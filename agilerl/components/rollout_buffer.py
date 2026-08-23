# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import OrderedDict
from collections.abc import Generator, Sequence
from typing import Any, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from tensordict import TensorDict, TensorDictBase

from agilerl.typing import (
    ArrayOrTensor,
    BPTTSequenceType,
    ObservationType,
)
from agilerl.utils.algo_utils import (
    extract_sequences_from_episode,
    get_num_actions,
    get_obs_shape,
    is_str_keyed_dict,
    maybe_add_batch_dim,
    narrow_tensor,
)


def _narrow_leaf(value: object) -> torch.Tensor | TensorDict:
    """Narrow a TensorDict child to the tensor or sub-collection it holds."""
    if isinstance(value, TensorDict):
        return value
    assert isinstance(value, torch.Tensor), (
        f"Expected a tensor or TensorDict leaf, got {type(value).__name__}."
    )
    return value


_T = TypeVar("_T")


def _require_prepared(value: _T | None) -> _T:
    """Return ``value`` unwrapped, or raise if sequences were never prepared."""
    if value is None:
        msg = (
            "Attempting to fetch minibatches before preparing sequences. "
            "Call prepare_sequence_tensors() first."
        )
        raise ValueError(msg)
    return value


def _index_tensordict(td: TensorDict, index: torch.Tensor | slice) -> TensorDict:
    """Index or slice ``td``, narrowing the widened ``__getitem__`` return."""
    view = td[index]
    assert isinstance(view, TensorDict), (
        f"Expected a TensorDict view, got {type(view).__name__}."
    )
    return view


def _is_numpy_tree(
    obj: object,
) -> TypeGuard[dict[str, npt.NDArray | dict[str, npt.NDArray]]]:
    """Narrow ``TensorDict.numpy()``'s result to the nested numpy form."""
    return isinstance(obj, dict)


class RolloutBuffer:
    """Rollout buffer for collecting experiences and computing advantages for RL algorithms.
    This buffer is designed to handle vectorized environments efficiently.

    :param capacity: Maximum number of timesteps to store in the buffer (per environment).
    :type capacity: int
    :param observation_space: Observation space of the environment.
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment.
    :type action_space: gym.spaces.Space
    :param num_envs: Number of parallel environments.
    :type num_envs: int
    :param device: Device to store tensors on, defaults to "cpu".
    :type device: str, optional
    :param gae_lambda: Lambda parameter for GAE, defaults to 0.95.
    :type gae_lambda: float, optional
    :param gamma: Discount factor, defaults to 0.99.
    :type gamma: float, optional
    :param recurrent: Whether to store hidden states, defaults to False.
    :type recurrent: bool, optional
    :param hidden_state_architecture: Architecture of hidden states if used, defaults to None.
    :type hidden_state_architecture: dict[str, tuple[int, int, int]], optional
    :param use_gae: Whether to compute GAE advantages, defaults to True.
    :type use_gae: bool, optional
    :param wrap_at_capacity: Whether to wrap the buffer at capacity, defaults to False. This is especially useful
        for OFF-policy algorithms, ON-policy algorithms should leave this as False in most cases.
    :type wrap_at_capacity: bool, optional
    :param max_seq_len: Maximum sequence length for BPTT, defaults to None.
    :type max_seq_len: int, optional
    """

    padded_data: TensorDict | None
    unpadded_data: TensorDict | None
    unpadded_slices: list[torch.Tensor] | None
    num_sequences: int | None
    max_sequence_length: int | None

    def __init__(
        self,
        capacity: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int = 1,
        device: str = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        recurrent: bool = False,
        hidden_state_architecture: dict[str, tuple[int, ...]] | None = None,
        use_gae: bool = True,
        wrap_at_capacity: bool = False,
        max_seq_len: int | None = None,
        bptt_sequence_type: BPTTSequenceType = BPTTSequenceType.CHUNKED,
    ) -> None:
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.recurrent = recurrent
        self.hidden_state_architecture = hidden_state_architecture
        self.use_gae = use_gae
        self.wrap_at_capacity = wrap_at_capacity
        self.max_seq_len = max_seq_len
        self.bptt_sequence_type = bptt_sequence_type

        self.pos = 0
        self.full = False
        self.num_sequences = None
        self.max_sequence_length = None
        self.unpadded_slices = None
        self.padded_data = None
        self.unpadded_data = None
        self._initialize_buffers()

    def size(self) -> int:
        """Get current number of transitions stored in the buffer.

        :return: Current number of transitions.
        :rtype: int
        """
        return (self.capacity if self.full else self.pos) * self.num_envs

    def reset(self) -> None:
        """Reset the buffer pointer and full flag."""
        self.pos = 0
        self.full = False
        self.num_sequences = None
        self.max_sequence_length = None
        self.unpadded_slices = None
        self.padded_data = None
        self.unpadded_data = None

    def _maybe_reshape_obs(
        self,
        obs: torch.Tensor,
        space: spaces.Space,
    ) -> torch.Tensor:
        """Reshape observation to the correct shape.

        :param obs: Observation to reshape.
        :type obs: torch.Tensor
        :param space: Observation space.
        :type space: spaces.Space
        :return: Reshaped observation.
        :rtype: torch.Tensor
        """
        if isinstance(space, spaces.Discrete) and obs.ndim < 2:
            obs = obs.unsqueeze(-1)

        # maybe_add_batch_dim preserves the input type at runtime, but its
        # annotation isn't generic over it, so the return is widened.
        return maybe_add_batch_dim(obs, space)

    def _initialize_buffers(self) -> None:
        """Initialize buffer arrays with correct shapes for vectorized environments."""
        # Determine shapes and dtypes for all expected fields
        obs_shape = get_obs_shape(self.observation_space)
        num_actions = get_num_actions(self.action_space)

        # Create a source TensorDict with appropriately sized tensors
        # The tensors will be on the CPU by default, can be moved to device later if needed.
        source_dict = OrderedDict()
        if isinstance(obs_shape, dict):  # Nested structure for Dict spaces
            obs_dict = OrderedDict()
            for key, shape in obs_shape.items():
                obs_dict[key] = torch.zeros(
                    (self.capacity, self.num_envs, *shape),
                    dtype=torch.float32,
                )

            source_dict["observations"] = obs_dict
            source_dict["next_observations"] = {
                key: torch.zeros_like(tensor) for key, tensor in obs_dict.items()
            }
        else:
            # For non-Dict spaces, use regular tensor allocation. Tuple
            # observation spaces are not supported by the rollout buffer, so
            # the shape is a flat tuple of ints here.
            leaf_shape: list[int] = []
            for dim in obs_shape:
                assert isinstance(dim, int), (
                    "Tuple observation spaces are not supported by the rollout buffer."
                )
                leaf_shape.append(dim)
            source_dict["observations"] = torch.zeros(
                (self.capacity, self.num_envs, *leaf_shape),
                dtype=torch.float32,
            )
            source_dict["next_observations"] = torch.zeros(
                (self.capacity, self.num_envs, *leaf_shape),
                dtype=torch.float32,
            )

        # Add other standard fields
        source_dict.update(
            {
                "actions": torch.zeros(
                    (self.capacity, self.num_envs, num_actions),
                    dtype=torch.float32,
                ),
                "rewards": torch.zeros(
                    (self.capacity, self.num_envs),
                    dtype=torch.float32,
                ),
                "dones": torch.zeros((self.capacity, self.num_envs), dtype=torch.bool),
                "values": torch.zeros(
                    (self.capacity, self.num_envs),
                    dtype=torch.float32,
                ),
                "log_probs": torch.zeros(
                    (self.capacity, self.num_envs),
                    dtype=torch.float32,
                ),
                "advantages": torch.zeros(
                    (self.capacity, self.num_envs),
                    dtype=torch.float32,
                ),
                "returns": torch.zeros(
                    (self.capacity, self.num_envs),
                    dtype=torch.float32,
                ),
            },
        )

        if self.recurrent:
            if self.hidden_state_architecture is None:
                msg = "hidden_state_architecture must be provided if recurrent=True"
                raise ValueError(
                    msg,
                )
            # self.hidden_state_architecture is dict[str, tuple[num_layers, num_envs_at_ppo_init, hidden_size]]
            # For buffer storage, each hidden state key should map to a tensor of shape:
            # (capacity, num_envs, num_layers, hidden_size)
            source_dict["hidden_states"] = {
                key: torch.zeros(
                    (
                        self.capacity,
                        self.num_envs,
                        self.hidden_state_architecture[key][0],  # num_layers
                        self.hidden_state_architecture[key][2],  # hidden_size
                    ),
                    dtype=torch.float32,
                )
                for key in self.hidden_state_architecture.keys()
            }
            # `next_hidden_states` might not be strictly necessary if we only store initial hidden states
            # for sequences. If we store step-by-step next_hidden_states, it would be:
            # source_dict["next_hidden_states"] = { ... similar structure ... }

        # Initialize the main buffer as a TensorDict with batch_size [capacity, num_envs]
        self.buffer = TensorDict(
            source_dict,
            batch_size=[self.capacity, self.num_envs],
            device="cpu",  # Keep buffer on CPU for memory efficiency, move to device in get_tensor_batch
        )

    def add(
        self,
        obs: ObservationType,
        action: ArrayOrTensor,
        reward: float | npt.NDArray,
        done: bool | npt.NDArray,
        value: float | npt.NDArray,
        log_prob: float | npt.NDArray,
        next_obs: ObservationType | None = None,
        hidden_state: dict[str, torch.Tensor] | None = None,
        next_hidden_state: (
            dict[str, torch.Tensor] | None
        ) = None,  # Not used if only initial hidden states are stored
        episode_start: bool | npt.NDArray | None = None,
        action_mask: ArrayOrTensor | None = None,
    ) -> None:
        """Add a new batch of observations and associated data from vectorized environments to the buffer.

        :param obs: Current observation batch (shape: (num_envs, *obs_shape))
        :type obs: ObservationType
        :param action: Action batch taken (shape: (num_envs, *action_shape))
        :type action: ArrayOrTensor
        :param reward: Reward batch received (shape: (num_envs,))
        :type reward: float | npt.NDArray
        :param done: Done flag batch (shape: (num_envs,))
        :type done: bool | npt.NDArray
        :param value: Value estimate batch (shape: (num_envs,))
        :type value: float | npt.NDArray
        :param log_prob: Log probability batch of the actions (shape: (num_envs,))
        :type log_prob: float | npt.NDArray
        :param next_obs: Next observation batch (shape: (num_envs, *obs_shape)), defaults to None
        :type next_obs: ObservationType | None
        :param hidden_state: Current hidden state batch (shape: (num_envs, hidden_size)), defaults to None
        :type hidden_state: dict[str, torch.Tensor] | None
        :param next_hidden_state: Next hidden state batch (shape: (num_envs, hidden_size)), defaults to None
        :type next_hidden_state: dict[str, torch.Tensor] | None
        :param episode_start: Episode start flag batch (shape: (num_envs,)), defaults to None
        :type episode_start: bool | npt.NDArray | None
        :param action_mask: Action mask batch (shape: (num_envs, mask_size)), 1=legal 0=illegal, defaults to None
        :type action_mask: ArrayOrTensor | None
        """
        if self.pos == self.capacity:
            if self.wrap_at_capacity:
                self.pos = 0
            else:
                msg = (
                    f"Buffer has reached capacity ({self.capacity} transitions) but received more transitions. "
                    "Either increase buffer capacity or set wrap_at_capacity=True."
                )
                raise ValueError(
                    msg,
                )

        # Prepare data as a dictionary of tensors for the current time step
        current_step_data = OrderedDict()

        # Convert inputs to tensors and ensure correct device (CPU for buffer storage)
        # Also ensure they have the (num_envs, ...) shape
        if isinstance(self.observation_space, spaces.Dict):
            assert is_str_keyed_dict(obs), (
                "Expected a dict observation for a Dict observation space."
            )
            obs_dict = OrderedDict()
            for key, item in obs.items():
                sub_space = self.observation_space.spaces[key]
                obs_tensor = torch.as_tensor(item, device="cpu")
                obs_dict[key] = self._maybe_reshape_obs(obs_tensor, sub_space)

            current_step_data["observations"] = obs_dict
        else:
            obs_tensor = torch.as_tensor(obs, device="cpu")
            current_step_data["observations"] = self._maybe_reshape_obs(
                obs_tensor,
                self.observation_space,
            )

        # Actions
        action_tensor = torch.as_tensor(action, device="cpu")
        current_step_data["actions"] = action_tensor.reshape(self.num_envs, -1)

        # Rewards
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device="cpu")
        current_step_data["rewards"] = reward_tensor.reshape(self.num_envs)

        # Dones
        done_tensor = torch.as_tensor(done, dtype=torch.bool, device="cpu")
        current_step_data["dones"] = done_tensor.reshape(self.num_envs)

        # Values
        value_tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        current_step_data["values"] = value_tensor.reshape(self.num_envs)

        # Log_probs
        log_prob_tensor = torch.as_tensor(log_prob, dtype=torch.float32, device="cpu")
        current_step_data["log_probs"] = log_prob_tensor.reshape(self.num_envs)

        # Next Observations
        if next_obs is not None:
            if isinstance(self.observation_space, spaces.Dict):
                assert is_str_keyed_dict(next_obs), (
                    "Expected a dict observation for a Dict observation space."
                )
                next_obs_dict = OrderedDict()
                for key, item in next_obs.items():
                    sub_space = self.observation_space.spaces[key]
                    next_obs_tensor = torch.as_tensor(item, device="cpu")
                    next_obs_dict[key] = self._maybe_reshape_obs(
                        next_obs_tensor,
                        sub_space,
                    )

                current_step_data["next_observations"] = next_obs_dict
            else:
                next_obs_tensor = torch.as_tensor(next_obs, device="cpu")
                current_step_data["next_observations"] = self._maybe_reshape_obs(
                    next_obs_tensor,
                    self.observation_space,
                )

        # Episode Starts
        if episode_start is not None:
            episode_start_tensor = torch.as_tensor(
                episode_start,
                dtype=torch.bool,
                device="cpu",
            )
            current_step_data["episode_starts"] = episode_start_tensor.reshape(
                self.num_envs,
            )
        else:
            current_step_data["episode_starts"] = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device="cpu",
            )

        # Action Masks (lazily allocated on first use to avoid wasting memory
        # for discrete action spaces that don't use masking)
        if action_mask is not None:
            action_mask_tensor = torch.as_tensor(
                action_mask, dtype=torch.bool, device="cpu"
            ).reshape(self.num_envs, -1)

            if "action_masks" not in self.buffer.keys():
                mask_size = action_mask_tensor.shape[-1]
                self.buffer["action_masks"] = torch.ones(
                    (self.capacity, self.num_envs, mask_size),
                    dtype=torch.bool,
                )

            current_step_data["action_masks"] = action_mask_tensor

        # Hidden States (assuming they are dictionaries of tensors)
        if self.recurrent and hidden_state is not None:
            # hidden_state is dict[str, Tensor] from PPO -> {key: (layers, num_envs, size)}
            # We need to populate current_step_data["hidden_states"]
            # with {key: (num_envs, layers, size)}
            current_step_data["hidden_states"] = {}
            for key, ppo_tensor_val in hidden_state.items():
                current_step_data["hidden_states"][key] = ppo_tensor_val.permute(
                    1,
                    0,
                    2,
                )  # Shape: (num_envs, layers, size)

        # Create a TensorDict for the current step's data
        # This will have batch_size [num_envs]
        current_td_slice = TensorDict(
            current_step_data,
            batch_size=[self.num_envs],
            device="cpu",
        )

        # Assign this slice to the buffer at the current position
        # self.buffer has batch_size [capacity, num_envs]
        # self.buffer[self.pos] should be a TensorDict with batch_size [num_envs]
        # This assignment will correctly place the data, including the nested hidden_states TD.
        self.buffer[self.pos] = current_td_slice

        # Update buffer position
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: ArrayOrTensor,
        last_done: ArrayOrTensor,
    ) -> None:
        """Compute returns and advantages for the stored experiences using GAE or Monte Carlo.

        :param last_value: Value estimate for the last observation in each environment (shape: (num_envs,))
        :type last_value: ArrayOrTensor
        :param last_done: Done flag for the last state in each environment (shape: (num_envs,))
        :type last_done: ArrayOrTensor
        """
        # Convert inputs to numpy arrays if they are tensors, and ensure correct shape
        if isinstance(last_value, torch.Tensor):
            last_value_np = last_value.cpu().numpy().reshape(self.num_envs)
        else:
            last_value_np = np.asarray(last_value).reshape(self.num_envs)

        if isinstance(last_done, torch.Tensor):
            last_done_np = last_done.cpu().numpy().reshape(self.num_envs)
        else:
            last_done_np = np.asarray(last_done).reshape(self.num_envs)

        buffer_size = self.capacity if self.full else self.pos

        # Temporary numpy arrays for computation, will be assigned back to TensorDict
        advantages_np = np.zeros((buffer_size, self.num_envs), dtype=np.float32)
        returns_np = np.zeros((buffer_size, self.num_envs), dtype=np.float32)

        # Get necessary data from TensorDict as numpy arrays for computation
        # Slicing to buffer_size for all components.
        rewards_np = narrow_tensor(self.buffer["rewards"])[:buffer_size].cpu().numpy()
        dones_np = narrow_tensor(self.buffer["dones"])[:buffer_size].cpu().numpy()
        values_np = narrow_tensor(self.buffer["values"])[:buffer_size].cpu().numpy()

        if self.use_gae:
            last_gae_lambda = np.zeros(self.num_envs, dtype=np.float32)
            for t in reversed(range(buffer_size)):
                if t == buffer_size - 1:
                    next_non_terminal = 1.0 - last_done_np.astype(float)
                    next_values = last_value_np.astype(float)
                else:
                    next_non_terminal = 1.0 - dones_np[t + 1].astype(float)
                    next_values = values_np[t + 1]

                delta = (
                    rewards_np[t]
                    + self.gamma * next_values * next_non_terminal
                    - values_np[t]
                )
                advantages_np[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )
            returns_np = advantages_np + values_np
        else:
            # Monte Carlo returns
            last_returns_np = last_value_np.astype(float) * (
                1.0 - last_done_np.astype(float)
            )
            for t in reversed(range(buffer_size)):
                returns_np[t] = last_returns_np = rewards_np[
                    t
                ] + self.gamma * last_returns_np * (1.0 - dones_np[t].astype(float))
            advantages_np = returns_np - values_np

        # Assign computed advantages and returns back to the TensorDict
        self.buffer["advantages"][:buffer_size] = torch.from_numpy(advantages_np)
        self.buffer["returns"][:buffer_size] = torch.from_numpy(returns_np)

    def get(
        self,
        batch_size: int | None = None,
    ) -> dict[str, npt.NDArray | dict[str, npt.NDArray]]:
        """Get data from the buffer, flattened and optionally sampled into minibatches.

        :param batch_size: Size of the minibatch to sample. If None, returns all data. Defaults to None.
        :type batch_size: int | None
        :return: Dictionary containing flattened buffer data arrays.
        :rtype: dict[str, npt.NDArray | dict[str, npt.NDArray]]
        """
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            # Return empty dictionary or raise error if buffer is empty
            return {}

        # Get a view of the buffer up to the current position and for all envs
        # This slice will have batch_size [buffer_size, num_envs]
        valid_buffer_data = self.buffer[:buffer_size]
        assert isinstance(valid_buffer_data, TensorDict)

        # Reshape to flatten the num_envs dimension into the first batch dimension
        # New batch_size will be [buffer_size * num_envs]
        flattened_td = valid_buffer_data.view(-1)

        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer size {total_samples}. Returning all data.",
                    stacklevel=2,
                )
                # Convert the flattened TensorDict to the old dictionary format
                # For hidden_states, we need to handle the nested TensorDict structure
                return self._convert_td_to_np_dict(flattened_td)

            indices = np.random.choice(total_samples, size=batch_size, replace=False)
            sampled_td = flattened_td[indices]
            # Convert the sampled TensorDict to the old dictionary format
            return self._convert_td_to_np_dict(sampled_td)
        return self._convert_td_to_np_dict(flattened_td)

    def get_tensor_batch(
        self,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> TensorDict:
        """Get data from the buffer as PyTorch tensors, flattened and optionally sampled.
        The output is a TensorDict.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None.
        :type batch_size: int | None
        :param device: Device to put tensors on, defaults to None (uses self.device).
        :type device: str | None
        :return: TensorDict containing the data.
        :rtype: TensorDict
        """
        target_device = device or self.device
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            # Empty buffer: return an empty TensorDict so callers can uniformly
            # rely on TensorDict semantics (e.g. ``is_empty()``).
            return TensorDict({}, batch_size=[], device=target_device)

        # Get a view of the buffer up to the current position and for all envs
        # This slice will have batch_size [buffer_size, num_envs]
        # All tensors inside self.buffer are already torch.Tensors on CPU
        valid_buffer_data_view = _index_tensordict(
            self.buffer, slice(None, buffer_size)
        )

        # Reshape to flatten the num_envs dimension into the first batch dimension
        # New batch_size will be [buffer_size * num_envs]
        # .view(-1) avoids a copy when the layout is contiguous
        flattened_td: TensorDict = valid_buffer_data_view.view(-1)

        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer_size {total_samples}. "
                    "Returning all data.",
                    stacklevel=2,
                )
                # Move the whole flattened_td to the target device
                return flattened_td.to(target_device)

            indices = torch.randperm(total_samples, device="cpu")[
                :batch_size
            ]  # Sample on CPU then move
            sampled_td = flattened_td[indices]
            assert isinstance(sampled_td, TensorDict)

            return sampled_td.to(target_device)
        # Return all flattened data, moved to the target device
        return flattened_td.to(target_device)

    def _convert_td_to_np_dict(
        self,
        td: TensorDict,
    ) -> dict[str, npt.NDArray | dict[str, npt.NDArray]]:
        """Convert a TensorDict to a dictionary of numpy arrays.

        :param td: TensorDict to convert.
        :type td: TensorDict
        :return: Dictionary of numpy arrays.
        :rtype: dict[str, npt.NDArray | dict[str, npt.NDArray]]
        """
        # TensorDict.numpy() returns the nested {key: ndarray | {sub_key: ndarray}}
        # form directly, moving tensors off-device as it goes.
        converted = td.cpu().numpy()
        assert _is_numpy_tree(converted), (
            f"Expected a dict from TensorDict.numpy(), got {type(converted).__name__}."
        )
        return converted

    @staticmethod
    def _pad_sequences(
        sequences: Sequence[torch.Tensor | TensorDict],
        target_length: int | None = None,
    ) -> torch.Tensor | TensorDict:
        """Pad sequences using zeros. If target_length is provided, the sequences are padded to the target length.
        Otherwise, the sequences are padded to the length of the longest sequence. Results in a tensor of
        shape (B, T, *). We provide the option to pad to a specified target length but in general these should be padded
        to the maximum length of the sequences in the batch (i.e. using `torch.nn.utils.rnn.pad_sequence`).

        :param sequences: The sequences to be padded.
        :type sequences: Sequence[torch.Tensor | TensorDict]
        :param target_length: The target length to pad the sequences to. If None, the sequences are padded to the length of the longest sequence.
        :type target_length: int | None
        :return: The padded sequence.
        :rtype: torch.Tensor | TensorDict
        """
        # Handle nested tensors (sequences are homogeneous: if the first element
        # is a tensor collection, all of them are), regrouping by key and recursing.
        first = sequences[0]
        if isinstance(first, TensorDict):
            # Sequences are homogeneous, so the isinstance filter keeps them all
            # while narrowing the element type to TensorDict.
            nested_tds = [seq for seq in sequences if isinstance(seq, TensorDict)]
            sequences_T: dict[Any, list[torch.Tensor | TensorDict]] = {
                key: [_narrow_leaf(nested_td.get(key)) for nested_td in nested_tds]
                for key in first.keys()
            }
            return TensorDict(
                {
                    key: RolloutBuffer._pad_sequences(sequences_T[key], target_length)
                    for key in sequences_T
                },
            )

        # Homogeneous with a tensor first element, so every entry is a tensor.
        tensors = [seq for seq in sequences if isinstance(seq, torch.Tensor)]

        # If target_length is provided, pad the sequences to the target length
        if target_length is not None:
            padded_sequences = []
            for seq in tensors:
                current_length = seq.size(0)
                pad_amount = target_length - current_length
                pad_spec = [0, 0] * (seq.dim() - 1) + [0, pad_amount]
                padded_sequences.append(
                    torch.nn.functional.pad(seq, pad_spec, mode="constant", value=0),
                )

            return torch.stack(padded_sequences)

        # Otherwise, pad the sequences to the length of the longest sequence
        return torch.nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True,
            padding_value=0,
        )

    def _extract_td_sequences(
        self,
        episode: TensorDict,
        max_seq_len: int,
    ) -> list[TensorDict]:
        """TensorDict counterpart of :func:`extract_sequences_from_episode` for
        :param episode: The nested episode to split into sequences.
        :type episode: TensorDict
        :param max_seq_len: The maximum sequence length.
        :type max_seq_len: int
        :return: The sequences extracted from the episode.
        :rtype: list[TensorDict]
        """
        length = len(episode)
        sequence_type = self.bptt_sequence_type
        if sequence_type == BPTTSequenceType.CHUNKED:
            num_chunks = max(1, length // max_seq_len)
            starts: range = range(0, num_chunks * max_seq_len, max_seq_len)
        elif sequence_type == BPTTSequenceType.MAXIMUM:
            starts = range(length - max_seq_len + 1)
        elif sequence_type == BPTTSequenceType.FIFTY_PERCENT_OVERLAP:
            starts = range(0, length - max_seq_len + 1, max_seq_len // 2)
        else:
            msg = f"Received unrecognized sequence type: {sequence_type}"
            raise NotImplementedError(msg)
        sequences: list[TensorDict] = []
        for start in starts:
            sequence = episode[start : start + max_seq_len]
            assert isinstance(sequence, TensorDict)
            sequences.append(sequence)
        return sequences

    def _get_complete_sequences(
        self,
        data: torch.Tensor | TensorDict,
        episode_done_indices: list[list[int]],
    ) -> tuple[list[torch.Tensor | TensorDict], int]:
        """Split the provided data into sequences. If `self.max_seq_len` is not set, the entire episode
        is used as a sequence. Otherwise, the episode is split into sequences of length `self.max_seq_len`.
        If the episode is shorter than `self.max_seq_len`, the entire episode is used as a sequence.

        :param data: The data to be split into sequences.
        :type data: torch.Tensor | TensorDict
        :param episode_done_indices: The indices of done signals.
        :type episode_done_indices: list[list[int]]
        :return: A list of sequences and the length of the longest sequence.
        :rtype: tuple[list[torch.Tensor | TensorDict], int]
        """
        max_seq_len = self.max_seq_len
        sequences: list[torch.Tensor | TensorDict] = []
        max_length = 1
        for env_idx in range(self.num_envs):
            start_index = 0
            for done_index in episode_done_indices[env_idx]:
                # Split trajectory into episodes. Nested buffer values (hidden
                # states, dict observations) are TensorDicts; leaves are tensors.
                if isinstance(data, torch.Tensor):
                    episode: torch.Tensor | TensorDict = data[
                        start_index : done_index + 1, env_idx
                    ]
                else:
                    sliced = data[start_index : done_index + 1, env_idx]
                    assert isinstance(sliced, TensorDict), (
                        "Nested buffer values are TensorDicts."
                    )
                    episode = sliced

                # Split episodes into sequences for truncated BPTT
                # If max_seq_len is not set, use the entire episode as a sequence
                # NOTE: It may be the case that we provide a max_seq_len but we have episodes
                # that are shorter than max_seq_len. In this case, we will use the entire episode
                # as a sequence, and later pad the shorter episodes to the max_seq_len.
                if (max_seq_len is not None) and (len(episode) >= max_seq_len):
                    # Extract sequences from the episode
                    if isinstance(episode, torch.Tensor):
                        sequences.extend(
                            extract_sequences_from_episode(
                                episode=episode,
                                max_seq_len=max_seq_len,
                                sequence_type=self.bptt_sequence_type,
                            )
                        )
                    else:
                        sequences.extend(
                            self._extract_td_sequences(episode, max_seq_len)
                        )
                else:
                    sequences.append(episode)

                max_length = max(max_length, len(episode))
                start_index = done_index + 1

        return sequences, max_length

    def prepare_sequence_tensors(self, device: str | None = None) -> None:
        """Prepare padded and unpadded TensorDicts with all of the possible sequences in the
        :param device: Device to put tensors on, defaults to None (uses self.device).
        :type device: str | None
        """
        if not self.recurrent:
            msg = "prepare_sequence_tensors() is only supported when recurrent=True."
            raise ValueError(
                msg,
            )

        if self.size() == 0:
            msg = "Attempting to prepare sequences with empty buffer."
            raise ValueError(msg)

        target_device = device or self.device
        buffer_size = self.capacity if self.full else self.pos

        # Get a view of the buffer up to the current position and for all envs
        # This slice will have batch_size [buffer_size, num_envs]
        # All tensors inside self.buffer are already torch.Tensors on CPU
        valid_buffer_data_view = _index_tensordict(
            self.buffer, slice(None, buffer_size)
        )

        # Split data into sequences and apply zero-padding
        # Retrieve the indices of dones as these are the last step of a whole episode
        dones = valid_buffer_data_view["dones"]
        assert isinstance(dones, torch.Tensor)
        episode_done_indices: list[list[int]] = []
        for env_idx in range(self.num_envs):
            env_dones: torch.Tensor = dones[:, env_idx]
            env_dones_list = env_dones.nonzero().squeeze(-1).tolist()
            episode_done_indices.append(env_dones_list)

            # Mark the end of the buffer as the end of an episode
            if (
                len(episode_done_indices[env_idx]) == 0
                or episode_done_indices[env_idx][-1] != buffer_size - 1
            ):
                episode_done_indices[env_idx].append(buffer_size - 1)

        # Get the indices of unpadded sequences
        flat_timesteps = torch.arange(buffer_size * self.num_envs).reshape(
            buffer_size,
            self.num_envs,
        )
        timestep_sequences, _ = self._get_complete_sequences(
            flat_timesteps,
            episode_done_indices,
        )
        # flat_timesteps is a tensor, so every returned slice is a tensor.
        unpadded_slices: list[torch.Tensor] = []
        for timestep_slice in timestep_sequences:
            assert isinstance(timestep_slice, torch.Tensor)
            unpadded_slices.append(timestep_slice)

        self.unpadded_slices = unpadded_slices

        keys_to_pad = ["observations", "actions", "hidden_states"]
        if "action_masks" in valid_buffer_data_view.keys():
            keys_to_pad.append("action_masks")
        valid_data_to_pad: TensorDictBase = valid_buffer_data_view.select(
            *keys_to_pad,
        ).clone()

        valid_data_to_pad["pad_mask"] = torch.ones(
            valid_data_to_pad.batch_size,
            dtype=torch.bool,
        )

        # Create a TensorDict with all of the possible sequences, padded to the same length
        padded_data_source = OrderedDict()
        for key, value in valid_data_to_pad.items():
            # Split data into episodes or sequences
            # NOTE: nested TensorDict values (e.g. hidden states) are handled by
            # the TensorDict path of _get_complete_sequences
            sequences, max_sequence_length = self._get_complete_sequences(
                _narrow_leaf(value),
                episode_done_indices,
            )

            # Apply zero-padding to ensure that each episode has the same length
            # NOTE: In https://github.com/MarcoMeter/recurrent-ppo-truncated-bpttm sequences
            # are padded to the maximum episode length, whereas these should be padded to the
            # specified max_sequence_length for truncated BPTT to see its benefits.
            padded_data_source[key] = RolloutBuffer._pad_sequences(
                sequences,
                target_length=None,
            )

        self.max_sequence_length = (
            min(max_sequence_length, self.max_seq_len)
            if self.max_seq_len is not None
            else max_sequence_length
        )
        self.num_sequences = len(sequences)

        padded_td = TensorDict(
            source=padded_data_source,
            batch_size=[self.num_sequences, self.max_sequence_length],
        )

        # Now, create the "initial_hidden_states" part for the output TensorDict
        # This will be a nested TensorDict with batch_size [actual_batch_size]
        if self.recurrent and padded_td.get("hidden_states", None) is not None:
            initial_hidden_states = OrderedDict()
            full_hidden_sequences: TensorDict = padded_td.get("hidden_states")
            for h_key, h_val_tensor_sequences in full_hidden_sequences.items():
                initial_hidden_states[h_key] = h_val_tensor_sequences[:, 0].clone()

            # Use set_non_tensor for the dictionary of initial hidden state tensors
            padded_td.set_non_tensor("initial_hidden_states", initial_hidden_states)

            # Exclude original full hidden_states sequence
            padded_td = padded_td.exclude("hidden_states")

        # Select unpadded data for training
        unpadded_data_source: TensorDict = valid_buffer_data_view.select(
            "log_probs",
            "advantages",
            "values",
            "returns",
        )

        # Flatten the TensorDict's and move to the target device
        self.padded_data = padded_td.view(-1).to(target_device)
        self.unpadded_data = unpadded_data_source.view(-1).to(target_device)

    def get_minibatch_sequences(
        self,
        batch_size: int,
    ) -> Generator[tuple[TensorDict, TensorDict], None, None]:
        """Get a minibatch of sequences from the buffer.

        :param batch_size: The number of sequences to sample.
        :type batch_size: int
        :return: A TensorDict containing the minibatch of sequences.
        :rtype: Generator[tuple[TensorDict, TensorDict], None, None]
        """
        unpadded_slices = _require_prepared(self.unpadded_slices)
        padded_data = _require_prepared(self.padded_data)
        unpadded_data = _require_prepared(self.unpadded_data)
        total_sequences = _require_prepared(self.num_sequences)
        max_sequence_length = _require_prepared(self.max_sequence_length)

        # Determine the number of sequences per mini batch
        if batch_size > total_sequences:
            warnings.warn(
                f"Batch size {batch_size} is larger than the number of sequences "
                f"({total_sequences}), using batch_size = {total_sequences}.",
                stacklevel=2,
            )
            batch_size = total_sequences

        num_batches = total_sequences // batch_size
        num_sequences_per_batch = [batch_size] * num_batches
        remainder = total_sequences % batch_size

        if remainder > 0:
            num_sequences_per_batch.append(remainder)

        # Prepare indices, but only shuffle the sequence indices and not the
        # entire batch to ensure that sequences are maintained as a whole.
        indices = torch.arange(
            start=0,
            end=total_sequences * max_sequence_length,
        ).reshape(total_sequences, max_sequence_length)

        sequence_indices = torch.randperm(total_sequences)

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            sequences_samples = sequence_indices[start:end]
            padded_indices = indices[sequences_samples].view(-1)

            # Unpadded and flat indices are used to sample unpadded training data
            minibatch_seq_indices: list[int] = sequences_samples.tolist()
            unpadded_slice_lists = [
                unpadded_slices[idx].tolist() for idx in minibatch_seq_indices
            ]
            unpadded_indices = [
                int(item) for sublist in unpadded_slice_lists for item in sublist
            ]
            unpadded_indices_tensor = torch.as_tensor(
                unpadded_indices, device=indices.device
            )

            padded = _index_tensordict(padded_data, padded_indices)
            if self.recurrent and padded.get("initial_hidden_states", None) is not None:
                batch_hidden_states = {}
                initial_hidden_states: dict[str, Any] = padded.get_non_tensor(
                    "initial_hidden_states",
                )
                for key, value in initial_hidden_states.items():
                    batch_hidden_states[key] = value[sequences_samples].clone()

                padded.set_non_tensor("initial_hidden_states", batch_hidden_states)

            unpadded = _index_tensordict(unpadded_data, unpadded_indices_tensor)
            start = end
            yield padded, unpadded

    def __getstate__(self) -> dict[str, Any]:
        """Get the state dictionary for pickling, ensuring arrays are copied.

        :return: State dictionary.
        :rtype: dict[str, Any]
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state dictionary when unpickling, re-initializing buffers.

        :param state: State dictionary.
        :type state: dict[str, Any]
        """
        self.__dict__.update(state)

        # Let's assume self.buffer is correctly loaded by `self.__dict__.update(state)`.
        # We should verify its integrity or re-initialize if it's somehow corrupted/not present.
        if not hasattr(self, "buffer") or not isinstance(self.buffer, TensorDict):
            warnings.warn(
                "TensorDict buffer not found or invalid during unpickling. Re-initializing.",
                stacklevel=2,
            )
            if not hasattr(self, "observation_space") or not hasattr(
                self,
                "action_space",
            ):
                warnings.warn(
                    "Observation or action space missing during RolloutBuffer unpickling. Buffer might be invalid.",
                    stacklevel=2,
                )
                return

            self._initialize_buffers()
            self.buffer = self.buffer.to("cpu")  # Ensure the buffer is on CPU
        else:
            # Verify batch_size and keys if necessary
            expected_batch_size = torch.Size([self.capacity, self.num_envs])
            if self.buffer.batch_size != expected_batch_size:
                warnings.warn(
                    f"Loaded TensorDict has batch_size {self.buffer.batch_size}, expected {expected_batch_size}. Re-initializing.",
                    stacklevel=2,
                )
                self._initialize_buffers()

            # Ensure the loaded buffer is on the correct device (CPU) as expected by internal logic
            self.buffer = self.buffer.to("cpu")
