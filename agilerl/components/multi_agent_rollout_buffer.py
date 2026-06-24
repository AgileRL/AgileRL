import warnings
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict

from agilerl.typing import (
    ArrayOrTensor,
    BPTTSequenceType,
    ObservationType,
    TorchObsType,
)
from agilerl.utils.algo_utils import (
    get_num_actions,
    get_obs_shape,
    maybe_add_batch_dim,
)


class MultiAgentRolloutBuffer:
    """Rollout buffer for collecting experiences and computing advantages for multi-agent RL algorithms.
    This buffer is designed to handle vectorized multi-agent environments efficiently.

    :param capacity: Maximum number of timesteps to store in the buffer (per environment).
    :type capacity: int
    :param observation_spaces: Observation spaces for each agent.
    :type observation_spaces: dict[str, gym.spaces.Space]
    :param action_spaces: Action spaces for each agent.
    :type action_spaces: dict[str, gym.spaces.Space]
    :param agent_ids: List of agent IDs.
    :type agent_ids: list[str]
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

    padded_data: TensorDict
    unpadded_data: TensorDict
    unpadded_slices: list[torch.Tensor]

    def __init__(
        self,
        capacity: int,
        observation_spaces: dict[str, spaces.Space],
        action_spaces: dict[str, spaces.Space],
        agent_ids: list[str],
        num_envs: int = 1,
        device: str = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        recurrent: bool = False,
        hidden_state_architecture: dict[str, tuple[int, int, int]] | None = None,
        use_gae: bool = True,
        wrap_at_capacity: bool = False,
        max_seq_len: int | None = None,
        bptt_sequence_type: BPTTSequenceType = BPTTSequenceType.CHUNKED,
    ) -> None:
        self.capacity = capacity
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.agent_ids = agent_ids
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
        obs: TorchObsType,
        space: spaces.Space,
    ) -> TorchObsType:
        """Reshape observation to the correct shape.

        :param obs: Observation to reshape.
        :type obs: TorchObsType
        :param space: Observation space.
        :type space: spaces.Space
        :return: Reshaped observation.
        :rtype: TorchObsType
        """
        if isinstance(space, spaces.Discrete) and obs.ndim < 2:
            obs = obs.unsqueeze(-1)

        return maybe_add_batch_dim(obs, space)

    def _initialize_buffers(self) -> None:
        """Initialize buffer arrays with correct shapes for vectorized multi-agent environments."""
        # Create a source TensorDict with appropriately sized tensors
        source_dict = OrderedDict()

        # Handle observations for each agent
        obs_dict = OrderedDict()
        for agent_id, obs_space in self.observation_spaces.items():
            obs_shape = get_obs_shape(obs_space)
            if isinstance(obs_space, spaces.Dict):
                agent_obs_dict = OrderedDict()
                for key, shape in obs_shape.items():
                    agent_obs_dict[key] = torch.zeros(
                        (self.capacity, self.num_envs, *shape),
                        dtype=torch.float32,
                    )
                obs_dict[agent_id] = agent_obs_dict
            else:
                obs_dict[agent_id] = torch.zeros(
                    (self.capacity, self.num_envs, *obs_shape),
                    dtype=torch.float32,
                )

        source_dict["observations"] = obs_dict
        source_dict["next_observations"] = {
            agent_id: torch.zeros_like(tensor) for agent_id, tensor in obs_dict.items()
        }

        # Handle actions for each agent
        action_dict = OrderedDict()
        for agent_id, action_space in self.action_spaces.items():
            num_actions = get_num_actions(action_space)
            action_dict[agent_id] = torch.zeros(
                (self.capacity, self.num_envs, num_actions),
                dtype=torch.float32,
            )
        source_dict["actions"] = action_dict

        # Add shared fields
        source_dict.update(
            {
                "rewards": torch.zeros(
                    (self.capacity, self.num_envs, len(self.agent_ids)),
                    dtype=torch.float32,
                ),
                "dones": torch.zeros((self.capacity, self.num_envs), dtype=torch.bool),
                "values": torch.zeros(
                    (self.capacity, self.num_envs, len(self.agent_ids)),
                    dtype=torch.float32,
                ),
                "log_probs": torch.zeros(
                    (self.capacity, self.num_envs, len(self.agent_ids)),
                    dtype=torch.float32,
                ),
                "advantages": torch.zeros(
                    (self.capacity, self.num_envs, len(self.agent_ids)),
                    dtype=torch.float32,
                ),
                "returns": torch.zeros(
                    (self.capacity, self.num_envs, len(self.agent_ids)),
                    dtype=torch.float32,
                ),
            },
        )

        if self.recurrent:
            if self.hidden_state_architecture is None:
                msg = "hidden_state_architecture must be provided if recurrent=True"
                raise ValueError(msg)
            # For multi-agent, hidden states per agent
            hidden_states_dict = OrderedDict()
            for agent_id in self.agent_ids:
                if agent_id in self.hidden_state_architecture:
                    arch = self.hidden_state_architecture[agent_id]
                    hidden_states_dict[agent_id] = torch.zeros(
                        (
                            self.capacity,
                            self.num_envs,
                            arch[0],  # num_layers
                            arch[2],  # hidden_size
                        ),
                        dtype=torch.float32,
                    )
            source_dict["hidden_states"] = hidden_states_dict

        # Initialize the main buffer as a TensorDict with batch_size [capacity, num_envs]
        self.buffer = TensorDict(
            source_dict,
            batch_size=[self.capacity, self.num_envs],
            device="cpu",  # Keep buffer on CPU for memory efficiency
        )

    def add(
        self,
        obs: dict[str, ObservationType],
        action: dict[str, ArrayOrTensor],
        reward: dict[str, float | np.ndarray],
        done: bool | np.ndarray,
        value: dict[str, float | np.ndarray],
        log_prob: dict[str, float | np.ndarray],
        next_obs: dict[str, ObservationType] | None = None,
        hidden_state: dict[str, ArrayOrTensor] | None = None,
        next_hidden_state: dict[str, ArrayOrTensor] | None = None,
        episode_start: bool | np.ndarray | None = None,
        action_mask: dict[str, ArrayOrTensor] | None = None,
    ) -> None:
        """Add a new batch of observations and associated data from vectorized multi-agent environments to the buffer.

        :param obs: Current observation batch for each agent (shape: {agent_id: (num_envs, *obs_shape)})
        :type obs: dict[str, ObservationType]
        :param action: Action batch taken for each agent (shape: {agent_id: (num_envs, *action_shape)})
        :type action: dict[str, ArrayOrTensor]
        :param reward: Reward batch received for each agent (shape: {agent_id: (num_envs,)})
        :type reward: dict[str, float | np.ndarray]
        :param done: Done flag batch (shape: (num_envs,))
        :type done: bool | np.ndarray
        :param value: Value estimate batch for each agent (shape: {agent_id: (num_envs,)})
        :type value: dict[str, float | np.ndarray]
        :param log_prob: Log probability batch of the actions for each agent (shape: {agent_id: (num_envs,)})
        :type log_prob: dict[str, float | np.ndarray]
        :param next_obs: Next observation batch for each agent (shape: {agent_id: (num_envs, *obs_shape)}), defaults to None
        :type next_obs: dict[str, ObservationType] | None
        :param hidden_state: Current hidden state batch for each agent (shape: {agent_id: (num_envs, hidden_size)}), defaults to None
        :type hidden_state: dict[str, ArrayOrTensor] | None
        :param next_hidden_state: Next hidden state batch for each agent (shape: {agent_id: (num_envs, hidden_size)}), defaults to None
        :type next_hidden_state: dict[str, ArrayOrTensor] | None
        :param episode_start: Episode start flag batch (shape: (num_envs,)), defaults to None
        :type episode_start: bool | np.ndarray | None
        :param action_mask: Action mask batch for each agent (shape: {agent_id: (num_envs, mask_size)}), 1=legal 0=illegal, defaults to None
        :type action_mask: dict[str, ArrayOrTensor] | None
        """
        if self.pos == self.capacity:
            if self.wrap_at_capacity:
                self.pos = 0
            else:
                msg = (
                    f"Buffer has reached capacity ({self.capacity} transitions) but received more transitions. "
                    "Either increase buffer capacity or set wrap_at_capacity=True."
                )
                raise ValueError(msg)

        # Prepare data as a dictionary of tensors for the current time step
        current_step_data = OrderedDict()

        # Handle observations
        obs_dict = OrderedDict()
        for agent_id, agent_obs in obs.items():
            obs_space = self.observation_spaces[agent_id]
            if isinstance(obs_space, spaces.Dict):
                agent_obs_dict = OrderedDict()
                for key, item in agent_obs.items():
                    sub_space = obs_space.spaces[key]
                    obs_tensor = torch.as_tensor(item, device="cpu")
                    agent_obs_dict[key] = self._maybe_reshape_obs(obs_tensor, sub_space)
                obs_dict[agent_id] = agent_obs_dict
            else:
                obs_tensor = torch.as_tensor(agent_obs, device="cpu")
                obs_dict[agent_id] = self._maybe_reshape_obs(obs_tensor, obs_space)
        current_step_data["observations"] = obs_dict

        # Handle actions
        action_dict = OrderedDict()
        for agent_id, agent_action in action.items():
            action_tensor = torch.as_tensor(agent_action, device="cpu")
            action_dict[agent_id] = action_tensor.reshape(self.num_envs, -1)
        current_step_data["actions"] = action_dict

        # Handle rewards
        reward_list = [reward[agent_id] for agent_id in self.agent_ids]
        reward_tensor = torch.stack(
            [
                torch.as_tensor(r, dtype=torch.float32, device="cpu")
                for r in reward_list
            ],
            dim=1,
        )
        current_step_data["rewards"] = reward_tensor.reshape(
            self.num_envs, len(self.agent_ids)
        )

        # Dones
        done_tensor = torch.as_tensor(done, dtype=torch.bool, device="cpu")
        current_step_data["dones"] = done_tensor.reshape(self.num_envs)

        # Values
        value_list = [value[agent_id] for agent_id in self.agent_ids]
        value_tensor = torch.stack(
            [torch.as_tensor(v, dtype=torch.float32, device="cpu") for v in value_list],
            dim=1,
        )
        current_step_data["values"] = value_tensor.reshape(
            self.num_envs, len(self.agent_ids)
        )

        # Log_probs
        log_prob_list = [log_prob[agent_id] for agent_id in self.agent_ids]
        log_prob_tensor = torch.stack(
            [
                torch.as_tensor(lp, dtype=torch.float32, device="cpu")
                for lp in log_prob_list
            ],
            dim=1,
        )
        current_step_data["log_probs"] = log_prob_tensor.reshape(
            self.num_envs, len(self.agent_ids)
        )

        # Next Observations
        if next_obs is not None:
            next_obs_dict = OrderedDict()
            for agent_id, agent_next_obs in next_obs.items():
                obs_space = self.observation_spaces[agent_id]
                if isinstance(obs_space, spaces.Dict):
                    agent_next_obs_dict = OrderedDict()
                    for key, item in agent_next_obs.items():
                        sub_space = obs_space.spaces[key]
                        next_obs_tensor = torch.as_tensor(item, device="cpu")
                        agent_next_obs_dict[key] = self._maybe_reshape_obs(
                            next_obs_tensor, sub_space
                        )
                    next_obs_dict[agent_id] = agent_next_obs_dict
                else:
                    next_obs_tensor = torch.as_tensor(agent_next_obs, device="cpu")
                    next_obs_dict[agent_id] = self._maybe_reshape_obs(
                        next_obs_tensor, obs_space
                    )
            current_step_data["next_observations"] = next_obs_dict

        # Episode Starts
        if episode_start is not None:
            episode_start_tensor = torch.as_tensor(
                episode_start,
                dtype=torch.bool,
                device="cpu",
            )
            current_step_data["episode_starts"] = episode_start_tensor.reshape(
                self.num_envs
            )
        else:
            current_step_data["episode_starts"] = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device="cpu",
            )

        # Action Masks (lazily allocated on first use)
        if action_mask is not None:
            action_mask_dict = OrderedDict()
            for agent_id, agent_mask in action_mask.items():
                action_mask_tensor = torch.as_tensor(
                    agent_mask, dtype=torch.bool, device="cpu"
                ).reshape(self.num_envs, -1)
                action_mask_dict[agent_id] = action_mask_tensor

            if "action_masks" not in self.buffer.keys():
                mask_dict = OrderedDict()
                for agent_id, agent_mask_tensor in action_mask_dict.items():
                    mask_size = agent_mask_tensor.shape[-1]
                    mask_dict[agent_id] = torch.ones(
                        (self.capacity, self.num_envs, mask_size),
                        dtype=torch.bool,
                    )
                self.buffer["action_masks"] = mask_dict

            current_step_data["action_masks"] = action_mask_dict

        # Hidden States
        if self.recurrent and hidden_state is not None:
            hidden_states_dict = OrderedDict()
            for agent_id, ppo_tensor_val in hidden_state.items():
                if agent_id in self.hidden_state_architecture:
                    hidden_states_dict[agent_id] = ppo_tensor_val.permute(
                        1, 0, 2
                    )  # Shape: (num_envs, layers, size)
            current_step_data["hidden_states"] = hidden_states_dict

        # Create a TensorDict for the current step's data
        current_td_slice = TensorDict(
            current_step_data,
            batch_size=[self.num_envs],
            device="cpu",
        )

        # Assign this slice to the buffer at the current position
        self.buffer[self.pos] = current_td_slice

        # Update buffer position
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: dict[str, ArrayOrTensor],
        last_done: ArrayOrTensor,
    ) -> None:
        """Compute returns and advantages for the stored experiences using GAE or Monte Carlo.

        :param last_value: Value estimate for the last observation in each environment for each agent (shape: {agent_id: (num_envs,)})
        :type last_value: dict[str, ArrayOrTensor]
        :param last_done: Done flag for the last state in each environment (shape: (num_envs,))
        :type last_done: ArrayOrTensor
        """
        # Convert inputs to tensors
        if isinstance(last_done, torch.Tensor):
            last_done_np = last_done.cpu().numpy().reshape(self.num_envs)
        else:
            last_done_np = np.asarray(last_done).reshape(self.num_envs)

        buffer_size = self.capacity if self.full else self.pos

        # Temporary numpy arrays for computation
        advantages_np = np.zeros(
            (buffer_size, self.num_envs, len(self.agent_ids)), dtype=np.float32
        )
        returns_np = np.zeros(
            (buffer_size, self.num_envs, len(self.agent_ids)), dtype=np.float32
        )

        # Get necessary data from TensorDict as numpy arrays
        rewards_np = self.buffer["rewards"][:buffer_size].cpu().numpy()
        dones_np = self.buffer["dones"][:buffer_size].cpu().numpy()
        values_np = self.buffer["values"][:buffer_size].cpu().numpy()

        if self.use_gae:
            last_gae_lambdas = np.zeros(
                (self.num_envs, len(self.agent_ids)), dtype=np.float32
            )
            for t in reversed(range(buffer_size)):
                if t == buffer_size - 1:
                    next_non_terminal = 1.0 - last_done_np.astype(float)
                    next_values = np.array(
                        [last_value[agent_id] for agent_id in self.agent_ids]
                    ).T
                else:
                    next_non_terminal = 1.0 - dones_np[t + 1].astype(float)
                    next_values = values_np[t + 1]

                delta = (
                    rewards_np[t]
                    + self.gamma * next_values * next_non_terminal[:, None]
                    - values_np[t]
                )
                advantages_np[t] = last_gae_lambdas = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * next_non_terminal[:, None]
                    * last_gae_lambdas
                )
            returns_np = advantages_np + values_np
        else:
            # Monte Carlo returns
            last_returns_np = (
                np.array([last_value[agent_id] for agent_id in self.agent_ids]).T
                * (1.0 - last_done_np.astype(float))[:, None]
            )
            for t in reversed(range(buffer_size)):
                returns_np[t] = last_returns_np = (
                    rewards_np[t]
                    + self.gamma
                    * last_returns_np
                    * (1.0 - dones_np[t].astype(float))[:, None]
                )
            advantages_np = returns_np - values_np

        # Assign computed advantages and returns back to the TensorDict
        self.buffer["advantages"][:buffer_size] = torch.from_numpy(advantages_np)
        self.buffer["returns"][:buffer_size] = torch.from_numpy(returns_np)

    def get(
        self,
        batch_size: int | None = None,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Get data from the buffer, flattened and optionally sampled into minibatches.

        :param batch_size: Size of the minibatch to sample. If None, returns all data. Defaults to None.
        :type batch_size: int | None
        :return: Dictionary containing flattened buffer data arrays.
        :rtype: dict[str, np.ndarray | dict[str, np.ndarray]]
        """
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            return {}

        # Get a view of the buffer up to the current position
        valid_buffer_data: TensorDict = self.buffer[:buffer_size]

        # Reshape to flatten the num_envs dimension
        flattened_td = valid_buffer_data.view(-1)

        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer size {total_samples}. Returning all data.",
                    stacklevel=2,
                )
                return self._convert_td_to_np_dict(flattened_td)

            indices = np.random.choice(total_samples, size=batch_size, replace=False)
            sampled_td = flattened_td[indices]
            return self._convert_td_to_np_dict(sampled_td)
        return self._convert_td_to_np_dict(flattened_td)

    def get_tensor_batch(
        self,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get data from the buffer as PyTorch tensors, flattened and optionally sampled.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None.
        :type batch_size: int | None
        :param device: Device to put tensors on, defaults to None (uses self.device).
        :type device: str | None
        :return: TensorDict containing the data.
        :rtype: dict[str, torch.Tensor | dict[str, torch.Tensor]]
        """
        target_device = device or self.device
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            return {}

        # Get a view of the buffer up to the current position
        valid_buffer_data_view: TensorDict = self.buffer[:buffer_size]

        # Reshape to flatten the num_envs dimension
        flattened_td: TensorDict = valid_buffer_data_view.view(-1)

        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer_size {total_samples}. "
                    "Returning all data.",
                    stacklevel=2,
                )
                return flattened_td.to(target_device)

            indices = torch.randperm(total_samples, device="cpu")[:batch_size]
            sampled_td: TensorDict = flattened_td[indices]

            return sampled_td.to(target_device)
        return flattened_td.to(target_device)

    def _convert_td_to_np_dict(
        self,
        td: TensorDict,
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Convert a TensorDict to a dictionary of numpy arrays.

        :param td: TensorDict to convert.
        :type td: TensorDict
        :return: Dictionary of numpy arrays.
        :rtype: dict[str, np.ndarray | dict[str, np.ndarray]]
        """
        np_dict = {}
        for key, value in td.items():
            if key in [
                "observations",
                "next_observations",
                "actions",
                "action_masks",
            ] and isinstance(value, dict):
                np_dict[key] = {k: v.cpu().numpy() for k, v in value.items()}
            elif key == "hidden_states" and isinstance(value, dict):
                np_dict[key] = {k: v.cpu().numpy() for k, v in value.items()}
            else:
                np_dict[key] = value.cpu().numpy()
        return np_dict

    # BPTT methods would need to be adapted for multi-agent, but for now we'll skip them
    # as the single-agent implementation is complex and may not be immediately needed

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

        if not hasattr(self, "buffer") or not isinstance(self.buffer, TensorDict):
            warnings.warn(
                "TensorDict buffer not found or invalid during unpickling. Re-initializing.",
                stacklevel=2,
            )
            if (
                not hasattr(self, "observation_spaces")
                or not hasattr(self, "action_spaces")
                or not hasattr(self, "agent_ids")
            ):
                warnings.warn(
                    "Spaces or agent_ids missing during MultiAgentRolloutBuffer unpickling. Buffer might be invalid.",
                    stacklevel=2,
                )
                return

            self._initialize_buffers()
            self.buffer = self.buffer.to("cpu")
        else:
            expected_batch_size = torch.Size([self.capacity, self.num_envs])
            if self.buffer.batch_size != expected_batch_size:
                warnings.warn(
                    f"Loaded TensorDict has batch_size {self.buffer.batch_size}, expected {expected_batch_size}. Re-initializing.",
                    stacklevel=2,
                )
                self._initialize_buffers()
