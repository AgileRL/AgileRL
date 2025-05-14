import random  # Added to support random sequence sampling for BPTT
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

from agilerl.typing import ArrayOrTensor, ObservationType


class RolloutBuffer:
    """
    Rollout buffer for collecting experiences and computing advantages for RL algorithms.
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
    :param hidden_state_size: Size of hidden states if used, defaults to None.
    :type hidden_state_size: int, optional
    :param use_gae: Whether to compute GAE advantages, defaults to True.
    :type use_gae: bool, optional
    :param wrap_at_capacity: Whether to wrap the buffer at capacity, defaults to False. This is especially useful for OFF-policy algorithms, ON-policy algorithms should leave this as False in most cases.
    :type wrap_at_capacity: bool, optional
    :param max_seq_len: Maximum sequence length for BPTT, defaults to None.
    :type max_seq_len: int, optional
    """

    hidden_states: Optional[Dict[str, np.ndarray]] = None
    next_hidden_states: Optional[Dict[str, np.ndarray]] = None

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
        hidden_state_architecture: Optional[Dict[str, Tuple[int, int, int]]] = None,
        use_gae: bool = True,
        wrap_at_capacity: bool = False,
        max_seq_len: Optional[int] = None,  # Maximum sequence length for BPTT
    ):
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
        self.max_seq_len = (
            max_seq_len  # Store maximum sequence length for BPTT sampling
        )

        self.pos = 0
        self.full = False
        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """Initialize buffer arrays with correct shapes for vectorized environments."""
        if isinstance(self.observation_space, spaces.Discrete):
            obs_shape = (1,)
        elif isinstance(self.observation_space, spaces.Box):
            obs_shape = self.observation_space.shape
        elif isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            # Assuming Dict/Tuple spaces are handled element-wise later
            obs_shape = ()  # Placeholder, will be determined by actual data
        else:
            obs_shape = self.observation_space.shape

        # Initialize buffers with zeros, adding the num_envs dimension
        # Note: Handling Dict/Tuple spaces requires dynamic shape inference or preprocessing
        if obs_shape:
            self.observations = np.zeros(
                (self.capacity, self.num_envs, *obs_shape), dtype=np.float32
            )
            self.next_observations = np.zeros(
                (self.capacity, self.num_envs, *obs_shape), dtype=np.float32
            )
        else:  # Placeholder for Dict/Tuple - consider initializing with None or object dtype
            self.observations = np.empty((self.capacity, self.num_envs), dtype=object)
            self.next_observations = np.empty(
                (self.capacity, self.num_envs), dtype=object
            )

        self.episode_starts = np.zeros((self.capacity, self.num_envs), dtype=np.bool_)
        self.rewards = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_envs), dtype=np.bool_)
        self.values = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.capacity, self.num_envs), dtype=np.float32)

        # Initialize action buffer based on action space type
        if isinstance(self.action_space, spaces.Discrete):
            action_shape = ()
            action_dtype = np.int64
        elif isinstance(self.action_space, spaces.Box):
            action_shape = self.action_space.shape
            action_dtype = np.float32
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_shape = (len(self.action_space.nvec),)
            action_dtype = np.int64
        elif isinstance(self.action_space, spaces.MultiBinary):
            action_shape = (self.action_space.n,)
            action_dtype = np.int64
        else:
            # Attempt to handle other spaces, assuming they have a shape attribute
            try:
                action_shape = self.action_space.shape
                action_dtype = getattr(
                    self.action_space, "dtype", np.float32
                )  # Use float32 as default
            except AttributeError:
                raise TypeError(
                    f"Unsupported action space type without shape: {type(self.action_space)}"
                )

        # Initialize actions buffer based on determined shape and dtype
        # For Discrete spaces, action_shape is now empty, resulting in shape (capacity, num_envs)
        self.actions = np.zeros(
            (self.capacity, self.num_envs, *action_shape), dtype=action_dtype
        )

        if self.recurrent:
            # Initialize hidden states buffer as dict of numpy arrays
            if self.hidden_state_architecture is None:
                raise ValueError(
                    "hidden_state_architecture must be provided if recurrent=True"
                )

            # preallocate hidden states
            self.hidden_states = {
                key: np.zeros(
                    (self.capacity, *self.hidden_state_architecture[key]),
                    dtype=np.float32,
                )
                for key in self.hidden_state_architecture.keys()
            }
            self.next_hidden_states = {
                key: np.zeros(
                    (self.capacity, *self.hidden_state_architecture[key]),
                    dtype=np.float32,
                )
                for key in self.hidden_state_architecture.keys()
            }

    def add(
        self,
        obs: ObservationType,
        action: ArrayOrTensor,
        reward: Union[float, np.ndarray],
        done: Union[bool, np.ndarray],
        value: Union[float, np.ndarray],
        log_prob: Union[float, np.ndarray],
        next_obs: Optional[ObservationType] = None,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
        next_hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
        episode_start: Optional[Union[bool, np.ndarray]] = None,
    ) -> None:
        """
        Add a new batch of observations and associated data from vectorized environments to the buffer.

        :param obs: Current observation batch (shape: (num_envs, *obs_shape))
        :param action: Action batch taken (shape: (num_envs, *action_shape))
        :param reward: Reward batch received (shape: (num_envs,))
        :param done: Done flag batch (shape: (num_envs,))
        :param value: Value estimate batch (shape: (num_envs,))
        :param log_prob: Log probability batch of the actions (shape: (num_envs,))
        :param next_obs: Next observation batch (shape: (num_envs, *obs_shape)), defaults to None
        :param hidden_state: Current hidden state batch (shape: (num_envs, hidden_size)), defaults to None
        :param next_hidden_state: Next hidden state batch (shape: (num_envs, hidden_size)), defaults to None
        :param episode_start: Episode start flag batch (shape: (num_envs,)), defaults to None
        """
        if self.pos == self.capacity:
            if self.wrap_at_capacity:
                self.pos = 0
            else:
                raise ValueError(
                    f"Buffer has reached capacity ({self.capacity} transitions) but received more transitions. Either increase buffer capacity or set wrap_at_capacity=True."
                )

        if self.num_envs == 1:
            obs = np.expand_dims(obs, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = np.expand_dims(done, axis=0)
            value = np.expand_dims(value, axis=0)
            log_prob = np.expand_dims(log_prob, axis=0)

        # Convert tensors to numpy arrays if needed
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if next_obs is not None and isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu().numpy()
        if isinstance(done, torch.Tensor):
            done = done.cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.cpu().numpy()
        if hidden_state is not None and isinstance(hidden_state, torch.Tensor):
            hidden_state = hidden_state.cpu().numpy()
        if next_hidden_state is not None and isinstance(
            next_hidden_state, torch.Tensor
        ):
            next_hidden_state = next_hidden_state.cpu().numpy()
        if episode_start is not None and isinstance(episode_start, torch.Tensor):
            episode_start = episode_start.cpu().numpy()

        # Ensure inputs are at least 1D arrays for consistent batch handling
        reward = np.atleast_1d(reward)
        done = np.atleast_1d(done)
        value = np.atleast_1d(value)
        log_prob = np.atleast_1d(log_prob)
        if episode_start is not None:
            episode_start = np.atleast_1d(episode_start)

        # Check if input dimensions match num_envs
        if obs is not None and obs.shape[0] != self.num_envs:
            raise ValueError(
                f"Observation batch size {obs.shape[0]} does not match num_envs {self.num_envs}"
            )
        # Add similar checks for action, reward, done, value, log_prob, next_obs if needed

        # Store the data batch at the current position
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        if next_obs is not None:
            self.next_observations[self.pos] = next_obs

        if episode_start is not None:
            self.episode_starts[self.pos] = episode_start
        else:
            # Default episode_start to False if not provided
            self.episode_starts[self.pos] = np.zeros(self.num_envs, dtype=bool)

        # Store hidden states if enabled
        if self.recurrent:
            # Use hidden_state_architecture to validate hidden state shapes
            if hidden_state is not None:
                for key in self.hidden_state_architecture.keys():
                    state = hidden_state.get(key)
                    if state is None:
                        raise ValueError(f"Hidden state missing key: {key}")
                    # Convert state to numpy if it's a tensor
                    if isinstance(state, torch.Tensor):
                        state = state.cpu().numpy()
                    # Get expected shape from hidden_state_architecture
                    expected_shape = self.hidden_state_architecture[key]
                    if state.shape != expected_shape:
                        raise ValueError(
                            f"Hidden state['{key}'] shape {state.shape} does not match expected {expected_shape}"
                        )
                    self.hidden_states[key][self.pos] = state

            if next_hidden_state is not None:
                for key in self.hidden_state_architecture.keys():
                    next_state = next_hidden_state.get(key)
                    if next_state is None:
                        raise ValueError(f"Next hidden state missing key: {key}")
                    # Convert state to numpy if it's a tensor
                    if isinstance(next_state, torch.Tensor):
                        next_state = next_state.cpu().numpy()
                    # Get expected shape from hidden_state_architecture
                    expected_shape = self.hidden_state_architecture[key]
                    if next_state.shape != expected_shape:
                        raise ValueError(
                            f"Next hidden state['{key}'] shape {next_state.shape} does not match expected {expected_shape}"
                        )
                    self.next_hidden_states[key][self.pos] = next_state

        # Update buffer position
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: ArrayOrTensor, last_done: ArrayOrTensor
    ) -> None:
        """
        Compute returns and advantages for the stored experiences using GAE or Monte Carlo.

        :param last_value: Value estimate for the last observation in each environment (shape: (num_envs,))
        :param last_done: Done flag for the last state in each environment (shape: (num_envs,))
        """
        # Convert tensors to numpy
        if isinstance(last_value, torch.Tensor):
            last_value = last_value.cpu().numpy()
        if isinstance(last_done, torch.Tensor):
            last_done = last_done.cpu().numpy()

        # Ensure last_value and last_done have the correct shape
        last_value = np.atleast_1d(last_value).reshape(self.num_envs)
        last_done = np.atleast_1d(last_done).reshape(self.num_envs)

        buffer_size = self.capacity if self.full else self.pos

        if self.use_gae:
            last_gae_lambda = np.zeros(self.num_envs, dtype=np.float32)
            for t in reversed(range(buffer_size)):
                if t == buffer_size - 1:
                    next_non_terminal = 1.0 - last_done.astype(float)
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1].astype(float)
                    next_values = self.values[t + 1]

                delta = (
                    self.rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - self.values[t]
                )
                self.advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )
            self.returns = (
                self.advantages[:buffer_size] + self.values[:buffer_size]
            )  # Only assign up to current size
        else:
            # Monte Carlo returns
            last_returns = last_value * (1.0 - last_done.astype(float))
            for t in reversed(range(buffer_size)):
                self.returns[t] = last_returns = self.rewards[
                    t
                ] + self.gamma * last_returns * (1.0 - self.dones[t].astype(float))

            self.advantages = (
                self.returns - self.values[:buffer_size]
            )  # Only assign up to current size

    def get(
        self, batch_size: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Get data from the buffer, flattened and optionally sampled into minibatches.

        :param batch_size: Size of the minibatch to sample. If None, returns all data. Defaults to None.
        :return: Dictionary containing flattened buffer data arrays.
        """
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            # Return empty dictionary or raise error if buffer is empty
            return {}

        # Flatten the data across time and environment dimensions
        flattened_data = {}
        # Handle potential object dtype for obs/next_obs if Dict/Tuple spaces were used
        if self.observations.dtype == object:
            # Need careful stacking if obs are dicts/tuples
            # This is a simplified placeholder - real implementation might need more logic
            flattened_data["observations"] = np.concatenate(
                self.observations[:buffer_size].ravel()
            )
            if self.next_observations is not None:
                flattened_data["next_observations"] = np.concatenate(
                    self.next_observations[:buffer_size].ravel()
                )
        else:
            obs_shape = self.observations.shape[2:]
            flattened_data["observations"] = self.observations[:buffer_size].reshape(
                total_samples, *obs_shape
            )
            if self.next_observations is not None:
                flattened_data["next_observations"] = self.next_observations[
                    :buffer_size
                ].reshape(total_samples, *obs_shape)

        action_shape = self.actions.shape[2:]
        flattened_data["actions"] = self.actions[:buffer_size].reshape(
            total_samples, *action_shape
        )
        flattened_data["rewards"] = self.rewards[:buffer_size].reshape(total_samples)
        flattened_data["dones"] = self.dones[:buffer_size].reshape(total_samples)
        flattened_data["values"] = self.values[:buffer_size].reshape(total_samples)
        flattened_data["log_probs"] = self.log_probs[:buffer_size].reshape(
            total_samples
        )
        flattened_data["advantages"] = self.advantages[:buffer_size].reshape(
            total_samples
        )
        flattened_data["returns"] = self.returns[:buffer_size].reshape(total_samples)
        flattened_data["episode_starts"] = self.episode_starts[:buffer_size].reshape(
            total_samples
        )

        if self.recurrent:
            flattened_data["hidden_states"] = {}
            flattened_data["next_hidden_states"] = {}
            total_samples = (
                buffer_size * self.num_envs
            )  # Recalculate or ensure it's available

            for key in self.hidden_state_architecture.keys():
                if self.hidden_states is not None and key in self.hidden_states:
                    # Shape: (buffer_size, 1, num_envs, hidden_size)
                    data = self.hidden_states[key][:buffer_size]
                    # Swap num_envs and layer_dim axes, then reshape
                    flattened_data["hidden_states"][key] = data.swapaxes(1, 2).reshape(
                        total_samples,
                        self.hidden_state_architecture[key][0],
                        self.hidden_state_architecture[key][2],
                    )

                if (
                    self.next_hidden_states is not None
                    and key in self.next_hidden_states
                ):
                    # Shape: (buffer_size, 1, num_envs, hidden_size)
                    next_data = self.next_hidden_states[key][:buffer_size]
                    # Swap num_envs and layer_dim axes, then reshape
                    flattened_data["next_hidden_states"][key] = next_data.swapaxes(
                        1, 2
                    ).reshape(
                        total_samples,
                        self.hidden_state_architecture[key][0],
                        self.hidden_state_architecture[key][2],
                    )

        # Sample a minibatch if batch_size is specified
        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer size {total_samples}. Returning all data."
                )
                indices = np.arange(total_samples)
            else:
                indices = np.random.choice(
                    total_samples, size=batch_size, replace=False
                )

            # Sample from flattened data, handling the dict structure for hidden states
            sampled_batch = {}
            for key, data in flattened_data.items():
                if key in ["hidden_states", "next_hidden_states"] and isinstance(
                    data, dict
                ):
                    sampled_batch[key] = {
                        sub_key: sub_data[indices] for sub_key, sub_data in data.items()
                    }
                elif isinstance(data, np.ndarray):
                    sampled_batch[key] = data[indices]
                else:
                    sampled_batch[key] = data  # Should not happen ideally

            return sampled_batch
        else:
            # Return all flattened data
            return flattened_data

    def get_tensor_batch(
        self, batch_size: Optional[int] = None, device: Optional[str] = None
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get data from the buffer as PyTorch tensors, flattened and optionally sampled.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None.
        :param device: Device to put tensors on, defaults to None (uses self.device).
        :return: Dictionary with tensor data.
        """
        np_batch = self.get(batch_size)
        device = device or self.device

        tensor_batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        for key, data in np_batch.items():
            if key in ["hidden_states", "next_hidden_states"] and isinstance(
                data, dict
            ):
                tensor_batch[key] = {
                    sub_key: torch.from_numpy(sub_data).to(device)
                    for sub_key, sub_data in data.items()
                }
            elif isinstance(data, np.ndarray):
                # Handle potential object dtype for observations (Dict/Tuple)
                # This requires knowing how to convert the specific dict/tuple structure to tensors
                if data.dtype == object:
                    # Placeholder: Assumes data is a list/array of dicts/tuples that can be processed
                    # This part needs custom logic based on the actual structure
                    # Example: Convert list of dicts to a dict of tensors
                    if isinstance(data[0], dict):
                        tensor_batch[key] = {
                            k: torch.stack([torch.from_numpy(d[k]) for d in data]).to(
                                device
                            )
                            for k in data[0]
                        }
                    # Add handling for tuples if needed
                    else:
                        # Fallback or raise error if structure not handled
                        warnings.warn(
                            f"Cannot automatically convert object array '{key}' to tensor. Skipping."
                        )
                        tensor_batch[key] = (
                            data  # Keep as numpy object array? Or handle specific types
                        )
                else:
                    tensor_batch[key] = torch.from_numpy(data).to(device)
            else:
                # Should not happen if get() returns numpy arrays/dicts, but handle just in case
                tensor_batch[key] = data  # Or raise error?

        return tensor_batch

    def size(self) -> int:
        """
        Get current number of transitions stored in the buffer.

        :return: Current number of transitions.
        """
        return (self.capacity if self.full else self.pos) * self.num_envs

    def reset(self) -> None:
        """Reset the buffer pointer and full flag."""
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # New helper functions for truncated Backpropagation Through Time (BPTT)
    # ------------------------------------------------------------------

    def _sample_sequence_start_indices(
        self, seq_len: int, batch_size: int
    ) -> List[Tuple[int, int]]:
        """Helper to randomly sample (time_idx, env_idx) pairs that can serve as
        the starting positions for sequences of length ``seq_len``.

        :param seq_len: Desired sequence length.
        :param batch_size: Number of sequences to sample.
        :return: List of tuples (time_idx, env_idx) identifying the first element
                 of every sampled sequence.
        """
        buffer_size = self.capacity if self.full else self.pos

        if seq_len > buffer_size:
            raise ValueError(
                f"Requested sequence length {seq_len} exceeds current buffer size {buffer_size}."
            )

        # Compute valid starting indices along the time dimension
        max_start = buffer_size - seq_len
        valid_pairs: List[Tuple[int, int]] = [
            (t, env) for env in range(self.num_envs) for t in range(max_start + 1)
        ]

        if batch_size is None or batch_size >= len(valid_pairs):
            return valid_pairs  # Return all pairs if batch_size too large / None

        return random.sample(valid_pairs, batch_size)

    def get_sequences(
        self,
        seq_len: int,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Returns a dictionary with batched sequences suitable for truncated BPTT.

        The returned arrays have an additional leading batch dimension of size
        ``batch_size`` and a time dimension of size ``seq_len``.

        :param seq_len: Length of each sequence in timesteps.
        :param batch_size: Number of sequences to sample. If None, returns all
                           possible sequences.
        :return: Dictionary mirroring :pyfunc:`get`, but with sequences.
        """
        # Sample starting indices
        start_indices = self._sample_sequence_start_indices(seq_len, batch_size)

        actual_batch_size = len(start_indices)

        if actual_batch_size == 0:
            return {}

        # Helper to stack sequences along batch dimension
        def _stack_seq(array: np.ndarray) -> np.ndarray:
            seqs = [
                array[t : t + seq_len, env_idx] for t, env_idx in start_indices
            ]  # Each with shape (seq_len, *dims)
            return np.stack(seqs, axis=0)  # (batch, seq_len, *dims)

        seq_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}

        # Handle observations (possibly object dtype)
        if self.observations.dtype == object:
            seq_data["observations"] = np.array(
                [
                    np.concatenate(self.observations[t : t + seq_len, env_idx])
                    for t, env_idx in start_indices
                ],
                dtype=object,
            )
        else:
            seq_data["observations"] = _stack_seq(self.observations)

        if self.next_observations is not None:
            if self.next_observations.dtype == object:
                seq_data["next_observations"] = np.array(
                    [
                        np.concatenate(self.next_observations[t : t + seq_len, env_idx])
                        for t, env_idx in start_indices
                    ],
                    dtype=object,
                )
            else:
                seq_data["next_observations"] = _stack_seq(self.next_observations)

        # Scalar / vector data
        for name, array in [
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
            ("values", self.values),
            ("log_probs", self.log_probs),
            ("advantages", self.advantages),
            ("returns", self.returns),
            ("episode_starts", self.episode_starts),
        ]:
            seq_data[name] = _stack_seq(array)

        # Handle recurrent states
        if self.recurrent and self.hidden_states is not None:
            seq_data["initial_hidden_states"] = {}
            for key in self.hidden_state_architecture.keys():
                # Shape in buffer: (time, layer, env, hidden)
                init_h = np.stack(
                    [
                        self.hidden_states[key][t, :, env_idx]
                        for t, env_idx in start_indices
                    ],
                    axis=0,
                )  # (batch, layer, hidden)
                seq_data["initial_hidden_states"][key] = init_h

        return seq_data

    def get_sequence_tensor_batch(
        self,
        seq_len: int,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Same as :pyfunc:`get_sequences` but returns PyTorch tensors on the
        specified device.

        :param seq_len: Length of each sequence.
        :param batch_size: Number of sequences to sample. If None, returns all.
        :param device: Torch device to move tensors to. Defaults to
                       :pyattr:`self.device`.
        :return: Dictionary with tensor sequences.
        """
        device = device or self.device
        np_batch = self.get_sequences(seq_len=seq_len, batch_size=batch_size)

        tensor_batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        for key, data in np_batch.items():
            if key == "initial_hidden_states" and isinstance(data, dict):
                tensor_batch[key] = {
                    k: torch.from_numpy(v).to(device) for k, v in data.items()
                }
            elif isinstance(data, dict):
                tensor_batch[key] = {
                    k: torch.from_numpy(v).to(device) for k, v in data.items()
                }
            elif isinstance(data, np.ndarray):
                if data.dtype == object:
                    # See note in get_tensor_batch
                    warnings.warn(
                        f"Cannot automatically convert object array '{key}' to tensor. Skipping."
                    )
                    tensor_batch[key] = data
                else:
                    tensor_batch[key] = torch.from_numpy(data).to(device)
            else:
                tensor_batch[key] = data

        return tensor_batch

    def get_specific_sequences_tensor_batch(
        self,
        seq_len: int,
        sequence_coords: List[Tuple[int, int]], # List of (env_idx, time_idx_in_env_rollout)
        device: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Returns a dictionary with batched sequences for specific, pre-determined
        starting coordinates, as PyTorch tensors.

        The returned arrays have a leading batch dimension (size = len(sequence_coords))
        and a time dimension of size ``seq_len``.

        :param seq_len: Length of each sequence in timesteps.
        :param sequence_coords: A list of (env_idx, time_idx) tuples.
                                Each tuple specifies the starting environment and time
                                for a sequence within that environment's rollout.
        :param device: Torch device to move tensors to. Defaults to self.device.
        :return: Dictionary with tensor sequences.
        """
        device = device or self.device
        actual_batch_size = len(sequence_coords)

        if actual_batch_size == 0:
            return {}

        # Helper to stack sequences along batch dimension for specific coordinates
        def _stack_specific_seq(array: np.ndarray) -> np.ndarray:
            # array has shape (capacity, num_envs, *dims)
            # sequence_coords are (env_idx, time_idx_in_env_rollout)
            # We need to slice: array[time_idx : time_idx + seq_len, env_idx]
            seqs = [
                array[time_idx : time_idx + seq_len, env_idx] for env_idx, time_idx in sequence_coords
            ]  # Each element is (seq_len, *dims)
            return np.stack(seqs, axis=0)  # (batch_size, seq_len, *dims)

        seq_data_np: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}

        # Handle observations
        if self.observations.dtype == object:
            seq_data_np["observations"] = np.array(
                [
                    np.concatenate(self.observations[time_idx : time_idx + seq_len, env_idx])
                    for env_idx, time_idx in sequence_coords
                ],
                dtype=object,
            )
        else:
            seq_data_np["observations"] = _stack_specific_seq(self.observations)

        # Handle next_observations if they exist
        if hasattr(self, 'next_observations') and self.next_observations is not None:
            if self.next_observations.dtype == object:
                seq_data_np["next_observations"] = np.array(
                    [
                        np.concatenate(self.next_observations[time_idx : time_idx + seq_len, env_idx])
                        for env_idx, time_idx in sequence_coords
                    ],
                    dtype=object,
                )
            else:
                seq_data_np["next_observations"] = _stack_specific_seq(self.next_observations)


        # Scalar / vector data
        for name, array_attr_name in [
            ("actions", "actions"),
            ("rewards", "rewards"),
            ("dones", "dones"),
            ("values", "values"),
            ("log_probs", "log_probs"),
            ("advantages", "advantages"),
            ("returns", "returns"),
            ("episode_starts", "episode_starts"),
        ]:
            array = getattr(self, array_attr_name)
            seq_data_np[name] = _stack_specific_seq(array)

        # Handle recurrent states (initial hidden state for each sequence)
        if self.recurrent and self.hidden_states is not None:
            seq_data_np["initial_hidden_states"] = {}
            for key in self.hidden_state_architecture.keys():
                # self.hidden_states[key] has shape (capacity, num_layers, num_envs, hidden_dim_per_layer)
                # We need hidden_states[key][time_idx, :, env_idx, :]
                init_h_list = []
                for env_idx, time_idx in sequence_coords:
                    # Correct slicing for hidden states:
                    # hidden_states[key] has shape e.g. (capacity, layers, envs, size)
                    # We need the state at time_idx for env_idx.
                    # It should be (layers, size) for that specific (time_idx, env_idx)
                    # The batch dimension for hidden states in RolloutBuffer.add is (num_envs, layers, hidden_size)
                    # So, in self.hidden_states[key], it's (capacity, num_envs, layers, hidden_size) if architecture is (layers, 1, hidden_size)
                    # Or (capacity, layers, num_envs, hidden_size) if architecture is (layers, num_envs, hidden_size) - check add()
                    # Current RolloutBuffer.add expects hidden_state of shape (num_envs, layers, size) for each key in the dict
                    # And stores it as self.hidden_states[key][pos] = state (where state is (num_envs, layers, size))
                    # So self.hidden_states[key] is (capacity, num_envs, layers, size)

                    # The target shape for PPO BPTT's initial_hidden_states for a batch of sequences is:
                    # (batch_size_of_sequences, num_layers, hidden_size_per_layer)
                    # init_h_list.append(self.hidden_states[key][time_idx, env_idx, :, :])
                    # Corrected slicing:
                    init_h_list.append(self.hidden_states[key][time_idx, :, env_idx, :])

                seq_data_np["initial_hidden_states"][key] = np.stack(init_h_list, axis=0)


        # Convert numpy batch to tensor batch
        tensor_batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        for key, data in seq_data_np.items():
            if key == "initial_hidden_states" and isinstance(data, dict):
                tensor_batch[key] = {
                    k: torch.from_numpy(v).to(device) for k, v in data.items()
                }
            elif isinstance(data, dict): # For observations if they are dicts
                 tensor_batch[key] = {
                    k: torch.from_numpy(v).to(device) for k, v in data.items()
                }
            elif isinstance(data, np.ndarray):
                if data.dtype == object:
                    warnings.warn(
                        f"Cannot automatically convert object array '{key}' in get_specific_sequences_tensor_batch to tensor. Skipping conversion for this key."
                    )
                    tensor_batch[key] = data # Keep as numpy object array
                else:
                    tensor_batch[key] = torch.from_numpy(data).to(device)
            else:
                tensor_batch[key] = data # Should not happen

        return tensor_batch

    def __getstate__(self) -> Dict[str, Any]:
        """Gets the state dictionary for pickling, ensuring arrays are copied."""
        state = self.__dict__.copy()
        # Explicitly copy numpy arrays to avoid issues with read-only views after unpickling
        buffer_size = self.capacity if self.full else self.pos
        array_keys = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "values",
            "log_probs",
            "next_observations",
            "advantages",
            "returns",
            "episode_starts",
        ]
        for key in array_keys:
            if hasattr(self, key) and getattr(self, key) is not None:
                # Slice up to current size and copy
                state[key] = np.array(getattr(self, key)[:buffer_size], copy=True)

        # Handle hidden_states dictionary (contains tensors or numpy arrays)
        if self.recurrent and self.hidden_states is not None:
            state["hidden_states"] = {}
            for key, val in self.hidden_states.items():
                # Slice up to current size and copy
                sliced_val = val[:buffer_size]
                state["hidden_states"][key] = (
                    sliced_val.clone()
                    if isinstance(sliced_val, torch.Tensor)
                    else np.array(sliced_val, copy=True)
                )

        if self.recurrent and self.next_hidden_states is not None:
            state["next_hidden_states"] = {}
            for key, val in self.next_hidden_states.items():
                # Slice up to current size and copy
                sliced_val = val[:buffer_size]
                state["next_hidden_states"][key] = (
                    sliced_val.clone()
                    if isinstance(sliced_val, torch.Tensor)
                    else np.array(sliced_val, copy=True)
                )

        # Remove attributes that might not be easily serializable or needed
        # state.pop('observation_space', None)
        # state.pop('action_space', None)
        # We need observation_space and action_space for _initialize_buffers in __setstate__

        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the state dictionary when unpickling, re-initializing buffers."""
        self.__dict__.update(state)

        # Re-initialize buffers to ensure they are correctly sized and writable
        # We need observation_space and action_space for this, ensured they are in state
        if not hasattr(self, "observation_space") or not hasattr(self, "action_space"):
            warnings.warn(
                "Observation or action space missing during RolloutBuffer unpickling. Buffer might be invalid."
            )
            return  # Cannot properly reinitialize

        self._initialize_buffers()  # Creates fresh, writable arrays

        # Restore data into the newly created buffers
        buffer_size = self.capacity if self.full else self.pos
        array_keys = [
            "observations",
            "actions",
            "rewards",
            "dones",
            "values",
            "log_probs",
            "next_observations",
            "advantages",
            "returns",
            "episode_starts",
        ]
        for key in array_keys:
            if key in state and hasattr(self, key) and getattr(self, key) is not None:
                saved_array = state[key]
                # Ensure the loaded array length matches the expected buffer size based on pos/full
                current_buffer_len = min(len(saved_array), buffer_size)
                if current_buffer_len > 0:
                    getattr(self, key)[:current_buffer_len] = saved_array[
                        :current_buffer_len
                    ]

        # Restore hidden states
        if self.recurrent:
            if "hidden_states" in state and self.hidden_states is not None:
                saved_hidden_states = state["hidden_states"]
                for key, saved_array_val in saved_hidden_states.items():
                    if key in self.hidden_states:
                        # Ensure the loaded array length matches the expected buffer size
                        current_buffer_len = min(len(saved_array_val), buffer_size)
                        if current_buffer_len > 0:
                            self.hidden_states[key][:current_buffer_len] = (
                                saved_array_val[:current_buffer_len]
                            )

            if "next_hidden_states" in state and self.next_hidden_states is not None:
                saved_next_hidden_states = state["next_hidden_states"]
                for key, saved_array_val in saved_next_hidden_states.items():
                    if key in self.next_hidden_states:
                        # Ensure the loaded array length matches the expected buffer size
                        current_buffer_len = min(len(saved_array_val), buffer_size)
                        if current_buffer_len > 0:
                            self.next_hidden_states[key][:current_buffer_len] = (
                                saved_array_val[:current_buffer_len]
                            )
