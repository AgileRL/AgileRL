from copy import deepcopy
import warnings
import random  # Added to support random sequence sampling for BPTT
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from gymnasium import spaces

from agilerl.typing import ArrayOrTensor, TorchObsType, ObservationType


def _get_space_shape(space: spaces.Space) -> Tuple[int, ...]:
    """Helper to get shape from different gym spaces."""
    if isinstance(space, spaces.Discrete):
        return (1,)
    elif isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.MultiDiscrete):
        return (len(space.nvec),)
    elif isinstance(space, spaces.MultiBinary):
        return (space.n,)
    else:
        # Fallback for spaces with shape attribute
        try:
            return space.shape
        except AttributeError:
            raise TypeError(f"Unsupported space type without shape: {type(space)}")


def _get_space_dtype(space: spaces.Space) -> np.dtype:
    """Helper to get dtype from different gym spaces."""
    if isinstance(space, spaces.Discrete):
        return np.int64
    elif isinstance(space, spaces.Box):
        return space.dtype
    elif isinstance(space, spaces.MultiDiscrete):
        return np.int64  # Match gym common practice
    elif isinstance(space, spaces.MultiBinary):
        return np.int8  # Match gym common practice
    else:
        # Fallback for spaces with dtype attribute, default to float32
        return getattr(space, "dtype", np.float32)


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
    :param recurrent_state_keys: Keys for hidden states if using dict, defaults to ["h", "c"].
    :type recurrent_state_keys: List[str], optional
    :param hidden_state_size: Size of hidden states if used, defaults to None.
    :type hidden_state_size: int, optional
    :param use_gae: Whether to compute GAE advantages, defaults to True.
    :type use_gae: bool, optional
    :param wrap_at_capacity: Whether to wrap the buffer at capacity, defaults to False. This is especially useful for OFF-policy algorithms, ON-policy algorithms should leave this as False in most cases.
    :type wrap_at_capacity: bool, optional
    :param max_seq_len: Maximum sequence length for BPTT, defaults to None.
    :type max_seq_len: int, optional
    """

    # Type hints for observation buffers which can be single array or dict of arrays
    observations: Union[np.ndarray, Dict[str, np.ndarray]]
    next_observations: Union[np.ndarray, Dict[str, np.ndarray]]
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
        recurrent_state_keys: Optional[List[str]] = ["h", "c"],
        hidden_state_size: Optional[int] = None,
        use_gae: bool = True,
        wrap_at_capacity: bool = False,
        max_seq_len: Optional[int] = None,  # Maximum sequence length for BPTT
    ):
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        self.is_dict_obs = isinstance(observation_space, spaces.Dict)
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.recurrent = recurrent
        self.recurrent_state_keys = recurrent_state_keys if recurrent else []
        self.hidden_state_size = hidden_state_size
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
        # Initialize observation buffers based on space type
        if self.is_dict_obs:
            self.observations = {}
            self.next_observations = {}
            for key, space in self.observation_space.spaces.items():
                obs_shape = _get_space_shape(space)
                obs_dtype = _get_space_dtype(space)
                self.observations[key] = np.zeros(
                    (self.capacity, self.num_envs, *obs_shape), dtype=obs_dtype
                )
                self.next_observations[key] = np.zeros(
                    (self.capacity, self.num_envs, *obs_shape), dtype=obs_dtype
                )
        else:
            obs_shape = _get_space_shape(self.observation_space)
            obs_dtype = _get_space_dtype(self.observation_space)
            self.observations = np.zeros(
                (self.capacity, self.num_envs, *obs_shape), dtype=obs_dtype
            )
            self.next_observations = np.zeros(
                (self.capacity, self.num_envs, *obs_shape), dtype=obs_dtype
            )

        # Initialize other standard buffers
        self.episode_starts = np.zeros((self.capacity, self.num_envs), dtype=np.bool_)
        self.rewards = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, self.num_envs), dtype=np.bool_)
        self.values = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.capacity, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.capacity, self.num_envs), dtype=np.float32)

        # Initialize action buffer based on action space type
        action_shape = _get_space_shape(self.action_space)
        action_dtype = _get_space_dtype(self.action_space)
        self.actions = np.zeros(
            (self.capacity, self.num_envs, *action_shape), dtype=action_dtype
        )

        # Initialize hidden state buffers if recurrent
        if self.recurrent:
            if self.hidden_state_size is None:
                raise ValueError("hidden_state_size must be provided if recurrent=True")
            if not self.recurrent_state_keys:
                raise ValueError(
                    "recurrent_state_keys must be provided if recurrent=True"
                )

            # Assuming num_layers * directions = 1 based on EvolvableLSTM implementation
            hidden_layer_shape = (1, self.num_envs, self.hidden_state_size)

            self.hidden_states = {
                key: np.zeros(
                    (self.capacity, *hidden_layer_shape),
                    dtype=np.float32,
                )
                for key in self.recurrent_state_keys
            }
            self.next_hidden_states = {
                key: np.zeros(
                    (self.capacity, *hidden_layer_shape),
                    dtype=np.float32,
                )
                for key in self.recurrent_state_keys
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
        Add a new batch of transitions to the buffer. Handles Dict observations.

        :param obs: Current observation batch. Expected type depends on `self.is_dict_obs`:
                    - If False: `np.ndarray` of shape `(num_envs, *obs_shape)`.
                    - If True: `Dict[str, np.ndarray]` where each value has shape `(num_envs, *value_shape)`.
        :param action: Action batch taken (shape: `(num_envs, *action_shape)`).
        :param reward: Reward batch received (shape: `(num_envs,)`).
        :param done: Done flag batch (shape: `(num_envs,)`).
        :param value: Value estimate batch (shape: `(num_envs,)`).
        :param log_prob: Log probability batch of the actions (shape: `(num_envs,)`).
        :param next_obs: Next observation batch. Type matches `obs`. Defaults to None.
        :param hidden_state: Current hidden state dictionary. Each value shape `(1, num_envs, hidden_size)`. Defaults to None.
        :param next_hidden_state: Next hidden state dictionary. Type matches `hidden_state`. Defaults to None.
        :param episode_start: Episode start flag batch (shape: `(num_envs,)`). Defaults to None.
        """
        if self.pos == self.capacity:
            if self.wrap_at_capacity:
                self.pos = 0
            else:
                raise ValueError(
                    f"Buffer has reached capacity ({self.capacity} transitions) but received more transitions. Either increase buffer capacity or set wrap_at_capacity=True."
                )

        # Ensure inputs are numpy arrays
        action = self._to_numpy(action)
        reward = np.atleast_1d(self._to_numpy(reward))
        done = np.atleast_1d(self._to_numpy(done))
        value = np.atleast_1d(self._to_numpy(value))
        log_prob = np.atleast_1d(self._to_numpy(log_prob)).squeeze(-1)

        if episode_start is not None:
            episode_start = np.atleast_1d(self._to_numpy(episode_start))

        # Handle observations (Dict or standard)
        if self.is_dict_obs:
            if not isinstance(obs, dict):
                raise TypeError(f"Expected obs to be a dict, got {type(obs)}")
            if next_obs is not None and not isinstance(next_obs, dict):
                raise TypeError(f"Expected next_obs to be a dict, got {type(next_obs)}")

            for key in self.observations:
                obs_value = self._to_numpy(obs[key])
                if obs_value.shape[0] != self.num_envs:
                    raise ValueError(
                        f"Observation['{key}'] batch size {obs_value.shape[0]} != num_envs {self.num_envs}"
                    )
                self.observations[key][self.pos] = obs_value
                if next_obs is not None:
                    next_obs_value = self._to_numpy(next_obs[key])
                    if next_obs_value.shape[0] != self.num_envs:
                        raise ValueError(
                            f"Next Observation['{key}'] batch size {next_obs_value.shape[0]} != num_envs {self.num_envs}"
                        )
                    self.next_observations[key][self.pos] = next_obs_value
        else:
            obs = self._to_numpy(obs)
            if obs.shape[0] != self.num_envs:
                raise ValueError(
                    f"Observation batch size {obs.shape[0]} != num_envs {self.num_envs}"
                )
            self.observations[self.pos] = obs
            if next_obs is not None:
                next_obs = self._to_numpy(next_obs)
                if next_obs.shape[0] != self.num_envs:
                    raise ValueError(
                        f"Next observation batch size {next_obs.shape[0]} != num_envs {self.num_envs}"
                    )
                self.next_observations[self.pos] = next_obs

        # Store standard data
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.episode_starts[self.pos] = (
            episode_start
            if episode_start is not None
            else np.zeros(self.num_envs, dtype=bool)
        )

        # Store hidden states if enabled
        if self.recurrent:
            expected_shape = (1, self.num_envs, self.hidden_state_size)
            if hidden_state is not None:
                if not isinstance(hidden_state, dict):
                    raise TypeError(
                        f"Expected hidden_state to be a dict, got {type(hidden_state)}"
                    )
                for key in self.recurrent_state_keys:
                    state = hidden_state.get(key)
                    if state is None:
                        raise ValueError(f"Hidden state missing key: {key}")
                    state = self._to_numpy(state)
                    if state.shape != expected_shape:
                        raise ValueError(
                            f"Hidden state['{key}'] shape {state.shape} != expected {expected_shape}"
                        )
                    self.hidden_states[key][self.pos] = state

            if next_hidden_state is not None:
                if not isinstance(next_hidden_state, dict):
                    raise TypeError(
                        f"Expected next_hidden_state to be a dict, got {type(next_hidden_state)}"
                    )
                for key in self.recurrent_state_keys:
                    next_state = next_hidden_state.get(key)
                    if next_state is None:
                        raise ValueError(f"Next hidden state missing key: {key}")
                    next_state = self._to_numpy(next_state)
                    if next_state.shape != expected_shape:
                        raise ValueError(
                            f"Next hidden state['{key}'] shape {next_state.shape} != expected {expected_shape}"
                        )
                    self.next_hidden_states[key][self.pos] = next_state

        # Update buffer position
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def _to_numpy(
        self, data: Union[ArrayOrTensor, Dict[str, ArrayOrTensor]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Convert single tensor or dict of tensors to numpy arrays."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, dict):
            return {k: self._to_numpy(v) for k, v in data.items()}
        elif isinstance(data, (np.ndarray, float, int, bool)):
            return data  # Already numpy or scalar
        else:
            # Attempt conversion for lists/tuples, assuming homogeneous elements
            try:
                return np.array(data)
            except Exception as e:
                raise TypeError(
                    f"Unsupported type for conversion to numpy: {type(data)}. Error: {e}"
                )

    def compute_returns_and_advantages(
        self, last_value: ArrayOrTensor, last_done: ArrayOrTensor
    ) -> None:
        """
        Compute returns and advantages for the stored experiences using GAE or Monte Carlo.

        :param last_value: Value estimate for the last observation in each environment (shape: (num_envs,))
        :param last_done: Done flag for the last state in each environment (shape: (num_envs,))
        """
        # Convert tensors to numpy
        last_value = self._to_numpy(last_value)
        last_done = self._to_numpy(last_done)

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
            # Ensure returns are only calculated for the filled part of the buffer
            valid_returns = self.advantages[:buffer_size] + self.values[:buffer_size]
            self.returns[:buffer_size] = valid_returns
        else:
            # Monte Carlo returns
            last_returns = last_value * (1.0 - last_done.astype(float))
            for t in reversed(range(buffer_size)):
                current_rewards = self.rewards[t]
                current_dones = self.dones[t]
                self.returns[t] = last_returns = (
                    current_rewards
                    + self.gamma * last_returns * (1.0 - current_dones.astype(float))
                )

            # Calculate advantages based on computed returns for the filled part
            valid_advantages = self.returns[:buffer_size] - self.values[:buffer_size]
            self.advantages[:buffer_size] = valid_advantages

    def get(
        self, batch_size: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Get data from the buffer, flattened and optionally sampled into minibatches.
        Handles Dict observations correctly.

        :param batch_size: Size of the minibatch to sample. If None, returns all data. Defaults to None.
        :return: Dictionary containing flattened buffer data arrays. For Dict observations,
                 'observations' and 'next_observations' are dictionaries mapping keys to flattened arrays.
        """
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            warnings.warn("Buffer is empty, returning empty dictionary.")
            return {}

        # Generate indices for sampling or for taking all data
        if batch_size is None:
            indices = np.arange(total_samples)
        elif batch_size > total_samples:
            warnings.warn(
                f"Batch size {batch_size} is larger than buffer size {total_samples}. Returning all data."
            )
            indices = np.arange(total_samples)
        else:
            indices = np.random.choice(total_samples, size=batch_size, replace=False)

        # Function to flatten and sample a standard array
        def flatten_and_sample(
            arr: np.ndarray, shape_suffix: Tuple[int, ...]
        ) -> np.ndarray:
            # Reshape (buffer_size, num_envs, *shape_suffix) -> (total_samples, *shape_suffix)
            flattened_arr = arr[:buffer_size].reshape(total_samples, *shape_suffix)
            return flattened_arr[indices]

        sampled_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}

        # Handle observations
        if self.is_dict_obs:
            sampled_data["observations"] = {}
            sampled_data["next_observations"] = {}
            for key, space in self.observation_space.spaces.items():
                obs_shape_suffix = _get_space_shape(space)
                sampled_data["observations"][key] = flatten_and_sample(
                    self.observations[key], obs_shape_suffix
                )
                sampled_data["next_observations"][key] = flatten_and_sample(
                    self.next_observations[key], obs_shape_suffix
                )
        else:
            obs_shape_suffix = _get_space_shape(self.observation_space)
            sampled_data["observations"] = flatten_and_sample(
                self.observations, obs_shape_suffix
            )
            sampled_data["next_observations"] = flatten_and_sample(
                self.next_observations, obs_shape_suffix
            )

        # Handle actions
        action_shape_suffix = _get_space_shape(self.action_space)
        sampled_data["actions"] = flatten_and_sample(self.actions, action_shape_suffix)

        # Handle scalar/1D data
        for name, arr in [
            ("rewards", self.rewards),
            ("dones", self.dones),
            ("values", self.values),
            ("log_probs", self.log_probs),
            ("advantages", self.advantages),
            ("returns", self.returns),
            ("episode_starts", self.episode_starts),
        ]:
            sampled_data[name] = flatten_and_sample(arr, ())

        # Handle recurrent states
        if self.recurrent and self.hidden_states is not None:
            sampled_data["hidden_states"] = {}
            # Shape in buffer: (buffer_size, layer=1, num_envs, hidden_size)
            # Desired output shape: (batch_size, layer=1, hidden_size)
            hidden_shape_suffix = (1, self.hidden_state_size)  # Assuming layer dim is 1

            for key in self.recurrent_state_keys:
                # Reshape to (total_samples, layer=1, hidden_size)
                flattened_hidden = (
                    self.hidden_states[key][:buffer_size]
                    .swapaxes(1, 2)
                    .reshape(total_samples, *hidden_shape_suffix)
                )
                sampled_data["hidden_states"][key] = flattened_hidden[indices]

            # Optionally handle next_hidden_states if they are stored and needed
            if self.next_hidden_states is not None:
                sampled_data["next_hidden_states"] = {}
                for key in self.recurrent_state_keys:
                    flattened_next_hidden = (
                        self.next_hidden_states[key][:buffer_size]
                        .swapaxes(1, 2)
                        .reshape(total_samples, *hidden_shape_suffix)
                    )
                    sampled_data["next_hidden_states"][key] = flattened_next_hidden[
                        indices
                    ]

        return sampled_data

    def get_tensor_batch(
        self, batch_size: Optional[int] = None, device: Optional[str] = None
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get data from the buffer as PyTorch tensors, flattened and optionally sampled.
        Handles Dict observations correctly.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None.
        :param device: Device to put tensors on, defaults to None (uses self.device).
        :return: Dictionary with tensor data. For Dict observations, 'observations' and
                 'next_observations' are dictionaries mapping keys to tensors.
        """
        np_batch = self.get(batch_size)
        target_device = torch.device(device or self.device)

        tensor_batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}

        for key, data in np_batch.items():
            if isinstance(data, dict):
                # Handles dicts for observations, hidden_states, next_hidden_states
                tensor_batch[key] = {
                    sub_key: torch.from_numpy(sub_data).to(target_device)
                    for sub_key, sub_data in data.items()
                }
            elif isinstance(data, np.ndarray):
                # Handles standard numpy arrays
                tensor_batch[key] = torch.from_numpy(data).to(target_device)
            else:
                # Should ideally not happen if get() returns numpy arrays/dicts
                warnings.warn(
                    f"Unexpected data type '{type(data)}' for key '{key}' in batch. Skipping tensor conversion."
                )
                tensor_batch[key] = data  # Keep as is or raise error?

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
        # Ensure sequences do not wrap around the buffer unless wrap_at_capacity=True (not handled here yet)
        # For now, assume sequences must be contiguous within the current fill level.
        max_start_time = buffer_size - seq_len
        if max_start_time < 0:
            return []  # Cannot form sequences of length seq_len

        valid_pairs: List[Tuple[int, int]] = [
            (t, env) for env in range(self.num_envs) for t in range(max_start_time + 1)
        ]

        if not valid_pairs:
            return []

        if batch_size is None or batch_size >= len(valid_pairs):
            return valid_pairs  # Return all pairs if batch_size too large / None

        return random.sample(valid_pairs, batch_size)

    def get_sequences(
        self,
        seq_len: int,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Returns a dictionary with batched sequences suitable for truncated BPTT.
        Handles Dict observations correctly.

        The returned arrays have an additional leading batch dimension of size
        ``batch_size`` and a time dimension of size ``seq_len``.

        :param seq_len: Length of each sequence in timesteps.
        :param batch_size: Number of sequences to sample. If None, returns all
                           possible sequences.
        :return: Dictionary mirroring :pyfunc:`get`, but with sequences. For Dict
                 observations, 'observations' and 'next_observations' are dictionaries
                 mapping keys to sequence arrays. 'initial_hidden_states' is also
                 a dictionary if recurrent.
        """
        # Sample starting indices (time_idx, env_idx)
        start_indices = self._sample_sequence_start_indices(seq_len, batch_size)

        actual_batch_size = len(start_indices)

        if actual_batch_size == 0:
            warnings.warn(
                "Could not sample any valid sequences, returning empty dictionary."
            )
            return {}

        # Helper to stack sequences along a new batch dimension
        # Input array shape: (buffer_capacity, num_envs, *dims)
        # Output array shape: (actual_batch_size, seq_len, *dims)
        def _stack_seq(array: np.ndarray) -> np.ndarray:
            # Extract sequence slices for each start index
            # slice shape: (seq_len, *dims)
            seqs = [array[t : t + seq_len, env_idx] for t, env_idx in start_indices]
            # Stack along a new batch dimension (axis 0)
            return np.stack(seqs, axis=0)

        seq_data: Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]] = {}

        # Handle observations
        if self.is_dict_obs:
            seq_data["observations"] = {}
            seq_data["next_observations"] = {}
            for key in self.observations:
                seq_data["observations"][key] = _stack_seq(self.observations[key])
                seq_data["next_observations"][key] = _stack_seq(
                    self.next_observations[key]
                )
        else:
            seq_data["observations"] = _stack_seq(self.observations)
            seq_data["next_observations"] = _stack_seq(self.next_observations)

        # Handle standard data (actions, rewards, etc.)
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

        # Handle recurrent states: We need the hidden state *at the beginning* of each sequence
        if self.recurrent and self.hidden_states is not None:
            seq_data["initial_hidden_states"] = {}
            for key in self.recurrent_state_keys:
                # Buffer shape: (time, layer, env, hidden)
                # We need the state at time 't' for environment 'env_idx'
                # Output shape: (batch, layer, hidden)
                init_h = np.stack(
                    [
                        # Get state from specific time 't' and env 'env_idx'
                        # Squeeze env dim as we are stacking batch dim
                        self.hidden_states[key][t, :, env_idx, :]
                        for t, env_idx in start_indices
                    ],
                    axis=0,  # Stack along the new batch dimension
                )
                seq_data["initial_hidden_states"][key] = init_h

        return seq_data

    def get_sequence_tensor_batch(
        self,
        seq_len: int,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Same as :pyfunc:`get_sequences` but returns PyTorch tensors on the
        specified device. Handles Dict observations correctly.

        :param seq_len: Length of each sequence.
        :param batch_size: Number of sequences to sample. If None, returns all.
        :param device: Torch device to move tensors to. Defaults to
                       :pyattr:`self.device`.
        :return: Dictionary with tensor sequences. For Dict observations, 'observations'
                 and 'next_observations' are dictionaries mapping keys to tensors.
                 'initial_hidden_states' is also a dictionary if recurrent.
        """
        target_device = torch.device(device or self.device)
        np_batch = self.get_sequences(seq_len=seq_len, batch_size=batch_size)

        if not np_batch:  # Handle empty batch case from get_sequences
            return {}

        tensor_batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        for key, data in np_batch.items():
            if isinstance(data, dict):
                # Handles dicts for observations, next_observations, initial_hidden_states
                tensor_batch[key] = {
                    k: torch.from_numpy(v).to(target_device) for k, v in data.items()
                }
            elif isinstance(data, np.ndarray):
                # Handles standard numpy arrays
                tensor_batch[key] = torch.from_numpy(data).to(target_device)
            else:
                warnings.warn(
                    f"Unexpected data type '{type(data)}' for key '{key}' in sequence batch. Skipping tensor conversion."
                )
                tensor_batch[key] = data

        return tensor_batch

    # Removed add_trajectory, _add_trajectory_segment, get_trajectories,
    # clear_trajectories, to_legacy_format, get_flat_batch
    # as they are less applicable/more complex with the vectorized approach.
