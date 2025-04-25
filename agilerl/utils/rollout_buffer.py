from copy import deepcopy
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from gymnasium import spaces

from agilerl.typing import ArrayOrTensor, TorchObsType, ObservationType


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
        recurrent_state_keys: Optional[List[str]] = ["h", "c"],
        hidden_state_size: Optional[int] = None,
        use_gae: bool = True,
        wrap_at_capacity: bool = False,
    ):
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.recurrent = recurrent
        self.recurrent_state_keys = recurrent_state_keys if recurrent else []
        self.hidden_state_size = hidden_state_size
        self.use_gae = use_gae
        self.wrap_at_capacity = wrap_at_capacity

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
            expected_shape = (1, self.num_envs, self.hidden_state_size)
            if hidden_state is not None:
                for key in self.recurrent_state_keys:
                    state = hidden_state.get(key)
                    if state is None:
                        raise ValueError(f"Hidden state missing key: {key}")
                    # Convert state to numpy if it's a tensor
                    if isinstance(state, torch.Tensor):
                        state = state.cpu().numpy()
                    # Validate shape before storing
                    if state.shape != expected_shape:
                        raise ValueError(
                            f"Hidden state['{key}'] shape {state.shape} does not match expected {expected_shape}"
                        )
                    self.hidden_states[key][self.pos] = state

            if next_hidden_state is not None:
                for key in self.recurrent_state_keys:
                    next_state = next_hidden_state.get(key)
                    if next_state is None:
                        raise ValueError(f"Next hidden state missing key: {key}")
                    # Convert state to numpy if it's a tensor
                    if isinstance(next_state, torch.Tensor):
                        next_state = next_state.cpu().numpy()
                    # Validate shape before storing
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

            for key in self.recurrent_state_keys:
                if self.hidden_states is not None and key in self.hidden_states:
                    data = self.hidden_states[key][
                        :buffer_size
                    ]  # Shape: (buffer_size, 1, num_envs, hidden_size)
                    # Swap num_envs and layer_dim axes, then reshape
                    flattened_data["hidden_states"][key] = data.swapaxes(1, 2).reshape(
                        total_samples, 1, self.hidden_state_size
                    )

                if (
                    self.next_hidden_states is not None
                    and key in self.next_hidden_states
                ):
                    next_data = self.next_hidden_states[key][
                        :buffer_size
                    ]  # Shape: (buffer_size, 1, num_envs, hidden_size)
                    # Swap num_envs and layer_dim axes, then reshape
                    flattened_data["next_hidden_states"][key] = next_data.swapaxes(
                        1, 2
                    ).reshape(total_samples, 1, self.hidden_state_size)

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

    # Removed add_trajectory, _add_trajectory_segment, get_trajectories,
    # clear_trajectories, to_legacy_format, get_flat_batch
    # as they are less applicable/more complex with the vectorized approach.
