from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
from gymnasium import spaces

from agilerl.typing import ArrayOrTensor, TorchObsType, ObservationType


class RolloutBuffer:
    """
    Rollout buffer for collecting experiences and computing advantages for RL algorithms.

    This buffer stores trajectories with potentially variable sequence lengths,
    supports hidden states, and handles vectorized environments.

    :param capacity: Maximum number of timesteps in the buffer
    :type capacity: int
    :param observation_space: Observation space of the environment
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment
    :type action_space: gym.spaces.Space
    :param device: Device to store tensors on, defaults to "cpu"
    :type device: str, optional
    :param gae_lambda: Lambda parameter for GAE, defaults to 0.95
    :type gae_lambda: float, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param recurrent: Whether to store hidden states, defaults to False
    :type recurrent: bool, optional
    :param hidden_size: Size of hidden states if used, defaults to None
    :type hidden_size: Tuple[int], optional
    :param use_gae: Whether to compute GAE advantages, defaults to True
    :type use_gae: bool, optional
    """

    def __init__(
        self,
        capacity: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        recurrent: bool = False,
        hidden_size: Optional[Tuple[int]] = None,
        use_gae: bool = True,
    ):
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.recurrent = recurrent
        self.hidden_size = hidden_size
        self.use_gae = use_gae

        self.pos = 0
        self.full = False
        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """Initialize buffer arrays with correct shapes."""
        if isinstance(self.observation_space, spaces.Discrete):
            obs_shape = (self.capacity, 1)
        elif isinstance(self.observation_space, spaces.Box):
            obs_shape = (self.capacity,) + self.observation_space.shape
        elif isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            # Special handling for Dict and Tuple spaces - will be done in add() method
            obs_shape = (self.capacity,)
        else:
            obs_shape = (self.capacity,) + self.observation_space.shape

        # Initialize buffers with zeros
        self.observations = np.zeros(obs_shape, dtype=np.float32)
        self.next_observations = np.zeros(obs_shape, dtype=np.float32)
        self.episode_starts = np.zeros((self.capacity,), dtype=np.bool_)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.values = np.zeros((self.capacity,), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity,), dtype=np.float32)
        self.advantages = np.zeros((self.capacity,), dtype=np.float32)
        self.returns = np.zeros((self.capacity,), dtype=np.float32)

        # Initialize action buffer based on action space type
        if isinstance(self.action_space, spaces.Discrete):
            self.actions = np.zeros((self.capacity,), dtype=np.int64)
        elif isinstance(self.action_space, spaces.Box):
            self.actions = np.zeros(
                (self.capacity,) + self.action_space.shape, dtype=np.float32
            )
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            self.actions = np.zeros(
                (self.capacity, len(self.action_space.nvec)), dtype=np.int64
            )
        elif isinstance(self.action_space, spaces.MultiBinary):
            self.actions = np.zeros(
                (self.capacity, self.action_space.n), dtype=np.int64
            )
        else:
            self.actions = np.zeros(
                (self.capacity,) + self.action_space.shape, dtype=np.float32
            )

        # Initialize hidden states if needed
        if self.recurrent and self.hidden_size is not None:
            self.hidden_states = np.zeros(
                (self.capacity,) + self.hidden_size, dtype=np.float32
            )
            self.next_hidden_states = np.zeros(
                (self.capacity,) + self.hidden_size, dtype=np.float32
            )
        else:
            self.hidden_states = None
            self.next_hidden_states = None

        # For managing variable-length trajectories
        self.trajectory_indices = []
        self.active_trajectories = []

    def add(
        self,
        obs: ObservationType,
        action: ArrayOrTensor,
        reward: Union[float, np.ndarray],
        done: Union[bool, np.ndarray],
        value: Union[float, np.ndarray],
        log_prob: Union[float, np.ndarray],
        next_obs: Optional[ObservationType] = None,
        hidden_state: Optional[ArrayOrTensor] = None,
        next_hidden_state: Optional[ArrayOrTensor] = None,
        episode_start: Optional[Union[bool, np.ndarray]] = None,
    ) -> None:
        """
        Add a new observation and associated data to the buffer.

        :param obs: Current observation
        :param action: Action taken
        :param reward: Reward received
        :param done: Whether the episode ended
        :param value: Value estimate
        :param log_prob: Log probability of the action
        :param next_obs: Next observation, defaults to None
        :param hidden_state: Current hidden state (for RNNs), defaults to None
        :param next_hidden_state: Next hidden state (for RNNs), defaults to None
        :param episode_start: Whether this is the start of an episode, defaults to None
        """
        # Convert to numpy arrays if needed
        if isinstance(reward, (int, float)):
            reward = np.array([reward])
        if isinstance(done, bool):
            done = np.array([done])
        if isinstance(value, (int, float)):
            value = np.array([value])
        if isinstance(log_prob, (int, float)):
            log_prob = np.array([log_prob])

        # Handle different observation space types
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        if next_obs is not None and isinstance(next_obs, torch.Tensor):
            next_obs = next_obs.cpu().numpy()

        # Handle action space types
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Store the data
        self.observations[self.pos] = obs

        # Handle different action shapes - ensure proper assignment
        if isinstance(self.action_space, spaces.Discrete):
            # For discrete action spaces, store a scalar
            if hasattr(action, "item"):
                self.actions[self.pos] = action.item()
            elif hasattr(action, "__len__") and len(action) == 1:
                self.actions[self.pos] = action[0]
            else:
                self.actions[self.pos] = action
        else:
            # For other action spaces, store the array
            self.actions[self.pos] = action

        # Store scalar values to avoid numpy warnings
        self.rewards[self.pos] = (
            reward.item()
            if hasattr(reward, "item")
            else reward[0] if hasattr(reward, "__len__") else reward
        )
        self.dones[self.pos] = (
            done.item()
            if hasattr(done, "item")
            else done[0] if hasattr(done, "__len__") else done
        )
        self.values[self.pos] = (
            value.item()
            if hasattr(value, "item")
            else value[0] if hasattr(value, "__len__") else value
        )
        self.log_probs[self.pos] = (
            log_prob.item()
            if hasattr(log_prob, "item")
            else log_prob[0] if hasattr(log_prob, "__len__") else log_prob
        )

        if next_obs is not None:
            self.next_observations[self.pos] = next_obs

        if episode_start is not None:
            if isinstance(episode_start, bool):
                episode_start = np.array([episode_start])
            self.episode_starts[self.pos] = (
                episode_start.item()
                if hasattr(episode_start, "item")
                else (
                    episode_start[0]
                    if hasattr(episode_start, "__len__")
                    else episode_start
                )
            )

        # Store hidden states if enabled
        if self.recurrent and hidden_state is not None:
            if isinstance(hidden_state, torch.Tensor):
                hidden_state = hidden_state.cpu().numpy()
            self.hidden_states[self.pos] = hidden_state

        if self.recurrent and next_hidden_state is not None:
            if isinstance(next_hidden_state, torch.Tensor):
                next_hidden_state = next_hidden_state.cpu().numpy()
            self.next_hidden_states[self.pos] = next_hidden_state

        # Update buffer position
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True
            self.pos = 0

    def add_trajectory(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        next_observations: Optional[np.ndarray] = None,
        hidden_states: Optional[np.ndarray] = None,
        next_hidden_states: Optional[np.ndarray] = None,
        episode_starts: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a full trajectory of data to the buffer.

        :param observations: Array of observations
        :param actions: Array of actions
        :param rewards: Array of rewards
        :param dones: Array of done flags
        :param values: Array of values
        :param log_probs: Array of log probabilities
        :param next_observations: Array of next observations, defaults to None
        :param hidden_states: Array of hidden states, defaults to None
        :param next_hidden_states: Array of next hidden states, defaults to None
        :param episode_starts: Array of episode start flags, defaults to None
        """
        traj_len = len(observations)
        start_pos = self.pos

        # Check if there's enough space in the buffer
        if start_pos + traj_len > self.capacity:
            # Not enough space - add what we can and wrap around
            first_part = self.capacity - start_pos
            self._add_trajectory_segment(
                observations[:first_part],
                actions[:first_part],
                rewards[:first_part],
                dones[:first_part],
                values[:first_part],
                log_probs[:first_part],
                (
                    next_observations[:first_part]
                    if next_observations is not None
                    else None
                ),
                hidden_states[:first_part] if hidden_states is not None else None,
                (
                    next_hidden_states[:first_part]
                    if next_hidden_states is not None
                    else None
                ),
                episode_starts[:first_part] if episode_starts is not None else None,
                start_pos,
            )

            # Add the second part at the beginning of the buffer
            self._add_trajectory_segment(
                observations[first_part:],
                actions[first_part:],
                rewards[first_part:],
                dones[first_part:],
                values[first_part:],
                log_probs[first_part:],
                (
                    next_observations[first_part:]
                    if next_observations is not None
                    else None
                ),
                hidden_states[first_part:] if hidden_states is not None else None,
                (
                    next_hidden_states[first_part:]
                    if next_hidden_states is not None
                    else None
                ),
                episode_starts[first_part:] if episode_starts is not None else None,
                0,
            )

            self.pos = traj_len - first_part
            self.full = True
            # Store trajectory indices with wrapped indices
            indices = list(range(start_pos, self.capacity)) + list(range(0, self.pos))
            self.trajectory_indices.append(indices)
        else:
            # Enough space to add the full trajectory
            self._add_trajectory_segment(
                observations,
                actions,
                rewards,
                dones,
                values,
                log_probs,
                next_observations,
                hidden_states,
                next_hidden_states,
                episode_starts,
                start_pos,
            )

            self.pos += traj_len
            if self.pos == self.capacity:
                self.full = True
                self.pos = 0

            # Store trajectory indices
            self.trajectory_indices.append(list(range(start_pos, start_pos + traj_len)))

    def _add_trajectory_segment(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        next_observations: Optional[np.ndarray],
        hidden_states: Optional[np.ndarray],
        next_hidden_states: Optional[np.ndarray],
        episode_starts: Optional[np.ndarray],
        start_pos: int,
    ) -> None:
        """Helper method to add a segment of a trajectory to the buffer."""
        segment_len = len(observations)
        end_pos = start_pos + segment_len

        self.observations[start_pos:end_pos] = observations
        self.actions[start_pos:end_pos] = actions
        self.rewards[start_pos:end_pos] = rewards
        self.dones[start_pos:end_pos] = dones
        self.values[start_pos:end_pos] = values
        self.log_probs[start_pos:end_pos] = log_probs

        if next_observations is not None:
            self.next_observations[start_pos:end_pos] = next_observations

        if episode_starts is not None:
            self.episode_starts[start_pos:end_pos] = episode_starts

        if self.recurrent and hidden_states is not None:
            self.hidden_states[start_pos:end_pos] = hidden_states

        if self.recurrent and next_hidden_states is not None:
            self.next_hidden_states[start_pos:end_pos] = next_hidden_states

    def compute_returns_and_advantages(
        self, last_value: Optional[float] = None, last_done: Optional[bool] = None
    ) -> None:
        """
        Compute returns and advantages for the stored trajectories.

        :param last_value: Value of the last observation, for bootstrapping, defaults to None
        :param last_done: Done flag of the last observation, defaults to None
        """
        # If using entire buffer
        if not self.trajectory_indices:
            self._compute_returns_and_advantages_for_buffer(last_value, last_done)
            return

        # Compute for each trajectory separately
        for indices in self.trajectory_indices:
            rewards = self.rewards[indices]
            dones = self.dones[indices]
            values = self.values[indices]

            # If trajectory is complete, last_value is the last value in the trajectory
            # Otherwise, use the provided last_value
            traj_last_value = (
                values[-1] if dones[-1] or last_value is None else last_value
            )
            traj_last_done = dones[-1] if last_done is None else last_done

            if self.use_gae:
                advantages = np.zeros_like(rewards)
                last_gae_lambda = 0

                # Compute GAE
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_non_terminal = 1.0 - float(traj_last_done)
                        next_value = traj_last_value
                    else:
                        next_non_terminal = 1.0 - float(dones[t + 1])
                        next_value = values[t + 1]

                    delta = (
                        rewards[t]
                        + self.gamma * next_value * next_non_terminal
                        - values[t]
                    )
                    advantages[t] = last_gae_lambda = (
                        delta
                        + self.gamma
                        * self.gae_lambda
                        * next_non_terminal
                        * last_gae_lambda
                    )

                # Store advantages and returns
                for i, idx in enumerate(indices):
                    self.advantages[idx] = advantages[i]
                    self.returns[idx] = advantages[i] + values[i]
            else:
                # Monte Carlo returns
                returns = np.zeros_like(rewards)
                last_return = traj_last_value * (1.0 - float(traj_last_done))

                for t in reversed(range(len(rewards))):
                    returns[t] = last_return = rewards[t] + self.gamma * last_return * (
                        1.0 - float(dones[t])
                    )

                # Store returns and simple advantages
                for i, idx in enumerate(indices):
                    self.returns[idx] = returns[i]
                    self.advantages[idx] = returns[i] - values[i]

    def _compute_returns_and_advantages_for_buffer(
        self, last_value: Optional[float] = None, last_done: Optional[bool] = None
    ) -> None:
        """Compute returns and advantages for the entire buffer."""
        if self.use_gae:
            # GAE Advantages
            last_gae_lambda = 0

            for t in reversed(range(self.size())):
                if t == self.size() - 1:
                    next_non_terminal = 1.0 - float(
                        last_done if last_done is not None else self.dones[t]
                    )
                    next_value = (
                        last_value if last_value is not None else self.values[t]
                    )
                else:
                    next_non_terminal = 1.0 - float(self.dones[t + 1])
                    next_value = self.values[t + 1]

                delta = (
                    self.rewards[t]
                    + self.gamma * next_value * next_non_terminal
                    - self.values[t]
                )
                self.advantages[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )

            self.returns = self.advantages + self.values
        else:
            # Monte Carlo returns
            last_return = last_value if last_value is not None else 0.0
            if last_done is not None:
                last_return *= 1.0 - float(last_done)

            for t in reversed(range(self.size())):
                self.returns[t] = last_return = self.rewards[
                    t
                ] + self.gamma * last_return * (1.0 - float(self.dones[t]))

            self.advantages = self.returns - self.values

    def get(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get all data from the buffer, optionally as a random batch.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None
        :return: Dictionary with all stored buffer data
        """
        if batch_size is None:
            # Return all valid data
            if self.full:
                indices = np.arange(self.capacity)
            else:
                indices = np.arange(self.pos)
        else:
            # Sample random batch
            buffer_size = self.capacity if self.full else self.pos
            indices = np.random.choice(buffer_size, size=batch_size, replace=False)

        batch = {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "dones": self.dones[indices],
            "values": self.values[indices],
            "log_probs": self.log_probs[indices],
            "advantages": self.advantages[indices],
            "returns": self.returns[indices],
            "episode_starts": self.episode_starts[indices],
        }

        if self.next_observations[0] is not None:
            batch["next_observations"] = self.next_observations[indices]

        if self.recurrent and self.hidden_states is not None:
            batch["hidden_states"] = self.hidden_states[indices]

        if self.recurrent and self.next_hidden_states is not None:
            batch["next_hidden_states"] = self.next_hidden_states[indices]

        return batch

    def get_trajectories(self) -> List[Dict[str, ArrayOrTensor]]:
        """
        Get all stored trajectories as a list of dictionaries.

        :return: List of dictionaries with trajectory data
        """
        trajectories = []

        for indices in self.trajectory_indices:
            traj = {
                "observations": self.observations[indices],
                "actions": self.actions[indices],
                "rewards": self.rewards[indices],
                "dones": self.dones[indices],
                "values": self.values[indices],
                "log_probs": self.log_probs[indices],
                "advantages": self.advantages[indices],
                "returns": self.returns[indices],
                "episode_starts": self.episode_starts[indices],
            }

            if self.next_observations[0] is not None:
                traj["next_observations"] = self.next_observations[indices]

            if self.recurrent and self.hidden_states is not None:
                traj["hidden_states"] = self.hidden_states[indices]

            if self.recurrent and self.next_hidden_states is not None:
                traj["next_hidden_states"] = self.next_hidden_states[indices]

            trajectories.append(traj)

        return trajectories

    def get_flat_batch(self, batch_size: Optional[int] = None) -> Tuple:
        """
        Get data from the buffer in the legacy flat tuple format.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None
        :return: Tuple with buffer data in legacy format
        """
        data = self.get(batch_size)

        # Return in legacy format for backward compatibility
        return (
            data["observations"],
            data["actions"],
            data["log_probs"],
            data["advantages"],
            data["returns"],
            data["values"],
        )

    def get_tensor_batch(
        self, batch_size: Optional[int] = None, device: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get data from the buffer as PyTorch tensors.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None
        :param device: Device to put tensors on, defaults to None (uses self.device)
        :return: Dictionary with tensor data
        """
        batch = self.get(batch_size)
        device = device or self.device

        tensor_batch = {}
        for key, data in batch.items():
            if isinstance(data, np.ndarray):
                tensor_batch[key] = torch.from_numpy(data).to(device)
            else:
                tensor_batch[key] = data

        return tensor_batch

    def size(self) -> int:
        """
        Get current size of buffer.

        :return: Current size of buffer
        """
        return self.capacity if self.full else self.pos

    def reset(self) -> None:
        """Reset the buffer to empty state."""
        self.pos = 0
        self.full = False
        self.trajectory_indices = []
        self.active_trajectories = []

    def clear_trajectories(self) -> None:
        """Clear trajectory metadata but keep buffer data."""
        self.trajectory_indices = []
        self.active_trajectories = []

    def to_legacy_format(self) -> Tuple:
        """
        Convert buffer data to legacy tuple format used by old AgileRL algorithms.

        :return: Tuple of experiences in legacy format
        """
        if self.full:
            size = self.capacity
        else:
            size = self.pos

        if size == 0:
            return tuple([np.array([]) for _ in range(8)])

        # Legacy format: (states, actions, log_probs, rewards, dones, values, next_state, next_done)
        next_state = self.next_observations[size - 1 : size]
        next_done = np.array([self.dones[size - 1]])

        return (
            self.observations[:size],
            self.actions[:size],
            self.log_probs[:size],
            self.rewards[:size],
            self.dones[:size],
            self.values[:size],
            next_state,
            next_done,
        )
