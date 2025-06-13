from collections import deque
from typing import Deque, Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase, is_tensor_collection

from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree
from agilerl.typing import ArrayOrTensor

DataType = Union[Dict[str, ArrayOrTensor], TensorDict]


class ReplayBuffer:
    """A circular replay buffer for off-policy learning using a TensorDict as storage.

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param device: Device to store the transitions on
    :type device: Optional[Union[str, torch.device]], optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_size = max_size
        self.device = device
        self.dtype = dtype
        self.counter = 0
        self.initialized = False

        self._cursor = 0
        self._size = 0
        self._storage: Optional[TensorDict] = None

    @property
    def storage(self) -> TensorDict:
        """Storage of the buffer."""
        return self._storage

    @property
    def size(self) -> int:
        """Number of transitions in the buffer."""
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value

    @property
    def is_full(self) -> bool:
        return len(self) == self.max_size

    def __len__(self) -> int:
        return self._size

    def _init(self, data: TensorDict) -> None:
        """Initialize the buffer given the passed data. For each key,
        we inspect the shape of the value and initialize the storage
        tensor with the correct shape.

        :param data: Data to initialize the buffer with
        :type data: TensorDict
        """
        _data: TensorDict = data[0]
        self._storage = torch.zeros_like(_data.expand((self.max_size, *_data.shape)))
        self.initialized = True

    def add(self, data: TensorDict) -> None:
        """Add a transition to the buffer.

        :param data: Transition to add to the buffer
        :type data: Union[TensorDict, Dict[str, Any]]
        """
        # Initialize storage
        data = data.to(self.device)
        _n_transitions = data.shape[0]

        # Ensure all tensors in data have proper dimensions beyond batch dimension
        # Handles the case of scalar observations that become (batch_size,)
        # instead of (batch_size, 1)
        for key, value in data.items():
            if is_tensor_collection(value):
                value: TensorDictBase = value
                for k, v in value.items():
                    if v.ndim == 1:
                        value[k] = v.reshape(_n_transitions, 1)
            else:
                if value.ndim == 1:
                    value = value.reshape(_n_transitions, 1)

            data[key] = value

        if self._storage is None:
            self._init(data)

        # Add to circular storage
        start = self._cursor
        end = self._cursor + _n_transitions
        if end > self.max_size:
            n = self.max_size - start
            self._storage[start:] = data[:n]
            self._storage[: _n_transitions - n] = data[n:]
        else:
            self._storage[start:end] = data

        # Update cursor and size
        self._cursor = end % self.max_size
        self._size = min(self._size + _n_transitions, self.max_size)
        self.counter += _n_transitions

    def sample(self, batch_size: int, return_idx: bool = False) -> TensorDict:
        """Sample a batch of transitions.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Boolean flag to return index of samples randomly selected, defaults to False
        :type return_idx: bool, optional
        :return: TensorDict containing sampled experiences
        :rtype: TensorDict
        """
        # Ensure samples are unique
        indices = torch.randperm(self.size)[:batch_size]
        samples: TensorDict = self._storage[indices]

        if return_idx:
            samples["idxs"] = indices

        return samples

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self._size = 0
        self._cursor = 0
        self._storage = None
        self.initialized = False


class MultiStepReplayBuffer(ReplayBuffer):
    """A circular replay buffer for n-step returns in off-policy learning.

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param n_step: Number of steps to accumulate reward over
    :type n_step: int
    :param gamma: Discount factor
    :type gamma: float
    :param device: Device to store the transitions on
    :type device: Optional[Union[str, torch.device]], optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        n_step: int = 3,
        gamma: float = 0.99,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(max_size, device, dtype)

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer: Deque[TensorDict] = deque(maxlen=n_step)
        self.reward_key = "reward"
        self.done_key = None
        self.ns_key = "next_obs"

    def add(self, data: TensorDict) -> Optional[TensorDict]:
        """Add a transition to the n-step buffer and potentially to the replay buffer.

        :param data: Transition to add to the buffer
        :type data: TensorDict
        :return: First transition in the n-step buffer
        :rtype: Optional[TensorDict]
        """
        # Add to n-step buffer
        data = data.to(self.device)
        self.n_step_buffer.append(data)

        # If buffer is not full yet, don't process n-step return
        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate n-step return
        n_step_data = self._get_n_step_info()

        # Add to replay buffer
        super().add(n_step_data)
        return self.n_step_buffer[0]

    def sample_from_indices(self, idxs: torch.Tensor) -> TensorDict:
        """Sample a batch of transitions from the buffer using the provided indices.

        :param idxs: Indices of the transitions to sample
        :type idxs: torch.Tensor
        :return: TensorDict containing sampled experiences
        :rtype: TensorDict
        """
        return self.storage[idxs]

    def _get_n_step_info(self) -> TensorDict:
        """Calculate the n-step return information.

        :return: Transition with n-step return
        :rtype: TensorDict
        """
        # Copy the first transition as a base
        first_transition: TensorDict = self.n_step_buffer[0].clone()

        # Get the reward key based on what's available in the transition
        if not self.initialized:
            assert (
                self.reward_key in self.n_step_buffer[0]
            ), f"Reward key not found in transition. Expected key: {self.reward_key}"
            assert (
                self.ns_key in self.n_step_buffer[0]
            ), f"Next observation key not found in transition. Expected key: {self.ns_key}"

            done_key = None
            expected_keys = ["done", "termination", "terminated"]
            for key in expected_keys:
                if key in self.n_step_buffer[0]:
                    done_key = key
                    break

            assert (
                done_key is not None
            ), f"No done/termination key found in transition. Expected keys: {expected_keys}"
            self.done_key = done_key

        # Start with reward from first transition
        n_step_reward: torch.Tensor = first_transition[self.reward_key]
        n_step_reward = n_step_reward.clone()

        # Get the last next_state and done flag
        for i, transition in enumerate(list(self.n_step_buffer)[1:]):
            # Add discounted reward
            reward: torch.Tensor = transition[self.reward_key]
            n_step_reward += reward * (self.gamma ** (i + 1))

            # Update next_state and done flag
            done: torch.Tensor = transition[self.done_key]
            next_obs: torch.Tensor = transition[self.ns_key]
            first_transition[self.ns_key] = next_obs.clone()
            first_transition[self.done_key] = done.clone()

            if done.bool().any():  # Stop if episode terminated
                break

        # Update the reward with n-step return
        first_transition[self.reward_key] = n_step_reward

        return first_transition


class PrioritizedReplayBuffer(ReplayBuffer):
    """A prioritized replay buffer for off-policy learning as introduced in the paper
    'Prioritized Experience Replay' (Schaul et al., 2015).

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param alpha: How much prioritization to use (0 - no prioritization, 1 - full prioritization)
    :type alpha: float
    :param device: Device to store the transitions on.
    :type device: Optional[Union[str, torch.device]], optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(max_size, device, dtype)
        self.alpha = alpha
        self.max_priority = 1.0
        self.tree_ptr = 0

        # Find the closest power of 2 capacity for the segment trees
        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2

        # Initialize segment trees
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(self, data: TensorDict) -> None:
        """Add a transition to the buffer.

        :param data: Transition to add to the buffer
        :type data: TensorDict
        """
        # Add to replay buffer
        super().add(data)

        # Add max priority for new entries
        n_transitions = data.shape[0]
        for i in range(n_transitions):
            self._update_priority(self.tree_ptr, self.max_priority)
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def _update_priority(self, idx: int, priority: float) -> None:
        """Update the priority of an experience in the buffer.

        :param idx: Index of the experience
        :type idx: int
        :param priority: New priority value
        :type priority: float
        """
        assert 0 <= idx < self.max_size

        # Apply alpha to priority
        priority_alpha = priority**self.alpha

        # Update trees
        self.sum_tree[idx] = priority_alpha
        self.min_tree[idx] = priority_alpha

        # Update max priority
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> TensorDict:
        """Sample a batch of transitions based on priorities.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param beta: Beta parameter for importance sampling, defaults to 0.4
        :type beta: float, optional
        :return: Batch of transitions
        :rtype: TensorDict
        """
        # Sample indices based on priorities
        indices = self._sample_proportional(batch_size)

        # Gather transitions
        samples: TensorDict = self.storage[indices]
        samples = samples.clone()

        # Calculate importance sampling weights
        weights = self._calculate_weights(indices, beta)

        # Add weights and indices to the batch
        samples["weights"] = weights.unsqueeze(1)
        samples["idxs"] = indices.unsqueeze(1)

        return samples

    def _sample_proportional(self, batch_size: int) -> torch.Tensor:
        """Sample indices based on their priorities.

        :param batch_size: Number of samples
        :type batch_size: int
        :return: Sampled indices
        :rtype: torch.Tensor
        """
        indices = torch.zeros(batch_size, dtype=torch.int64)

        # Get the sum of all priorities and section length
        total_priority = self.sum_tree.sum()
        segment = total_priority / batch_size

        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            # Sample uniformly from segment and retrieve corresponding index
            upperbound = torch.rand(1).item() * (b - a) + a
            idx = self.sum_tree.retrieve(upperbound)
            indices[i] = idx

        return indices

    def _calculate_weights(self, indices: torch.Tensor, beta: float) -> torch.Tensor:
        """Calculate importance sampling weights for prioritized replay.

        :param indices: Sampled indices
        :type indices: torch.Tensor
        :param beta: Beta parameter for importance sampling
        :type beta: float
        :return: Weights for the sampled transitions
        :rtype: torch.Tensor
        """
        # Create a tensor for weights
        batch_size = len(indices)
        weights = torch.zeros(batch_size, device=self.device)

        # Find the min probability from the min tree
        p_min = self.min_tree.min() / self.sum_tree.sum()

        # Calculate the max weight value
        max_weight = (p_min * self.size) ** -beta

        # Calculate weights for each index
        for i, idx in enumerate(indices):
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * self.size) ** -beta
            weights[i] = weight / max_weight  # Normalize

        return weights

    def update_priorities(
        self, indices: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        """Update priorities of the sampled transitions.

        :param indices: Indices of transitions to update
        :type indices: torch.Tensor
        :param priorities: New priorities
        :type priorities: torch.Tensor
        """
        for idx, priority in zip(indices, priorities):
            # Handle small priorities
            priority = max(priority.item(), 1e-5)

            # Update the priority
            self._update_priority(idx.item(), priority)
