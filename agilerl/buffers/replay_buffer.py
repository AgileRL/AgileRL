from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from tensordict import TensorDict

from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree


class Sampler(ABC):
    """Abstract base class for replay buffer samplers."""

    @abstractmethod
    def sample(self, buffer_size: int, batch_size: int) -> torch.Tensor:
        """Sample indices from the buffer.

        Args:
            buffer_size: Current size of the buffer
            batch_size: Number of indices to sample

        Returns:
            Tensor of sampled indices
        """
        pass


class RandomSampler(Sampler):
    """Random uniform sampling from the buffer."""

    def sample(self, buffer_size: int, batch_size: int) -> torch.Tensor:
        return torch.randint(0, buffer_size, (batch_size,))


class PrioritizedSampler(Sampler):
    """Prioritized sampling from the buffer using segment trees."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        """Initialize the prioritized sampler.

        Args:
            capacity: Maximum capacity of the buffer
            alpha: Priority exponent (0 for uniform sampling, 1 for pure prioritized)
            beta: Initial importance sampling weight
            beta_increment: How much to increment beta by each update
        """
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        # Capacity must be positive and a power of 2
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def sample(self, buffer_size: int, batch_size: int) -> torch.Tensor:
        """Sample indices using priority-based sampling.

        Args:
            buffer_size: Current size of the buffer
            batch_size: Number of indices to sample

        Returns:
            Tensor of sampled indices
        """
        indices = self._sample_proportional(buffer_size, batch_size)
        self._last_indices = indices
        self._last_weights = torch.tensor([self._calculate_weight(i) for i in indices])
        return torch.tensor(indices)

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)
        self.beta = min(1.0, self.beta + self.beta_increment)

    def _sample_proportional(self, buffer_size: int, batch_size: int) -> List[int]:
        """Sample indices based on proportions.

        Args:
            buffer_size: Current size of the buffer
            batch_size: Number of indices to sample

        Returns:
            List of sampled indices
        """
        indices = []
        p_total = self.sum_tree.sum(0, buffer_size - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = torch.rand(1).item() * (b - a) + a
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int) -> float:
        """Calculate the weight of the experience at idx.

        Args:
            idx: Index of the experience

        Returns:
            Weight of the experience
        """
        # Get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.sum_tree.capacity) ** (-self.beta)

        # Calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.sum_tree.capacity) ** (-self.beta)
        weight = weight / max_weight

        return weight


class ReplayBuffer:
    """Base class for replay buffers using TensorDict storage."""

    def __init__(
        self,
        max_size: int,
        sampler: Optional[Sampler] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize the replay buffer.

        Args:
            max_size: Maximum number of transitions to store
            sampler: Sampler to use for sampling transitions (defaults to RandomSampler)
            device: Device to store the buffer on
        """
        self.max_size = max_size
        self.device = device if device is not None else "cpu"
        self.sampler = sampler if sampler is not None else RandomSampler()
        self._size = 0
        self._storage: Optional[TensorDict] = None

    @property
    def size(self) -> int:
        """Current number of transitions in the buffer."""
        return self._size

    def add(self, data: Union[TensorDict, Dict[str, Any]]) -> None:
        """Add a transition to the buffer.

        Args:
            data: Transition data as TensorDict or dict
        """
        if not isinstance(data, TensorDict):
            data = TensorDict(data, batch_size=[])

        if self._storage is None:
            # Initialize storage with first transition
            self._storage = TensorDict(
                {
                    k: torch.zeros((self.max_size,) + v.shape, device=self.device)
                    for k, v in data.items()
                },
                batch_size=[self.max_size],
            )

        # Add to storage
        idx = self._size % self.max_size
        for k, v in data.items():
            self._storage[k][idx] = v.to(self.device)

        self._size = min(self._size + 1, self.max_size)

    def sample(self, batch_size: int) -> TensorDict:
        """Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            TensorDict containing the sampled transitions
        """
        if self._size == 0:
            raise RuntimeError("Buffer is empty")

        indices = self.sampler.sample(self._size, batch_size)
        batch = self._storage[indices]

        # Add importance weights if using PrioritizedSampler
        if isinstance(self.sampler, PrioritizedSampler):
            batch["weights"] = self.sampler._last_weights.to(self.device)
            batch["indices"] = indices

        return batch

    def update_priorities(
        self, indices: torch.Tensor, priorities: torch.Tensor
    ) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        if isinstance(self.sampler, PrioritizedSampler):
            self.sampler.update_priorities(indices.tolist(), priorities.tolist())

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self._size = 0
        self._storage = None
