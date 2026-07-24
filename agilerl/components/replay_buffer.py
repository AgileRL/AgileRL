from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from tensordict import TensorDict

from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree

# Typed to return ``torch.Tensor`` so callers don't narrow the widened
# ``TensorDict.__getitem__`` return; the n-step keys always hold tensors.
if TYPE_CHECKING:

    def _read_tensor(transition: TensorDict, key: str) -> torch.Tensor: ...

else:

    def _read_tensor(transition: TensorDict, key: str) -> torch.Tensor:
        return transition[key]


class ReplayBuffer:
    """A circular replay buffer for off-policy learning using a TensorDict as storage.

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param device: Device to store the transitions on
    :type device: str | torch.device | None, optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_size = max_size
        self.device = device
        self.dtype = dtype
        self.counter = 0
        self.initialized = False

        self._cursor = 0
        self._size = 0
        self._storage: TensorDict | None = None

    @property
    def storage(self) -> TensorDict | None:
        """Storage of the buffer, or ``None`` if no data has been added yet."""
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

    @staticmethod
    def _normalize_dims(data: TensorDict, n: int) -> TensorDict:
        """Give every scalar ``(batch,)`` leaf a trailing feature dim ``(batch, 1)``.

        Recurses in place into nested tensor collections, so a single
        implementation handles flat, dict-observation, and nested multi-agent
        (``field -> agent_id -> tensor``) layouts uniformly at any depth -
        higher-dimensional leaves (e.g. images) are left untouched.

        :param data: Data to normalize
        :type data: TensorDict
        :param n: Number of transitions
        :type n: int
        :return: Normalized data
        :rtype: TensorDict
        """
        for key, item in data.items():
            if isinstance(item, TensorDict):
                # Nested collections in buffer storage are TensorDicts
                ReplayBuffer._normalize_dims(item, n)
            elif item.ndim == 1:
                data[key] = item.reshape(n, 1)

        return data

    def _init(self, data: TensorDict) -> TensorDict:
        """Initialize the buffer given the passed data. For each key,
        we inspect the shape of the value and initialize the storage
        tensor with the correct shape.

        :param data: Data to initialize the buffer with
        :type data: TensorDict
        :return: The initialized storage
        :rtype: TensorDict
        """
        _data = data[0]
        assert isinstance(_data, TensorDict)
        self._storage = _data.expand((self.max_size, *_data.shape)).clone()
        self._storage.zero_()
        self.initialized = True
        return self._storage

    def add(self, data: TensorDict) -> TensorDict | None:
        """Add a transition to the buffer.

        :param data: Transition to add to the buffer
        :type data: TensorDict | dict[str, Any]
        :return: The first transition leaving the n-step window for n-step buffers, None otherwise
        :rtype: TensorDict | None
        """
        # Initialize storage
        data = data.to(self.device)
        _n_transitions = data.shape[0]

        # Ensure all tensors in data have proper dimensions beyond batch dimension
        # Handles the case of scalar observations that become (batch_size,)
        # instead of (batch_size, 1)
        data = self._normalize_dims(data, _n_transitions)

        storage = self._storage if self._storage is not None else self._init(data)

        # Add to circular storage
        start = self._cursor
        end = self._cursor + _n_transitions
        if end > self.max_size:
            n = self.max_size - start
            storage[start:] = data[:n]
            storage[: _n_transitions - n] = data[n:]
        else:
            storage[start:end] = data

        # Update cursor and size
        self._cursor = end % self.max_size
        self._size = min(self._size + _n_transitions, self.max_size)
        self.counter += _n_transitions

    def _sample_indices(self, k: int) -> torch.Tensor:
        """Draw ``k`` storage indices in ``[0, size)``.

        Small buffers sample **without replacement** via ``torch.randperm`` (no
        duplicate transitions within a minibatch). Once the buffer is large enough
        we switch to **with replacement** - a single O(``k``) ``torch.randint``
        draw, which avoids ``randperm``'s O(``size``) shuffle of the whole buffer.

        :param k: Number of indices to draw.
        :type k: int
        :return: 1-D tensor of ``k`` indices.
        :rtype: torch.Tensor
        """
        if k <= 0:
            return torch.empty(0, dtype=torch.long)

        # Switch to with-replacement sampling once the current fill reaches 16384.
        # Below this the O(size) randperm shuffle is cheap (< ~30us) and keeps
        # samples unique; above it the shuffle grows (~2ms at 1e6) while the
        # expected duplicate fraction of with-replacement (~ k / (2 * size)) stays
        # under 1% for any typical batch (0.78% at batch 256), and with-replacement
        # is the standard for experience replay. Uses the live fill (not max_size)
        # since a partially-full buffer collides far more than its eventual size.
        if self.size >= 16384:
            return torch.randint(0, self.size, (k,))

        # Otherwise sample without replacement (no intra-batch duplicates).
        return torch.randperm(self.size)[:k]

    def sample(
        self,
        batch_size: int,
        return_idx: bool = False,
        *,
        beta: float = 0.4,
    ) -> TensorDict:
        """Sample a batch of transitions.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Boolean flag to return index of samples randomly selected, defaults to False
        :type return_idx: bool, optional
        :param beta: Unused; accepted for API compatibility with prioritized replay buffers
        :type beta: float, optional
        :return: TensorDict containing sampled experiences
        :rtype: TensorDict
        """
        assert self._storage is not None, "Cannot sample from an empty buffer."

        indices = self._sample_indices(min(batch_size, self.size))
        samples = self._storage[indices]
        assert isinstance(samples, TensorDict)

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
    :type device: str | torch.device | None, optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        n_step: int = 3,
        gamma: float = 0.99,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(max_size, device, dtype)

        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer: deque[TensorDict] = deque(maxlen=n_step)
        self.reward_key = "reward"
        self.done_key: str | None = None
        self.ns_key = "next_obs"

    def add(self, data: TensorDict) -> TensorDict | None:
        """Add a transition to the n-step buffer and potentially to the replay buffer.

        :param data: Transition to add to the buffer
        :type data: TensorDict
        :return: First transition in the n-step buffer
        :rtype: TensorDict | None
        """
        # Add to n-step buffer
        data = data.to(self.device)
        self.n_step_buffer.append(data)

        # If buffer is not full yet, don't process n-step return
        if len(self.n_step_buffer) < self.n_step:
            return None

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
        assert self._storage is not None, "Cannot sample from an empty buffer."
        samples = self._storage[idxs]
        assert isinstance(samples, TensorDict)
        return samples

    def _get_n_step_info(self) -> TensorDict:
        """Calculate the n-step return information.

        :return: Transition with n-step return
        :rtype: TensorDict
        """
        # Copy the first transition as a base
        first_transition: TensorDict = self.n_step_buffer[0].clone()

        # Get the reward key based on what's available in the transition
        if not self.initialized:
            assert self.reward_key in self.n_step_buffer[0], (
                f"Reward key not found in transition. Expected key: {self.reward_key}"
            )
            assert self.ns_key in self.n_step_buffer[0], (
                f"Next observation key not found in transition. Expected key: {self.ns_key}"
            )

            done_key = None
            expected_keys = ["done", "termination", "terminated"]
            for key in expected_keys:
                if key in self.n_step_buffer[0]:
                    done_key = key
                    break

            assert done_key is not None, (
                f"No done/termination key found in transition. Expected keys: {expected_keys}"
            )
            self.done_key = done_key

        done_key = self.done_key
        assert done_key is not None, "Done key is resolved on the first transition."

        n_step_reward = _read_tensor(first_transition, self.reward_key).clone()

        # Get the last next_state and done flag
        for i, transition in enumerate(list(self.n_step_buffer)[1:]):
            # Add discounted reward
            reward = _read_tensor(transition, self.reward_key)
            n_step_reward += reward * (self.gamma ** (i + 1))

            # Update next_state and done flag
            done = _read_tensor(transition, done_key)
            next_obs = _read_tensor(transition, self.ns_key)
            first_transition[self.ns_key] = next_obs.clone()
            first_transition[done_key] = done.clone()

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
    :type device: str | torch.device | None, optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        device: str | torch.device = "cpu",
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

        # Assign max priority to the new entries in one vectorised tree update
        # (priority_alpha is constant across the batch).
        n_transitions = data.shape[0]
        priority_alpha = self.max_priority**self.alpha
        idxs = (self.tree_ptr + np.arange(n_transitions)) % self.max_size
        values = np.full(n_transitions, priority_alpha, dtype=np.float64)
        self.sum_tree.update_batch(idxs, values)
        self.min_tree.update_batch(idxs, values)
        self.tree_ptr = (self.tree_ptr + n_transitions) % self.max_size

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

    def sample(
        self,
        batch_size: int,
        return_idx: bool = False,
        *,
        beta: float = 0.4,
    ) -> TensorDict:
        """Sample a batch of transitions based on priorities.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Unused; indices are always included in the batch
        :type return_idx: bool, optional
        :param beta: Beta parameter for importance sampling, defaults to 0.4
        :type beta: float, optional
        :return: Batch of transitions
        :rtype: TensorDict
        """
        assert self._storage is not None, "Cannot sample from an empty buffer."

        # Sample indices based on priorities
        indices = self._sample_proportional(batch_size)

        sampled = self._storage[indices]
        assert isinstance(sampled, TensorDict)
        samples = sampled.clone()

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
        # Stratified sampling: one uniform per segment, then a single vectorised
        # descent of the sum-tree for the whole batch (no Python per-sample loop).
        # upperbound_i = segment * (i + u_i),  u_i ~ U[0, 1).
        total_priority = self.sum_tree.sum()
        segment = total_priority / batch_size
        u = torch.rand(batch_size).numpy()
        upperbounds = (np.arange(batch_size) + u) * segment
        indices = self.sum_tree.retrieve_batch(upperbounds)

        return torch.as_tensor(indices, dtype=torch.int64)

    def _calculate_weights(self, indices: torch.Tensor, beta: float) -> torch.Tensor:
        """Calculate importance sampling weights for prioritized replay.

        :param indices: Sampled indices
        :type indices: torch.Tensor
        :param beta: Beta parameter for importance sampling
        :type beta: float
        :return: Weights for the sampled transitions
        :rtype: torch.Tensor
        """
        # Total priority is loop-invariant - compute it once.
        total_priority = self.sum_tree.sum()

        # Min probability -> maximum (normalising) weight.
        p_min = self.min_tree.min() / total_priority
        max_weight = (p_min * self.size) ** -beta

        # Gather the sampled priorities and compute every weight in one
        # vectorised numpy op (no Python per-element loop).
        idx_np = torch.as_tensor(indices).flatten().cpu().numpy()
        p_samples = self.sum_tree.get_batch(idx_np) / total_priority
        weights = (p_samples * self.size) ** -beta / max_weight
        return torch.as_tensor(weights, dtype=torch.float32, device=self.device)

    def update_priorities(
        self,
        indices: torch.Tensor | npt.NDArray,
        priorities: torch.Tensor | npt.NDArray,
    ) -> None:
        """Update priorities of the sampled transitions.

        :param indices: Indices of transitions to update
        :type indices: torch.Tensor | npt.NDArray
        :param priorities: New priorities
        :type priorities: torch.Tensor | npt.NDArray
        """
        # float64 matches the original max(priority.item(), 1e-5) clamp precision.
        idx_np = torch.as_tensor(indices).flatten().cpu().numpy()
        if idx_np.size == 0:
            return

        priorities = (
            torch.as_tensor(priorities, dtype=torch.float64)
            .clamp_min(1e-5)
            .flatten()
            .cpu()
            .numpy()
        )
        priority_alpha = priorities**self.alpha
        self.sum_tree.update_batch(idx_np, priority_alpha)
        self.min_tree.update_batch(idx_np, priority_alpha)
        self.max_priority = max(self.max_priority, float(priorities.max()))


# Any off-policy replay buffer, canonical here so consumers share one definition.
BufferType = ReplayBuffer | PrioritizedReplayBuffer | MultiStepReplayBuffer
