from typing import Dict, Optional, Union

import torch
from tensordict import TensorDict

from agilerl.typing import ArrayOrTensor

DataType = Union[Dict[str, ArrayOrTensor], TensorDict]


class ReplayBuffer:
    """A circular replay buffer for off-policy learning using a TensorDict as storage.

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param device: Device to store the transitions on.
    :type device: Optional[Union[str, torch.device]], optional
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

    @property
    def is_full(self) -> bool:
        return len(self) == self.max_size

    def __len__(self) -> int:
        return self._size

    def _init(self, data: TensorDict, is_vectorised: bool) -> None:
        """Initialize the buffer given the passed data. For each key,
        we inspect the shape of the value and initialize the storage
        tensor with the correct shape.

        :param data: Data to initialize the buffer with
        :type data: TensorDict
        :param is_vectorised: Whether the data is vectorised or not
        :type is_vectorised: bool
        """
        _data = data[0] if is_vectorised else data
        self._storage = torch.empty_like(_data.expand((self.max_size, *_data.shape)))

    def add(self, data: TensorDict, is_vectorised: bool = False) -> None:
        """Add a transition to the buffer.

        :param data: Transition to add to the buffer
        :type data: Union[TensorDict, Dict[str, Any]]
        :param is_vectorised: Whether the data is vectorised or not
        :type is_vectorised: bool
        """
        # Initialize storage
        data = data.to(self.device)
        if self._storage is None:
            self._init(data, is_vectorised)

        if not is_vectorised:
            data = data.expand((1, *data.shape))

        # Add to storage considering circularity of buffer
        _n_transitions = data.shape[0]
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

    def sample(self, batch_size: int, return_idx: bool = False) -> TensorDict:
        """Sample a batch of transitions.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Boolean flag to return index of samples randomly selected, defaults to False
        :type return_idx: bool, optional
        :return: Tuple of sampled experiences
        :rtype: tuple
        """
        indices = torch.randint(0, self.size, (batch_size,))
        samples: TensorDict = self._storage[indices]

        if return_idx:
            samples["idxs"] = indices

        return samples

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self._size = 0
        self._cursor = 0
        self._storage = None


class NStepReplayBuffer(ReplayBuffer): ...


class PrioritizedReplayBuffer(ReplayBuffer): ...
