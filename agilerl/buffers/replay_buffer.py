from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict

from agilerl.typing import ArrayOrTensor

DataType = Union[Dict[str, ArrayOrTensor], TensorDict]


class ReplayBuffer:
    """A circular replay buffer for off-policy reinforcement learning
    using a TensorDict as storage.

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param ndim: the number of dimensions to be accounted for when measuring the storage size.
        For instance, a storage of shape ``[3, 4]`` has capacity ``3`` if ``ndim=1`` and ``12`` if ``ndim=2``.
        Defaults to ``1``.
    :type ndim: int, optional
    :param device: Device to store the transitions on.
    :type device: Optional[Union[str, torch.device]], optional
    """

    def __init__(self, max_size: int, device: Union[str, torch.device] = "cpu") -> None:

        self.max_size = max_size
        self.device = device

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
        _data: TensorDict = data.copy()
        _data = _data.to(self.device)
        _data = _data[0] if is_vectorised else _data
        self._storage = torch.empty_like(_data.expand((self.max_size, *_data.shape)))

    def add(
        self, data: Union[TensorDict, Dict[str, Any]], is_vectorised: bool = False
    ) -> None:
        """Add a transition to the buffer.

        :param data: Transition to add to the buffer
        :type data: Union[TensorDict, Dict[str, Any]]
        :param is_vectorised: Whether the data is vectorised or not
        :type is_vectorised: bool
        """
        if not isinstance(data, TensorDict):
            data = TensorDict(data, batch_size=[])

        # Initialize storage
        if self._storage is None:
            self._init(data, is_vectorised)

        # Add to storage
        data = data.to(self.device)
        if not is_vectorised:
            data = data.expand((1, *data.shape))

        _n_transitions = data.shape[0]
        start = self._cursor
        end = self._cursor + _n_transitions
        self._storage[start:end] = data
        self._cursor = end % self.max_size
        self._size = min(self._size + _n_transitions, self.max_size)

    def sample(self, batch_size: int) -> TensorDict:
        """Sample a batch of transitions.

        :param batch_size: Number of transitions to sample
        :type batch_size: int
        :return: Sampled transitions
        :rtype: TensorDict
        """
        indices = torch.randint(0, self.size, (batch_size,))
        return self._storage[indices]

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self._size = 0
        self._cursor = 0
        self._storage = None
