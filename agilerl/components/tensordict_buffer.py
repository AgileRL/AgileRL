from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict


class ReplayBuffer:
    """A replay buffer for storing experiences in off-policy reinforcement learning. It uses
    TensorDict for efficient storage and sampling.

    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience, e.g. ['state', 'action', 'reward']
    :type field_names: list[Union[str, Tuple[str, ...], Dict[str, Any]]]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(
        self,
        memory_size: int,
        field_names: List[Union[str, Tuple[str, ...], Dict[str, Any]]],
        device: Optional[str] = None,
    ):
        assert memory_size > 0, "Memory size must be greater than zero."
        assert len(field_names) > 0, "Field names must contain at least one field name."

        self.memory_size = memory_size
        self.field_names = field_names
        self.device = device
        self.memory = self._initialize_memory(field_names, memory_size)
        self.counter = 0

    def _initialize_memory(
        self,
        field_names: List[Union[str, Tuple[str, ...], Dict[str, Any]]],
        memory_size: int,
    ) -> TensorDict:
        """Initialize the TensorDict memory structure based on field names.

        :param field_names: List of field names that can be strings, tuples of strings, or dicts
        :type field_names: List[Union[str, Tuple[str, ...], Dict[str, Any]]]
        :param memory_size: Maximum size of the replay buffer
        :type memory_size: int
        :return: TensorDict containing empty tensors for each field
        :rtype: TensorDict
        """
        memory_structure = {}
        for field in field_names:
            if isinstance(field, str):
                memory_structure[field] = torch.empty((memory_size,))
            elif isinstance(field, tuple):
                memory_structure[field] = tuple(
                    torch.empty((memory_size,)) for _ in field
                )
            elif isinstance(field, dict):
                memory_structure[field] = {
                    k: torch.empty((memory_size,)) for k in field.keys()
                }

        return TensorDict(memory_structure, batch_size=[memory_size])

    def __len__(self) -> int:
        """Returns the current size of internal memory."""
        return min(self.counter, self.memory_size)

    def _add(self, *args: Any) -> None:
        """Adds experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        idx = self.counter % self.memory_size
        for i, field in enumerate(self.field_names):
            if isinstance(field, str):
                self.memory[field][idx] = args[i]
            elif isinstance(field, tuple):
                for j, subfield in enumerate(field):
                    self.memory[field][j][idx] = args[i][j]
            elif isinstance(field, dict):
                for key in field.keys():
                    self.memory[field][key][idx] = args[i][key]

        self.counter += 1

    def sample(self, batch_size: int, return_idx: bool = False) -> Tuple[Any, ...]:
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Boolean flag to return index of samples randomly selected, defaults to False
        :type return_idx: bool, optional
        :return: Tuple of sampled experiences
        :rtype: tuple
        """
        idxs = torch.randint(0, len(self), (batch_size,))
        sampled_transitions: TensorDict = self.memory[idxs]

        if return_idx:
            return tuple(sampled_transitions.values()), idxs
        else:
            return tuple(sampled_transitions.values())

    def save_to_memory(self, *args: Any) -> None:
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        self._add(*args)
