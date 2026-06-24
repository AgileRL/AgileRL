from collections import deque, namedtuple
from numbers import Number
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from agilerl.typing import NumpyObsType

NpTransitionType = Number | np.ndarray | dict[str, np.ndarray]
TorchTransitionType = torch.Tensor | dict[str, torch.Tensor]


class MultiAgentReplayBuffer:
    """The Multi-Agent Experience Replay Buffer class. Used to store multiple agents'
    experiences and allow off-policy learning.

    :param memory_size: Maximum length of the replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: list[str]
    :param agent_ids: Names of all agents that will act in the environment
    :type agent_ids: list[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str | None
    """

    def __init__(
        self,
        memory_size: int,
        field_names: list[str],
        agent_ids: list[str],
        device: str | None = None,
    ) -> None:
        assert memory_size > 0, "Memory size must be greater than zero."
        assert len(field_names) > 0, "Field names must contain at least one field name."
        assert len(agent_ids) > 0, "Agent ids must contain at least one agent id."

        self.memory_size: int = memory_size
        self.memory: deque = deque(maxlen=memory_size)
        self.field_names: list[str] = field_names
        self.experience: type = namedtuple("Experience", self.field_names)
        self.counter: int = 0
        self.device: str | None = device
        self.agent_ids: list[str] = agent_ids

        # TensorDict storage
        self._storage: TensorDict | None = None
        self._cursor = 0
        self._size = 0
        self._initialized = False

    def __len__(self) -> int:
        """Return the current size of internal memory.

        :return: Length of the memory
        :rtype: int
        """
        return self._size

    def _init_storage(self, data: dict[str, dict[str, Any]]) -> None:
        """Initialize the TensorDict storage given sample data.

        :param data: Sample transition data
        :type data: dict[str, dict[str, Any]]
        """
        storage_dict = {}
        for field in self.field_names:
            field_data = data[field]
            agent_dict = {}
            for agent_id in self.agent_ids:
                agent_data = field_data[agent_id]
                if isinstance(agent_data, torch.Tensor):
                    tensor = agent_data.unsqueeze(0)  # Add batch dim
                else:
                    tensor = torch.tensor(agent_data, dtype=torch.float32).unsqueeze(0)
                agent_dict[agent_id] = torch.zeros_like(
                    tensor.expand((self.memory_size, *tensor.shape[1:]))
                )
            storage_dict[field] = TensorDict(agent_dict, batch_size=[self.memory_size])

        self._storage = TensorDict(storage_dict, batch_size=[self.memory_size])
        self._initialized = True

    def _add_transition(self, data: dict[str, dict[str, Any]]) -> None:
        """Add a transition to the TensorDict storage.

        :param data: Transition data
        :type data: dict[str, dict[str, Any]]
        """
        if not self._initialized:
            self._init_storage(data)

        for field in self.field_names:
            for agent_id in self.agent_ids:
                agent_data = data[field][agent_id]
                if not isinstance(agent_data, torch.Tensor):
                    agent_data = torch.tensor(agent_data, dtype=torch.float32)
                if self.device is not None:
                    agent_data = agent_data.to(self.device)
                self._storage[field][agent_id][self._cursor] = agent_data

        self._cursor = (self._cursor + 1) % self.memory_size
        self._size = min(self._size + 1, self.memory_size)
        self.counter += 1

    @staticmethod
    def stack_transitions(transitions: list[NumpyObsType]) -> NumpyObsType:
        """Stacks transitions into a single array/dictionary/tuple of arrays.

        :param transitions: List of transitions
        :type transitions: list[NumpyObsType]

        :return: Stacked transitions
        :rtype: NumpyObsType
        """
        # Identify the type of the transition
        field_type = type(transitions[0])

        # Stack the transitions into a single array or tuple/dictionary of arrays
        ts = []
        for item in transitions:
            if field_type is dict:
                converted = {k: np.array(v) for k, v in item.items()}
            elif field_type is tuple:
                converted = tuple(np.array(v) for v in item)
            else:
                converted = item

            ts.append(converted)

        if field_type is dict:
            _ts = {}
            for k in ts[0].keys():
                kts = np.array([t[k] for t in ts])
                _ts[k] = np.expand_dims(kts, axis=1) if kts.ndim == 1 else kts

            ts = _ts
        elif field_type is tuple:
            _ts = []
            for i in range(len(ts[0])):
                its = np.array([t[i] for t in ts])
                _ts.append(np.expand_dims(its, axis=1) if its.ndim == 1 else its)

            ts = tuple(_ts)
        else:
            ts = np.array(ts)
            if ts.ndim == 1:
                ts = np.expand_dims(ts, axis=1)

        return ts

    def save_to_memory_single_env(self, *args: dict[str, NumpyObsType]) -> None:
        """Save experience to memory.

        :param args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        :type args: Any
        """
        # Convert to the internal format
        transition_data = {field: args[i] for i, field in enumerate(self.field_names)}
        self._add_transition(transition_data)

    def save_to_memory_vect_envs(self, *args: dict[str, NumpyObsType]) -> None:
        """Save multiple experiences to memory.

        :param args: Variable length argument list. Contains batched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        :type args: Any
        """
        # Reorganize batched data
        args = self._reorganize_dicts(*args)
        for transition in zip(*args, strict=False):
            transition_data = {
                field: transition[i] for i, field in enumerate(self.field_names)
            }
            self._add_transition(transition_data)

    def _reorganize_dicts(
        self,
        *args: dict[str, NumpyObsType],
    ) -> tuple[list[dict[str, NumpyObsType]], ...]:
        """Reorganizes dictionaries from vectorized to unvectorized experiences.

        :param args: Variable length argument list of dictionaries
        :type args: dict[str, np.ndarray]
        :return: Reorganized dictionaries
        :rtype: tuple[list[dict[str, np.ndarray]], ...]
        """

        def maybe_to_array(value: np.ndarray | list[float] | list[int]) -> np.ndarray:
            return np.array(value) if not isinstance(value, np.ndarray) else value

        results = [[] for _ in range(len(args))]
        num_entries = len(next(iter(args[0].values())))
        for i in range(num_entries):
            for j, arg in enumerate(args):
                new_dict = {}
                for key, value in arg.items():
                    if isinstance(value, dict):
                        new_dict[key] = {
                            k: maybe_to_array(v[i]) for k, v in value.items()
                        }
                    elif isinstance(value, tuple):
                        new_dict[key] = tuple(maybe_to_array(v[i]) for v in value)
                    else:
                        new_dict[key] = maybe_to_array(value[i])

                results[j].append(new_dict)

        return tuple(results)

    def sample(self, batch_size: int, *args: Any) -> tuple:
        """Return sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param args: Additional arguments
        :type args: Any
        :return: Sampled experiences
        :rtype: Tuple
        """
        if self._size == 0:
            return tuple({} for _ in self.field_names)

        # Sample indices
        indices = torch.randperm(self._size)[:batch_size]

        # Sample from TensorDict
        sampled = self._storage[indices]

        # Convert back to the expected format
        transition = {}
        for field in self.field_names:
            field_dict = {}
            for agent_id in self.agent_ids:
                agent_data = sampled[field][agent_id]
                # Convert to numpy for compatibility
                field_dict[agent_id] = agent_data.cpu().numpy()
            transition[field] = field_dict

        return tuple(transition.values())
