import random
from collections import deque, namedtuple
from numbers import Number
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike

from agilerl.typing import NumpyObsType
from agilerl.utils.algo_utils import obs_to_tensor

NpTransitionType = Union[Number, ArrayLike, Dict[str, ArrayLike]]
TorchTransitionType = Union[torch.Tensor, Dict[str, torch.Tensor]]


class MultiAgentReplayBuffer:
    """The Multi-Agent Experience Replay Buffer class. Used to store multiple agents'
    experiences and allow off-policy learning.

    :param memory_size: Maximum length of the replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: List[str]
    :param agent_ids: Names of all agents that will act in the environment
    :type agent_ids: List[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: Optional[str]
    """

    def __init__(
        self,
        memory_size: int,
        field_names: List[str],
        agent_ids: List[str],
        device: Optional[str] = None,
    ):
        assert memory_size > 0, "Memory size must be greater than zero."
        assert len(field_names) > 0, "Field names must contain at least one field name."
        assert len(agent_ids) > 0, "Agent ids must contain at least one agent id."

        self.memory_size: int = memory_size
        self.memory: Deque = deque(maxlen=memory_size)
        self.field_names: List[str] = field_names
        self.experience: NamedTuple = namedtuple(
            "Experience", field_names=self.field_names
        )
        self.counter: int = 0
        self.device: Optional[str] = device
        self.agent_ids: List[str] = agent_ids

    def __len__(self) -> int:
        """
        Returns the current size of internal memory.

        :return: Length of the memory
        :rtype: int
        """
        return len(self.memory)

    @staticmethod
    def stack_transitions(transitions: List[NumpyObsType]) -> NumpyObsType:
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
        for ft in transitions:
            if field_type is dict:
                ft = {k: np.array(v) for k, v in ft.items()}
            elif field_type is tuple:
                ft = tuple(np.array(v) for v in ft)

            ts.append(ft)

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

    def _add(self, *args: Dict[str, NumpyObsType]) -> None:
        """
        Adds experience to memory.

        :param args: Variable length argument list for experience fields
        :type args: Any
        """
        e = self.experience(*args)
        self.memory.append(e)

    def _process_transition(
        self, experiences: List[NamedTuple], np_array: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Returns transition dictionary from experiences.

        :param experiences: List of experiences
        :type experiences: List[NamedTuple]
        :param np_array: Flag to return numpy arrays instead of tensors, defaults to False
        :type np_array: bool, optional
        :return: Transition dictionary
        :rtype: Dict[str, Dict[str, Any]]
        """
        transition = {field: {} for field in self.field_names}
        experiences_filtered = [e for e in experiences if e is not None]

        for field in self.field_names:
            is_binary_field = field in [
                "done",
                "termination",
                "terminated",
                "truncation",
                "truncated",
            ]

            for agent_id in self.agent_ids:
                # Get field values for each agent
                ts = [getattr(e, field)[agent_id] for e in experiences_filtered]

                # Stack transitions if necessary
                ts = MultiAgentReplayBuffer.stack_transitions(ts)

                if is_binary_field and not np.isnan(ts).any():
                    ts = ts.astype(np.uint8)

                if not np_array:
                    ts = obs_to_tensor(ts, self.device)

                transition[field][agent_id] = ts

        return transition

    def sample(self, batch_size: int, *args: Any) -> Tuple:
        """
        Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param args: Additional arguments
        :type args: Any
        :return: Sampled experiences
        :rtype: Tuple
        """
        experiences = random.sample(self.memory, k=batch_size)
        transition = self._process_transition(experiences)
        return tuple(transition.values())

    def save_to_memory_single_env(self, *args: Dict[str, NumpyObsType]) -> None:
        """
        Saves experience to memory.

        :param args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        :type args: Any
        """
        self._add(*args)
        self.counter += 1

    def _reorganize_dicts(
        self, *args: Dict[str, NumpyObsType]
    ) -> Tuple[List[Dict[str, NumpyObsType]], ...]:
        """
        Reorganizes dictionaries from vectorized to unvectorized experiences.

        :param args: Variable length argument list of dictionaries
        :type args: Dict[str, np.ndarray]
        :return: Reorganized dictionaries
        :rtype: Tuple[List[Dict[str, np.ndarray]], ...]
        """

        def maybe_to_array(value):
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

    def save_to_memory_vect_envs(self, *args: Dict[str, NumpyObsType]) -> None:
        """
        Saves multiple experiences to memory.

        :param args: Variable length argument list. Contains batched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        :type args: Any
        """
        args = self._reorganize_dicts(*args)
        for transition in zip(*args):
            self._add(*transition)
            self.counter += 1

    def save_to_memory(
        self, *args: Dict[str, NumpyObsType], is_vectorised: bool = False
    ) -> None:
        """
        Applies appropriate save_to_memory function depending on whether
        the environment is vectorized or not.

        :param args: Variable length argument list. Contains batched or unbatched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        :type args: Any
        :param is_vectorised: Boolean flag indicating if the environment has been vectorized
        :type is_vectorised: bool
        """
        if is_vectorised:
            self.save_to_memory_vect_envs(*args)
        else:
            self.save_to_memory_single_env(*args)
