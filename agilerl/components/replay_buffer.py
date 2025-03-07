import random
from collections import deque, namedtuple
from numbers import Number
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike

from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree
from agilerl.typing import NumpyObsType
from agilerl.utils.algo_utils import obs_to_tensor

NpTransitionType = Union[Number, ArrayLike, Dict[str, ArrayLike]]
TorchTransitionType = Union[torch.Tensor, Dict[str, torch.Tensor]]


class ReplayBuffer:
    """The Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: list[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(
        self, memory_size: int, field_names: List[str], device: Optional[str] = None
    ):
        assert memory_size > 0, "Memory size must be greater than zero."
        assert len(field_names) > 0, "Field names must contain at least one field name."

        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.field_names = field_names
        self.experience = namedtuple("Experience", field_names=self.field_names)
        self.counter = 0  # update cycle counter
        self.device = device

    def __len__(self) -> int:
        """Returns the current size of internal memory."""
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
                ts.append({k: np.expand_dims(v, axis=0) for k, v in ft.items()})
            elif field_type is tuple:
                ts.append(tuple(np.expand_dims(v, axis=0) for v in ft))
            else:
                ts.append(np.expand_dims(ft, axis=0))

        if field_type is dict:
            ts = {k: np.vstack([t[k] for t in ts]) for k in ts[0].keys()}
        elif field_type is tuple:
            ts = tuple(np.vstack([t[i] for t in ts]) for i in range(len(ts[0])))
        else:
            ts = np.vstack(ts)

        return ts

    def _add(self, *args: Any) -> None:
        """Adds experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        e = self.experience(*args)
        self.memory.append(e)

    def _process_transition(
        self, experiences: List[NamedTuple], np_array: bool = False
    ) -> Dict[str, Any]:
        """Returns transition dictionary from experiences.

        :param experiences: List of experiences
        :type experiences: list
        :param np_array: Flag to return numpy arrays instead of torch tensors, defaults to False
        :type np_array: bool, optional
        :return: Transition dictionary
        :rtype: dict
        """
        transition = {}
        for field in self.field_names:
            # Extract all of the transitions for the current field
            field_transitions: NpTransitionType = [
                getattr(e, field) for e in experiences if e is not None
            ]

            # Stack the transitions into a single array or tuple/dictionary of arrays
            ts = ReplayBuffer.stack_transitions(field_transitions)

            # Handle integer fields
            if field in [
                "done",
                "termination",
                "terminated",
                "truncation",
                "truncated",
            ]:
                ts = ts.astype(np.uint8)

            # Convert to torch tensor if specified
            if not np_array:
                ts = obs_to_tensor(ts, self.device)

            transition[field] = ts

        return transition

    def sample(
        self, batch_size: int, return_idx: bool = False, np_array: bool = False
    ) -> Tuple[Any, ...]:
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Boolean flag to return index of samples randomly selected, defaults to False
        :type return_idx: bool, optional
        :param np_array: Flag to return numpy arrays instead of torch tensors, defaults to False
        :type np_array: bool, optional
        :return: Tuple of sampled experiences
        :rtype: tuple
        """
        if return_idx:
            idxs = np.random.choice(len(self.memory), size=batch_size, replace=False)
            experiences = list(map(lambda i: self.memory[i], idxs))
            transition = self._process_transition(experiences, np_array)
            transition["idxs"] = idxs
        else:
            experiences = random.sample(self.memory, k=batch_size)
            transition = self._process_transition(experiences, np_array)

        return tuple(transition.values())

    def save_to_memory_single_env(self, *args: Any) -> None:
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        self._add(*args)
        self.counter += 1

    def save_to_memory_vect_envs(self, *args: Any) -> None:
        """Saves multiple experiences to memory.

        :param *args: Variable length argument list. Contains batched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        """
        for transition in zip(*args):
            self._add(*transition)
            self.counter += 1

    def save_to_memory(self, *args: Any, is_vectorised: bool = False) -> None:
        """Applies appropriate save_to_memory function depending on whether
        the environment is vectorised or not.

        :param *args: Variable length argument list. Contains batched or unbatched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        :param is_vectorised: Boolean flag indicating if the environment has been vectorised
        :type is_vectorised: bool
        """
        if is_vectorised:
            self.save_to_memory_vect_envs(*args)
        else:
            self.save_to_memory_single_env(*args)


class MultiStepReplayBuffer(ReplayBuffer):
    """The Multi-step Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: list[str]
    :param num_envs: Number of parallel environments for training
    :type num_envs: int
    :param n_step: Step number to calculate n-step td error, defaults to 3
    :type n_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(
        self,
        memory_size: int,
        field_names: List[str],
        num_envs: int,
        n_step: int = 3,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ):
        super().__init__(memory_size, field_names, device)
        assert (
            "reward" in field_names
        ), "Reward must be saved in replay buffer under the field name 'reward'."
        assert (
            "next_state" in field_names
        ), "Next state must be saved in replay buffer under the field name 'next_state'."
        assert (
            "done" in field_names
            or "termination" in field_names
            or "terminated" in field_names
        ), "Done/termination must be saved in replay buffer under the field name 'done', 'termination', or 'terminated'."

        self.num_envs = num_envs
        self.n_step_buffers = [deque(maxlen=n_step) for _ in range(num_envs)]
        self.args_deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def save_to_memory_single_env(self, *args: Any) -> Tuple[Any, ...]:
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        :return: The first saved transition
        :rtype: tuple
        """
        self.args_deque.append(args)
        transition = self.experience(*args)
        self.n_step_buffers[0].append(transition)

        # single step transition is not ready
        if len(self.n_step_buffers[0]) < self.n_step:
            return ()

        # make a n-step transition
        args = self._get_n_step_info(self.n_step_buffers[0], self.gamma)
        self._add(*args)
        self.counter += 1

        return self.args_deque[0]

    def save_to_memory_vect_envs(self, *args: Any) -> Tuple[Any, ...]:
        """Saves multiple experiences to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        :return: The first saved transition
        :rtype: tuple
        """
        self.args_deque.append(args)
        for buffer, *transition in zip(self.n_step_buffers, *args):
            transition = self.experience(*transition)
            buffer.append(transition)

        # single step transition is not ready
        if any(len(buffer) < self.n_step for buffer in self.n_step_buffers):
            return ()
        else:
            for buffer in self.n_step_buffers:
                # make a n-step transition
                single_step_args = self._get_n_step_info(buffer, self.gamma)
                self._add(*single_step_args)
                self.counter += 1

            return self.args_deque[0]

    def sample_from_indices(self, idxs: List[int]) -> Tuple[Any, ...]:
        """Returns sample of experiences from memory using provided indices.

        :param idxs: Indices to sample
        :type idxs: list[int]
        :return: Tuple of sampled experiences
        :rtype: tuple
        """
        experiences = list(map(lambda i: self.memory[i], idxs))
        transition = self._process_transition(experiences)
        return tuple(transition.values())

    def _get_n_step_info(
        self, n_step_buffer: Deque[NamedTuple], gamma: float
    ) -> Tuple[Any, ...]:
        """Returns n step reward, next_state, and done, as well as other saved transition elements, in order.

        :param n_step_buffer: Buffer containing n-step transitions
        :type n_step_buffer: deque
        :param gamma: Discount factor
        :type gamma: float
        :return: Tuple containing n-step transition elements
        :rtype: tuple
        """
        # info of the last transition
        t = [n_step_buffer[0]]
        transition = self._process_transition(t, np_array=True)

        vect_reward = transition["reward"][0]
        vect_next_state = transition["next_state"][0]
        if "done" in transition.keys():
            vect_done = transition["done"][0]
        elif "termination" in transition.keys():
            vect_done = transition["termination"][0]
        else:
            vect_done = transition["terminated"][0]

        for idx, ts in enumerate(list(n_step_buffer)[1:]):
            if not vect_done:
                vect_r, vect_n_s = (ts.reward, ts.next_state)

                if "done" in transition.keys():
                    vect_d = ts.done
                elif "termination" in transition.keys():
                    vect_d = ts.termination
                else:
                    vect_d = ts.terminated

                vect_reward += vect_r * gamma ** (idx + 1)
                vect_done = np.array([vect_d])
                vect_next_state = vect_n_s

        transition["reward"] = vect_reward
        transition["next_state"] = vect_next_state
        if "done" in transition.keys():
            transition["done"] = vect_done
        elif "termination" in transition.keys():
            transition["termination"] = vect_done
        else:
            transition["terminated"] = vect_done
        transition["state"] = transition["state"][0]
        transition["action"] = transition["action"][0]

        return tuple(transition.values())


class PrioritizedReplayBuffer(MultiStepReplayBuffer):
    """The Prioritized Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: list[str]
    :param num_envs: Number of parallel environments for training
    :type num_envs: int
    :param alpha: Alpha parameter for prioritized replay buffer, defaults to 0.6
    :type alpha: float, optional
    :param n_step: Step number to calculate n-step td error, defaults to 1
    :type n_step: int, optional
    :param gamma: Discount factor, defaults to 0.99
    :type gamma: float, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(
        self,
        memory_size: int,
        field_names: List[str],
        num_envs: int,
        alpha: float = 0.6,
        n_step: int = 1,
        gamma: float = 0.99,
        device: Optional[str] = None,
    ):
        super().__init__(memory_size, field_names, num_envs, n_step, gamma, device)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.memory_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def _add(self, *args: Any) -> None:
        """Adds experience to memory and updates priority trees.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        super()._add(*args)
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.memory_size

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Any, ...]:
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param beta: Beta parameter for importance sampling, defaults to 0.4
        :type beta: float, optional
        :return: Tuple of sampled experiences
        :rtype: tuple
        """
        idxs = self._sample_proportional(batch_size)
        experiences = [self.memory[i] for i in idxs]
        transition = self._process_transition(experiences)

        weights = torch.from_numpy(
            np.array([self._calculate_weight(i, beta) for i in idxs])
        ).float()

        if self.device is not None:
            weights = weights.to(self.device)

        transition["weights"] = weights
        transition["idxs"] = idxs

        return tuple(transition.values())

    def update_priorities(self, idxs: List[int], priorities: List[float]) -> None:
        """Update priorities of sampled transitions.

        :param idxs: Indices of sampled transitions
        :type idxs: list[int]
        :param priorities: New priorities of sampled transitions
        :type priorities: list[float]
        """
        for idx, priority in zip(idxs, priorities):
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size: int) -> List[int]:
        """Sample indices based on proportions.

        :param batch_size: Sample size
        :type batch_size: int
        :return: List of sampled indices
        :rtype: list[int]
        """
        idxs = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            idxs.append(idx)
        return idxs

    def _calculate_weight(self, idx: int, beta: float) -> float:
        """Calculate the weight of the experience at idx.

        :param idx: Index of the experience
        :type idx: int
        :param beta: Beta parameter for importance sampling
        :type beta: float
        :return: Weight of the experience
        :rtype: float
        """
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
