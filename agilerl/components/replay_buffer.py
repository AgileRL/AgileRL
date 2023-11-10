import random
from collections import deque, namedtuple

import numpy as np
import torch

from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    """The Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param action_dim: Action dimension
    :type action_dim: int
    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: list[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(self, action_dim, memory_size, field_names, device=None):
        assert action_dim > 0, "Action dimension must be greater than zero."
        assert memory_size > 0, "Mmeory size must be greater than zero."
        assert len(field_names) > 0, "Field names must contain at least one field name."

        self.action_dim = action_dim
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.field_names = field_names
        self.experience = namedtuple("Experience", field_names=self.field_names)
        self.counter = 0  # update cycle counter
        self.device = device

    def __len__(self):
        return len(self.memory)

    def _add(self, *args):
        """Adds experience to memory."""
        e = self.experience(*args)
        self.memory.append(e)

    def _process_transition(self, experiences, np_array=False):
        """Returns transition dictionary from experiences."""
        transition = {}
        for field in self.field_names:
            ts = [getattr(e, field) for e in experiences if e is not None]
            ts = np.vstack(ts)

            # Handle numpy stacking
            if field in ["done", "termination", "truncation"]:
                ts = ts.astype(np.uint8)

            if not np_array:
                # Handle torch tensor creation
                ts = torch.from_numpy(ts).float()

                # Place on device
                if self.device is not None:
                    ts = ts.to(self.device)

            transition[field] = ts
        return transition

    def sample(self, batch_size):
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        """
        experiences = random.sample(self.memory, k=batch_size)
        transition = self._process_transition(experiences)
        return tuple(transition.values())

    def save2memorySingleEnv(self, *args):
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        self._add(*args)
        self.counter += 1

    def save2memoryVectEnvs(self, *args):
        """Saves multiple experiences to memory.

        :param *args: Variable length argument list. Contains batched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        """
        for transition in zip(*args):
            self._add(*transition)
            self.counter += 1

    def save2memory(self, *args, is_vectorised=False):
        """Applies appropriate save2memory function depending on whether
        the environment is vectorised or not.

        :param *args: Variable length argument list. Contains batched or unbatched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        :param is_vectorised: Boolean flag indicating if the environment has been vectorised
        :type is_vectorised: bool
        """
        if is_vectorised:
            self.save2memoryVectEnvs(*args)
        else:
            self.save2memorySingleEnv(*args)


class MultiStepReplayBuffer(ReplayBuffer):
    """The Multi-step Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param action_dim: Action dimension
    :type action_dim: int
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
        action_dim,
        memory_size,
        field_names,
        num_envs,
        n_step=3,
        gamma=0.99,
        device=None,
    ):
        super().__init__(action_dim, memory_size, field_names, device)
        assert (
            "reward" in field_names
        ), "Reward must be saved in replay buffer under the field name 'reward'."
        assert (
            "next_state" in field_names
        ), "Next state must be saved in replay buffer under the field name 'next_state'."
        assert (
            "done" in field_names or "termination" in field_names
        ), "Done/termination must be saved in replay buffer under the field name 'done' or 'termination."
        self.num_envs = num_envs
        self.n_step_buffers = [deque(maxlen=n_step) for i in range(num_envs)]
        self.n_step = n_step
        self.gamma = gamma

    def save2memorySingleEnv(self, *args):
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        transition = self.experience(*args)
        self.n_step_buffers[0].append(transition)

        # single step transition is not ready
        if len(self.n_step_buffers[0]) < self.n_step:
            return ()

        # make a n-step transition
        args = self._get_n_step_info(self.n_step_buffers[0], self.gamma)
        self._add(*args)
        self.counter += 1

        return transition

    def save2memoryVectEnvs(self, *args):
        """Saves multiple experiences to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        for buffer, *transition in zip(self.n_step_buffers, *args):
            transition = self.experience(*transition)
            buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffers[0]) < self.n_step:
            return ()
        else:
            for buffer in self.n_step_buffers:
                # make a n-step transition
                single_step_args = self._get_n_step_info(buffer, self.gamma)
                self._add(*single_step_args)
                self.counter += 1

            return args

    def sample_from_indices(self, idxs):
        """Returns sample of experiences from memory using provided indices.

        :param idxs: Indices to sample
        :type idxs: list[int]
        """
        experiences = [self.memory[i] for i in idxs]
        transition = self._process_transition(experiences)
        return tuple(transition.values())

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Returns n step reward, next_state, and done, as well as other saved transition elements, in order."""
        # info of the last transition
        t = [n_step_buffer[-1]]
        transition = self._process_transition(t, np_array=True)

        vect_reward = transition["reward"]
        vect_next_state = transition["next_state"]
        if "done" in transition.keys():
            vect_done = transition["done"]
        else:
            vect_done = transition["termination"]

        for ts in reversed(list(n_step_buffer)[:-1]):
            vect_r, vect_n_s, vect_d = (
                ts.reward,
                ts.next_state,
                ts.done if "done" in transition.keys() else ts.termination,
            )

            vect_reward = vect_r + gamma * vect_reward * (1 - vect_d)
            vect_next_state, vect_done = (
                (np.expand_dims(vect_n_s, 0), np.array([[vect_d]]))
                if vect_d
                else (vect_next_state, vect_done)
            )

        transition["reward"] = vect_reward
        transition["next_state"] = vect_next_state
        if "done" in transition.keys():
            transition["done"] = vect_done
        else:
            transition["termination"] = vect_done

        return tuple(transition.values())


class PrioritizedReplayBuffer(MultiStepReplayBuffer):
    """The Prioritized Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param action_dim: Action dimension
    :type action_dim: int
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
        action_dim,
        memory_size,
        field_names,
        num_envs,
        alpha=0.6,
        n_step=1,
        gamma=0.99,
        device=None,
    ):
        super().__init__(
            action_dim, memory_size, field_names, num_envs, n_step, gamma, device
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.memory_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def _add(self, *args):
        super()._add(*args)
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.memory_size

    def sample(self, batch_size, beta=0.4):
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        """
        idxs = self._sample_proprtional(batch_size)
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

    def update_priorities(self, idxs, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(idxs, priorities):
            self.sum_tree[idx] = priority**self.alpha
            self.min_tree[idx] = priority**self.alpha
            self.max_priority = max(self.max_priority, priority)

    def _sample_proprtional(self, batch_size):
        """Sample indices based on proportions.

        :param batch_size: Sample size
        :type batch_size: int
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

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
