import random
from collections import deque, namedtuple

import numpy as np
import torch

from agilerl.components.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    """The Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param n_actions: Action dimension
    :type n_actions: int
    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: List[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional
    """

    def __init__(self, action_dim, memory_size, field_names, device=None):
        self.n_actions = action_dim
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=field_names)
        self.counter = 0  # update cycle counter
        self.device = device

    def __len__(self):
        return len(self.memory)

    def _add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        """
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None], axis=0)
        ).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        )
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float()
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences if e is not None], axis=0)
        ).float()
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float()

        if self.device is not None:
            states, actions, rewards, next_states, dones = (
                states.to(self.device),
                actions.to(self.device),
                rewards.to(self.device),
                next_states.to(self.device),
                dones.to(self.device),
            )

        return (states, actions, rewards, next_states, dones)

    def save2memory(self, state, action, reward, next_state, done):
        """Saves experience to memory.

        :param state: Environment observation
        :type state: float or List[float]
        :param action: Action in environment
        :type action: float or List[float]
        :param reward: Reward from environment
        :type reward: float
        :param next_state: Environment observation of next state
        :type next_state: float or List[float]
        :param done: True if environment episode finished, else False
        :type done: bool
        """
        self._add(state, action, reward, next_state, done)
        self.counter += 1

    def save2memoryVectEnvs(self, states, actions, rewards, next_states, dones):
        """Saves multiple experiences to memory.

        :param states: Multiple environment observations in a batch
        :type states: List[float] or List[List[float]]
        :param actions: Multiple actions in environment a batch
        :type actions: List[float] or List[List[float]]
        :param rewards: Multiple rewards from environment in a batch
        :type rewards: List[float]
        :param next_states: Multiple environment observations of next states in a batch
        :type next_states: List[float] or List[List[float]]
        :param dones: True if environment episodes finished, else False, in a batch
        :type dones: List[bool]
        """
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self._add(state, action, reward, next_state, done)
            self.counter += 1


class MultiStepReplayBuffer(ReplayBuffer):
    """The Multi-step Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param n_actions: Action dimension
    :type n_actions: int
    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: List[str]
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
        self.num_envs = num_envs
        self.n_step_buffers = [deque(maxlen=n_step) for i in range(num_envs)]
        self.n_step = n_step
        self.gamma = gamma

    def save2memory(self, state, action, reward, next_state, done):
        """Saves experience to memory.

        :param state: Environment observation
        :type state: float or List[float]
        :param action: Action in environment
        :type action: float or List[float]
        :param reward: Reward from environment
        :type reward: float
        :param next_state: Environment observation of next state
        :type next_state: float or List[float]
        :param done: True if environment episode finished, else False
        :type done: bool
        """
        transition = self.experience(state, action, reward, next_state, done)
        self.n_step_buffers[0].append(transition)

        # single step transition is not ready
        if len(self.n_step_buffers[0]) < self.n_step:
            return ()

        # make a n-step transition
        state, action, reward, next_state, done = self._get_n_step_info(
            self.n_step_buffers[0], self.gamma
        )
        self._add(state, action, reward, next_state, done)
        self.counter += 1

        return transition

    def save2memoryVectEnvs(self, states, actions, rewards, next_states, dones):
        """Saves multiple experiences to memory.

        :param states: Multiple environment observations in a batch
        :type states: List[float] or List[List[float]]
        :param actions: Multiple actions in environment a batch
        :type actions: List[float] or List[List[float]]
        :param rewards: Multiple rewards from environment in a batch
        :type rewards: List[float]
        :param next_states: Multiple environment observations of next states in a batch
        :type next_states: List[float] or List[List[float]]
        :param dones: True if environment episodes finished, else False, in a batch
        :type dones: List[bool]
        """
        for state, action, reward, next_state, done, buffer in zip(
            states, actions, rewards, next_states, dones, self.n_step_buffers
        ):
            transition = self.experience(state, action, reward, next_state, done)
            buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffers[0]) < self.n_step:
            return ()
        else:
            for buffer in self.n_step_buffers:
                # make a n-step transition
                state, action, reward, next_state, done = self._get_n_step_info(
                    buffer, self.gamma
                )
                self._add(state, action, reward, next_state, done)
                self.counter += 1

            return states, actions, rewards, next_states, dones

    def sample_from_indices(self, idxs):
        """Returns sample of experiences from memory using provided indices.

        :param idxs: Indices to sample
        :type idxs: List[int]
        """
        experiences = [self.memory[i] for i in idxs]

        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None], axis=0)
        ).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        )
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float()
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences if e is not None], axis=0)
        ).float()
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float()

        if self.device is not None:
            states, actions, rewards, next_states, dones = (
                states.to(self.device),
                actions.to(self.device),
                rewards.to(self.device),
                next_states.to(self.device),
                dones.to(self.device),
            )

        return (states, actions, rewards, next_states, dones)

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step reward, next_state, and done."""
        # info of the last transition
        t = n_step_buffer[-1]
        vect_state, vect_action = t.state, t.action
        vect_reward, vect_next_state, vect_done = t.reward, t.next_state, t.done

        for transition in reversed(list(n_step_buffer)[:-1]):
            vect_r, vect_n_s, vect_d = (
                transition.reward,
                transition.next_state,
                transition.done,
            )

            vect_reward = vect_r + gamma * vect_reward * (1 - vect_d)
            vect_next_state, vect_done = (
                (vect_n_s, vect_d) if vect_d else (vect_next_state, vect_done)
            )

        return vect_state, vect_action, vect_reward, vect_next_state, vect_done


class PrioritizedReplayBuffer(MultiStepReplayBuffer):
    """The Prioritized Experience Replay Buffer class. Used to store experiences and allow
    off-policy learning.

    :param n_actions: Action dimension
    :type n_actions: int
    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: List[str]
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

    def _add(self, state, action, reward, next_state, done):
        super()._add(state, action, reward, next_state, done)
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

        states = torch.from_numpy(
            np.stack([e.state for e in experiences if e is not None], axis=0)
        ).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        )
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float()
        next_states = torch.from_numpy(
            np.stack([e.next_state for e in experiences if e is not None], axis=0)
        ).float()
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float()
        weights = torch.from_numpy(
            np.array([self._calculate_weight(i, beta) for i in idxs])
        ).float()

        if self.device is not None:
            states, actions, rewards, next_states, dones, weights = (
                states.to(self.device),
                actions.to(self.device),
                rewards.to(self.device),
                next_states.to(self.device),
                dones.to(self.device),
                weights.to(self.device),
            )

        return (states, actions, rewards, next_states, dones, weights, idxs)

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
