import torch
import numpy as np
from collections import deque, namedtuple
import random


class ReplayBuffer():
    """The Experience Replay Buffer class. Used to store experiences and allow off-policy learning.

    :param n_actions: Action dimension
    :type n_actions: int
    :param memory_size: Maximum length of replay buffer
    :type memory_size: int
    :param field_names: Field names for experience named tuple, e.g. ['state', 'action', 'reward']
    :type field_names: List[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    """

    def __init__(self, action_dim, memory_size, field_names, device='cpu'):
        self.n_actions = action_dim
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=field_names)
        self.counter = 0    # update cycle counter
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

        states = torch.from_numpy(np.stack(
            [e.state for e in experiences if e is not None], axis=0)).to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack(
            [e.next_state for e in experiences if e is not None], axis=0)).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(
            np.uint8)).float().to(self.device)

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
                states, actions, rewards, next_states, dones):
            self._add(state, action, reward, next_state, done)
            self.counter += 1
