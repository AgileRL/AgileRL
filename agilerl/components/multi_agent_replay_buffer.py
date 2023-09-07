import random
from collections import deque, namedtuple

import numpy as np
import torch


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
    :type device: str, optional

    """

    def __init__(self, memory_size, field_names, agent_ids, device=None):
        self.memory = memory_size
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=field_names)
        self.counter = 0
        self.device = device
        self.agent_ids = agent_ids

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

        if self.device is not None:
            states = {
                agent_id: torch.from_numpy(
                    np.stack([e.state[agent_id] for e in experiences])
                ).to(self.device)
                for agent_id in self.agent_ids
            }
            actions = {
                agent_id: torch.from_numpy(
                    np.stack([e.action[agent_id] for e in experiences])
                ).to(self.device)
                for agent_id in self.agent_ids
            }
            rewards = {
                agent_id: torch.from_numpy(
                    np.vstack([e.reward[agent_id] for e in experiences])
                )
                .float()
                .to(self.device)
                for agent_id in self.agent_ids
            }
            next_states = {
                agent_id: torch.from_numpy(
                    np.stack([e.next_state[agent_id] for e in experiences])
                )
                .float()
                .to(self.device)
                for agent_id in self.agent_ids
            }
            dones = {
                agent_id: torch.from_numpy(
                    np.vstack([e.done[agent_id] for e in experiences]).astype(np.uint8)
                )
                .float()
                .to(self.device)
                for agent_id in self.agent_ids
            }
        else:
            states = {
                agent_id: torch.from_numpy(
                    np.stack([e.state[agent_id] for e in experiences])
                )
                for agent_id in self.agent_ids
            }
            actions = {
                agent_id: torch.from_numpy(
                    np.stack([e.action[agent_id] for e in experiences])
                )
                for agent_id in self.agent_ids
            }
            rewards = {
                agent_id: torch.from_numpy(
                    np.vstack([e.reward[agent_id] for e in experiences])
                ).float()
                for agent_id in self.agent_ids
            }
            next_states = {
                agent_id: torch.from_numpy(
                    np.stack([e.next_state[agent_id] for e in experiences])
                ).float()
                for agent_id in self.agent_ids
            }
            dones = {
                agent_id: torch.from_numpy(
                    np.vstack([e.done[agent_id] for e in experiences]).astype(np.uint8)
                ).float()
                for agent_id in self.agent_ids
            }

        return states, actions, rewards, next_states, dones

    def save2memory(self, state, action, reward, next_state, done):
        """Saves experience to memory.

        :param state: Environment observation
        :type state: Dict[str, numpy.Array]
        :param action: Action in environment
        :type action: Dict[str, numpy.Array]
        :param reward: Reward from environment
        :type reward: dict[str, int]
        :param next_state: Environment observation of next state
        :type next_state: Dict[str, numpy.Array]
        :param done: True if environment episode finished, else False
        :type done: Dict[str, bool]
        """
        self._add(state, action, reward, next_state, done)
        self.counter += 1
