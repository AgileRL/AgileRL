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
    :type field_names: list[str]
    :param agent_ids: Names of all agents that will act in the environment
    :type agent_ids: list[str]
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to None
    :type device: str, optional

    """

    def __init__(self, memory_size, field_names, agent_ids, device=None):
        assert memory_size > 0, "Mmeory size must be greater than zero."
        assert len(field_names) > 0, "Field names must contain at least one field name."
        assert len(agent_ids) > 0, "Agent ids must contain at least one agent id."

        self.memory = memory_size
        self.memory = deque(maxlen=memory_size)
        self.field_names = field_names
        self.experience = namedtuple("Experience", field_names=self.field_names)
        self.counter = 0
        self.device = device
        self.agent_ids = agent_ids

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
            field_dict = {}
            for agent_id in self.agent_ids:
                ts = [getattr(e, field)[agent_id] for e in experiences if e is not None]

                # Handle numpy stacking
                ts = np.vstack(ts)

                if field in ["done", "termination", "truncation"]:
                    ts = ts.astype(np.uint8)

                if not np_array:
                    # Handle torch tensor creation
                    ts = torch.from_numpy(ts).float()

                    # Place on device
                    if self.device is not None:
                        ts = ts.to(self.device)

                field_dict[agent_id] = ts
            transition[field] = field_dict
        return transition

    def sample(self, batch_size):
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        """
        experiences = random.sample(self.memory, k=batch_size)
        transition = self._process_transition(experiences)
        return tuple(transition.values())

    def save2memory(self, *args):
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains batched or unbatched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        """
        self._add(*args)
        self.counter += 1
