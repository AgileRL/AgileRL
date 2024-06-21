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

        self.memory_size = memory_size
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
                ts = np.stack(ts, axis=0)
                if len(ts.shape) == 1:
                    ts = np.expand_dims(ts, axis=1)

                if field in [
                    "done",
                    "termination",
                    "terminated",
                    "truncation",
                    "truncated",
                ]:
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

    def sample(self, batch_size, *args):
        """Returns sample of experiences from memory.

        :param batch_size: Number of samples to return
        :type batch_size: int
        """
        experiences = random.sample(self.memory, k=batch_size)
        transition = self._process_transition(experiences)
        return tuple(transition.values())

    def save_to_memory_single_env(self, *args):
        """Saves experience to memory.

        :param *args: Variable length argument list. Contains transition elements in consistent order,
            e.g. state, action, reward, next_state, done
        """
        self._add(*args)
        self.counter += 1

    def _reorganize_dicts(self, *args):
        """Reorgansizes dictionaries from vectorised to unvectorized experiences.

        Example input:
        {"agent1": np.array([[1, 2], [3, 4]]), "agent2": np.array([[1, 2], [3, 4]])}

        Example output:
        [{'agent1': array([[1, 2]]), 'agent2': array([[1, 2]])},
         {'agent1': array([[3, 4]]), 'agent2': array([[3, 4]])}]
        """
        results = [[] for _ in range(len(args))]
        num_entries = len(next(iter(args[0].values())))
        for i in range(num_entries):
            for j, arg in enumerate(args):
                new_dict = {
                    key: (
                        np.array(value[i]) if type(value[i]) != np.ndarray else value[i]
                    )
                    for key, value in arg.items()
                }
                results[j].append(new_dict)
        return tuple(results)

    def save_to_memory_vect_envs(self, *args):
        """Saves multiple experiences to memory.

        :param *args: Variable length argument list. Contains batched transition elements in consistent order,
            e.g. states, actions, rewards, next_states, dones
        """
        args = self._reorganize_dicts(*args)
        for transition in zip(*args):
            self._add(*transition)
            self.counter += 1

    def save_to_memory(self, *args, is_vectorised=False):
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
