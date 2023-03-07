import torch
import numpy as np
from collections import deque, namedtuple
import random

class ReplayBuffer():
    def __init__(self, n_actions, memory_size, field_names, device):
        self.n_actions = n_actions
        self.memory = deque(maxlen = memory_size)
        self.experience = namedtuple("Experience", field_names=field_names)
        self.counter = 0    # update cycle counter
        self.device = device

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def save2memory(self, state, action, reward, next_state, done):
        self.add(state, action, reward, next_state, done)
        self.counter += 1

    def save2memoryVectEnvs(self, states, actions, rewards, next_states, dones):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.add(state, action, reward, next_state, done)
            self.counter += 1