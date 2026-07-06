"""Importable dummy gym envs for manifest entrypoint resolution in tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DictObsEnv(gym.Env):
    """Single-agent env with a Dict observation space (needs EvolvableMultiInput)."""

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Dict(
            {
                "a": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                "b": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
