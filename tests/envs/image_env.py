"""Minimal image-observation environment for testing CNN architectures."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DummyImageEnv(gym.Env):
    """Gymnasium environment with a ``Box(84, 84, 3)`` observation space.

    Used by test manifests to exercise CNN encoder paths without requiring
    heavy external packages (ALE, MuJoCo, etc.).
    """

    metadata = {"render_modes": []}

    def __init__(self, **kwargs):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}
