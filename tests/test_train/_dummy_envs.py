# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Importable dummy gym envs for manifest entrypoint resolution in tests."""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


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


class ImageObsEnv(gym.Env):
    """Single-agent env with a 3-D Box observation space (needs EvolvableCNN)."""

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 32, 32), dtype=np.uint8
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


class HeteroParallelEnv(ParallelEnv):
    """Two agents: one Dict-obs (multiinput), one vector-obs (mlp)."""

    metadata: ClassVar[dict] = {"name": "hetero_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["dict_agent", "vec_agent"]
        self.render_mode = render_mode
        self._obs = {
            "dict_agent": spaces.Dict(
                {"a": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)}
            ),
            "vec_agent": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        }
        self._act = {
            "dict_agent": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            "vec_agent": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        }

    def observation_space(self, agent):
        return self._obs[agent]

    def action_space(self, agent):
        return self._act[agent]

    def reset(self, *, seed=None, options=None):
        self.agents = list(self.possible_agents)
        return (
            {a: self._obs[a].sample() for a in self.agents},
            {a: {} for a in self.agents},
        )

    def step(self, actions):
        obs = {a: self._obs[a].sample() for a in self.agents}
        return (
            obs,
            dict.fromkeys(self.agents, 0.0),
            dict.fromkeys(self.agents, False),
            dict.fromkeys(self.agents, False),
            {a: {} for a in self.agents},
        )
