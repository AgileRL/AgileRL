# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces


class ObsDependentRewardEnv:
    def __init__(self) -> None:
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        self.observation_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }
        self.action_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }

        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[1]])},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[0.2, 0.8]]),
                "other_agent_0": np.array([[0.8, 0.2]]),
            },
            {
                "agent_0": np.array([[0.8, 0.2]]),
                "other_agent_0": np.array([[0.2, 0.8]]),
            },
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = self.last_obs
        reward = (
            {"agent_0": 1, "other_agent_0": 0}
            if self.last_obs["agent_0"] == 0
            else {"agent_0": 0, "other_agent_0": 1}
        )  # Reward depends on observation
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
            ],
        )
        info = {}
        return self.last_obs, info

class ObsDependentRewardImageEnv:
    def __init__(self) -> None:
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.last_obs = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        self.observation_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
        }
        self.action_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }

        self.sample_obs = [
            {
                "agent_0": np.zeros((1, 1, 3, 3)),
                "other_agent_0": np.zeros((1, 1, 3, 3)),
            },
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[0.2, 0.8]]),
                "other_agent_0": np.array([[0.8, 0.2]]),
            },
            {
                "agent_0": np.array([[0.8, 0.2]]),
                "other_agent_0": np.array([[0.2, 0.8]]),
            },
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = self.last_obs
        reward = (
            {"agent_0": 1, "other_agent_0": 0}
            if np.mean(self.last_obs["agent_0"]) == 0
            else {"agent_0": 0, "other_agent_0": 1}
        )  # Reward depends on observation
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
            ],
        )
        info = {}
        return self.last_obs, info

class ObsDependentRewardContActionsEnv:
    def __init__(self) -> None:
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        self.observation_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }
        self.action_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1,)),
        }

        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[1]])},
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.2]]), "other_agent_0": np.array([[0.0]])},
            {"agent_0": np.array([[0.8]]), "other_agent_0": np.array([[0.6]])},
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = self.last_obs
        reward = (
            {"agent_0": 1, "other_agent_0": 0}
            if self.last_obs["agent_0"] == 0
            else {"agent_0": 0, "other_agent_0": 1}
        )  # Reward depends on observation
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
            ],
        )
        info = {}
        return self.last_obs, info

class ObsDependentRewardContActionsImageEnv:
    def __init__(self) -> None:
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        self.observation_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
        }
        self.action_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1,)),
        }

        self.sample_obs = [
            {
                "agent_0": np.zeros((1, 1, 3, 3)),
                "other_agent_0": np.zeros((1, 1, 3, 3)),
            },
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.2]]), "other_agent_0": np.array([[0.0]])},
            {"agent_0": np.array([[0.8]]), "other_agent_0": np.array([[0.6]])},
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 1.0},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = self.last_obs
        reward = (
            {"agent_0": 1, "other_agent_0": 0}
            if np.mean(self.last_obs["agent_0"]) == 0
            else {"agent_0": 0, "other_agent_0": 1}
        )  # Reward depends on observation
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
            ],
        )
        info = {}
        return self.last_obs, info
