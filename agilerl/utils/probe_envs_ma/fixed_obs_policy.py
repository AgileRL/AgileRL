# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces


class FixedObsPolicyEnv:
    def __init__(self) -> None:
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        self.observation_space = {
            "agent_0": spaces.Discrete(1),
            "other_agent_0": spaces.Discrete(1),
        }
        self.action_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }

        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        reward = {
            "agent_0": [1, -1][int(np.asarray(action["agent_0"]).flat[0])],
            "other_agent_0": [-1, 1][int(np.asarray(action["other_agent_0"]).flat[0])],
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return observation, info

class FixedObsPolicyImageEnv:
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
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        reward = {
            "agent_0": [1, -1][int(np.asarray(action["agent_0"]).flat[0])],
            "other_agent_0": [-1, 1][int(np.asarray(action["other_agent_0"]).flat[0])],
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return observation, info

class FixedObsPolicyContActionsEnv:
    def __init__(self) -> None:
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        self.observation_space = {
            "agent_0": spaces.Discrete(1),
            "other_agent_0": spaces.Discrete(1),
        }
        self.action_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1,)),
        }

        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])},
        ]
        self.sample_actions = [
            {"agent_0": np.array([[1.0]]), "other_agent_0": np.array([[0.0]])},
        ]
        self.q_values = [
            {"agent_0": 0.0, "other_agent_0": 0.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {"agent_0": np.array([1.0]), "other_agent_0": np.array([0.0])},
        ]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        reward = {
            "agent_0": -((1 - action["agent_0"]) ** 2),
            "other_agent_0": -((0 - action["other_agent_0"]) ** 2),
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return observation, info

class FixedObsPolicyContActionsImageEnv:
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
            "agent_0": spaces.Box(0.0, 1.0, (1,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1,)),
        }

        self.sample_obs = [
            {
                "agent_0": np.zeros((1, 1, 3, 3)),
                "other_agent_0": np.zeros((1, 1, 3, 3)),
            },
        ]
        self.sample_actions = [
            {"agent_0": np.array([[1.0]]), "other_agent_0": np.array([[0.0]])},
        ]
        self.q_values = [
            {"agent_0": 0.0, "other_agent_0": 0.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {"agent_0": np.array([1.0]), "other_agent_0": np.array([0.0])},
        ]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        reward = {
            "agent_0": -((1 - action["agent_0"]) ** 2),
            "other_agent_0": -((0 - action["other_agent_0"]) ** 2),
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return observation, info
