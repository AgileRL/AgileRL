# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces


class DiscountedRewardEnv:
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
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {"agent_0": np.array([1]), "other_agent_0": np.array([1])}
        reward = (
            {"agent_0": 1, "other_agent_0": 0.5}
            if self.last_obs["agent_0"] == 1
            else {"agent_0": 0, "other_agent_0": 0}
        )  # Reward depends on observation  # Reward depends on observation
        terminated = {
            agent: obs[0] for agent, obs in self.last_obs.items()
        }  # Terminate after second step
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        self.last_obs = {"agent_0": np.array([1]), "other_agent_0": np.array([1])}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return self.last_obs, info

class DiscountedRewardImageEnv:
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
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {
            "agent_0": np.ones((1, 3, 3)),
            "other_agent_0": np.ones((1, 3, 3)),
        }
        reward = (
            {"agent_0": 1, "other_agent_0": 0.5}
            if np.mean(self.last_obs["agent_0"]) == 1
            else {"agent_0": 0, "other_agent_0": 0}
        )  # Reward depends on observation  # Reward depends on observation
        terminated = {
            "agent_0": int(np.mean(self.last_obs["agent_0"])),
            "other_agent_0": int(np.mean(self.last_obs["agent_0"])),
        }  # Terminate after second step
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        self.last_obs = {
            "agent_0": np.ones((1, 3, 3)),
            "other_agent_0": np.ones((1, 3, 3)),
        }
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return self.last_obs, info

class DiscountedRewardContActionsEnv:
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
            {"agent_0": np.array([[0.2]]), "other_agent_0": np.array([[0.4]])},
            {"agent_0": np.array([[0.8]]), "other_agent_0": np.array([[0.1]])},
        ]
        self.q_values = [
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {"agent_0": np.array([1]), "other_agent_0": np.array([1])}
        reward = (
            {"agent_0": 1, "other_agent_0": 0.5}
            if self.last_obs["agent_0"] == 1
            else {"agent_0": 0, "other_agent_0": 0}
        )  # Reward depends on observation  # Reward depends on observation
        terminated = {
            agent: obs[0] for agent, obs in self.last_obs.items()
        }  # Terminate after second step
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        self.last_obs = {"agent_0": np.array([1]), "other_agent_0": np.array([1])}
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return self.last_obs, info

class DiscountedRewardContActionsImageEnv:
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
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.2]]), "other_agent_0": np.array([[0.4]])},
            {"agent_0": np.array([[0.8]]), "other_agent_0": np.array([[0.1]])},
        ]
        self.q_values = [
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 0.99, "other_agent_0": 0.495},
            {"agent_0": 1.0, "other_agent_0": 0.5},
        ]  # Correct V values to learn, s table
        self.policy_values = [None, None]

    def step(
        self,
        action: dict[str, npt.NDArray] | npt.NDArray,
    ) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        observation = {
            "agent_0": np.ones((1, 3, 3)),
            "other_agent_0": np.ones((1, 3, 3)),
        }
        reward = (
            {"agent_0": 1, "other_agent_0": 0.5}
            if np.mean(self.last_obs["agent_0"]) == 1
            else {"agent_0": 0, "other_agent_0": 0}
        )  # Reward depends on observation  # Reward depends on observation
        terminated = {
            "agent_0": int(np.mean(self.last_obs["agent_0"])),
            "other_agent_0": int(np.mean(self.last_obs["agent_0"])),
        }  # Terminate after second step
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        self.last_obs = {
            "agent_0": np.ones((1, 3, 3)),
            "other_agent_0": np.ones((1, 3, 3)),
        }
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self.last_obs = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return self.last_obs, info
