import random

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from tqdm import trange


class ConstantRewardEnv:
    def __init__(self):
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.observation_space = {
            "agent_0": spaces.Discrete(1),
            "other_agent_0": spaces.Discrete(1),
        }
        self.action_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }

        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])}
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.2, 0.8]]), "other_agent_0": np.array([[0.8, 0.2]])}
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct V values to learn, s table
        self.policy_values = [None]

    def step(self, action):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        reward = {"agent_0": 1, "other_agent_0": 0}  # Constant reward of 1
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return observation, info


class ConstantRewardImageEnv:
    def __init__(self):
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.observation_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
        }
        self.action_space = {
            "agent_0": spaces.Discrete(2),
            "other_agent_0": spaces.Discrete(2),
        }

        self.sample_obs = [
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))}
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.2, 0.8]]), "other_agent_0": np.array([[0.8, 0.2]])}
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct V values to learn, s table
        self.policy_values = [None]

    def step(self, action):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        reward = {"agent_0": 1, "other_agent_0": 0}  # Constant reward of 1
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return observation, info


class ConstantRewardContActionsEnv:
    def __init__(self):
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.observation_space = {
            "agent_0": spaces.Discrete(1),
            "other_agent_0": spaces.Discrete(1),
        }
        self.action_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1,)),
        }

        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])}
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.0]]), "other_agent_0": np.array([[1.0]])}
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct V values to learn, s table
        self.policy_values = [None]

    def step(self, action):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        reward = {"agent_0": 1, "other_agent_0": 0}  # Constant reward
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return observation, info


class ConstantRewardContActionsImageEnv:
    def __init__(self):
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents
        self.max_num_agents = len(self.possible_agents)
        self.num_agents = len(self.agents)

        self.observation_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1, 3, 3)),
        }
        self.action_space = {
            "agent_0": spaces.Box(0.0, 1.0, (1,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (1,)),
        }

        self.sample_obs = [
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))}
        ]
        self.sample_actions = [
            {"agent_0": np.array([[0.0]]), "other_agent_0": np.array([[1.0]])}
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [
            {"agent_0": 1.0, "other_agent_0": 0.0}
        ]  # Correct V values to learn, s table
        self.policy_values = [None]

    def step(self, action):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        reward = {"agent_0": 1, "other_agent_0": 0}  # Constant reward
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return observation, info


class ObsDependentRewardEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
            ]
        )
        info = {}
        return self.last_obs, info


class ObsDependentRewardImageEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
            ]
        )
        info = {}
        return self.last_obs, info


class ObsDependentRewardContActionsEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
            ]
        )
        info = {}
        return self.last_obs, info


class ObsDependentRewardContActionsImageEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
            ]
        )
        info = {}
        return self.last_obs, info


class DiscountedRewardEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return self.last_obs, info


class DiscountedRewardImageEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return self.last_obs, info


class DiscountedRewardContActionsEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return self.last_obs, info


class DiscountedRewardContActionsImageEnv:
    def __init__(self):
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

    def step(self, action):
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

    def reset(self):
        self.last_obs = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return self.last_obs, info


class FixedObsPolicyEnv:
    def __init__(self):
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
            {"agent_0": 1.0, "other_agent_0": 1.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {"agent_0": np.array([[1.0, 0.0]]), "other_agent_0": np.array([[0.0, 1.0]])}
        ]

    def step(self, action):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        reward = {
            "agent_0": [1, -1][action["agent_0"]],
            "other_agent_0": [-1, 1][action["other_agent_0"]],
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return observation, info


class FixedObsPolicyImageEnv:
    def __init__(self):
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
            {"agent_0": 1.0, "other_agent_0": 1.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {"agent_0": np.array([[1.0, 0.0]]), "other_agent_0": np.array([[0.0, 1.0]])}
        ]

    def step(self, action):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        reward = {
            "agent_0": [1, -1][action["agent_0"]],
            "other_agent_0": [-1, 1][action["other_agent_0"]],
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return observation, info


class FixedObsPolicyContActionsEnv:
    def __init__(self):
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
            {"agent_0": 0.0, "other_agent_0": 0.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {"agent_0": np.array([1.0]), "other_agent_0": np.array([0.0])}
        ]

    def step(self, action):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        reward = {
            "agent_0": -((1 - action["agent_0"]) ** 2),
            "other_agent_0": -((0 - action["other_agent_0"]) ** 2),
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"agent_0": np.array([0]), "other_agent_0": np.array([0])}
        info = {}
        return observation, info


class FixedObsPolicyContActionsImageEnv:
    def __init__(self):
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
            {"agent_0": 0.0, "other_agent_0": 0.0}
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {"agent_0": np.array([1.0]), "other_agent_0": np.array([0.0])}
        ]

    def step(self, action):
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

    def reset(self):
        observation = {
            "agent_0": np.zeros((1, 3, 3)),
            "other_agent_0": np.zeros((1, 3, 3)),
        }
        info = {}
        return observation, info


class PolicyEnv:
    def __init__(self):
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
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 0.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 0.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(self, action):
        observation = self.last_obs
        reward = {
            "agent_0": action["agent_0"] == self.last_obs["agent_0"],
            "other_agent_0": action["other_agent_0"] != self.last_obs["other_agent_0"],
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
                {"agent_0": np.array([0]), "other_agent_0": np.array([1])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([0])},
            ]
        )
        info = {}
        return self.last_obs, info


class PolicyImageEnv:
    def __init__(self):
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
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 0.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 0.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(self, action):
        observation = self.last_obs
        reward = {
            "agent_0": action["agent_0"] == np.mean(self.last_obs["agent_0"]),
            "other_agent_0": action["other_agent_0"]
            != np.mean(self.last_obs["other_agent_0"]),
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
            ]
        )
        info = {}
        return self.last_obs, info


class PolicyContActionsEnv:
    def __init__(self):
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
            "agent_0": spaces.Box(0.0, 1.0, (2,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (2,)),
        }
        self.sample_obs = [
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 0.0]]),
            },
            {
                "agent_0": np.array([[1.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 1.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 0.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 0.0},
            {"agent_0": -2.0, "other_agent_0": -2.0},
            {"agent_0": -2.0, "other_agent_0": -2.0},
            {"agent_0": -1.0, "other_agent_0": -1.0},
            {"agent_0": -1.0, "other_agent_0": -1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(self, action):
        observation = self.last_obs
        reward = {}
        if self.last_obs["agent_0"]:  # last obs = 1, policy should be [0, 1]
            reward["agent_0"] = -((0 - action["agent_0"][0]) ** 2) - (
                (1 - action["agent_0"][1]) ** 2
            )
        else:  # last obs = 0, policy should be [1, 0]
            reward["agent_0"] = -((1 - action["agent_0"][0]) ** 2) - (
                (0 - action["agent_0"][1]) ** 2
            )
        if self.last_obs["other_agent_0"]:  # last obs = 1, policy should be [1, 0]
            reward["other_agent_0"] = -((1 - action["other_agent_0"][0]) ** 2) - (
                (0 - action["other_agent_0"][1]) ** 2
            )
        else:  # last obs = 0, policy should be [0, 1]
            reward["other_agent_0"] = -((0 - action["other_agent_0"][0]) ** 2) - (
                (1 - action["other_agent_0"][1]) ** 2
            )
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
                {"agent_0": np.array([0]), "other_agent_0": np.array([1])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([0])},
            ]
        )
        info = {}
        return self.last_obs, info


class PolicyContActionsImageEnv:
    def __init__(self):
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
            "agent_0": spaces.Box(0.0, 1.0, (2,)),
            "other_agent_0": spaces.Box(0.0, 1.0, (2,)),
        }
        self.sample_obs = [
            {
                "agent_0": np.zeros((1, 1, 3, 3)),
                "other_agent_0": np.zeros((1, 1, 3, 3)),
            },
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 0.0]]),
            },
            {
                "agent_0": np.array([[1.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 1.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 0.0, "other_agent_0": 0.0},
            {"agent_0": 0.0, "other_agent_0": 0.0},
            {"agent_0": -2.0, "other_agent_0": -2.0},
            {"agent_0": -2.0, "other_agent_0": -2.0},
            {"agent_0": -1.0, "other_agent_0": -1.0},
            {"agent_0": -1.0, "other_agent_0": -1.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(self, action):
        observation = self.last_obs
        reward = {}
        # First, deal with agent_0
        if np.mean(self.last_obs["agent_0"]):  # last obs = 1, policy should be [0, 1]
            reward["agent_0"] = -((0 - action["agent_0"][0]) ** 2) - (
                (1 - action["agent_0"][1]) ** 2
            )
        else:  # last obs = 0, policy should be [1, 0]
            reward["agent_0"] = -((1 - action["agent_0"][0]) ** 2) - (
                (0 - action["agent_0"][1]) ** 2
            )

        # other_agent_0 should learn the opposite behaviour
        if np.mean(
            self.last_obs["other_agent_0"]
        ):  # last obs = 1, policy should be [1, 0]
            reward["other_agent_0"] = -((1 - action["other_agent_0"][0]) ** 2) - (
                (0 - action["other_agent_0"][1]) ** 2
            )
        else:  # last obs = 0, policy should be [0, 1]
            reward["other_agent_0"] = -((0 - action["other_agent_0"][0]) ** 2) - (
                (1 - action["other_agent_0"][1]) ** 2
            )

        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
            ]
        )
        info = {}
        return self.last_obs, info


class MultiPolicyEnv:
    def __init__(self):
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
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
            {"agent_0": np.array([[0]]), "other_agent_0": np.array([[1]])},
            {"agent_0": np.array([[1]]), "other_agent_0": np.array([[0]])},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 2.0, "other_agent_0": 2.0},
            {"agent_0": 2.0, "other_agent_0": 2.0},
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 0.0, "other_agent_0": 3.0},
            {"agent_0": 0.0, "other_agent_0": 3.0},
            {"agent_0": 3.0, "other_agent_0": 0.0},
            {"agent_0": 3.0, "other_agent_0": 0.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(self, action):
        observation = self.last_obs
        reward = {
            "agent_0": 2 * (action["agent_0"] == self.last_obs["agent_0"])
            + (action["other_agent_0"] == self.last_obs["other_agent_0"]),
            "other_agent_0": 2
            * (action["other_agent_0"] != self.last_obs["other_agent_0"])
            + (action["agent_0"] != self.last_obs["agent_0"]),
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.array([0]), "other_agent_0": np.array([0])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([1])},
                {"agent_0": np.array([0]), "other_agent_0": np.array([1])},
                {"agent_0": np.array([1]), "other_agent_0": np.array([0])},
            ]
        )
        info = {}
        return self.last_obs, info


class MultiPolicyImageEnv:
    def __init__(self):
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
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
            {"agent_0": np.zeros((1, 1, 3, 3)), "other_agent_0": np.ones((1, 1, 3, 3))},
            {"agent_0": np.ones((1, 1, 3, 3)), "other_agent_0": np.zeros((1, 1, 3, 3))},
        ]
        self.sample_actions = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
        ]
        self.q_values = [
            {"agent_0": 2.0, "other_agent_0": 2.0},
            {"agent_0": 2.0, "other_agent_0": 2.0},
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 1.0, "other_agent_0": 1.0},
            {"agent_0": 0.0, "other_agent_0": 3.0},
            {"agent_0": 0.0, "other_agent_0": 3.0},
            {"agent_0": 3.0, "other_agent_0": 0.0},
            {"agent_0": 3.0, "other_agent_0": 0.0},
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]
        self.policy_values = [
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
            {
                "agent_0": np.array([[1.0, 0.0]]),
                "other_agent_0": np.array([[1.0, 0.0]]),
            },
            {
                "agent_0": np.array([[0.0, 1.0]]),
                "other_agent_0": np.array([[0.0, 1.0]]),
            },
        ]

    def step(self, action):
        observation = self.last_obs
        reward = {
            "agent_0": 2
            * (np.mean(action["agent_0"]) == np.mean(self.last_obs["agent_0"]))
            + (
                np.mean(action["other_agent_0"])
                == np.mean(self.last_obs["other_agent_0"])
            ),
            "other_agent_0": 2
            * (
                np.mean(action["other_agent_0"])
                != np.mean(self.last_obs["other_agent_0"])
            )
            + (np.mean(action["agent_0"]) != np.mean(self.last_obs["agent_0"])),
        }  # Reward depends on action
        terminated = {"agent_0": True, "other_agent_0": True}
        truncated = {"agent_0": False, "other_agent_0": False}
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice(
            [
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
                {"agent_0": np.zeros((1, 3, 3)), "other_agent_0": np.ones((1, 3, 3))},
                {"agent_0": np.ones((1, 3, 3)), "other_agent_0": np.zeros((1, 3, 3))},
            ]
        )
        info = {}
        return self.last_obs, info


def prepare_ma_states(states, observation_space, device="cpu"):
    processed_states = {}
    for agent_id, state in states.items():
        agent_space = observation_space[agent_id]
        if isinstance(agent_space, spaces.Discrete):
            processed_states[agent_id] = (
                nn.functional.one_hot(
                    torch.Tensor(state).long(), num_classes=agent_space.n
                )
                .float()
                .squeeze(1)
                .to(device)
            )
        else:
            processed_states[agent_id] = torch.Tensor(state).to(device)
    return processed_states


def prepare_ma_actions(actions, device="cpu"):
    actions = {
        agent_id: torch.Tensor(action).to(device)
        for (agent_id, action) in actions.items()
    }
    return actions


def check_policy_q_learning_with_probe_env(
    env, algo_class, algo_args, memory, learn_steps=1000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, vect_noise_dim=1, device=device)

    state, _ = env.reset()
    agent.set_training_mode(True)
    for _ in range(1000):
        # Make vectorized
        state = {agent_id: np.expand_dims(s, 0) for agent_id, s in state.items()}
        processed_action, raw_action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(processed_action)
        reward = {
            agent_id: np.expand_dims(np.array(r), 0) for agent_id, r in reward.items()
        }
        done = {
            agent_id: np.expand_dims(np.array(d), 0) for agent_id, d in done.items()
        }
        mem_next_state = {
            agent_id: np.expand_dims(ns, 0) for agent_id, ns in next_state.items()
        }
        memory.save_to_memory(
            state, raw_action, reward, mem_next_state, done, is_vectorised=True
        )
        state = next_state
        if done[agent.agent_ids[0]]:
            state, _ = env.reset()

    # Learn from experiences
    for _ in trange(learn_steps):
        experiences = memory.sample(agent.batch_size)

        # Learn according to agent's RL algorithm
        agent.learn(experiences)

    agent.set_training_mode(False)
    with torch.no_grad():
        for agent_id in agent.agent_ids:
            actor = agent.actors[agent_id]
            critic = agent.critics[agent_id]
            for sample_obs, sample_action, q_values, policy_values in zip(
                env.sample_obs, env.sample_actions, env.q_values, env.policy_values
            ):

                state = prepare_ma_states(sample_obs, agent.observation_space, device)

                if q_values is not None:
                    action = prepare_ma_actions(sample_action, device)
                    stacked_actions = torch.cat(list(action.values()), dim=1)
                    predicted_q_values = (
                        critic(state, stacked_actions).detach().cpu().numpy()[0]
                    )
                    # print("---")
                    # print(agent_id, "q", q_values[agent_id], predicted_q_values)
                    # assert np.allclose(q_values[agent_id], predicted_q_values, atol=0.1):
                    if not np.allclose(
                        q_values[agent_id], predicted_q_values, atol=0.1
                    ):
                        print(agent_id, "q", q_values[agent_id], predicted_q_values)

                if policy_values is not None:
                    predicted_policy_values = (
                        actor(state[agent_id]).detach().cpu().numpy()[0]
                    )

                    # print(agent_id, "pol", policy_values[agent_id], predicted_policy_values)
                    # assert np.allclose(policy_values[agent_id], predicted_policy_values, atol=0.1)
                    if not np.allclose(
                        policy_values[agent_id], predicted_policy_values, atol=0.1
                    ):
                        print(
                            agent_id,
                            "pol",
                            policy_values[agent_id],
                            predicted_policy_values,
                        )


def check_on_policy_learning_with_probe_env(
    env, algo_class, algo_args, learn_steps=1000, device="cpu", discrete=True
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    for i in trange(learn_steps):

        state, _ = env.reset()
        states = {agent_id: [] for agent_id in agent.agent_ids}
        actions = {agent_id: [] for agent_id in agent.agent_ids}
        log_probs = {agent_id: [] for agent_id in agent.agent_ids}
        rewards = {agent_id: [] for agent_id in agent.agent_ids}
        dones = {agent_id: [] for agent_id in agent.agent_ids}
        values = {agent_id: [] for agent_id in agent.agent_ids}

        done = {agent_id: np.zeros((1,)) for agent_id in agent.agent_ids}

        for _ in range(100):
            # Make vectorized
            state = {agent_id: np.expand_dims(s, 0) for agent_id, s in state.items()}

            action, log_prob, _, value = agent.get_action(obs=state)

            action = {agent: act[0].squeeze() for agent, act in action.items()}
            log_prob = {agent: lp[0] for agent, lp in log_prob.items()}
            value = {agent: val[0] for agent, val in value.items()}

            next_state, reward, termination, truncation, info = env.step(action)

            next_done = {}
            for agent_id in agent.agent_ids:
                states[agent_id].append(state[agent_id])
                actions[agent_id].append(action[agent_id])
                log_probs[agent_id].append(log_prob[agent_id])
                rewards[agent_id].append(reward[agent_id])
                dones[agent_id].append(done[agent_id])
                values[agent_id].append(value[agent_id])
                next_done[agent_id] = np.logical_or(
                    termination[agent_id], truncation[agent_id]
                ).astype(np.int8)

            next_done = {agent: np.array([n_d]) for agent, n_d in next_done.items()}

            state = next_state
            done = next_done

            done = {
                agent_id: np.expand_dims(np.array(d), 0)
                for agent_id, d in termination.items()
            }
            if done[agent.agent_ids[0]]:
                state, _ = env.reset()

        # Learn from experiences
        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
            next_done,
        )
        _loss = agent.learn(experiences)
        # if i < 20:
        #     print("Loss = ", _loss)

    with torch.no_grad():
        for agent_id in agent.observation_space.keys():
            actor = agent.actors[agent_id]
            critic = agent.critics[agent_id]
            for sample_obs, v_values, policy_values in zip(
                env.sample_obs, env.v_values, env.policy_values
            ):
                state = prepare_ma_states(sample_obs, agent.observation_space, device)

                if v_values is not None:
                    predicted_v_values = (
                        critic(state[agent_id]).detach().cpu().numpy()[0]
                    )

                    print(agent_id, "v", v_values[agent_id], predicted_v_values)
                    # assert np.allclose(v_values[agent_id], predicted_v_values, atol=0.1):
                    # if not np.allclose(
                    #     v_values[agent_id], predicted_v_values, atol=0.1
                    # ):
                    #     print(
                    #         "FAILURE: ",
                    #         agent_id,
                    #         "v",
                    #         v_values[agent_id],
                    #         predicted_v_values,
                    #     )
                    # else:
                    #     print(
                    #         "SUCCESS: ",
                    #         agent_id,
                    #         "v",
                    #         v_values[agent_id],
                    #         predicted_v_values,
                    #     )

                if policy_values is not None:
                    if discrete:
                        _, _, _ = actor(state[agent_id])
                        predicted_policy_values = (
                            actor.head_net.dist.distribution.probs.detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        _, _, _ = actor(state[agent_id])
                        predicted_policy_values = (
                            actor.head_net.dist.distribution.loc.detach().cpu().numpy()
                        )
                    print(
                        agent_id,
                        "pol",
                        policy_values[agent_id],
                        predicted_policy_values,
                    )
                    # assert np.allclose(policy_values[agent_id], predicted_policy_values, atol=0.1)
                    # if not np.allclose(
                    #     policy_values[agent_id], predicted_policy_values, atol=0.1
                    # ):
                    #     print(
                    #         "FAILURE: ",
                    #         agent_id,
                    #         "pol",
                    #         policy_values[agent_id],
                    #         predicted_policy_values,
                    #     )
                    # else:
                    #     print(
                    #         "SUCCESS: ",
                    #         agent_id,
                    #         "pol",
                    #         policy_values[agent_id],
                    #         predicted_policy_values,
                    #     )


if __name__ == "__main__":
    from agilerl.algorithms import IPPO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vector_envs = [
        (ConstantRewardEnv(), 10),
        (ObsDependentRewardEnv(), 10),
        (DiscountedRewardEnv(), 30),
        (FixedObsPolicyEnv(), 10),
        (PolicyEnv(), 10),
        (MultiPolicyEnv(), 10),
    ]

    for env, learn_steps in vector_envs:
        algo_args = {
            "observation_spaces": [space for space in env.observation_space.values()],
            "action_spaces": [space for space in env.action_space.values()],
            "agent_ids": env.agents,
            "lr": 1e-2,
            "net_config": {
                "encoder_config": {"hidden_size": [16, 16], "init_layers": False},
                "head_config": {"hidden_size": [16], "init_layers": False},
            },
        }

        check_on_policy_learning_with_probe_env(
            env, IPPO, algo_args, learn_steps, device, discrete=True
        )

    image_envs = [
        (ConstantRewardImageEnv(), 10),
        (ObsDependentRewardImageEnv(), 30),
        (DiscountedRewardImageEnv(), 60),
        (FixedObsPolicyImageEnv(), 10),
        (PolicyImageEnv(), 20),
        (MultiPolicyImageEnv(), 20),
    ]

    for env, learn_steps in image_envs:
        algo_args = {
            "observation_spaces": [space for space in env.observation_space.values()],
            "action_spaces": [space for space in env.action_space.values()],
            "agent_ids": env.agents,
            "lr": 1e-2,
            "net_config": {
                "encoder_config": {
                    "channel_size": [16],
                    "kernel_size": [3],
                    "stride_size": [1],
                },
                "head_config": {"hidden_size": [32], "output_activation": "Sigmoid"},
            },
            "normalize_images": False,
        }

        check_on_policy_learning_with_probe_env(
            env, IPPO, algo_args, learn_steps, device, discrete=True
        )

    cont_vector_envs = [
        (ConstantRewardContActionsEnv(), 10),
        (ObsDependentRewardContActionsEnv(), 10),
        (DiscountedRewardContActionsEnv(), 50),
        (FixedObsPolicyContActionsEnv(), 30),
        (PolicyContActionsEnv(), 30),
    ]

    for env, learn_steps in cont_vector_envs:
        algo_args = {
            "observation_spaces": [space for space in env.observation_space.values()],
            "action_spaces": [space for space in env.action_space.values()],
            "agent_ids": env.agents,
            "lr": 1e-2,
            "net_config": {
                "encoder_config": {"hidden_size": [16, 16], "init_layers": False},
                "head_config": {"hidden_size": [16], "init_layers": False},
            },
        }

        check_on_policy_learning_with_probe_env(
            env, IPPO, algo_args, learn_steps, device, discrete=False
        )

    cont_image_envs = [
        (ConstantRewardContActionsImageEnv(), 10),
        (ObsDependentRewardContActionsImageEnv(), 30),
        (DiscountedRewardContActionsImageEnv(), 80),
        (FixedObsPolicyContActionsImageEnv(), 30),
        (PolicyContActionsImageEnv(), 90),
    ]

    for env, learn_steps in cont_image_envs:
        algo_args = {
            "observation_spaces": [space for space in env.observation_space.values()],
            "action_spaces": [space for space in env.action_space.values()],
            "agent_ids": env.agents,
            "lr": 1e-2,
            "net_config": {
                "encoder_config": {
                    "channel_size": [16],
                    "kernel_size": [3],
                    "stride_size": [1],
                },
                "head_config": {"hidden_size": [32], "output_activation": "Sigmoid"},
            },
            "normalize_images": False,
        }

        check_on_policy_learning_with_probe_env(
            env, IPPO, algo_args, learn_steps, device, discrete=False
        )
