import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from tqdm import trange

from agilerl.components.data import Transition


class ConstantRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [np.zeros((1, 1))]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = 0
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = 0
        info = {}
        return observation, info


class ConstantRewardImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 0.0, (1, 3, 3))
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [np.zeros((1, 1, 3, 3))]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((1, 3, 3))
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((1, 3, 3))
        info = {}
        return observation, info


class ConstantRewardDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 0.0, (1, 3, 3))}
        )
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [{"discrete": np.array([[0]]), "box": np.zeros((1, 1, 3, 3))}]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        info = {}
        return observation, info


class ConstantRewardContActionsEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.zeros((1, 1))]
        self.sample_actions = [[[1.0]]]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = 0
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = 0
        info = {}
        return observation, info


class ConstantRewardContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 0.0, (1, 3, 3))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.zeros((1, 1, 3, 3))]
        self.sample_actions = [[[1.0]]]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V value to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((1, 3, 3))
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((1, 3, 3))
        info = {}
        return observation, info


class ConstantRewardContActionsDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 0.0, (1, 3, 3))}
        )
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [{"discrete": np.array([[0]]), "box": np.zeros((1, 1, 3, 3))}]
        self.sample_actions = [np.array([[1.0]])]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        info = {}
        return observation, info


class ObsDependentRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.last_obs = 1
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[-1.0], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        reward = -1 if self.last_obs == 0 else 1  # Reward depends on observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([0, 1])
        info = {}
        return self.last_obs, info


class ObsDependentRewardImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Discrete(1)
        self.last_obs = np.ones((1, 3, 3))
        self.sample_obs = [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[-1.0], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        reward = (
            -1 if np.mean(self.last_obs) == 0.0 else 1
        )  # Reward depends on observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))])
        info = {}
        return self.last_obs, info


class ObsDependentRewardDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Discrete(1)
        self.last_obs = {"discrete": 1, "box": np.ones((1, 3, 3))}
        self.sample_obs = [
            {
                "discrete": [np.array([[0]]), np.array([[1]])],
                "box": [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))],
            }
        ]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[-1.0], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = {
            "discrete": self.last_obs["discrete"],
            "box": self.last_obs["box"],
        }
        reward = (
            -1 if np.mean(self.last_obs["box"]) != self.last_obs["discrete"] else 1
        )  # Reward depends on observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = {
            "discrete": random.choice([0, 1]),
            "box": random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))]),
        }
        info = {}
        return self.last_obs, info


class ObsDependentRewardContActionsEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = 1
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table
        self.policy_values = [None]  # Correct policy to learn
        self.v_values = [[-1.0], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        reward = -1 if self.last_obs == 0 else 1  # Reward depends on observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([0, 1])
        info = {}
        return self.last_obs, info


class ObsDependentRewardContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = np.ones((1, 3, 3))
        self.sample_obs = [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))]
        self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[-1.0], [1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = self.last_obs
        reward = (
            -1 if np.mean(self.last_obs) == 0.0 else 1
        )  # Reward depends on observationspaces.Box(0.0, 1.0
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))])
        info = {}
        return self.last_obs, info


class ObsDependentRewardContActionsDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = {"discrete": 1, "box": np.ones((1, 3, 3))}
        self.sample_obs = [
            {
                "discrete": [np.array([[0]]), np.array([[1]])],
                "box": [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))],
            }
        ]
        self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[-1.0], [1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = {
            "discrete": self.last_obs["discrete"],
            "box": self.last_obs["box"],
        }
        reward = (
            -1 if np.mean(self.last_obs["box"]) != self.last_obs["discrete"] else 1
        )  # Reward depends on observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = {
            "discrete": random.choice([0, 1]),
            "box": random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))]),
        }
        info = {}
        return self.last_obs, info


class DiscountedRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.last_obs = 0
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = 1
        reward = self.last_obs  # Reward depends on observation
        terminated = self.last_obs  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = 1
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = 0
        info = {}
        return self.last_obs, info


class DiscountedRewardImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Discrete(1)
        self.last_obs = np.zeros((1, 3, 3))
        self.sample_obs = [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = np.ones((1, 3, 3))
        reward = np.mean(self.last_obs)  # Reward depends on observation
        terminated = int(np.mean(self.last_obs))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = np.ones((1, 3, 3))
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = np.zeros((1, 3, 3))
        info = {}
        return self.last_obs, info


class DiscountedRewardDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Discrete(1)
        self.last_obs = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        self.sample_obs = [
            {
                "discrete": [np.array([[0]]), np.array([[1]])],
                "box": [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))],
            }
        ]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = {"discrete": 1, "box": np.ones((1, 3, 3))}
        reward = (
            np.mean(self.last_obs["box"]) + self.last_obs["discrete"]
        )  # Reward depends on observation
        terminated = int(np.mean(self.last_obs["box"]))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = {"discrete": 1, "box": np.ones((1, 3, 3))}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        info = {}
        return self.last_obs, info


class DiscountedRewardContActionsEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = 0
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = 1
        reward = self.last_obs  # Reward depends on observation
        terminated = self.last_obs  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = 1
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = 0
        info = {}
        return self.last_obs, info


class DiscountedRewardContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = np.zeros((1, 3, 3))
        self.sample_obs = [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))]
        self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = np.ones((1, 3, 3))
        reward = np.mean(self.last_obs)  # Reward depends on observation
        terminated = int(np.mean(self.last_obs))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = np.ones((1, 3, 3))
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = np.zeros((1, 3, 3))
        info = {}
        return self.last_obs, info


class DiscountedRewardContActionsDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        self.sample_obs = [
            {"discrete": np.array([[0]]), "box": np.zeros((1, 1, 3, 3))},
            {"discrete": np.array([[1]]), "box": np.ones((1, 1, 3, 3))},
        ]
        self.sample_actions = self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = {"discrete": 1, "box": np.ones((1, 3, 3))}
        reward = (
            np.mean(self.last_obs["box"]) + self.last_obs["discrete"]
        )  # Reward depends on observation
        terminated = int(np.mean(self.last_obs["box"]))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = {"discrete": 1, "box": np.ones((1, 3, 3))}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        info = {}
        return self.last_obs, info


class FixedObsPolicyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [np.array([[0]])]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[0.0, 1.0]]  # Correct policy to learn

    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        observation = 0
        reward = [-1, 1][action]  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = 0
        info = {}
        return observation, info


class FixedObsPolicyImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 0.0, (1, 3, 3))
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [np.zeros((1, 1, 3, 3))]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[0.0, 1.0]]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((1, 3, 3))
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        reward = [-1, 1][action]  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((1, 3, 3))
        info = {}
        return observation, info


class FixedObsPolicyDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 0.0, (1, 3, 3))}
        )
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [{"discrete": np.array([[0]]), "box": np.zeros((1, 1, 3, 3))}]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[0.0, 1.0]]  # Correct policy to learn

    def step(self, action):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        reward = [-1, 1][action]  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        info = {}
        return observation, info


class FixedObsPolicyContActionsEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.array([[0]])]
        self.sample_actions = [np.array([[1.0]])]
        self.q_values = np.array([[0.0]])  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[1.0]]  # Correct policy to learn

    def step(self, action):
        observation = 0
        reward = -((1 - action[0]) ** 2)  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = 0
        info = {}
        return observation, info


class FixedObsPolicyContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.zeros((1, 1, 3, 3))]
        self.sample_actions = [np.array([[1.0]])]
        self.q_values = np.array([[0.0]])  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[1.0]]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((1, 3, 3))
        reward = -((1 - action[0]) ** 2)  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((1, 3, 3))
        info = {}
        return observation, info


class FixedObsPolicyContActionsDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [{"discrete": np.array([[0]]), "box": np.zeros((1, 1, 3, 3))}]
        self.sample_actions = [np.array([[1.0]])]
        self.q_values = np.array([[0.0]])  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[1.0]]  # Correct policy to learn

    def step(self, action):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        reward = -((1 - action[0]) ** 2)  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        info = {}
        return observation, info


class PolicyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.last_obs = 0
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.q_values = [
            [1.0, -1.0],
            [-1.0, 1.0],
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        reward = (
            1 if action == self.last_obs else -1
        )  # Reward depends on action in observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([0, 1])
        info = {}
        return self.last_obs, info


class PolicyImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Discrete(2)
        self.last_obs = np.ones((1, 3, 3))
        self.sample_obs = [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))]
        self.q_values = [
            [1.0, -1.0],
            [-1.0, 1.0],
        ]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        reward = (
            1 if action == int(np.mean(self.last_obs)) else -1
        )  # Reward depends on action in observation
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))])
        info = {}
        return self.last_obs, info


class PolicyDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(2), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Discrete(2)
        self.last_obs = {"discrete": 1, "box": np.ones((1, 3, 3))}
        self.sample_obs = [
            {"discrete": 0, "box": np.zeros((1, 3, 3))},
            {"discrete": 0, "box": np.ones((1, 3, 3))},
            {"discrete": 1, "box": np.zeros((1, 3, 3))},
            {"discrete": 1, "box": np.ones((1, 3, 3))},
        ]
        self.q_values = [
            [1.0, -1.0],  # discrete=0, box=0
            [-1.0, -1.0],  # discrete=0, box=1
            [-1.0, -1.0],  # discrete=1, box=0
            [-1.0, 1.0],  # discrete=1, box=1
        ]
        self.v_values = [None]

    def step(self, action):
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        observation = {
            "discrete": self.last_obs["discrete"],
            "box": self.last_obs["box"],
        }
        reward = (
            1
            if action == self.last_obs["discrete"]
            and action == int(np.mean(self.last_obs["box"]))
            else -1
        )
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = {
            "discrete": random.choice([0, 1]),
            "box": random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))]),
        }
        info = {}
        return self.last_obs, info


class PolicyContActionsEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Box(0.0, 1.0, (2,))
        self.last_obs = 0
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.sample_actions = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
        self.q_values = [[0.0], [0.0]]  # Correct Q values to learn
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[1.0, 0.0], [0.0, 1.0]]  # Correct policy to learn

    def step(self, action):
        observation = self.last_obs
        if self.last_obs:  # last obs = 1, policy should be [0, 1]
            reward = -((0 - action[0]) ** 2) - (1 - action[1]) ** 2
        else:  # last obs = 0, policy should be [1, 0]
            reward = -((1 - action[0]) ** 2) - (0 - action[1]) ** 2
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([0, 1])
        info = {}
        return self.last_obs, info


class PolicyContActionsImageEnvSimple(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = np.zeros((1, 3, 3))
        self.sample_obs = [
            np.zeros((1, 1, 3, 3)),
            np.zeros((1, 1, 3, 3)),
            np.ones((1, 1, 3, 3)),
            np.ones((1, 1, 3, 3)),
        ]
        self.sample_actions = [
            np.array([[0.0]]),
            np.array([[1.0]]),
            np.array([[0.0]]),
            np.array([[1.0]]),
        ]
        self.q_values = [[0.0], [-1.0], [-1.0], [0.0]]  # Correct Q values to learn
        self.policy_values = [[0.0], [0.0], [1.0], [1.0]]  # Correct policy to learn
        self.v_values = [None]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        if int(np.mean(self.last_obs)):  # last obs = 1, policy should be [1]
            reward = -((1 - action[0]) ** 2)
        else:  # last obs = 0, policy should be [0]
            reward = -((0 - action[0]) ** 2)
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        # self.last_obs = random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))])
        if int(np.mean(self.last_obs)):
            self.last_obs = np.zeros((1, 3, 3))
        else:
            self.last_obs = np.ones((1, 3, 3))

        info = {}
        return self.last_obs, info


class PolicyContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (1, 3, 3))
        self.action_space = spaces.Box(0.0, 1.0, (2,))
        self.last_obs = np.zeros((1, 3, 3))
        self.sample_obs = [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))]
        self.sample_actions = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
        self.q_values = [[0.0], [0.0]]  # Correct Q values to learn
        self.policy_values = [[1.0, 0.0], [0.0, 1.0]]  # Correct policy to learn
        self.v_values = [None]  # Correct V values to learn, s table

    def step(self, action):
        observation = self.last_obs
        if int(np.mean(self.last_obs)):  # last obs = 1, policy should be [0, 1]
            reward = -((0 - action[0]) ** 2) - (1 - action[1]) ** 2
        else:  # last obs = 0, policy should be [1, 0]
            reward = -((1 - action[0]) ** 2) - (0 - action[1]) ** 2
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))])
        info = {}
        return self.last_obs, info


class PolicyContActionsDictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict(
            {"discrete": spaces.Discrete(1), "box": spaces.Box(0.0, 1.0, (1, 3, 3))}
        )
        self.action_space = spaces.Box(0.0, 1.0, (2,))
        self.last_obs = {"discrete": 0, "box": np.zeros((1, 3, 3))}
        self.sample_obs = [
            {"discrete": np.array([[0]]), "box": np.zeros((1, 1, 3, 3))},
            {"discrete": np.array([[1]]), "box": np.ones((1, 1, 3, 3))},
        ]
        self.sample_actions = [[1.0, 0.0], 0.0, 1.0]
        self.q_values = [[0.0], [0.0]]  # Correct Q values to learn
        self.policy_values = [[1.0, 0.0], [0.0, 1.0]]  # Correct policy to learn
        self.v_values = [None]  # Correct V values to learn, s table

    def step(self, action):
        observation = {
            "discrete": self.last_obs["discrete"],
            "box": self.last_obs["box"],
        }
        if self.last_obs["discrete"] and int(
            np.mean(self.last_obs["box"])
        ):  # last obs = 1, policy should be [0, 1]
            reward = -((0 - action[0]) ** 2) - (1 - action[1]) ** 2
        else:  # last obs = 0, policy should be [1, 0]
            reward = -((1 - action[0]) ** 2) - (0 - action[1]) ** 2
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = {
            "discrete": random.choice([0, 1]),
            "box": random.choice([np.zeros((1, 3, 3)), np.ones((1, 3, 3))]),
        }
        info = {}
        return self.last_obs, info


def check_q_learning_with_probe_env(
    env, algo_class, algo_args, memory, learn_steps=10000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    state, _ = env.reset()
    for _ in range(1000):
        if isinstance(state, dict):
            state = {k: np.expand_dims(v, 0) for k, v in state.items()}
        else:
            state = np.expand_dims(state, 0)
        action = agent.get_action(state, epsilon=1)
        next_state, reward, done, _, _ = env.step(action)
        transition = Transition(
            obs=state, action=action, reward=reward, next_obs=next_state, done=done
        ).to_tensordict()
        transition = transition.unsqueeze(0)
        transition.batch_size = [1]
        memory.add(transition)
        state = next_state
        if done:
            state, _ = env.reset()

    # Learn from experiences
    for i in trange(learn_steps):
        experiences = memory.sample(agent.batch_size)
        # Learn according to agent's RL algorithm
        loss = agent.learn(experiences)
        if i < 20:
            print("Loss = ", loss)

    for sample_obs, q_values in zip(env.sample_obs, env.q_values):
        predicted_q_values = agent.actor(sample_obs).detach().cpu().numpy()[0]
        print("q_values", q_values)
        print("predicted_q_values", predicted_q_values)
        # assert np.allclose(
        #     q_values, predicted_q_values, atol=0.1
        # ), f"{q_values} != {predicted_q_values}"


def check_policy_q_learning_with_probe_env(
    env, algo_class, algo_args, memory, learn_steps=10000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    state, _ = env.reset()
    for _ in range(5000):
        action = (
            (agent.action_space.high - agent.action_space.low)
            * np.random.rand(1, agent.action_dim).astype("float32")
        ) + agent.action_space.low
        action = action[0]
        next_state, reward, done, _, _ = env.step(action)
        transition = Transition(
            obs=state, action=action, reward=reward, next_obs=next_state, done=done
        ).to_tensordict()
        transition = transition.unsqueeze(0)
        transition.batch_size = [1]
        memory.add(transition)
        state = next_state
        if done:
            state, _ = env.reset()

    # Learn from experiences
    for i in trange(learn_steps):
        experiences = memory.sample(agent.batch_size)
        # Learn according to agent's RL algorithm
        loss = agent.learn(experiences)
        if i < 20:
            print("Loss = ", loss)

    for sample_obs, sample_action, q_values, policy_values in zip(
        env.sample_obs, env.sample_actions, env.q_values, env.policy_values
    ):
        if isinstance(sample_obs, dict):
            state = {
                k: torch.tensor(v).float().to(device) for k, v in sample_obs.items()
            }
        else:
            state = torch.tensor(sample_obs).float().to(device)

        agent.critic.eval()
        agent.actor.eval()
        action = torch.tensor(sample_action).float().to(device)
        predicted_q_values = agent.critic(state, action).detach().cpu().numpy()[0]
        print("q", q_values, predicted_q_values)
        # assert np.allclose(
        #     q_values, predicted_q_values, atol=0.15
        # ), f"{q_values} != {predicted_q_values}"

        if policy_values is not None:
            predicted_policy_values = agent.actor(sample_obs).detach().cpu().numpy()[0]

            print("pol", policy_values, predicted_policy_values)
            # assert np.allclose(
            #     policy_values, predicted_policy_values, atol=0.2
            # ), f"{policy_values} != {predicted_policy_values}"


def check_policy_on_policy_with_probe_env(
    env,
    algo_class,
    algo_args,
    learn_steps=5000,
    device="cpu",
    discrete=True,
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    for i in trange(learn_steps):
        state, _ = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        done = 0

        for j in range(200):
            if isinstance(state, dict):
                state = {k: np.expand_dims(v, 0) for k, v in state.items()}
            else:
                state = np.expand_dims(state, 0)

            action, log_prob, _, value = agent.get_action(state)

            action = action[0]
            log_prob = log_prob[0]
            value = value[0]
            next_state, reward, term, trunc, _ = env.step(action)
            next_done = np.logical_or(term, trunc).astype(np.int8)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state
            done = next_done

            if done:
                state, _ = env.reset()

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
        loss = agent.learn(experiences)
        if i < 20:
            print("Loss = ", loss)

    for sample_obs, v_values, policy_values in zip(
        env.sample_obs, env.v_values, env.policy_values
    ):
        if isinstance(sample_obs, dict):
            state = {
                k: torch.tensor(v).float().to(device) for k, v in sample_obs.items()
            }
        else:
            state = torch.tensor(sample_obs).float().to(device)

        if v_values is not None:
            predicted_v_values = agent.critic(state).detach().cpu().numpy()[0]
            print("v", v_values, predicted_v_values)
            # assert np.allclose(
            #     v_values, predicted_v_values, atol=0.2
            # ), f"{v_values} != {predicted_v_values}"

        if policy_values is not None:
            # Assumes it is always a discrete action space
            _, _, _ = agent.actor(state)
            predicted_policy_values = (
                agent.actor.head_net.dist.distribution.probs.detach().cpu().numpy()
            )

            print("pol", policy_values, predicted_policy_values)
            # assert np.allclose(
            #     policy_values, predicted_policy_values, atol=0.2
            # ), f"{policy_values} != {predicted_policy_values}"
