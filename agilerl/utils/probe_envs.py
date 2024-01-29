import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ConstantRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [np.zeros((1, 1))]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table

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
        self.observation_space = spaces.Box(0.0, 0.0, (3, 64, 64))
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [np.zeros((1, 3, 64, 64))]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table

    def step(self, action):
        observation = np.zeros((3, 64, 64))
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 64, 64))
        info = {}
        return observation, info


class ObsDependentRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.last_obs = 1
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table

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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 64, 64))
        self.action_space = spaces.Discrete(1)
        self.last_obs = np.ones((3, 64, 64))
        self.sample_obs = [np.zeros((1, 3, 64, 64)), np.ones((1, 3, 64, 64))]
        self.q_values = [[-1.0], [1.0]]  # Correct Q values to learn, s x a table

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
        self.last_obs = random.choice([np.zeros((3, 64, 64)), np.ones((3, 64, 64))])
        info = {}
        return self.last_obs, info


class DiscountedRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.last_obs = 0
        self.sample_obs = [np.array([[1, 0]]), np.array([[0, 1]])]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table

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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 64, 64))
        self.action_space = spaces.Discrete(1)
        self.last_obs = np.zeros((3, 64, 64))
        self.sample_obs = [np.zeros((1, 3, 64, 64)), np.ones((1, 3, 64, 64))]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table

    def step(self, action):
        observation = np.ones((3, 64, 64))
        reward = np.mean(self.last_obs)  # Reward depends on observation
        terminated = int(np.mean(self.last_obs))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = np.ones((3, 64, 64))
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = np.zeros((3, 64, 64))
        info = {}
        return self.last_obs, info


class FixedObsPolicyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [np.array([[0]])]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table

    def step(self, action):
        observation = 0
        reward = [-1, 1][action[0]]  # Reward depends on action
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
        self.observation_space = spaces.Box(0.0, 0.0, (3, 64, 64))
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [np.zeros((1, 3, 64, 64))]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table

    def step(self, action):
        observation = np.zeros((3, 64, 64))
        reward = [-1, 1][action[0]]  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 64, 64))
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 64, 64))
        self.action_space = spaces.Discrete(2)
        self.last_obs = np.ones((3, 64, 64))
        self.sample_obs = [np.zeros((1, 3, 64, 64)), np.ones((1, 3, 64, 64))]
        self.q_values = [
            [1.0, -1.0],
            [-1.0, 1.0],
        ]  # Correct Q values to learn, s x a table

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
        self.last_obs = random.choice([np.zeros((3, 64, 64)), np.ones((3, 64, 64))])
        info = {}
        return self.last_obs, info


def check_with_probe_env(env, algo_class, algo_args, learn_steps=1000, device="cpu"):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, lr=0.01, device=device)

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        action_dim=algo_args["action_dim"],  # Number of agent actions
        memory_size=1000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    state, _ = env.reset()
    for _ in range(500):
        action = agent.getAction(np.expand_dims(state, 0), epsilon=1)
        next_state, reward, done, _, _ = env.step(action)
        memory.save2memory(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

    # Learn from experiences
    for _ in trange(learn_steps):
        experiences = memory.sample(agent.batch_size)
        # Learn according to agent's RL algorithm
        agent.learn(experiences)

    for sample_obs, q_values in zip(env.sample_obs, env.q_values):
        predicted_q_values = agent.actor(sample_obs).detach().cpu().numpy()[0]
        assert q_values == pytest.approx(predicted_q_values, 0.1)


if __name__ == "__main__":
    import pytest
    import torch
    from tqdm import trange

    from agilerl.algorithms.dqn import DQN
    from agilerl.components.replay_buffer import ReplayBuffer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vector_envs = [
        (ConstantRewardEnv(), 1000),
        (ObsDependentRewardEnv(), 1000),
        (DiscountedRewardEnv(), 3000),
        (FixedObsPolicyEnv(), 1000),
        (PolicyEnv(), 1000),
    ]

    for env, learn_steps in vector_envs:
        algo_args = {
            "state_dim": (env.observation_space.n,),
            "action_dim": env.action_space.n,
            "one_hot": True if env.observation_space.n > 1 else False,
        }

        check_with_probe_env(env, DQN, algo_args, learn_steps, device)

    image_envs = [
        (ConstantRewardImageEnv(), 1000),
        (ObsDependentRewardImageEnv(), 1000),
        (DiscountedRewardImageEnv(), 5000),
        (FixedObsPolicyImageEnv(), 1000),
        (PolicyImageEnv(), 1000),
    ]

    for env, learn_steps in image_envs:
        algo_args = {
            "state_dim": (env.observation_space.shape),
            "action_dim": env.action_space.n,
            "one_hot": False,
            "net_config": {
                "arch": "cnn",  # Network architecture
                "h_size": [32],  # Network hidden size
                "c_size": [32, 32],  # CNN channel size
                "k_size": [8, 4],  # CNN kernel size
                "s_size": [4, 2],  # CNN stride size
                "normalize": True,  # Normalize image from range [0,255] to [0,1]
            },
        }

        check_with_probe_env(env, DQN, algo_args, learn_steps, device)
