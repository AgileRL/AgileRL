import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from tqdm import trange


class ConstantRewardEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [np.zeros((1, 1))]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table

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
        self.observation_space = spaces.Box(0.0, 0.0, (3, 32, 32))
        self.action_space = spaces.Discrete(1)
        self.sample_obs = [np.zeros((1, 3, 32, 32))]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = np.zeros((3, 32, 32))
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 32, 32))
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
        self.observation_space = spaces.Box(0.0, 0.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.zeros((1, 3, 32, 32))]
        self.sample_actions = [[[1.0]]]
        self.q_values = [[1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[1.0]]  # Correct V value to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((3, 32, 32))
        reward = 1  # Constant reward of 1
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 32, 32))
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Discrete(1)
        self.last_obs = np.ones((3, 32, 32))
        self.sample_obs = [np.zeros((1, 3, 32, 32)), np.ones((1, 3, 32, 32))]
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
        self.last_obs = random.choice([np.zeros((3, 32, 32)), np.ones((3, 32, 32))])
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = np.ones((3, 32, 32))
        self.sample_obs = [np.zeros((1, 3, 32, 32)), np.ones((1, 3, 32, 32))]
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
        self.last_obs = random.choice([np.zeros((3, 32, 32)), np.ones((3, 32, 32))])
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Discrete(1)
        self.last_obs = np.zeros((3, 32, 32))
        self.sample_obs = [np.zeros((1, 3, 32, 32)), np.ones((1, 3, 32, 32))]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table

    def step(self, action):
        observation = np.ones((3, 32, 32))
        reward = np.mean(self.last_obs)  # Reward depends on observation
        terminated = int(np.mean(self.last_obs))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = np.ones((3, 32, 32))
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = np.zeros((3, 32, 32))
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = np.zeros((3, 32, 32))
        self.sample_obs = [np.zeros((1, 3, 32, 32)), np.ones((1, 3, 32, 32))]
        self.sample_actions = [[[1.0]], [[1.0]]]
        self.q_values = [[0.99], [1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [[0.99], [1.0]]  # Correct V values to learn, s table
        self.policy_values = [None]  # Correct policy to learn

    def step(self, action):
        observation = np.ones((3, 32, 32))
        reward = np.mean(self.last_obs)  # Reward depends on observation
        terminated = int(np.mean(self.last_obs))  # Terminate after second step
        truncated = False
        info = {}
        self.last_obs = np.ones((3, 32, 32))
        return observation, reward, terminated, truncated, info

    def reset(self):
        self.last_obs = np.zeros((3, 32, 32))
        info = {}
        return self.last_obs, info


class FixedObsPolicyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [np.array([[0]])]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table

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
        self.observation_space = spaces.Box(0.0, 0.0, (3, 32, 32))
        self.action_space = spaces.Discrete(2)
        self.sample_obs = [np.zeros((1, 3, 32, 32))]
        self.q_values = [[-1.0, 1.0]]  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table

    def step(self, action):
        observation = np.zeros((3, 32, 32))
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        reward = [-1, 1][action]  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 32, 32))
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.sample_obs = [np.zeros((1, 3, 32, 32))]
        self.sample_actions = [np.array([[1.0]])]
        self.q_values = np.array([[0.0]])  # Correct Q values to learn, s x a table
        self.v_values = [None]  # Correct V values to learn, s table
        self.policy_values = [[1.0]]  # Correct policy to learn

    def step(self, action):
        observation = np.zeros((3, 32, 32))
        reward = -((1 - action[0]) ** 2)  # Reward depends on action
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self):
        observation = np.zeros((3, 32, 32))
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Discrete(2)
        self.last_obs = np.ones((3, 32, 32))
        self.sample_obs = [np.zeros((1, 3, 32, 32)), np.ones((1, 3, 32, 32))]
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
        self.last_obs = random.choice([np.zeros((3, 32, 32)), np.ones((3, 32, 32))])
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
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (1,))
        self.last_obs = np.zeros((3, 32, 32))
        self.sample_obs = [
            np.zeros((1, 3, 32, 32)),
            np.zeros((1, 3, 32, 32)),
            np.ones((1, 3, 32, 32)),
            np.ones((1, 3, 32, 32)),
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
        # self.last_obs = random.choice([np.zeros((3, 32, 32)), np.ones((3, 32, 32))])
        if int(np.mean(self.last_obs)):
            self.last_obs = np.zeros((3, 32, 32))
        else:
            self.last_obs = np.ones((3, 32, 32))

        info = {}
        return self.last_obs, info


class PolicyContActionsImageEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(0.0, 1.0, (3, 32, 32))
        self.action_space = spaces.Box(0.0, 1.0, (2,))
        self.last_obs = np.zeros((3, 32, 32))
        self.sample_obs = [np.zeros((1, 3, 32, 32)), np.ones((1, 3, 32, 32))]
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
        self.last_obs = random.choice([np.zeros((3, 32, 32)), np.ones((3, 32, 32))])
        info = {}
        return self.last_obs, info


def check_q_learning_with_probe_env(
    env, algo_class, algo_args, memory, learn_steps=1000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    state, _ = env.reset()
    for _ in range(500):
        action = agent.get_action(np.expand_dims(state, 0), epsilon=1)
        next_state, reward, done, _, _ = env.step(action)
        memory.save_to_memory(state, action, reward, next_state, done)
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
        assert np.allclose(q_values, predicted_q_values, atol=0.1)


def check_policy_q_learning_with_probe_env(
    env, algo_class, algo_args, memory, learn_steps=1000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    state, _ = env.reset()
    for _ in range(5000):
        action = (
            (agent.max_action - agent.min_action)
            * np.random.rand(1, agent.action_dim).astype("float32")
        ) + agent.min_action
        action = action[0]
        next_state, reward, done, _, _ = env.step(action)
        memory.save_to_memory(state, action, reward, next_state, done)
        state = next_state
        if done:
            state, _ = env.reset()

    # Learn from experiences
    for _ in trange(learn_steps):
        experiences = memory.sample(agent.batch_size)
        # Learn according to agent's RL algorithm
        agent.learn(experiences)

    for sample_obs, sample_action, q_values, policy_values in zip(
        env.sample_obs, env.sample_actions, env.q_values, env.policy_values
    ):
        state = torch.tensor(sample_obs).float().to(device)
        action = torch.tensor(sample_action).float().to(device)
        if agent.arch == "mlp":
            input_combined = torch.cat([state, action], 1)
            predicted_q_values = agent.critic(input_combined).detach().cpu().numpy()[0]
        else:
            predicted_q_values = agent.critic(state, action).detach().cpu().numpy()[0]
        # print("---")
        # print("q", q_values, predicted_q_values)
        assert np.allclose(q_values, predicted_q_values, atol=0.1)

        if policy_values is not None:
            predicted_policy_values = agent.actor(sample_obs).detach().cpu().numpy()[0]

            # print("pol", policy_values, predicted_policy_values)
            assert np.allclose(policy_values, predicted_policy_values, atol=0.1)


def check_policy_on_policy_with_probe_env(
    env, algo_class, algo_args, learn_steps=5000, device="cpu"
):
    print(f"Probe environment: {type(env).__name__}")

    agent = algo_class(**algo_args, device=device)

    for _ in trange(learn_steps):
        state, _ = env.reset()
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        truncs = []

        for _ in range(100):
            action, log_prob, _, value = agent.get_action(np.expand_dims(state, 0))
            action = action[0]
            log_prob = log_prob[0]
            value = value[0]
            next_state, reward, done, trunc, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            truncs.append(trunc)

            state = next_state
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
        )
        agent.learn(experiences)

    for sample_obs, v_values in zip(env.sample_obs, env.v_values):
        state = torch.tensor(sample_obs).float().to(device)
        if v_values is not None:
            predicted_v_values = agent.critic(state).detach().cpu().numpy()[0]
            # print("---")
            # print("v", v_values, predicted_v_values)
            assert np.allclose(v_values, predicted_v_values, atol=0.1)

    if hasattr(env, "sample_actions"):
        for sample_action, policy_values in zip(env.sample_actions, env.policy_values):
            action = torch.tensor(sample_action).float().to(device)
            if policy_values is not None:
                predicted_policy_values = (
                    agent.actor(sample_obs).detach().cpu().numpy()[0]
                )
                # print("pol", policy_values, predicted_policy_values)
                assert np.allclose(policy_values, predicted_policy_values, atol=0.1)


# if __name__ == "__main__":
#     from agilerl.algorithms.ddpg import DDPG
#     from agilerl.algorithms.dqn import DQN
#     from agilerl.components.replay_buffer import ReplayBuffer

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     vector_envs = [
#         (ConstantRewardEnv(), 1000),
#         (ObsDependentRewardEnv(), 1000),
#         (DiscountedRewardEnv(), 3000),
#         (FixedObsPolicyEnv(), 1000),
#         (PolicyEnv(), 1000),
#     ]

#     for env, learn_steps in vector_envs:
#         algo_args = {
#             "state_dim": (env.observation_space.n,),
#             "action_dim": env.action_space.n,
#             "one_hot": True if env.observation_space.n > 1 else False,
#             "lr": 1e-2,
#         }

#         field_names = ["state", "action", "reward", "next_state", "done"]
#         memory = ReplayBuffer(
#             memory_size=1000,  # Max replay buffer size
#             field_names=field_names,  # Field names to store in memory
#             device=device,
#         )

#         check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)

#     image_envs = [
#         (ConstantRewardImageEnv(), 1000),
#         (ObsDependentRewardImageEnv(), 1000),
#         (DiscountedRewardImageEnv(), 5000),
#         (FixedObsPolicyImageEnv(), 1000),
#         (PolicyImageEnv(), 1000),
#     ]

#     for env, learn_steps in image_envs:
#         algo_args = {
#             "state_dim": (env.observation_space.shape),
#             "action_dim": env.action_space.n,
#             "one_hot": False,
#             "net_config": {
#                 "arch": "cnn",  # Network architecture
#                 "hidden_size": [32],  # Network hidden size
#                 "channel_size": [32, 32],  # CNN channel size
#                 "kernel_size": [8, 4],  # CNN kernel size
#                 "stride_size": [4, 2],  # CNN stride size
#                 "normalize": False,  # Normalize image from range [0,255] to [0,1]
#             },
#             "lr": 1e-2,
#         }

#         field_names = ["state", "action", "reward", "next_state", "done"]
#         memory = ReplayBuffer(
#             memory_size=1000,  # Max replay buffer size
#             field_names=field_names,  # Field names to store in memory
#             device=device,
#         )

#         check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)

#     cont_vector_envs = [
#         (ConstantRewardContActionsEnv(), 1000),
#         (ObsDependentRewardContActionsEnv(), 1000),
#         (DiscountedRewardContActionsEnv(), 5000),
#         (FixedObsPolicyContActionsEnv(), 3000),
#         (PolicyContActionsEnv(), 3000),
#     ]

#     for env, learn_steps in cont_vector_envs:
#         algo_args = {
#             "state_dim": (env.observation_space.n,),
#             "action_dim": env.action_space.shape[0],
#             "one_hot": True if env.observation_space.n > 1 else False,
#             "max_action": 1.0,
#             "min_action": 0.0,
#             "lr_actor": 1e-2,
#             "lr_critic": 1e-2,
#         }

#         field_names = ["state", "action", "reward", "next_state", "done"]
#         memory = ReplayBuffer(
#             memory_size=1000,  # Max replay buffer size
#             field_names=field_names,  # Field names to store in memory
#             device=device,
#         )

#         check_policy_q_learning_with_probe_env(
#             env, DDPG, algo_args, memory, learn_steps, device
#         )

#     image_envs = [
#         (ConstantRewardContActionsImageEnv(), 1000),
#         (ObsDependentRewardContActionsImageEnv(), 3000),
#         (DiscountedRewardContActionsImageEnv(), 7000),
#         (FixedObsPolicyContActionsImageEnv(), 3000),
#         (PolicyContActionsImageEnvSimple(), 4000),
#         (PolicyContActionsImageEnv(), 5000),
#     ]

#     for env, learn_steps in image_envs:
#         algo_args = {
#             "state_dim": (env.observation_space.shape),
#             "action_dim": env.action_space.shape[0],
#             "one_hot": False,
#             "net_config": {
#                 "arch": "cnn",  # Network architecture
#                 "hidden_size": [64],  # Network hidden size
#                 "channel_size": [32, 32],  # CNN channel size
#                 "kernel_size": [8, 4],  # CNN kernel size
#                 "stride_size": [4, 2],  # CNN stride size
#                 "normalize": False,  # Normalize image from range [0,255] to [0,1]
#             },
#             "max_action": 1.0,
#             "min_action": 0.0,
#             "policy_freq": 2,
#         }

#         field_names = ["state", "action", "reward", "next_state", "done"]
#         memory = ReplayBuffer(
#             memory_size=1000,  # Max replay buffer size
#             field_names=field_names,  # Field names to store in memory
#             device=device,
#         )

#         check_policy_q_learning_with_probe_env(
#             env, DDPG, algo_args, memory, learn_steps, device
#         )
