import random

import gymnasium as gym
import numpy as np
import pandas as pd


class Skill(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """The Skill class, used in curriculum learning to teach agents skills. This class works as a
    wrapper around an environment that alters the reward to encourage learning of a particular skill.

    :param env: Environment to learn in
    :type env: Gymnasium-style environment
    """

    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Use custom reward
        return self.skill_reward(observation, reward, terminated, truncated, info)

    def skill_reward(self, observation, reward, terminated, truncated, info):
        return observation, reward, terminated, truncated, info


class BanditEnv:
    """The Bandit learning environment class. Turns a labelled dataset into a reinforcement learning,
    Gym-style environment.

    :param features: Dataset features
    :type features: Pandas DataFrame
    :param targets: Dataset targets corresponding to features
    :type features: Pandas DataFrame
    """

    def __init__(self, features, targets):
        self.arms = int(targets.nunique()[0])
        self.context_dim = (len(np.array(features.loc[0])) * self.arms,)

        self.features = features
        self.targets = pd.factorize(targets.values.ravel())[0]
        self.prev_reward = np.zeros(self.arms)

    def _new_state_and_target_action(self):
        # Randomly select next context
        r = random.randint(0, len(self.features) - 1)

        # Create contextual input to bandit and corresponding target
        context = np.array(self.features.loc[r])
        target = self.targets[r]
        next_state = np.zeros((self.arms, *self.context_dim))
        for i, j in zip(range(self.arms), range(0, self.context_dim[0], len(context))):
            next_state[i, j : j + len(context)] = context
        return next_state, target

    def step(self, k):
        # Calculate reward from action in previous state
        reward = self.prev_reward[k]

        # Now decide on next state
        next_state, target = self._new_state_and_target_action()

        # Save reward for next call to step()
        next_reward = np.zeros(self.arms)
        next_reward[target] = 1
        self.prev_reward = next_reward
        return next_state, reward

    def reset(self):
        next_state, target = self._new_state_and_target_action()
        next_reward = np.zeros(self.arms)
        next_reward[target] = 1
        self.prev_reward = next_reward
        return next_state
