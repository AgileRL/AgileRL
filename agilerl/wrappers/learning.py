import random
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class Skill(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """The Skill class, used in curriculum learning to teach agents skills. This class works as a
    wrapper around an environment that alters the reward to encourage learning of a particular skill.

    :param env: Environment to learn in
    :type env: Gymnasium-style environment
    """

    def __init__(self, env: gym.Env) -> None:
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the environment and return the observation, reward, terminated, truncated, and info.

        :param action: Action
        :type action: Any
        :return: Tuple of (observation, reward, terminated, truncated, info)
        :rtype: tuple[Any, float, bool, bool, dict[str, Any]]
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.skill_reward(observation, reward, terminated, truncated, info)

    def skill_reward(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Calculate the reward for the given observation, reward, terminated, truncated, and info.

        :param observation: Observation
        :type observation: Any
        :param reward: Reward
        :type reward: float
        :param terminated: Terminated
        :type terminated: bool
        :param truncated: Truncated
        :type truncated: bool
        :param info: Info
        :type info: dict[str, Any]
        :return: Tuple of (observation, reward, terminated, truncated, info)
        :rtype: tuple[Any, float, bool, bool, dict[str, Any]]
        """
        return observation, reward, terminated, truncated, info


class BanditEnv:
    """The Bandit learning environment class. Turns a labelled dataset into a reinforcement learning,
    Gym-style environment.

    :param features: Dataset features
    :type features: pd.DataFrame
    :param targets: Dataset targets corresponding to features
    :type targets: pd.DataFrame
    """

    def __init__(self, features: pd.DataFrame, targets: pd.DataFrame) -> None:
        # Define the number of arms and the context dimension
        self.arms = int(targets.nunique().iloc[0])
        self.context_dim = (len(np.array(features.loc[0])) * self.arms,)

        self.features = features
        self.targets = pd.factorize(targets.values.ravel())[0]
        self.prev_reward = np.zeros(self.arms, dtype=np.float32)
        self.num_envs = 1  # follow vec-env interface

        # Define the observation and action spaces
        self.single_observation_space = spaces.Box(
            low=features.values.min(),
            high=features.values.max(),
            shape=self.context_dim,
            dtype=np.float32,
        )
        self.single_action_space = spaces.Discrete(self.arms)

    def _new_state_and_target_action(self) -> tuple[np.ndarray, int]:
        """Generate a new state and target action.

        :return: Tuple of (state, target)
        :rtype: tuple[np.ndarray, int]
        """
        # Randomly select next context
        r = random.randint(0, len(self.features) - 1)

        # Create contextual input to bandit and corresponding target
        context = np.array(self.features.loc[r], dtype=np.float32)
        target = self.targets[r]
        next_state = np.zeros((self.arms, *self.context_dim), dtype=np.float32)
        for i, j in zip(
            range(self.arms), range(0, self.context_dim[0], len(context)), strict=False
        ):
            next_state[i, j : j + len(context)] = context

        return next_state, target

    def step(self, k: int) -> tuple[np.ndarray, float]:
        """Step the environment and return the state and reward.

        :param k: Action
        :type k: int
        :return: Tuple of (state, reward)
        :rtype: tuple[np.ndarray, float]
        """
        # Calculate reward from action in previous state
        reward = self.prev_reward[k]

        # Now decide on next state
        next_state, target = self._new_state_and_target_action()

        # Save reward for next call to step()
        next_reward = np.zeros(self.arms, dtype=np.float32)
        next_reward[target] = 1
        self.prev_reward = next_reward

        return next_state, float(reward)

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial state.

        :return: Initial state
        :rtype: np.ndarray
        """
        next_state, target = self._new_state_and_target_action()
        next_reward = np.zeros(self.arms, dtype=np.float32)
        next_reward[target] = 1
        self.prev_reward = next_reward
        return next_state
