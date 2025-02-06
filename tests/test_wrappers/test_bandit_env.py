import random

import numpy as np
import pandas as pd

from agilerl.wrappers.learning import BanditEnv


# BanditEnv class creates an environment from a dataset
def test_create_environment():
    features = pd.DataFrame(np.random.uniform(0, 1, size=(10, 1)), columns=["features"])
    targets = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)), columns=["targets"])

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_space = env.arms

    assert isinstance(env, BanditEnv)
    assert isinstance(action_space, int)
    assert isinstance(context_dim, tuple)
    assert action_space == int(targets.nunique()[0])
    assert context_dim == (len(np.array(features.loc[0])) * int(targets.nunique()),)


# BanditEnv class returns the reward output
def test_return_state_reward():
    features = pd.DataFrame(np.random.uniform(0, 1, size=(10, 1)), columns=["features"])
    targets = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)), columns=["targets"])

    env = BanditEnv(features, targets)  # Create environment
    action_space = env.arms

    state = env.reset()
    action = random.randint(0, action_space - 1)
    prev_reward = env.prev_reward
    new_state, reward = env.step(action)

    assert state.shape == new_state.shape
    assert isinstance(reward, (int, float))
    assert reward == prev_reward[action]
