import random

import numpy as np
from ucimlrepo import fetch_ucirepo

from agilerl.wrappers.learning import BanditEnv


# BanditEnv class creates an environment from a dataset
def test_create_environment():
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    assert isinstance(env, BanditEnv)
    assert isinstance(action_dim, int)
    assert isinstance(context_dim, tuple)
    assert action_dim == int(targets.nunique())
    assert context_dim == (len(np.array(features.loc[0])) * int(targets.nunique()),)


# BanditEnv class returns the reward output
def test_return_state_reward():
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    action_dim = env.arms

    state = env.reset()
    action = random.randint(0, action_dim - 1)
    prev_reward = env.prev_reward
    new_state, reward = env.step(action)

    assert state.shape == new_state.shape
    assert isinstance(reward, (int, float))
    assert reward == prev_reward[action]
