import numpy as np
import pytest

from agilerl.algorithms.ppo import PPO
from agilerl.rollouts.on_policy import (
    _collect_rollouts,
    collect_rollouts,
    collect_rollouts_recurrent,
)


class DummyEnv:
    def __init__(self, state_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.vect = vect
        self.num_envs = num_envs
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.n_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


def test_collect_rollouts_use_rollout_buffer_false_raises(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=False,
        learn_step=5,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    with pytest.raises(RuntimeError, match="use_rollout_buffer=True"):
        collect_rollouts(ppo, env, n_steps=5)
    ppo.clean_up()


def test_collect_rollouts_returns_scores(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    result = collect_rollouts(ppo, env, n_steps=4)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert isinstance(result[0], list)
    ppo.clean_up()


def test_collect_rollouts_recurrent_returns_scores(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
        recurrent=True,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    result = collect_rollouts_recurrent(ppo, env, n_steps=4)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert isinstance(result[0], list)
    ppo.clean_up()


def test_collect_rollouts_warm_start_with_last_obs(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    obs, info = env.reset()
    result = _collect_rollouts(
        ppo,
        env,
        n_steps=2,
        last_obs=obs,
        last_done=np.zeros(1),
        last_scores=np.zeros(1),
        last_info=info,
        recurrent=False,
    )
    assert isinstance(result, tuple)
    assert len(result) == 5
    ppo.clean_up()


def test_collect_rollouts_n_steps_default(vector_space, discrete_space):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=8,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    result = collect_rollouts(ppo, env)
    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    ppo.clean_up()


@pytest.mark.parametrize("recurrent", [False, True])
def test_collect_rollouts_recurrent_warm_start(vector_space, discrete_space, recurrent):
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        use_rollout_buffer=True,
        learn_step=4,
        num_envs=1,
        recurrent=recurrent,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    obs, info = env.reset()
    if recurrent:
        ppo.hidden_state = ppo.get_initial_hidden_state(1)
    result = _collect_rollouts(
        ppo,
        env,
        n_steps=2,
        last_obs=obs,
        last_done=np.zeros(1),
        last_scores=np.zeros(1),
        last_info=info,
        recurrent=recurrent,
    )
    assert isinstance(result, tuple)
    assert len(result) == 5
    ppo.clean_up()
