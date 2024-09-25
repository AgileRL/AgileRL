import numpy as np
import pettingzoo
import pytest

from agilerl.vector.pz_async_vec_env import AsyncVectorPZEnv
from tests.test_vectorization import parallel_env_disc

# @pytest.fixture
# def make_env_lambda(env):
#     return lambda: parallel_env_cont()

# def test_vectors():
#     a = AsyncVectorPZEnv([lambda:simple_speaker_listener_v4.parallel_env()])

#     print(a.obs_boundaries)
#     print(a.obs_space_width)

#     assert False


def create_env():
    def make_env():
        # env = parallel_env_disc()
        env = pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
            continuous_actions=True
        )
        return env

    return make_env


@pytest.mark.parametrize("env", [(parallel_env_disc)])
def test_reset(env):
    num_envs = 25
    single_env = pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
        continuous_actions=True
    )
    vec_env = AsyncVectorPZEnv([create_env() for _ in range(num_envs)])
    obs, info = vec_env.reset(seed=12)

    # Ensure num_envs observations are returned
    for agent, ob in obs.items():
        assert ob.shape[0] == num_envs
        assert single_env.observation_space(agent).shape[0] == ob.shape[1]
        assert not np.all(ob == ob[0, :])


def test_vector_step():
    num_envs = 2
    # single_env = pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    vec_env = AsyncVectorPZEnv([create_env() for _ in range(num_envs)])
    obs, info = vec_env.reset(seed=12)

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}

    state, reward, term, trunc, info = vec_env.step(actions)
