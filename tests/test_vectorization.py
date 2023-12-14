import numpy as np
import pytest
from pettingzoo.mpe import simple_speaker_listener_v4
from pettingzoo.atari import space_invaders_v2
from unittest.mock import patch
from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper


@pytest.fixture
def petting_zoo_env_cont_actions():
    return simple_speaker_listener_v4.parallel_env(
        max_cycles=25, continuous_actions=True
    )


@pytest.fixture
def petting_zoo_env_disc_actions():
    return simple_speaker_listener_v4.parallel_env(
        max_cycles=25, continuous_actions=False
    )


@pytest.fixture
def atari():
    return space_invaders_v2.parallel_env()


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_vectorisation_wrapper_petting_zoo_reset(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    vec_env.close()
    for agent in vec_env.agents:
        assert len(observations[agent]) == n_envs
        assert len(infos[agent]) == n_envs


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_vectorisation_wrapper_petting_zoo_step(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    for step in range(25):
        actions = {
            agent: [(vec_env.action_space(agent).sample(),) for n in range(n_envs)]
            for agent in vec_env.agents
        }
        print("ACTIONS", actions)
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        for agent in vec_env.agents:
            assert len(observations[agent]) == n_envs
            assert len(rewards[agent]) == n_envs
            assert len(terminations[agent]) == n_envs
            assert len(truncations[agent]) == n_envs
            assert len(infos[agent]) == n_envs
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)


@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
    ],
)
def test_cont_action_observation_spaces(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    vec_env.reset()
    print(env.action_space("speaker_0").shape[0])
    print(vec_env.action_space("speaker_0").shape[0])
    assert [env.action_space(agent).shape[0] for agent in env.agents] == [
        vec_env.action_space(agent).shape[0] for agent in vec_env.agents
    ]
    assert [env.observation_space(agent).shape for agent in env.agents] == [
        vec_env.observation_space(agent).shape for agent in vec_env.agents
    ]


@pytest.mark.parametrize("env", ["petting_zoo_env_disc_actions", "atari"])
def test_disc_action_observation_spaces(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    vec_env.reset()
    assert [env.action_space(agent).n for agent in env.agents] == [
        vec_env.action_space(agent).n for agent in vec_env.agents
    ]
    assert [env.observation_space(agent).shape for agent in env.agents] == [
        vec_env.observation_space(agent).shape for agent in vec_env.agents
    ]


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_basic_attributes(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    assert env.agents == vec_env.agents
    assert env.num_agents == vec_env.num_agents
    assert env.possible_agents == vec_env.possible_agents
    assert env.max_num_agents == vec_env.max_num_agents


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_close(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    vec_env.render()
    with patch("agilerl.wrappers.pettingzoo_wrappers.PettingZooVectorizationParallelWrapper.close") as mock_close:
        vec_env.close()
        mock_close.assert_called()
    pass


if __name__ == "__main__":
    pass
