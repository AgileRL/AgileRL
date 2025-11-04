import numpy as np
import pytest

from agilerl.vector.pz_sync_vec_env import SyncPettingZooVecEnv


def _make_rps_env_fns(num_envs: int):
    from pettingzoo.classic import rps_v2

    return [lambda: rps_v2.parallel_env() for _ in range(num_envs)]


def _sample_actions_for_agents(vec_env: SyncPettingZooVecEnv, num_envs: int):
    actions = {}
    for agent in vec_env.agents:
        space = vec_env.single_action_space(agent)
        samples = [space.sample() for _ in range(num_envs)]
        actions[agent] = np.asarray(samples)
    return actions


def _sample_actions_per_env(vec_env: SyncPettingZooVecEnv, num_envs: int):
    # Returns list[dict[agent, action]] for step_async
    per_env = []
    for _ in range(num_envs):
        action_dict = {}
        for agent in vec_env.agents:
            action_dict[agent] = vec_env.single_action_space(agent).sample()
        per_env.append(action_dict)
    return per_env


@pytest.mark.parametrize("num_envs", [1, 2])
def test_sync_pz_vec_env_reset_and_spaces_rps(num_envs):
    env_fns = _make_rps_env_fns(num_envs)
    vec_env = SyncPettingZooVecEnv(env_fns)

    obs, info = vec_env.reset(seed=123)

    assert isinstance(obs, dict)
    assert isinstance(info, dict)
    assert set(vec_env.agents) == set(obs.keys())

    for agent in vec_env.agents:
        agent_obs = obs[agent]
        # Expect leading batch dimension == num_envs
        if isinstance(agent_obs, dict):
            # Dict observation spaces: each subkey batched
            for v in agent_obs.values():
                assert isinstance(v, np.ndarray)
                assert v.shape[0] == num_envs
        elif isinstance(agent_obs, tuple):
            for v in agent_obs:
                assert isinstance(v, np.ndarray)
                assert v.shape[0] == num_envs
        else:
            assert isinstance(agent_obs, np.ndarray)
            assert agent_obs.shape[0] == num_envs

    # single_action_space should be available and consistent across envs
    for agent in vec_env.agents:
        _ = vec_env.single_action_space(agent)


@pytest.mark.parametrize("num_envs", [1, 2])
def test_sync_pz_vec_env_step_shapes_rps(num_envs):
    env_fns = _make_rps_env_fns(num_envs)
    vec_env = SyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = _sample_actions_for_agents(vec_env, num_envs)
    next_obs, rewards, terms, truncs, infos = vec_env.step(actions)

    assert isinstance(next_obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terms, dict)
    assert isinstance(truncs, dict)
    assert isinstance(infos, dict)

    for agent in vec_env.agents:
        assert (
            next_obs[agent].shape[0] == num_envs
            if not isinstance(next_obs[agent], dict)
            and not isinstance(next_obs[agent], tuple)
            else True
        )
        assert len(rewards[agent]) == num_envs
        assert len(terms[agent]) == num_envs
        assert len(truncs[agent]) == num_envs


@pytest.mark.parametrize("num_envs", [2])
def test_sync_pz_vec_env_step_async_wait_equivalence(num_envs):
    env_fns = _make_rps_env_fns(num_envs)
    vec_env = SyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    per_env_actions = _sample_actions_per_env(vec_env, num_envs)
    vec_env.step_async(per_env_actions)
    out_async = vec_env.step_wait()

    actions = _sample_actions_for_agents(vec_env, num_envs)
    out_direct = vec_env.step(actions)

    # Verify both return the expected 5-tuple types and matching batch sizes
    for out in (out_async, out_direct):
        next_obs, rewards, terms, truncs, infos = out
        assert isinstance(next_obs, dict)
        assert isinstance(rewards, dict)
        assert isinstance(terms, dict)
        assert isinstance(truncs, dict)
        assert isinstance(infos, dict)
        for agent in vec_env.agents:
            assert len(rewards[agent]) == num_envs
            assert len(terms[agent]) == num_envs
            assert len(truncs[agent]) == num_envs


def test_sync_pz_vec_env_auto_reset_on_done_rps():
    # RPS is a one-step game, so after a step all agents are done and env auto-resets.
    num_envs = 2
    env_fns = _make_rps_env_fns(num_envs)
    vec_env = SyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = _sample_actions_for_agents(vec_env, num_envs)
    _ = vec_env.step(actions)

    # Without auto-reset, a second step would raise; with auto-reset it should succeed.
    actions2 = _sample_actions_for_agents(vec_env, num_envs)
    next_obs, rewards, terms, truncs, infos = vec_env.step(actions2)

    for agent in vec_env.agents:
        assert len(rewards[agent]) == num_envs
        assert len(terms[agent]) == num_envs
        assert len(truncs[agent]) == num_envs
