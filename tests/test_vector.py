"""Tests for pz_async_vec_env.py and pz_vec_env.py"""

import multiprocessing as mp
import operator
import os
import signal
import time
from itertools import accumulate
from multiprocessing import Process
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector.utils import CloudpickleWrapper
from pettingzoo.mpe import simple_speaker_listener_v4
from pettingzoo.sisl import pursuit_v4

from agilerl.vector.pz_async_vec_env import (
    AsyncPettingZooVecEnv,
    AsyncState,
    Observations,
    PettingZooExperienceSpec,
    SharedMemory,
    _async_worker,
)
from agilerl.vector.pz_vec_env import PettingZooVecEnv
from tests.pz_vector_test_utils import CustomSpace, GenericTestEnv, term_env


class DummyRecv:
    def __init__(self, cmd, data):
        self.call_count = 0
        self.cmd = cmd
        self.data = data

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > 1:
            return "close", None
        else:
            return self.cmd, self.data


def actions_to_list_helper(actions):
    passed_actions_list = [[] for _ in list(actions.values())[0]]
    for env_idx, _ in enumerate(list(actions.values())[0]):
        for possible_agent in actions.keys():
            passed_actions_list[env_idx].append(actions[possible_agent][env_idx])

    return passed_actions_list


@pytest.fixture
def pz_experience_spec(env_fns):
    env = env_fns[0]()
    return PettingZooExperienceSpec(env, 8)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(8)]]
)
def test_create_async_pz_vector_env(pz_experience_spec, use_exp_handler, env_fns):
    if use_exp_handler:
        exp_handler = pz_experience_spec
    else:
        exp_handler = None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=exp_handler)
    if exp_handler is not None:
        assert pz_experience_spec.single_action_space
        assert pz_experience_spec.action_space
        assert pz_experience_spec.single_observation_space
        assert pz_experience_spec.observation_space
        assert pz_experience_spec.observation_widths
        assert pz_experience_spec.observation_boundaries
        assert pz_experience_spec.observation_shapes
    assert env.num_envs == 8
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(8)]]
)
def test_reset_async_pz_vector_env(seed, pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    agents = env.possible_agents[:]
    observations, infos = env.reset(seed=seed)
    env.close()
    for agent in agents:
        assert isinstance(env.observation_space(agent), Box)
        assert isinstance(observations[agent], np.ndarray)
        assert observations[agent].dtype == env.observation_space(agent).dtype
        assert (
            observations[agent].shape
            == (8,) + env.single_observation_space(agent).shape
        )
        assert observations[agent].shape == env.observation_space(agent).shape
    assert isinstance(infos, dict)
    assert set(agents).issubset(set(infos.keys()))

    try:
        env_fns = [simple_speaker_listener_v4.parallel_env for _ in range(8)]
        env = AsyncPettingZooVecEnv(env_fns, experience_spec)
    finally:
        env.close()
    for agent in agents:
        assert isinstance(env.observation_space(agent), Box)
        assert isinstance(observations[agent], np.ndarray)
        assert observations[agent].dtype == env.observation_space(agent).dtype
        assert (
            observations[agent].shape
            == (8,) + env.single_observation_space(agent).shape
        )
        assert observations[agent].shape == env.observation_space(agent).shape
        assert set(agents).issubset(set(infos.keys()))


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(render_mode="rgb_array")
            for _ in range(8)
        ]
    ],
)
def test_render_async_pz_vector_env(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    assert env.render_mode == "rgb_array"

    env.reset()
    rendered_frames = env.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == env.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    env.close()


@pytest.mark.parametrize("use_exp_handler", [True])
@pytest.mark.parametrize("use_single_action_space", [False, True])
@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(8)
        ]
    ],
)
def test_step_async_pz_vector_env(
    use_single_action_space, use_exp_handler, pz_experience_spec, env_fns
):
    try:
        env_fns = [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(8)
        ]
        experience_spec = pz_experience_spec if use_exp_handler else None
        env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
        env.reset()
        if use_single_action_space:
            actions = {
                agent: [env.single_action_space(agent).sample() for _ in range(8)]
                for agent in env.agents
            }
        else:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(actions)
        for agent in env.agents:
            assert isinstance(env.single_action_space(agent), Discrete)
            assert isinstance(env.action_space(agent), MultiDiscrete)
            assert isinstance(env.observation_space(agent), Box)
            assert isinstance(observations[agent], np.ndarray)
            assert observations[agent].dtype == env.observation_space(agent).dtype
            assert (
                observations[agent].shape
                == (8,) + env.single_observation_space(agent).shape
            )
            assert observations[agent].shape == env.observation_space(agent).shape
            assert isinstance(rewards[agent], np.ndarray)
            assert isinstance(rewards[agent][0], (float, np.floating))
            assert rewards[agent].ndim == 1
            assert rewards[agent].size == 8
            assert isinstance(terminations[agent], np.ndarray)
            assert terminations[agent].dtype == np.bool_
            assert terminations[agent].ndim == 1
            assert terminations[agent].size == 8
            assert isinstance(truncations[agent], np.ndarray)
            assert truncations[agent].dtype == np.bool_
            assert truncations[agent].ndim == 1
            assert truncations[agent].size == 8
        env.close()

    except Exception as e:
        env.close()
        raise e


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(
                render_mode="rgb_array", continuous_actions=False
            )
            for _ in range(4)
        ]
    ],
)
def test_call_async_pz_vector_env(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()

    images = env.call("render")
    max_num_agents = env.call("max_num_agents")
    env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert images[i].shape[-1] == 3
        assert isinstance(images[i][0], np.ndarray)

    assert isinstance(max_num_agents, tuple)
    assert len(max_num_agents) == 4
    for i in range(4):
        assert isinstance(max_num_agents[i], int)
        assert max_num_agents[i] == 2


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(2)
        ]
    ],
)
def test_get_attr_async_pz_vector_env(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    env.set_attr("test_attribute", [1, 2])
    test_attribute = env.get_attr("test_attribute")
    assert test_attribute == (1, 2)
    env.close()


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(1)
        ]
    ],
)
def test_set_attr_make_values_list(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)

    env.set_attr(name="test", values=1)
    env.close()


def test_env_order_preserved():
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
        for _ in range(16)
    ]
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()
    for obs_view in env.observations.obs_view:
        obs_view[:] = 0
    actions = [0, 1]
    rand_env = np.random.randint(0, 16)
    env.parent_pipes[rand_env].send(("step", actions))
    env.parent_pipes[rand_env].recv()
    for idx, agent in enumerate(env.agents):
        assert not np.array_equal(
            env.observations.obs_view[idx][rand_env, :],
            np.zeros_like(env.experience_spec.observation_shapes[agent]),
        )
    env.close()


def raise_error_reset(self, seed=None, options=None):
    if seed == 1:
        raise ValueError("Error in reset")
    return {
        agent: self.observation_space(agent).sample() for agent in self.possible_agents
    }, {agent: {} for agent in self.possible_agents}


def raise_error_step(self, action):
    if list(action.values())[0] >= 1:
        raise ValueError(f"Error in step with {action}")

    def pz_dict(transition, agents):
        return {agent: transition for agent in self.possible_agents}

    return (
        {
            agent: self.observation_space(agent).sample()
            for agent in self.possible_agents
        },
        pz_dict(0, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict({}, self.possible_agents),
    )


@pytest.mark.parametrize("use_exp_handler", [True, False])
def test_async_vector_subenv_error(use_exp_handler):
    env_list = [
        lambda: GenericTestEnv(reset_func=raise_error_reset, step_func=raise_error_step)
    ]
    exp_handler = (
        PettingZooExperienceSpec(env_list[0](), 2) if use_exp_handler else None
    )
    envs = AsyncPettingZooVecEnv(env_list * 2, exp_handler)

    with pytest.raises(ValueError, match="Error in reset"):
        envs.reset(seed=[1, 0])

    envs.close()
    del exp_handler
    env_list = [
        lambda: GenericTestEnv(reset_func=raise_error_reset, step_func=raise_error_step)
    ]
    exp_handler = (
        PettingZooExperienceSpec(env_list[0](), 3) if use_exp_handler else None
    )
    envs = AsyncPettingZooVecEnv(env_list * 3, exp_handler)

    with pytest.raises(ValueError, match="Error in step"):
        envs.step({"agent_0": [0, 1, 2]})

    envs.close()


@pytest.mark.parametrize("use_exp_handler", [True, False])
def test_custom_space_error(use_exp_handler):
    num_envs = 4
    env_fns = [
        lambda: GenericTestEnv(
            action_space=CustomSpace(), observation_space=CustomSpace()
        )
    ] * num_envs
    with pytest.raises(ValueError):
        exp_handler = (
            PettingZooExperienceSpec(env_fns[0](), num_envs)
            if use_exp_handler
            else None
        )
        AsyncPettingZooVecEnv(env_fns, exp_handler)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_reset_async_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.reset_async()

    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_reset_wait_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    with pytest.raises(NoAsyncCallError):
        env.reset_async()
        env._state = AsyncState.DEFAULT
        env.reset_wait()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_step_async_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.step_async(actions=None)
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_step_wait_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.step_wait()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_async_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.call_async("test")
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_wait_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.call_wait()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_exception_worker(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    with pytest.raises(ValueError):
        env.call("reset")
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_set_attr_val_error(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    with pytest.raises(ValueError):
        env.set_attr("test", values=[1, 2, 3])
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_set_attr_exception(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.set_attr("test", values=[1, 2])
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_close_extras_warning(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    with patch.object(gym.logger, "warn") as mock_logger_warn:
        env.close_extras(timeout=None)
        mock_logger_warn.assert_called_once
    env.close()


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_close_extras_terminate(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    env.close_extras(terminate=True)

    for p in env.processes:
        assert not p.is_alive()


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_poll_pipe_envs(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env.parent_pipes[0] = None
    result = env._poll_pipe_envs(timeout=1)
    assert not result
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_assert_is_running(pz_experience_spec, use_exp_handler, env_fns):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env.closed = True
    with pytest.raises(ClosedEnvironmentError):
        env._assert_is_running()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_step_wait_timeout_async_pz_vector_env(
    pz_experience_spec, use_exp_handler, env_fns
):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_STEP
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.step_wait(timeout=1)
        env.close()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize("use_exp_handler", [True, False])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_wait_timeout_async_pz_vector_env(
    pz_experience_spec, use_exp_handler, env_fns
):
    experience_spec = pz_experience_spec if use_exp_handler else None
    env = AsyncPettingZooVecEnv(env_fns, experience_spec=experience_spec)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.call_wait(timeout=1)
        env.close()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


@pytest.mark.parametrize(
    "transition_name", ["reward", "truncation", "termination", "info", "observation"]
)
def test_get_placeholder_value(transition_name):
    env_fns = [lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]
    env = AsyncPettingZooVecEnv(env_fns)
    if transition_name != "observation":
        val = env.experience_spec.get_placeholder_value("agent", transition_name)
        if transition_name == "reward":
            assert val == 0
        if transition_name == "truncated" or transition_name == "terminated":
            assert val
        if transition_name == "info":
            assert val == {}
    else:
        env.reset()
        output = env.experience_spec.get_placeholder_value(
            agent="speaker_0",
            transition_name=transition_name,
            observations=env.observations,
        )
        assert isinstance(output, np.ndarray)
    env.close()


def test_add_info_dictionaries():
    info_list = [
        {
            "agent_0": {
                "env_defined_actions": np.array([1, 2, 3]),
                "action_mask": np.array([1, 0]),
            },
            "agent_1": {},
        },
        {
            "agent_0": {},
            "agent_1": {
                "env_defined_actions": np.array([5, 6, 7]),
                "action_mask": np.array([0, 1]),
            },
        },
        {"agent_0": {}, "agent_1": {}},
    ]
    env_fns = [lambda: simple_speaker_listener_v4.parallel_env() for _ in range(3)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    assert np.all(
        vector_infos["agent_0"]["env_defined_actions"]
        == np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
    )
    assert np.all(
        vector_infos["agent_0"]["action_mask"] == np.array([[1, 0], [0, 0], [0, 0]])
    )
    assert np.all(
        vector_infos["agent_1"]["env_defined_actions"]
        == np.array([[0, 0, 0], [5, 6, 7], [0, 0, 0]])
    )
    assert np.all(
        vector_infos["agent_1"]["action_mask"] == np.array([[0, 0], [0, 1], [0, 0]])
    )
    env.close()


def test_add_info_int():
    info_list = [
        {"agent_0": 1.0},
        {"agent_1": 1},
    ]
    env_fns = [lambda: GenericTestEnv() for _ in range(3)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {"agent_0": {}}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    env.close()


def test_add_info_unknown_objects():
    info_list = [
        {"agent_0": "string"},
        {"agent_1": "string"},
        {"agent_2": None},
    ]
    env_fns = [lambda: GenericTestEnv() for _ in range(3)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {"agent_0": {}}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)
    env.close()


def test_create_experience_handler():
    num_envs = 8
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env() for _ in range(num_envs)
    ]
    env = simple_speaker_listener_v4.parallel_env()
    experience_spec = PettingZooExperienceSpec(env_fns[0](), num_envs)

    assert experience_spec.single_action_space == {
        agent: env.action_space(agent) for agent in env.possible_agents
    }
    assert experience_spec.single_observation_space == {
        agent: env.observation_space(agent) for agent in env.possible_agents
    }
    assert experience_spec.observation_widths == {
        agent: int(np.prod(obs.shape))
        for agent, obs in experience_spec.single_observation_space.items()
    }
    assert experience_spec.observation_boundaries == [0] + list(
        accumulate(experience_spec.observation_widths.values(), operator.add)
    )
    assert experience_spec.total_observation_width == int(
        np.sum(list(experience_spec.observation_widths.values()))
    )


def test_worker_reset():
    env_fn = [lambda: simple_speaker_listener_v4.parallel_env() for _ in range(1)][0]
    env = env_fn()
    env.reset()
    exp_handler = PettingZooExperienceSpec(env_fn(), 3)
    shared_memory = SharedMemory(3, exp_handler, mp)
    observations = Observations(shared_memory.shared_memory, exp_handler, 3)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn),
            child_pipe,
            parent_pipe,
            observations,
            queue,
            exp_handler,
        ),
    )
    p.start()
    child_pipe.close()
    parent_pipe.send(("reset", {}))
    results, success = parent_pipe.recv()
    assert success
    assert len(results) == 2
    assert list(sorted(results.keys())) == sorted(env.aec_env.agents)
    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_step_simple():
    num_envs = 1
    env_fn = [
        lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
        for _ in range(num_envs)
    ]

    vec_env = AsyncPettingZooVecEnv(env_fn)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    exp_handler = PettingZooExperienceSpec(env_fn[0](), num_envs)
    shared_memory = SharedMemory(num_envs, exp_handler, mp)
    observations = Observations(shared_memory.shared_memory, exp_handler, num_envs)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn[0]),
            child_pipe,
            parent_pipe,
            observations,
            queue,
            exp_handler,
        ),
    )
    p.start()
    child_pipe.close()
    # Reset the environment before stepping
    parent_pipe.send(("reset", {}))
    parent_pipe.recv()
    time.sleep(1)

    parent_pipe.send(("step", actions[0]))
    results, success = parent_pipe.recv()
    assert success
    for dic in results:
        assert list(sorted(dic.keys())) == sorted(vec_env.agents)

    rewards, term, trunc, _ = results

    # state check
    assert observations["speaker_0"].shape == (num_envs,) + (3,)
    assert observations["listener_0"].shape == (num_envs,) + (11,)
    assert isinstance(observations["speaker_0"], np.ndarray)
    assert isinstance(observations["listener_0"], np.ndarray)

    # rewards check
    assert isinstance(rewards["speaker_0"], float)
    assert isinstance(rewards["listener_0"], float)

    # term check
    assert isinstance(term["speaker_0"], bool)
    assert isinstance(term["listener_0"], bool)

    # trunc check
    assert isinstance(trunc["speaker_0"], bool)
    assert isinstance(trunc["listener_0"], bool)

    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_step_autoreset():
    num_envs = 1
    env_fn = [lambda: term_env() for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fn)
    vec_env.reset()
    actions = {
        agent: np.array([vec_env.single_action_space(agent).sample()])
        for agent in vec_env.agents
    }
    vec_env.close()
    actions = actions_to_list_helper(actions)
    exp_handler = PettingZooExperienceSpec(env_fn[0](), num_envs)
    shared_memory = SharedMemory(num_envs, exp_handler, mp)
    observations = Observations(shared_memory.shared_memory, exp_handler, num_envs)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn[0]),
            child_pipe,
            parent_pipe,
            observations,
            queue,
            exp_handler,
        ),
    )
    p.start()
    child_pipe.close()
    # Reset the environment before stepping
    parent_pipe.send(("reset", {}))
    parent_pipe.recv()
    time.sleep(1)

    # print("ACTIONS", actions[0].dtype)
    parent_pipe.send(("step", actions[0]))
    results, success = parent_pipe.recv()
    assert success
    for dic in results:
        assert list(sorted(dic.keys())) == sorted(vec_env.agents)

    # Send step again and autoreset should be True
    parent_pipe.send(("step", actions[0]))
    parent_pipe.recv()
    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_runtime_error():
    num_envs = 1
    env_fn = [
        lambda: simple_speaker_listener_v4.parallel_env() for _ in range(num_envs)
    ][0]
    env = env_fn()
    env.reset()
    exp_handler = PettingZooExperienceSpec(env_fn(), num_envs)
    shared_memory = SharedMemory(num_envs, exp_handler, mp)
    observations = Observations(shared_memory.shared_memory, exp_handler, num_envs)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn),
            child_pipe,
            parent_pipe,
            observations,
            queue,
            exp_handler,
        ),
    )
    p.start()
    child_pipe.close()
    parent_pipe.send(("Unknown", {}))
    _, success = parent_pipe.recv()
    assert not success
    _, exctype, value, _ = queue.get()
    assert exctype.__name__ == "RuntimeError"
    assert isinstance(value, RuntimeError)
    assert (
        str(value)
        == "Received unknown command `Unknown`. Must be one of [`reset`, `step`, `close`, `_call`, `_setattr`, `_check_spaces`]."
    )
    os.kill(p.pid, signal.SIGTERM)


def test_observations_vector():
    num_envs = 1
    agents = ["speaker_0", "listener_0"]
    env_fn = [
        lambda: simple_speaker_listener_v4.parallel_env() for _ in range(num_envs)
    ][0]
    exp_handler = PettingZooExperienceSpec(env_fn(), num_envs)
    shared_memory = SharedMemory(num_envs, exp_handler, mp)
    observations = Observations(
        shared_memory.shared_memory, exp_spec=exp_handler, num_envs=num_envs
    )

    assert (
        observations.__str__()
        == observations.__repr__()
        == "{'speaker_0': array([[0., 0., 0.]], dtype=float32), 'listener_0': array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)}"
    )
    ob = {"speaker_0": np.ones((1, 3)), "listener_0": np.ones((1, 11))}
    observations.set_env_obs(0, ob)
    # assert np.all(observations.get_env_obs(0) == np.ones((1, 14), dtype=np.float32))
    assert "speaker_0" in observations
    assert len(observations) == 2
    keys = []
    vals = []
    for key, val in observations.items():
        keys.append(key)
        vals.append(val)
    assert keys == ["speaker_0", "listener_0"]
    assert np.all(np.concatenate(vals, axis=1) == np.ones((1, 14), dtype=np.float32))
    assert list(observations.keys()) == agents
    assert np.all(
        np.concatenate(list(observations.values()), axis=1)
        == np.ones((1, 14), dtype=np.float32)
    )
    assert np.all(observations.get("speaker_0") == np.ones((1, 3), dtype=np.float32))
    assert observations.get("agent") is None
    assert (
        str(next(iter(observations)))
        == "('speaker_0', array([[1., 1., 1.]], dtype=float32))"
    )


def test_observations_image():
    num_envs = 1
    env_fn = [lambda: pursuit_v4.parallel_env() for _ in range(num_envs)][0]
    exp_handler = PettingZooExperienceSpec(env_fn(), num_envs)
    shared_memory = SharedMemory(num_envs, exp_handler, mp)
    observations = Observations(
        shared_memory.shared_memory, exp_spec=exp_handler, num_envs=num_envs
    )

    for agent in exp_handler.agents:
        assert isinstance(observations[agent], np.ndarray)
        assert observations[agent].shape == (1, 7, 7, 3)

    # assert observations.get_agent_obs("pursuer_0", flat=True).shape == (
    #     1,
    #     exp_handler.observation_widths["pursuer_0"],
    # )
    # assert (
    #     observations.get_agent_obs("pursuer_0", flat=False).shape
    #     == (1,) + exp_handler.observation_shapes["pursuer_0"]
    # )


def test_observations_states():
    num_envs = 1
    env_fn = [lambda: pursuit_v4.parallel_env() for _ in range(num_envs)][0]
    exp_handler = PettingZooExperienceSpec(env_fn(), num_envs)
    shared_memory = SharedMemory(num_envs, exp_handler, mp)
    observations = Observations(
        shared_memory.shared_memory, exp_spec=exp_handler, num_envs=num_envs
    )
    state = observations.__getstate__()
    assert state["obs_view"] is None
    observations.__setstate__(state)
    assert isinstance(observations.obs_view, list)
    for ob in observations.obs_view:
        assert isinstance(ob, np.ndarray)


# Test for pz_vec_env.py
def test_vec_env_reset():
    vec_env = PettingZooVecEnv(3, ["agent_0"])
    vec_env.reset()


def test_vec_env_step():
    vec_env = PettingZooVecEnv(3, ["agent_0"])
    vec_env.step_async([])
    vec_env.step_wait()


def test_vec_env_render():
    vec_env = PettingZooVecEnv(3, ["agent_0"])
    with pytest.raises(NotImplementedError):
        vec_env.render()


def test_vec_env_closed():
    vec_env = PettingZooVecEnv(3, ["agent_0"])
    vec_env.closed = True
    vec_env.close()


def test_vec_env_close_extras():
    vec_env = PettingZooVecEnv(3, ["agent_0"])
    vec_env.close_extras()


def test_vec_env_unwrapped():
    vec_env = PettingZooVecEnv(3, ["agent_0"])
    vec_env.unwrapped


def test_delete_async_pz_vec_env():
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
        for _ in range(2)
    ]
    env = AsyncPettingZooVecEnv(env_fns)
    assert len(env.processes) > 0  # Ensure subprocesses were created
    for process in env.processes:
        assert process.is_alive()
    processes = env.processes
    env.__del__()
    for p in processes:
        assert not p.is_alive()
