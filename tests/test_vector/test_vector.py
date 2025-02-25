"""Tests for pz_async_vec_env.py and pz_vec_env.py"""

import multiprocessing as mp
import os
import signal
import time
from multiprocessing import Process
from multiprocessing.sharedctypes import SynchronizedArray
from unittest.mock import patch

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pytest
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector.utils import CloudpickleWrapper
from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_speaker_listener_v4
from pettingzoo.sisl import pursuit_v4

from agilerl.vector.pz_async_vec_env import (  # PettingZooExperienceSpec,; SharedMemory,
    AsyncPettingZooVecEnv,
    AsyncState,
    Observations,
    _async_worker,
    get_placeholder_value,
    set_env_obs,
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


# @pytest.fixture
# def pz_experience_spec():
#     return PettingZooExperienceSpec(8)


@pytest.fixture(autouse=True)
def clean_process_fixture():
    """Fixture to ensure processes are cleaned up between tests"""
    # Before each test
    yield
    # After each test - forcibly terminate any stray processes
    for p in mp.active_children():
        p.terminate()
        p.join(timeout=1.0)


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(8)]]
)
def test_create_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.single_action_space
    assert env.action_space
    assert env.single_observation_space
    assert env.observation_space
    assert env.observation_widths
    assert env.observation_boundaries
    assert env.observation_shapes
    assert env.num_envs == 8
    for val in env._obs_buffer.values():
        assert isinstance(val, SynchronizedArray)
    assert isinstance(env.observations, Observations)
    assert env.processes
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(8)]]
)
def test_reset_async_pz_vector_env(seed, env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
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
        env = AsyncPettingZooVecEnv(env_fns)
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


@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(render_mode="rgb_array")
            for _ in range(8)
        ]
    ],
)
def test_render_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.render_mode == "rgb_array"

    env.reset()
    rendered_frames = env.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == env.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    env.close()


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
def test_step_async_pz_vector_env(use_single_action_space, env_fns):
    try:
        env_fns = [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(8)
        ]
        env = AsyncPettingZooVecEnv(env_fns)
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
def test_call_async_pz_vector_env(env_fns):
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


@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(2)
        ]
    ],
)
def test_get_attr_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr("test_attribute", [1, 2])
    test_attribute = env.get_attr("test_attribute")
    assert test_attribute == (1, 2)
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
            for _ in range(1)
        ]
    ],
)
def test_set_attr_make_values_list(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr(name="test", values=1)
    assert env.call("test")[0] == 1
    env.close()


def test_env_order_preserved():
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=False)
        for _ in range(16)
    ]
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()
    for obs_view in env.observations.obs_view.values():
        print("obs view", obs_view)
        obs_view[:] = 0
    actions = [0, 1]
    rand_env = np.random.randint(0, 16)
    env.parent_pipes[rand_env].send(("step", actions))
    env.parent_pipes[rand_env].recv()
    for agent in env.agents:
        assert not np.array_equal(
            env.observations.obs_view[agent][
                rand_env
                * env.observation_widths[agent] : (rand_env + 1)
                * env.observation_widths[agent]
            ],
            np.zeros_like(env.observation_shapes[agent]),
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


def test_async_vector_subenv_error():
    env_list = [
        lambda: GenericTestEnv(reset_func=raise_error_reset, step_func=raise_error_step)
    ]
    envs = AsyncPettingZooVecEnv(env_list * 2)

    with pytest.raises(ValueError, match="Error in reset"):
        envs.reset(seed=[1, 0])

    envs.close()
    env_list = [
        lambda: GenericTestEnv(reset_func=raise_error_reset, step_func=raise_error_step)
    ]
    envs = AsyncPettingZooVecEnv(env_list * 3)

    with pytest.raises(ValueError, match="Error in step"):
        envs.step({"agent_0": [0, 1, 2]})

    envs.close()


def test_custom_space_error():
    num_envs = 4
    env_fns = [
        lambda: GenericTestEnv(
            action_space=CustomSpace(), observation_space=CustomSpace()
        )
    ] * num_envs
    with pytest.raises(ValueError):
        AsyncPettingZooVecEnv(env_fns)


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_reset_async_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.reset_async()
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_reset_wait_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    with pytest.raises(NoAsyncCallError):
        env.reset_async()
        env._state = AsyncState.DEFAULT
        env.reset_wait()
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_step_async_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.step_async(actions=None)
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_step_wait_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.step_wait()
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_async_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.call_async("test")
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_wait_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.call_wait()
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_exception_worker(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    with pytest.raises(ValueError):
        env.call("reset")
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_set_attr_val_error(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    with pytest.raises(ValueError):
        env.set_attr("test", values=[1, 2, 3])
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_set_attr_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.set_attr("test", values=[1, 2])
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_close_extras_warning(env_fns):
    env = AsyncPettingZooVecEnv(
        env_fns,
    )
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    with patch.object(gym.logger, "warn") as mock_logger_warn:
        env.close_extras(timeout=None)
        mock_logger_warn.assert_called_once
    env.close()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_close_extras_terminate(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    env.close_extras(terminate=True)

    for p in env.processes:
        assert not p.is_alive()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_poll_pipe_envs(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.parent_pipes[0] = None
    result = env._poll_pipe_envs(timeout=1)
    assert not result
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_assert_is_running(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.closed = True
    with pytest.raises(ClosedEnvironmentError):
        env._assert_is_running()
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_step_wait_timeout_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_STEP
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.step_wait(timeout=1)
        env.close()
    env.__del__()


@pytest.mark.parametrize(
    "env_fns", [[lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]]
)
def test_call_wait_timeout_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.call_wait(timeout=1)
        env.close()
    env.__del__()


@pytest.mark.parametrize(
    "transition_name", ["reward", "truncated", "terminated", "info", "observation"]
)
def test_get_placeholder_value(transition_name):
    env_fns = [lambda: simple_speaker_listener_v4.parallel_env() for _ in range(2)]
    env = AsyncPettingZooVecEnv(env_fns)
    if transition_name != "observation":
        val = get_placeholder_value("agent", transition_name)
        if transition_name == "reward":
            assert val == 0
        if transition_name == "truncated":
            assert not val
        if transition_name == "terminated":
            assert val
        if transition_name == "info":
            assert val == {}
    else:
        env.reset()
        output = get_placeholder_value(
            agent="speaker_0",
            transition_name=transition_name,
            observation_shapes=env.observations,
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


def test_worker_reset():
    env_fns = [lambda: simple_speaker_listener_v4.parallel_env() for _ in range(1)]
    env_fn = env_fns[0]
    env = env_fn()
    env.reset()
    vec_env = AsyncPettingZooVecEnv(env_fns)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
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
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
        for _ in range(num_envs)
    ]

    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
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
    assert vec_env.observations["speaker_0"].shape == (num_envs,) + (3,)
    assert vec_env.observations["listener_0"].shape == (num_envs,) + (11,)
    assert isinstance(vec_env.observations["speaker_0"], np.ndarray)
    assert isinstance(vec_env.observations["listener_0"], np.ndarray)

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
    env_fns = [lambda: term_env() for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()
    actions = {
        agent: np.array([vec_env.single_action_space(agent).sample()])
        for agent in vec_env.agents
    }
    vec_env.close()
    actions = actions_to_list_helper(actions)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
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

    # Send step again and autoreset should be True
    parent_pipe.send(("step", actions[0]))
    parent_pipe.recv()
    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_runtime_error():
    num_envs = 1
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env() for _ in range(num_envs)
    ]
    vec_env = AsyncPettingZooVecEnv(env_fns)
    env_fn = env_fns[0]
    env = env_fn()
    env.reset()
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    try:
        p = Process(
            target=_async_worker,
            args=(
                0,
                CloudpickleWrapper(env_fn),
                child_pipe,
                parent_pipe,
                vec_env._obs_buffer,
                queue,
                vec_env.observation_shapes,
                vec_env.observation_widths,
                vec_env.observation_dtypes,
                vec_env.agents,
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
    finally:
        # Clean up resources
        os.kill(p.pid, signal.SIGTERM)
        p.join(timeout=1)  # Wait for process to terminate
        if p.is_alive():
            p.kill()  # Force kill if still alive
        p.join()  # Final join

        # Close pipes and queue
        parent_pipe.close()
        queue.close()
        queue.join_thread()


def test_observations_vector():
    num_envs = 1
    agents = ["speaker_0", "listener_0"]
    env_fns = [
        lambda: simple_speaker_listener_v4.parallel_env() for _ in range(num_envs)
    ]
    vec_env = AsyncPettingZooVecEnv(env_fns)
    assert (
        vec_env.observations.__str__()
        == vec_env.observations.__repr__()
        == "{'speaker_0': array([0., 0., 0.], dtype=float32), 'listener_0': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}"
    ), vec_env.observations.__str__()
    ob = {"speaker_0": np.ones((1, 3)), "listener_0": np.ones((1, 11))}
    set_env_obs(
        0,
        ob,
        vec_env._obs_buffer,
        vec_env.observation_widths,
        # vec_env.observation_dtypes,
    )
    assert "speaker_0" in vec_env.observations
    assert len(vec_env.observations) == 2
    keys = []
    vals = []
    for key, val in vec_env.observations.items():
        keys.append(key)
        vals.append(val)
    assert keys == ["speaker_0", "listener_0"]
    assert np.all(np.concatenate(vals, axis=1) == np.ones((1, 14), dtype=np.float32))
    assert list(vec_env.observations.keys()) == agents
    assert np.all(
        np.concatenate(list(vec_env.observations.values()), axis=1)
        == np.ones((1, 14), dtype=np.float32)
    )
    assert np.all(
        vec_env.observations.get("speaker_0") == np.ones((1, 3), dtype=np.float32)
    )
    assert vec_env.observations.get("agent") is None
    assert (
        str(next(iter(vec_env.observations)))
        == "('speaker_0', array([[1., 1., 1.]], dtype=float32))"
    )
    vec_env.close()


def test_observations_image():
    num_envs = 1
    env_fns = [lambda: pursuit_v4.parallel_env() for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fns)

    for agent in vec_env.agents:
        assert isinstance(vec_env.observations[agent], np.ndarray)
        assert vec_env.observations[agent].shape == (1, 7, 7, 3)
    vec_env.close()


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


class DictSpaceTestEnv(ParallelEnv):
    """Test environment with dictionary observation spaces"""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "dict_space_test_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            "agent_0": {
                "position": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "velocity": np.array([0.01, 0.02], dtype=np.float32),
            },
            "agent_1": {
                "position": np.array([0.4, 0.5, 0.6], dtype=np.float32),
                "velocity": np.array([0.03, 0.04], dtype=np.float32),
            },
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations = {
            "agent_0": {
                "position": np.array([0.2, 0.3, 0.4], dtype=np.float32),
                "velocity": np.array([0.02, 0.03], dtype=np.float32),
            },
            "agent_1": {
                "position": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "velocity": np.array([0.04, 0.05], dtype=np.float32),
            },
        }
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Dict(
            {
                "position": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "velocity": Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
            }
        )

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.ones((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


class TupleSpaceTestEnv(ParallelEnv):
    """Test environment with tuple observation spaces"""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "tuple_space_test_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            "agent_0": (
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.01, 0.02], dtype=np.float32),
            ),
            "agent_1": (
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
                np.array([0.03, 0.04], dtype=np.float32),
            ),
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations = {
            "agent_0": (
                np.array([0.2, 0.3, 0.4], dtype=np.float32),
                np.array([0.02, 0.03], dtype=np.float32),
            ),
            "agent_1": (
                np.array([0.5, 0.6, 0.7], dtype=np.float32),
                np.array([0.04, 0.05], dtype=np.float32),
            ),
        }
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Tuple(
            (
                Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
            )
        )

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.ones((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


# Tests for Dictionary Observation Spaces


@pytest.mark.parametrize("env_fns", [[lambda: DictSpaceTestEnv() for _ in range(8)]])
def test_create_async_pz_vector_env_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.single_action_space
    assert env.action_space
    assert env.single_observation_space
    assert env.observation_space
    assert env.observation_widths
    assert env.observation_boundaries
    assert env.observation_shapes
    assert env.num_envs == 8

    # Test dictionary-specific properties
    for agent in env.possible_agents:
        assert isinstance(env.single_observation_space(agent), spaces.Dict)
        assert isinstance(env.observation_shapes[agent], dict)
        assert "position" in env.observation_shapes[agent]
        assert "velocity" in env.observation_shapes[agent]
        assert env.observation_shapes[agent]["position"] == (3,)
        assert env.observation_shapes[agent]["velocity"] == (2,)

    for val in env._obs_buffer.values():
        assert isinstance(val, SynchronizedArray)
    assert isinstance(env.observations, Observations)
    assert env.processes
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize("env_fns", [[lambda: DictSpaceTestEnv() for _ in range(8)]])
def test_reset_async_pz_vector_env_dict_space(seed, env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    agents = env.possible_agents[:]
    observations, infos = env.reset(seed=seed)

    for agent in agents:
        assert isinstance(env.observation_space(agent), spaces.Dict)
        assert isinstance(observations[agent], dict)

        # Check position
        assert "position" in observations[agent]
        assert observations[agent]["position"].dtype == np.float32
        assert observations[agent]["position"].shape == (8, 3)

        # Check velocity
        assert "velocity" in observations[agent]
        assert observations[agent]["velocity"].dtype == np.float32
        assert observations[agent]["velocity"].shape == (8, 2)

    assert isinstance(infos, dict)
    assert set(agents).issubset(set(infos.keys()))
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: DictSpaceTestEnv(render_mode="rgb_array") for _ in range(8)]],
)
def test_render_async_pz_vector_env_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.render_mode == "rgb_array"

    env.reset()
    rendered_frames = env.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == env.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    env.close()


@pytest.mark.parametrize("env_fns", [[lambda: DictSpaceTestEnv() for _ in range(4)]])
def test_step_async_pz_vector_env_dict_space(env_fns):
    try:
        env = AsyncPettingZooVecEnv(env_fns)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(actions)

        for agent in env.agents:
            assert isinstance(env.observation_space(agent), spaces.Dict)

            # Check position
            assert "position" in observations[agent]
            assert observations[agent]["position"].dtype == np.float32
            assert observations[agent]["position"].shape == (4, 3)

            # Check velocity
            assert "velocity" in observations[agent]
            assert observations[agent]["velocity"].dtype == np.float32
            assert observations[agent]["velocity"].shape == (4, 2)

            # Check rewards, terminations, truncations
            assert isinstance(rewards[agent], np.ndarray)
            assert rewards[agent].ndim == 1
            assert rewards[agent].size == 4

            assert isinstance(terminations[agent], np.ndarray)
            assert terminations[agent].dtype == np.bool_
            assert terminations[agent].ndim == 1
            assert terminations[agent].size == 4

            assert isinstance(truncations[agent], np.ndarray)
            assert truncations[agent].dtype == np.bool_
            assert truncations[agent].ndim == 1
            assert truncations[agent].size == 4

        env.close()
    except Exception as e:
        env.close()
        raise e


# Tests for spaces.Tuple Observation Spaces


@pytest.mark.parametrize("env_fns", [[lambda: TupleSpaceTestEnv() for _ in range(8)]])
def test_create_async_pz_vector_env_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.single_action_space
    assert env.action_space
    assert env.single_observation_space
    assert env.observation_space
    assert env.observation_widths
    assert env.observation_boundaries
    assert env.observation_shapes
    assert env.num_envs == 8

    # Test tuple-specific properties
    for agent in env.possible_agents:
        assert isinstance(env.single_observation_space(agent), spaces.Tuple)
        assert isinstance(env.observation_shapes[agent], list)
        assert len(env.observation_shapes[agent]) == 2
        assert env.observation_shapes[agent][0] == (3,)
        assert env.observation_shapes[agent][1] == (2,)

    for val in env._obs_buffer.values():
        assert isinstance(val, SynchronizedArray)
    assert isinstance(env.observations, Observations)
    assert env.processes
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize("env_fns", [[lambda: TupleSpaceTestEnv() for _ in range(8)]])
def test_reset_async_pz_vector_env_tuple_space(seed, env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    agents = env.possible_agents[:]
    observations, infos = env.reset(seed=seed)

    for agent in agents:
        assert isinstance(env.observation_space(agent), spaces.Tuple)
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 2

        # Check first element (position)
        assert observations[agent][0].dtype == np.float32
        assert observations[agent][0].shape == (8, 3)

        # Check second element (velocity)
        assert observations[agent][1].dtype == np.float32
        assert observations[agent][1].shape == (8, 2)

    assert isinstance(infos, dict)
    assert set(agents).issubset(set(infos.keys()))
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: TupleSpaceTestEnv(render_mode="rgb_array") for _ in range(8)]],
)
def test_render_async_pz_vector_env_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.render_mode == "rgb_array"

    env.reset()
    rendered_frames = env.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == env.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    env.close()


@pytest.mark.parametrize("env_fns", [[lambda: TupleSpaceTestEnv() for _ in range(4)]])
def test_step_async_pz_vector_env_tuple_space(env_fns):
    try:
        env = AsyncPettingZooVecEnv(env_fns)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(actions)

        for agent in env.agents:
            assert isinstance(env.observation_space(agent), spaces.Tuple)
            assert isinstance(observations[agent], tuple)
            assert len(observations[agent]) == 2

            # Check first element (position)
            assert observations[agent][0].dtype == np.float32
            assert observations[agent][0].shape == (4, 3)

            # Check second element (velocity)
            assert observations[agent][1].dtype == np.float32
            assert observations[agent][1].shape == (4, 2)

            # Check rewards, terminations, truncations
            assert isinstance(rewards[agent], np.ndarray)
            assert rewards[agent].ndim == 1
            assert rewards[agent].size == 4

            assert isinstance(terminations[agent], np.ndarray)
            assert terminations[agent].dtype == np.bool_
            assert terminations[agent].ndim == 1
            assert terminations[agent].size == 4

            assert isinstance(truncations[agent], np.ndarray)
            assert truncations[agent].dtype == np.bool_
            assert truncations[agent].ndim == 1
            assert truncations[agent].size == 4

        env.close()
    except Exception as e:
        env.close()
        raise e


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: TupleSpaceTestEnv(render_mode="rgb_array") for _ in range(4)]],
)
def test_call_async_pz_vector_env_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()

    images = env.call("render")
    max_num_agents = env.call("possible_agents")
    env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert images[i].shape[-1] == 3
        assert isinstance(images[i], np.ndarray)

    assert isinstance(max_num_agents, tuple)
    assert len(max_num_agents) == 4


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: DictSpaceTestEnv(render_mode="rgb_array") for _ in range(4)]],
)
def test_call_async_pz_vector_env_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()

    images = env.call("render")
    max_num_agents = env.call("possible_agents")
    env.close()

    assert isinstance(images, tuple)
    assert len(images) == 4
    for i in range(4):
        assert images[i].shape[-1] == 3
        assert isinstance(images[i], np.ndarray)

    assert isinstance(max_num_agents, tuple)
    assert len(max_num_agents) == 4


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: DictSpaceTestEnv() for _ in range(2)]],
)
def test_get_attr_async_pz_vector_env_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr("test_attribute", [1, 2])
    test_attribute = env.get_attr("test_attribute")
    assert test_attribute == (1, 2)
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: TupleSpaceTestEnv() for _ in range(2)]],
)
def test_get_attr_async_pz_vector_env_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr("test_attribute", [1, 2])
    test_attribute = env.get_attr("test_attribute")
    assert test_attribute == (1, 2)
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: DictSpaceTestEnv() for _ in range(1)]],
)
def test_set_attr_make_values_list_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr(name="test", values=1)
    assert env.call("test")[0] == 1
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: TupleSpaceTestEnv() for _ in range(1)]],
)
def test_set_attr_make_values_list_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr(name="test", values=1)
    assert env.call("test")[0] == 1
    env.close()


# Test placeholder values for dict and tuple spaces
def test_get_placeholder_value_dict_space():
    observation_shapes = {"agent_0": {"position": (3,), "velocity": (2,)}}

    placeholder = get_placeholder_value("agent_0", "observation", observation_shapes)
    assert isinstance(placeholder, dict)
    assert "position" in placeholder
    assert "velocity" in placeholder
    assert placeholder["position"].shape == (3,)
    assert placeholder["velocity"].shape == (2,)
    assert np.all(placeholder["position"] == -1)
    assert np.all(placeholder["velocity"] == -1)


def test_get_placeholder_value_tuple_space():
    observation_shapes = {"agent_0": [(3,), (2,)]}

    placeholder = get_placeholder_value("agent_0", "observation", observation_shapes)
    assert isinstance(placeholder, tuple)
    assert len(placeholder) == 2
    assert placeholder[0].shape == (3,)
    assert placeholder[1].shape == (2,)
    assert np.all(placeholder[0] == -1)
    assert np.all(placeholder[1] == -1)


def test_worker_dict_space():
    """Test worker function with Dict observation spaces"""
    num_envs = 1
    env_fns = [lambda: DictSpaceTestEnv() for _ in range(num_envs)]

    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()

    # Reset the environment before stepping
    parent_pipe.send(("reset", {}))
    results, success = parent_pipe.recv()
    assert success

    # Step the environment
    parent_pipe.send(("step", actions[0]))
    results, success = parent_pipe.recv()
    assert success

    rewards, term, trunc, _ = results

    # Check observation structure
    for agent in vec_env.agents:
        assert isinstance(vec_env.observation_space(agent), spaces.Dict)

    # Check rewards
    assert isinstance(rewards["agent_0"], float)

    # Check termination/truncation
    assert isinstance(term["agent_0"], bool)
    assert isinstance(trunc["agent_0"], bool)

    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_tuple_space():
    """Test worker function with Tuple observation spaces"""
    num_envs = 1
    env_fns = [lambda: TupleSpaceTestEnv() for _ in range(num_envs)]

    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()

    # Reset the environment before stepping
    parent_pipe.send(("reset", {}))
    results, success = parent_pipe.recv()
    assert success

    # Step the environment
    parent_pipe.send(("step", actions[0]))
    results, success = parent_pipe.recv()
    assert success

    rewards, term, trunc, _ = results

    # Check observation structure
    for agent in vec_env.agents:
        assert isinstance(vec_env.observation_space(agent), spaces.Tuple)

    # Check rewards
    assert isinstance(rewards["agent_0"], float)

    # Check termination/truncation
    assert isinstance(term["agent_0"], bool)
    assert isinstance(trunc["agent_0"], bool)

    parent_pipe.close()
    p.terminate()
    p.join()


def test_add_info_dict_space():
    """Test _add_info with dictionary values in info"""
    info_list = [
        {"agent_0": {"metrics": {"distance": 1.5, "energy": 0.5}}},
        {"agent_0": {"metrics": {"distance": 2.0, "energy": 0.3}}},
    ]
    env_fns = [lambda: DictSpaceTestEnv() for _ in range(2)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {}

    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    assert "agent_0" in vector_infos
    assert "metrics" in vector_infos["agent_0"]
    assert "distance" in vector_infos["agent_0"]["metrics"]
    assert vector_infos["agent_0"]["metrics"]["distance"][0] == 1.5
    assert vector_infos["agent_0"]["metrics"]["distance"][1] == 2.0

    env.close()


def test_observations_dict_buffer():
    """Test Observations class with dictionary spaces"""
    num_envs = 2
    env_fns = [lambda: DictSpaceTestEnv() for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # Test reset to ensure Observations class is working
    observations, _ = env.reset()

    for agent in env.agents:
        # Check that observations are properly structured
        assert isinstance(observations[agent], dict)
        assert "position" in observations[agent]
        assert "velocity" in observations[agent]

        # Check shapes
        assert observations[agent]["position"].shape == (num_envs, 3)
        assert observations[agent]["velocity"].shape == (num_envs, 2)

    # Check the string representation
    obs_string = str(env.observations)
    assert "position" in obs_string
    assert "velocity" in obs_string

    env.close()


def test_observations_tuple_buffer():
    """Test Observations class with tuple spaces"""
    num_envs = 2
    env_fns = [lambda: TupleSpaceTestEnv() for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # Test reset to ensure Observations class is working
    observations, _ = env.reset()

    for agent in env.agents:
        # Check that observations are properly structured
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 2

        # Check shapes
        assert observations[agent][0].shape == (num_envs, 3)
        assert observations[agent][1].shape == (num_envs, 2)

    # Check the string representation
    obs_string = str(env.observations)
    assert "agent_0" in obs_string
    assert "agent_1" in obs_string

    env.close()


class ComplexDictSpaceTestEnv(ParallelEnv):
    """Test environment with dictionary observation spaces containing both vector and image data"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "complex_dict_space_test_v0",
    }

    def __init__(self, render_mode=None):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            "agent_0": {
                "position": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "velocity": np.array([0.01, 0.02], dtype=np.float32),
                "image": np.ones((16, 16, 3), dtype=np.uint8) * 100,
            },
            "agent_1": {
                "position": np.array([0.4, 0.5, 0.6], dtype=np.float32),
                "velocity": np.array([0.03, 0.04], dtype=np.float32),
                "image": np.ones((16, 16, 3), dtype=np.uint8) * 200,
            },
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations = {
            "agent_0": {
                "position": np.array([0.2, 0.3, 0.4], dtype=np.float32),
                "velocity": np.array([0.02, 0.03], dtype=np.float32),
                "image": np.ones((16, 16, 3), dtype=np.uint8) * 150,
            },
            "agent_1": {
                "position": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "velocity": np.array([0.04, 0.05], dtype=np.float32),
                "image": np.ones((16, 16, 3), dtype=np.uint8) * 250,
            },
        }
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Dict(
            {
                "position": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "velocity": Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
                "image": Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8),
            }
        )

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.ones((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


class ComplexTupleSpaceTestEnv(ParallelEnv):
    """Test environment with tuple observation spaces containing both vector and image data"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "complex_tuple_space_test_v0",
    }

    def __init__(self, render_mode=None):
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            "agent_0": (
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.01, 0.02], dtype=np.float32),
                np.ones((16, 16, 3), dtype=np.uint8) * 100,
            ),
            "agent_1": (
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
                np.array([0.03, 0.04], dtype=np.float32),
                np.ones((16, 16, 3), dtype=np.uint8) * 200,
            ),
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations = {
            "agent_0": (
                np.array([0.2, 0.3, 0.4], dtype=np.float32),
                np.array([0.02, 0.03], dtype=np.float32),
                np.ones((16, 16, 3), dtype=np.uint8) * 150,
            ),
            "agent_1": (
                np.array([0.5, 0.6, 0.7], dtype=np.float32),
                np.array([0.04, 0.05], dtype=np.float32),
                np.ones((16, 16, 3), dtype=np.uint8) * 250,
            ),
        }
        rewards = {agent: 1.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Tuple(
            (
                Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
                Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8),
            )
        )

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.ones((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


# Tests for Complex Dict Space (vector + image)
@pytest.mark.parametrize(
    "env_fns", [[lambda: ComplexDictSpaceTestEnv() for _ in range(4)]]
)
def test_create_async_pz_vector_env_complex_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.single_action_space
    assert env.action_space
    assert env.single_observation_space
    assert env.observation_space
    assert env.observation_widths
    assert env.observation_boundaries
    assert env.observation_shapes
    assert env.num_envs == 4

    # Test dictionary-specific properties
    for agent in env.possible_agents:
        assert isinstance(env.single_observation_space(agent), spaces.Dict)
        assert isinstance(env.observation_shapes[agent], dict)

        # Check vector parts
        assert "position" in env.observation_shapes[agent]
        assert "velocity" in env.observation_shapes[agent]
        assert env.observation_shapes[agent]["position"] == (3,)
        assert env.observation_shapes[agent]["velocity"] == (2,)

        # Check image part
        assert "image" in env.observation_shapes[agent]
        assert env.observation_shapes[agent]["image"] == (16, 16, 3)

    for val in env._obs_buffer.values():
        assert isinstance(val, SynchronizedArray)
    assert isinstance(env.observations, Observations)
    assert env.processes
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize(
    "env_fns", [[lambda: ComplexDictSpaceTestEnv() for _ in range(4)]]
)
def test_reset_async_pz_vector_env_complex_dict_space(seed, env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    agents = env.possible_agents[:]
    observations, infos = env.reset(seed=seed)

    for agent in agents:
        assert isinstance(env.observation_space(agent), spaces.Dict)
        assert isinstance(observations[agent], dict)

        # Check position
        assert "position" in observations[agent]
        assert observations[agent]["position"].dtype == np.float32
        assert observations[agent]["position"].shape == (4, 3)

        # Check velocity
        assert "velocity" in observations[agent]
        assert observations[agent]["velocity"].dtype == np.float32
        assert observations[agent]["velocity"].shape == (4, 2)

        # Check image
        assert "image" in observations[agent]
        assert observations[agent]["image"].shape == (4, 16, 16, 3)

    assert isinstance(infos, dict)
    assert set(agents).issubset(set(infos.keys()))
    env.close()


@pytest.mark.parametrize(
    "env_fns", [[lambda: ComplexDictSpaceTestEnv() for _ in range(3)]]
)
def test_step_async_pz_vector_env_complex_dict_space(env_fns):
    try:
        env = AsyncPettingZooVecEnv(env_fns)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(actions)

        for agent in env.agents:
            assert isinstance(env.observation_space(agent), spaces.Dict)

            # Check position
            assert "position" in observations[agent]
            assert observations[agent]["position"].dtype == np.float32
            assert observations[agent]["position"].shape == (3, 3)

            # Check velocity
            assert "velocity" in observations[agent]
            assert observations[agent]["velocity"].dtype == np.float32
            assert observations[agent]["velocity"].shape == (3, 2)

            # Check image
            assert "image" in observations[agent]
            assert observations[agent]["image"].shape == (3, 16, 16, 3)

            # Check rewards, terminations, truncations
            assert isinstance(rewards[agent], np.ndarray)
            assert rewards[agent].ndim == 1
            assert rewards[agent].size == 3

            assert isinstance(terminations[agent], np.ndarray)
            assert terminations[agent].dtype == np.bool_
            assert terminations[agent].ndim == 1
            assert terminations[agent].size == 3

            assert isinstance(truncations[agent], np.ndarray)
            assert truncations[agent].dtype == np.bool_
            assert truncations[agent].ndim == 1
            assert truncations[agent].size == 3

        env.close()
    except Exception as e:
        env.close()
        raise e


# Tests for Complex Tuple Space (vector + image)
@pytest.mark.parametrize(
    "env_fns", [[lambda: ComplexTupleSpaceTestEnv() for _ in range(4)]]
)
def test_create_async_pz_vector_env_complex_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.single_action_space
    assert env.action_space
    assert env.single_observation_space
    assert env.observation_space
    assert env.observation_widths
    assert env.observation_boundaries
    assert env.observation_shapes
    assert env.num_envs == 4

    # Test tuple-specific properties
    for agent in env.possible_agents:
        assert isinstance(env.single_observation_space(agent), spaces.Tuple)
        assert isinstance(env.observation_shapes[agent], list)
        assert len(env.observation_shapes[agent]) == 3

        # Check vector parts
        assert env.observation_shapes[agent][0] == (3,)
        assert env.observation_shapes[agent][1] == (2,)

        # Check image part
        assert env.observation_shapes[agent][2] == (16, 16, 3)

    for val in env._obs_buffer.values():
        assert isinstance(val, SynchronizedArray)
    assert isinstance(env.observations, Observations)
    assert env.processes
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize(
    "env_fns", [[lambda: ComplexTupleSpaceTestEnv() for _ in range(4)]]
)
def test_reset_async_pz_vector_env_complex_tuple_space(seed, env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    agents = env.possible_agents[:]
    observations, infos = env.reset(seed=seed)

    for agent in agents:
        assert isinstance(env.observation_space(agent), spaces.Tuple)
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 3

        # Check first element (position)
        assert observations[agent][0].dtype == np.float32
        assert observations[agent][0].shape == (4, 3)

        # Check second element (velocity)
        assert observations[agent][1].dtype == np.float32
        assert observations[agent][1].shape == (4, 2)

        # Check third element (image)
        assert observations[agent][2].shape == (4, 16, 16, 3)

    assert isinstance(infos, dict)
    assert set(agents).issubset(set(infos.keys()))
    env.close()


@pytest.mark.parametrize(
    "env_fns", [[lambda: ComplexTupleSpaceTestEnv() for _ in range(3)]]
)
def test_step_async_pz_vector_env_complex_tuple_space(env_fns):
    try:
        env = AsyncPettingZooVecEnv(env_fns)
        env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(actions)

        for agent in env.agents:
            assert isinstance(env.observation_space(agent), spaces.Tuple)
            assert isinstance(observations[agent], tuple)
            assert len(observations[agent]) == 3

            # Check first element (position)
            assert observations[agent][0].dtype == np.float32
            assert observations[agent][0].shape == (3, 3)

            # Check second element (velocity)
            assert observations[agent][1].dtype == np.float32
            assert observations[agent][1].shape == (3, 2)

            # Check third element (image)
            assert observations[agent][2].shape == (3, 16, 16, 3)

            # Check rewards, terminations, truncations
            assert isinstance(rewards[agent], np.ndarray)
            assert rewards[agent].ndim == 1
            assert rewards[agent].size == 3

            assert isinstance(terminations[agent], np.ndarray)
            assert terminations[agent].dtype == np.bool_
            assert terminations[agent].ndim == 1
            assert terminations[agent].size == 3

            assert isinstance(truncations[agent], np.ndarray)
            assert truncations[agent].dtype == np.bool_
            assert truncations[agent].ndim == 1
            assert truncations[agent].size == 3

        env.close()
    except Exception as e:
        env.close()
        raise e


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: ComplexTupleSpaceTestEnv(render_mode="rgb_array") for _ in range(4)]],
)
def test_render_async_pz_vector_env_complex_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.render_mode == "rgb_array"

    env.reset()
    rendered_frames = env.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == env.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: ComplexDictSpaceTestEnv(render_mode="rgb_array") for _ in range(4)]],
)
def test_render_async_pz_vector_env_complex_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.render_mode == "rgb_array"

    env.reset()
    rendered_frames = env.render()
    assert isinstance(rendered_frames, tuple)
    assert len(rendered_frames) == env.num_envs
    assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: ComplexTupleSpaceTestEnv(render_mode="rgb_array") for _ in range(4)]],
)
def test_call_async_pz_vector_env_complex_tuple_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()

    images = env.call("render")
    max_num_agents = env.call("possible_agents")
    assert images
    assert max_num_agents
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[lambda: ComplexDictSpaceTestEnv(render_mode="rgb_array") for _ in range(4)]],
)
def test_call_async_pz_vector_env_complex_dict_space(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()

    images = env.call("render")
    max_num_agents = env.call("possible_agents")
    assert images
    assert max_num_agents
    env.close()


def test_place_holder_value_complex_dict_space():
    """Test getting placeholder value for complex dictionary space"""
    obs_shapes = {"agent_0": {"position": (3,), "velocity": (2,), "image": (16, 16, 3)}}

    placeholder = get_placeholder_value("agent_0", "observation", obs_shapes)
    assert isinstance(placeholder, dict)
    assert "position" in placeholder
    assert "velocity" in placeholder
    assert "image" in placeholder

    assert placeholder["position"].shape == (3,)
    assert placeholder["velocity"].shape == (2,)
    assert placeholder["image"].shape == (16, 16, 3)

    assert np.all(placeholder["position"] == -1)
    assert np.all(placeholder["velocity"] == -1)
    assert np.all(placeholder["image"] == -1)


def test_place_holder_value_complex_tuple_space():
    """Test getting placeholder value for complex tuple space"""
    obs_shapes = {"agent_0": [(3,), (2,), (16, 16, 3)]}

    placeholder = get_placeholder_value("agent_0", "observation", obs_shapes)
    assert isinstance(placeholder, tuple)
    assert len(placeholder) == 3

    assert placeholder[0].shape == (3,)
    assert placeholder[1].shape == (2,)
    assert placeholder[2].shape == (16, 16, 3)

    assert np.all(placeholder[0] == -1)
    assert np.all(placeholder[1] == -1)
    assert np.all(placeholder[2] == -1)


def test_worker_complex_dict_space():
    """Test worker function with complex Dict observation spaces"""
    num_envs = 1
    env_fns = [lambda: ComplexDictSpaceTestEnv() for _ in range(num_envs)]

    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()

    # Reset the environment before stepping
    parent_pipe.send(("reset", {}))
    results, success = parent_pipe.recv()
    assert success

    # Step the environment
    parent_pipe.send(("step", actions[0]))
    results, success = parent_pipe.recv()
    assert success

    rewards, term, trunc, _ = results

    # Check observation structure
    for agent in vec_env.agents:
        assert isinstance(vec_env.observation_space(agent), spaces.Dict)

    # Check rewards
    assert isinstance(rewards["agent_0"], float)

    # Check termination/truncation
    assert isinstance(term["agent_0"], bool)
    assert isinstance(trunc["agent_0"], bool)

    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_complex_tuple_space():
    """Test worker function with complex Tuple observation spaces"""
    num_envs = 1
    env_fns = [lambda: ComplexTupleSpaceTestEnv() for _ in range(num_envs)]

    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.observation_shapes,
            vec_env.observation_widths,
            vec_env.observation_dtypes,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()

    # Reset the environment before stepping
    parent_pipe.send(("reset", {}))
    results, success = parent_pipe.recv()
    assert success

    # Step the environment
    parent_pipe.send(("step", actions[0]))
    results, success = parent_pipe.recv()
    assert success

    rewards, term, trunc, _ = results

    # Check observation structure
    for agent in vec_env.agents:
        assert isinstance(vec_env.observation_space(agent), spaces.Tuple)

    # Check rewards
    assert isinstance(rewards["agent_0"], float)

    # Check termination/truncation
    assert isinstance(term["agent_0"], bool)
    assert isinstance(trunc["agent_0"], bool)

    parent_pipe.close()
    p.terminate()
    p.join()
