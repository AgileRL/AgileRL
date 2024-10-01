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
import pettingzoo
import pytest
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector.utils import CloudpickleWrapper

from agilerl.vector.pz_async_vec_env import (
    AsyncState,
    AsyncVectorPettingZooEnv,
    PettingZooExperienceHandler,
    _async_worker,
    dict_to_1d_array,
)
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
def pz_experience_handler(env_fns):
    return PettingZooExperienceHandler(env_fns[0], 8)


# # FIXME why are we getting an io error for use_exp_handler True
# @pytest.mark.parametrize("use_exp_handler", [True])  # False needs to be added back in
# @pytest.mark.parametrize(
#     "env_fns", [[lambda: pettingzoo.mpe.simple_v3.parallel_env() for _ in range(1)]]
# )
# def test_create_async_pz_vector_env(pz_experience_handler, use_exp_handler, env_fns):
#     # try:

#     experience_handler = pz_experience_handler if use_exp_handler else None
#     env = AsyncVectorPettingZooEnv(env_fns, experience_handler=experience_handler)
#     pids = [p.pid for p in env.processes]
#     assert env.num_envs == 1
#     env.reset()
#     actions = {
#         agent: [env.single_action_space(agent).sample() for _ in range(1)]
#         for agent in env.agents
#     }
#     observations, rewards, terminations, truncations, _ = env.step(actions)
#     env.render()
#     env.close()
#     assert False

# except Exception as e:
#     env.close()
#     raise e


@pytest.mark.parametrize("seed", [1, None])
def test_reset_async_pz_vector_env(seed):
    try:
        env_fns = [
            lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
            for _ in range(8)
        ]
        env = AsyncVectorPettingZooEnv(env_fns)
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
            env_fns = [
                pettingzoo.mpe.simple_speaker_listener_v4.parallel_env for _ in range(8)
            ]
            env = AsyncVectorPettingZooEnv(env_fns)
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

    except Exception as e:
        env.close()
        raise e


def test_render_async_pz_vector_env():
    try:
        env_fns = [
            lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
                render_mode="rgb_array"
            )
            for _ in range(8)
        ]
        env = AsyncVectorPettingZooEnv(env_fns)
        assert env.render_mode == "rgb_array"

        env.reset()
        rendered_frames = env.render()
        assert isinstance(rendered_frames, tuple)
        assert len(rendered_frames) == env.num_envs
        assert all(isinstance(frame, np.ndarray) for frame in rendered_frames)
        env.close()

    except Exception as e:
        env.close()
        raise e


@pytest.mark.parametrize("use_single_action_space", [False, True])
def test_step_async_pz_vector_env(use_single_action_space):
    try:
        env_fns = [
            lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
                continuous_actions=False
            )
            for _ in range(8)
        ]
        env = AsyncVectorPettingZooEnv(env_fns)
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


def test_call_async_pz_vector_env():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
            render_mode="rgb_array", continuous_actions=False
        )
        for _ in range(4)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    try:
        env.reset()

        images = env.call("render")
        max_num_agents = env.call("max_num_agents")
        env.close()

        assert isinstance(images, tuple)
        assert len(images) == 4
        for i in range(4):
            print(images[i].shape)
            assert images[i].shape[-1] == 3
            assert isinstance(images[i][0], np.ndarray)

        assert isinstance(max_num_agents, tuple)
        assert len(max_num_agents) == 4
        for i in range(4):
            assert isinstance(max_num_agents[i], int)
            assert max_num_agents[i] == 2
    except Exception as e:
        env.close()
        raise e


def test_get_attr_async_pz_vector_env():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
            continuous_actions=False
        )
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    try:
        env.set_attr("test_attribute", [1, 2])
        test_attribute = env.get_attr("test_attribute")
        assert test_attribute == (1, 2)
        env.close()
    except Exception as e:
        env.close()
        raise e


def test_set_attr_make_values_list():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
            continuous_actions=False
        )
        for _ in range(1)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    env.set_attr(name="test", values=1)
    env.close()


def raise_error_reset(self, seed=None, options=None):
    if seed == 1:
        print("Raising error")
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
    envs = AsyncVectorPettingZooEnv(
        [
            lambda: GenericTestEnv(
                reset_func=raise_error_reset, step_func=raise_error_step
            )
        ]
        * 2
    )

    with pytest.raises(ValueError, match="Error in reset"):
        envs.reset(seed=[1, 0])

    envs.close()

    envs = AsyncVectorPettingZooEnv(
        [
            lambda: GenericTestEnv(
                reset_func=raise_error_reset, step_func=raise_error_step
            )
        ]
        * 3
    )

    with pytest.raises(ValueError, match="Error in step"):
        envs.step({"agent_0": [0, 1, 2]})  # np.array([[0], [1], [2]])})

    envs.close()


def test_custom_space_error():
    num_envs = 4
    env_fns = [
        lambda: GenericTestEnv(
            action_space=CustomSpace(), observation_space=CustomSpace()
        )
    ] * num_envs
    with pytest.raises(ValueError):
        AsyncVectorPettingZooEnv(env_fns)


def test_reset_async_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.reset_async()

    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_reset_wait_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    with pytest.raises(NoAsyncCallError):
        env.reset_async()
        env._state = AsyncState.DEFAULT
        env.reset_wait()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_step_async_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.step_async(actions=None)
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_step_wait_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.step_wait()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_call_async_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.call_async("test")
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_call_wait_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.call_wait()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_call_exception_worker():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    with pytest.raises(ValueError):
        env.call("reset")
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_set_attr_val_error():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    with pytest.raises(ValueError):
        env.set_attr("test", values=[1, 2, 3])
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_set_attr_exception():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.set_attr("test", values=[1, 2])
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_close_extras_warning():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    with patch.object(gym.logger, "warn") as mock_logger_warn:
        env.close_extras(timeout=None)
        mock_logger_warn.assert_called_once
    env.close()


def test_close_extras_terminate():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    env.close_extras(terminate=True)

    for p in env.processes:
        assert not p.is_alive()


def test_poll_pipe_envs():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env.parent_pipes[0] = None
    result = env._poll_pipe_envs(timeout=1)
    assert not result
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_assert_is_running():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env.closed = True
    with pytest.raises(ClosedEnvironmentError):
        env._assert_is_running()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_step_wait_timeout_async_pz_vector_env():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    pids = [p.pid for p in env.processes]
    env._state = AsyncState.WAITING_STEP
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.step_wait(timeout=1)
        env.close()
    for pid in pids:
        os.kill(pid, signal.SIGTERM)


def test_call_wait_timeout_async_pz_vector_env():
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
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
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(2)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
    if transition_name != "observation":
        val = env.experience_handler.get_placeholder_value(0, "agent", transition_name)
        if transition_name == "reward":
            assert val == 0
        if transition_name == "truncation" or transition_name == "termination":
            assert val
        if transition_name == "info":
            assert val == {}
    else:
        env.reset()
        output = env.experience_handler.get_placeholder_value(
            index=0,
            agent="speaker_0",
            transition_name=transition_name,
            shared_memory=env._obs_buffer,
        )
        assert isinstance(output, np.ndarray)
    env.close()


def test_read_obs_from_shared_memory():
    shared_memory = mp.RawArray("d", 28)  # width 16
    env_fn = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(1)
    ][0]
    exp_handler = PettingZooExperienceHandler(env_fn, 2)

    # Write to the shared memory object
    destination = np.frombuffer(shared_memory, dtype=float)
    np.copyto(
        destination[:3], 3 * np.ones((1, 3), dtype=np.float32)
    )  # shapes of simple speaker listener obs
    np.copyto(destination[3:14], 4 * np.ones((1, 11), dtype=np.float32))

    speaker_obs = exp_handler.read_obs_from_shared_memory(shared_memory, "speaker_0", 0)

    listener_obs = exp_handler.read_obs_from_shared_memory(
        shared_memory, "listener_0", 0
    )
    assert np.all(speaker_obs == 3 * np.ones((1, 3)))
    assert np.all(listener_obs == 4 * np.ones((1, 11)))


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
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(3)
    ]
    env = AsyncVectorPettingZooEnv(env_fns)
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
    env = AsyncVectorPettingZooEnv(env_fns)
    vector_infos = {"agent_0": {}}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    env.close()


def test_add_info_unknown_objects():
    info_list = [
        {"agent_0": "string"},
        {"agent_1": "string"},
    ]
    env_fns = [lambda: GenericTestEnv() for _ in range(3)]
    env = AsyncVectorPettingZooEnv(env_fns)
    vector_infos = {"agent_0": {}}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    env.close()


def test_create_experience_handler():
    num_envs = 8
    env_fns = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(num_envs)
    ]
    env = pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
    experience_handler = PettingZooExperienceHandler(env_fns[0], num_envs)

    assert experience_handler.single_action_space_dict == {
        agent: env.action_space(agent) for agent in env.possible_agents
    }
    assert experience_handler.single_observation_space_dict == {
        agent: env.observation_space(agent) for agent in env.possible_agents
    }
    assert experience_handler.observation_widths == {
        agent: int(np.prod(obs.shape))
        for agent, obs in experience_handler.single_observation_space_dict.items()
    }
    assert experience_handler.observation_boundaries == [0] + list(
        accumulate(experience_handler.observation_widths.values(), operator.add)
    )
    assert experience_handler.total_observation_width == int(
        np.sum(list(experience_handler.observation_widths.values()))
    )


def test_worker_reset():
    env_fn = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(1)
    ][0]
    env = env_fn()
    env.reset()
    exp_handler = PettingZooExperienceHandler(env_fn, 3)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn),
            child_pipe,
            parent_pipe,
            None,
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
    for dic in results:
        assert list(sorted(dic.keys())) == sorted(env.aec_env.agents)
    parent_pipe.close()
    p.terminate()
    p.join()


@pytest.mark.parametrize("env_type", ["PZ"])
def test_worker_step(env_type):
    env_fn = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env(
            continuous_actions=True
        )
        for _ in range(1)
    ]

    vec_env = AsyncVectorPettingZooEnv(env_fn)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    exp_handler = PettingZooExperienceHandler(env_fn[0], 1)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn[0]),
            child_pipe,
            parent_pipe,
            None,
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

    states, rewards, term, trunc, _ = results

    # state check
    assert states["speaker_0"].shape == (3,)
    assert states["listener_0"].shape == (11,)
    assert isinstance(states["speaker_0"], np.ndarray)
    assert isinstance(states["listener_0"], np.ndarray)

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
    env_fn = [lambda: term_env() for _ in range(1)]
    vec_env = AsyncVectorPettingZooEnv(env_fn)
    vec_env.reset()

    actions = {agent: vec_env.action_space(agent).sample() for agent in vec_env.agents}
    vec_env.close()
    actions = actions_to_list_helper(actions)
    exp_handler = PettingZooExperienceHandler(env_fn[0], 1)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn[0]),
            child_pipe,
            parent_pipe,
            None,
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

    # Send step again and autoreset should be True
    parent_pipe.send(("step", actions[0]))
    parent_pipe.recv()
    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_runtime_error():
    env_fn = [
        lambda: pettingzoo.mpe.simple_speaker_listener_v4.parallel_env()
        for _ in range(1)
    ][0]
    env = env_fn()
    env.reset()
    exp_handler = PettingZooExperienceHandler(env_fn, 3)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn),
            child_pipe,
            parent_pipe,
            None,
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


@pytest.mark.parametrize(
    "input_dict",
    [({"agent_0": 6}), ({"agent_0": np.array([1, 0, 1])}), ({"agent_0": [2, 3, 4]})],
)
def test_dict_to_1d_array(input_dict):
    output = dict_to_1d_array(input_dict)
    assert isinstance(output, np.ndarray)


@pytest.mark.parametrize(
    "input_dict",
    [
        ({"agent_0": (6,)}),
    ],
)
def test_dict_to_1d_array_error(input_dict):
    with pytest.raises(TypeError):
        dict_to_1d_array(input_dict)
