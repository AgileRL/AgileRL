"""Tests for pz_async_vec_env.py and pz_vec_env.py"""

import multiprocessing as mp
import os
import signal
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    NoAsyncCallError,
)
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from gymnasium.vector.utils import CloudpickleWrapper
from pettingzoo import ParallelEnv
from tests.pz_vector_test_utils import (
    GenericTestEnv,
    SpeakerListenerLikeEnv,
    speaker_listener_like_env,
    term_env,
)

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import (  # PettingZooExperienceSpec,; SharedMemory,
    AsyncPettingZooVecEnv,
    AsyncState,
    Observations,
    _async_worker,
    create_shared_memory,
    get_placeholder_value,
    write_to_shared_memory,
)
from agilerl.vector.pz_vec_env import PettingZooVecEnv


class DummyRecv:
    def __init__(self, cmd, data):
        self.call_count = 0
        self.cmd = cmd
        self.data = data

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > 1:
            return "close", None
        return self.cmd, self.data


class DictSpaceTestEnv(ParallelEnv):
    """Test environment with dictionary observation spaces"""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "dict_space_test_v0"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            "agent_0": {
                "position": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "velocity": np.array([0.01, 0.02], dtype=np.float32),
            },
            "other_agent_0": {
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
            "other_agent_0": {
                "position": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "velocity": np.array([0.04, 0.05], dtype=np.float32),
            },
        }
        rewards = dict.fromkeys(self.agents, 1.0)
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Dict(
            {
                "position": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "velocity": Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
            },
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
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.agents = self.possible_agents.copy()
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        observations = {
            "agent_0": (
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.01, 0.02], dtype=np.float32),
            ),
            "other_agent_0": (
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
            "other_agent_0": (
                np.array([0.5, 0.6, 0.7], dtype=np.float32),
                np.array([0.04, 0.05], dtype=np.float32),
            ),
        }
        rewards = dict.fromkeys(self.agents, 1.0)
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Tuple(
            (
                Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
            ),
        )

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.ones((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


class ComplexDictSpaceTestEnv(ParallelEnv):
    """Test environment with dictionary observation spaces containing both vector and image data"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "complex_dict_space_test_v0",
    }

    def __init__(self, render_mode=None):
        self.possible_agents = ["agent_0", "other_agent_0"]
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
            "other_agent_0": {
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
            "other_agent_0": {
                "position": np.array([0.5, 0.6, 0.7], dtype=np.float32),
                "velocity": np.array([0.04, 0.05], dtype=np.float32),
                "image": np.ones((16, 16, 3), dtype=np.uint8) * 250,
            },
        }
        rewards = dict.fromkeys(self.agents, 1.0)
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Dict(
            {
                "position": Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "velocity": Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
                "image": Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8),
            },
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
        self.possible_agents = ["agent_0", "other_agent_0"]
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
            "other_agent_0": (
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
            "other_agent_0": (
                np.array([0.5, 0.6, 0.7], dtype=np.float32),
                np.array([0.04, 0.05], dtype=np.float32),
                np.ones((16, 16, 3), dtype=np.uint8) * 250,
            ),
        }
        rewards = dict.fromkeys(self.agents, 1.0)
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {agent: {} for agent in self.agents}
        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return spaces.Tuple(
            (
                Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32),
                Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8),
            ),
        )

    def action_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return np.ones((64, 64, 3), dtype=np.uint8)
        return None

    def close(self):
        pass


def make_observation_views(env_cls, num_envs=1):
    env = env_cls()
    obs_spaces = {
        agent: env.observation_space(agent) for agent in env.possible_agents
    }
    agents = env.possible_agents.copy()
    env.close()
    shared_memory = create_shared_memory(num_envs, obs_spaces, mp.get_context())
    observations = Observations(shared_memory, obs_spaces, num_envs)
    return agents, obs_spaces, shared_memory, observations


def actions_to_list_helper(actions):
    passed_actions_list = [[] for _ in list(actions.values())[0]]
    for env_idx, _ in enumerate(list(actions.values())[0]):
        for possible_agent in actions:
            passed_actions_list[env_idx].append(actions[possible_agent][env_idx])

    return passed_actions_list


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
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_create_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    assert env.single_action_space
    assert env.action_space
    assert env.single_observation_space
    assert env.observation_space
    assert env.num_envs == 2
    for val in env._obs_buffer.values():
        assert isinstance(val, mp.sharedctypes.SynchronizedArray)
    assert isinstance(env.observations, Observations)
    assert env.processes
    env.reset()
    env.close()


@pytest.mark.parametrize("seed", [1, None])
@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_reset_async_pz_vector_env(seed, env_fns):
    num_envs = 2
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
            == (num_envs,) + env.single_observation_space(agent).shape
        )
        assert observations[agent].shape == env.observation_space(agent).shape
    assert isinstance(infos, dict)
    assert set(agents).issubset(set(infos.keys()))

    try:
        env_fns = [speaker_listener_like_env for _ in range(num_envs)]
        env = AsyncPettingZooVecEnv(env_fns)
    finally:
        env.close()
    for agent in agents:
        assert isinstance(env.observation_space(agent), Box)
        assert isinstance(observations[agent], np.ndarray)
        assert observations[agent].dtype == env.observation_space(agent).dtype
        assert (
            observations[agent].shape
            == (num_envs,) + env.single_observation_space(agent).shape
        )
        assert observations[agent].shape == env.observation_space(agent).shape
        assert set(agents).issubset(set(infos.keys()))


@pytest.mark.parametrize(
    "env_fns",
    [
        [lambda: speaker_listener_like_env(render_mode="rgb_array") for _ in range(2)],
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
        [lambda: speaker_listener_like_env(continuous_actions=False) for _ in range(2)],
    ],
)
def test_step_async_pz_vector_env(use_single_action_space, env_fns):
    num_envs = 2
    try:
        env = AsyncPettingZooVecEnv(env_fns)
        env.reset()
        if use_single_action_space:
            actions = {
                agent: [
                    env.single_action_space(agent).sample() for _ in range(num_envs)
                ]
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
                == (num_envs,) + env.single_observation_space(agent).shape
            )
            assert observations[agent].shape == env.observation_space(agent).shape
            assert isinstance(rewards[agent], np.ndarray)
            assert isinstance(rewards[agent][0], (float, np.floating))
            assert rewards[agent].ndim == 1
            assert rewards[agent].size == num_envs
            assert isinstance(terminations[agent], np.ndarray)
            assert terminations[agent].dtype == bool
            assert terminations[agent].ndim == 1
            assert terminations[agent].size == num_envs
            assert isinstance(truncations[agent], np.ndarray)
            assert truncations[agent].dtype == bool
            assert truncations[agent].ndim == 1
            assert truncations[agent].size == num_envs
        env.close()

    except Exception as e:
        env.close()
        raise e


@pytest.mark.parametrize(
    "env_fns",
    [
        [
            lambda: speaker_listener_like_env(
                render_mode="rgb_array",
                continuous_actions=False,
            )
            for _ in range(2)
        ],
    ],
)
def test_call_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    num_envs = 2
    env.reset()

    images = env.call("render")
    max_num_agents = env.call("max_num_agents")
    env.close()

    assert isinstance(images, tuple)
    assert len(images) == num_envs
    for i in range(num_envs):
        assert images[i].shape[-1] == 3
        assert isinstance(images[i][0], np.ndarray)

    assert isinstance(max_num_agents, tuple)
    assert len(max_num_agents) == num_envs
    for i in range(num_envs):
        assert isinstance(max_num_agents[i], int)
        assert max_num_agents[i] == 2


@pytest.mark.parametrize(
    "env_fns",
    [
        [lambda: speaker_listener_like_env(continuous_actions=False) for _ in range(2)],
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
        [lambda: speaker_listener_like_env(continuous_actions=False) for _ in range(1)],
    ],
)
def test_set_attr_make_values_list(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.set_attr(name="test", values=1)
    assert env.call("test")[0] == 1
    env.close()


def test_env_order_preserved():
    num_envs = 4
    env_fns = [
        lambda: speaker_listener_like_env(continuous_actions=False)
        for _ in range(num_envs)
    ]
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset()
    for obs_view in env.observations.obs_view.values():
        obs_view[:] = 0
    actions = {
        agent_id: env.single_action_space(agent_id).sample()
        for agent_id in env.possible_agents
    }
    rand_env = np.random.randint(0, num_envs)
    env.parent_pipes[rand_env].send(("step", actions))
    env.parent_pipes[rand_env].recv()
    for agent in env.agents:
        size = int(np.prod(env.single_observation_space(agent).shape))
        assert not np.array_equal(
            env.observations.obs_view[agent][rand_env * size : (rand_env + 1) * size],
            np.zeros_like(env.single_observation_space(agent).shape),
        )
    env.close()


def raise_error_reset(self, seed=None, options=None):
    if seed == 1:
        msg = "Error in reset"
        raise ValueError(msg)
    return {
        agent: self.observation_space(agent).sample() for agent in self.possible_agents
    }, {agent: {} for agent in self.possible_agents}


def raise_error_step(self, action):
    if list(action.values())[0] >= 1:
        msg = f"Error in step with {action}"
        raise ValueError(msg)

    def pz_dict(transition, agents):
        return dict.fromkeys(self.possible_agents, transition)

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


def raising_worker_env_constructor():
    if mp.current_process().name != "MainProcess":
        msg = "Error creating env"
        raise RuntimeError(msg)
    return GenericTestEnv()


def test_async_vector_subenv_error():
    env_list = [
        lambda: GenericTestEnv(
            reset_func=raise_error_reset,
            step_func=raise_error_step,
        ),
    ]
    envs = AsyncPettingZooVecEnv(env_list * 2)

    with pytest.raises(ValueError, match="Error in reset"):
        envs.reset(seed=[1, 0])

    envs.close()


@pytest.mark.parametrize(
    "env_fns",
    [[raising_worker_env_constructor]],
)
def test_async_vector_subenv_init_error(env_fns):
    envs = AsyncPettingZooVecEnv(env_fns)

    with pytest.raises(RuntimeError, match="Error creating env"):
        envs.reset()

    envs.close(terminate=True)
    env_list = [
        lambda: GenericTestEnv(
            reset_func=raise_error_reset,
            step_func=raise_error_step,
        ),
    ]
    envs = AsyncPettingZooVecEnv(env_list * 2)

    envs.reset()
    with pytest.raises(ValueError, match="Error in step"):
        envs.step({"agent_0": [0, 1]})

    envs.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_reset_async_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.reset_async()
    env.close(terminate=True)


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_reset_wait_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    with pytest.raises(NoAsyncCallError):
        env.reset_async()
        env._state = AsyncState.DEFAULT
        env.reset_wait()
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_step_async_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_RESET
    with pytest.raises(AlreadyPendingCallError):
        env.step_async(actions=None)
    env.close(terminate=True)


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_step_wait_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.step_wait()
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_call_async_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.call_async("test")
    env.close(terminate=True)


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_call_wait_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.DEFAULT
    with pytest.raises(NoAsyncCallError):
        env.call_wait()
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_call_exception_worker(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    with pytest.raises(ValueError):
        env.call("reset")
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_set_attr_val_error(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    with pytest.raises(ValueError):
        env.set_attr("test", values=[1, 2, 3])
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_set_attr_exception(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(AlreadyPendingCallError):
        env.set_attr("test", values=[1, 2])
    env.close(terminate=True)


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_close_extras_warning(env_fns):
    env = AsyncPettingZooVecEnv(
        env_fns,
    )
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    with patch.object(gym.logger, "warn") as mock_logger_warn:
        env.close_extras(timeout=None)
        mock_logger_warn.assert_called_once()
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_close_extras_terminate(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.reset_async()
    env._state = AsyncState.WAITING_RESET
    env.close_extras(terminate=True)

    for p in env.processes:
        assert not p.is_alive()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_poll_pipe_envs(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.parent_pipes[0] = None
    result = env._poll_pipe_envs(timeout=1)
    assert not result
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_poll_pipe_envs_ready(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    original_pipes = env.parent_pipes

    class ReadyPipe:
        closed = False

        def poll(self, _timeout):
            return True

    env.parent_pipes = [ReadyPipe() for _ in original_pipes]
    try:
        assert env._poll_pipe_envs(timeout=1.0)
    finally:
        env.parent_pipes = original_pipes
        env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_assert_is_running(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env.closed = True
    with pytest.raises(ClosedEnvironmentError):
        env._assert_is_running()
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_step_wait_timeout_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_STEP
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.step_wait(timeout=1)
        env.close()
    env.close()


@pytest.mark.parametrize(
    "env_fns",
    [[speaker_listener_like_env for _ in range(2)]],
)
def test_call_wait_timeout_async_pz_vector_env(env_fns):
    env = AsyncPettingZooVecEnv(env_fns)
    env._state = AsyncState.WAITING_CALL
    with pytest.raises(mp.TimeoutError):
        env.parent_pipes[0] = None
        env.call_wait(timeout=1)
        env.close()
    env.close()


@pytest.mark.parametrize(
    "transition_name",
    ["reward", "truncated", "terminated", "info", "observation"],
)
def test_get_placeholder_value(transition_name):
    env_fns = [speaker_listener_like_env for _ in range(2)]
    env = AsyncPettingZooVecEnv(env_fns)
    if transition_name != "observation":
        val = get_placeholder_value("agent", transition_name)
        if transition_name == "reward":
            assert np.isnan(val)
        if transition_name == "truncated":
            assert np.isnan(val)
        if transition_name == "terminated":
            assert np.isnan(val)
        if transition_name == "info":
            assert val == {}
    else:
        env.reset()
        output = get_placeholder_value(
            agent="speaker_0",
            transition_name=transition_name,
            obs_spaces=env._single_observation_spaces,
        )
        assert isinstance(output, np.ndarray)
    env.close()


def test_get_placeholder_value_observation_without_spaces():
    output = get_placeholder_value(
        agent="speaker_0",
        transition_name="observation",
        obs_spaces=None,
    )
    assert output is None


def test_get_placeholder_value_unknown_transition():
    output = get_placeholder_value(agent="speaker_0", transition_name="unknown")
    assert output is None


def test_add_info_dictionaries():
    info_list = [
        {
            "agent_0": {
                "env_defined_actions": np.array([1, 2, 3]),
                "action_mask": np.array([1, 0]),
            },
            "other_agent_0": {},
        },
        {
            "agent_0": {},
            "other_agent_0": {
                "env_defined_actions": np.array([5, 6, 7]),
                "action_mask": np.array([0, 1]),
            },
        },
        {"agent_0": {}, "other_agent_0": {}},
    ]
    env_fns = [speaker_listener_like_env for _ in range(3)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    assert np.all(
        vector_infos["agent_0"]["env_defined_actions"]
        == np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]]),
    )
    assert np.all(
        vector_infos["agent_0"]["action_mask"] == np.array([[1, 0], [0, 0], [0, 0]]),
    )
    assert np.all(
        vector_infos["other_agent_0"]["env_defined_actions"]
        == np.array([[0, 0, 0], [5, 6, 7], [0, 0, 0]]),
    )
    assert np.all(
        vector_infos["other_agent_0"]["action_mask"]
        == np.array([[0, 0], [0, 1], [0, 0]]),
    )
    env.close()


def test_add_info_int():
    info_list = [
        {"agent_0": 1.0},
        {"other_agent_0": 1},
    ]
    env_fns = [GenericTestEnv for _ in range(3)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {"agent_0": {}}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)

    env.close()


def test_add_info_unknown_objects():
    info_list = [
        {"agent_0": "string"},
        {"other_agent_0": "string"},
        {"agent_2": None},
    ]
    env_fns = [GenericTestEnv for _ in range(3)]
    env = AsyncPettingZooVecEnv(env_fns)
    vector_infos = {"agent_0": {}}
    for i, info in enumerate(info_list):
        vector_infos = env._add_info(vector_infos, info, i)
    env.close()


def test_worker_reset():
    env_fns = [speaker_listener_like_env for _ in range(1)]
    env_fn = env_fns[0]
    env = env_fn()
    env.reset()
    vec_env = AsyncPettingZooVecEnv(env_fns)
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = mp.Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fn),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()
    parent_pipe.send(("reset", {}))
    results, success = parent_pipe.recv()
    assert success
    assert len(results) == 2
    assert sorted(results.keys()) == sorted(env.possible_agents)
    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_step_simple():
    num_envs = 1
    env_fns = [
        lambda: speaker_listener_like_env(continuous_actions=True)
        for _ in range(num_envs)
    ]

    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()

    actions = {
        agent: vec_env.single_action_space(agent).sample() for agent in vec_env.agents
    }
    vec_env.close()
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = mp.Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()
    # Reset the environment before stepping. ``parent_pipe.recv`` already
    # synchronises with the worker; the previous ``time.sleep(1)`` was a
    # leftover hand-tuning that just inflated runtime.
    parent_pipe.send(("reset", {}))
    parent_pipe.recv()

    parent_pipe.send(("step", actions))
    results, success = parent_pipe.recv()
    assert success
    for dic in results:
        assert sorted(dic.keys()) == sorted(vec_env.agents)

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
    env_fns = [term_env for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fns)
    vec_env.reset()
    actions = {
        agent: np.array([vec_env.single_action_space(agent).sample()])
        for agent in vec_env.agents
    }
    vec_env.close()
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    p = mp.Process(
        target=_async_worker,
        args=(
            0,
            CloudpickleWrapper(env_fns[0]),
            child_pipe,
            parent_pipe,
            vec_env._obs_buffer,
            queue,
            vec_env.agents,
        ),
    )
    p.start()
    child_pipe.close()
    # Reset the environment before stepping. ``recv`` is the synchronisation
    # primitive; the previous ``time.sleep(1)`` here was redundant.
    parent_pipe.send(("reset", {}))
    parent_pipe.recv()
    parent_pipe.send(("step", actions))
    results, success = parent_pipe.recv()
    assert success
    for dic in results:
        assert sorted(dic.keys()) == sorted(vec_env.agents)

    # Send step again and autoreset should be True
    parent_pipe.send(("step", actions))
    parent_pipe.recv()
    parent_pipe.close()
    p.terminate()
    p.join()


def test_worker_runtime_error():
    num_envs = 1
    env_fns = [speaker_listener_like_env for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fns)
    env_fn = env_fns[0]
    env = env_fn()
    env.reset()
    parent_pipe, child_pipe = mp.Pipe()
    queue = mp.Queue()
    try:
        p = mp.Process(
            target=_async_worker,
            args=(
                0,
                CloudpickleWrapper(env_fn),
                child_pipe,
                parent_pipe,
                vec_env._obs_buffer,
                queue,
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
    env_fns = [speaker_listener_like_env for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fns)
    assert (
        vec_env.observations.__str__()
        == vec_env.observations.__repr__()
        == "{'speaker_0': array([0., 0., 0.], dtype=float32), 'listener_0': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}"
    ), vec_env.observations.__str__()
    ob = {"speaker_0": np.ones((1, 3)), "listener_0": np.ones((1, 11))}
    write_to_shared_memory(
        0,
        ob,
        vec_env._obs_buffer,
        vec_env._single_observation_spaces,
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
        == np.ones((1, 14), dtype=np.float32),
    )
    assert np.all(
        vec_env.observations.get("speaker_0") == np.ones((1, 3), dtype=np.float32),
    )
    assert vec_env.observations.get("agent") is None
    assert (
        str(next(iter(vec_env.observations)))
        == "('speaker_0', array([[1., 1., 1.]], dtype=float32))"
    )
    vec_env.close()


class ImageObsTestEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "image_obs_test_v0"}

    def __init__(self):
        self.possible_agents = ["pursuer_0", "pursuer_1"]
        self.agents = self.possible_agents.copy()
        self.render_mode = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        return {
            agent: np.zeros((7, 7, 3), dtype=np.uint8) for agent in self.agents
        }, {agent: {} for agent in self.agents}

    def step(self, actions):
        return (
            {agent: np.zeros((7, 7, 3), dtype=np.uint8) for agent in self.agents},
            dict.fromkeys(self.agents, 0.0),
            dict.fromkeys(self.agents, False),
            dict.fromkeys(self.agents, False),
            {agent: {} for agent in self.agents},
        )

    def observation_space(self, agent):
        return Box(low=0, high=255, shape=(7, 7, 3), dtype=np.uint8)

    def action_space(self, agent):
        return Discrete(2)


def test_observations_image():
    num_envs = 1
    env_fns = [ImageObsTestEnv for _ in range(num_envs)]
    vec_env = AsyncPettingZooVecEnv(env_fns)

    # Reset the environment to initialize observations
    vec_env.reset()

    # Create a test observation with known values
    test_obs = {
        agent: np.ones((1, 7, 7, 3), dtype=np.uint8) * 255 for agent in vec_env.agents
    }

    # Write the test observation to shared memory
    write_to_shared_memory(
        0,
        test_obs,
        vec_env._obs_buffer,
        vec_env._single_observation_spaces,
    )

    # Check if the values were correctly written to shared memory
    for agent in vec_env.agents:
        assert isinstance(vec_env.observations[agent], np.ndarray)
        assert vec_env.observations[agent].shape == (1, 7, 7, 3)
        assert np.all(vec_env.observations[agent] == 255)

    vec_env.close()


def test_write_to_shared_memory_dict_image():
    """Test writing dictionary observations with images to shared memory"""
    agents, obs_spaces, shared_memory, observations = make_observation_views(
        ComplexDictSpaceTestEnv
    )

    # Create test observation
    test_obs = {
        "agent_0": {
            "position": np.ones((3,), dtype=np.float32),
            "velocity": np.ones((2,), dtype=np.float32) * 2,
            "image": np.ones((16, 16, 3), dtype=np.uint8) * 100,
        },
        "other_agent_0": {
            "position": np.ones((3,), dtype=np.float32) * 3,
            "velocity": np.ones((2,), dtype=np.float32) * 4,
            "image": np.ones((16, 16, 3), dtype=np.uint8) * 200,
        },
    }

    # Write to shared memory
    write_to_shared_memory(0, test_obs, shared_memory, obs_spaces)

    # Check if values were correctly written
    for agent in agents:
        assert np.allclose(
            observations[agent]["position"][0],
            test_obs[agent]["position"],
        )
        assert np.allclose(
            observations[agent]["velocity"][0],
            test_obs[agent]["velocity"],
        )
        assert np.all(observations[agent]["image"][0] == test_obs[agent]["image"])


def test_write_to_shared_memory_tuple_image():
    """Test writing tuple observations with images to shared memory"""
    agents, obs_spaces, shared_memory, observations = make_observation_views(
        ComplexTupleSpaceTestEnv
    )

    # Create test observation
    test_obs = {
        "agent_0": (
            np.ones((3,), dtype=np.float32),
            np.ones((2,), dtype=np.float32) * 2,
            np.ones((16, 16, 3), dtype=np.uint8) * 100,
        ),
        "other_agent_0": (
            np.ones((3,), dtype=np.float32) * 3,
            np.ones((2,), dtype=np.float32) * 4,
            np.ones((16, 16, 3), dtype=np.uint8) * 200,
        ),
    }

    # Write to shared memory
    write_to_shared_memory(0, test_obs, shared_memory, obs_spaces)

    # Check if values were correctly written
    for agent in agents:
        assert np.allclose(observations[agent][0][0], test_obs[agent][0])
        assert np.allclose(observations[agent][1][0], test_obs[agent][1])
        assert np.all(observations[agent][2][0] == test_obs[agent][2])


dummy_action_spaces = {"agent_0": gym.spaces.Box(0, 1, (4,))}
dummy_observation_spaces = {"agent_0": gym.spaces.Box(0, 1, (4,))}


# Test for pz_vec_env.py
def test_vec_env_reset_not_implemented():
    vec_env = PettingZooVecEnv(
        3,
        dummy_observation_spaces,
        dummy_action_spaces,
        ["agent_0"],
    )
    with pytest.raises(NotImplementedError, match="Subclasses must implement reset"):
        vec_env.reset()


def test_vec_env_step_not_implemented():
    vec_env = PettingZooVecEnv(
        3,
        dummy_observation_spaces,
        dummy_action_spaces,
        ["agent_0"],
    )
    with pytest.raises(
        NotImplementedError, match="Subclasses must implement step_async"
    ):
        vec_env.step_async([])
    with pytest.raises(
        NotImplementedError, match="Subclasses must implement step_wait"
    ):
        vec_env.step_wait()


def test_vec_env_step_skips_nan_actions():
    class DummyStepVecEnv(PettingZooVecEnv):
        def __init__(self):
            super().__init__(
                3,
                dummy_observation_spaces,
                dummy_action_spaces,
                ["agent_0"],
            )
            self.passed_actions = None

        def step_async(self, actions):
            self.passed_actions = actions

        def step_wait(self):
            return {}, {}, {}, {}, {}

    vec_env = DummyStepVecEnv()
    actions = {"agent_0": np.array([np.nan, 1.0, np.nan], dtype=np.float32)}
    vec_env.step(actions)
    assert vec_env.passed_actions == [{}, {"agent_0": 1}, {}]


def test_vec_env_render():
    vec_env = PettingZooVecEnv(
        3,
        dummy_observation_spaces,
        dummy_action_spaces,
        ["agent_0"],
    )
    with pytest.raises(NotImplementedError):
        vec_env.render()


def test_vec_env_closed():
    vec_env = PettingZooVecEnv(
        3,
        dummy_observation_spaces,
        dummy_action_spaces,
        ["agent_0"],
    )
    vec_env.closed = True
    vec_env.close()


def test_vec_env_close_extras():
    vec_env = PettingZooVecEnv(
        3,
        dummy_observation_spaces,
        dummy_action_spaces,
        ["agent_0"],
    )
    vec_env.close_extras()


def test_vec_env_unwrapped():
    vec_env = PettingZooVecEnv(
        3,
        dummy_observation_spaces,
        dummy_action_spaces,
        ["agent_0"],
    )
    _ = vec_env.unwrapped


def test_delete_async_pz_vec_env():
    env_fns = [
        lambda: speaker_listener_like_env(continuous_actions=False) for _ in range(2)
    ]
    env = AsyncPettingZooVecEnv(env_fns)
    assert len(env.processes) > 0  # Ensure subprocesses were created
    for process in env.processes:
        assert process.is_alive()
    processes = env.processes
    env.close()
    for p in processes:
        assert not p.is_alive()


def test_dict_space_env():
    """Test environment with dictionary observation spaces"""
    num_envs = 2
    env_fns = [DictSpaceTestEnv for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # Check spaces
    for agent in env.agents:
        assert isinstance(env.single_observation_space(agent), spaces.Dict)
        assert isinstance(
            env.single_observation_space(agent).spaces["position"],
            spaces.Box,
        )
        assert isinstance(
            env.single_observation_space(agent).spaces["velocity"],
            spaces.Box,
        )

    # Test reset
    observations, infos = env.reset()
    for agent in env.agents:
        assert isinstance(observations[agent], dict)
        assert "position" in observations[agent]
        assert "velocity" in observations[agent]
        assert observations[agent]["position"].shape == (num_envs, 3)
        assert observations[agent]["velocity"].shape == (num_envs, 2)

    # Test step
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent in env.agents:
        assert isinstance(observations[agent], dict)
        assert "position" in observations[agent]
        assert "velocity" in observations[agent]
        assert observations[agent]["position"].shape == (num_envs, 3)
        assert observations[agent]["velocity"].shape == (num_envs, 2)
        assert rewards[agent].shape == (num_envs,)

    env.close()


def test_tuple_space_env():
    """Test environment with tuple observation spaces"""
    num_envs = 2
    env_fns = [TupleSpaceTestEnv for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # Check spaces
    for agent in env.agents:
        assert isinstance(env.single_observation_space(agent), spaces.Tuple)
        assert isinstance(env.single_observation_space(agent).spaces[0], spaces.Box)
        assert isinstance(env.single_observation_space(agent).spaces[1], spaces.Box)

    # Test reset
    observations, infos = env.reset()
    for agent in env.agents:
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 2
        assert observations[agent][0].shape == (num_envs, 3)
        assert observations[agent][1].shape == (num_envs, 2)

    # Test step
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent in env.agents:
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 2
        assert observations[agent][0].shape == (num_envs, 3)
        assert observations[agent][1].shape == (num_envs, 2)
        assert rewards[agent].shape == (num_envs,)

    env.close()


def test_complex_dict_space_env():
    """Test environment with complex dictionary observation spaces (containing images)"""
    num_envs = 2
    env_fns = [ComplexDictSpaceTestEnv for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # Check spaces
    for agent in env.agents:
        assert isinstance(env.single_observation_space(agent), spaces.Dict)
        assert isinstance(
            env.single_observation_space(agent).spaces["position"],
            spaces.Box,
        )
        assert isinstance(
            env.single_observation_space(agent).spaces["velocity"],
            spaces.Box,
        )
        assert isinstance(
            env.single_observation_space(agent).spaces["image"],
            spaces.Box,
        )

    # Test reset
    observations, infos = env.reset()
    for agent in env.agents:
        assert isinstance(observations[agent], dict)
        assert "position" in observations[agent]
        assert "velocity" in observations[agent]
        assert "image" in observations[agent]
        assert observations[agent]["position"].shape == (num_envs, 3)
        assert observations[agent]["velocity"].shape == (num_envs, 2)
        assert observations[agent]["image"].shape == (num_envs, 16, 16, 3)

    # Test step
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent in env.agents:
        assert isinstance(observations[agent], dict)
        assert "position" in observations[agent]
        assert "velocity" in observations[agent]
        assert "image" in observations[agent]
        assert observations[agent]["position"].shape == (num_envs, 3)
        assert observations[agent]["velocity"].shape == (num_envs, 2)
        assert observations[agent]["image"].shape == (num_envs, 16, 16, 3)
        assert rewards[agent].shape == (num_envs,)

    env.close()


def test_complex_tuple_space_env():
    """Test environment with complex tuple observation spaces (containing images)"""
    num_envs = 2
    env_fns = [ComplexTupleSpaceTestEnv for _ in range(num_envs)]
    env = AsyncPettingZooVecEnv(env_fns)

    # Check spaces
    for agent in env.agents:
        assert isinstance(env.single_observation_space(agent), spaces.Tuple)
        assert isinstance(env.single_observation_space(agent).spaces[0], spaces.Box)
        assert isinstance(env.single_observation_space(agent).spaces[1], spaces.Box)
        assert isinstance(env.single_observation_space(agent).spaces[2], spaces.Box)

    # Test reset
    observations, infos = env.reset()
    for agent in env.agents:
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 3
        assert observations[agent][0].shape == (num_envs, 3)
        assert observations[agent][1].shape == (num_envs, 2)
        assert observations[agent][2].shape == (num_envs, 16, 16, 3)

    # Test step
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    for agent in env.agents:
        assert isinstance(observations[agent], tuple)
        assert len(observations[agent]) == 3
        assert observations[agent][0].shape == (num_envs, 3)
        assert observations[agent][1].shape == (num_envs, 2)
        assert observations[agent][2].shape == (num_envs, 16, 16, 3)
        assert rewards[agent].shape == (num_envs,)

    env.close()


def test_write_to_shared_memory_dict():
    """Test writing dictionary observations to shared memory"""
    agents, obs_spaces, shared_memory, observations = make_observation_views(
        DictSpaceTestEnv
    )

    # Create test observation
    test_obs = {
        "agent_0": {
            "position": np.ones((3,), dtype=np.float32),
            "velocity": np.ones((2,), dtype=np.float32) * 2,
        },
        "other_agent_0": {
            "position": np.ones((3,), dtype=np.float32) * 3,
            "velocity": np.ones((2,), dtype=np.float32) * 4,
        },
    }

    # Write to shared memory
    write_to_shared_memory(0, test_obs, shared_memory, obs_spaces)

    # Check if values were correctly written
    for agent in agents:
        assert np.allclose(
            observations[agent]["position"][0],
            test_obs[agent]["position"],
        )
        assert np.allclose(
            observations[agent]["velocity"][0],
            test_obs[agent]["velocity"],
        )


def test_write_to_shared_memory_tuple():
    """Test writing tuple observations to shared memory"""
    agents, obs_spaces, shared_memory, observations = make_observation_views(
        TupleSpaceTestEnv
    )

    # Create test observation
    test_obs = {
        "agent_0": (
            np.ones((3,), dtype=np.float32),
            np.ones((2,), dtype=np.float32) * 2,
        ),
        "other_agent_0": (
            np.ones((3,), dtype=np.float32) * 3,
            np.ones((2,), dtype=np.float32) * 4,
        ),
    }

    # Write to shared memory
    write_to_shared_memory(0, test_obs, shared_memory, obs_spaces)

    # Check if values were correctly written
    for agent in agents:
        assert np.allclose(observations[agent][0][0], test_obs[agent][0])
        assert np.allclose(observations[agent][1][0], test_obs[agent][1])


def test_placeholder_dict_space():
    """Test placeholder values for dictionary observation spaces"""
    _, obs_spaces, _, _ = make_observation_views(DictSpaceTestEnv)

    placeholder = get_placeholder_value(
        agent="agent_0",
        transition_name="observation",
        obs_spaces=obs_spaces,
    )

    assert isinstance(placeholder, dict)
    assert "position" in placeholder
    assert "velocity" in placeholder
    assert placeholder["position"].shape == (3,)
    assert placeholder["velocity"].shape == (2,)


def test_placeholder_tuple_space():
    """Test placeholder values for tuple observation spaces"""
    _, obs_spaces, _, _ = make_observation_views(TupleSpaceTestEnv)

    placeholder = get_placeholder_value(
        agent="agent_0",
        transition_name="observation",
        obs_spaces=obs_spaces,
    )

    assert isinstance(placeholder, tuple)
    assert len(placeholder) == 2
    assert placeholder[0].shape == (3,)
    assert placeholder[1].shape == (2,)


# Helper function to create a replay buffer and add transitions
def create_replay_buffer_with_transitions(env, memory_size=10):
    buffer = MultiAgentReplayBuffer(
        memory_size=memory_size,
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=env.possible_agents,
    )
    env.reset()
    for _ in range(memory_size):
        actions = {
            agent: env.action_space(agent).sample() for agent in env.possible_agents
        }
        obs, rewards, dones, truncated, infos = env.step(actions)
        buffer.save_to_memory(obs, actions, rewards, obs, dones, is_vectorised=True)
    return buffer


@pytest.mark.parametrize("env_cls", [DictSpaceTestEnv, TupleSpaceTestEnv])
def test_replay_buffer_with_various_spaces(env_cls):
    env_fns = [env_cls for _ in range(2)]
    env = AsyncPettingZooVecEnv(env_fns)

    buffer = create_replay_buffer_with_transitions(env)
    batch_size = 5
    sampled_transitions = buffer.sample(batch_size)

    # Check that the sampled transitions have the correct structure
    for field, agent_data in zip(buffer.field_names, sampled_transitions, strict=False):
        for agent_id, data in agent_data.items():
            assert agent_id in env.possible_agents
            if field == "state":
                obs_space = env.single_observation_space(agent_id)
                if isinstance(obs_space, spaces.Dict):
                    for key in obs_space.spaces:
                        assert key in data
                        assert data[key].shape[0] == batch_size
                elif isinstance(obs_space, spaces.Tuple):
                    assert len(data) == len(obs_space.spaces)
                    for _i, sub_data in enumerate(data):
                        assert sub_data.shape[0] == batch_size
                else:
                    assert data.shape[0] == batch_size

    env.close()
