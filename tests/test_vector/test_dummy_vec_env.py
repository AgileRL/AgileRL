"""Comprehensive tests for DummyVecEnv and PzDummyVecEnv."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from gymnasium import spaces

from agilerl.vector.dummy_vec_env import DummyVecEnv, PzDummyVecEnv, _pz_placeholder


# ---------------------------------------------------------------------------
# Lightweight fake environments (no external dependency)
# ---------------------------------------------------------------------------


class FakeGymEnv:
    """Minimal gymnasium-compatible environment with Discrete actions."""

    def __init__(self, *, obs_shape: tuple = (4,), n_actions: int = 2) -> None:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape)
        self.action_space = spaces.Discrete(n_actions)
        self.render_mode = "rgb_array"
        self.spec = None
        self.custom_attr = "hello"
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        self._step_count = 0
        return np.ones(self.observation_space.shape, dtype=np.float32), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        obs = np.full(self.observation_space.shape, float(action), dtype=np.float32)
        done = self._step_count >= 5
        return obs, 1.0, done, False, {"action_taken": action}

    def render(self) -> str:
        return "frame"

    def close(self) -> None:
        pass


class FakeGymEnvContinuous:
    """Minimal gymnasium-compatible environment with Box actions."""

    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.render_mode = None
        self.spec = None

    def reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.array([action[0], 0.0], dtype=np.float32), 0.5, False, False, {}

    def render(self):
        return None

    def close(self):
        pass


class FakePzEnv:
    """Minimal PettingZoo ParallelEnv with configurable termination."""

    def __init__(
        self,
        agents: list[str] | None = None,
        *,
        terminate_after: int = 100,
    ) -> None:
        self.possible_agents = agents or ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._terminate_after = terminate_after
        self._step_count = 0

    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        self.agents = list(self.possible_agents)
        obs = {a: np.ones(3, dtype=np.float32) for a in self.possible_agents}
        return obs, {"reset": True}

    def step(self, actions: dict[str, Any]):
        self._step_count += 1
        done = self._step_count >= self._terminate_after
        obs = {
            a: np.full(3, float(actions.get(a, 0)), dtype=np.float32)
            for a in self.agents
        }
        rewards = {a: 1.0 for a in self.agents}
        terminated = {a: done for a in self.agents}
        truncated = {a: False for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, rewards, terminated, truncated, info

    def close(self):
        pass


# ---------------------------------------------------------------------------
# DummyVecEnv
# ---------------------------------------------------------------------------


class TestDummyVecEnvInit:
    def test_num_envs_is_one(self):
        env = DummyVecEnv(FakeGymEnv())
        assert env.num_envs == 1

    def test_single_spaces_match_underlying(self):
        inner = FakeGymEnv(obs_shape=(6,), n_actions=3)
        env = DummyVecEnv(inner)
        assert env.single_observation_space == inner.observation_space
        assert env.single_action_space == inner.action_space

    def test_batched_observation_space_shape(self):
        inner = FakeGymEnv(obs_shape=(4,))
        env = DummyVecEnv(inner)
        assert env.observation_space.shape == (1, 4)

    def test_batched_action_space(self):
        inner = FakeGymEnv(n_actions=5)
        env = DummyVecEnv(inner)
        assert env.action_space.shape == (1,)

    def test_render_mode_forwarded(self):
        env = DummyVecEnv(FakeGymEnv())
        assert env.render_mode == "rgb_array"

    def test_spec_forwarded(self):
        env = DummyVecEnv(FakeGymEnv())
        assert env.spec is None


class TestDummyVecEnvReset:
    def test_obs_has_batch_dim(self):
        env = DummyVecEnv(FakeGymEnv(obs_shape=(4,)))
        obs, info = env.reset()
        assert obs.shape == (1, 4)

    def test_obs_values(self):
        env = DummyVecEnv(FakeGymEnv())
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs[0], np.ones(4, dtype=np.float32))

    def test_info_returned(self):
        env = DummyVecEnv(FakeGymEnv())
        _, info = env.reset(seed=42)
        assert info == {"seed": 42}


class TestDummyVecEnvStep:
    def test_output_shapes(self):
        env = DummyVecEnv(FakeGymEnv(obs_shape=(4,)))
        env.reset()
        obs, reward, done, trunc, info = env.step(np.array([1]))
        assert obs.shape == (1, 4)
        assert reward.shape == (1,)
        assert done.shape == (1,)
        assert trunc.shape == (1,)

    def test_reward_value(self):
        env = DummyVecEnv(FakeGymEnv())
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0]))
        assert reward[0] == 1.0

    def test_discrete_action_cast_to_int(self):
        inner = MagicMock()
        inner.observation_space = spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = spaces.Discrete(3)
        inner.reset.return_value = (np.zeros(2), {})
        inner.step.return_value = (np.zeros(2), 0.0, False, False, {})

        env = DummyVecEnv(inner)
        env.reset()
        env.step(np.array([2.0]))
        passed_action = inner.step.call_args[0][0]
        assert isinstance(passed_action, int)
        assert passed_action == 2

    def test_continuous_action_not_cast(self):
        inner = FakeGymEnvContinuous()
        env = DummyVecEnv(inner)
        env.reset()
        obs, _, _, _, _ = env.step(np.array([[0.5]]))
        assert obs.shape == (1, 2)
        np.testing.assert_allclose(obs[0, 0], 0.5)

    def test_info_passthrough(self):
        env = DummyVecEnv(FakeGymEnv())
        env.reset()
        _, _, _, _, info = env.step(np.array([1]))
        assert info["action_taken"] == 1

    def test_done_flag(self):
        inner = (
            FakeGymEnv(terminate_after=1)
            if hasattr(FakeGymEnv, "terminate_after")
            else FakeGymEnv()
        )
        env = DummyVecEnv(inner)
        env.reset()
        for _ in range(4):
            env.step(np.array([0]))
        _, _, done, _, _ = env.step(np.array([0]))
        assert done[0] is True or done[0] == True  # noqa: E712


class TestDummyVecEnvMisc:
    def test_render_delegates(self):
        env = DummyVecEnv(FakeGymEnv())
        assert env.render() == "frame"

    def test_close_delegates(self):
        inner = MagicMock()
        inner.observation_space = spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = spaces.Discrete(2)
        env = DummyVecEnv(inner)
        env.close()
        inner.close.assert_called_once()

    def test_getattr_forwards(self):
        env = DummyVecEnv(FakeGymEnv())
        assert env.custom_attr == "hello"

    def test_getattr_raises_for_missing(self):
        env = DummyVecEnv(FakeGymEnv())
        with pytest.raises(AttributeError):
            _ = env.nonexistent_attribute


# ---------------------------------------------------------------------------
# _pz_placeholder helper
# ---------------------------------------------------------------------------


class TestPzPlaceholder:
    OBS_SPACES = {
        "a0": spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),
        "a1": spaces.Box(low=0, high=1, shape=(5,), dtype=np.float64),
    }

    def test_reward_is_nan(self):
        assert np.isnan(_pz_placeholder("a0", "reward", self.OBS_SPACES))

    def test_terminated_is_nan(self):
        assert np.isnan(_pz_placeholder("a0", "terminated", self.OBS_SPACES))

    def test_truncated_is_nan(self):
        assert np.isnan(_pz_placeholder("a0", "truncated", self.OBS_SPACES))

    def test_info_is_empty_dict(self):
        assert _pz_placeholder("a0", "info", self.OBS_SPACES) == {}

    def test_observation_shape_and_dtype(self):
        obs = _pz_placeholder("a1", "observation", self.OBS_SPACES)
        assert obs.shape == (5,)
        assert obs.dtype == np.float64
        np.testing.assert_array_equal(obs, np.zeros(5))


# ---------------------------------------------------------------------------
# PzDummyVecEnv
# ---------------------------------------------------------------------------


class TestPzDummyVecEnvInit:
    def test_num_envs_is_one(self):
        env = PzDummyVecEnv(FakePzEnv())
        assert env.num_envs == 1

    def test_agents_list(self):
        env = PzDummyVecEnv(FakePzEnv(["p1", "p2", "p3"]))
        assert env.agents == ["p1", "p2", "p3"]
        assert env.num_agents == 3

    def test_observation_space_callable(self):
        env = PzDummyVecEnv(FakePzEnv())
        space = env.single_observation_space("agent_0")
        assert space.shape == (3,)

    def test_action_space_callable(self):
        env = PzDummyVecEnv(FakePzEnv())
        space = env.single_action_space("agent_0")
        assert isinstance(space, spaces.Discrete)

    def test_batched_observation_space(self):
        env = PzDummyVecEnv(FakePzEnv())
        space = env.observation_space("agent_0")
        assert space.shape == (1, 3)

    def test_render_mode(self):
        env = PzDummyVecEnv(FakePzEnv())
        assert env.render_mode is None


class TestPzDummyVecEnvReset:
    def test_obs_is_per_agent_dict(self):
        env = PzDummyVecEnv(FakePzEnv())
        obs, info = env.reset()
        assert set(obs.keys()) == {"agent_0", "agent_1"}

    def test_obs_has_batch_dim(self):
        env = PzDummyVecEnv(FakePzEnv())
        obs, _ = env.reset()
        assert obs["agent_0"].shape == (1, 3)
        assert obs["agent_1"].shape == (1, 3)

    def test_obs_values(self):
        env = PzDummyVecEnv(FakePzEnv())
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs["agent_0"][0], np.ones(3, dtype=np.float32))

    def test_info_returned(self):
        env = PzDummyVecEnv(FakePzEnv())
        _, info = env.reset()
        assert info == {"reset": True}

    def test_seed_forwarded(self):
        inner = MagicMock()
        inner.possible_agents = ["a0"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = ({"a0": np.zeros(2)}, {})

        env = PzDummyVecEnv(inner)
        env.reset(seed=99)
        inner.reset.assert_called_once_with(seed=99, options=None)

    def test_inactive_agent_gets_zero_obs(self):
        inner = MagicMock()
        inner.possible_agents = ["a0", "a1"]
        inner.observation_space = lambda a: spaces.Box(
            low=0, high=1, shape=(3,), dtype=np.float32
        )
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = ({"a0": np.ones(3)}, {})

        env = PzDummyVecEnv(inner)
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs["a1"][0], np.zeros(3))


class TestPzDummyVecEnvStep:
    def _make_env(self, **kwargs) -> PzDummyVecEnv:
        return PzDummyVecEnv(FakePzEnv(**kwargs))

    def test_output_structure(self):
        env = self._make_env()
        env.reset()
        actions = {"agent_0": np.array([0]), "agent_1": np.array([1])}
        obs, rew, term, trunc, info = env.step(actions)

        assert set(obs.keys()) == {"agent_0", "agent_1"}
        assert set(rew.keys()) == {"agent_0", "agent_1"}
        assert set(term.keys()) == {"agent_0", "agent_1"}
        assert set(trunc.keys()) == {"agent_0", "agent_1"}

    def test_obs_batch_dim(self):
        env = self._make_env()
        env.reset()
        actions = {"agent_0": np.array([0]), "agent_1": np.array([1])}
        obs, _, _, _, _ = env.step(actions)
        assert obs["agent_0"].shape == (1, 3)

    def test_reward_batch_dim(self):
        env = self._make_env()
        env.reset()
        actions = {"agent_0": np.array([0]), "agent_1": np.array([0])}
        _, rew, _, _, _ = env.step(actions)
        assert rew["agent_0"].shape == (1,)
        assert rew["agent_0"][0] == 1.0

    def test_discrete_action_cast_to_int(self):
        inner = MagicMock()
        inner.possible_agents = ["a0"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(3)
        inner.reset.return_value = ({"a0": np.zeros(2)}, {})
        inner.step.return_value = (
            {"a0": np.zeros(2)},
            {"a0": 0.0},
            {"a0": False},
            {"a0": False},
            {"a0": {}},
        )

        env = PzDummyVecEnv(inner)
        env.reset()
        env.step({"a0": np.array([2.0])})
        passed = inner.step.call_args[0][0]
        assert isinstance(passed["a0"], int)
        assert passed["a0"] == 2

    def test_nan_action_filters_agent(self):
        inner = MagicMock()
        inner.possible_agents = ["a0", "a1"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = (
            {"a0": np.zeros(2), "a1": np.zeros(2)},
            {},
        )
        inner.step.return_value = (
            {"a0": np.zeros(2)},
            {"a0": 1.0},
            {"a0": False},
            {"a0": False},
            {},
        )

        env = PzDummyVecEnv(inner)
        env.reset()
        env.step({"a0": np.array([1]), "a1": np.array([np.nan])})

        passed = inner.step.call_args[0][0]
        assert "a0" in passed
        assert "a1" not in passed

    def test_inactive_agent_gets_nan_reward(self):
        inner = MagicMock()
        inner.possible_agents = ["a0", "a1"]
        inner.observation_space = lambda a: spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = (
            {"a0": np.zeros(2), "a1": np.zeros(2)},
            {},
        )
        inner.step.return_value = (
            {"a0": np.zeros(2)},
            {"a0": 1.0},
            {"a0": False},
            {"a0": False},
            {},
        )

        env = PzDummyVecEnv(inner)
        env.reset()
        _, rew, term, trunc, _ = env.step({"a0": np.array([0]), "a1": np.array([0])})

        assert np.isnan(rew["a1"][0])
        assert np.isnan(term["a1"][0])
        assert np.isnan(trunc["a1"][0])

    def test_inactive_agent_gets_zero_obs(self):
        inner = MagicMock()
        inner.possible_agents = ["a0", "a1"]
        inner.observation_space = lambda a: spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = (
            {"a0": np.zeros(2), "a1": np.zeros(2)},
            {},
        )
        inner.step.return_value = (
            {"a0": np.ones(2)},
            {"a0": 1.0},
            {"a0": False},
            {"a0": False},
            {},
        )

        env = PzDummyVecEnv(inner)
        env.reset()
        obs, _, _, _, _ = env.step({"a0": np.array([0]), "a1": np.array([0])})
        np.testing.assert_array_equal(obs["a1"][0], np.zeros(2))

    def test_auto_reset_on_all_done(self):
        env = self._make_env(terminate_after=1)
        env.reset()
        actions = {"agent_0": np.array([0]), "agent_1": np.array([0])}
        obs, _, term, _, _ = env.step(actions)

        assert term["agent_0"][0] == True  # noqa: E712
        # After auto-reset, obs should be the reset obs (all ones)
        np.testing.assert_array_equal(obs["agent_0"][0], np.ones(3, dtype=np.float32))

    def test_no_auto_reset_when_not_all_done(self):
        inner = MagicMock()
        inner.possible_agents = ["a0", "a1"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = (
            {"a0": np.zeros(2), "a1": np.zeros(2)},
            {},
        )
        step_obs = np.array([0.5, 0.5])
        inner.step.return_value = (
            {"a0": step_obs, "a1": step_obs},
            {"a0": 1.0, "a1": 1.0},
            {"a0": True, "a1": False},
            {"a0": False, "a1": False},
            {},
        )

        env = PzDummyVecEnv(inner)
        env.reset()
        obs, _, _, _, _ = env.step({"a0": np.array([0]), "a1": np.array([0])})
        np.testing.assert_array_equal(obs["a0"][0], step_obs)
        # reset should NOT have been called a second time
        assert inner.reset.call_count == 1


class TestPzDummyVecEnvStepAsyncWait:
    def test_step_async_then_wait(self):
        env = PzDummyVecEnv(FakePzEnv())
        env.reset()
        env.step_async([{"agent_0": 0, "agent_1": 1}])
        obs, rew, term, trunc, info = env.step_wait()

        assert obs["agent_0"].shape == (1, 3)
        assert rew["agent_0"].shape == (1,)

    def test_step_wait_without_async_raises(self):
        env = PzDummyVecEnv(FakePzEnv())
        env.reset()
        with pytest.raises(RuntimeError, match="step_async"):
            env.step_wait()

    def test_step_async_filters_missing_agents(self):
        inner = MagicMock()
        inner.possible_agents = ["a0", "a1"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.reset.return_value = (
            {"a0": np.zeros(2), "a1": np.zeros(2)},
            {},
        )
        inner.step.return_value = (
            {"a0": np.zeros(2), "a1": np.zeros(2)},
            {"a0": 1.0, "a1": 1.0},
            {"a0": False, "a1": False},
            {"a0": False, "a1": False},
            {},
        )

        env = PzDummyVecEnv(inner)
        env.reset()
        # Only pass action for a0; a1 missing -> gets np.nan -> filtered
        env.step_async([{"a0": 1}])
        env.step_wait()

        passed = inner.step.call_args[0][0]
        assert "a0" in passed
        assert "a1" not in passed


class TestPzDummyVecEnvMisc:
    def test_render_delegates(self):
        inner = MagicMock()
        inner.possible_agents = ["a0"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)
        inner.render.return_value = "rendered"

        env = PzDummyVecEnv(inner)
        assert env.render() == "rendered"

    def test_close_extras_delegates(self):
        inner = MagicMock()
        inner.possible_agents = ["a0"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)

        env = PzDummyVecEnv(inner)
        env.close_extras()
        inner.close.assert_called_once()

    def test_close_calls_close_extras(self):
        inner = MagicMock()
        inner.possible_agents = ["a0"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)

        env = PzDummyVecEnv(inner)
        env.close()
        inner.close.assert_called_once()
        assert env.closed is True

    def test_close_idempotent(self):
        inner = MagicMock()
        inner.possible_agents = ["a0"]
        inner.observation_space = lambda a: spaces.Box(low=0, high=1, shape=(2,))
        inner.action_space = lambda a: spaces.Discrete(2)

        env = PzDummyVecEnv(inner)
        env.close()
        env.close()
        inner.close.assert_called_once()
