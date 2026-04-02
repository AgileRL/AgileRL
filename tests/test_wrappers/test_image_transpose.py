"""Tests for agilerl.wrappers.image_transpose."""

from __future__ import annotations

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from agilerl.wrappers.image_transpose import (
    ImageTranspose,
    PettingZooImageTranspose,
    _transpose_obs,
    _transpose_space,
    is_channels_last,
    needs_image_transpose,
)


# ---------------------------------------------------------------------------
# is_channels_last
# ---------------------------------------------------------------------------


class TestIsChannelsLast:
    def test_hwc_image(self):
        assert is_channels_last(spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8))

    def test_chw_image(self):
        assert not is_channels_last(
            spaces.Box(0, 255, shape=(3, 84, 84), dtype=np.uint8)
        )

    def test_single_channel_hwc(self):
        assert is_channels_last(spaces.Box(0, 255, shape=(84, 84, 1), dtype=np.uint8))

    def test_single_channel_chw(self):
        assert not is_channels_last(
            spaces.Box(0, 255, shape=(1, 84, 84), dtype=np.uint8)
        )

    def test_stacked_frames_chw(self):
        assert not is_channels_last(
            spaces.Box(0, 255, shape=(4, 84, 84), dtype=np.uint8)
        )

    def test_stacked_frames_hwc(self):
        assert is_channels_last(spaces.Box(0, 255, shape=(84, 84, 4), dtype=np.uint8))

    def test_2d_not_image(self):
        assert not is_channels_last(spaces.Box(0, 1, shape=(4, 4), dtype=np.float32))

    def test_1d_not_image(self):
        assert not is_channels_last(spaces.Box(0, 1, shape=(8,), dtype=np.float32))

    def test_non_box_returns_false(self):
        assert not is_channels_last(spaces.Discrete(5))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# needs_image_transpose
# ---------------------------------------------------------------------------


class TestNeedsImageTranspose:
    def test_hwc_box(self):
        assert needs_image_transpose(spaces.Box(0, 255, (84, 84, 3), np.uint8))

    def test_chw_box(self):
        assert not needs_image_transpose(spaces.Box(0, 255, (3, 84, 84), np.uint8))

    def test_dict_with_hwc(self):
        s = spaces.Dict(
            {
                "image": spaces.Box(0, 255, (84, 84, 3), np.uint8),
                "velocity": spaces.Box(-1, 1, (3,), np.float32),
            }
        )
        assert needs_image_transpose(s)

    def test_dict_without_hwc(self):
        s = spaces.Dict(
            {
                "velocity": spaces.Box(-1, 1, (3,), np.float32),
            }
        )
        assert not needs_image_transpose(s)

    def test_tuple_with_hwc(self):
        s = spaces.Tuple(
            (
                spaces.Box(0, 255, (84, 84, 3), np.uint8),
                spaces.Discrete(5),
            )
        )
        assert needs_image_transpose(s)

    def test_discrete(self):
        assert not needs_image_transpose(spaces.Discrete(5))


# ---------------------------------------------------------------------------
# _transpose_space
# ---------------------------------------------------------------------------


class TestTransposeSpace:
    def test_box_3d(self):
        s = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        t = _transpose_space(s)
        assert t.shape == (3, 84, 84)

    def test_dict(self):
        s = spaces.Dict(
            {
                "img": spaces.Box(0, 255, (84, 84, 3), np.uint8),
                "vel": spaces.Box(-1, 1, (3,), np.float32),
            }
        )
        t = _transpose_space(s)
        assert t["img"].shape == (3, 84, 84)
        assert t["vel"].shape == (3,)

    def test_tuple(self):
        s = spaces.Tuple(
            (
                spaces.Box(0, 255, (84, 84, 3), np.uint8),
                spaces.Discrete(5),
            )
        )
        t = _transpose_space(s)
        assert t.spaces[0].shape == (3, 84, 84)
        assert isinstance(t.spaces[1], spaces.Discrete)

    def test_passthrough_non_image(self):
        s = spaces.Discrete(10)
        assert _transpose_space(s) is s


# ---------------------------------------------------------------------------
# _transpose_obs
# ---------------------------------------------------------------------------


class TestTransposeObs:
    def test_3d_array(self):
        space = spaces.Box(0, 255, (4, 6, 3), np.uint8)
        obs = np.zeros((4, 6, 3), dtype=np.uint8)
        result = _transpose_obs(obs, space)
        assert result.shape == (3, 4, 6)

    def test_dict_obs(self):
        space = spaces.Dict(
            {
                "img": spaces.Box(0, 255, (4, 6, 3), np.uint8),
                "vel": spaces.Box(-1, 1, (3,), np.float32),
            }
        )
        obs = {
            "img": np.zeros((4, 6, 3), dtype=np.uint8),
            "vel": np.zeros((3,), dtype=np.float32),
        }
        result = _transpose_obs(obs, space)
        assert result["img"].shape == (3, 4, 6)
        assert result["vel"].shape == (3,)

    def test_passthrough(self):
        space = spaces.Discrete(5)
        obs = 3
        assert _transpose_obs(obs, space) == 3


# ---------------------------------------------------------------------------
# ImageTranspose (Gymnasium wrapper)
# ---------------------------------------------------------------------------


class TestImageTranspose:
    def test_hwc_env(self):
        env = gym.make("CarRacing-v3")
        wrapped = ImageTranspose(env)
        assert wrapped.observation_space.shape == (3, 96, 96)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (3, 96, 96)
        env.close()

    def test_transpose_values_roundtrip(self):
        env = gym.make("CarRacing-v3")
        wrapped = ImageTranspose(env)
        obs, _ = wrapped.reset(seed=42)
        roundtripped = obs.transpose(1, 2, 0)
        assert roundtripped.shape == (96, 96, 3)
        wrapped.close()

    def test_step_returns_transposed(self):
        env = gym.make("CarRacing-v3")
        wrapped = ImageTranspose(env)
        wrapped.reset(seed=42)
        action = wrapped.action_space.sample()
        obs, _, _, _, _ = wrapped.step(action)
        assert obs.shape == (3, 96, 96)
        wrapped.close()

    def test_non_image_passthrough(self):
        env = gym.make("CartPole-v1")
        wrapped = ImageTranspose(env)
        assert wrapped.observation_space.shape == (4,)
        obs, _ = wrapped.reset(seed=42)
        assert obs.shape == (4,)
        wrapped.close()

    def test_dict_observation_space(self):
        space = spaces.Dict(
            {
                "image": spaces.Box(0, 255, (84, 84, 3), np.uint8),
                "vector": spaces.Box(-1, 1, (10,), np.float32),
            }
        )
        env = MagicMock(spec=gym.Env)
        env.observation_space = space
        env.action_space = spaces.Discrete(4)
        wrapped = ImageTranspose(env)
        assert wrapped.observation_space["image"].shape == (3, 84, 84)
        assert wrapped.observation_space["vector"].shape == (10,)


# ---------------------------------------------------------------------------
# PettingZooImageTranspose
# ---------------------------------------------------------------------------


def _make_mock_pz_env(
    obs_shape: tuple[int, ...] = (84, 84, 3),
    n_agents: int = 2,
):
    """Create a mock PettingZoo ParallelEnv with image observations."""
    agents = [f"agent_{i}" for i in range(n_agents)]
    obs_space = spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)
    act_space = spaces.Discrete(5)

    env = MagicMock()
    env.possible_agents = agents
    env.agents = list(agents)
    env.metadata = {"render_modes": ["human"]}
    env.observation_space = MagicMock(side_effect=lambda a: obs_space)
    env.action_space = MagicMock(side_effect=lambda a: act_space)
    env.unwrapped = env

    def mock_reset(seed=None, options=None):
        obs = {a: np.zeros(obs_shape, dtype=np.uint8) for a in agents}
        info = {a: {} for a in agents}
        return obs, info

    def mock_step(actions):
        obs = {a: np.ones(obs_shape, dtype=np.uint8) for a in agents}
        rewards = {a: 0.0 for a in agents}
        terms = {a: False for a in agents}
        truncs = {a: False for a in agents}
        infos = {a: {} for a in agents}
        return obs, rewards, terms, truncs, infos

    env.reset = MagicMock(side_effect=mock_reset)
    env.step = MagicMock(side_effect=mock_step)
    env.close = MagicMock()
    env.render = MagicMock(return_value=None)
    return env


class TestPettingZooImageTranspose:
    def test_hwc_obs_spaces(self):
        env = _make_mock_pz_env(obs_shape=(84, 84, 3))
        wrapped = PettingZooImageTranspose(env)
        for agent in wrapped.possible_agents:
            assert wrapped.observation_space(agent).shape == (3, 84, 84)

    def test_reset_transposes(self):
        env = _make_mock_pz_env(obs_shape=(84, 84, 3))
        wrapped = PettingZooImageTranspose(env)
        obs, info = wrapped.reset(seed=42)
        for agent in wrapped.possible_agents:
            assert obs[agent].shape == (3, 84, 84)

    def test_step_transposes(self):
        env = _make_mock_pz_env(obs_shape=(84, 84, 3))
        wrapped = PettingZooImageTranspose(env)
        wrapped.reset(seed=42)
        actions = {a: 0 for a in wrapped.possible_agents}
        obs, rewards, terms, truncs, infos = wrapped.step(actions)
        for agent in wrapped.possible_agents:
            assert obs[agent].shape == (3, 84, 84)

    def test_action_space_delegated(self):
        env = _make_mock_pz_env()
        wrapped = PettingZooImageTranspose(env)
        for agent in wrapped.possible_agents:
            assert isinstance(wrapped.action_space(agent), spaces.Discrete)

    def test_close_delegated(self):
        env = _make_mock_pz_env()
        wrapped = PettingZooImageTranspose(env)
        wrapped.close()
        env.close.assert_called_once()

    def test_render_delegated(self):
        env = _make_mock_pz_env()
        wrapped = PettingZooImageTranspose(env)
        wrapped.render()
        env.render.assert_called_once()

    def test_getattr_delegates_unknown_attrs(self):
        env = _make_mock_pz_env()
        env.state_space = spaces.Box(0, 1, shape=(10,))
        wrapped = PettingZooImageTranspose(env)
        assert wrapped.state_space is env.state_space

    def test_getattr_raises_for_missing(self):
        env = _make_mock_pz_env()
        # Replace the inner env with a plain object so missing attrs
        # actually raise AttributeError (MagicMock auto-creates them).
        wrapped = PettingZooImageTranspose(env)
        wrapped.env = type("StubEnv", (), {})()
        with pytest.raises(AttributeError):
            _ = wrapped.nonexistent_attribute
