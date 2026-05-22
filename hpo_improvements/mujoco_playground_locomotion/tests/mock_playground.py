"""Deterministic mock of a MuJoCo Playground MJX env for wrapper unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from mujoco_playground import registry


@struct.dataclass
class MockPlaygroundState:
    """Minimal Playground-style state registered as a JAX pytree."""

    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    data: jax.Array
    metrics: dict = struct.field(pytree_node=False, default_factory=dict)


class _MockMjModel:
    actuator_ctrlrange = np.array([[-1.0, 1.0]] * 6, dtype=np.float32)


class MockPlaygroundEnv:
    """Single-env mock whose ``reset``/``step`` are deterministic in the PRNG key."""

    observation_size = 4
    mj_model = _MockMjModel()

    def __init__(
        self, *, episode_done_step: int = 5, reward_scale: float = 0.1
    ) -> None:
        self._episode_done_step = episode_done_step
        self._reward_scale = reward_scale

    @staticmethod
    def _key_to_id(key: jax.Array) -> jax.Array:
        """Map a PRNG key to a stable scalar id (unique across typical split keys)."""
        return jax.random.randint(key, (), 0, 1_000_000).astype(jnp.float32)

    def reset(self, key: jax.Array) -> MockPlaygroundState:
        """Return deterministic initial state derived from the PRNG key."""
        env_id = self._key_to_id(key)
        obs = jnp.arange(self.observation_size, dtype=jnp.float32) + env_id * 0.01
        data = jnp.array([env_id, 0.0], dtype=jnp.float32)
        return MockPlaygroundState(
            obs=obs,
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            data=data,
            metrics={},
        )

    def step(
        self, state: MockPlaygroundState, action: jax.Array
    ) -> MockPlaygroundState:
        """Advance one step with simple linear observation/reward dynamics."""
        step_count = state.data[1] + 1.0
        env_id = state.data[0]
        reward = jnp.sum(action) * jnp.float32(self._reward_scale)
        obs = state.obs + action[: self.observation_size] * jnp.float32(0.05)
        done = jnp.where(step_count >= self._episode_done_step, 1.0, 0.0)
        data = jnp.array([env_id, step_count], dtype=jnp.float32)
        return state.replace(
            obs=obs,
            reward=reward,
            done=done,
            data=data,
        )


def mock_playground_config(*, episode_length: int = 10) -> SimpleNamespace:
    """Config object returned by ``registry.get_default_config``."""
    return SimpleNamespace(impl="jax", episode_length=episode_length, sim_dt=0.01)


def install_mock_registry(
    monkeypatch: Any,
    *,
    episode_length: int = 10,
    episode_done_step: int = 5,
    reward_scale: float = 0.1,
) -> MockPlaygroundEnv:
    """Patch ``mujoco_playground.registry`` so wrappers load :class:`MockPlaygroundEnv`."""
    shared = MockPlaygroundEnv(
        episode_done_step=episode_done_step,
        reward_scale=reward_scale,
    )

    def _get_default_config(_name: str) -> SimpleNamespace:
        return mock_playground_config(episode_length=episode_length)

    def _load(_name: str, config: SimpleNamespace) -> MockPlaygroundEnv:
        return shared

    monkeypatch.setattr(registry, "get_default_config", _get_default_config)
    monkeypatch.setattr(registry, "load", _load)
    return shared
