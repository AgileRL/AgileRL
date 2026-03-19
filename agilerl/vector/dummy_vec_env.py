"""Lightweight single-env wrapper that exposes the ``VectorEnv`` interface.

When a plain ``gymnasium.Env`` is passed to a Trainer, it is automatically
wrapped in a :class:`DummyVecEnv` so that all downstream code can assume
a uniform vectorized-environment API (``num_envs``,
``single_observation_space``, etc.) without branching.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Env, spaces
from gymnasium.vector.utils import batch_space


class DummyVecEnv:
    """Wraps a single :class:`gymnasium.Env` with a ``VectorEnv``-like API.

    Observations returned by :meth:`reset` and :meth:`step` always carry a
    leading batch dimension of size 1, and actions are expected to have the
    same leading dimension (which is stripped before forwarding to the
    underlying environment).

    :param env: The environment to wrap.
    :type env: gymnasium.Env
    """

    def __init__(self, env: Env) -> None:
        self._env = env
        self.num_envs: int = 1
        self.single_observation_space: spaces.Space = env.observation_space
        self.single_action_space: spaces.Space = env.action_space
        self.observation_space: spaces.Space = batch_space(env.observation_space, 1)
        self.action_space: spaces.Space = batch_space(env.action_space, 1)
        self.render_mode: str | None = getattr(env, "render_mode", None)
        self.spec = getattr(env, "spec", None)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return batched observation.

        :param seed: Random seed for the reset.
        :type seed: int | None
        :param options: Additional options for the reset.
        :type options: dict[str, Any] | None
        :returns: A tuple of ``(obs, info)`` with a leading batch dim on *obs*.
        :rtype: tuple[np.ndarray, dict[str, Any]]
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return np.expand_dims(obs, axis=0), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Take a step in the environment.

        :param action: Batched action array (shape ``(1, ...)``).
        :type action: np.ndarray
        :returns: A tuple of ``(obs, reward, terminated, truncated, info)``
            with leading batch dimensions.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]
        """
        scalar_action = action[0]
        if isinstance(self.single_action_space, spaces.Discrete):
            scalar_action = int(scalar_action)

        obs, reward, terminated, truncated, info = self._env.step(scalar_action)
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            info,
        )

    def render(self) -> Any:
        """Render the environment.

        :returns: Render output from the wrapped environment.
        :rtype: Any
        """
        return self._env.render()

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped environment."""
        return getattr(self._env, name)
