"""Lightweight single-env wrappers that expose vectorized-env interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium import Env, spaces
from gymnasium.vector.utils import batch_space

from agilerl.typing import PzStepReturn
from agilerl.vector.pz_vec_env import PettingZooVecEnv

if TYPE_CHECKING:
    from pettingzoo import ParallelEnv


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


def _pz_placeholder(
    agent: str,
    name: str,
    obs_spaces: dict[str, spaces.Space],
) -> Any:
    """Return a NaN/zero placeholder for an inactive PettingZoo agent.

    Mirrors the convention used by :class:`AsyncPettingZooVecEnv`.
    """
    if name in ("reward", "terminated", "truncated"):
        return np.nan
    if name == "info":
        return {}
    space = obs_spaces[agent]
    return np.zeros(space.shape, dtype=space.dtype)


class PzDummyVecEnv(PettingZooVecEnv):
    """Wraps a single PettingZoo :class:`ParallelEnv` with a vectorized API.

    Observations, rewards, terminations, and truncations returned by
    :meth:`reset` and :meth:`step` always carry a leading batch dimension of
    size 1 per agent.  Actions are expected to have the same leading
    dimension, which is stripped before forwarding to the underlying
    environment.

    Inactive agents receive NaN rewards/dones and zero-filled observations,
    matching the convention used by :class:`AsyncPettingZooVecEnv`.

    :param env: A PettingZoo ``ParallelEnv`` instance.
    :type env: pettingzoo.ParallelEnv
    """

    def __init__(self, env: ParallelEnv) -> None:
        self._env = env

        possible_agents: list[str] = env.possible_agents
        obs_spaces: dict[str, spaces.Space] = {
            agent: env.observation_space(agent) for agent in possible_agents
        }
        act_spaces: dict[str, spaces.Space] = {
            agent: env.action_space(agent) for agent in possible_agents
        }

        super().__init__(
            num_envs=1,
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            possible_agents=possible_agents,
            render_mode=getattr(env, "render_mode", None),
        )
        self._pending_actions = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment and return batched observations.

        :param seed: Random seed for the reset.
        :type seed: int | None
        :param options: Additional options forwarded to the underlying env.
        :type options: dict[str, Any] | None
        :returns: ``(obs, info)`` where *obs* is a dict of arrays with shape
            ``(1, ...)``.
        :rtype: tuple[dict[str, np.ndarray], dict[str, Any]]
        """
        obs, info = self._env.reset(seed=seed, options=options)

        batched_obs: dict[str, np.ndarray] = {}
        for agent in self.agents:
            if agent in obs:
                batched_obs[agent] = np.expand_dims(np.asarray(obs[agent]), axis=0)
            else:
                batched_obs[agent] = np.expand_dims(
                    _pz_placeholder(
                        agent, "observation", self._single_observation_spaces
                    ),
                    axis=0,
                )

        return batched_obs, info

    def step(
        self,
        actions: dict[str, np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> PzStepReturn:
        """Take a step in the environment.

        :param actions: Dict of batched actions per agent, each with shape
            ``(1, ...)``.  NaN actions are filtered (agent treated as inactive).
        :type actions: dict[str, np.ndarray]
        :returns: ``(obs, rewards, terminated, truncated, info)`` with leading
            batch dimension of 1 on all per-agent arrays.
        :rtype: PzStepReturn
        """
        # Strip batch dimension and filter NaN (inactive) agents
        scalar_actions: dict[str, Any] = {}
        for agent_id, action in actions.items():
            act = np.asarray(action[0])
            if np.isnan(act).all():
                continue
            if isinstance(self._single_action_spaces[agent_id], spaces.Discrete):
                act = int(act.flat[0])

            scalar_actions[agent_id] = act

        obs, reward, terminated, truncated, info = self._env.step(scalar_actions)

        # Batch all outputs, filling placeholders for inactive agents
        batched_obs: dict[str, np.ndarray] = {}
        batched_reward: dict[str, np.ndarray] = {}
        batched_terminated: dict[str, np.ndarray] = {}
        batched_truncated: dict[str, np.ndarray] = {}

        for agent in self.agents:
            if agent in obs:
                batched_obs[agent] = np.expand_dims(
                    np.asarray(obs[agent]),
                    axis=0,
                )
            else:
                batched_obs[agent] = np.expand_dims(
                    _pz_placeholder(
                        agent,
                        "observation",
                        self._single_observation_spaces,
                    ),
                    axis=0,
                )

            batched_reward[agent] = np.array(
                [reward[agent] if agent in reward else np.nan],
            )
            batched_terminated[agent] = np.array(
                [terminated[agent] if agent in terminated else np.nan],
            )
            batched_truncated[agent] = np.array(
                [truncated[agent] if agent in truncated else np.nan],
            )

        # Auto-reset when all agents are done
        if all(
            t | tr
            for t, tr in zip(terminated.values(), truncated.values(), strict=False)
        ):
            reset_obs, _reset_info = self._env.reset()
            for agent in self.agents:
                if agent in reset_obs:
                    batched_obs[agent] = np.expand_dims(
                        np.asarray(reset_obs[agent]),
                        axis=0,
                    )

        return batched_obs, batched_reward, batched_terminated, batched_truncated, info

    def step_async(self, actions: list[dict[str, Any]]) -> None:
        """Store actions for :meth:`step_wait` (synchronous passthrough).

        :param actions: List of dictionaries of length num_envs, each sub dictionary contains
            actions for each agent in a given environment
        :type actions: list[dict[str, Any]]

        :raises RuntimeError: If :meth:`step_async` is called before :meth:`step_wait`.
        """
        self._pending_actions = actions

    def step_wait(self, timeout: float | None = None) -> PzStepReturn:
        """Execute the step stored by :meth:`step_async`.

        :param timeout: Number of seconds before the call to :meth:`step_wait` times out. If `None`, the call to :meth:`step_wait` never times out, defaults to 0
        :type timeout: int | float | None, optional

        :return: Tuple of observations, rewards, terminated, truncated, infos
        :rtype: PzStepReturn
        """
        if self._pending_actions is not None:
            actions_dict: dict[str, np.ndarray] = {}
            for agent in self.agents:
                vals = [
                    env_actions.get(agent, np.nan)
                    for env_actions in self._pending_actions
                ]
                actions_dict[agent] = np.array(vals)
            del self._pending_actions
            return self.step(actions_dict)
        msg = "step_async() must be called before step_wait()"
        raise RuntimeError(msg)

    def render(self) -> Any:
        """Render the underlying environment.

        :returns: Render output from the wrapped environment.
        :rtype: Any
        """
        return self._env.render()

    def close_extras(
        self,
        *,
        timeout: float | None = None,
        terminate: bool = False,
        **kwargs: Any,
    ) -> None:
        """Close the wrapped environment.

        :param timeout: Number of seconds before the call to :meth:`close_extras` times out. If `None`, the call to :meth:`close_extras` never times out, defaults to 0
        :type timeout: int | float | None, optional
        :param terminate: Whether to terminate the environment, defaults to False
        :type terminate: bool, optional
        :param **kwargs: Additional keyword arguments to pass to the underlying environment's close method
        :type **kwargs: Any
        """
        self._env.close()
