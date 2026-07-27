"""Lightweight single-env wrappers that expose vectorized-env interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
from gymnasium import Env, spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from agilerl.typing import (
    ActionType,
    ArrayOrTensor,
    InfosDict,
    NumpyObsType,
    PzStepReturn,
)
from agilerl.vector.pz_vec_env import PettingZooVecEnv

if TYPE_CHECKING:
    from gymnasium.core import RenderFrame
    from pettingzoo import ParallelEnv


class DummyVecEnv(VectorEnv):
    """Wraps a single :class:`gymnasium.Env` with a ``VectorEnv``-like API.

    Observations returned by :meth:`reset` and :meth:`step` always carry a
    leading batch dimension of size 1, and actions are expected to have the
    same leading dimension (which is stripped before forwarding to the
    underlying environment).

    Episodes auto-reset following the gymnasium >= 1.0 next-step convention
    (matching :class:`gymnasium.vector.SyncVectorEnv`): the step after a
    termination/truncation resets the environment and returns the reset
    observation with zero reward and both done flags ``False``.

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
        self._autoreset = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """Reset the environment and return batched observation.

        :param seed: Random seed for the reset.
        :type seed: int | None
        :param options: Additional options for the reset.
        :type options: dict[str, Any] | None
        :returns: A tuple of ``(obs, info)`` with a leading batch dim on *obs*.
        :rtype: tuple[npt.NDArray, dict[str, Any]]
        """
        obs, info = self._env.reset(seed=seed, options=options)
        self._autoreset = False
        return np.expand_dims(obs, axis=0), info

    def step(
        self, actions: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, dict[str, Any]]:
        """Take a step in the environment.

        If the previous step ended the episode, the environment is reset
        instead (next-step autoreset) and the reset observation is returned
        with zero reward and ``False`` done flags.

        :param actions: Batched action array (shape ``(1, ...)``).
        :type actions: npt.NDArray
        :returns: A tuple of ``(obs, reward, terminated, truncated, info)``
            with leading batch dimensions.
        :rtype: tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, dict[str, Any]]
        """
        if self._autoreset:
            obs, info = self._env.reset()
            self._autoreset = False
            return (
                np.expand_dims(obs, axis=0),
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                info,
            )

        scalar_action = actions[0]
        if isinstance(self.single_action_space, spaces.Discrete):
            scalar_action = int(scalar_action)

        obs, reward, terminated, truncated, info = self._env.step(scalar_action)
        self._autoreset = bool(terminated) or bool(truncated)
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            info,
        )

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Render the environment.

        :returns: Render output from the wrapped environment.
        :rtype: tuple[gymnasium.core.RenderFrame, ...] | None
        """
        frame = self._env.render()
        if frame is None:
            return None
        # A single frame becomes a length-1 tuple; a frame list (rgb_array_list)
        # is spread so each element is a RenderFrame, per the VectorEnv contract.
        return tuple(frame) if isinstance(frame, list) else (frame,)

    def close(self, **kwargs: Any) -> None:
        """Close the wrapped environment."""
        self._env.close()

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- forwards arbitrary attributes from the wrapped env
        """Forward attribute access to the wrapped environment."""
        return getattr(self._env, name)


@overload
def _pz_placeholder(
    agent: str,
    name: Literal["observation"],
    obs_spaces: dict[str, spaces.Space],
) -> npt.NDArray: ...


@overload
def _pz_placeholder(
    agent: str,
    name: str,
    obs_spaces: dict[str, spaces.Space],
) -> float | dict[str, Any] | npt.NDArray: ...


def _pz_placeholder(
    agent: str,
    name: str,
    obs_spaces: dict[str, spaces.Space],
) -> float | dict[str, Any] | npt.NDArray:
    """Return a NaN/zero placeholder for an inactive PettingZoo agent."""
    if name in ("reward", "terminated", "truncated"):
        return np.nan
    if name == "info":
        return {}
    space = obs_spaces[agent]
    assert space.shape is not None  # placeholder only used for flat obs spaces
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
    ) -> tuple[dict[str, NumpyObsType], InfosDict]:
        """Reset the environment and return batched observations.

        :param seed: Random seed for the reset.
        :type seed: int | None
        :param options: Additional options forwarded to the underlying env.
        :type options: dict[str, Any] | None
        :returns: ``(obs, info)`` where *obs* is a dict of arrays with shape
            ``(1, ...)``.
        :rtype: tuple[dict[str, NumpyObsType], InfosDict]
        """
        obs, info = self._env.reset(seed=seed, options=options)

        batched_obs: dict[str, NumpyObsType] = {}
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
        actions: Mapping[str, ArrayOrTensor],
        *args: Any,
        **kwargs: Any,
    ) -> PzStepReturn:
        """Take a step in the environment.

        :param actions: Dict of batched actions per agent, each with shape
            ``(1, ...)``.  Values may be arrays or tensors (converted via
            ``np.asarray``).  NaN actions are filtered (agent treated as inactive).
        :type actions: Mapping[str, ArrayOrTensor]
        :returns: ``(obs, rewards, terminated, truncated, info)`` with leading
            batch dimension of 1 on all per-agent arrays.
        :rtype: PzStepReturn
        """
        # Strip batch dimension and filter NaN (inactive) agents
        scalar_actions: dict[str, ActionType] = {}
        for agent_id, action in actions.items():
            act = np.asarray(action[0])
            if np.isnan(act).all():
                continue
            if isinstance(self._single_action_spaces[agent_id], spaces.Discrete):
                act = int(act.flat[0])

            scalar_actions[agent_id] = act

        obs, reward, terminated, truncated, info = self._env.step(scalar_actions)

        # Batch all outputs, filling placeholders for inactive agents
        batched_obs: dict[str, NumpyObsType] = {}
        batched_reward: dict[str, npt.NDArray] = {}
        batched_terminated: dict[str, npt.NDArray] = {}
        batched_truncated: dict[str, npt.NDArray] = {}

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

    def step_async(self, actions: list[dict[str, ActionType]]) -> None:
        """Store actions for :meth:`step_wait` (synchronous passthrough).

        :param actions: List of dictionaries of length num_envs, each sub dictionary contains
            actions for each agent in a given environment
        :type actions: list[dict[str, ActionType]]

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
            actions_dict: dict[str, npt.NDArray] = {}
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

    def render(self) -> None | npt.NDArray | str | list:
        """Render the underlying environment.

        :returns: Render output from the wrapped environment.
        :rtype: None | npt.NDArray | str | list
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
