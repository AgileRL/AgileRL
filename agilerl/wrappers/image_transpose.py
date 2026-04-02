"""Environment wrappers that transpose image observations from HWC to CHW.

Provides :class:`ImageTranspose` for Gymnasium environments and
:class:`PettingZooImageTranspose` for PettingZoo parallel environments.
Both wrappers unconditionally transpose 3-D Box observation subspaces
from ``(H, W, C)`` to ``(C, H, W)`` format.

Use :func:`needs_image_transpose` to check whether an observation space
would benefit from wrapping before applying the wrapper.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from agilerl.typing import NumpyObsType


def is_channels_last(space: spaces.Box) -> bool:
    """Detect whether a 3-D Box space is in channels-last (HWC) format.

    Uses a simple heuristic: if the smallest dimension is last, the
    observation is assumed to be ``(H, W, C)`` — i.e. channels-last.

    :param space: A gymnasium Box space.
    :type space: spaces.Box
    :returns: ``True`` when the space looks like a channels-last image.
    :rtype: bool
    """
    if not (isinstance(space, spaces.Box) and len(space.shape) == 3):
        return False
    return int(np.argmin(space.shape)) == 2


def needs_image_transpose(observation_space: spaces.Space) -> bool:
    """Check whether *any* image subspace requires a channels-last → first transpose.

    Recursively inspects :class:`~gymnasium.spaces.Dict` and
    :class:`~gymnasium.spaces.Tuple` spaces.

    :param observation_space: The observation space to inspect.
    :type observation_space: spaces.Space
    :returns: ``True`` if at least one Box subspace is channels-last.
    :rtype: bool
    """
    if isinstance(observation_space, spaces.Box):
        return is_channels_last(observation_space)
    if isinstance(observation_space, spaces.Dict):
        return any(needs_image_transpose(s) for s in observation_space.spaces.values())
    if isinstance(observation_space, spaces.Tuple):
        return any(needs_image_transpose(s) for s in observation_space.spaces)
    return False


def _transpose_space(space: spaces.Space) -> spaces.Space:
    """Return a copy of *space* with all 3-D Box subspaces transposed to CHW.

    :param space: Space to transpose
    :type space: spaces.Space
    :return: Transposed space
    :rtype: spaces.Space
    """
    if isinstance(space, spaces.Box) and len(space.shape) == 3:
        low = space.low.transpose(2, 0, 1)
        high = space.high.transpose(2, 0, 1)
        return spaces.Box(low=low, high=high, dtype=space.dtype)

    if isinstance(space, spaces.Dict):
        return spaces.Dict(
            {key: _transpose_space(s) for key, s in space.spaces.items()}
        )

    if isinstance(space, spaces.Tuple):
        return spaces.Tuple(tuple(_transpose_space(s) for s in space.spaces))

    return space


def _transpose_obs(
    observation: NumpyObsType, original_space: spaces.Space
) -> NumpyObsType:
    """Transpose 3-D observations from HWC to CHW.

    :param observation: Observation
    :type observation: NumpyObsType
    :param original_space: Original observation space
    :type original_space: spaces.Space
    :return: Transposed observation
    :rtype: np.ndarray
    """
    if isinstance(original_space, spaces.Box) and len(original_space.shape) == 3:
        return np.asarray(observation).transpose(2, 0, 1)

    if isinstance(original_space, spaces.Dict):
        return {
            key: _transpose_obs(observation[key], original_space[key])
            for key in observation
        }

    if isinstance(original_space, spaces.Tuple):
        return tuple(
            _transpose_obs(o, s)
            for o, s in zip(observation, original_space.spaces, strict=True)
        )

    return observation


class ImageTranspose(gym.ObservationWrapper):
    """Transpose image observations from channels-last (HWC) to channels-first (CHW).

    Example::

        import gymnasium as gym
        from agilerl.wrappers.image_transpose import ImageTranspose

        env = gym.make("ALE/Pong-v5")   # obs shape (210, 160, 3) — HWC
        env = ImageTranspose(env)        # obs shape (3, 210, 160) — CHW

    :param env: The gymnasium environment to wrap.
    :type env: gym.Env
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._original_obs_space = env.observation_space
        self.observation_space = _transpose_space(self._original_obs_space)

    def observation(self, observation: NumpyObsType) -> NumpyObsType:
        return _transpose_obs(observation, self._original_obs_space)


class PettingZooImageTranspose(ParallelEnv):
    """Transpose image observations for PettingZoo parallel environments.

    Behaves identically to :class:`ImageTranspose` but wraps a
    :class:`~pettingzoo.utils.env.ParallelEnv` instead.  Per-agent
    observation spaces are transposed individually.

    Example::

        from agilerl.wrappers.image_transpose import PettingZooImageTranspose

        env = my_parallel_env()
        env = PettingZooImageTranspose(env)

    :param env: A PettingZoo parallel environment.
    :type env: ParallelEnv
    """

    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType]) -> None:
        self.env = env
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents

        self._original_obs_spaces: dict[AgentID, spaces.Space] = {}
        self._transposed_obs_spaces: dict[AgentID, spaces.Space] = {}

        for agent in self.possible_agents:
            original = env.observation_space(agent)
            self._original_obs_spaces[agent] = original
            self._transposed_obs_spaces[agent] = _transpose_space(original)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def observation_space(self, agent: AgentID) -> spaces.Space:
        """Return the transposed observation space for the given agent.

        :param agent: Agent ID
        :type agent: AgentID
        :return: Transposed observation space
        :rtype: spaces.Space
        """
        return self._transposed_obs_spaces[agent]

    def action_space(self, agent: AgentID) -> spaces.Space:
        """Return the action space for the given agent.

        :param agent: Agent ID
        :type agent: AgentID
        :return: Action space
        :rtype: spaces.Space
        """
        return self.env.action_space(agent)

    @property
    def unwrapped(self) -> ParallelEnv:
        """Return the unwrapped environment.

        :return: Unwrapped environment
        :rtype: ParallelEnv
        """
        return self.env.unwrapped

    @property
    def state(self) -> np.ndarray:
        """Return the state of the environment.

        :return: State of the environment
        :rtype: np.ndarray
        """
        return self.env.state

    def _transpose_agent_obs(
        self, observations: dict[AgentID, NumpyObsType]
    ) -> dict[AgentID, NumpyObsType]:
        """Transpose the observations for the given agents.

        :param observations: Observations
        :type observations: dict[AgentID, NumpyObsType]
        :return: Transposed observations
        :rtype: dict[AgentID, NumpyObsType]
        """
        return {
            agent: _transpose_obs(obs, self._original_obs_spaces[agent])
            for agent, obs in observations.items()
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, NumpyObsType], dict[AgentID, dict]]:
        """Reset the environment and return the initial observations and info.

        :param seed: Random seed
        :type seed: int | None
        :param options: Options
        :type options: dict | None
        :return: Tuple of (observations, info)
        :rtype: tuple[dict[AgentID, NumpyObsType], dict[AgentID, dict]]
        """
        self.np_random, _ = seeding.np_random(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        return self._transpose_agent_obs(obs), info

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, NumpyObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """Step the environment and return the observations, rewards, terminations, truncations, and info.

        :param actions: Actions
        :type actions: dict[AgentID, ActionType]
        :return: Tuple of (observations, rewards, terminations, truncations, info)
        :rtype: tuple[dict[AgentID, ObsType], dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict]]
        """
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.agents = self.env.agents
        return (
            self._transpose_agent_obs(obs),
            rewards,
            terminations,
            truncations,
            infos,
        )

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def close(self) -> None:
        return self.env.close()
