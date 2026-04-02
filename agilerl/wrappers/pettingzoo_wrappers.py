from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
from gymnasium.utils import seeding
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

if TYPE_CHECKING:
    from gymnasium import spaces


class PettingZooAutoResetParallelWrapper(ParallelEnv):
    """Wrapper to automatically reset the environment when all agents terminate or truncate.

    :param env: The environment to wrap
    :type env: ParallelEnv[AgentID, ObsType, ActionType]
    """

    env: ParallelEnv[AgentID, ObsType, ActionType]
    metadata: dict[str, Any]
    possible_agents: list[AgentID]
    state_space: spaces.Space | None
    np_random: np.random.RandomState
    agents: list[AgentID]

    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType]) -> None:
        self.env = env
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents

        # Not every environment has the .state_space attribute implemented
        with contextlib.suppress(AttributeError):
            self.state_space = (
                self.env.state_space  # pyright: ignore[reportGeneralTypeIssues]
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        """Reset the environment and return the initial observations and info.

        :param seed: Random seed, defaults to None
        :type seed: int | None, optional
        :param options: Options dictionary, defaults to None
        :type options: dict | None, optional
        :return: Tuple of (observations, infos)
        :rtype: tuple[dict[str, np.ndarray], dict[str, Any]]
        """
        self.np_random, _ = seeding.np_random(seed)

        res, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        return res, info

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """Step the environment and return the observations, rewards, terminations, truncations, and info.

        :param actions: Actions dictionary
        :type actions: dict[str, ActionType]
        :return: Tuple of (observations, rewards, terminations, truncations, infos)
        :rtype: tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]
        """
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        if np.all(list(terminations.values()) or list(truncations.values())):
            obs, infos = self.env.reset()
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None | np.ndarray | str | list:
        """Render the environment.

        :return: Rendered environment
        :rtype: None | np.ndarray | str | list
        """
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        return self.env.close()

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

    def observation_space(self, agent: AgentID) -> spaces.Space:
        """Return the observation space for the given agent.

        :param agent: Agent ID
        :type agent: str
        :return: Observation space
        :rtype: spaces.Space
        """
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> spaces.Space:
        """Return the action space for the given agent.

        :param agent: Agent ID
        :type agent: str
        :return: Action space
        :rtype: spaces.Space
        """
        return self.env.action_space(agent)
