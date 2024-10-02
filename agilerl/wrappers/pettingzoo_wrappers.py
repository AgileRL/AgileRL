from __future__ import annotations

import types

import gymnasium.spaces
import numpy as np
from gymnasium.utils import seeding
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from agilerl.utils.multiprocessing_env import SubprocVecEnv


class PettingZooParallelWrapper(ParallelEnv):
    def __init__(
        self, env: ParallelEnv[AgentID, ObsType, ActionType] | types.ModuleType
    ):
        if not isinstance(env, ParallelEnv):
            env = env.parallel_env()
        self.env = env
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = (
                self.env.state_space  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            pass

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.np_random, _ = seeding.np_random(seed)

        res, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        return res, info

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def close(self) -> None:
        return self.env.close()

    @property
    def unwrapped(self) -> ParallelEnv:
        return self.env.unwrapped

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)


class PettingZooAutoResetParallelWrapper(ParallelEnv):
    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType]):
        self.env = env
        self.metadata = env.metadata
        self.possible_agents = env.possible_agents

        # Not every environment has the .state_space attribute implemented
        try:
            self.state_space = (
                self.env.state_space  # pyright: ignore[reportGeneralTypeIssues]
            )
        except AttributeError:
            pass

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.np_random, _ = seeding.np_random(seed)

        res, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        return res, info

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        if np.all(list(terminations.values())) | np.any(list(truncations.values())):
            obs, infos = self.env.reset()
        return obs, rewards, terminations, truncations, infos

    def render(self) -> None | np.ndarray | str | list:
        return self.env.render()

    def close(self) -> None:
        return self.env.close()

    @property
    def unwrapped(self) -> ParallelEnv:
        return self.env.unwrapped

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.action_space(agent)


class PettingZooVectorizationParallelWrapper(PettingZooAutoResetParallelWrapper):
    """Wrapper to vectorize custom PettingZoo ParallelEnv environments, instantiated through
    env = CustomEnv()

    :param env: Custom PettingZoo environment
    :type env: ParallelEnv
    :param n_envs: The number of environments to run in parallel
    :type n_envs: int
    """

    def __init__(
        self,
        env: ParallelEnv[AgentID, ObsType, ActionType],
        n_envs: int,
    ):
        super().__init__(env=env)
        self.num_envs = n_envs
        self.env = SubprocVecEnv(
            [lambda: env for _ in range(n_envs)], enable_autoreset=False
        )
        return


class DefaultPettingZooVectorizationParallelWrapper(PettingZooParallelWrapper):
    """Wrapper to vectorize PetttingZoo library parallel environments, instantiated through
    env = pettingzoo_env.parallel_env()

    :param env: Default PettingZoo environment
    :type env: types.ModuleType
    :param n_envs: The number of environments to run in parallel
    :type n_envs: int
    :param enable_autoreset: Boolean flag to enable environment auto-resetting
    :type enable_autoreset: bool
    :param env_args: Dictionary of arguments for your environment instance
    :type env_args: dict
    """

    def __init__(
        self, env: types.ModuleType, n_envs: int, enable_autoreset=True, env_args={}
    ):
        super().__init__(env=env)
        self.num_envs = n_envs
        self.env = SubprocVecEnv(
            [lambda: env for _ in range(n_envs)],
            enable_autoreset=enable_autoreset,
            env_args=env_args,
        )
        return


# class PettingZooVectorWrapper:
#     def __init__(env, n_envs):
