from __future__ import annotations

import gymnasium as gym
import gymnasium.spaces
import numpy as np
from gymnasium.utils import seeding
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv

from agilerl.utils.multiprocessing_env import SubprocVecEnv


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
        if np.any(list(terminations.values())) or np.any(list(truncations.values())):
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
    def __init__(self, env: ParallelEnv[AgentID, ObsType, ActionType], n_envs: int):
        super().__init__(env=env)
        self.num_envs = n_envs
        self.env = SubprocVecEnv([lambda: self.env for _ in range(n_envs)])
        return


class Gym2PZWrapper:
    """
    Wrapper to make make any gymnasium environment conform to the PettingZoo API. Allows
    single agent environments to be used with multi-agent algorithms for benchmarking purposes.

    :param env: Gymnasium environment
    :type env: gymnasium.Env
    """

    def __init__(self, env: gym.Env) -> None:
        self.metadata = env.metadata
        self.possible_agents = ["agent_0"]
        self.env = env
        self.agents = ["agent_0"]
        self.name = "agent_0"
        self.observation_spaces = {self.name: env.observation_space}
        self.action_spaces = {self.name: env.action_space}
        self.num_agents = len(self.agents)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        return {self.name: obs}, {0: info}

    def step(self, action):
        action = action[self.name]
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            {self.name: obs},
            {self.name: reward},
            {self.name: terminated},
            {self.name: truncated},
            {0: info},
        )

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]
