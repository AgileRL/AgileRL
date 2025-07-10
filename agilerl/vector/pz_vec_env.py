from typing import Any, Dict, List, Optional

import numpy as np
from gymnasium.spaces import Space
from gymnasium.vector.utils import batch_space

from agilerl.typing import ActionType, PzStepReturn


class PettingZooVecEnv:
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py

    :param num_envs: Number of environments to vectorize
    :type num_envs: int
    :param observation_spaces: Dictionary of observation spaces
    :type observation_spaces: dict[str, gymnasium.spaces.Space]
    :param action_spaces: List of action spaces
    :type action_spaces: list[gymnasium.spaces.Space]
    :param possible_agents: List of possible agents
    :type possible_agents: list[str]
    """

    metadata: Dict[str, Any] = {}
    render_mode: Optional[str] = None
    closed: bool = False
    num_envs: int
    agents: List[str]
    num_agents: int

    def __init__(
        self,
        num_envs: int,
        observation_spaces: Dict[str, Space],
        action_spaces: List[Space],
        possible_agents: List[str],
    ):
        self.num_envs = num_envs
        self.agents = possible_agents
        self.num_agents = len(self.agents)
        self._single_observation_spaces = observation_spaces
        self._single_action_spaces = action_spaces
        self._observation_spaces = {
            agent: batch_space(space, self.num_envs)
            for agent, space in observation_spaces.items()
        }
        self._action_spaces = {
            agent: batch_space(space, self.num_envs)
            for agent, space in action_spaces.items()
        }
        self.action_space = self._get_action_space
        self.observation_space = self._get_observation_space
        self.single_action_space = self._get_single_action_space
        self.single_observation_space = self._get_single_observation_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reset all the environments and return two dictionaries of batched observations and infos.

        :param seed: Random seed, defaults to None
        :type seed: None | int, optional
        :param options: Options dictionary
        :type options: dict[str, Any]
        """
        pass

    def step_async(self, actions: List[Dict[str, ActionType]]) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.

        :param actions: List of dictionaries of length num_envs, each sub dictionary contains
        actions for each agent in a given environment
        :type actions: list[dict[str, int | float | np.ndarray]]
        """
        pass

    def step_wait(self) -> PzStepReturn:
        """
        Wait for the step taken with step_async().
        """
        pass

    def step(self, actions: Dict[str, np.ndarray]) -> PzStepReturn:
        """Take an action for each parallel environment

        :param actions: Dictionary of vectorized actions for each agent.
        :type actions: dict[str, np.ndarray]

        :return: Tuple of observations, rewards, terminated, truncated, infos
        :rtype: tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]
        """
        passed_actions_list = []
        for env_idx, _ in enumerate(list(actions.values())[0]):
            env_actions = {}
            for agent_id, action in actions.items():
                agent_action = action[env_idx]

                # Skip if the action is NaN
                if np.isnan(agent_action).all():
                    continue

                agent_action = (
                    int(agent_action) if np.isscalar(agent_action) else agent_action
                )
                env_actions[agent_id] = agent_action

            passed_actions_list.append(env_actions)
        assert (
            len(passed_actions_list) == self.num_envs
        ), "Number of actions passed to the step function must be equal to the number of vectorized environments"

        self.step_async(passed_actions_list)
        return self.step_wait()

    def render(self) -> Any:
        """Returns the rendered frames from the parallel environments."""
        raise NotImplementedError(
            f"{self.__str__()} render function is not implemented."
        )

    def close(self, **kwargs: Any) -> None:
        """
        Clean up the environments' resources.
        """
        if self.closed:
            return

        self.close_extras(**kwargs)
        self.closed = True

    def close_extras(self, **kwargs: Any) -> None:
        """Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    @property
    def unwrapped(self) -> "PettingZooVecEnv":
        """Return the base environment."""
        return self

    def _get_single_action_space(self, agent):
        """Get an agents single action space

        :param agent: Name of agent
        :type agent: str
        """
        return self._single_action_spaces[agent]

    def _get_action_space(self, agent):
        """Get an agents action space

        :param agent: Name of agent
        :type agent: str
        """
        return self._action_spaces[agent]

    def _get_single_observation_space(self, agent):
        """Get an agents single observation space

        :param agent: Name of agent
        :type agent: str
        """
        return self._single_observation_spaces[agent]

    def _get_observation_space(self, agent):
        """Get an agents observation space

        :param agent: Name of agent
        :type agent: str
        """
        return self._observation_spaces[agent]
