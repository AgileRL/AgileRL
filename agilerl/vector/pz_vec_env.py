from typing import Any


class PettingZooVecEnv:
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py

    :param num_envs: Number of environments to vectorize
    :type num_envs: int
    :param possible_agents: List of possible agents
    :type possible_agents: list[str]
    """

    metadata: dict[str, Any] = {}
    render_mode: str | None = None
    closed: bool = False

    num_envs: int

    def __init__(self, num_envs, possible_agents):
        self.num_envs = num_envs
        self.agents = possible_agents
        self.num_agents = len(self.agents)

    def reset(self, seed=None, options=None):
        """
        Reset all the environments and return two dictionaries of batched observations and infos.

        :param seed: Random seed, defaults to None
        :type seed: None | int, optional
        :param options: Options dictionary
        :type options: dict[str, Any]
        """
        pass

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.

        :param actions: List of lists of length num_envs, each sub list contains actions for each agent in a given environment
        :type actions: list[list[int | float | np.ndarray]]
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        """
        pass

    def step(self, actions):
        """Take an action for each parallel environment

        :param actions: Dictionary of vectorized actions for each agent.
        :type actions: dict[str, np.ndarray]
        """
        passed_actions_list = [[] for _ in list(actions.values())[0]]
        for env_idx, _ in enumerate(list(actions.values())[0]):
            for possible_agent in self.agents:
                passed_actions_list[env_idx].append(actions[possible_agent][env_idx])
        assert (
            len(passed_actions_list) == self.num_envs
        ), "Number of actions passed to the step function must be equal to the number of vectorized environments"
        self.step_async(passed_actions_list)
        return self.step_wait()

    def render(self):
        """Returns the rendered frames from the parallel environments."""
        raise NotImplementedError(
            f"{self.__str__()} render function is not implemented."
        )

    def close(self, **kwargs):
        """
        Clean up the environments' resources.
        """
        if self.closed:
            return

        self.close_extras(**kwargs)
        self.closed = True

    def close_extras(self, **kwargs: Any):
        """Clean up the extra resources e.g. beyond what's in this base class."""
        pass

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self
