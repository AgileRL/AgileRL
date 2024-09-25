from typing import Any

import numpy as np
from gymnasium.utils import seeding


class VecEnv:
    """An abstract asynchronous, vectorized environment

    References:
        https://github.com/openai/baselines/tree/master/baselines/common/vec_env
        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/vector/vector_env.py
    """

    metadata: dict[str, Any] = {}
    render_mode: str | None = None
    closed: bool = False

    num_envs: int

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def __init__(self, num_envs, possible_agents):
        self.num_envs = num_envs
        self.agents = possible_agents
        self.num_agents = len(self.agents)

    def reset(self, seed=None, options=None):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        if seed is not None:
            self._np_random, self._np_random_ = seeding.np_random(seed)

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def step(self, actions):
        print("ACTIONS")
        print(actions)
        passed_actions_list = [[] for _ in list(actions.values())[0]]
        for env_idx, _ in enumerate(list(actions.values())[0]):
            for possible_agent in self.agents:
                passed_actions_list[env_idx].append(actions[possible_agent][env_idx])
        assert (
            len(passed_actions_list) == self.num_envs
        ), "Number of actions passed to the step function must be equal to the number of vectorized environments"
        self.step_async(passed_actions_list)
        step_wait = self.step_wait()
        return step_wait

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
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value
        self._np_random_seed = -1

    @property
    def np_random_seed(self) -> int | None:
        """Returns the environment's internal :attr:`_np_random_seed` that if not set will first initialise with a random int as seed.

        If :attr:`np_random_seed` was set directly instead of through :meth:`reset` or :meth:`set_np_random_through_seed`,
        the seed will take the value -1.

        Returns:
            int: the seed of the current `np_random` or -1, if the seed of the rng is unknown
        """
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random_seed

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self

    def _add_info(self, vector_infos, indi_info, idx):
        """Handle env defined actions"""
        pass
        # indi_info = {
        #     "agent_0": {"env_defined_actions" : np.array()}
        # }
        # for key, value in indi_info.items():
        #     if key not in vector_infos.keys():
        #         vector_infos[key] = {}
        #     if "env_defined_actions" in value.keys():
        #         vector_infos[key]["env_defined_actions"] = np.zeros((self.num_envs, ))
