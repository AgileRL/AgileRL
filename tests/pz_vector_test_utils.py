import types
from collections.abc import Callable

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

ROCK = 0
PAPER = 1
SCISSORS = 2
NONE = 3
MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
NUM_ITERS = 100
REWARD_MAP = {
    (ROCK, ROCK): (0, 0),
    (ROCK, PAPER): (-1, 1),
    (ROCK, SCISSORS): (1, -1),
    (PAPER, ROCK): (1, -1),
    (PAPER, PAPER): (0, 0),
    (PAPER, SCISSORS): (-1, 1),
    (SCISSORS, ROCK): (-1, 1),
    (SCISSORS, PAPER): (1, -1),
    (SCISSORS, SCISSORS): (0, 0),
}


class term_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.seed = self.dummy_seed

    def dummy_seed(self, seed):
        pass

    def observation_space(self, agent):
        return Discrete(2)  # 0, 2, shape=(2,), dtype=float)

    def action_space(self, agent):
        return Box(0, 2, shape=(1,), dtype=float)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: NONE for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = (0, 0)
        terminations = {agent: True for agent in self.agents}
        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}
        observations = {
            self.agents[i]: actions[self.agents[1 - i]].astype(int)
            for i in range(len(self.agents))
        }
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        if env_truncation:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


def basic_reset_func(
    self,
    *,
    seed: int | None = None,
    options: dict | None = None,
):
    """A basic reset function that will pass the environment check using random actions from the observation space."""
    return {
        agent: self.observation_space(agent).sample() for agent in self.possible_agents
    }, {agent: {"options": options} for agent in self.possible_agents}


def old_reset_func(self):
    """An old reset function that will pass the environment check using random actions from the observation space."""
    super(GenericTestEnv, self).reset()
    return {
        agent: self.observation_space(agent).sample() for agent in self.possible_agent
    }


def basic_step_func(self, action):
    """A step function that follows the basic step api that will pass the environment check using random actions from the observation space."""

    def pz_dict(transition, agents):
        return {agent: transition for agent in self.possible_agents}

    return (
        {
            agent: self.observation_space(agent).sample()
            for agent in self.possible_agents
        },
        pz_dict(0, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict({}, self.possible_agents),
    )


def old_step_func(self, action):
    """A step function that follows the old step api that will pass the environment check using random actions from the observation space."""

    def pz_dict(transition, agents):
        return {agent: transition for agent in self.possible_agents}

    return (
        {
            agent: self.observation_space(agent).sample()
            for agent in self.possible_agents
        },
        pz_dict(0, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict(False, self.possible_agents),
        pz_dict({}, self.possible_agents),
    )


def basic_render_func(self):
    """Basic render fn that does nothing."""
    pass


class GenericTestEnv(ParallelEnv):
    """A generic testing environment for use in testing with modified environments are required."""

    def __init__(
        self,
        action_space: spaces.Space = spaces.Box(0, 1, (1,)),
        observation_space: spaces.Space = spaces.Box(0, 1, (1,)),
        reset_func: Callable = basic_reset_func,
        step_func: Callable = basic_step_func,
        render_func: Callable = basic_render_func,
        metadata={"render_modes": []},
        render_mode=None,
    ):
        """Generic testing environment constructor.

        Args:
            action_space: The environment action space
            observation_space: The environment observation space
            reset_func: The environment reset function
            step_func: The environment step function
            render_func: The environment render function
            metadata: The environment metadata
            render_mode: The render mode of the environment
            spec: The environment spec
        """
        self.metadata = metadata
        self.render_mode = render_mode
        self.possible_agents = ["agent_0"]
        self.action_space_dict = {"agent_0": action_space}
        self.observation_space_dict = {"agent_0": observation_space}

        if observation_space is not None:
            self.observation_space = self.action_space_func
        if action_space is not None:
            self.action_space = self.observation_space_func

        if reset_func is not None:
            self.reset = types.MethodType(reset_func, self)
        if step_func is not None:
            self.step = types.MethodType(step_func, self)
        if render_func is not None:
            self.render = types.MethodType(render_func, self)

    def action_space_func(self, agent):
        return self.action_space_dict[agent]

    def observation_space_func(self, agent):
        return self.observation_space_dict[agent]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Resets the environment."""
        # If you need a default working reset function, use `basic_reset_fn` above
        raise NotImplementedError("TestingEnv reset_fn is not set.")

    def step(self, action):
        """Steps through the environment."""
        raise NotImplementedError("TestingEnv step_fn is not set.")

    def render(self):
        """Renders the environment."""
        raise NotImplementedError("testingEnv render_fn is not set.")


class CustomSpace(gym.Space):
    """Minimal custom observation space."""

    shape = (4,)

    def sample(self):
        """Generates a sample from the custom space."""
        return self.np_random.integers(0, 10, ())

    def contains(self, x):
        """Check if the element `x` is contained within the space."""
        return 0 <= x <= 10

    def __eq__(self, other):
        """Check if the two spaces are equal."""
        return isinstance(other, CustomSpace)
