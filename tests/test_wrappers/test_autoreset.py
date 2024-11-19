import functools
from unittest.mock import patch

import gymnasium
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv

from agilerl.wrappers.pettingzoo_wrappers import PettingZooAutoResetParallelWrapper

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


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: NONE for agent in self.agents}
        infos = {agent: self.num_moves for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = REWARD_MAP[
            (actions[self.agents[0]], actions[self.agents[1]])
        ]

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}
        terminations = {agent: env_truncation for agent in self.agents}
        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: int(actions[self.agents[1 - i]])
            for i in range(len(self.agents))
        }
        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: self.num_moves for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos


def test_autoreset_wrapper_autoreset():
    """Tests the autoreset wrapper actually automatically resets correctly."""
    env = parallel_env(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)
    observations, infos = env.reset()
    with patch(
        f"{__name__}.parallel_env.reset", wraps=env.env.reset
    ) as autoreset_patch:
        # Environment truncates after 100 steps, so we expect 1 reset.
        for _ in range(100):
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        autoreset_patch.assert_called()
        autoreset_patch.reset_mock()
        # Environment truncates after 100 steps, so we expect 5 resets.
        for _ in range(500):
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        autoreset_patch.assert_called()
        assert autoreset_patch.call_count == 5
        autoreset_patch.reset_mock()
        # Environment truncates after 100 steps, so we expect no reset.
        for _ in range(99):
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            observations, rewards, terminations, truncations, infos = env.step(actions)
        autoreset_patch.assert_not_called()


def test_return_unwrapped():
    env = parallel_env(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)
    observations, infos = env.reset()
    unwrapped = env.unwrapped
    assert isinstance(unwrapped, ParallelEnv)


def test_return_state():
    env = parallel_env(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)
    observations, infos = env.reset()
    state = env.state
    assert state == observations


if __name__ == "__main__":
    test_autoreset_wrapper_autoreset()
