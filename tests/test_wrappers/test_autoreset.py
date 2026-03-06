import functools
from unittest.mock import patch

import gymnasium
import pytest
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


class RPSParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        """The init method takes in environment arguments and should define the following attributes:
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
            zip(
                self.possible_agents,
                list(range(len(self.possible_agents))),
                strict=False,
            ),
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.cache  # noqa: B019
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(4)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.cache  # noqa: B019
    def action_space(self, agent):
        return Discrete(3)

    def render(self):
        """Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode.",
            )
            return

        if len(self.agents) == 2:
            string = f"Current state: Agent1: {MOVES[self.state[self.agents[0]]]} , Agent2: {MOVES[self.state[self.agents[1]]]}"
        else:
            string = "Game over"
        print(string)

    def close(self):
        """Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """

    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = dict.fromkeys(self.agents, NONE)
        infos = dict.fromkeys(self.agents, self.num_moves)
        self.state = observations

        return observations, infos

    def step(self, actions):
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, other_agent_1: item_2}
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

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = dict.fromkeys(self.agents, env_truncation)
        terminations = dict.fromkeys(self.agents, env_truncation)
        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: int(actions[self.agents[1 - i]])
            for i in range(len(self.agents))
        }
        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = dict.fromkeys(self.agents, self.num_moves)

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos


def test_autoreset_wrapper_autoreset():
    """Tests the autoreset wrapper actually automatically resets correctly."""
    env = RPSParallelEnv(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)
    env.reset()
    with patch(
        f"{__name__}.RPSParallelEnv.reset",
        wraps=env.env.reset,
    ) as autoreset_patch:
        # Environment truncates after 100 steps, so we expect 1 reset.
        for _ in range(100):
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _ = env.step(actions)
        autoreset_patch.assert_called()
        autoreset_patch.reset_mock()
        # Environment truncates after 100 steps, so we expect 5 resets.
        for _ in range(500):
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _ = env.step(actions)
        autoreset_patch.assert_called()
        assert autoreset_patch.call_count == 5
        autoreset_patch.reset_mock()
        # Environment truncates after 100 steps, so we expect no reset.
        for _ in range(99):
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            _ = env.step(actions)
        autoreset_patch.assert_not_called()


def test_return_unwrapped():
    env = RPSParallelEnv(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)
    env.reset()
    unwrapped = env.unwrapped
    assert isinstance(unwrapped, ParallelEnv)


def test_return_state():
    env = RPSParallelEnv(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)
    observations, _ = env.reset()
    state = env.state
    assert state == observations


def test_pettingzoo_wrapper_close():
    """Test PettingZooAutoResetParallelWrapper.close()."""
    env = RPSParallelEnv(render_mode=None)
    wrapped = PettingZooAutoResetParallelWrapper(env)
    wrapped.reset()
    result = wrapped.close()
    assert result is None


def test_pettingzoo_wrapper_state_property():
    """Test PettingZooAutoResetParallelWrapper.state property."""
    env = RPSParallelEnv(render_mode=None)
    wrapped = PettingZooAutoResetParallelWrapper(env)
    observations, _ = wrapped.reset()
    state = wrapped.state
    assert state is not None
    assert state == observations


@pytest.mark.parametrize("agent_id", ["player_0", "player_1"])
def test_pettingzoo_wrapper_autoreset_on_all_done(agent_id):
    """Test autoreset when all terminations/truncations."""
    env = RPSParallelEnv(render_mode=None)
    wrapped = PettingZooAutoResetParallelWrapper(env)
    wrapped.reset()
    # Step until truncation (100 steps) to trigger autoreset
    for _ in range(101):
        actions = {
            agent: wrapped.action_space(agent).sample() for agent in wrapped.agents
        }
        _, _, _, _, _ = wrapped.step(actions)
        if not wrapped.agents:
            break
    assert len(wrapped.agents) == 2
    assert wrapped.observation_space(agent_id) is not None
    assert wrapped.action_space(agent_id) is not None


@pytest.mark.parametrize("agent_id", ["player_0", "player_1"])
def test_pettingzoo_wrapper_delegates_spaces(agent_id):
    env = RPSParallelEnv(render_mode=None)
    wrapped = PettingZooAutoResetParallelWrapper(env)
    obs_space = wrapped.observation_space(agent_id)
    act_space = wrapped.action_space(agent_id)
    assert obs_space == env.observation_space(agent_id)
    assert act_space == env.action_space(agent_id)


def test_pettingzoo_wrapper_render():
    env = RPSParallelEnv(render_mode=None)
    wrapped = PettingZooAutoResetParallelWrapper(env)
    result = wrapped.render()
    assert result is None


def test_pettingzoo_wrapper_init_suppresses_missing_state_space():
    env = RPSParallelEnv(render_mode=None)
    assert not hasattr(env, "state_space")
    wrapped = PettingZooAutoResetParallelWrapper(env)
    assert not hasattr(wrapped, "state_space")


def test_pettingzoo_wrapper_init_sets_state_space_when_present():
    env = RPSParallelEnv(render_mode=None)
    env.state_space = "custom_state"
    wrapped = PettingZooAutoResetParallelWrapper(env)
    assert wrapped.state_space == "custom_state"


if __name__ == "__main__":
    test_autoreset_wrapper_autoreset()
