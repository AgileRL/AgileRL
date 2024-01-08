import functools
from unittest.mock import patch

import gymnasium
import pytest
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper

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


class parallel_env_disc(ParallelEnv):
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
        infos = {agent: {} for agent in self.agents}
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
            (int(actions[self.agents[0]]), int(actions[self.agents[1]]))
        ]

        terminations = {agent: False for agent in self.agents}

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # current observation is just the other player's most recent action
        observations = {
            self.agents[i]: int(actions[self.agents[1 - i]])
            for i in range(len(self.agents))
        }
        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def seed(self, seed):
        self.seed = seed


class parallel_env_cont(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    def observation_space(self, agent):
        return Box(0, 2, shape=(2,), dtype=float)

    def action_space(self, agent):
        return Box(0, 2, shape=(2,), dtype=float)

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
        terminations = {agent: False for agent in self.agents}
        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}
        observations = {
            self.agents[i]: actions[self.agents[1 - i]] for i in range(len(self.agents))
        }
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        if env_truncation:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


class parallel_env_atari(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    def observation_space(self, agent):
        return Box(0, 255, shape=(32, 32, 3), dtype=int)

    def action_space(self, agent):
        return Discrete(3)

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
        terminations = {agent: False for agent in self.agents}
        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}
        observations = {
            self.agents[i]: self.action_space(self.agents[i]).sample()
            for i in range(len(self.agents))
        }
        self.state = observations
        infos = {agent: {} for agent in self.agents}
        if env_truncation:
            self.agents = []
        return observations, rewards, terminations, truncations, infos


@pytest.fixture
def petting_zoo_env_cont_actions():
    return parallel_env_cont()


@pytest.fixture
def petting_zoo_env_disc_actions():
    return parallel_env_disc()


@pytest.fixture
def atari():
    return parallel_env_atari()


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_vectorisation_wrapper_petting_zoo_reset(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    vec_env.close()
    for agent in vec_env.agents:
        assert len(observations[agent]) == n_envs
        assert len(infos[agent]) == n_envs


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_vectorisation_wrapper_petting_zoo_step(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    for step in range(25):
        actions = {
            agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
            for agent in vec_env.agents
        }
        print("ACTIONS", actions)
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        for agent in vec_env.agents:
            assert len(observations[agent]) == n_envs
            assert len(rewards[agent]) == n_envs
            assert len(terminations[agent]) == n_envs
            assert len(truncations[agent]) == n_envs
            assert len(infos[agent]) == n_envs
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)


@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
    ],
)
def test_cont_action_observation_spaces(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    vec_env.reset()
    print(env.action_space("speaker_0").shape[0])
    print(vec_env.action_space("speaker_0").shape[0])
    assert [env.action_space(agent).shape[0] for agent in env.agents] == [
        vec_env.action_space(agent).shape[0] for agent in vec_env.agents
    ]
    assert [env.observation_space(agent).shape for agent in env.agents] == [
        vec_env.observation_space(agent).shape for agent in vec_env.agents
    ]


@pytest.mark.parametrize("env", ["petting_zoo_env_disc_actions", "atari"])
def test_disc_action_observation_spaces(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    vec_env.reset()
    assert [env.action_space(agent).n for agent in env.agents] == [
        vec_env.action_space(agent).n for agent in vec_env.agents
    ]
    assert [env.observation_space(agent).shape for agent in env.agents] == [
        vec_env.observation_space(agent).shape for agent in vec_env.agents
    ]


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_basic_attributes(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    assert env.agents == vec_env.agents
    assert env.num_agents == vec_env.num_agents
    assert env.possible_agents == vec_env.possible_agents
    assert env.max_num_agents == vec_env.max_num_agents


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_close(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    print(actions)
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    vec_env.render()
    with patch(
        "agilerl.wrappers.pettingzoo_wrappers.PettingZooVectorizationParallelWrapper.close"
    ) as mock_close:
        vec_env.close()
        mock_close.assert_called()
    pass


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_seed(env, request):
    env = request.getfixturevalue(env)
    n_envs = 4
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    vec_env.env.seed(0)


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_subproc_close(env, request):
    env = request.getfixturevalue(env)
    n_envs = 1
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()

    class DummyRemote:
        def recv(self, *args):
            return None

        def send(self, *args):
            pass

    vec_env.env.remotes = [DummyRemote()]

    vec_env.env.waiting = True
    vec_env.env.close()
    vec_env.env.close()

    assert vec_env.env.closed is True


@pytest.mark.parametrize(
    "env", ["petting_zoo_env_cont_actions", "petting_zoo_env_disc_actions", "atari"]
)
def test_sample_personas(env, request):
    env = request.getfixturevalue(env)

    def dummy_sp(is_train, is_val, path):
        return None

    env.sample_personas = dummy_sp

    n_envs = 1
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()

    result = vec_env.env.sample_personas(False)

    assert result is None
