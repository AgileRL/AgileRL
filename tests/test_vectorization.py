import copy
import functools
from typing import Any, Literal
from unittest.mock import patch

import gymnasium
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from pettingzoo.mpe import simple_adversary_v3

from agilerl.wrappers.pettingzoo_wrappers import (
    DefaultPettingZooVectorizationParallelWrapper,
    PettingZooAutoResetParallelWrapper,
    PettingZooParallelWrapper,
    PettingZooVectorizationParallelWrapper,
)

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
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: np.random.randint(100) for agent in self.agents}
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


class parallel_env_disc_action_masking(ParallelEnv):
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

        for agent in self.agents:
            observations[agent] = {}
            observations[agent]["observation"] = 1
            observations[agent]["action_mask"] = np.ones(self.action_space(agent).n)

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
        observations = {}

        for agent in self.agents:
            observations[agent] = {}
            observations[agent]["observation"] = int(actions[agent])
            observations[agent]["action_mask"] = np.ones(self.action_space(agent).n)

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
        self.seed = self.dummy_seed

    def dummy_seed(self, seed):
        pass

    def observation_space(self, agent):
        return Box(0, 2, shape=(2,), dtype=float)

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


class parallel_env_cont_agent_masking(ParallelEnv):
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
        return Box(0, 2, shape=(1,), dtype=float)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: NONE for agent in self.agents}
        infos = {
            agent: {
                "env_defined_actions": np.random.normal(
                    size=(self.action_space(agent).shape)
                )
            }
            for agent in self.agents
        }
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

        if env_truncation:
            self.agents = []
        infos = {
            agent: (
                {
                    "env_defined_actions": np.random.normal(
                        size=(self.action_space(agent).shape)
                    )
                }
                if idx % 2 == 0
                else {"env_defined_actions": None}
            )
            for idx, agent in enumerate(self.agents)
        }
        return observations, rewards, terminations, truncations, infos


class parallel_env_atari(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.seed = self.dummy_seed

    def dummy_seed(self, value):
        pass

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


class terminal_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, terminal_mode, render_mode=None):
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode
        self.terminal_mode = terminal_mode

    def observation_space(self, agent):
        return Box(0, 2, shape=(2,), dtype=float)

    def action_space(self, agent):
        return Box(0, 2, shape=(1,), dtype=float)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        observations = {agent: NONE for agent in self.agents}
        infos = {agent: {"reset"} for agent in self.agents}
        self.state = observations
        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        rewards = {}
        rewards[self.agents[0]], rewards[self.agents[1]] = (0, 0)
        if self.terminal_mode == "term_only":
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
        elif self.terminal_mode == "trunc_only":
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
        else:
            terminations = {
                agent: True if idx % 2 == 0 else False
                for idx, agent in enumerate(self.agents)
            }
            truncations = {
                agent: False if terminations[agent] else True for agent in self.agents
            }
        observations = {
            self.agents[i]: actions[self.agents[1 - i]] for i in range(len(self.agents))
        }
        self.state = observations
        infos = {
            agent: {
                "env_defined_actions": np.random.normal(
                    size=(self.action_space(agent).shape)
                )
            }
            for agent in self.agents
        }
        return observations, rewards, terminations, truncations, infos


class error_env(ParallelEnv):
    metadata = {}

    def __init__(self):
        self.agents = self.possible_agents = ["a1", "a2"]

    def step(self, action, *args, **kwargs):
        raise Exception

    def reset(self, *args, **kwargs):
        return ({a: 1 for a in self.agents}, {a: {} for a in self.agents})

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)


@pytest.fixture
def make_error_env():
    return error_env()


@pytest.fixture
def native_pz_env(env):
    return env


@pytest.fixture
def petting_zoo_env_cont_actions():
    return parallel_env_cont()


@pytest.fixture
def petting_zoo_env_disc_actions():
    return parallel_env_disc()


@pytest.fixture
def petting_zoo_terminal_env(terminal_mode):
    return terminal_env(terminal_mode)


@pytest.fixture
def atari():
    return parallel_env_atari()


@pytest.fixture
def pettingzoo_env_disc_action_masking():
    return parallel_env_disc_action_masking()


@pytest.fixture
def pettingzoo_env_cont_agent_masking():
    return parallel_env_cont_agent_masking()


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_vectorisation_wrapper_petting_zoo_reset(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    vec_env.close()

    for agent in vec_env.agents:
        if hasattr(observations[agent], "get"):
            assert len(observations[agent]["observation"]) == n_envs
            assert len(observations[agent]["action_mask"]) == n_envs
        else:
            assert len(observations[agent]) == n_envs
        assert isinstance(infos[agent], dict)


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_vectorisation_wrapper_petting_zoo_step(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()

    for step in range(25):
        actions = {
            agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
            for agent in vec_env.agents
        }
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        for agent in vec_env.agents:
            if isinstance(observations[agent], dict):
                assert len(observations[agent]["observation"]) == n_envs
            else:
                assert len(observations[agent]) == n_envs
            assert len(rewards[agent]) == n_envs
            assert len(terminations[agent]) == n_envs
            assert len(truncations[agent]) == n_envs
            assert len(infos[agent]) == 0
            assert isinstance(observations, dict)
            assert isinstance(rewards, dict)
            assert isinstance(terminations, dict)
            assert isinstance(truncations, dict)
            assert isinstance(infos, dict)


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
    ],
)
def test_cont_action_observation_spaces(
    env: Literal["petting_zoo_env_cont_actions"],
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    env.reset()
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    vec_env.reset()
    assert [env.action_space(agent).shape[0] for agent in env.agents] == [
        vec_env.action_space(agent).shape[0] for agent in vec_env.agents
    ]
    assert [env.observation_space(agent).shape for agent in env.agents] == [
        vec_env.observation_space(agent).shape for agent in vec_env.agents
    ]


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    ["petting_zoo_env_disc_actions", "atari", "pettingzoo_env_disc_action_masking"],
)
def test_disc_action_observation_spaces(
    env: (
        Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    env.reset()
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    vec_env.reset()
    assert [env.action_space(agent).n for agent in env.agents] == [
        vec_env.action_space(agent).n for agent in vec_env.agents
    ]
    assert [env.observation_space(agent).shape for agent in env.agents] == [
        vec_env.observation_space(agent).shape for agent in vec_env.agents
    ]


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_basic_attributes(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    env.reset()
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    assert env.agents == vec_env.agents
    assert env.num_agents == vec_env.num_agents
    assert env.possible_agents == vec_env.possible_agents
    assert env.max_num_agents == vec_env.max_num_agents


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_close(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    vec_env.render()
    with patch(
        "agilerl.wrappers.pettingzoo_wrappers.PettingZooVectorizationParallelWrapper.close"
    ) as mock_close:
        vec_env.close()
        mock_close.assert_called()
    pass


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_seed(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    vec_env.env.seed(0)
    vec_env.close()


@pytest.mark.parametrize("n_envs", [1])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_subproc_close(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()

    class DummyRemote:
        closed = False

        def recv(self, *args):
            return None, True

        def send(self, *args):
            pass

        def close(self, *args):
            pass

    vec_env.env.parent_remotes = [DummyRemote()]

    vec_env.env.waiting = True
    vec_env.env.close()
    vec_env.env.close()

    assert vec_env.env.closed is True


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize(
    "env",
    [
        "petting_zoo_env_cont_actions",
        "petting_zoo_env_disc_actions",
        "atari",
        "pettingzoo_env_disc_action_masking",
    ],
)
def test_sample_personas(
    env: (
        Literal["petting_zoo_env_cont_actions"]
        | Literal["petting_zoo_env_disc_actions"]
        | Literal["atari"]
        | Literal["pettingzoo_env_disc_action_masking"]
    ),
    request: pytest.FixtureRequest,
    n_envs: Literal[1] | Literal[4],
):
    env = request.getfixturevalue(env)

    def dummy_sp(is_train, is_val, path):
        return None

    env.sample_personas = dummy_sp
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    result = vec_env.env.sample_personas(False)
    vec_env.close()
    assert result is None


@pytest.mark.parametrize(
    "n_envs",
    [
        # 1,
        4
    ],
)
@pytest.mark.parametrize("env", ["pettingzoo_env_disc_action_masking"])
def test_action_masking(
    env: Literal["pettingzoo_env_disc_action_masking"],
    request: pytest.FixtureRequest,
    n_envs: Literal[4],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }

    observations, rewards, terminations, truncations, infos = vec_env.step(actions)

    for agent in vec_env.agents:
        assert "observation" in observations[agent].keys()
        assert "action_mask" in observations[agent].keys()
    for agent in vec_env.agents:
        assert observations[agent]["action_mask"].shape == (
            n_envs,
            vec_env.action_space(agent).n,
        )
    vec_env.close()


@pytest.mark.parametrize("n_envs", [4])
@pytest.mark.parametrize("env", ["pettingzoo_env_cont_agent_masking"])
def test_agent_masking(
    env: Literal["pettingzoo_env_cont_agent_masking"],
    request: pytest.FixtureRequest,
    n_envs: Literal[4],
):
    env = request.getfixturevalue(env)
    vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
    observations, infos = vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    observations, rewards, terminations, truncations, infos = vec_env.step(actions)
    for agent, dic in infos.items():
        assert "env_defined_actions" in dic.keys()
        assert dic["env_defined_actions"].shape == (
            n_envs,
            vec_env.action_space(agent).shape[0],
        )

    # for agent in vec_env.agents:
    #     assert observations["action_mask"][agent].shape == (n_envs, vec_env.action_space(agent).n)
    vec_env.close()


def test_pz_native_env_reset():
    num_envs = 2
    env = DefaultPettingZooVectorizationParallelWrapper(simple_adversary_v3, num_envs)

    reset_result = env.reset()
    env_agents = copy.deepcopy(env.agents)
    assert isinstance(reset_result, tuple)

    obs, info = reset_result
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    assert list(obs.keys()) == env_agents
    assert list(info.keys()) == env_agents

    for v in obs.values():
        assert isinstance(v, np.ndarray)
        assert v.shape[0] == num_envs

    env.close()


@pytest.mark.parametrize(
    "env, continuous_actions",
    [(simple_adversary_v3, True), (simple_adversary_v3, False)],
)
def test_pz_native_env_step(env: Any, native_pz_env: Any, continuous_actions: bool):
    num_envs = 1
    env = DefaultPettingZooVectorizationParallelWrapper(
        native_pz_env,
        num_envs,
        enable_autoreset=True,
        env_args={"N": 1, "continuous_actions": continuous_actions},
    )
    _ = env.reset()
    env_agents = copy.deepcopy(env.agents)
    actions = {
        agent: np.array([env.action_space(agent).sample() for n in range(num_envs)])
        for agent in env.agents
    }
    *transition, info = env.step(actions)
    env.close()
    for t in transition:
        assert list(t.keys()) == env_agents
        for k, v in t.items():
            assert isinstance(v, np.ndarray)
            assert v.shape[0] == num_envs
    assert list(info.keys()) == env_agents
    for k, v in info.items():
        assert isinstance(v, dict)


@pytest.mark.parametrize(
    "env",
    ["petting_zoo_env_disc_actions"],
)
def test_render_vectorized_pz_default_wrapper(
    env: Literal["petting_zoo_env_disc_actions"], request: pytest.FixtureRequest
):
    env = request.getfixturevalue(env)
    env = PettingZooParallelWrapper(env)
    env.render()


@pytest.mark.parametrize(
    "env, wrapper",
    [
        ("petting_zoo_env_disc_actions", PettingZooParallelWrapper),
        ("petting_zoo_env_disc_actions", PettingZooAutoResetParallelWrapper),
    ],
)
def test_unwrapped_property(
    env: Literal["petting_zoo_env_disc_actions"],
    request: pytest.FixtureRequest,
    wrapper: Any | PettingZooAutoResetParallelWrapper,
):
    env = request.getfixturevalue(env)
    wrapped_env = wrapper(env)
    assert env.unwrapped == wrapped_env.unwrapped


@pytest.mark.parametrize(
    "env, wrapper",
    [
        ("petting_zoo_env_disc_actions", PettingZooParallelWrapper),
        ("petting_zoo_env_disc_actions", PettingZooAutoResetParallelWrapper),
    ],
)
def test_state_property(
    env: Literal["petting_zoo_env_disc_actions"],
    request: pytest.FixtureRequest,
    wrapper: Any | PettingZooAutoResetParallelWrapper,
):
    env = request.getfixturevalue(env)
    wrapped_env = wrapper(env)
    assert env.state == wrapped_env.state


@pytest.mark.parametrize(
    "env, wrapper",
    [
        ("petting_zoo_env_disc_actions", PettingZooParallelWrapper),
        ("petting_zoo_env_disc_actions", PettingZooAutoResetParallelWrapper),
    ],
)
def test_observation_space_property(
    env: Literal["petting_zoo_env_disc_actions"],
    request: pytest.FixtureRequest,
    wrapper: Any | PettingZooAutoResetParallelWrapper,
):
    env = request.getfixturevalue(env)
    wrapped_env = wrapper(env)
    for agent in env.possible_agents:
        assert env.observation_space(agent) == wrapped_env.observation_space(agent)


@pytest.mark.parametrize(
    "terminal_mode", [("term_only"), ("trunc_only"), ("term_and_trunc")]
)
def test_auto_reset_wrapper(
    petting_zoo_terminal_env: terminal_env,
    terminal_mode: (
        Literal["term_only"] | Literal["trunc_only"] | Literal["term_and_trunc"]
    ),
):
    env = PettingZooAutoResetParallelWrapper(petting_zoo_terminal_env)
    actions = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
    env.reset()
    with patch.object(env.env, "reset", side_effect=lambda: ({}, {})) as mock_reset:
        env.step(actions)
        mock_reset.assert_called_once

    env.close()


@pytest.mark.parametrize(
    "env",
    [(simple_adversary_v3)],
)
def test_auto_reset_subproc(
    env: Any,
    native_pz_env: Any,
):
    n_envs = 4
    max_cycles = 25
    vec_env = DefaultPettingZooVectorizationParallelWrapper(
        env=native_pz_env, n_envs=n_envs, enable_autoreset=True
    )
    vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    for i in range(max_cycles):
        *_, trunc, _ = vec_env.step(actions)

    assert all(all(t) for t in trunc.values())

    *_, trunc, _ = vec_env.step(actions)
    assert not all(all(t) for t in trunc.values())
    vec_env.close()


def test_exception_throws_from_within_subproc(make_error_env):
    n_envs = 2
    vec_env = PettingZooVectorizationParallelWrapper(make_error_env, n_envs=n_envs)
    vec_env.reset()
    actions = {
        agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
        for agent in vec_env.agents
    }
    with pytest.raises(Exception):
        vec_env.step(actions)

    vec_env.close()


def test_env_defined_action_none_to_nan():
    pass


def test_exception_when_incorrect_num_actions():
    pass
