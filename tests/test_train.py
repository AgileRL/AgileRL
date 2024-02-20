import os
import random
from unittest.mock import ANY, MagicMock, patch

import dill
import numpy as np
import pytest
import torch
from accelerate import Accelerator

import agilerl.training.train_bandits
import agilerl.training.train_multi_agent
import agilerl.training.train_off_policy
import agilerl.training.train_offline
import agilerl.training.train_on_policy
from agilerl.algorithms.cqn import CQN
from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.algorithms.maddpg import MADDPG
from agilerl.algorithms.matd3 import MATD3
from agilerl.algorithms.neural_ts_bandit import NeuralTS
from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.algorithms.ppo import PPO
from agilerl.algorithms.td3 import TD3
from agilerl.training.train_bandits import train_bandits
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import makeMultiAgentVectEnvs


class DummyEnv:
    def __init__(self, state_size, action_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.action_size = action_size
        self.vect = vect
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.n_envs = num_envs
            self.num_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), "info_string"

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            "info_string",
        )


class DummyBanditEnv:
    def __init__(self, state_size, arms):
        self.arms = arms
        self.state_size = (arms,) + state_size
        self.action_size = 1
        self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size)

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.rand(1),
        )


class DummyAgentOffPolicy:
    def __init__(self, batch_size, env, beta=None):
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.action_dim = env.action_size
        self.batch_size = batch_size
        self.beta = beta
        self.learn_step = 1
        self.scores = []
        self.steps = [0]
        self.fitness = []
        self.mut = "mutation"
        self.index = 1

    def getAction(self, *args):
        return np.random.rand(self.action_size)

    def learn(self, experiences, n_step=False, per=False):
        if n_step and per:
            return random.random(), None, None
        elif n_step or per:
            return random.random(), None, None
        else:
            return random.random()

    def test(self, env, swap_channels, max_steps, loop):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def saveCheckpoint(self, path):
        empty_dic = {}
        torch.save(empty_dic, path, pickle_module=dill)
        return True

    def loadCheckpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyAgentOnPolicy(DummyAgentOffPolicy):
    def __init__(self, batch_size, env):
        super().__init__(batch_size, env)

    def learn(self, *args, **kwargs):
        return random.random()

    def getAction(self, *args):
        return tuple(np.random.randn(self.action_size) for _ in range(4))

    def test(self, env, swap_channels, max_steps, loop):
        return super().test(env, swap_channels, max_steps, loop)

    def saveCheckpoint(self, path):
        return super().saveCheckpoint(path)

    def loadCheckpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyBandit:
    def __init__(self, batch_size, bandit_env, beta=None):
        self.state_size = bandit_env.state_size
        self.action_size = bandit_env.action_size
        self.action_dim = bandit_env.action_size
        self.batch_size = batch_size
        self.beta = beta
        self.learn_step = 1
        self.scores = []
        self.steps = [0]
        self.regret = [0]
        self.fitness = []
        self.mut = "mutation"
        self.index = 1

    def getAction(self, *args):
        return np.random.randint(self.action_size)

    def learn(self, experiences):
        return random.random()

    def test(self, env, swap_channels, max_steps, loop):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def saveCheckpoint(self, path):
        empty_dic = {}
        torch.save(empty_dic, path, pickle_module=dill)
        return True

    def loadCheckpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyMultiEnv:
    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.state_size = self.state_dims
        self.action_dims = action_dims
        self.action_size = self.action_dims
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = ["agent_0", "agent_1"]
        self.metadata = None
        self.observation_space = None
        self.action_space = None

    def reset(self, seed=None, options=None):
        return {agent: np.random.rand(*self.state_dims) for agent in self.agents}, {
            "info_string": None,
            "agent_mask": {"agent_0": False, "agent_1": True},
            "env_defined_actions": {"agent_0": np.array([0, 1]), "agent_1": None},
        }

    def step(self, action):
        return (
            {agent: np.random.rand(*self.state_dims) for agent in self.agents},
            {agent: np.random.randint(0, 5) for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            {agent: "info_string" for agent in self.agents},
        )


class DummyMultiAgent(DummyAgentOffPolicy):
    def __init__(self, batch_size, env, *args):
        super().__init__(batch_size, env, *args)
        self.agents = ["agent_0", "agent_1"]
        self.lr_actor = 0.001
        self.lr_critic = 0.01
        self.discrete_actions = False

    def getAction(self, *args):
        return {agent: np.random.randn(self.action_size) for agent in self.agents}, None

    def learn(self, experiences):
        return {
            "actors": {"agent_0": random.random(), "agent_1": random.random()},
            "critics": {"agent_0": random.random(), "agent_1": random.random()},
        }

    def test(self, env, swap_channels, max_steps, loop):
        return super().test(env, swap_channels, max_steps, loop)

    def saveCheckpoint(self, path):
        return super().saveCheckpoint(path)

    def loadCheckpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyTournament:
    def __init__(self):
        pass

    def select(self, pop):
        return pop[0], pop


class DummyMutations:
    def __init__(self):
        pass

    def mutation(self, pop, pre_training_mut=False):
        return pop


class DummyMemory:
    def __init__(self):
        self.counter = 0
        self.state_size = None
        self.action_size = None
        self.next_state_size = None

    def save2memoryVectEnvs(self, states, actions, rewards, next_states, dones):
        if self.state_size is None:
            self.state_size, *_ = (state.shape for state in states)
            self.action_size, *_ = (action.shape for action in actions)
            self.next_state_size, *_ = (next_state.shape for next_state in next_states)
            self.counter += 1

        one_step_transition = (
            np.random.randn(*self.state_size),
            np.random.randn(*self.action_size),
            np.random.uniform(0, 400),
            np.random.choice([True, False]),
            np.random.randn(*self.next_state_size),
        )

        return one_step_transition

    def save2memory(self, state, action, reward, next_state, done, is_vectorised=False):
        if is_vectorised:
            self.save2memoryVectEnvs(state, action, reward, next_state, done)
        else:
            self.state_size = state.shape
            self.action_size = action.shape
            self.next_state_size = next_state.shape
            self.counter += 1

    def __len__(self):
        return 1000

    def sample(self, batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, list):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*self.state_size)
            actions = np.random.randn(*self.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*self.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*self.state_size) for _ in range(batch_size)]
            )
            actions = np.array(
                [np.random.randn(*self.action_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            next_states = np.array(
                [np.random.randn(*self.next_state_size) for _ in range(batch_size)]
            )

        if beta is None:
            return states, actions, rewards, dones, next_states

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        return states, actions, rewards, dones, next_states, weights, idxs

    def update_priorities(self, idxs, priorities):
        return


class DummyNStepMemory(DummyMemory):
    def __init__(self):
        super().__init__()

    def save2memory(self, state, action, reward, next_state, done, is_vectorised):
        return super().save2memory(state, action, reward, next_state, done)

    def save2memoryVectEnvs(self, state, action, reward, next_state, done):
        self.state_size = state.shape
        self.action_size = action.shape
        self.next_state_size = next_state.shape
        self.counter += 1

        one_step_transition = (
            np.random.randn(*self.state_size),
            np.random.randn(*self.action_size),
            np.random.uniform(0, 400),
            np.random.randn(*self.next_state_size),
            np.random.choice([True, False]),
        )

        return one_step_transition

    def __len__(self):
        return super().__len__()

    def sample_n_step(self, *args):
        return super().sample(*args)

    def sample_per(self, *args):
        return super().sample(*args)

    def sample_from_indices(self, *args):
        return super().sample(*args)


class DummyBanditMemory:
    def __init__(self):
        self.counter = 0
        self.state_size = None
        self.action_size = 1

    def save2memoryVectEnvs(self, states, rewards):
        if self.state_size is None:
            self.state_size, *_ = (state.shape for state in states)
            self.counter += 1

    def save2memory(self, state, reward, is_vectorised=False):
        if is_vectorised:
            self.save2memoryVectEnvs(state, reward)
        else:
            self.state_size = state.shape
            self.counter += 1

    def __len__(self):
        return 1000

    def sample(self, batch_size, *args):
        if batch_size == 1:
            states = np.random.randn(*self.state_size)
            rewards = np.random.uniform(0, 400)
        else:
            states = np.array(
                [np.random.randn(*self.state_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])

        return states, rewards


class DummyMultiMemory(DummyMemory):
    def __init__(self):
        super().__init__()
        self.agents = ["agent_0", "agent_1"]

    def save2memory(self, state, action, reward, next_state, done, is_vectorised=False):
        self.state_size = list(state.values())[0].shape
        self.action_size = list(action.values())[0].shape
        self.next_state_size = list(next_state.values())[0].shape
        self.counter += 1

    def sample(self, batch_size, *args):
        states = {
            agent: np.array(
                [np.random.randn(*self.state_size) for _ in range(batch_size)]
            )
            for agent in self.agents
        }
        actions = {
            agent: np.array(
                [np.random.randn(*self.action_size) for _ in range(batch_size)]
            )
            for agent in self.agents
        }
        rewards = {
            agent: np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            for agent in self.agents
        }
        dones = {
            agent: np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            for agent in self.agents
        }
        next_states = {
            agent: np.array(
                [np.random.randn(*self.next_state_size) for _ in range(batch_size)]
            )
            for agent in self.agents
        }

        return states, actions, rewards, dones, next_states


@pytest.fixture
def env(state_size, action_size, vect):
    return DummyEnv(state_size, action_size, vect)


@pytest.fixture
def bandit_env(state_size, action_size):
    return DummyBanditEnv(state_size, action_size)


@pytest.fixture
def multi_env(state_size, action_size):
    return DummyMultiEnv(state_size, action_size)


@pytest.fixture
def population_off_policy(env):
    return [DummyAgentOffPolicy(5, env, 0.4) for _ in range(6)]


@pytest.fixture
def population_on_policy(env):
    return [DummyAgentOnPolicy(5, env) for _ in range(6)]


@pytest.fixture
def population_bandit(bandit_env):
    return [DummyBandit(5, bandit_env) for _ in range(6)]


@pytest.fixture
def population_multi_agent(multi_env):
    return [DummyMultiAgent(5, multi_env) for _ in range(6)]


@pytest.fixture
def tournament():
    return DummyTournament()


@pytest.fixture
def mutations():
    return DummyMutations()


@pytest.fixture
def memory():
    return DummyMemory()


@pytest.fixture
def n_step_memory():
    return DummyNStepMemory()


@pytest.fixture
def bandit_memory():
    return DummyBanditMemory()


@pytest.fixture
def multi_memory():
    return DummyMultiMemory()


@pytest.fixture
def mocked_agent_off_policy(env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.state_size = env.state_size
    mock_agent.action_size = 2
    mock_agent.action_dim = 2
    mock_agent.beta = 0.4
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.getAction.side_effect = lambda state: np.random.randn(env.action_size)
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: random.random()
    mock_agent.saveCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.loadCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_agent_on_policy(env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.state_size = env.state_size
    mock_agent.action_size = env.action_size
    mock_agent.beta = 0.4
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.getAction.side_effect = lambda state: tuple(
        np.random.randn(env.action_size) for _ in range(4)
    )
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: random.random()
    mock_agent.saveCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.loadCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_bandit(bandit_env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.state_size = bandit_env.state_size
    mock_agent.action_size = 2
    mock_agent.action_dim = 2
    mock_agent.beta = 0.4
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.regret = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.getAction.side_effect = lambda state: np.random.randint(
        bandit_env.action_size
    )
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: random.random()
    mock_agent.saveCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.loadCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_multi_agent(multi_env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.lr = 0.1
    mock_agent.agents = ["agent_0", "agent_1"]
    mock_agent.state_size = multi_env.state_size
    mock_agent.action_size = multi_env.action_size
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.discrete_actions = False

    def getAction(*args):
        return {
            agent: np.random.randn(mock_agent.action_size)
            for agent in mock_agent.agents
        }, None

    mock_agent.getAction.side_effect = getAction
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: {
        "actors": {"agent_0": random.random(), "agent_1": random.random()},
        "critics": {"agent_0": random.random(), "agent_1": random.random()},
    }
    mock_agent.saveCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.loadCheckpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10

    def save2memoryVectEnvs(states, actions, rewards, next_states, dones):
        if mock_memory.state_size is None:
            mock_memory.state_size, *_ = (state.shape for state in states)
            mock_memory.action_size, *_ = (action.shape for action in actions)
            mock_memory.next_state_size, *_ = (
                next_state.shape for next_state in next_states
            )
            mock_memory.counter += 1

        one_step_transition = (
            np.random.randn(*mock_memory.state_size),
            np.random.randn(*mock_memory.action_size),
            np.random.uniform(0, 400),
            np.random.choice([True, False]),
            np.random.randn(*mock_memory.next_state_size),
        )
        return one_step_transition

    mock_memory.save2memoryVectEnvs.side_effect = save2memoryVectEnvs

    def save2memory(state, action, reward, next_state, done, is_vectorised=False):
        if is_vectorised:
            mock_memory.save2memoryVectEnvs(state, action, reward, next_state, done)
        else:
            mock_memory.state_size = state.shape
            mock_memory.action_size = action.shape
            mock_memory.next_state_size = next_state.shape
            mock_memory.counter += 1

    # Assigning the save2memory function to the MagicMock
    mock_memory.save2memory.side_effect = save2memory

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, list):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*mock_memory.state_size)
            actions = np.random.randn(*mock_memory.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*mock_memory.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            actions = np.array(
                [np.random.randn(*mock_memory.action_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            next_states = np.array(
                [
                    np.random.randn(*mock_memory.next_state_size)
                    for _ in range(batch_size)
                ]
            )
        if beta is None:
            return states, actions, rewards, dones, next_states

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        return states, actions, rewards, dones, next_states, weights, idxs

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample

    def update_priorities(idxs, priorities):
        return None

    mock_memory.update_priorities.side_effect = update_priorities

    return mock_memory


@pytest.fixture
def mocked_n_step_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10000

    def save2memoryVectEnvs(state, action, reward, next_state, done):
        mock_memory.state_size = state.shape
        mock_memory.action_size = action.shape
        mock_memory.next_state_size = next_state.shape
        mock_memory.counter += 1

        one_step_transition = (
            np.random.randn(*mock_memory.state_size),
            np.random.randn(*mock_memory.action_size),
            np.random.uniform(0, 400),
            np.random.randn(*mock_memory.next_state_size),
            np.random.choice([True, False]),
        )
        return one_step_transition

    mock_memory.save2memoryVectEnvs.side_effect = save2memoryVectEnvs

    def save2memory(state, action, reward, next_state, done, is_vectorised):
        if is_vectorised:
            mock_memory.save2memoryVectEnvs(state, action, reward, next_state, done)
        else:
            mock_memory.state_size = state.shape
            mock_memory.action_size = action.shape
            mock_memory.next_state_size = next_state.shape
            mock_memory.counter += 1

    # Assigning the save2memory function to the MagicMock
    mock_memory.save2memory.side_effect = save2memory

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, list):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*mock_memory.state_size)
            actions = np.random.randn(*mock_memory.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*mock_memory.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            actions = np.array(
                [np.random.randn(*mock_memory.action_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            next_states = np.array(
                [
                    np.random.randn(*mock_memory.next_state_size)
                    for _ in range(batch_size)
                ]
            )
        if beta is None:
            return states, actions, rewards, dones, next_states

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        return states, actions, rewards, dones, next_states, weights, idxs

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample
    mock_memory.sample_n_step.side_effect = sample
    mock_memory.sample_per.side_effect = sample
    mock_memory.sample_from_indices.side_effect = sample

    return mock_memory


@pytest.fixture
def mocked_bandit_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.__len__.return_value = 10

    def save2memoryVectEnvs(states, rewards):
        if mock_memory.state_size is None:
            mock_memory.state_size, *_ = (state.shape for state in states)
            mock_memory.counter += 1

    mock_memory.save2memoryVectEnvs.side_effect = save2memoryVectEnvs

    def save2memory(state, reward, is_vectorised=False):
        if is_vectorised:
            mock_memory.save2memoryVectEnvs(state, reward)
        else:
            mock_memory.state_size = state.shape
            mock_memory.counter += 1

    # Assigning the save2memory function to the MagicMock
    mock_memory.save2memory.side_effect = save2memory

    def sample(batch_size, *args):
        if batch_size == 1:
            states = np.random.randn(*mock_memory.state_size)
            rewards = np.random.uniform(0, 400)
        else:
            states = np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])

        return states, rewards

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample

    return mock_memory


@pytest.fixture
def mocked_multi_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10000
    mock_memory.agents = ["agent_0", "agent_1"]

    def save2memory(state, action, reward, next_state, done, is_vectorised=False):
        mock_memory.state_size = list(state.values())[0].shape
        mock_memory.action_size = list(action.values())[0].shape
        mock_memory.next_state_size = list(next_state.values())[0].shape
        mock_memory.counter += 1

    # Assigning the save2memory function to the MagicMock
    mock_memory.save2memory.side_effect = save2memory

    def sample(batch_size, *args):
        states = {
            agent: np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            for agent in mock_memory.agents
        }
        actions = {
            agent: np.array(
                [np.random.randn(*mock_memory.action_size) for _ in range(batch_size)]
            )
            for agent in mock_memory.agents
        }
        rewards = {
            agent: np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            for agent in mock_memory.agents
        }
        dones = {
            agent: np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)]
            )
            for agent in mock_memory.agents
        }
        next_states = {
            agent: np.array(
                [
                    np.random.randn(*mock_memory.next_state_size)
                    for _ in range(batch_size)
                ]
            )
            for agent in mock_memory.agents
        }

        return states, actions, rewards, dones, next_states

    mock_memory.sample.side_effect = sample

    return mock_memory


@pytest.fixture
def mocked_env(state_size, action_size, vect=True, num_envs=2):
    mock_env = MagicMock()
    mock_env.state_size = state_size
    mock_env.action_size = action_size
    mock_env.vect = vect
    if mock_env.vect:
        mock_env.state_size = (num_envs,) + mock_env.state_size
        mock_env.n_envs = num_envs
        mock_env.num_envs = num_envs
    else:
        mock_env.n_envs = 1

    def reset():
        return np.random.rand(*mock_env.state_size), "info_string"

    mock_env.reset.side_effect = reset

    def step(action):
        return (
            np.random.rand(*mock_env.state_size),
            np.random.randint(0, 5, mock_env.n_envs),
            np.random.randint(0, 2, mock_env.n_envs),
            np.random.randint(0, 2, mock_env.n_envs),
            "info_string",
        )

    mock_env.step.side_effect = step

    return mock_env


@pytest.fixture
def mocked_bandit_env(state_size, action_size):
    mock_env = MagicMock()
    mock_env.state_size = (action_size,) + state_size
    mock_env.action_size = 1
    mock_env.n_envs = 1

    def reset():
        return np.random.rand(*mock_env.state_size)

    mock_env.reset.side_effect = reset

    def step(action):
        return (
            np.random.rand(*mock_env.state_size),
            np.random.rand(mock_env.n_envs),
        )

    mock_env.step.side_effect = step

    return mock_env


@pytest.fixture
def mocked_multi_env(state_size, action_size):
    mock_env = MagicMock()
    mock_env.state_size = state_size
    mock_env.action_size = action_size
    mock_env.agents = ["agent_0", "agent_1"]
    mock_env.reset.side_effect = lambda *args: (
        {agent: np.random.rand(*mock_env.state_size) for agent in mock_env.agents},
        {"info_string": None, "agent_mask": True, "env_defined_actions": True},
    )
    mock_env.step.side_effect = lambda *args: (
        {agent: np.random.rand(*mock_env.state_size) for agent in mock_env.agents},
        {agent: np.random.randint(0, 5) for agent in mock_env.agents},
        {agent: np.random.randint(0, 2) for agent in mock_env.agents},
        {agent: np.random.randint(0, 2) for agent in mock_env.agents},
        {"info_string": None},
    )

    return mock_env


@pytest.fixture
def mocked_mutations():
    mock_mutations = MagicMock()

    def mutation(pop, pre_training_mut=False):
        return pop

    mock_mutations.mutation.side_effect = mutation
    return mock_mutations


@pytest.fixture
def mocked_tournament():
    mock_tournament = MagicMock()

    def select(pop):
        return pop[0], pop

    mock_tournament.select.side_effect = select
    return mock_tournament


@pytest.fixture
def offline_init_hp():
    return {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "DOUBLE": False,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
        "DATASET": "../data/cartpole/cartpole_v1.1.0.h5",
    }


@pytest.fixture
def dummy_h5py_data(action_size, state_size):
    # Create a dummy h5py dataset
    dataset = {key: None for key in ["actions", "observations", "rewards"]}
    dataset["actions"] = np.array([np.random.randn(action_size) for _ in range(10)])
    dataset["observations"] = np.array(
        [np.random.randn(*state_size) for _ in range(10)]
    )
    dataset["rewards"] = np.array([np.random.randint(0, 5) for _ in range(10)])
    dataset["terminals"] = np.array(
        [np.random.choice([True, False]) for _ in range(10)]
    )

    return dataset


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_off_policy(env, population_off_policy, tournament, mutations, memory):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, per, n_step, algo",
    [
        ((6,), 2, True, False, False, DQN),
        ((6,), 2, False, False, False, DDPG),
        ((6,), 2, False, False, False, TD3),
        ((6,), 2, True, True, True, RainbowDQN),
    ],
)
def test_train_off_policy_agent_calls_made(
    env,
    algo,
    mocked_agent_off_policy,
    tournament,
    mutations,
    memory,
    per,
    n_step,
    n_step_memory,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        if not isinstance(algo, RainbowDQN):
            n_step_memory = None
            per = False
            n_step = False
        mock_population = [mocked_agent_off_policy for _ in range(6)]

        pop, pop_fitnesses = train_off_policy(
            env,
            "env_name",
            "algo",
            mock_population,
            memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            n_step=n_step,
            per=per,
            noisy=True,
            n_step_memory=n_step_memory,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            save_elite=True,
        )

        mocked_agent_off_policy.getAction.assert_called()
        mocked_agent_off_policy.learn.assert_called()
        mocked_agent_off_policy.test.assert_called()
        if accelerator is not None:
            mocked_agent_off_policy.wrap_models.assert_called()
            mocked_agent_off_policy.unwrap_models.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, False)])
def test_train_off_policy_save_elite_warning(
    env, population_off_policy, tournament, mutations, memory
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, False)])
def test_train_off_policy_checkpoint_warning(
    env, population_off_policy, tournament, mutations, memory
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, False)])
def test_actions_histogram(env, population_off_policy, tournament, mutations, memory):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "DQN",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_train_off_policy_replay_buffer_calls(
    mocked_memory, env, population_off_policy, tournament, mutations
):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        mocked_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_memory.save2memory.assert_called()
    mocked_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, per",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_off_policy_alternate_buffer_calls(
    env,
    mocked_memory,
    population_off_policy,
    tournament,
    mutations,
    mocked_n_step_memory,
    per,
):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=mocked_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=True,
        per=per,
        noisy=False,
        n_step_memory=mocked_n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_n_step_memory.save2memoryVectEnvs.assert_called()
    mocked_memory.save2memoryVectEnvs.assert_called()
    if per:
        mocked_n_step_memory.sample_from_indices.assert_called()
        mocked_memory.update_priorities.assert_called()
    else:
        mocked_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_off_policy_env_calls(
    mocked_env, memory, population_off_policy, tournament, mutations
):
    pop, pop_fitnesses = train_off_policy(
        mocked_env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_env.step.assert_called()
    mocked_env.reset.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_off_policy_tourn_mut_calls(
    env, memory, population_off_policy, mocked_tournament, mocked_mutations
):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_mutations.mutation.assert_called()
    mocked_tournament.select.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((250, 160, 3), 2, False)])
def test_train_off_policy_rgb_input(
    env, population_off_policy, tournament, mutations, memory
):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=False,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, per",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_off_policy_using_alternate_buffers(
    env, memory, population_off_policy, tournament, mutations, n_step_memory, per
):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=True,
        per=per,
        noisy=False,
        n_step_memory=n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((3, 64, 64), 2, True)])
def test_train_off_policy_using_alternate_buffers_rgb(
    env, memory, population_off_policy, tournament, mutations, n_step_memory
):
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=True,
        per=True,
        noisy=False,
        n_step_memory=n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_off_policy_distributed(
    env, population_off_policy, tournament, mutations, memory
):
    accelerator = Accelerator()
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_wandb_init_log(env, population_off_policy, tournament, mutations, memory):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_off_policy.wandb.login") as _, patch(
        "agilerl.training.train_off_policy.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_off_policy.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_off_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_off_policy.train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator",
    [
        ((6,), 2, True, True),
        ((6,), 2, True, False),
    ],
)
def test_wandb_init_log_distributed(
    env, population_off_policy, tournament, mutations, memory, accelerator
):
    if accelerator:
        accelerator = Accelerator()
    else:
        accelerator = None
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_off_policy.wandb.login") as _, patch(
        "agilerl.training.train_off_policy.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_off_policy.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_off_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_off_policy.train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            accelerator=accelerator,
            wandb_api_key="testing",
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_early_stop_wandb(env, population_off_policy, tournament, mutations, memory):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_off_policy.wandb.login") as _, patch(
        "agilerl.training.train_off_policy.wandb.init"
    ) as _, patch("agilerl.training.train_off_policy.wandb.log") as _, patch(
        "agilerl.training.train_off_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_off_policy.train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            target=-10000,
            swap_channels=False,
            n_episodes=110,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            n_step=False,
            per=False,
            noisy=True,
            n_step_memory=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_train_off_policy_save_elite(
    env, population_off_policy, tournament, mutations, memory
):
    elite_path = "checkpoint.pt"
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path=elite_path,
    )
    assert os.path.isfile(elite_path)
    os.remove(elite_path)


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_save_checkpoint(
    env, population_off_policy, tournament, mutations, memory, accelerator_flag
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = "checkpoint"
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        n_step=False,
        per=False,
        noisy=True,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{10}.pt")
        os.remove(f"{checkpoint_path}_{i}_{10}.pt")


@pytest.mark.parametrize("state_size, action_size, vect, algo", [((6,), 2, True, PPO)])
def test_train_on_policy_agent_calls_made(
    env, algo, mocked_agent_on_policy, tournament, mutations
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        mock_population = [mocked_agent_on_policy for _ in range(6)]
        pop, pop_fitnesses = train_on_policy(
            env,
            "env_name",
            "algo",
            mock_population,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        mocked_agent_on_policy.getAction.assert_called()
        mocked_agent_on_policy.learn.assert_called()
        mocked_agent_on_policy.test.assert_called()
        if accelerator is not None:
            mocked_agent_on_policy.wrap_models.assert_called()
            mocked_agent_on_policy.unwrap_models.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, False)])
def test_train_on_policy_save_elite_warning(
    env,
    population_on_policy,
    tournament,
    mutations,
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, False)])
def test_train_on_policy_checkpoint_warning(
    env,
    population_on_policy,
    tournament,
    mutations,
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_on_policy_env_calls(
    mocked_env, population_on_policy, tournament, mutations
):
    pop, pop_fitnesses = train_on_policy(
        mocked_env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_env.step.assert_called()
    mocked_env.reset.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_on_policy_tourn_mut_calls(
    env, population_on_policy, mocked_tournament, mocked_mutations
):
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_mutations.mutation.assert_called()
    mocked_tournament.select.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_on_policy(env, population_on_policy, tournament, mutations):
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize("state_size, action_size, vect", [((250, 160, 3), 2, False)])
def test_train_on_policy_rgb_input(env, population_on_policy, tournament, mutations):
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_on_policy_distributed(env, population_on_policy, tournament, mutations):
    accelerator = Accelerator()
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator",
    [((6,), 2, True, False), ((6,), 2, True, True)],
)
def test_wandb_init_log_on_policy(
    env, population_on_policy, tournament, mutations, accelerator
):
    if accelerator:
        accelerator = Accelerator()
    else:
        accelerator = None
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_on_policy.wandb.login") as _, patch(
        "agilerl.training.train_on_policy.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_on_policy.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_on_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_on_policy.train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            accelerator=accelerator,
            wandb_api_key="testing",
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called()
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", [((6,), 2, True)])
def test_early_stop_wandb_on_policy(env, population_on_policy, tournament, mutations):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_on_policy.wandb.login") as _, patch(
        "agilerl.training.train_on_policy.wandb.init"
    ) as _, patch("agilerl.training.train_on_policy.wandb.log") as _, patch(
        "agilerl.training.train_on_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_on_policy.train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            target=-10000,
            swap_channels=False,
            n_episodes=110,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_on_policy_save_elite(
    env, population_on_policy, tournament, mutations, accelerator_flag
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    elite_path = "elite"
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path=elite_path,
        accelerator=accelerator,
    )
    assert os.path.isfile(f"{elite_path}.pt")
    os.remove(f"{elite_path}.pt")


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_on_policy_save_checkpoint(
    env, population_on_policy, tournament, mutations, accelerator_flag
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = "checkpoint"
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{10}.pt")
        os.remove(f"{checkpoint_path}_{i}_{10}.pt")


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_agent(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        net_config=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_agent_distributed(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    accelerator = Accelerator()
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        net_config=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_multi_agent)


def test_train_multi_agent_agent_masking():
    pass


@pytest.mark.parametrize("state_size, action_size", [((250, 160, 3), 2)])
def test_train_multi_agent_rgb(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        net_config=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("state_size, action_size", [((250, 160, 3), 2)])
def test_train_multi_agent_rgb_vectorized(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    env = makeMultiAgentVectEnvs(multi_env)
    env.reset()
    pop, pop_fitnesses = train_multi_agent(
        env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        net_config=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_save_elite_warning(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_multi_agent(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            INIT_HP=None,
            MUT_P=None,
            net_config=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_checkpoint_warning(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_multi_agent(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            INIT_HP=None,
            MUT_P=None,
            net_config=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, False), ((6,), 2, True)]
)
def test_train_multi_wandb_init_log(
    multi_env,
    population_multi_agent,
    multi_memory,
    tournament,
    mutations,
    accelerator_flag,
):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_multi_agent.wandb.login") as _, patch(
        "agilerl.training.train_multi_agent.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_multi_agent.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_multi_agent.wandb.finish"
    ) as mock_wandb_finish:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent.train_multi_agent(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            multi_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            net_config=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            accelerator=accelerator,
            wandb_api_key="testing",
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called()
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_multi_agent_early_stop(
    multi_env, population_multi_agent, multi_memory, tournament, mutations
):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_multi_agent.wandb.login") as _, patch(
        "agilerl.training.train_multi_agent.wandb.init"
    ) as _, patch("agilerl.training.train_multi_agent.wandb.log") as _, patch(
        "agilerl.training.train_multi_agent.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent.train_multi_agent(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            multi_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            net_config=None,
            swap_channels=False,
            target=-10000,
            n_episodes=110,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, algo",
    [
        ((6,), 2, MADDPG),
        ((6,), 2, MATD3),
    ],
)
def test_train_multi_agent_calls(
    multi_env, mocked_multi_agent, multi_memory, tournament, mutations, algo
):
    for accelerator_flag in [False, True]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None

        mock_population = [mocked_multi_agent for _ in range(6)]

        pop, pop_fitnesses = train_multi_agent(
            multi_env,
            "env_name",
            "algo",
            mock_population,
            multi_memory,
            INIT_HP=None,
            MUT_P=None,
            net_config=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=50,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        mocked_multi_agent.getAction.assert_called()
        mocked_multi_agent.learn.assert_called()
        mocked_multi_agent.test.assert_called()
        if accelerator is not None:
            mocked_multi_agent.wrap_models.assert_called()
            mocked_multi_agent.unwrap_models.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_env_calls(
    mocked_multi_env, multi_memory, population_multi_agent, tournament, mutations
):
    pop, pop_fitnesses = train_multi_agent(
        mocked_multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_multi_env.step.assert_called()
    mocked_multi_env.reset.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_tourn_mut_calls(
    multi_env, multi_memory, population_multi_agent, mocked_tournament, mocked_mutations
):
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_tournament.select.assert_called()
    mocked_mutations.mutation.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_memory_calls(
    multi_env, mocked_multi_memory, population_multi_agent, tournament, mutations
):
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        mocked_multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_multi_memory.sample.assert_called()
    mocked_multi_memory.save2memory.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_multi_save_elite(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    elite_path = "elite"
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path=elite_path,
        accelerator=accelerator,
    )
    assert os.path.isfile(f"{elite_path}.pt")
    os.remove(f"{elite_path}.pt")


@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_multi_save_checkpoint(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = "checkpoint"
    pop, pop_fitnesses = train_multi_agent(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{10}.pt")
        os.remove(f"{checkpoint_path}_{i}_{10}.pt")


@pytest.mark.parametrize(
    "state_size, action_size, vect, swap_channels",
    [
        ((6,), 2, True, False),
        ((250, 160, 3), 2, False, True),
    ],
)
def test_train_offline(
    env,
    population_off_policy,
    memory,
    swap_channels,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None

        pop, pop_fitness = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            INIT_HP=offline_init_hp,
            MUT_P=None,
            swap_channels=swap_channels,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_offline_save_elite_warning(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitness = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            INIT_HP=offline_init_hp,
            MUT_P=None,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_offline_save_checkpoint_warning(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitness = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            INIT_HP=offline_init_hp,
            MUT_P=None,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_offline_wandb_calls(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        MUT_P = {
            "NO_MUT": 0.4,
            "ARCH_MUT": 0.2,
            "PARAMS_MUT": 0.2,
            "ACT_MUT": 0.2,
            "RL_HP_MUT": 0.2,
        }
        with patch("agilerl.training.train_offline.wandb.login") as _, patch(
            "agilerl.training.train_offline.wandb.init"
        ) as mock_wandb_init, patch(
            "agilerl.training.train_offline.wandb.log"
        ) as mock_wandb_log, patch(
            "agilerl.training.train_offline.wandb.finish"
        ) as mock_wandb_finish:
            # Call the function that should trigger wandb.init
            agilerl.training.train_offline.train_offline(
                env,
                "env_name",
                dummy_h5py_data,
                "algo",
                population_off_policy,
                memory,
                INIT_HP=offline_init_hp,
                MUT_P=MUT_P,
                swap_channels=False,
                n_episodes=10,
                max_steps=5,
                evo_epochs=1,
                evo_loop=1,
                tournament=tournament,
                mutation=mutations,
                wb=True,
                accelerator=accelerator,
                wandb_api_key="testing",
            )

            # Assert that wandb.init was called with expected arguments
            mock_wandb_init.assert_called_once_with(
                project=ANY,
                name=ANY,
                config=ANY,
            )
            # Assert that wandb.log was called with expected log parameters
            mock_wandb_log.assert_called()
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_offline_early_stop(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        MUT_P = {
            "NO_MUT": 0.4,
            "ARCH_MUT": 0.2,
            "PARAMS_MUT": 0.2,
            "ACT_MUT": 0.2,
            "RL_HP_MUT": 0.2,
        }
        with patch("agilerl.training.train_offline.wandb.login") as _, patch(
            "agilerl.training.train_offline.wandb.init"
        ) as _, patch("agilerl.training.train_offline.wandb.log") as _, patch(
            "agilerl.training.train_offline.wandb.finish"
        ) as mock_wandb_finish:
            # Call the function that should trigger wandb.init
            agilerl.training.train_offline.train_offline(
                env,
                "env_name",
                dummy_h5py_data,
                "algo",
                population_off_policy,
                memory,
                INIT_HP=offline_init_hp,
                MUT_P=MUT_P,
                swap_channels=False,
                n_episodes=110,
                target=-10000,
                max_steps=5,
                evo_epochs=1,
                evo_loop=1,
                tournament=tournament,
                mutation=mutations,
                wb=True,
                accelerator=accelerator,
                wandb_api_key="testing",
            )
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, algo",
    [
        ((6,), 2, True, CQN),
    ],
)
def test_offline_agent_calls(
    env,
    mocked_agent_off_policy,
    memory,
    algo,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        mock_population = [mocked_agent_off_policy for _ in range(6)]

        pop, pop_fitnesses = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            mock_population,
            memory,
            INIT_HP=offline_init_hp,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        mocked_agent_off_policy.learn.assert_called()
        mocked_agent_off_policy.test.assert_called()
        if accelerator is not None:
            mocked_agent_off_policy.wrap_models.assert_called()
            mocked_agent_off_policy.unwrap_models.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_offline_memory_calls(
    env,
    population_off_policy,
    mocked_memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None

        pop, pop_fitnesses = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            mocked_memory,
            INIT_HP=offline_init_hp,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )
        mocked_memory.save2memory.assert_called()
        mocked_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_offline_mut_tourn_calls(
    env,
    population_off_policy,
    memory,
    mocked_tournament,
    mocked_mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None

        pop, pop_fitnesses = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            INIT_HP=offline_init_hp,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=mocked_tournament,
            mutation=mocked_mutations,
            wb=False,
            accelerator=accelerator,
        )
        mocked_tournament.select.assert_called()
        mocked_mutations.mutation.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_offline_save_elite(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    elite_path = "elite"
    pop, pop_fitnesses = train_offline(
        env,
        "env_name",
        dummy_h5py_data,
        "algo",
        population_off_policy,
        memory,
        INIT_HP=offline_init_hp,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
        save_elite=True,
        elite_path=elite_path,
    )
    assert os.path.isfile(f"{elite_path}.pt")
    os.remove(f"{elite_path}.pt")


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_offline_save_checkpoint(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = "checkpoint"
    pop, pop_fitnesses = train_offline(
        env,
        "env_name",
        dummy_h5py_data,
        "algo",
        population_off_policy,
        memory,
        INIT_HP=offline_init_hp,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{10}.pt")
        os.remove(f"{checkpoint_path}_{i}_{10}.pt")


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_bandit(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_bandit)


@pytest.mark.parametrize(
    "state_size, action_size, algo",
    [
        ((6,), 2, NeuralTS),
        ((6,), 2, NeuralUCB),
    ],
)
def test_train_bandit_agent_calls_made(
    bandit_env,
    mocked_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        mock_population = [mocked_bandit for _ in range(6)]
        print(mocked_bandit)
        print(mock_population)

        pop, pop_fitnesses = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            mock_population,
            bandit_memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            save_elite=True,
        )

        mocked_bandit.getAction.assert_called()
        mocked_bandit.learn.assert_called()
        mocked_bandit.test.assert_called()
        if accelerator is not None:
            mocked_bandit.wrap_models.assert_called()
            mocked_bandit.unwrap_models.assert_called()


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_bandit_save_elite_warning(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_bandit_checkpoint_warning(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=5,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_bandit_actions_histogram(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "DQN",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_bandit)


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_bandit_replay_buffer_calls(
    mocked_bandit_memory, bandit_env, population_bandit, tournament, mutations
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        mocked_bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_bandit_memory.save2memory.assert_called()
    mocked_bandit_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_bandit_bandit_env_calls(
    mocked_bandit_env, bandit_memory, population_bandit, tournament, mutations
):
    pop, pop_fitnesses = train_bandits(
        mocked_bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_bandit_env.step.assert_called()
    mocked_bandit_env.reset.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_bandit_tourn_mut_calls(
    bandit_env, bandit_memory, population_bandit, mocked_tournament, mocked_mutations
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_mutations.mutation.assert_called()
    mocked_tournament.select.assert_called()


@pytest.mark.parametrize("state_size, action_size", [((250, 160, 3), 2)])
def test_train_bandit_rgb_input(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_bandit)


@pytest.mark.parametrize(
    "state_size, action_size",
    [((6,), 2)],
)
def test_train_bandit_using_alternate_buffers(
    bandit_env,
    bandit_memory,
    population_bandit,
    tournament,
    mutations,
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        memory=bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_bandit)


@pytest.mark.parametrize("state_size, action_size", [((3, 64, 64), 2)])
def test_train_bandit_using_alternate_buffers_rgb(
    bandit_env, bandit_memory, population_bandit, tournament, mutations
):
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        memory=bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_bandit)


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_bandit_distributed(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    accelerator = Accelerator()
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_bandit)


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_bandit_wandb_init_log(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 1,
        "LAMBDA": 1,
        "REG": 0.000625,
        "LEARN_STEP": 1,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_bandits.wandb.login") as _, patch(
        "agilerl.training.train_bandits.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_bandits.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_bandits.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_bandits.train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "steps_per_agent": ANY,
                "train/mean_score": ANY,
                "train/mean_regret": ANY,
                "train/best_regret": ANY,
                "train/mean_loss": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, accelerator",
    [
        ((6,), 2, True),
        ((6,), 2, False),
    ],
)
def test_bandit_wandb_init_log_distributed(
    bandit_env, population_bandit, tournament, mutations, bandit_memory, accelerator
):
    if accelerator:
        accelerator = Accelerator()
    else:
        accelerator = None
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 1,
        "LAMBDA": 1,
        "REG": 0.000625,
        "LEARN_STEP": 1,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_bandits.wandb.login") as _, patch(
        "agilerl.training.train_bandits.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_bandits.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_bandits.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_bandits.train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            n_episodes=10,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            accelerator=accelerator,
            wandb_api_key="testing",
        )

        # Assert that wandb.init was called with expected arguments
        mock_wandb_init.assert_called_once_with(
            project=ANY,
            name=ANY,
            config=ANY,
        )
        # Assert that wandb.log was called with expected log parameters
        mock_wandb_log.assert_called_with(
            {
                "global_step": ANY,
                "steps_per_agent": ANY,
                "train/mean_score": ANY,
                "train/mean_regret": ANY,
                "train/best_regret": ANY,
                "train/mean_loss": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            }
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_bandit_early_stop_wandb(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    INIT_HP = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 1,
        "LAMBDA": 1,
        "REG": 0.000625,
        "LEARN_STEP": 1,
        "CHANNELS_LAST": False,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    MUT_P = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with patch("agilerl.training.train_bandits.wandb.login") as _, patch(
        "agilerl.training.train_bandits.wandb.init"
    ) as _, patch("agilerl.training.train_bandits.wandb.log") as _, patch(
        "agilerl.training.train_bandits.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_bandits.train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            target=-10000,
            swap_channels=False,
            n_episodes=110,
            max_steps=5,
            evo_epochs=1,
            evo_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_bandit_save_elite(
    bandit_env, population_bandit, tournament, mutations, bandit_memory
):
    elite_path = "checkpoint.pt"
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path=elite_path,
    )
    assert os.path.isfile(elite_path)
    os.remove(elite_path)


@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag",
    [((6,), 2, True), ((6,), 2, False)],
)
def test_bandit_train_save_checkpoint(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = "checkpoint"
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        n_episodes=10,
        max_steps=5,
        evo_epochs=5,
        evo_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{10}.pt")
        os.remove(f"{checkpoint_path}_{i}_{10}.pt")
