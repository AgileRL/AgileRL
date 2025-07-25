import os
import random
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import dill
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import ParallelEnv
from tensordict import TensorDict

import agilerl.training.train_bandits
import agilerl.training.train_multi_agent_off_policy
import agilerl.training.train_off_policy
import agilerl.training.train_offline
import agilerl.training.train_on_policy
from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.components.data import Transition
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.training.train_bandits import train_bandits
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import make_multi_agent_vect_envs


class DummyEnv:
    def __init__(self, state_size, action_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = Discrete(action_size)
        self.vect = vect
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.num_envs = num_envs
            self.n_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


class DummyBanditEnv:
    def __init__(self, state_size, arms):
        self.arms = arms
        self.state_size = (arms,) + state_size
        self.action_size = 1
        self.num_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size)

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.rand(1),
        )


class DummyAgentOffPolicy:
    def __init__(self, batch_size, env, beta=None, algo="DQN"):
        self.algo = algo
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.action_dim = env.action_size
        self.batch_size = batch_size
        self.training = True
        self.beta = beta
        self.learn_step = 1
        self.scores = []
        self.steps = [0]
        self.fitness = []
        self.mut = "mutation"
        self.index = 1

    def set_training_mode(self, training):
        self.training = training

    def get_action(self, *args, **kwargs):
        return np.random.rand(self.action_size)

    def learn(self, experiences, n_experiences=None, per=False):
        if n_experiences is not None or per:
            return random.random(), None, None
        else:
            return random.random()

    def test(self, env, swap_channels, max_steps, loop):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def save_checkpoint(self, path):
        empty_dic = {}
        torch.save(empty_dic, path, pickle_module=dill)
        return True

    def load_checkpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return

    def reset_action_noise(self, *args, **kwargs):
        return


class DummyAgentOnPolicy(DummyAgentOffPolicy):
    def __init__(self, batch_size, env):
        super().__init__(batch_size, env)
        self.learn_step = 128
        self.action_space = Box(0, 1, (1,))
        self.actor = MagicMock()
        self.actor.squash_output = False
        self.actor.scale_action = lambda x: x
        self.actor.action_space = self.action_space

        self.registry = MagicMock()
        self.rollout_buffer = MagicMock()
        self.rollout_buffer.reset.side_effect = lambda: None
        self.rollout_buffer.add.side_effect = lambda *args, **kwargs: None
        self.registry.policy.side_effect = lambda: "actor"
        self.use_rollout_buffer = False
        self.num_envs = 2

    def learn(self, *args, **kwargs):
        return random.random()

    def get_action(self, *args, **kwargs):
        return tuple(np.random.randn(self.action_size) for _ in range(4))

    def _get_action_and_values(self, *args, **kwargs):
        return tuple(torch.randn(self.action_size) for _ in range(5))

    def test(self, env, swap_channels, max_steps, loop):
        return super().test(env, swap_channels, max_steps, loop)

    def preprocess_observation(self, obs):
        return obs

    def save_checkpoint(self, path):
        return super().save_checkpoint(path)

    def load_checkpoint(self, *args):
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

    def get_action(self, *args):
        return np.random.randint(self.action_size)

    def learn(self, experiences):
        return random.random()

    def test(self, env, swap_channels, max_steps, loop):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def save_checkpoint(self, path):
        empty_dic = {}
        torch.save(empty_dic, path, pickle_module=dill)
        return True

    def load_checkpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return


class DummyMultiEnv(ParallelEnv):
    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.state_size = self.state_dims
        self.action_dims = action_dims
        self.action_size = self.action_dims
        self.agents = ["agent_0", "other_agent_0"]
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.render_mode = None
        self.metadata = None
        self.info = {
            agent: {
                "env_defined_actions": (
                    None if agent == "other_agent_0" else np.array([0, 1])
                )
            }
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        return {
            agent: np.random.rand(*self.state_dims) for agent in self.agents
        }, self.info

    def step(self, action):
        return (
            {agent: np.random.rand(*self.state_dims) for agent in self.agents},
            {agent: np.random.randint(0, 5) for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            self.info,
        )

    def action_space(self, agent):
        return Discrete(5)

    def observation_space(self, agent):
        return Box(0, 255, self.state_dims)


class DummyMultiAgent(DummyAgentOffPolicy):
    def __init__(self, batch_size, env, on_policy, *args):
        super().__init__(batch_size, env, *args)
        self.agent_ids = ["agent_0", "other_agent_0"]
        self.shared_agent_ids = ["agent", "other_agent"]
        self.lr_actor = 0.001
        self.lr_critic = 0.01
        self.lr = 0.01
        self.num_envs = 1
        self.on_policy = on_policy
        self.actors = {
            "agent_0": MagicMock(),
            "other_agent_0": MagicMock(),
        }
        self.actors["agent_0"].squash_output = False
        self.actors["agent_0"].scale_action = lambda x: x
        self.actors["other_agent_0"].squash_output = False
        self.actors["other_agent_0"].scale_action = lambda x: x
        self.possible_action_spaces = Dict(
            {
                "agent_0": Discrete(2),
                "other_agent_0": Box(0, 1, (2,)),
            }
        )
        self.possible_observation_spaces = Dict(
            {
                "agent_0": Box(0, 1, env.state_dims),
                "other_agent_0": Box(0, 1, env.state_dims),
            }
        )
        self.action_space = deepcopy(self.possible_action_spaces)
        self.observation_space = deepcopy(self.possible_observation_spaces)

        self.registry = MagicMock()
        self.registry.policy.side_effect = lambda: "actors"

    def get_group_id(self, agent_id: str) -> str:
        return agent_id.rsplit("_", 1)[0] if isinstance(agent_id, str) else agent_id

    def has_grouped_agents(self) -> bool:
        return True

    def get_action(self, *args, **kwargs):
        output_dict = {
            agent: np.random.randn(self.num_envs, self.action_size)
            for agent in self.agent_ids
        }
        if self.on_policy:
            return output_dict, output_dict, output_dict, output_dict

        return output_dict, output_dict

    def learn(self, experiences):
        if self.on_policy:
            return {
                "agent_0": (random.random()),
                "other_agent_0": (random.random()),
            }
        return {
            "agent_0": (random.random(), random.random()),
            "other_agent_0": (random.random(), random.random()),
        }

    def test(self, env, swap_channels, max_steps, loop, sum_scores):
        rand_int = np.random.uniform(0, 400)
        rand_int = (rand_int / 2, rand_int / 2) if not sum_scores else rand_int
        self.fitness.append(rand_int)
        return rand_int

    def get_env_defined_actions(self, info, agents):
        env_defined_actions = {
            agent: info[agent].get("env_defined_action", None) for agent in agents
        }

        if all(eda is None for eda in env_defined_actions.values()):
            return
        return env_defined_actions

    def save_checkpoint(self, path):
        return super().save_checkpoint(path)

    def load_checkpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return

    def reset_action_noise(self, *args, **kwargs):
        return

    def assemble_grouped_outputs(self, agent_outputs, vect_dim):
        return {
            "agent_0": np.random.randn(vect_dim, self.action_size),
            "other_agent_0": np.random.randn(vect_dim, self.action_size),
        }

    def extract_inactive_agents(self, obs):
        return {}, obs


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


class DummyMemory(ReplayBuffer):
    def __init__(self):
        self.size = 0
        self.counter = 0
        self.state_size = None
        self.action_size = None
        self.next_state_size = None

    def add(self, data: TensorDict) -> None:
        return self.save_to_memory_vect_envs(data)

    def save_to_memory_vect_envs(self, data: TensorDict):
        if self.state_size is None:
            self.state_size = data["obs"].shape
            self.action_size = data["action"].shape
            self.next_state_size = data["next_obs"].shape

        self.size += 1
        self.counter += 1
        one_step_transition = Transition(
            obs=np.random.randn(*self.state_size),
            action=np.random.randn(*self.action_size),
            reward=np.random.uniform(0, 400),
            done=np.random.choice([True, False]),
            next_obs=np.random.randn(*self.next_state_size),
        )
        return one_step_transition.to_tensordict()

    def __len__(self):
        return 1000

    def sample(self, batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, (list, torch.Tensor)):
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

        sample_transition = TensorDict(
            {
                "obs": states,
                "action": actions,
                "reward": rewards,
                "next_obs": next_states,
                "done": dones,
            },
            batch_size=[batch_size],
        )
        if beta is None:
            return sample_transition

        idxs = [np.random.randn(1) for _ in range(batch_size)]
        weights = [i for i in range(batch_size)]

        sample_transition["weights"] = torch.tensor(weights)
        sample_transition["idxs"] = torch.tensor(idxs)

        return sample_transition

    def update_priorities(self, idxs, priorities):
        return


class DummyNStepMemory(DummyMemory, MultiStepReplayBuffer):
    def __init__(self):
        super().__init__()

    def save_to_memory_vect_envs(self, data: TensorDict):
        self.num_envs = data["obs"].shape[0]
        self.state_size = data["obs"].shape
        self.action_size = data["action"].shape
        self.next_state_size = data["next_obs"].shape
        self.size += 1

        one_step_transition = Transition(
            obs=np.random.randn(*self.state_size),
            action=np.random.randn(*self.action_size),
            reward=np.random.uniform(0, 400, self.num_envs),
            next_obs=np.random.randn(*self.next_state_size),
            done=np.random.choice([True, False], self.num_envs),
        )
        one_step_transition.batch_size = [self.num_envs]
        return one_step_transition.to_tensordict()

    def add(self, data: TensorDict):
        return self.save_to_memory_vect_envs(data)

    def __len__(self):
        return super().__len__()

    def sample_n_step(self, *args):
        return super().sample(*args)

    def sample_per(self, *args):
        return super().sample(*args)

    def sample_from_indices(self, *args):
        return super().sample(*args)


class DummyBanditMemory(ReplayBuffer):
    def __init__(self):
        self.counter = 0
        self.state_size = None
        self.size = 0
        self.action_size = 1

    def save_to_memory_vect_envs(self, data: TensorDict):
        if self.state_size is None:
            self.state_size, *_ = (state.shape for state in data["obs"])

        self.counter += 1
        self.size += 1

    def add(self, data: TensorDict):
        self.save_to_memory_vect_envs(data)

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

        sample_transition = TensorDict(
            {"obs": states, "reward": rewards}, batch_size=[batch_size]
        )
        return sample_transition


class DummyMultiMemory:
    def __init__(self):
        self.counter = 0
        self.state_size = None
        self.action_size = None
        self.next_state_size = None
        self.agents = ["agent_0", "other_agent_1"]

    def __len__(self):
        return 1000

    def save_to_memory(
        self, state, action, reward, next_state, done, is_vectorised=False
    ):
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
def population_multi_agent(multi_env, on_policy):
    return [DummyMultiAgent(5, multi_env, on_policy) for _ in range(6)]


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
    mock_agent.get_action.side_effect = (
        lambda state, *args, **kwargs: np.random.randint(
            env.action_size, size=(env.n_envs,)
        )
    )
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    if algo in [RainbowDQN]:
        mock_agent.learn.side_effect = lambda experiences, **kwargs: (
            random.random(),
            random.random(),
            random.random(),
        )
    else:
        mock_agent.learn.side_effect = lambda experiences, **kwargs: random.random()
    mock_agent.save_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.load_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None
    if algo in [DDPG, TD3]:
        mock_agent.reset_action_noise.side_effect = lambda *args, **kwargs: None
    mock_agent.algo = {
        DQN: "DQN",
        RainbowDQN: "Rainbow DQN",
        DDPG: "DDPG",
        TD3: "TD3",
        CQN: "CQN",
    }[algo]

    return mock_agent


@pytest.fixture
def mocked_agent_on_policy(env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.state_size = env.state_size
    mock_agent.action_size = env.action_size
    mock_agent.action_space = env.action_space
    mock_agent.beta = 0.4
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.get_action.side_effect = lambda state, *args, **kwargs: tuple(
        np.random.randn(env.action_size) for _ in range(4)
    )
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: random.random()
    mock_agent.save_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.load_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.algo = "PPO"

    mock_agent.registry = MagicMock()
    mock_agent.registry.policy = lambda: "actor"
    mock_agent.actor = MagicMock()
    mock_agent.actor.squash_output = False

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
    mock_agent.get_action.side_effect = (
        lambda state, *args, **kwargs: np.random.randint(bandit_env.action_size)
    )
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    mock_agent.learn.side_effect = lambda experiences: random.random()
    mock_agent.save_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.load_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None

    return mock_agent


@pytest.fixture
def mocked_multi_agent(multi_env, algo):
    mock_agent = MagicMock(spec=algo)
    mock_agent.learn_step = 1
    mock_agent.batch_size = 5
    mock_agent.lr = 0.1
    mock_agent.agent_ids = ["agent_0", "other_agent_0"]
    mock_agent.shared_agent_ids = ["agent", "other_agent"]
    mock_agent.state_size = multi_env.state_size
    mock_agent.action_size = multi_env.action_size
    mock_agent.scores = []
    mock_agent.steps = [0]
    mock_agent.fitness = []
    mock_agent.mut = "mutation"
    mock_agent.index = 1
    mock_agent.possible_action_spaces = Dict(
        {
            agent_id: multi_env.action_space(agent_id)
            for agent_id in mock_agent.agent_ids
        }
    )
    mock_agent.possible_observation_spaces = Dict(
        {
            agent_id: multi_env.observation_space(agent_id)
            for agent_id in mock_agent.agent_ids
        }
    )
    mock_agent.action_space = deepcopy(mock_agent.possible_action_spaces)
    mock_agent.observation_space = deepcopy(mock_agent.possible_observation_spaces)

    mock_agent.get_group_id.side_effect = lambda x: (
        x.rsplit("_", 1)[0] if isinstance(x, str) else x
    )
    mock_agent.registry = MagicMock()
    mock_agent.registry.policy.side_effect = lambda: "actors"
    mock_agent.has_grouped_agents.side_effect = lambda: algo == IPPO
    mock_agent.actors = {agent_id: MagicMock() for agent_id in mock_agent.agent_ids}

    def get_action(*args, **kwargs):
        out = {
            agent: np.random.randn(mock_agent.action_size)
            for agent in mock_agent.agent_ids
        }
        if algo == IPPO:
            return out, out, out, out
        return out, out

    mock_agent.get_action.side_effect = get_action
    mock_agent.test.side_effect = lambda *args, **kwargs: np.random.uniform(0, 400)
    if algo == IPPO:
        mock_agent.learn.side_effect = lambda experiences: {
            "agent_0": random.random(),
            "other_agent_0": random.random(),
        }
    else:
        mock_agent.learn.side_effect = lambda experiences: {
            "agent_0": (random.random(), random.random()),
            "other_agent_0": (random.random(), random.random()),
        }
    mock_agent.save_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.load_checkpoint.side_effect = lambda *args, **kwargs: None
    mock_agent.wrap_models.side_effect = lambda *args, **kwargs: None
    mock_agent.unwrap_models.side_effect = lambda *args, **kwargs: None
    if algo != IPPO:
        mock_agent.reset_action_noise.side_effect = lambda *args, **kwargs: None
    mock_agent.algo = {MADDPG: "MADDPG", MATD3: "MATD3", IPPO: "IPPO"}[algo]

    return mock_agent


@pytest.fixture
def mocked_per_memory():
    mock_memory = MagicMock(spec=PrioritizedReplayBuffer)
    mock_memory.counter = 0
    mock_memory.size = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10

    def add(data: TensorDict):
        if mock_memory.state_size is None:
            mock_memory.num_envs = data["obs"].shape[0]
            mock_memory.state_size = data["obs"].shape
            mock_memory.action_size = data["action"].shape
            mock_memory.next_state_size = data["next_obs"].shape

        mock_memory.counter += 1
        mock_memory.size += 1

        one_step_transition = Transition(
            obs=np.random.randn(*mock_memory.state_size),
            action=np.random.randn(*mock_memory.action_size),
            reward=np.random.uniform(0, 400, mock_memory.num_envs),
            done=np.random.choice([True, False], mock_memory.num_envs),
            next_obs=np.random.randn(*mock_memory.next_state_size),
        )
        return one_step_transition.to_tensordict()

    # Assigning the save_to_memory function to the MagicMock
    mock_memory.add.side_effect = add

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, (list, torch.Tensor)):
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

        sample_transition = Transition(
            obs=states,
            action=actions,
            reward=rewards,
            done=dones,
            next_obs=next_states,
            batch_size=[batch_size],
        ).to_tensordict()

        sample_transition["weights"] = torch.tensor(weights)
        sample_transition["idxs"] = torch.tensor(idxs)

        return sample_transition

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample

    def update_priorities(idxs, priorities):
        return None

    mock_memory.update_priorities.side_effect = update_priorities

    return mock_memory


@pytest.fixture
def mocked_memory():
    mock_memory = MagicMock(spec=ReplayBuffer)
    mock_memory.counter = 0
    mock_memory.size = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10

    def add(data: TensorDict):
        if mock_memory.state_size is None:
            mock_memory.num_envs = data["obs"].shape[0]
            mock_memory.state_size = data["obs"].shape
            mock_memory.action_size = data["action"].shape
            mock_memory.next_state_size = data["next_obs"].shape

        mock_memory.counter += 1
        mock_memory.size += 1

        one_step_transition = Transition(
            obs=np.random.randn(*mock_memory.state_size),
            action=np.random.randn(*mock_memory.action_size),
            reward=np.random.uniform(0, 400, mock_memory.num_envs),
            done=np.random.choice([True, False], mock_memory.num_envs),
            next_obs=np.random.randn(*mock_memory.next_state_size),
        )
        return one_step_transition.to_tensordict()

    # Assigning the save_to_memory function to the MagicMock
    mock_memory.add.side_effect = add

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, (list, torch.Tensor)):
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

        sample_transition = Transition(
            obs=states,
            action=actions,
            reward=rewards,
            done=dones,
            next_obs=next_states,
            batch_size=[batch_size],
        ).to_tensordict()

        if beta is None:
            return sample_transition

        idxs = [np.random.randn(1) for _ in range(batch_size)]

        sample_transition["idxs"] = torch.tensor(idxs)

        return sample_transition

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample

    return mock_memory


@pytest.fixture
def mocked_n_step_memory():
    mock_memory = MagicMock(spec=MultiStepReplayBuffer)
    mock_memory.counter = 0
    mock_memory.size = 0
    mock_memory.state_size = None
    mock_memory.action_size = None
    mock_memory.next_state_size = None
    mock_memory.__len__.return_value = 10000

    def add(data: TensorDict):
        if mock_memory.state_size is None:
            mock_memory.num_envs = data["obs"].shape[0]
            mock_memory.state_size = data["obs"].shape
            mock_memory.action_size = data["action"].shape
            mock_memory.next_state_size = data["next_obs"].shape

        mock_memory.size += 1
        mock_memory.counter += 1

        one_step_transition = Transition(
            obs=np.random.randn(*mock_memory.state_size),
            action=np.random.randn(*mock_memory.action_size),
            reward=np.random.uniform(0, 400, mock_memory.num_envs),
            done=np.random.choice([True, False], mock_memory.num_envs),
            next_obs=np.random.randn(*mock_memory.next_state_size),
        )
        return one_step_transition.to_tensordict()

    # Assigning the save_to_memory function to the MagicMock
    mock_memory.add.side_effect = add

    def sample(batch_size, beta=None, *args):
        # Account for sample_from_indices
        if isinstance(batch_size, (list, torch.Tensor)):
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

        sample_transition = Transition(
            obs=states,
            action=actions,
            reward=rewards,
            done=dones,
            next_obs=next_states,
            batch_size=[batch_size],
        ).to_tensordict()

        sample_transition["weights"] = torch.tensor(weights)
        sample_transition["idxs"] = torch.tensor(idxs)

        return sample_transition

    # Assigning the sample function to the MagicMock
    mock_memory.sample.side_effect = sample
    mock_memory.sample_from_indices.side_effect = sample

    return mock_memory


@pytest.fixture
def mocked_bandit_memory():
    mock_memory = MagicMock()
    mock_memory.counter = 0
    mock_memory.state_size = None
    mock_memory.__len__.return_value = 10

    def add(data: TensorDict):
        if mock_memory.state_size is None:
            mock_memory.state_size = data["obs"].shape
            mock_memory.counter += 1

    mock_memory.add.side_effect = add

    def sample(batch_size, *args):
        if batch_size == 1:
            states = np.random.randn(*mock_memory.state_size)
            rewards = np.random.uniform(0, 400)
        else:
            states = np.array(
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)]
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])

        sample_transition = TensorDict(
            {"obs": states, "reward": rewards}, batch_size=[batch_size]
        )
        return sample_transition

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
    mock_memory.agents = ["agent_0", "other_agent_0"]

    def save_to_memory(state, action, reward, next_state, done, is_vectorised=False):
        mock_memory.state_size = list(state.values())[0].shape
        mock_memory.action_size = list(action.values())[0].shape
        mock_memory.next_state_size = list(next_state.values())[0].shape
        mock_memory.counter += 1

    # Assigning the save_to_memory function to the MagicMock
    mock_memory.save_to_memory.side_effect = save_to_memory

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
        mock_env.num_envs = num_envs
    else:
        mock_env.num_envs = 1

    def reset():
        return np.random.rand(*mock_env.state_size), {}

    mock_env.reset.side_effect = reset

    def step(action):
        return (
            np.random.rand(*mock_env.state_size),
            np.random.randint(0, 5, mock_env.num_envs),
            np.random.randint(0, 2, mock_env.num_envs),
            np.random.randint(0, 2, mock_env.num_envs),
            {},
        )

    mock_env.step.side_effect = step

    return mock_env


@pytest.fixture
def mocked_bandit_env(state_size, action_size):
    mock_env = MagicMock()
    mock_env.state_size = (action_size,) + state_size
    mock_env.action_size = 1
    mock_env.num_envs = 1

    def reset():
        return np.random.rand(*mock_env.state_size)

    mock_env.reset.side_effect = reset

    def step(action):
        return (
            np.random.rand(*mock_env.state_size),
            np.random.rand(mock_env.num_envs),
        )

    mock_env.step.side_effect = step

    return mock_env


@pytest.fixture
def mocked_multi_env(state_size, action_size):
    mock_env = MagicMock(spec=DummyMultiEnv)
    mock_env.state_size = state_size
    mock_env.action_size = action_size
    mock_env.agents = ["agent_0", "other_agent_0"]
    mock_env.reset.side_effect = lambda *args: (
        {agent: np.random.rand(*mock_env.state_size) for agent in mock_env.agents},
        {
            agent: {
                "env_defined_actions": (
                    None if agent == "other_agent_0" else np.array([0, 1])
                )
            }
            for agent in mock_env.agents
        },
    )
    mock_env.step.side_effect = lambda *args: (
        {agent: np.random.rand(*mock_env.state_size) for agent in mock_env.agents},
        {agent: np.random.randint(0, 5) for agent in mock_env.agents},
        {agent: np.random.randint(0, 2) for agent in mock_env.agents},
        {agent: np.random.randint(0, 2) for agent in mock_env.agents},
        {
            agent: {
                "env_defined_actions": (
                    None if agent == "other_agent_0" else np.array([0, 1])
                )
            }
            for agent in mock_env.agents
        },
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_off_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect, per, n_step, algo, num_envs, learn_step",
    [
        ((6,), 2, False, False, False, DQN, 1, 2),
        ((6,), 2, False, False, False, DDPG, 1, 2),
        ((6,), 2, False, False, False, TD3, 1, 2),
        ((6,), 2, True, False, False, DQN, 2, 1),
        ((6,), 2, True, False, False, DDPG, 2, 1),
        ((6,), 2, True, False, False, TD3, 2, 1),
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
    num_envs,
    learn_step,
):
    for accelerator_flag in [True, False]:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        n_step_memory = None
        per = False
        n_step = False
        mock_population = [mocked_agent_off_policy for _ in range(6)]
        for agent in mock_population:
            agent.learn_step = learn_step

        if env.vect:
            env.num_envs = num_envs

        pop, pop_fitnesses = train_off_policy(
            env,
            "env_name",
            "algo",
            mock_population,
            memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step=n_step,
            per=per,
            n_step_memory=n_step_memory,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            save_elite=True,
        )

        mocked_agent_off_policy.get_action.assert_called()
        mocked_agent_off_policy.learn.assert_called()
        mocked_agent_off_policy.test.assert_called()
        if accelerator is not None:
            mocked_agent_off_policy.wrap_models.assert_called()
            mocked_agent_off_policy.unwrap_models.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, algo, num_envs, learn_step, n_step, per",
    [
        ((6,), 2, False, RainbowDQN, 1, 2, True, False),
        ((6,), 2, True, RainbowDQN, 2, 1, True, False),
        ((6,), 2, False, RainbowDQN, 1, 2, True, True),
        ((6,), 2, True, RainbowDQN, 2, 1, True, True),
        ((6,), 2, False, RainbowDQN, 1, 2, False, False),
        ((6,), 2, True, RainbowDQN, 2, 1, False, False),
    ],
)
def test_train_off_policy_agent_calls_made_rainbow(
    env,
    algo,
    mocked_agent_off_policy,
    tournament,
    mutations,
    memory,
    per,
    n_step,
    n_step_memory,
    num_envs,
    learn_step,
):
    accelerator = None
    n_step_memory = n_step_memory if n_step else None
    mock_population = [mocked_agent_off_policy for _ in range(6)]
    for agent in mock_population:
        agent.learn_step = learn_step
    env.n_envs = num_envs

    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "Rainbow DQN",
        mock_population,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=n_step,
        per=per,
        n_step_memory=n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
        save_elite=True,
    )

    mocked_agent_off_policy.get_action.assert_called()
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step=False,
            per=False,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step=False,
            per=False,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_memory.add.assert_called()
    mocked_memory.sample.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, per",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_off_policy_alternate_buffer_calls(
    env,
    mocked_memory,
    mocked_per_memory,
    population_off_policy,
    tournament,
    mutations,
    mocked_n_step_memory,
    per,
):
    mocked_memory = mocked_memory if not per else mocked_per_memory
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=mocked_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=True,
        per=per,
        n_step_memory=mocked_n_step_memory,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_n_step_memory.add.assert_called()
    mocked_memory.add.assert_called()
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=True,
        per=per,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=True,
        per=True,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step=False,
            per=False,
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
                "fps": ANY,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step=False,
            per=False,
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
                "fps": ANY,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step=False,
            per=False,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
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
    env, population_off_policy, tournament, mutations, memory, accelerator_flag, tmpdir
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, pop_fitnesses = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step=False,
        per=False,
        n_step_memory=None,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")
        os.remove(f"{checkpoint_path}_{i}_{50}.pt")


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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        mocked_agent_on_policy.get_action.assert_called()
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_mutations.mutation.assert_called()
    mocked_tournament.select.assert_called()


@pytest.mark.parametrize(
    "state_size, action_size, vect, use_rollout_buffer",
    [((6,), 2, True, True), ((6,), 2, False, False)],
)
def test_train_on_policy(
    env, population_on_policy, tournament, mutations, use_rollout_buffer
):
    if use_rollout_buffer:
        for agent in population_on_policy:
            agent.use_rollout_buffer = True

    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=256,
        evo_steps=256,
        eval_loop=1,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
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
            max_steps=50,
            evo_steps=10,
            eval_loop=1,
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
            max_steps=500,
            evo_steps=10,
            eval_loop=1,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
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
    env, population_on_policy, tournament, mutations, accelerator_flag, tmpdir
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, pop_fitnesses = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=500,
        evo_steps=500,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{512}.pt")
        os.remove(f"{checkpoint_path}_{i}_{512}.pt")


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, sum_scores", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_multi_agent_off_policy(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
    sum_scores,
):
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        sum_scores=sum_scores,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, sum_scores, swap_channels",
    [((6,), 2, True, False), ((6,), 2, False, False), ((250, 160, 3), 2, False, True)],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_train_multi_agent_on_policy(
    multi_env,
    population_multi_agent,
    on_policy,
    tournament,
    mutations,
    sum_scores,
    swap_channels,
    accelerator,
):
    pop, pop_fitnesses = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=swap_channels,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        sum_scores=sum_scores,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_agent_off_policy_distributed(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
):
    accelerator = Accelerator()
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        accelerator=accelerator,
    )

    assert len(pop) == len(population_multi_agent)


def test_train_multi_agent_off_policy_agent_masking():
    pass


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", [((250, 160, 3), 2)])
def test_train_multi_agent_off_policy_rgb(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
):
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=True,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", [((250, 160, 3), 2)])
def test_train_multi_agent_off_policy_rgb_vectorized(
    multi_env,
    population_multi_agent,
    multi_memory,
    on_policy,
    tournament,
    mutations,
    state_size,
    action_size,
):
    env = make_multi_agent_vect_envs(
        DummyMultiEnv, num_envs=4, state_dims=state_size, action_dims=action_size
    )
    for agent in population_multi_agent:
        agent.num_envs = 4
        agent.scores = [1]
    env.reset()
    pop, pop_fitnesses = train_multi_agent_off_policy(
        env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=10,
        evo_steps=5,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
    )
    assert len(pop) == len(population_multi_agent)
    env.close()


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize("state_size, action_size", [((250, 160, 3), 2)])
def test_train_multi_agent_on_policy_rgb_vectorized(
    multi_env,
    population_multi_agent,
    multi_memory,
    on_policy,
    tournament,
    mutations,
    state_size,
    action_size,
):
    env = make_multi_agent_vect_envs(
        DummyMultiEnv, num_envs=4, state_dims=state_size, action_dims=action_size
    )
    for agent in population_multi_agent:
        agent.num_envs = 4
        agent.scores = [1]
    env.reset()
    pop, pop_fitnesses = train_multi_agent_on_policy(
        env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=10,
        evo_steps=5,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
    )
    assert len(pop) == len(population_multi_agent)
    env.close()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_save_elite_warning(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_save_elite_warning_on_policy(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
):
    warning_string = "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_checkpoint_warning(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize("state_size, action_size", [((6,), 2)])
def test_train_multi_checkpoint_warning_on_policy(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
):
    warning_string = "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    with pytest.warns(match=warning_string):
        pop, pop_fitnesses = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, False), ((6,), 2, True)]
)
def test_train_multi_wandb_init_log(
    multi_env,
    population_multi_agent,
    multi_memory,
    on_policy,
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
    with patch("agilerl.training.train_multi_agent_off_policy.wandb.login") as _, patch(
        "agilerl.training.train_multi_agent_off_policy.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_multi_agent_off_policy.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_multi_agent_off_policy.wandb.finish"
    ) as mock_wandb_finish:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            multi_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            max_steps=50,
            evo_steps=10,
            eval_loop=1,
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


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, False), ((6,), 2, True)]
)
def test_train_multi_wandb_init_log_on_policy(
    multi_env,
    population_multi_agent,
    multi_memory,
    on_policy,
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
    with patch("agilerl.training.train_multi_agent_on_policy.wandb.login") as _, patch(
        "agilerl.training.train_multi_agent_on_policy.wandb.init"
    ) as mock_wandb_init, patch(
        "agilerl.training.train_multi_agent_on_policy.wandb.log"
    ) as mock_wandb_log, patch(
        "agilerl.training.train_multi_agent_on_policy.wandb.finish"
    ) as mock_wandb_finish:
        if accelerator_flag:
            accelerator = Accelerator()
        else:
            accelerator = None
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_on_policy.train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            max_steps=50,
            evo_steps=10,
            eval_loop=1,
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


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_multi_agent_early_stop(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
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
    with patch("agilerl.training.train_multi_agent_off_policy.wandb.login") as _, patch(
        "agilerl.training.train_multi_agent_off_policy.wandb.init"
    ) as _, patch(
        "agilerl.training.train_multi_agent_off_policy.wandb.log"
    ) as _, patch(
        "agilerl.training.train_multi_agent_off_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            multi_memory,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            target=-10000,
            max_steps=500,
            evo_steps=10,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_multi_agent_early_stop_on_policy(
    multi_env, population_multi_agent, on_policy, multi_memory, tournament, mutations
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
    with patch("agilerl.training.train_multi_agent_on_policy.wandb.login") as _, patch(
        "agilerl.training.train_multi_agent_on_policy.wandb.init"
    ) as _, patch("agilerl.training.train_multi_agent_on_policy.wandb.log") as _, patch(
        "agilerl.training.train_multi_agent_on_policy.wandb.finish"
    ) as mock_wandb_finish:
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_on_policy.train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            INIT_HP=INIT_HP,
            MUT_P=MUT_P,
            swap_channels=False,
            target=-10000,
            max_steps=500,
            evo_steps=10,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            wandb_api_key="testing",
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, algo, accelerator_flag",
    [
        ((6,), 2, MADDPG, False),
        ((6,), 2, MATD3, True),
    ],
)
def test_train_multi_agent_off_policy_calls(
    multi_env,
    mocked_multi_agent,
    multi_memory,
    on_policy,
    tournament,
    mutations,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None

    mock_population = [mocked_multi_agent for _ in range(6)]

    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        mock_population,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    for agent in mock_population:
        agent.get_action.assert_called()
        agent.learn.assert_called()
        agent.test.assert_called()
        if accelerator is not None:
            agent.wrap_models.assert_called()
            agent.unwrap_models.assert_called()


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, algo, accelerator_flag",
    [
        ((6,), 2, IPPO, False),
        ((6,), 2, IPPO, True),
    ],
)
def test_train_multi_agent_onpolicy_calls(
    multi_env,
    mocked_multi_agent,
    multi_memory,
    on_policy,
    tournament,
    mutations,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None

    mock_population = [mocked_multi_agent for _ in range(6)]

    pop, pop_fitnesses = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        mock_population,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
    )

    for agent in mock_population:
        agent.get_action.assert_called()
        agent.learn.assert_called()
        agent.test.assert_called()
        if accelerator is not None:
            agent.wrap_models.assert_called()
            agent.unwrap_models.assert_called()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_env_calls(
    mocked_multi_env,
    multi_memory,
    population_multi_agent,
    on_policy,
    tournament,
    mutations,
):
    pop, pop_fitnesses = train_multi_agent_off_policy(
        mocked_multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_multi_env.step.assert_called()
    mocked_multi_env.reset.assert_called()


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_env_calls_on_policy(
    mocked_multi_env,
    multi_memory,
    population_multi_agent,
    on_policy,
    tournament,
    mutations,
):
    pop, pop_fitnesses = train_multi_agent_on_policy(
        mocked_multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_multi_env.step.assert_called()
    mocked_multi_env.reset.assert_called()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_tourn_mut_calls(
    multi_env,
    multi_memory,
    population_multi_agent,
    on_policy,
    mocked_tournament,
    mocked_mutations,
):
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_tournament.select.assert_called()
    mocked_mutations.mutation.assert_called()


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_tourn_mut_calls_on_policy(
    multi_env,
    multi_memory,
    population_multi_agent,
    on_policy,
    mocked_tournament,
    mocked_mutations,
):
    pop, pop_fitnesses = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=mocked_tournament,
        mutation=mocked_mutations,
        wb=False,
    )
    mocked_tournament.select.assert_called()
    mocked_mutations.mutation.assert_called()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_multi_memory_calls(
    multi_env,
    mocked_multi_memory,
    population_multi_agent,
    on_policy,
    tournament,
    mutations,
):
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        mocked_multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_multi_memory.sample.assert_called()
    mocked_multi_memory.save_to_memory.assert_called()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_multi_save_elite(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    on_policy,
    accelerator_flag,
):
    accelerator = Accelerator() if accelerator_flag else None
    elite_path = "elite"
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path=elite_path,
        accelerator=accelerator,
    )
    assert os.path.isfile(f"{elite_path}.pt")
    os.remove(f"{elite_path}.pt")


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_multi_save_elite_on_policy(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    on_policy,
    accelerator_flag,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    elite_path = "elite"
    pop, pop_fitnesses = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        save_elite=True,
        elite_path=elite_path,
        accelerator=accelerator,
    )
    assert os.path.isfile(f"{elite_path}.pt")
    os.remove(f"{elite_path}.pt")


@pytest.mark.parametrize("on_policy", [False])
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
    tmpdir,
):
    accelerator = Accelerator() if accelerator_flag else None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, pop_fitnesses = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")
        os.remove(f"{checkpoint_path}_{i}_{50}.pt")


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_multi_save_checkpoint_on_policy(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    accelerator_flag,
    tmpdir,
):
    accelerator = Accelerator() if accelerator_flag else None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, pop_fitnesses = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")
        os.remove(f"{checkpoint_path}_{i}_{50}.pt")


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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [
        ((6,), 2, True, False),
        ((6,), 2, True, True),
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
    accelerator_flag,
):
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
            max_steps=50,
            evo_steps=10,
            eval_loop=1,
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
                target=-10000,
                max_steps=50,
                evo_steps=10,
                eval_loop=1,
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
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
        accelerator = Accelerator() if accelerator_flag else None
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )
        mocked_memory.add.assert_called()
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
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
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
    tmpdir,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
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
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        accelerator=accelerator,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
    )
    for i in range(6):  # iterate through the population indices
        assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")
        os.remove(f"{checkpoint_path}_{i}_{50}.pt")


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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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

        pop, pop_fitnesses = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            mock_population,
            bandit_memory,
            INIT_HP=None,
            MUT_P=None,
            swap_channels=False,
            max_steps=50,
            episode_steps=5,
            evo_steps=25,
            eval_steps=5,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            save_elite=True,
        )

        mocked_bandit.get_action.assert_called()
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
            max_steps=50,
            episode_steps=5,
            evo_steps=25,
            eval_steps=5,
            eval_loop=1,
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
            max_steps=50,
            episode_steps=5,
            evo_steps=25,
            eval_steps=5,
            eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_bandit_memory.add.assert_called()
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
            max_steps=50,
            episode_steps=5,
            evo_steps=25,
            eval_steps=5,
            eval_loop=1,
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
            max_steps=50,
            episode_steps=5,
            evo_steps=25,
            eval_steps=5,
            eval_loop=1,
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
            max_steps=550,
            episode_steps=5,
            evo_steps=25,
            eval_steps=5,
            eval_loop=1,
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
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
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
    tmpdir,
):
    if accelerator_flag:
        accelerator = Accelerator()
    else:
        accelerator = None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, pop_fitnesses = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        INIT_HP=None,
        MUT_P=None,
        swap_channels=False,
        max_steps=50,
        episode_steps=5,
        evo_steps=25,
        eval_steps=5,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
        checkpoint=10,
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
    )
    for i in range(6):  # iterate through the population indices
        for s in range(5):
            assert os.path.isfile(f"{checkpoint_path}_{i}_{10*(s+1)}.pt")
            os.remove(f"{checkpoint_path}_{i}_{10*(s+1)}.pt")


# LEAVE LAST, TEMPORARY TO DELETE SAVED MODELS
# TODO: Properly handle saving/deletion in tests
def test_remove_saved_models():
    if os.path.exists("models"):
        shutil.rmtree("models")
