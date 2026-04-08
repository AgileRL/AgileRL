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
import agilerl
import agilerl.rollouts.on_policy
from accelerate import Accelerator
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import ParallelEnv
from tensordict import TensorDict

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
from agilerl.algorithms.core.base import MultiAgentRLAlgorithm
from agilerl.metrics import AgentMetrics, MultiAgentMetrics
from agilerl.training.train_bandits import train_bandits
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import make_multi_agent_vect_envs

# Common parametrize constants
_FLAT_VECT = [((6,), 2, True)]
_FLAT_NOVECT = [((6,), 2, False)]
_FLAT_BOTH = [((6,), 2, True), ((6,), 2, False)]
_IMG_NOVECT = [((250, 160, 3), 2, False)]
_IMG_VECT = [((3, 64, 64), 2, True)]
_FLAT = [((6,), 2)]
_IMG = [((250, 160, 3), 2)]
_IMG_SQUARE = [((3, 64, 64), 2)]


class DummyEnv:
    def __init__(self, state_size, action_size, vect=True, num_envs=2):
        self._single_state_size = tuple(state_size)
        self.state_size = state_size
        self.action_size = action_size
        self.observation_space = Box(0.0, 1.0, self._single_state_size)
        self.action_space = Box(-1.0, 1.0, (action_size,))
        self.vect = vect
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.num_envs = num_envs
            self.n_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self, seed=None, options=None):
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        if not self.vect:
            return (
                np.random.rand(*self.state_size),
                float(np.random.randint(0, 5)),
                bool(np.random.randint(0, 2)),
                bool(np.random.randint(0, 2)),
                {},
            )
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
        self.observation_space = Box(0.0, 1.0, self.state_size)
        self.action_space = Discrete(self.action_size)
        self.num_envs = 1

    def reset(self, seed=None, options=None):
        return np.random.rand(*self.state_size)

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.rand(1),
        )


class DummyAgentOffPolicy:
    def __init__(
        self,
        batch_size,
        env,
        beta=None,
        algo="DQN",
        action_space=None,
        learn_step=1,
        actor=None,
    ):
        self.algo = algo
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.action_dim = env.action_size
        self.action_space = (
            action_space if action_space is not None else Discrete(env.action_size)
        )
        self.batch_size = batch_size
        self.training = True
        self.beta = beta
        self.learn_step = learn_step
        self.metrics = AgentMetrics()
        self.scores = self.metrics.scores
        self.steps = [self.metrics.steps]
        self.fitness = []
        self.steps_per_second = 0.0
        self.mut = "mutation"
        self.index = 1
        self.registry = MagicMock()
        self.registry.hp_config = None
        # Attributes required by train_off_policy for continuous action agents (DDPG/TD3)
        self.action_low = torch.as_tensor(
            [-1.0] * self.action_size,
            dtype=torch.float32,
        )
        self.action_high = torch.as_tensor(
            [1.0] * self.action_size,
            dtype=torch.float32,
        )
        self.actor = actor if actor is not None else MagicMock()
        self.actor.output_activation = "Tanh"

    def set_training_mode(self, training):
        self.training = training

    def get_action(self, *args, **kwargs):
        obs = args[0] if args else kwargs.get("obs")
        num_envs = (
            int(obs.shape[0]) if isinstance(obs, np.ndarray) and obs.ndim > 1 else 1
        )
        return np.random.rand(num_envs, self.action_size).astype(np.float32)

    def learn(self, experiences, n_experiences=None, per=False):
        loss = random.random()
        if n_experiences is not None or per:
            return loss, None, None
        return loss

    def test(self, env, max_steps=None, loop=3, **kwargs):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def init_evo_step(self):
        self.metrics.init_evo_step()

    def add_scores(self, scores):
        self.metrics.add_scores(scores)
        self.scores = self.metrics.scores

    def finalize_evo_step(self, num_steps):
        self.metrics.finalize_evo_step(num_steps)
        self.steps_per_second = self.metrics.steps_per_second
        self.steps = [self.metrics.steps]

    def save_checkpoint(self, path):
        torch.save({}, path, pickle_module=dill)
        return True

    def load_checkpoint(self, *args):
        return

    def wrap_models(self, *args):
        return

    def unwrap_models(self, *args):
        return

    def reset_action_noise(self, *args, **kwargs):
        return


class DummyAgentOnPolicy(DummyAgentOffPolicy):  # pylint: disable=overwritten-inherited-attribute
    def __init__(self, batch_size, env):
        actor = MagicMock()
        super().__init__(
            batch_size,
            env,
            action_space=Box(0, 1, (1,)),
            learn_step=128,
            actor=actor,
        )
        self.actor.squash_output = False
        self.actor.scale_action = lambda x: x
        self.actor.action_space = self.action_space

        self.registry = MagicMock()
        self.rollout_buffer = MagicMock()
        self.rollout_buffer.reset.side_effect = lambda: None
        self.rollout_buffer.add.side_effect = lambda *args, **kwargs: None
        self.registry.policy.side_effect = lambda: "actor"
        self.num_envs = 2

    def learn(self, *args, **kwargs):
        return random.random()

    def get_action(self, *args, **kwargs):
        return tuple(np.random.randn(self.action_size) for _ in range(4))

    def _get_action_and_values(self, *args, **kwargs):
        return tuple(torch.randn(self.action_size) for _ in range(5))

    def test(self, env, max_steps=None, loop=3, **kwargs):
        return super().test(env, max_steps, loop, **kwargs)

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


class DummyBandit(DummyAgentOffPolicy):
    def __init__(self, batch_size, bandit_env, beta=None):
        super().__init__(batch_size, bandit_env, beta=beta)
        self.regret = [0]

    def get_action(self, *args, **kwargs):
        return np.random.randint(self.action_size)


class ScalarDoneEnv:
    """Minimal env that returns scalar done (bool) instead of array."""

    def __init__(self):
        self.observation_space = Box(low=-1.0, high=1.0, shape=(1,))
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self.state_size = (1,)
        self.action_size = 1

    def reset(self, **kwargs):
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([0.0], dtype=np.float32), 1.0, True, False, {}


class DummyStochastic:
    """Stand-in for StochasticActor with configurable squash_output."""

    def __init__(self, squash_output=False, clip_low=-1.0, clip_high=1.0):
        self.squash_output = squash_output
        self._clip_low = clip_low
        self._clip_high = clip_high

    def scale_action(self, action):
        if self.squash_output:
            return np.clip(action, self._clip_low, self._clip_high)
        return action


class DummyCompiledPolicy:
    """Stand-in for a torch-compiled policy wrapping a DummyStochastic."""

    def __init__(self, orig_mod=None):
        self._orig_mod = orig_mod if orig_mod is not None else DummyStochastic()


class DummyMultiEnv(ParallelEnv):  # pylint: disable=overwritten-inherited-attribute
    """Mimics a vectorized multi-agent parallel environment with num_envs=1."""

    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.state_size = self.state_dims
        self.action_dims = action_dims
        self.action_size = self.action_dims
        self.num_envs = 1
        self.agents = ["agent_0", "other_agent_0"]
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.render_mode = None
        self.metadata = None
        self.info = {
            agent: {
                "env_defined_actions": (
                    None if agent == "other_agent_0" else np.array([0, 1])
                ),
            }
            for agent in self.agents
        }

    def reset(self, seed=None, options=None):
        return {
            agent: np.random.rand(self.num_envs, *self.state_dims)
            for agent in self.agents
        }, self.info

    def step(self, action):
        return (
            {
                agent: np.random.rand(self.num_envs, *self.state_dims)
                for agent in self.agents
            },
            {agent: np.random.rand(self.num_envs) for agent in self.agents},
            {
                agent: np.random.randint(0, 2, size=(self.num_envs,)).astype(bool)
                for agent in self.agents
            },
            {
                agent: np.random.randint(0, 2, size=(self.num_envs,)).astype(bool)
                for agent in self.agents
            },
            self.info,
        )

    def action_space(self, agent):
        return Discrete(5)

    def observation_space(self, agent):
        return Box(0, 255, self.state_dims)


class DummyMultiAgent(DummyAgentOffPolicy):
    def __init__(self, batch_size, env, on_policy, *args):
        possible_action_spaces = Dict(
            {
                "agent_0": Discrete(2),
                "other_agent_0": Box(0, 1, (2,)),
            },
        )
        possible_observation_spaces = Dict(
            {
                "agent_0": Box(0, 1, env.state_dims),
                "other_agent_0": Box(0, 1, env.state_dims),
            },
        )
        super().__init__(
            batch_size, env, *args, action_space=deepcopy(possible_action_spaces)
        )
        self.agent_ids = ["agent_0", "other_agent_0"]
        self.metrics = MultiAgentMetrics(self.agent_ids)
        self.scores = self.metrics.scores
        self.steps = [self.metrics.steps]
        self.shared_agent_ids = ["agent", "other_agent"]
        self.lr_actor = 0.001
        self.lr_critic = 0.01
        self.lr = 0.01
        self.num_envs = 1
        self.on_policy = on_policy
        self.torch_compiler = None
        self.actors = {
            "agent_0": MagicMock(),
            "other_agent_0": MagicMock(),
        }
        self.actors["agent_0"].squash_output = False
        self.actors["agent_0"].scale_action = lambda x: x
        self.actors["other_agent_0"].squash_output = False
        self.actors["other_agent_0"].scale_action = lambda x: x
        self.possible_action_spaces = possible_action_spaces
        self.possible_observation_spaces = possible_observation_spaces
        self.observation_space = deepcopy(possible_observation_spaces)
        self.get_action = (
            self._get_action_on_policy
            if self.on_policy
            else self._get_action_off_policy
        )

        self.registry = MagicMock()
        self.registry.policy.side_effect = lambda: "actors"

    def get_group_id(self, agent_id: str) -> str:
        return agent_id.rsplit("_", 1)[0] if isinstance(agent_id, str) else agent_id

    def has_grouped_agents(self) -> bool:
        return True

    def _get_action_on_policy(self, *args, **kwargs):
        output_dict = {
            agent: np.random.randn(self.num_envs, self.action_size)
            for agent in self.agent_ids
        }
        return output_dict, output_dict, output_dict, output_dict

    def _get_action_off_policy(self, *args, **kwargs):
        output_dict = {
            agent: np.random.randn(self.num_envs, self.action_size)
            for agent in self.agent_ids
        }
        return output_dict, output_dict

    def learn(self, experiences):  # pylint: disable=mixed-tuple-returns
        if self.on_policy:
            return {
                "agent_0": (random.random(),),
                "other_agent_0": (random.random(),),
            }
        return {
            "agent_0": (random.random(), random.random()),
            "other_agent_0": (random.random(), random.random()),
        }

    def test(
        self,
        env,
        max_steps=None,
        loop=3,
        sum_scores=True,
        **kwargs,
    ):
        raw_score = np.random.uniform(0, 400)
        result = (raw_score / 2, raw_score / 2) if not sum_scores else raw_score
        self.fitness.append(result)
        return result

    def get_env_defined_actions(self, info, agents):
        env_defined_actions = {
            agent: info[agent].get("env_defined_action", None) for agent in agents
        }

        if all(eda is None for eda in env_defined_actions.values()):
            return None
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


# Register the dummy multi-agent algorithm with the MultiAgentRLAlgorithm base class.
MultiAgentRLAlgorithm.register(DummyMultiAgent)


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
        super().__init__(max_size=0)
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
                [np.random.randn(*self.state_size) for _ in range(batch_size)],
            )
            actions = np.array(
                [np.random.randn(*self.action_size) for _ in range(batch_size)],
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)],
            )
            next_states = np.array(
                [np.random.randn(*self.next_state_size) for _ in range(batch_size)],
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
        weights = list(range(batch_size))

        sample_transition["weights"] = torch.tensor(weights)
        sample_transition["idxs"] = torch.tensor(idxs)

        return sample_transition

    def update_priorities(self, idxs, priorities):
        return


class DummyNStepMemory(DummyMemory, MultiStepReplayBuffer):  # pylint: disable=overwritten-inherited-attribute
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
        super().__init__(max_size=0)
        self.state_size = None
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
                [np.random.randn(*self.state_size) for _ in range(batch_size)],
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])

        return TensorDict(
            {"obs": states, "reward": rewards},
            batch_size=[batch_size],
        )


class DummyMultiMemory(ReplayBuffer):
    """TensorDict-based multi-agent replay buffer stub.

    Mirrors the API of :class:`MultiAgentReplayBuffer` that the
    multi-agent off-policy training loop expects (``add``, ``sample``,
    ``counter``, ``__len__``).
    """

    def __init__(self):
        super().__init__(max_size=0)
        self.state_size = None
        self.action_size = None
        self.next_state_size = None
        self.agents = ["agent_0", "other_agent_0"]

    def __len__(self):
        return 1000

    def add(self, data: TensorDict) -> None:
        obs_td = data["obs"]
        first_agent = next(iter(obs_td.keys()))
        if self.state_size is None:
            self.state_size = obs_td[first_agent].shape
            self.action_size = data["action"][first_agent].shape
            self.next_state_size = data["next_obs"][first_agent].shape
        self.size += 1
        self.counter += 1

    def sample(self, batch_size, *args):
        return TensorDict(
            {
                "obs": TensorDict(
                    {
                        agent: torch.randn(batch_size, *self.state_size[1:])
                        for agent in self.agents
                    },
                    batch_size=[batch_size],
                ),
                "action": TensorDict(
                    {
                        agent: torch.randn(batch_size, *self.action_size[1:])
                        for agent in self.agents
                    },
                    batch_size=[batch_size],
                ),
                "reward": TensorDict(
                    {agent: torch.randn(batch_size, 1) for agent in self.agents},
                    batch_size=[batch_size],
                ),
                "next_obs": TensorDict(
                    {
                        agent: torch.randn(batch_size, *self.next_state_size[1:])
                        for agent in self.agents
                    },
                    batch_size=[batch_size],
                ),
                "done": TensorDict(
                    {
                        agent: torch.randint(0, 2, (batch_size, 1)).float()
                        for agent in self.agents
                    },
                    batch_size=[batch_size],
                ),
            },
            batch_size=[batch_size],
        )


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


def _make_base_mock_agent(spec_cls, state_size, action_size, *, metrics=None):
    """Wire up the attributes every EvolvableAlgorithm mock needs."""
    mock = MagicMock(spec=spec_cls)
    mock.metrics = metrics or AgentMetrics()
    mock.learn_step = 1
    mock.batch_size = 5
    mock.state_size = state_size
    mock.action_size = action_size
    mock.beta = 0.4
    mock.scores = mock.metrics.scores
    mock.steps = [mock.metrics.steps]
    mock.steps_per_second = 0.0
    mock.fitness = []
    mock.mut = "mutation"
    mock.index = 1
    mock.registry = MagicMock()
    mock.registry.hp_config = None

    def _test_side_effect(*args, **kwargs):
        score = np.random.uniform(0, 400)
        mock.fitness.append(score)
        return score

    mock.test.side_effect = _test_side_effect
    mock.init_evo_step.side_effect = lambda: mock.metrics.init_evo_step()
    mock.add_scores.side_effect = lambda scores: mock.metrics.add_scores(scores)
    mock.finalize_evo_step.side_effect = lambda num_steps: (
        mock.metrics.finalize_evo_step(num_steps)
    )
    mock.learn.side_effect = lambda *args, **kwargs: random.random()
    mock.save_checkpoint.side_effect = lambda *a, **kw: None
    mock.load_checkpoint.side_effect = lambda *a, **kw: None
    mock.wrap_models.side_effect = lambda *a, **kw: None
    mock.unwrap_models.side_effect = lambda *a, **kw: None
    return mock


@pytest.fixture
def mocked_agent_off_policy(env, algo):
    mock_agent = _make_base_mock_agent(algo, env.state_size, 2)
    mock_agent.action_dim = 2

    if algo in [DDPG, TD3]:
        mock_agent.action_low = torch.as_tensor(
            [-1.0] * mock_agent.action_size,
            dtype=torch.float32,
        )
        mock_agent.action_high = torch.as_tensor(
            [1.0] * mock_agent.action_size,
            dtype=torch.float32,
        )
        mock_agent.actor = MagicMock()
        mock_agent.actor.output_activation = "Tanh"
        mock_agent.get_action.side_effect = lambda state, *args, **kwargs: (
            np.random.randn(env.n_envs, mock_agent.action_size).astype(np.float32)
        )
        mock_agent.reset_action_noise.side_effect = lambda *a, **kw: None
    else:
        mock_agent.get_action.side_effect = lambda state, *args, **kwargs: (
            np.random.randint(env.action_size, size=(env.n_envs,))
        )

    if algo == RainbowDQN:
        mock_agent.learn.side_effect = lambda experiences, **kwargs: (
            random.random(),
            random.random(),
            random.random(),
        )
    else:
        mock_agent.learn.side_effect = lambda experiences, **kwargs: random.random()

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
    mock_agent = _make_base_mock_agent(algo, env.state_size, env.action_size)
    mock_agent.action_space = env.action_space
    mock_agent.algo = "PPO"

    mock_agent.get_action.side_effect = lambda state, *args, **kwargs: tuple(
        np.random.randn(env.action_size) for _ in range(4)
    )

    num_envs = env.num_envs if hasattr(env, "num_envs") else 1
    mock_agent.num_envs = num_envs
    mock_agent.rollout_buffer = MagicMock()
    mock_agent.recurrent = False
    mock_agent.preprocess_observation.side_effect = lambda obs: obs
    mock_agent._get_action_and_values.side_effect = lambda *args, **kwargs: (
        torch.zeros(num_envs, env.action_size),
        torch.zeros(num_envs),
        torch.zeros(num_envs),
        torch.zeros(num_envs, 1),
        None,
    )
    mock_agent.registry.policy = lambda: "actor"
    mock_agent.actor = MagicMock()
    mock_agent.actor.squash_output = False
    return mock_agent


@pytest.fixture
def mocked_bandit(bandit_env, algo):
    mock_agent = _make_base_mock_agent(algo, bandit_env.state_size, 2)
    mock_agent.action_dim = 2
    mock_agent.regret = [0]

    mock_agent.get_action.side_effect = lambda state, *args, **kwargs: (
        np.random.randint(bandit_env.action_size)
    )
    mock_agent.learn.side_effect = lambda experiences: random.random()
    return mock_agent


@pytest.fixture
def mocked_multi_agent(multi_env, algo):
    agent_ids = ["agent_0", "other_agent_0"]
    mock_agent = _make_base_mock_agent(
        algo,
        multi_env.state_size,
        multi_env.action_size,
        metrics=MultiAgentMetrics(agent_ids),
    )
    mock_agent.lr = 0.1
    mock_agent.agent_ids = agent_ids
    mock_agent.shared_agent_ids = ["agent", "other_agent"]
    mock_agent.torch_compiler = None
    mock_agent.possible_action_spaces = Dict(
        {aid: multi_env.action_space(aid) for aid in agent_ids},
    )
    mock_agent.possible_observation_spaces = Dict(
        {aid: multi_env.observation_space(aid) for aid in agent_ids},
    )
    mock_agent.action_space = deepcopy(mock_agent.possible_action_spaces)
    mock_agent.observation_space = deepcopy(mock_agent.possible_observation_spaces)

    mock_agent.get_group_id.side_effect = lambda x: (
        x.rsplit("_", 1)[0] if isinstance(x, str) else x
    )
    mock_agent.registry.policy.side_effect = lambda: "actors"
    mock_agent.has_grouped_agents.side_effect = lambda: algo == IPPO
    mock_agent.actors = {aid: MagicMock() for aid in agent_ids}

    def get_action_on_policy(*args, **kwargs):
        out = {a: np.random.randn(1, mock_agent.action_size) for a in agent_ids}
        return out, out

    def get_action_off_policy(*args, **kwargs):
        out = {a: np.random.randn(1, mock_agent.action_size) for a in agent_ids}
        return out, out, out, out

    mock_agent.get_action.side_effect = (
        get_action_off_policy if algo == IPPO else get_action_on_policy
    )
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
    if algo != IPPO:
        mock_agent.reset_action_noise.side_effect = lambda *a, **kw: None
    mock_agent.algo = {MADDPG: "MADDPG", MATD3: "MATD3", IPPO: "IPPO"}[algo]
    return mock_agent


def _make_mock_replay_buffer(
    spec_cls,
    *,
    len_value=10,
    include_weights=True,
    include_sample_from_indices=False,
):
    """Build a MagicMock replay buffer with dynamic shape tracking."""
    mock = MagicMock(spec=spec_cls)
    mock.counter = 0
    mock.size = 0
    mock.state_size = None
    mock.action_size = None
    mock.next_state_size = None
    mock.__len__.return_value = len_value

    def add(data: TensorDict):
        if mock.state_size is None:
            mock.num_envs = data["obs"].shape[0]
            mock.state_size = data["obs"].shape
            mock.action_size = data["action"].shape
            mock.next_state_size = data["next_obs"].shape

        mock.counter += 1
        mock.size += 1

        t = Transition(
            obs=np.random.randn(*mock.state_size),
            action=np.random.randn(*mock.action_size),
            reward=np.random.uniform(0, 400, mock.num_envs),
            done=np.random.choice([True, False], mock.num_envs),
            next_obs=np.random.randn(*mock.next_state_size),
        )
        return t.to_tensordict()

    mock.add.side_effect = add

    def sample(batch_size, beta=None, *args):
        if isinstance(batch_size, (list, torch.Tensor)):
            batch_size = len(batch_size)

        if batch_size == 1:
            states = np.random.randn(*mock.state_size)
            actions = np.random.randn(*mock.action_size)
            rewards = np.random.uniform(0, 400)
            dones = np.random.choice([True, False])
            next_states = np.random.randn(*mock.next_state_size)
        else:
            states = np.array(
                [np.random.randn(*mock.state_size) for _ in range(batch_size)],
            )
            actions = np.array(
                [np.random.randn(*mock.action_size) for _ in range(batch_size)],
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])
            dones = np.array(
                [np.random.choice([True, False]) for _ in range(batch_size)],
            )
            next_states = np.array(
                [np.random.randn(*mock.next_state_size) for _ in range(batch_size)],
            )

        td = Transition(
            obs=states,
            action=actions,
            reward=rewards,
            done=dones,
            next_obs=next_states,
            batch_size=[batch_size],
        ).to_tensordict()

        if beta is not None:
            idxs = [np.random.randn(1) for _ in range(batch_size)]
            td["idxs"] = torch.tensor(idxs)
            if include_weights:
                td["weights"] = torch.tensor(list(range(batch_size)))
        return td

    mock.sample.side_effect = sample
    if include_sample_from_indices:
        mock.sample_from_indices.side_effect = sample
    if spec_cls is PrioritizedReplayBuffer:
        mock.update_priorities.side_effect = lambda idxs, priorities: None
    return mock


@pytest.fixture
def mocked_per_memory():
    return _make_mock_replay_buffer(PrioritizedReplayBuffer, include_weights=True)


@pytest.fixture
def mocked_memory():
    return _make_mock_replay_buffer(ReplayBuffer, include_weights=False)


@pytest.fixture
def mocked_n_step_memory():
    return _make_mock_replay_buffer(
        MultiStepReplayBuffer,
        len_value=10000,
        include_weights=True,
        include_sample_from_indices=True,
    )


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
                [np.random.randn(*mock_memory.state_size) for _ in range(batch_size)],
            )
            rewards = np.array([np.random.uniform(0, 400) for _ in range(batch_size)])

        return TensorDict(
            {"obs": states, "reward": rewards},
            batch_size=[batch_size],
        )

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

    def add(data):
        if mock_memory.state_size is None:
            mock_memory.state_size = data["obs", mock_memory.agents[0]].shape[1:]
        if mock_memory.action_size is None:
            mock_memory.action_size = data["action", mock_memory.agents[0]].shape[1:]
        if mock_memory.next_state_size is None:
            mock_memory.next_state_size = data["next_obs", mock_memory.agents[0]].shape[
                1:
            ]
        mock_memory.counter += data.shape[0]

    mock_memory.add.side_effect = add

    def sample(batch_size, *args):
        obs = TensorDict(
            {
                a: torch.randn(batch_size, *mock_memory.state_size)
                for a in mock_memory.agents
            },
            batch_size=[batch_size],
        )
        actions = TensorDict(
            {
                a: torch.randn(batch_size, *mock_memory.action_size)
                for a in mock_memory.agents
            },
            batch_size=[batch_size],
        )
        rewards = TensorDict(
            {a: torch.rand(batch_size, 1) for a in mock_memory.agents},
            batch_size=[batch_size],
        )
        dones = TensorDict(
            {a: torch.zeros(batch_size, 1) for a in mock_memory.agents},
            batch_size=[batch_size],
        )
        next_obs = TensorDict(
            {
                a: torch.randn(batch_size, *mock_memory.next_state_size)
                for a in mock_memory.agents
            },
            batch_size=[batch_size],
        )
        return TensorDict(
            {
                "obs": obs,
                "action": actions,
                "reward": rewards,
                "done": dones,
                "next_obs": next_obs,
            },
            batch_size=[batch_size],
        )

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
    mock_env.num_envs = 1
    mock_env.agents = ["agent_0", "other_agent_0"]
    mock_env.possible_agents = ["agent_0", "other_agent_0"]
    mock_env.reset.side_effect = lambda *args, **kwargs: (
        {
            agent: np.expand_dims(np.random.rand(*mock_env.state_size), 0)
            for agent in mock_env.agents
        },
        {
            agent: {
                "env_defined_actions": (
                    None if agent == "other_agent_0" else np.array([0, 1])
                ),
            }
            for agent in mock_env.agents
        },
    )
    mock_env.step.side_effect = lambda *args: (
        {
            agent: np.expand_dims(np.random.rand(*mock_env.state_size), 0)
            for agent in mock_env.agents
        },
        {
            agent: np.array([np.random.randint(0, 5)], dtype=np.float64)
            for agent in mock_env.agents
        },
        {
            agent: np.array([np.random.randint(0, 2)], dtype=bool)
            for agent in mock_env.agents
        },
        {
            agent: np.array([np.random.randint(0, 2)], dtype=bool)
            for agent in mock_env.agents
        },
        {
            agent: {
                "env_defined_actions": (
                    None if agent == "other_agent_0" else np.array([0, 1])
                ),
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
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
        "DATASET": "../data/cartpole/cartpole_v1.1.0.h5",
    }


@pytest.fixture
def dummy_h5py_data(action_size, state_size):
    # Create a dummy h5py dataset
    dataset = dict.fromkeys(["actions", "observations", "rewards"])
    dataset["actions"] = np.array([np.random.randn(action_size) for _ in range(10)])
    dataset["observations"] = np.array(
        [np.random.randn(*state_size) for _ in range(10)],
    )
    dataset["rewards"] = np.array([np.random.randint(0, 5) for _ in range(10)])
    dataset["terminals"] = np.array(
        [np.random.choice([True, False]) for _ in range(10)],
    )

    return dataset


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_BOTH)
def test_train_off_policy(env, population_off_policy, tournament, mutations, memory):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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
    "state_size, action_size, vect, algo, num_envs, learn_step",
    [
        ((6,), 2, False, DQN, 1, 2),
        ((6,), 2, False, DDPG, 1, 2),
        ((6,), 2, False, TD3, 1, 2),
        ((6,), 2, True, DQN, 2, 1),
        ((6,), 2, True, DDPG, 2, 1),
        ((6,), 2, True, TD3, 2, 1),
    ],
)
def test_train_off_policy_agent_calls_made(
    env,
    algo,
    mocked_agent_off_policy,
    tournament,
    mutations,
    memory,
    num_envs,
    learn_step,
):
    for accelerator_flag in [True, False]:
        accelerator = Accelerator() if accelerator_flag else None
        mock_population = [mocked_agent_off_policy for _ in range(6)]
        for agent in mock_population:
            agent.learn_step = learn_step

        if env.vect:
            env.num_envs = num_envs

        pop, _ = train_off_policy(
            env,
            "env_name",
            "algo",
            mock_population,
            memory,
            init_hp=None,
            mut_p=None,
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

    pop, _ = train_off_policy(
        env,
        "env_name",
        "Rainbow DQN",
        mock_population,
        memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_NOVECT)
def test_train_off_policy_save_elite_warning(
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
):
    warning_string = (
        "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            init_hp=None,
            mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_NOVECT)
def test_train_off_policy_checkpoint_warning(
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
):
    warning_string = (
        "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            init_hp=None,
            mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_NOVECT)
def test_actions_histogram(env, population_off_policy, tournament, mutations, memory):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "DQN",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_off_policy_replay_buffer_calls(
    mocked_memory,
    env,
    population_off_policy,
    tournament,
    mutations,
):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        mocked_memory,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=mocked_memory,
        init_hp=None,
        mut_p=None,
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
    mocked_env,
    memory,
    population_off_policy,
    tournament,
    mutations,
):
    pop, _ = train_off_policy(
        mocked_env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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
    env,
    memory,
    population_off_policy,
    mocked_tournament,
    mocked_mutations,
):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _IMG_NOVECT)
def test_train_off_policy_rgb_input(
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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
    env,
    memory,
    population_off_policy,
    tournament,
    mutations,
    n_step_memory,
    per,
):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _IMG_VECT)
def test_train_off_policy_using_alternate_buffers_rgb(
    env,
    memory,
    population_off_policy,
    tournament,
    mutations,
    n_step_memory,
):
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory=memory,
        init_hp=None,
        mut_p=None,
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
    "state_size, action_size, vect",
    [((6,), 2, True), ((6,), 2, False)],
)
def test_train_off_policy_distributed(
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
):
    accelerator = Accelerator()
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_wandb_init_log(env, population_off_policy, tournament, mutations, memory):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_off_policy.train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            init_hp=init_hp,
            mut_p=mut_p,
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
                "train/global_step": ANY,
                "train/steps_per_second": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            },
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
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
    accelerator,
):
    accelerator = Accelerator() if accelerator else None
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_off_policy.train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            init_hp=init_hp,
            mut_p=mut_p,
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
                "train/global_step": ANY,
                "train/steps_per_second": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            },
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_early_stop_wandb(env, population_off_policy, tournament, mutations, memory):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as _,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as _,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_off_policy.train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            init_hp=init_hp,
            mut_p=mut_p,
            target=-10000,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_off_policy_save_elite(
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
    tmp_path,
):
    elite_path = str(tmp_path / "checkpoint.pt")
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_save_checkpoint(
    env,
    population_off_policy,
    tournament,
    mutations,
    memory,
    accelerator_flag,
    tmpdir,
):
    accelerator = Accelerator() if accelerator_flag else None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, _ = train_off_policy(
        env,
        "env_name",
        "algo",
        population_off_policy,
        memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect, algo", [((6,), 2, True, PPO)])
def test_train_on_policy_agent_calls_made(
    env,
    algo,
    mocked_agent_on_policy,
    tournament,
    mutations,
):
    for accelerator_flag in [True, False]:
        accelerator = Accelerator() if accelerator_flag else None
        mock_population = [mocked_agent_on_policy for _ in range(6)]
        pop, _ = train_on_policy(
            env,
            "env_name",
            "algo",
            mock_population,
            init_hp=None,
            mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_NOVECT)
def test_train_on_policy_save_elite_warning(
    env,
    population_on_policy,
    tournament,
    mutations,
):
    warning_string = (
        "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            wb=False,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_NOVECT)
def test_train_on_policy_checkpoint_warning(
    env,
    population_on_policy,
    tournament,
    mutations,
):
    warning_string = (
        "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=None,
            mut_p=None,
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
    mocked_env,
    population_on_policy,
    tournament,
    mutations,
):
    pop, _ = train_on_policy(
        mocked_env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
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
    env,
    population_on_policy,
    mocked_tournament,
    mocked_mutations,
):
    pop, _ = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
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
    "state_size, action_size, vect", [((6,), 2, True), ((6,), 2, False)]
)
def test_train_on_policy(
    env,
    population_on_policy,
    tournament,
    mutations,
):
    pop, _ = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
        max_steps=256,
        evo_steps=256,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize("state_size, action_size, vect", _IMG_NOVECT)
def test_train_on_policy_rgb_input(env, population_on_policy, tournament, mutations):
    pop, _ = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )

    assert len(pop) == len(population_on_policy)


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [((6,), 2, True), ((6,), 2, False)],
)
def test_train_on_policy_distributed(env, population_on_policy, tournament, mutations):
    accelerator = Accelerator()
    pop, _ = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
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
    env,
    population_on_policy,
    tournament,
    mutations,
    accelerator,
):
    accelerator = Accelerator() if accelerator else None
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_on_policy.train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=init_hp,
            mut_p=mut_p,
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


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_early_stop_wandb_on_policy(env, population_on_policy, tournament, mutations):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as _,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as _,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_on_policy.train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=init_hp,
            mut_p=mut_p,
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


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_on_policy_save_elite(
    env,
    population_on_policy,
    tournament,
    mutations,
    accelerator_flag,
    tmp_path,
):
    accelerator = Accelerator() if accelerator_flag else None
    elite_path = str(tmp_path / "elite")
    pop, _ = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize(
    "state_size, action_size, vect, accelerator_flag",
    [((6,), 2, True, True), ((6,), 2, True, False)],
)
def test_train_on_policy_save_checkpoint(
    env,
    population_on_policy,
    tournament,
    mutations,
    accelerator_flag,
    tmpdir,
):
    accelerator = Accelerator() if accelerator_flag else None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, _ = train_on_policy(
        env,
        "env_name",
        "algo",
        population_on_policy,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, sum_scores",
    [((6,), 2, True), ((6,), 2, False)],
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
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        init_hp=None,
        mut_p=None,
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
    "state_size, action_size, sum_scores",
    [((6,), 2, True), ((6,), 2, False)],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
def test_train_multi_agent_on_policy(
    multi_env,
    population_multi_agent,
    on_policy,
    tournament,
    mutations,
    sum_scores,
    accelerator_flag,
):
    accelerator = Accelerator() if accelerator_flag else None
    pop, _ = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        init_hp=None,
        mut_p=None,
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
@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_agent_off_policy_distributed(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    accelerator = Accelerator()
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        init_hp=None,
        mut_p=None,
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
@pytest.mark.parametrize("state_size, action_size", _IMG)
def test_train_multi_agent_off_policy_rgb(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
    )

    assert len(pop) == len(population_multi_agent)


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", _IMG)
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
        DummyMultiEnv,
        num_envs=4,
        state_dims=state_size,
        action_dims=action_size,
    )
    for agent in population_multi_agent:
        agent.num_envs = 4
        agent.scores = [1]
    env.reset()
    pop, _ = train_multi_agent_off_policy(
        env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        memory=multi_memory,
        init_hp=None,
        mut_p=None,
        max_steps=10,
        evo_steps=5,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
    )
    assert len(pop) == len(population_multi_agent)
    env.close()


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize("state_size, action_size", _IMG)
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
        DummyMultiEnv,
        num_envs=4,
        state_dims=state_size,
        action_dims=action_size,
    )
    for agent in population_multi_agent:
        agent.num_envs = 4
        agent.scores = [1]
    env.reset()
    pop, _ = train_multi_agent_on_policy(
        env,
        "env_name",
        "algo",
        pop=population_multi_agent,
        init_hp=None,
        mut_p=None,
        max_steps=10,
        evo_steps=5,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
    )
    assert len(pop) == len(population_multi_agent)
    env.close()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_save_elite_warning(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    warning_string = (
        "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_save_elite_warning_on_policy(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    warning_string = (
        "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            save_elite=False,
            elite_path="path",
        )


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_checkpoint_warning(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    warning_string = (
        "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            tournament=tournament,
            mutation=mutations,
            checkpoint=None,
            checkpoint_path="path",
        )


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_checkpoint_warning_on_policy(
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    warning_string = (
        "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            init_hp=None,
            mut_p=None,
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
    "state_size, action_size, accelerator_flag",
    [((6,), 2, False), ((6,), 2, True)],
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
    init_hp = {
        "BATCH_SIZE": 128,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch(
            "agilerl.utils.utils.wandb.init",
        ) as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch(
            "agilerl.logger.wandb.log",
        ) as mock_wandb_log,
        patch(
            "agilerl.logger.wandb.finish",
        ) as mock_wandb_finish,
    ):
        accelerator = Accelerator() if accelerator_flag else None
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            multi_memory,
            init_hp=init_hp,
            mut_p=mut_p,
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
    "state_size, action_size, accelerator_flag",
    [((6,), 2, False), ((6,), 2, True)],
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
    init_hp = {
        "BATCH_SIZE": 128,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch(
            "agilerl.utils.utils.wandb.init",
        ) as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch(
            "agilerl.logger.wandb.log",
        ) as mock_wandb_log,
        patch(
            "agilerl.logger.wandb.finish",
        ) as mock_wandb_finish,
    ):
        accelerator = Accelerator() if accelerator_flag else None
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_on_policy.train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            init_hp=init_hp,
            mut_p=mut_p,
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
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as _,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as _,
        patch(
            "agilerl.logger.wandb.finish",
        ) as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            multi_memory,
            init_hp=init_hp,
            mut_p=mut_p,
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
    multi_env,
    population_multi_agent,
    on_policy,
    multi_memory,
    tournament,
    mutations,
):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-3,
        "GAMMA": 0.99,
        "LEARN_STEP": 1,
        "TAU": 1e-3,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as _,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as _,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_multi_agent_on_policy.train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            init_hp=init_hp,
            mut_p=mut_p,
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
    accelerator = Accelerator() if accelerator_flag else None

    mock_population = [mocked_multi_agent for _ in range(6)]

    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        mock_population,
        multi_memory,
        init_hp=None,
        mut_p=None,
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
    accelerator = Accelerator() if accelerator_flag else None

    mock_population = [mocked_multi_agent for _ in range(6)]

    pop, _ = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        mock_population,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_multi_agent_off_policy(
        mocked_multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_multi_agent_on_policy(
        mocked_multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        mocked_multi_memory,
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        tournament=tournament,
        mutation=mutations,
        wb=False,
    )
    mocked_multi_memory.sample.assert_called()
    mocked_multi_memory.add.assert_called()


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag",
    [((6,), 2, True), ((6,), 2, False)],
)
def test_train_multi_save_elite(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    on_policy,
    accelerator_flag,
    tmp_path,
):
    accelerator = Accelerator() if accelerator_flag else None
    elite_path = str(tmp_path / "elite")
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag",
    [((6,), 2, True), ((6,), 2, False)],
)
def test_train_multi_save_elite_on_policy(
    multi_env,
    population_multi_agent,
    tournament,
    mutations,
    multi_memory,
    on_policy,
    accelerator_flag,
    tmp_path,
):
    accelerator = Accelerator() if accelerator_flag else None
    elite_path = str(tmp_path / "elite")
    pop, _ = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("on_policy", [False])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag",
    [((6,), 2, True), ((6,), 2, False)],
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
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        multi_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("on_policy", [True])
@pytest.mark.parametrize(
    "state_size, action_size, accelerator_flag",
    [((6,), 2, True), ((6,), 2, False)],
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
    pop, _ = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        population_multi_agent,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize(
    "state_size, action_size, vect",
    [
        ((6,), 2, True),
    ],
)
def test_train_offline(
    env,
    population_off_policy,
    memory,
    tournament,
    mutations,
    offline_init_hp,
    dummy_h5py_data,
):
    for accelerator_flag in [True, False]:
        accelerator = Accelerator() if accelerator_flag else None

        pop, _ = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            init_hp=offline_init_hp,
            mut_p=None,
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
    warning_string = (
        "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            init_hp=offline_init_hp,
            mut_p=None,
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
    warning_string = (
        "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            init_hp=offline_init_hp,
            mut_p=None,
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
    accelerator = Accelerator() if accelerator_flag else None
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_offline.train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            init_hp=offline_init_hp,
            mut_p=mut_p,
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
        accelerator = Accelerator() if accelerator_flag else None
        mut_p = {
            "NO_MUT": 0.4,
            "ARCH_MUT": 0.2,
            "PARAMS_MUT": 0.2,
            "ACT_MUT": 0.2,
            "RL_HP_MUT": 0.2,
        }
        with (
            patch("agilerl.utils.utils.wandb.login") as _,
            patch("agilerl.utils.utils.wandb.init") as _,
            patch("agilerl.logger.wandb.run", new=MagicMock()),
            patch("agilerl.logger.wandb.log") as _,
            patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
        ):
            # Call the function that should trigger wandb.init
            agilerl.training.train_offline.train_offline(
                env,
                "env_name",
                dummy_h5py_data,
                "algo",
                population_off_policy,
                memory,
                init_hp=offline_init_hp,
                mut_p=mut_p,
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
        accelerator = Accelerator() if accelerator_flag else None
        mock_population = [mocked_agent_off_policy for _ in range(6)]

        pop, _ = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            mock_population,
            memory,
            init_hp=offline_init_hp,
            mut_p=None,
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
        pop, _ = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            mocked_memory,
            init_hp=offline_init_hp,
            mut_p=None,
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
        accelerator = Accelerator() if accelerator_flag else None

        pop, _ = train_offline(
            env,
            "env_name",
            dummy_h5py_data,
            "algo",
            population_off_policy,
            memory,
            init_hp=offline_init_hp,
            mut_p=None,
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
    tmp_path,
):
    accelerator = Accelerator() if accelerator_flag else None
    elite_path = str(tmp_path / "elite")
    pop, _ = train_offline(
        env,
        "env_name",
        dummy_h5py_data,
        "algo",
        population_off_policy,
        memory,
        init_hp=offline_init_hp,
        mut_p=None,
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
    accelerator = Accelerator() if accelerator_flag else None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, _ = train_offline(
        env,
        "env_name",
        dummy_h5py_data,
        "algo",
        population_off_policy,
        memory,
        init_hp=offline_init_hp,
        mut_p=None,
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


@pytest.mark.parametrize(
    "state_size, action_size",
    [
        ((6,), 2),
    ],
)
def test_train_bandit(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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
        accelerator = Accelerator() if accelerator_flag else None
        mock_population = [mocked_bandit for _ in range(6)]

        pop, _ = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            mock_population,
            bandit_memory,
            init_hp=None,
            mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_bandit_save_elite_warning(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    warning_string = (
        "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            init_hp=None,
            mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_bandit_checkpoint_warning(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    warning_string = (
        "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
    )
    with pytest.warns(match=warning_string):
        pop, _ = train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            init_hp=None,
            mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_bandit_actions_histogram(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "DQN",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_bandit_replay_buffer_calls(
    mocked_bandit_memory,
    bandit_env,
    population_bandit,
    tournament,
    mutations,
):
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        mocked_bandit_memory,
        init_hp=None,
        mut_p=None,
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
    mocked_bandit_env,
    bandit_memory,
    population_bandit,
    tournament,
    mutations,
):
    pop, _ = train_bandits(
        mocked_bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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
    bandit_env,
    bandit_memory,
    population_bandit,
    mocked_tournament,
    mocked_mutations,
):
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _IMG)
def test_train_bandit_rgb_input(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        memory=bandit_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _IMG_SQUARE)
def test_train_bandit_using_alternate_buffers_rgb(
    bandit_env,
    bandit_memory,
    population_bandit,
    tournament,
    mutations,
):
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        memory=bandit_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_bandit_distributed(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    accelerator = Accelerator()
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_bandit_wandb_init_log(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 1,
        "LAMBDA": 1,
        "REG": 0.000625,
        "LEARN_STEP": 1,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_bandits.train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            init_hp=init_hp,
            mut_p=mut_p,
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
                "train/global_step": ANY,
                "train/steps_per_second": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            },
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
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
    accelerator,
):
    accelerator = Accelerator() if accelerator else None
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 1,
        "LAMBDA": 1,
        "REG": 0.000625,
        "LEARN_STEP": 1,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as mock_wandb_init,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_bandits.train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            init_hp=init_hp,
            mut_p=mut_p,
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
                "train/global_step": ANY,
                "train/steps_per_second": ANY,
                "train/mean_score": ANY,
                "eval/mean_fitness": ANY,
                "eval/best_fitness": ANY,
            },
        )
        # Assert that wandb.finish was called
        mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_bandit_early_stop_wandb(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
):
    init_hp = {
        "BATCH_SIZE": 128,
        "LR": 1e-3,
        "GAMMA": 1,
        "LAMBDA": 1,
        "REG": 0.000625,
        "LEARN_STEP": 1,
        "POP_SIZE": 6,
        "MEMORY_SIZE": 20000,
    }
    mut_p = {
        "NO_MUT": 0.4,
        "ARCH_MUT": 0.2,
        "PARAMS_MUT": 0.2,
        "ACT_MUT": 0.2,
        "RL_HP_MUT": 0.2,
    }
    with (
        patch("agilerl.utils.utils.wandb.login") as _,
        patch("agilerl.utils.utils.wandb.init") as _,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as _,
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        # Call the function that should trigger wandb.init
        agilerl.training.train_bandits.train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            init_hp=init_hp,
            mut_p=mut_p,
            target=-10000,
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


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_bandit_save_elite(
    bandit_env,
    population_bandit,
    tournament,
    mutations,
    bandit_memory,
    tmp_path,
):
    elite_path = str(tmp_path / "checkpoint.pt")
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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
    accelerator = Accelerator() if accelerator_flag else None
    checkpoint_path = str(Path(tmpdir) / "checkpoint")
    pop, _ = train_bandits(
        bandit_env,
        "bandit_env_name",
        "algo",
        population_bandit,
        bandit_memory,
        init_hp=None,
        mut_p=None,
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
            assert os.path.isfile(f"{checkpoint_path}_{i}_{10 * (s + 1)}.pt")


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_off_policy_wandb_kwargs_update(env, memory):
    agent = DummyAgentOffPolicy(5, env, 0.4)

    with (
        patch("agilerl.utils.utils.init_wandb") as mock_init_wandb,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish"),
    ):
        train_off_policy(
            env,
            "env_name",
            "algo",
            [agent],
            memory,
            max_steps=2,
            evo_steps=2,
            wb=True,
            wandb_kwargs={"project": "custom_project", "name": "custom_run"},
            verbose=False,
        )

    kwargs = mock_init_wandb.call_args.kwargs
    assert kwargs["project"] == "custom_project"
    assert kwargs["name"] == "custom_run"


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_off_policy_per_nstep_none_branches(env):
    class CapturingPerAgent(DummyAgentOffPolicy):
        def __init__(self, batch_size, env):
            super().__init__(batch_size, env, 0.4)
            self.captured = []

        def learn(self, experiences, n_experiences=None, per=False):
            self.captured.append(n_experiences)
            return 0.1, torch.tensor([0]), torch.tensor([1.0])

    agent_gt = CapturingPerAgent(5, env)
    agent_gt.learn_step = 4
    train_off_policy(
        env,
        "env_name",
        "algo",
        [agent_gt],
        DummyMemory(),
        max_steps=4,
        evo_steps=4,
        per=True,
        n_step_memory=None,
        verbose=False,
    )
    assert any(item is None for item in agent_gt.captured)

    agent_le = CapturingPerAgent(5, env)
    agent_le.learn_step = 1
    train_off_policy(
        env,
        "env_name",
        "algo",
        [agent_le],
        DummyMemory(),
        max_steps=4,
        evo_steps=4,
        per=True,
        n_step_memory=None,
        verbose=False,
    )
    assert any(item is None for item in agent_le.captured)


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_off_policy_wandb_dqn_and_ddpg_loss_branches(env, monkeypatch):
    class DQNLossAgent(DummyAgentOffPolicy):
        def get_action(self, *args, **kwargs):
            return np.array([0, 1], dtype=int)

        def learn(self, experiences, n_experiences=None, per=False):
            return 0.25

    class DDPGLossAgent(DummyAgentOffPolicy):
        def learn(self, experiences, n_experiences=None, per=False):
            return (0.1, 0.2)

    dqn_agent = DQNLossAgent(5, env, 0.4)
    dqn_agent.steps = [0] * 100
    ddpg_agent = DDPGLossAgent(5, env, 0.4)
    ddpg_agent.steps = [0] * 100

    monkeypatch.setattr(agilerl.training.train_off_policy, "DQN", DQNLossAgent)
    monkeypatch.setattr(agilerl.training.train_off_policy, "DDPG", DDPGLossAgent)
    monkeypatch.setattr(agilerl.training.train_off_policy, "TD3", DDPGLossAgent)

    with (
        patch("agilerl.utils.utils.init_wandb"),
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log") as mock_wandb_log,
        patch("agilerl.logger.wandb.finish"),
    ):
        train_off_policy(
            env,
            "env_name",
            "algo",
            [dqn_agent],
            DummyMemory(),
            max_steps=4,
            evo_steps=4,
            wb=True,
            verbose=False,
        )
        dqn_log = mock_wandb_log.call_args[0][0]
        assert "train/global_step" in dqn_log
        assert "eval/mean_fitness" in dqn_log
        assert "train/mean_score" in dqn_log

        train_off_policy(
            env,
            "env_name",
            "algo",
            [ddpg_agent],
            DummyMemory(),
            max_steps=4,
            evo_steps=4,
            wb=True,
            verbose=False,
        )
        ddpg_log = mock_wandb_log.call_args[0][0]
        assert "train/global_step" in ddpg_log
        assert "eval/mean_fitness" in ddpg_log
        assert "train/mean_score" in ddpg_log


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_off_policy_early_stop_wb_branch(env):
    agent = DummyAgentOffPolicy(5, env, 0.4)
    agent.steps = [0] * 100

    with (
        patch("agilerl.utils.utils.init_wandb"),
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        train_off_policy(
            env,
            "env_name",
            "algo",
            [agent],
            DummyMemory(),
            max_steps=2,
            evo_steps=2,
            target=-1.0,
            wb=True,
            verbose=False,
        )
    mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_on_policy_wandb_kwargs_update(env):
    agent = DummyAgentOnPolicy(5, env)
    with (
        patch("agilerl.utils.utils.init_wandb") as mock_init_wandb,
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish"),
    ):
        train_on_policy(
            env,
            "env_name",
            "algo",
            [agent],
            max_steps=2,
            evo_steps=2,
            wb=True,
            wandb_kwargs={"project": "custom_project", "name": "custom_run"},
            verbose=False,
        )
    kwargs = mock_init_wandb.call_args.kwargs
    assert kwargs["project"] == "custom_project"
    assert kwargs["name"] == "custom_run"


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_on_policy_recurrent_collect_rollouts_import_branch(env, monkeypatch):
    agent = DummyAgentOnPolicy(5, env)
    agent.recurrent = True
    agent.learn_step = 1

    def fake_collect(*args, **kwargs):
        return [], None, None, None, None

    monkeypatch.setattr("agilerl.rollouts.collect_rollouts_recurrent", fake_collect)

    train_on_policy(
        env,
        "env_name",
        "algo",
        [agent],
        max_steps=1,
        evo_steps=1,
        wb=False,
        verbose=False,
    )


def test_train_on_policy_clip_box_without_squash_and_scalar_done(monkeypatch):
    monkeypatch.setattr(agilerl.rollouts.on_policy, "StochasticActor", DummyStochastic)

    env = ScalarDoneEnv()
    agent = DummyAgentOnPolicy(1, env)
    agent.action_space = Box(low=-1.0, high=1.0, shape=(1,))
    agent.actor = DummyStochastic(squash_output=False)
    agent.registry.policy.side_effect = lambda: "actor"
    agent.get_action = lambda *args, **kwargs: (
        np.array([2.5], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
        np.array([0.3], dtype=np.float32),
    )

    train_on_policy(
        env,
        "env_name",
        "algo",
        [agent],
        max_steps=1,
        evo_steps=1,
        wb=False,
        verbose=False,
    )


def test_train_on_policy_clip_box_with_squash(monkeypatch):
    monkeypatch.setattr(agilerl.rollouts.on_policy, "StochasticActor", DummyStochastic)

    env = ScalarDoneEnv()
    agent = DummyAgentOnPolicy(1, env)
    agent.action_space = Box(low=-1.0, high=1.0, shape=(1,))
    agent.actor = DummyStochastic(squash_output=True)
    agent.registry.policy.side_effect = lambda: "actor"
    agent.get_action = lambda *args, **kwargs: (
        np.array([2.5], dtype=np.float32),
        np.array([0.1], dtype=np.float32),
        np.array([0.2], dtype=np.float32),
        np.array([0.3], dtype=np.float32),
    )

    train_on_policy(
        env,
        "env_name",
        "algo",
        [agent],
        max_steps=1,
        evo_steps=1,
        wb=False,
        verbose=False,
    )


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_on_policy_early_stop_wb_branch(env):
    agent = DummyAgentOnPolicy(5, env)
    agent.steps = [0] * 100
    with (
        patch("agilerl.utils.utils.init_wandb"),
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        train_on_policy(
            env,
            "env_name",
            "algo",
            [agent],
            max_steps=2,
            evo_steps=2,
            target=-1.0,
            wb=True,
            verbose=False,
        )
    mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_agent_off_policy_learn_step_branch_and_early_stop(
    multi_env,
    multi_memory,
):
    agent = DummyMultiAgent(1, multi_env, on_policy=False)
    agent.learn_step = 2
    agent.steps = [0] * 100

    with (
        patch("agilerl.utils.utils.init_wandb"),
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            [agent],
            multi_memory,
            max_steps=2,
            evo_steps=2,
            target=-1.0,
            wb=True,
            verbose=False,
        )
    mock_wandb_finish.assert_called()


def test_train_multi_agent_off_policy_empty_population_rejected(multi_memory):
    class EmptyAgentEnv:
        agents = []
        possible_agents = []

    with pytest.raises(ValueError, match="at least one agent"):
        train_multi_agent_off_policy(
            EmptyAgentEnv(),
            "env_name",
            "algo",
            [],
            multi_memory,
            sum_scores=False,
            max_steps=1,
            evo_steps=1,
            wb=False,
            verbose=False,
        )


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_agent_on_policy_compiled_clip_and_early_stop(
    multi_env,
    monkeypatch,
):
    monkeypatch.setattr(
        agilerl.training.train_multi_agent_on_policy, "StochasticActor", DummyStochastic
    )

    agent = DummyMultiAgent(1, multi_env, on_policy=True)
    agent.torch_compiler = "compiled"
    agent.steps = [0] * 100
    agent.possible_action_spaces = Dict(
        {"agent_0": Box(0, 1, (2,)), "other_agent_0": Box(0, 1, (2,))}
    )
    agent.actors = {
        "agent_0": DummyCompiledPolicy(),
        "other_agent_0": DummyCompiledPolicy(),
    }

    with (
        patch("agilerl.utils.utils.init_wandb"),
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            [agent],
            sum_scores=True,
            max_steps=2,
            evo_steps=2,
            target=-1.0,
            wb=True,
            verbose=False,
        )
    mock_wandb_finish.assert_called()


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_agent_on_policy_compiled_clip_with_squash(
    multi_env,
    monkeypatch,
):
    monkeypatch.setattr(
        agilerl.training.train_multi_agent_on_policy,
        "StochasticActor",
        DummyStochastic,
    )

    squashed = DummyStochastic(squash_output=True, clip_low=0.0, clip_high=1.0)
    agent = DummyMultiAgent(1, multi_env, on_policy=True)
    agent.torch_compiler = "compiled"
    agent.possible_action_spaces = Dict(
        {"agent_0": Box(0, 1, (2,)), "other_agent_0": Box(0, 1, (2,))}
    )
    agent.actors = {
        "agent_0": DummyCompiledPolicy(squashed),
        "other_agent_0": DummyCompiledPolicy(squashed),
    }

    train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        [agent],
        sum_scores=True,
        max_steps=2,
        evo_steps=2,
        wb=False,
        verbose=False,
    )


@pytest.mark.parametrize("state_size, action_size", _FLAT)
def test_train_multi_agent_on_policy_nan_mean_score_branch(multi_env, monkeypatch):
    class OddIterPop(list):
        def __init__(self, *args):
            super().__init__(*args)
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            # Iteration call order inside train_multi_agent_on_policy is:
            # 1-3: setup list comprehensions, 4: while condition, 5: training loop
            # We make the training loop empty to keep pop_episode_scores == [].
            if self.iter_calls == 5:
                return iter([])
            return super().__iter__()

    class DummyPbar:
        def update(self, *args, **kwargs):
            return None

        def write(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class ToggleSum:
        def __init__(self):
            self.calls = 0

        def __call__(self, _):
            self.calls += 1
            return 0 if self.calls == 1 else 2

    monkeypatch.setattr(
        agilerl.training.train_multi_agent_on_policy,
        "default_progress_bar",
        lambda *args, **kwargs: DummyPbar(),
    )
    monkeypatch.setattr(
        agilerl.training.train_multi_agent_on_policy.np, "sum", ToggleSum()
    )

    pop = OddIterPop([DummyMultiAgent(1, multi_env, on_policy=True)])
    train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        pop,
        sum_scores=False,
        max_steps=1,
        evo_steps=1,
        wb=False,
        verbose=False,
    )


@pytest.mark.parametrize("state_size, action_size, vect", _FLAT_VECT)
def test_train_offline_minari_branch_and_early_stop(env, memory):
    agent = DummyAgentOffPolicy(5, env, 0.4)
    agent.steps = [0] * 100
    seed_transition = Transition(
        obs=np.random.randn(2, *env.state_size[1:]),
        action=np.random.randn(2, env.action_size),
        reward=np.random.uniform(0, 1, 2),
        done=np.random.choice([True, False], 2),
        next_obs=np.random.randn(2, *env.state_size[1:]),
    ).to_tensordict()
    seed_transition.batch_size = [2]
    memory.add(seed_transition)
    with (
        patch(
            "agilerl.training.train_offline.minari_to_agile_buffer",
            side_effect=lambda *_args, **_kwargs: memory,
        ) as mock_minari,
        patch("agilerl.utils.utils.init_wandb"),
        patch("agilerl.logger.wandb.run", new=MagicMock()),
        patch("agilerl.logger.wandb.log"),
        patch("agilerl.logger.wandb.finish") as mock_wandb_finish,
    ):
        train_offline(
            env,
            "env_name",
            {},
            "algo",
            [agent],
            memory,
            max_steps=2,
            evo_steps=2,
            minari_dataset_id="dummy_minari_id",
            wb=True,
            target=-1.0,
            verbose=False,
        )
    mock_minari.assert_called_once()
    mock_wandb_finish.assert_called()


# LEAVE LAST, TEMPORARY TO DELETE SAVED MODELS
# TODO: Properly handle saving/deletion in tests
def test_remove_saved_models():
    if os.path.exists("models"):
        shutil.rmtree("models")
