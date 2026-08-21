# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import os
import random
import shutil
import warnings
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import ClassVar
from unittest.mock import ANY, MagicMock, patch

import dill
import gymnasium as gym
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from pettingzoo import ParallelEnv
from tensordict import TensorDict

import agilerl
import agilerl.rollouts.on_policy
from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core.base import EvolvableAlgorithm, MultiAgentRLAlgorithm
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.data import Transition
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.hpo import function_preserving, regrama
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.metrics import AgentMetrics, MultiAgentMetrics
from agilerl.population import Population
from agilerl.training.train_bandits import train_bandits
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import make_multi_agent_vect_envs, run_selection_and_mutation
from agilerl.vector.pz_vec_env import PettingZooVecEnv
from tests.helper_functions import (
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
    grama_scores_for,
    rank_population_by_subpopulation,
    weakest_agent_index,
)

# Common parametrize constants
_FLAT_VECT = [((6,), 2, True)]
_FLAT_NOVECT = [((6,), 2, False)]
_FLAT_BOTH = [((6,), 2, True), ((6,), 2, False)]
_IMG_NOVECT = [((250, 160, 3), 2, False)]
_IMG_VECT = [((3, 64, 64), 2, True)]
_FLAT = [((6,), 2)]
_IMG = [((250, 160, 3), 2)]
_IMG_SQUARE = [((3, 64, 64), 2)]

_WANDB_SUMMARY_KEYS = (
    "train/global_step",
    "train/steps_per_second",
    "train/mean_score",
    "eval/mean_fitness",
    "eval/best_fitness",
)


def _assert_wandb_summary_log(mock_wandb_log: MagicMock) -> None:
    """Assert final wandb.log includes population summary keys (may include per-agent keys)."""
    mock_wandb_log.assert_called()
    logged = mock_wandb_log.call_args[0][0]
    for key in _WANDB_SUMMARY_KEYS:
        assert key in logged


def _make_multi_frequency_selection(seed: int = 0) -> MultiFrequencySelection:
    """Build a six-slot multi-frequency selection (2 subpops x 3) for trainer-routing tests.

    :param seed: Seed for the selection's RNG, defaults to 0.
    :type seed: int, optional
    :return: A six-slot multi-frequency selection with fast/slow subpopulation frequencies.
    :rtype: MultiFrequencySelection
    """
    return MultiFrequencySelection(
        population_size=6,
        n_subpopulations=2,
        evolution_frequency_ratios=[1, 2],
        n_winners=1,
        n_survivors=0,
        n_open_for_migration=1,
        n_losers=1,
        seed=seed,
    )


class DummyEnv(VectorEnv):
    """Minimal vectorized env double. ``vect`` selects ``num_envs`` (2 vs 1);"""

    def __init__(self, state_size, action_size, vect=True, num_envs=2):
        self._single_state_size = tuple(state_size)
        self.action_size = action_size
        self.vect = vect
        self.num_envs = num_envs if vect else 1
        self.n_envs = self.num_envs
        self.state_size = (self.num_envs, *self._single_state_size)
        self.single_observation_space = Box(0.0, 1.0, self._single_state_size)
        self.single_action_space = Box(-1.0, 1.0, (action_size,))
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self, seed=None, options=None):
        return np.random.rand(*self.state_size).astype(np.float32), {}

    def step(self, action):
        return (
            np.random.rand(*self.state_size).astype(np.float32),
            np.random.randint(0, 5, self.num_envs),
            np.random.randint(0, 2, self.num_envs),
            np.random.randint(0, 2, self.num_envs),
            {},
        )


class DummyBanditEnv:
    def __init__(self, state_size, arms):
        self.arms = arms
        self.state_size = (arms, *state_size)
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
        self.steps = self.metrics.steps
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
            # Prioritized / n-step path expects idxs + priorities for update_priorities
            if "idxs" in experiences.keys():
                idxs = experiences["idxs"]
            else:
                idxs = torch.tensor([0])
            priorities = np.ones(len(idxs), dtype=np.float32)
            return loss, idxs, priorities
        return loss

    def test(self, env, max_steps=None, loop=3, **kwargs):
        rand_int = np.random.uniform(0, 400)
        self.fitness.append(rand_int)
        return rand_int

    def init_training_step(self):
        self.metrics.init_training_step()

    def add_scores(self, scores):
        self.metrics.add_scores(scores)
        self.scores = self.metrics.scores

    def finalize_training_step(self, num_steps):
        self.metrics.finalize_training_step(num_steps)
        self.steps_per_second = self.metrics.steps_per_second
        self.steps = self.metrics.steps

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


class DummyMultiEnv(PettingZooVecEnv):  # pylint: disable=overwritten-inherited-attribute
    """Mimics a vectorized multi-agent parallel environment with num_envs=1."""

    def __init__(self, state_dims, action_dims):
        agents = ["agent_0", "other_agent_0"]
        super().__init__(
            num_envs=1,
            observation_spaces={agent: Box(0, 255, state_dims) for agent in agents},
            action_spaces={agent: Discrete(5) for agent in agents},
            possible_agents=agents,
        )
        self.state_dims = state_dims
        self.state_size = self.state_dims
        self.action_dims = action_dims
        self.action_size = self.action_dims
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


class DummyMultiParallelEnv(ParallelEnv):
    """Single PettingZoo ParallelEnv double with non-batched dict returns."""

    def __init__(self, state_dims, action_dims):
        self.state_dims = state_dims
        self.state_size = self.state_dims
        self.action_dims = action_dims
        self.action_size = self.action_dims
        self.agents = ["agent_0", "other_agent_0"]
        self.possible_agents = ["agent_0", "other_agent_0"]
        self.render_mode = None
        self.metadata = {"name": "dummy_multi_parallel_v0"}
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
            agent: np.random.rand(*self.state_dims) for agent in self.agents
        }, self.info

    def step(self, action):
        return (
            {agent: np.random.rand(*self.state_dims) for agent in self.agents},
            {agent: float(np.random.rand()) for agent in self.agents},
            {agent: bool(np.random.randint(0, 2)) for agent in self.agents},
            {agent: bool(np.random.randint(0, 2)) for agent in self.agents},
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
        self.steps = self.metrics.steps
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
        return pop[0], pop, None


class DummyMutations:
    def __init__(self, dormant_reset_param_mut=False):
        self.dormant_reset_param_mut = dormant_reset_param_mut

    def mutation(self, pop, pre_training_mut=False, indices=None):
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


class DummyPrioritizedMemory(DummyMemory, PrioritizedReplayBuffer):  # pylint: disable=overwritten-inherited-attribute
    def __init__(self):
        super().__init__()


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

    Mirrors the API of :class:`ReplayBuffer` that the
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
    mock.steps = mock.metrics.steps
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
    mock.init_training_step.side_effect = lambda: mock.metrics.init_training_step()
    mock.add_scores.side_effect = lambda scores: mock.metrics.add_scores(scores)
    mock.finalize_training_step.side_effect = lambda num_steps: (
        mock.metrics.finalize_training_step(num_steps)
    )
    mock.learn.side_effect = lambda *args, **kwargs: random.random()
    mock.save_checkpoint.side_effect = lambda *a, **kw: None
    mock.load_checkpoint.side_effect = lambda *a, **kw: None
    mock.wrap_models.side_effect = lambda *a, **kw: None
    mock.unwrap_models.side_effect = lambda *a, **kw: None
    return mock


def _instrument_callable_methods(agent, *method_names):
    """Wrap real methods in MagicMock so tests can use assert_called()."""
    for name in method_names:
        setattr(agent, name, MagicMock(side_effect=getattr(agent, name)))
    return agent


@pytest.fixture
def mocked_agent_off_policy(env, algo):
    # DummyAgentOffPolicy provides the `NStepAgent` members (batch_size, beta,
    # learn) that the prioritized/n-step path reads.
    if algo == RainbowDQN:
        agent = DummyAgentOffPolicy(5, env, 0.4, algo="Rainbow DQN")
        agent.action_dim = 2
        return _instrument_callable_methods(
            agent,
            "get_action",
            "learn",
            "test",
            "wrap_models",
            "unwrap_models",
            "set_training_mode",
            "init_training_step",
            "add_scores",
            "finalize_training_step",
            "save_checkpoint",
            "load_checkpoint",
            "reset_action_noise",
        )

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

    mock_agent.learn.side_effect = lambda experiences, **kwargs: random.random()

    mock_agent.algo = {
        DQN: "DQN",
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
    # ``spec=VectorEnv`` makes ``isinstance(env, VectorEnv)`` True so
    # train_off/on_policy skips DummyVecEnv wrapping (which requires real Spaces).
    mock_env = MagicMock(spec=VectorEnv)
    mock_env.action_size = action_size
    mock_env.vect = vect
    n_envs = num_envs if vect else 1
    mock_env.num_envs = n_envs
    mock_env.state_size = (n_envs, *state_size)
    mock_env.single_observation_space = Box(0.0, 1.0, tuple(state_size))
    mock_env.single_action_space = Box(-1.0, 1.0, (action_size,))
    mock_env.observation_space = batch_space(mock_env.single_observation_space, n_envs)
    mock_env.action_space = batch_space(mock_env.single_action_space, n_envs)

    def reset(seed=None, options=None):
        return np.random.rand(*mock_env.state_size).astype(np.float32), {}

    mock_env.reset.side_effect = reset

    def step(action):
        return (
            np.random.rand(*mock_env.state_size).astype(np.float32),
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
    mock_env.state_size = (action_size, *state_size)
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

    def mutation(pop, pre_training_mut=False, indices=None):
        return pop

    mock_mutations.mutation.side_effect = mutation
    return mock_mutations


@pytest.fixture
def mocked_tournament():
    mock_tournament = MagicMock()

    def select(pop):
        return pop[0], pop, None

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


class TestTrainOffPolicy:
    def test_real_rainbow_takes_the_rainbow_action_branch(self):
        """The action-selection dispatch is by concrete class."""
        import gymnasium as gym

        from agilerl.algorithms import RainbowDQN

        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        agent = RainbowDQN(
            vec_env.single_observation_space,
            vec_env.single_action_space,
            batch_size=4,
            learn_step=1,
        )
        pop, fitnesses = train_off_policy(
            vec_env,
            "CartPole-v1",
            "Rainbow DQN",
            [agent],
            PrioritizedReplayBuffer(max_size=100, device="cpu"),
            max_steps=12,
            evo_steps=12,
            eval_steps=2,
            eval_loop=1,
            verbose=False,
        )
        assert len(pop) == 1
        assert len(fitnesses) == 1

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_BOTH)
    def test_train_off_policy(
        self, env, population_off_policy, tournament, mutations, memory
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_off_policy)
        assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize(
        ("algo", "num_envs", "learn_step"), [(DQN, 2, 1), (DDPG, 2, 1), (TD3, 1, 2)]
    )
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_BOTH)
    def test_train_off_policy_agent_calls_made(
        self,
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
                env.n_envs = num_envs
                env.state_size = (num_envs, *env._single_state_size)

            _pop, _ = train_off_policy(
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
                n_step_memory=None,
                selection_strategy=tournament,
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
            mocked_agent_off_policy.get_action.assert_called()
            mocked_agent_off_policy.learn.assert_called()
            mocked_agent_off_policy.test.assert_called()
            if accelerator is not None:
                mocked_agent_off_policy.wrap_models.assert_called()
                mocked_agent_off_policy.unwrap_models.assert_called()

    @pytest.mark.parametrize(("per", "n_step"), [(False, True), (True, True)])
    @pytest.mark.parametrize(("num_envs", "learn_step"), [(2, 1)])
    @pytest.mark.parametrize("algo", [RainbowDQN])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_BOTH)
    def test_train_off_policy_agent_calls_made_rainbow(
        self,
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
        if env.vect:
            env.num_envs = num_envs
            env.n_envs = num_envs
            env.state_size = (num_envs, *env._single_state_size)
        buf = DummyPrioritizedMemory() if per else memory

        _pop, _ = train_off_policy(
            env,
            "env_name",
            "Rainbow DQN",
            mock_population,
            buf,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step_memory=n_step_memory,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            save_elite=True,
        )

        mocked_agent_off_policy.get_action.assert_called()
        mocked_agent_off_policy.learn.assert_called()
        mocked_agent_off_policy.test.assert_called()
        mocked_agent_off_policy.get_action.assert_called()
        mocked_agent_off_policy.learn.assert_called()
        mocked_agent_off_policy.test.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_NOVECT)
    def test_train_off_policy_save_elite_warning(
        self,
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
            _pop, _ = train_off_policy(
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
                n_step_memory=None,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                save_elite=False,
                elite_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_NOVECT)
    def test_train_off_policy_checkpoint_warning(
        self,
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
            _pop, _ = train_off_policy(
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
                n_step_memory=None,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                checkpoint=None,
                checkpoint_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_NOVECT)
    def test_actions_histogram(
        self, env, population_off_policy, tournament, mutations, memory
    ):
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_off_policy)
        assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_replay_buffer_calls(
        self,
        mocked_memory,
        env,
        population_off_policy,
        tournament,
        mutations,
    ):
        _pop, _ = train_off_policy(
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_memory.add.assert_called()
        mocked_memory.sample.assert_called()

    @pytest.mark.parametrize("per", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_alternate_buffer_calls(
        self,
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
        _pop, _ = train_off_policy(
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
            n_step_memory=mocked_n_step_memory,
            selection_strategy=tournament,
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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_env_calls(
        self,
        mocked_env,
        memory,
        population_off_policy,
        tournament,
        mutations,
    ):
        _pop, _ = train_off_policy(
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_env.step.assert_called()
        mocked_env.reset.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_tourn_mut_calls(
        self,
        env,
        memory,
        population_off_policy,
        mocked_tournament,
        mocked_mutations,
    ):
        _pop, _ = train_off_policy(
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
            n_step_memory=None,
            selection_strategy=mocked_tournament,
            mutation=mocked_mutations,
            wb=False,
        )
        mocked_mutations.mutation.assert_called()
        mocked_tournament.select.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _IMG_NOVECT)
    def test_train_off_policy_rgb_input(
        self,
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_off_policy)
        assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize("per", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_using_alternate_buffers(
        self,
        env,
        memory,
        population_off_policy,
        tournament,
        mutations,
        n_step_memory,
        per,
    ):
        buf = DummyPrioritizedMemory() if per else memory
        pop, _ = train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory=buf,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            n_step_memory=n_step_memory,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_off_policy)
        assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _IMG_VECT)
    def test_train_off_policy_using_alternate_buffers_rgb(
        self,
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
            n_step_memory=n_step_memory,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_off_policy)
        assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_distributed(
        self,
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        assert len(pop) == len(population_off_policy)
        assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_wandb_init_log(
        self, env, population_off_policy, tournament, mutations, memory
    ):
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
                n_step_memory=None,
                selection_strategy=tournament,
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
            _assert_wandb_summary_log(mock_wandb_log)
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize("accelerator", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_wandb_init_log_distributed(
        self,
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
                n_step_memory=None,
                selection_strategy=tournament,
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
            _assert_wandb_summary_log(mock_wandb_log)
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_early_stop_wandb(
        self, env, population_off_policy, tournament, mutations, memory
    ):
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
                n_step_memory=None,
                selection_strategy=tournament,
                mutation=mutations,
                wb=True,
                wandb_api_key="testing",
            )
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_save_elite(
        self,
        env,
        population_off_policy,
        tournament,
        mutations,
        memory,
        tmp_path,
    ):
        elite_path = str(tmp_path / "checkpoint.pt")
        _pop, _ = train_off_policy(
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            save_elite=True,
            elite_path=elite_path,
        )
        assert os.path.isfile(elite_path)

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_save_checkpoint(
        self,
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
        _pop, _ = train_off_policy(
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
            n_step_memory=None,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=10,
            checkpoint_path=checkpoint_path,
            accelerator=accelerator,
        )
        for i in range(6):  # iterate through the population indices
            assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_wandb_kwargs_update(self, env, memory):
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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_per_nstep_none_branches(self, env):
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
            DummyPrioritizedMemory(),
            max_steps=4,
            evo_steps=4,
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
            DummyPrioritizedMemory(),
            max_steps=4,
            evo_steps=4,
            n_step_memory=None,
            verbose=False,
        )
        assert any(item is None for item in agent_le.captured)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_wandb_dqn_and_ddpg_loss_branches(self, env, monkeypatch):
        class DQNLossAgent(DummyAgentOffPolicy):
            def get_action(self, *args, **kwargs):
                return np.array([0, 1], dtype=int)

            def learn(self, experiences, n_experiences=None, per=False):
                return 0.25

        class DDPGLossAgent(DummyAgentOffPolicy):
            def learn(self, experiences, n_experiences=None, per=False):
                return (0.1, 0.2)

        dqn_agent = DQNLossAgent(5, env, 0.4)
        dqn_agent.steps = 0
        ddpg_agent = DDPGLossAgent(5, env, 0.4)
        ddpg_agent.steps = 0

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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_early_stop_wb_branch(self, env):
        agent = DummyAgentOffPolicy(5, env, 0.4)
        agent.steps = 0

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


class TestTrainTargetEarlyReturn:
    """Cover ``population.should_stop`` early-return branches in train loops."""

    @staticmethod
    def _population_with_min_evo(agents: list) -> Population:
        return Population(agents=agents, min_evo_steps=0)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_returns_when_target_met(self, env):
        agent = DummyAgentOffPolicy(5, env, 0.4)
        agent.fitness = [100.0]
        population = self._population_with_min_evo([agent])

        with (
            patch(
                "agilerl.training.train_off_policy.Population",
                return_value=population,
            ),
            patch("agilerl.utils.utils.init_wandb"),
            patch("agilerl.logger.wandb.run", new=MagicMock()),
            patch("agilerl.logger.wandb.log"),
            patch.object(population, "finish") as mock_finish,
        ):
            agents, fitnesses = train_off_policy(
                env,
                "env_name",
                "algo",
                [agent],
                DummyMemory(),
                max_steps=100,
                evo_steps=2,
                target=-1.0,
                wb=True,
                verbose=False,
            )

        assert agents is population.agents
        assert fitnesses == population.last_fitnesses
        mock_finish.assert_called_once()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_returns_when_target_met(self, env):
        agent = DummyAgentOnPolicy(5, env)
        agent.fitness = [100.0]
        population = self._population_with_min_evo([agent])

        with (
            patch(
                "agilerl.training.train_on_policy.Population",
                return_value=population,
            ),
            patch("agilerl.utils.utils.init_wandb"),
            patch("agilerl.logger.wandb.run", new=MagicMock()),
            patch("agilerl.logger.wandb.log"),
            patch.object(population, "finish") as mock_finish,
        ):
            agents, fitnesses = train_on_policy(
                env,
                "env_name",
                "algo",
                [agent],
                max_steps=100,
                evo_steps=2,
                target=-1.0,
                wb=True,
                verbose=False,
            )

        assert agents is population.agents
        assert fitnesses == population.last_fitnesses
        mock_finish.assert_called_once()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_returns_when_target_met(self, env, dummy_h5py_data):
        agent = DummyAgentOffPolicy(5, env, 0.4)
        agent.fitness = [100.0]
        population = self._population_with_min_evo([agent])
        memory = DummyMemory()
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
                "agilerl.training.train_offline.Population",
                return_value=population,
            ),
            patch("agilerl.utils.utils.init_wandb"),
            patch("agilerl.logger.wandb.run", new=MagicMock()),
            patch("agilerl.logger.wandb.log"),
            patch.object(population, "finish") as mock_finish,
        ):
            agents, fitnesses = train_offline(
                env,
                "env_name",
                "algo",
                [agent],
                memory,
                dataset=dummy_h5py_data,
                max_steps=100,
                evo_steps=2,
                target=-1.0,
                wb=True,
                verbose=False,
            )

        assert agents is population.agents
        assert fitnesses == population.last_fitnesses
        mock_finish.assert_called_once()

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_off_policy_returns_when_target_met(
        self,
        multi_env,
        multi_memory,
    ):
        agent = DummyMultiAgent(1, multi_env, on_policy=False)
        agent.fitness = [100.0]
        population = self._population_with_min_evo([agent])
        should_stop_results: list[bool] = []

        def should_stop_spy(target):
            result = Population.should_stop(population, target)
            should_stop_results.append(result)
            return result

        with (
            patch(
                "agilerl.training.train_multi_agent_off_policy.Population",
                return_value=population,
            ),
            patch("agilerl.utils.utils.init_wandb"),
            patch("agilerl.logger.wandb.run", new=MagicMock()),
            patch("agilerl.logger.wandb.log"),
            patch.object(population, "finish") as mock_finish,
            patch.object(population, "should_stop", side_effect=should_stop_spy),
        ):
            agents, fitnesses = train_multi_agent_off_policy(
                multi_env,
                "env_name",
                "algo",
                [agent],
                multi_memory,
                max_steps=100,
                evo_steps=2,
                target=-1.0,
                wb=True,
                verbose=False,
            )

        assert agents is population.agents
        assert fitnesses == population.last_fitnesses
        mock_finish.assert_called_once()
        assert should_stop_results == [True]
        assert agent.steps < 100

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_on_policy_returns_when_target_met(self, multi_env):
        agent = DummyMultiAgent(1, multi_env, on_policy=True)
        agent.fitness = [100.0]
        population = self._population_with_min_evo([agent])
        should_stop_results: list[bool] = []

        def should_stop_spy(target):
            result = Population.should_stop(population, target)
            should_stop_results.append(result)
            return result

        with (
            patch(
                "agilerl.training.train_multi_agent_on_policy.Population",
                return_value=population,
            ),
            patch("agilerl.utils.utils.init_wandb"),
            patch("agilerl.logger.wandb.run", new=MagicMock()),
            patch("agilerl.logger.wandb.log"),
            patch.object(population, "finish") as mock_finish,
            patch.object(population, "should_stop", side_effect=should_stop_spy),
        ):
            agents, fitnesses = train_multi_agent_on_policy(
                multi_env,
                "env_name",
                "algo",
                [agent],
                max_steps=100,
                evo_steps=2,
                target=-1.0,
                wb=True,
                verbose=False,
            )

        assert agents is population.agents
        assert fitnesses == population.last_fitnesses
        mock_finish.assert_called_once()
        assert should_stop_results == [True]
        assert agent.steps < 100

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_requires_dataset_or_minari(self, env):
        agent = DummyAgentOffPolicy(5, env, 0.4)
        memory = DummyMemory()

        with pytest.raises(ValueError, match="minari_dataset_id"):
            train_offline(
                env,
                "env_name",
                "algo",
                [agent],
                memory,
                dataset=None,
                minari_dataset_id=None,
                max_steps=2,
                evo_steps=2,
                verbose=False,
            )


class TestTrainOnPolicy:
    @pytest.mark.parametrize(
        ("state_size", "action_size", "vect", "algo"), [((6,), 2, True, PPO)]
    )
    def test_train_on_policy_agent_calls_made(
        self,
        env,
        algo,
        mocked_agent_on_policy,
        tournament,
        mutations,
    ):
        for accelerator_flag in [True, False]:
            accelerator = Accelerator() if accelerator_flag else None
            mock_population = [mocked_agent_on_policy for _ in range(6)]
            _pop, _ = train_on_policy(
                env,
                "env_name",
                "algo",
                mock_population,
                init_hp=None,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
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
            mocked_agent_on_policy.get_action.assert_called()
            mocked_agent_on_policy.learn.assert_called()
            mocked_agent_on_policy.test.assert_called()
            if accelerator is not None:
                mocked_agent_on_policy.wrap_models.assert_called()
                mocked_agent_on_policy.unwrap_models.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_NOVECT)
    def test_train_on_policy_save_elite_warning(
        self,
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
            _pop, _ = train_on_policy(
                env,
                "env_name",
                "algo",
                population_on_policy,
                init_hp=None,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                save_elite=False,
                elite_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_NOVECT)
    def test_train_on_policy_checkpoint_warning(
        self,
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
            _pop, _ = train_on_policy(
                env,
                "env_name",
                "algo",
                population_on_policy,
                init_hp=None,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                checkpoint=None,
                checkpoint_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_env_calls(
        self,
        mocked_env,
        population_on_policy,
        tournament,
        mutations,
    ):
        _pop, _ = train_on_policy(
            mocked_env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_env.step.assert_called()
        mocked_env.reset.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_tourn_mut_calls(
        self,
        env,
        population_on_policy,
        mocked_tournament,
        mocked_mutations,
    ):
        _pop, _ = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=mocked_tournament,
            mutation=mocked_mutations,
            wb=False,
        )
        mocked_mutations.mutation.assert_called()
        mocked_tournament.select.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_on_policy)
        assert len(pop) == len(population_on_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _IMG_NOVECT)
    def test_train_on_policy_rgb_input(
        self, env, population_on_policy, tournament, mutations
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_on_policy)
        assert len(pop) == len(population_on_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_distributed(
        self, env, population_on_policy, tournament, mutations
    ):
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        assert len(pop) == len(population_on_policy)
        assert len(pop) == len(population_on_policy)

    @pytest.mark.parametrize("accelerator", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_wandb_init_log_on_policy(
        self,
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
                selection_strategy=tournament,
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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_early_stop_wandb_on_policy(
        self, env, population_on_policy, tournament, mutations
    ):
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
                selection_strategy=tournament,
                mutation=mutations,
                wb=True,
                wandb_api_key="testing",
            )
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_save_elite(
        self,
        env,
        population_on_policy,
        tournament,
        mutations,
        accelerator_flag,
        tmp_path,
    ):
        accelerator = Accelerator() if accelerator_flag else None
        elite_path = str(tmp_path / "elite")
        _pop, _ = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            save_elite=True,
            elite_path=elite_path,
            accelerator=accelerator,
        )
        assert os.path.isfile(f"{elite_path}.pt")

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_save_checkpoint(
        self,
        env,
        population_on_policy,
        tournament,
        mutations,
        accelerator_flag,
        tmpdir,
    ):
        accelerator = Accelerator() if accelerator_flag else None
        checkpoint_path = str(Path(tmpdir) / "checkpoint")
        _pop, _ = train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            init_hp=None,
            mut_p=None,
            max_steps=500,
            evo_steps=500,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=10,
            checkpoint_path=checkpoint_path,
            accelerator=accelerator,
        )
        for i in range(6):  # iterate through the population indices
            assert os.path.isfile(f"{checkpoint_path}_{i}_{512}.pt")

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_wandb_kwargs_update(self, env):
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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_recurrent_collect_rollouts_import_branch(
        self, env, monkeypatch
    ):
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

    def test_train_on_policy_clip_box_without_squash_and_scalar_done(self, monkeypatch):
        monkeypatch.setattr(
            agilerl.rollouts.on_policy, "StochasticActor", DummyStochastic
        )

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

    def test_train_on_policy_clip_box_with_squash(self, monkeypatch):
        monkeypatch.setattr(
            agilerl.rollouts.on_policy, "StochasticActor", DummyStochastic
        )

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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_early_stop_wb_branch(self, env):
        agent = DummyAgentOnPolicy(5, env)
        agent.steps = 0
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


class TestTrainMultiAgentOffPolicy:
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_wraps_env_without_num_envs(self, state_size, action_size, multi_memory):
        env = DummyMultiParallelEnv(state_size, action_size)
        agent = DummyMultiAgent(1, env, on_policy=False)
        wrapped = DummyMultiEnv(state_size, action_size)

        with patch(
            "agilerl.training.train_multi_agent_off_policy.PzDummyVecEnv",
            return_value=wrapped,
        ) as mock_wrap:
            train_multi_agent_off_policy(
                env,
                "env_name",
                "algo",
                [agent],
                multi_memory,
                max_steps=2,
                evo_steps=2,
                verbose=False,
            )

        mock_wrap.assert_called_once_with(env)

    @pytest.mark.parametrize("sum_scores", [True, False])
    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_off_policy(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            sum_scores=sum_scores,
        )

        assert len(pop) == len(population_multi_agent)

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_off_policy_distributed(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            accelerator=accelerator,
        )

        assert len(pop) == len(population_multi_agent)
        assert len(pop) == len(population_multi_agent)

    def test_train_multi_agent_off_policy_agent_masking(self):
        pass

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _IMG)
    def test_train_multi_agent_off_policy_rgb(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
        )

        assert len(pop) == len(population_multi_agent)
        assert len(pop) == len(population_multi_agent)

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _IMG)
    def test_train_multi_agent_off_policy_rgb_vectorized(
        self,
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
            DummyMultiParallelEnv,
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
            selection_strategy=tournament,
            mutation=mutations,
        )
        assert len(pop) == len(population_multi_agent)
        env.close()

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_save_elite_warning(
        self,
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
            _pop, _ = train_multi_agent_off_policy(
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
                selection_strategy=tournament,
                mutation=mutations,
                save_elite=False,
                elite_path="path",
            )

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_checkpoint_warning(
        self,
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
            _pop, _ = train_multi_agent_off_policy(
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
                selection_strategy=tournament,
                mutation=mutations,
                checkpoint=None,
                checkpoint_path="path",
            )

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_wandb_init_log(
        self,
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
                selection_strategy=tournament,
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
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_multi_agent_early_stop(
        self,
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
                selection_strategy=tournament,
                mutation=mutations,
                wb=True,
                wandb_api_key="testing",
            )
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize("algo", [MADDPG])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_off_policy_calls(
        self,
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
        mock_population = [mocked_multi_agent for _ in range(6)]

        _pop, _ = train_multi_agent_off_policy(
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
            selection_strategy=tournament,
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
        for agent in mock_population:
            agent.get_action.assert_called()
            agent.learn.assert_called()
            agent.test.assert_called()
            if accelerator is not None:
                agent.wrap_models.assert_called()
                agent.unwrap_models.assert_called()

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_env_calls(
        self,
        mocked_multi_env,
        multi_memory,
        population_multi_agent,
        on_policy,
        tournament,
        mutations,
    ):
        _pop, _ = train_multi_agent_off_policy(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_multi_env.step.assert_called()
        mocked_multi_env.reset.assert_called()

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_tourn_mut_calls(
        self,
        multi_env,
        multi_memory,
        population_multi_agent,
        on_policy,
        mocked_tournament,
        mocked_mutations,
    ):
        _pop, _ = train_multi_agent_off_policy(
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
            selection_strategy=mocked_tournament,
            mutation=mocked_mutations,
            wb=False,
        )
        mocked_tournament.select.assert_called()
        mocked_mutations.mutation.assert_called()

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_memory_calls(
        self,
        multi_env,
        mocked_multi_memory,
        population_multi_agent,
        on_policy,
        tournament,
        mutations,
    ):
        _pop, _ = train_multi_agent_off_policy(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_multi_memory.sample.assert_called()
        mocked_multi_memory.add.assert_called()

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_save_elite(
        self,
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
        _pop, _ = train_multi_agent_off_policy(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            save_elite=True,
            elite_path=elite_path,
            accelerator=accelerator,
        )
        assert os.path.isfile(f"{elite_path}.pt")

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_save_checkpoint(
        self,
        multi_env,
        population_multi_agent,
        tournament,
        mutations,
        multi_memory,
        accelerator_flag,
        on_policy,
        tmpdir,
    ):
        accelerator = Accelerator() if accelerator_flag else None
        checkpoint_path = str(Path(tmpdir) / "checkpoint")
        _pop, _ = train_multi_agent_off_policy(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=10,
            checkpoint_path=checkpoint_path,
            accelerator=accelerator,
        )
        for i in range(6):  # iterate through the population indices
            assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_off_policy_learn_step_branch_and_early_stop(
        self,
        multi_env,
        multi_memory,
    ):
        agent = DummyMultiAgent(1, multi_env, on_policy=False)
        agent.learn_step = 2
        agent.steps = 0

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

    def test_train_multi_agent_off_policy_empty_population_rejected(self, multi_memory):
        class EmptyAgentEnv:
            agents: ClassVar[list] = []
            possible_agents: ClassVar[list] = []

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


class TestTrainMultiAgentOnPolicy:
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_wraps_env_without_num_envs(self, state_size, action_size):
        env = DummyMultiParallelEnv(state_size, action_size)
        agent = DummyMultiAgent(1, env, on_policy=True)
        wrapped = DummyMultiEnv(state_size, action_size)

        with patch(
            "agilerl.training.train_multi_agent_on_policy.PzDummyVecEnv",
            return_value=wrapped,
        ) as mock_wrap:
            train_multi_agent_on_policy(
                env,
                "env_name",
                "algo",
                [agent],
                max_steps=2,
                evo_steps=2,
                verbose=False,
            )

        mock_wrap.assert_called_once_with(env)

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("sum_scores", [True, False])
    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_on_policy(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            sum_scores=sum_scores,
            accelerator=accelerator,
        )

        assert len(pop) == len(population_multi_agent)

    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _IMG)
    def test_train_multi_agent_on_policy_rgb_vectorized(
        self,
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
            DummyMultiParallelEnv,
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
            selection_strategy=tournament,
            mutation=mutations,
        )
        assert len(pop) == len(population_multi_agent)
        env.close()

    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_save_elite_warning_on_policy(
        self,
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
            _pop, _ = train_multi_agent_on_policy(
                multi_env,
                "env_name",
                "algo",
                pop=population_multi_agent,
                init_hp=None,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                save_elite=False,
                elite_path="path",
            )

    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_checkpoint_warning_on_policy(
        self,
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
            _pop, _ = train_multi_agent_on_policy(
                multi_env,
                "env_name",
                "algo",
                pop=population_multi_agent,
                init_hp=None,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                checkpoint=None,
                checkpoint_path="path",
            )

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_wandb_init_log_on_policy(
        self,
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
                selection_strategy=tournament,
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
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_multi_agent_early_stop_on_policy(
        self,
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
                selection_strategy=tournament,
                mutation=mutations,
                wb=True,
                wandb_api_key="testing",
            )
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize("algo", [IPPO])
    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_onpolicy_calls(
        self,
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

        _pop, _ = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            mock_population,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
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
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_env_calls_on_policy(
        self,
        mocked_multi_env,
        multi_memory,
        population_multi_agent,
        on_policy,
        tournament,
        mutations,
    ):
        _pop, _ = train_multi_agent_on_policy(
            mocked_multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_multi_env.step.assert_called()
        mocked_multi_env.reset.assert_called()

    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_tourn_mut_calls_on_policy(
        self,
        multi_env,
        multi_memory,
        population_multi_agent,
        on_policy,
        mocked_tournament,
        mocked_mutations,
    ):
        _pop, _ = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=mocked_tournament,
            mutation=mocked_mutations,
            wb=False,
        )
        mocked_tournament.select.assert_called()
        mocked_mutations.mutation.assert_called()

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_save_elite_on_policy(
        self,
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
        _pop, _ = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            save_elite=True,
            elite_path=elite_path,
            accelerator=accelerator,
        )
        assert os.path.isfile(f"{elite_path}.pt")

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_save_checkpoint_on_policy(
        self,
        multi_env,
        population_multi_agent,
        tournament,
        mutations,
        multi_memory,
        accelerator_flag,
        on_policy,
        tmpdir,
    ):
        accelerator = Accelerator() if accelerator_flag else None
        checkpoint_path = str(Path(tmpdir) / "checkpoint")
        _pop, _ = train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            population_multi_agent,
            init_hp=None,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=10,
            checkpoint_path=checkpoint_path,
            accelerator=accelerator,
        )
        for i in range(6):  # iterate through the population indices
            assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_on_policy_compiled_clip_and_early_stop(
        self,
        multi_env,
        monkeypatch,
    ):
        monkeypatch.setattr(
            agilerl.training.train_multi_agent_on_policy,
            "StochasticActor",
            DummyStochastic,
        )

        agent = DummyMultiAgent(1, multi_env, on_policy=True)
        agent.torch_compiler = "compiled"
        agent.steps = 0
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

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_on_policy_compiled_clip_with_squash(
        self,
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

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_on_policy_nan_mean_score_branch(
        self, multi_env, monkeypatch
    ):
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


class TestTrainOffline:
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline(
        self,
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
                "algo",
                population_off_policy,
                memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                accelerator=accelerator,
            )

            assert len(pop) == len(population_off_policy)
            assert len(pop) == len(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_save_elite_warning(
        self,
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
            _pop, _ = train_offline(
                env,
                "env_name",
                "algo",
                population_off_policy,
                memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                save_elite=False,
                elite_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_save_checkpoint_warning(
        self,
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
            _pop, _ = train_offline(
                env,
                "env_name",
                "algo",
                population_off_policy,
                memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                checkpoint=None,
                checkpoint_path="path",
            )

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_wandb_calls(
        self,
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
                "algo",
                population_off_policy,
                memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=mut_p,
                max_steps=50,
                evo_steps=10,
                eval_loop=1,
                selection_strategy=tournament,
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

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_early_stop(
        self,
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
                    "algo",
                    population_off_policy,
                    memory,
                    dataset=dummy_h5py_data,
                    init_hp=offline_init_hp,
                    mut_p=mut_p,
                    target=-10000,
                    max_steps=50,
                    evo_steps=10,
                    eval_loop=1,
                    selection_strategy=tournament,
                    mutation=mutations,
                    wb=True,
                    accelerator=accelerator,
                    wandb_api_key="testing",
                )
                # Assert that wandb.finish was called
                mock_wandb_finish.assert_called()

    @pytest.mark.parametrize("algo", [DQN])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_offline_agent_calls(
        self,
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

            _pop, _ = train_offline(
                env,
                "env_name",
                "algo",
                mock_population,
                memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                accelerator=accelerator,
            )

            mocked_agent_off_policy.learn.assert_called()
            mocked_agent_off_policy.test.assert_called()
            if accelerator is not None:
                mocked_agent_off_policy.wrap_models.assert_called()
                mocked_agent_off_policy.unwrap_models.assert_called()
            mocked_agent_off_policy.learn.assert_called()
            mocked_agent_off_policy.test.assert_called()
            if accelerator is not None:
                mocked_agent_off_policy.wrap_models.assert_called()
                mocked_agent_off_policy.unwrap_models.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_offline_memory_calls(
        self,
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
            _pop, _ = train_offline(
                env,
                "env_name",
                "algo",
                population_off_policy,
                mocked_memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                accelerator=accelerator,
            )
            mocked_memory.add.assert_called()
            mocked_memory.sample.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_offline_mut_tourn_calls(
        self,
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

            _pop, _ = train_offline(
                env,
                "env_name",
                "algo",
                population_off_policy,
                memory,
                dataset=dummy_h5py_data,
                init_hp=offline_init_hp,
                mut_p=None,
                max_steps=50,
                evo_steps=50,
                eval_loop=1,
                selection_strategy=mocked_tournament,
                mutation=mocked_mutations,
                wb=False,
                accelerator=accelerator,
            )
            mocked_tournament.select.assert_called()
            mocked_mutations.mutation.assert_called()

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_save_elite(
        self,
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
        _pop, _ = train_offline(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            dataset=dummy_h5py_data,
            init_hp=offline_init_hp,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            save_elite=True,
            elite_path=elite_path,
        )
        assert os.path.isfile(f"{elite_path}.pt")

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_save_checkpoint(
        self,
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
        _pop, _ = train_offline(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            dataset=dummy_h5py_data,
            init_hp=offline_init_hp,
            mut_p=None,
            max_steps=50,
            evo_steps=50,
            eval_loop=1,
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
            checkpoint=10,
            checkpoint_path=checkpoint_path,
        )
        for i in range(6):  # iterate through the population indices
            assert os.path.isfile(f"{checkpoint_path}_{i}_{50}.pt")

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_minari_branch_and_early_stop(self, env, memory):
        agent = DummyAgentOffPolicy(5, env, 0.4)
        agent.steps = 0
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
                "algo",
                [agent],
                memory,
                dataset={},
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


class TestTrainBandits:
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_bandit)
        assert len(pop) == len(population_bandit)

    @pytest.mark.parametrize("algo", [NeuralUCB])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_agent_calls_made(
        self,
        bandit_env,
        mocked_bandit,
        tournament,
        mutations,
        bandit_memory,
    ):
        for accelerator_flag in [True, False]:
            accelerator = Accelerator() if accelerator_flag else None
            mock_population = [mocked_bandit for _ in range(6)]

            _pop, _ = train_bandits(
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
                selection_strategy=tournament,
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
            mocked_bandit.get_action.assert_called()
            mocked_bandit.learn.assert_called()
            mocked_bandit.test.assert_called()
            if accelerator is not None:
                mocked_bandit.wrap_models.assert_called()
                mocked_bandit.unwrap_models.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_save_elite_warning(
        self,
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
            _pop, _ = train_bandits(
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
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                save_elite=False,
                elite_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_checkpoint_warning(
        self,
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
            _pop, _ = train_bandits(
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
                selection_strategy=tournament,
                mutation=mutations,
                wb=False,
                checkpoint=None,
                checkpoint_path="path",
            )

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_bandit_actions_histogram(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_bandit)
        assert len(pop) == len(population_bandit)

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_replay_buffer_calls(
        self,
        mocked_bandit_memory,
        bandit_env,
        population_bandit,
        tournament,
        mutations,
    ):
        _pop, _ = train_bandits(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_bandit_memory.add.assert_called()
        mocked_bandit_memory.sample.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_bandit_env_calls(
        self,
        mocked_bandit_env,
        bandit_memory,
        population_bandit,
        tournament,
        mutations,
    ):
        _pop, _ = train_bandits(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )
        mocked_bandit_env.step.assert_called()
        mocked_bandit_env.reset.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_tourn_mut_calls(
        self,
        bandit_env,
        bandit_memory,
        population_bandit,
        mocked_tournament,
        mocked_mutations,
    ):
        _pop, _ = train_bandits(
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
            selection_strategy=mocked_tournament,
            mutation=mocked_mutations,
            wb=False,
        )
        mocked_mutations.mutation.assert_called()
        mocked_tournament.select.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size"), _IMG)
    def test_train_bandit_rgb_input(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_bandit)
        assert len(pop) == len(population_bandit)

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_using_alternate_buffers(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_bandit)
        assert len(pop) == len(population_bandit)

    @pytest.mark.parametrize(("state_size", "action_size"), _IMG_SQUARE)
    def test_train_bandit_using_alternate_buffers_rgb(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
        )

        assert len(pop) == len(population_bandit)
        assert len(pop) == len(population_bandit)

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_distributed(
        self,
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            accelerator=accelerator,
        )

        assert len(pop) == len(population_bandit)
        assert len(pop) == len(population_bandit)

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_bandit_wandb_init_log(
        self,
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
                selection_strategy=tournament,
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
            _assert_wandb_summary_log(mock_wandb_log)
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize("accelerator", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_bandit_wandb_init_log_distributed(
        self,
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
                selection_strategy=tournament,
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
            _assert_wandb_summary_log(mock_wandb_log)
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_bandit_early_stop_wandb(
        self,
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
                selection_strategy=tournament,
                mutation=mutations,
                wb=True,
                wandb_api_key="testing",
            )
            # Assert that wandb.finish was called
            mock_wandb_finish.assert_called()

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandit_save_elite(
        self,
        bandit_env,
        population_bandit,
        tournament,
        mutations,
        bandit_memory,
        tmp_path,
    ):
        elite_path = str(tmp_path / "checkpoint.pt")
        _pop, _ = train_bandits(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            save_elite=True,
            elite_path=elite_path,
        )
        assert os.path.isfile(elite_path)

    @pytest.mark.parametrize("accelerator_flag", [False, True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_bandit_train_save_checkpoint(
        self,
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
        _pop, _ = train_bandits(
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
            selection_strategy=tournament,
            mutation=mutations,
            wb=False,
            checkpoint=10,
            checkpoint_path=checkpoint_path,
            accelerator=accelerator,
        )
        for i in range(6):  # iterate through the population indices
            for s in range(5):
                assert os.path.isfile(f"{checkpoint_path}_{i}_{10 * (s + 1)}.pt")


def _route_off_policy(get, **strategy_kwarg):
    pop, _ = train_off_policy(
        get("env"),
        "env_name",
        "algo",
        get("population_off_policy"),
        get("memory"),
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        n_step_memory=None,
        **strategy_kwarg,
        mutation=get("mutations"),
        wb=False,
    )
    return pop


def _route_on_policy(get, **strategy_kwarg):
    pop, _ = train_on_policy(
        get("env"),
        "env_name",
        "algo",
        get("population_on_policy"),
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        **strategy_kwarg,
        mutation=get("mutations"),
        wb=False,
    )
    return pop


def _route_multi_agent_off_policy(get, **strategy_kwarg):
    multi_env = get("multi_env")
    pop, _ = train_multi_agent_off_policy(
        multi_env,
        "env_name",
        "algo",
        pop=[DummyMultiAgent(5, multi_env, False) for _ in range(6)],
        memory=get("multi_memory"),
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        **strategy_kwarg,
        mutation=get("mutations"),
    )
    return pop


def _route_multi_agent_on_policy(get, **strategy_kwarg):
    multi_env = get("multi_env")
    pop, _ = train_multi_agent_on_policy(
        multi_env,
        "env_name",
        "algo",
        pop=[DummyMultiAgent(5, multi_env, True) for _ in range(6)],
        init_hp=None,
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        **strategy_kwarg,
        mutation=get("mutations"),
    )
    return pop


def _route_offline(get, **strategy_kwarg):
    pop, _ = train_offline(
        get("env"),
        "env_name",
        "algo",
        get("population_off_policy"),
        get("memory"),
        dataset=get("dummy_h5py_data"),
        init_hp=get("offline_init_hp"),
        mut_p=None,
        max_steps=50,
        evo_steps=50,
        eval_loop=1,
        **strategy_kwarg,
        mutation=get("mutations"),
        wb=False,
    )
    return pop


def _route_bandits(get, **strategy_kwarg):
    pop, _ = train_bandits(
        get("bandit_env"),
        "bandit_env_name",
        "algo",
        get("population_bandit"),
        get("bandit_memory"),
        init_hp=None,
        mut_p=None,
        max_steps=50,
        episode_steps=5,
        evo_steps=50,
        eval_steps=5,
        eval_loop=1,
        **strategy_kwarg,
        mutation=get("mutations"),
        wb=False,
    )
    return pop


# (trainer module, runner) per non-LLM trainer
_SELECTION_ROUTING_CASES = {
    "off-policy": ("agilerl.training.train_off_policy", _route_off_policy),
    "on-policy": ("agilerl.training.train_on_policy", _route_on_policy),
    "multi-agent off-policy": (
        "agilerl.training.train_multi_agent_off_policy",
        _route_multi_agent_off_policy,
    ),
    "multi-agent on-policy": (
        "agilerl.training.train_multi_agent_on_policy",
        _route_multi_agent_on_policy,
    ),
    "offline": ("agilerl.training.train_offline", _route_offline),
    "bandits": ("agilerl.training.train_bandits", _route_bandits),
}


class TestTrainerSelectionStrategyRouting:
    """Every trainer hands its selection strategy to the one shared entry point."""

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    @pytest.mark.parametrize(
        "case", list(_SELECTION_ROUTING_CASES), ids=list(_SELECTION_ROUTING_CASES)
    )
    def test_trainer_forwards_selection_strategy(
        self, request, case, state_size, action_size, vect
    ):
        module, route = _SELECTION_ROUTING_CASES[case]
        strategy = _make_multi_frequency_selection()

        with patch(
            f"{module}.run_selection_and_mutation",
            side_effect=lambda _strategy, **kwargs: kwargs["population"],
        ) as spy:
            pop = route(request.getfixturevalue, selection_strategy=strategy)

        spy.assert_called_once()
        assert spy.call_args.args[0] is strategy
        assert len(pop) == 6


class TestTrainerDeprecatedTournamentArgument:
    """The superseded ``tournament`` argument still drives evolution, with a warning.

    Every trainer folds it into ``selection_strategy`` via
    :func:`~agilerl.utils.utils.resolve_selection_strategy`, so callers written
    against the old signature keep working unchanged.
    """

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    @pytest.mark.parametrize(
        "case", list(_SELECTION_ROUTING_CASES), ids=list(_SELECTION_ROUTING_CASES)
    )
    def test_deprecated_tournament_argument_reaches_the_entry_point(
        self, request, case, state_size, action_size, vect
    ):
        module, route = _SELECTION_ROUTING_CASES[case]
        strategy = DummyTournament()

        with (
            patch(
                f"{module}.run_selection_and_mutation",
                side_effect=lambda _strategy, **kwargs: kwargs["population"],
            ) as spy,
            pytest.warns(DeprecationWarning, match="'tournament' argument"),
        ):
            pop = route(request.getfixturevalue, tournament=strategy)

        spy.assert_called_once()
        assert spy.call_args.args[0] is strategy
        assert len(pop) == 6


_CROSS_FAMILY_NET_CONFIG = {
    "encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 7}
}


def _single_agent_hp_config() -> HyperparameterConfig:
    return HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
        batch_size=RLParameter(min=8, max=64, dtype=int),
    )


def _multi_agent_hp_config() -> HyperparameterConfig:
    return HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=64, dtype=int),
    )


def _build_single_agent_population(algo_cls, net_config=_CROSS_FAMILY_NET_CONFIG):
    return algo_cls.population(
        size=8,
        observation_space=generate_random_box_space((4,)),
        action_space=generate_discrete_space(2),
        hp_config=_single_agent_hp_config(),
        net_config=net_config,
        device="cpu",
    )


def _build_maddpg_population(net_config=_CROSS_FAMILY_NET_CONFIG):
    return MADDPG.population(
        size=8,
        observation_space=generate_multi_agent_box_spaces(2, (4,)),
        action_space=generate_multi_agent_discrete_spaces(2, 2),
        agent_ids=["agent_0", "agent_1"],
        hp_config=_multi_agent_hp_config(),
        net_config=net_config,
        device="cpu",
    )


def _build_ippo_population(net_config=_CROSS_FAMILY_NET_CONFIG):
    return IPPO.population(
        size=8,
        observation_space=generate_multi_agent_box_spaces(2, (4,)),
        action_space=generate_multi_agent_discrete_spaces(2, 2),
        agent_ids=["agent_0", "agent_1"],
        hp_config=_single_agent_hp_config(),
        net_config=net_config,
        device="cpu",
    )


# One real population per non-LLM algorithm family the operator must support
_CROSS_FAMILY_CASES = {
    "off-policy (DQN)": ("DQN", lambda: _build_single_agent_population(DQN)),
    "multi-agent off-policy (MADDPG)": ("MADDPG", _build_maddpg_population),
    "multi-agent on-policy (IPPO)": ("IPPO", _build_ippo_population),
    "bandit (NeuralUCB)": (
        "NeuralUCB",
        lambda: _build_single_agent_population(NeuralUCB),
    ),
    "offline (CQN)": ("CQN", lambda: _build_single_agent_population(CQN)),
}


class TestMultiFrequencyCrossFamilyEvolution:
    """Multi-frequency selection evolves a real population of every algorithm family."""

    @pytest.mark.parametrize(
        "family", list(_CROSS_FAMILY_CASES), ids=list(_CROSS_FAMILY_CASES)
    )
    def test_evolves_a_real_population_of_every_family(self, family):
        algo_name, build_population = _CROSS_FAMILY_CASES[family]
        population = build_population()
        for agent in population:
            agent.subpopulation_id = agent.index // 4
        strategy = MultiFrequencySelection(
            population_size=8,
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
            seed=0,
        )
        mutation = Mutations(
            no_mutation=0.2,
            architecture=0.0,
            new_layer_prob=0.0,
            parameters=0.4,
            activation=0.0,
            rl_hp=0.4,
            mutation_sd=0.1,
            rand_seed=0,
            device="cpu",
        )

        for cycle in range(3):
            rank_population_by_subpopulation(population)
            doomed = {weakest_agent_index(population, subpop=0)}
            if cycle % 2 == 1:
                doomed.add(weakest_agent_index(population, subpop=1))

            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo=algo_name,
            )

            surviving = {a.index for a in population}
            assert not (doomed & surviving)  # the due subpops really did evolve
            assert len(population) == 8
            assert Counter(a.subpopulation_id for a in population) == Counter(
                {0: 4, 1: 4}
            )
            assert len({a.index for a in population}) == 8
            assert all(isinstance(a, EvolvableAlgorithm) for a in population)


def _try_remove_models_dir() -> bool:
    try:
        shutil.rmtree("models")
        return True
    except OSError:
        return not os.path.exists("models")


def test_remove_saved_models():
    if not os.path.exists("models"):
        return
    for _ in range(3):
        if _try_remove_models_dir():
            return
    shutil.rmtree("models", ignore_errors=True)


def _regrama_mutation() -> Mutations:
    """A ReGraMa-configured operator with the amplified band switched off."""
    return Mutations(
        no_mutation=0.2,
        architecture=0.0,
        new_layer_prob=0.0,
        parameters=0.4,
        activation=0.0,
        rl_hp=0.4,
        mutation_sd=0.1,
        rand_seed=0,
        device="cpu",
        dormant_reset_param_mut=True,
        amplified_gauss_param_mut=False,
        dormant_threshold=0.01,
    )


def _give_population_fitness(population) -> None:
    """Give each agent a distinct fitness so tournament selection can rank."""
    for position, agent in enumerate(population):
        agent.fitness = [float(position)]


def _mark_population_dormant(population) -> None:
    """Give every agent a snapshot in which every measured neuron is dormant."""
    for agent in population:
        agent.grama_scores = grama_scores_for(agent, fill=0.0)


def _pin_population_biases(population, value: float = 1.0) -> None:
    """Pin every one-dimensional bias of every evaluation network to a sentinel.

    The Gaussian pass only writes two-dimensional tensors, so a bias back at zero
    afterwards is a neuron ReGraMa reset.
    """
    with torch.no_grad():
        for agent in population:
            for _network_id, network in regrama.eval_networks(agent):
                for name, param in network.named_parameters():
                    if name.endswith("bias") and param.dim() == 1:
                        param.fill_(value)


def _population_zeroed_biases(population) -> int:
    """Count the pinned biases ReGraMa has zeroed across a whole population."""
    return sum(
        int(bool((value == 0).all()))
        for agent in population
        for _network_id, network in regrama.eval_networks(agent)
        for key, value in network.state_dict().items()
        if key.endswith("bias") and value.dim() == 1
    )


_pinned_weight = 5.0
# A pinned weight can only end up this close to zero if the random-reset band
# redrew it: ordinary noise on a 5.0 weight has a standard deviation of 0.5.
_reset_residual = 2.0


def _random_reset_band_mutation(*, random_reset_param_mut: bool) -> Mutations:
    """A parameter-mutation-only operator with the amplified band switched off."""
    return Mutations(
        no_mutation=0.0,
        architecture=0.0,
        new_layer_prob=0.0,
        parameters=1.0,
        activation=0.0,
        rl_hp=0.0,
        mutation_sd=0.1,
        rand_seed=0,
        device="cpu",
        amplified_gauss_param_mut=False,
        random_reset_param_mut=random_reset_param_mut,
    )


def _mutable_weights(population):
    """Yield every tensor the Gaussian pass is allowed to write."""
    for agent in population:
        for _network_id, network in regrama.eval_networks(agent):
            for key, tensor in network.state_dict().items():
                if tensor.dim() == 2 and "norm" not in key and "lstm" not in key:
                    yield tensor


def _pin_population_weights(population, value: float = _pinned_weight) -> None:
    """Pin every mutable weight of every evaluation network to a sentinel."""
    with torch.no_grad():
        for tensor in _mutable_weights(population):
            tensor.fill_(value)


def _smallest_pinned_magnitude(population) -> float:
    """Return the smallest weight left across the tensors pinned before mutating."""
    return min(tensor.abs().min().item() for tensor in _mutable_weights(population))


class TestRandomResetParameterMutationCrossFamilyEvolution:
    """The random-reset Gaussian band is switchable for every non-LLM family."""

    def evolve(self, family, *, random_reset_param_mut, cycles=3):
        """Run cycles of selection and parameter mutation over a pinned population."""
        algo_name, build_population = _CROSS_FAMILY_CASES[family]
        population = build_population()
        strategy = TournamentSelection(
            tournament_size=2,
            elitism=True,
            population_size=8,
        )
        mutation = _random_reset_band_mutation(
            random_reset_param_mut=random_reset_param_mut
        )

        smallest = _pinned_weight
        for _cycle in range(cycles):
            _give_population_fitness(population)
            _pin_population_weights(population)
            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo=algo_name,
            )
            smallest = min(smallest, _smallest_pinned_magnitude(population))
        return population, smallest

    @pytest.mark.parametrize(
        "family", list(_CROSS_FAMILY_CASES), ids=list(_CROSS_FAMILY_CASES)
    )
    def test_the_band_redraws_weights_when_left_on(self, family):
        population, smallest = self.evolve(family, random_reset_param_mut=True)

        assert len(population) == 8
        assert smallest < _reset_residual

    @pytest.mark.parametrize(
        "family", list(_CROSS_FAMILY_CASES), ids=list(_CROSS_FAMILY_CASES)
    )
    def test_switching_it_off_leaves_trained_weights_alone(self, family):
        population, smallest = self.evolve(family, random_reset_param_mut=False)

        assert len(population) == 8
        assert smallest > _reset_residual
        for agent in population:
            for _network_id, network in regrama.eval_networks(agent):
                assert all(
                    torch.isfinite(value).all()
                    for value in network.state_dict().values()
                )


class TestRegramaCrossFamilyEvolution:
    """ReGraMa evolves a real population of every non-LLM algorithm family."""

    @pytest.mark.parametrize(
        "family", list(_CROSS_FAMILY_CASES), ids=list(_CROSS_FAMILY_CASES)
    )
    def test_tournament_selection_evolves_every_family(self, family):
        algo_name, build_population = _CROSS_FAMILY_CASES[family]
        population = build_population()
        mutation = _regrama_mutation()
        strategy = TournamentSelection(
            tournament_size=2,
            elitism=True,
            population_size=8,
        )

        resets_seen = 0
        for _cycle in range(3):
            _give_population_fitness(population)
            _mark_population_dormant(population)
            _pin_population_biases(population)
            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo=algo_name,
            )
            resets_seen += _population_zeroed_biases(population)

            assert len(population) == 8
            assert all(isinstance(agent, EvolvableAlgorithm) for agent in population)
            for agent in population:
                for _network_id, network in regrama.eval_networks(agent):
                    assert all(
                        torch.isfinite(value).all()
                        for value in network.state_dict().values()
                    )

        assert resets_seen > 0

    @pytest.mark.parametrize(
        "family", list(_CROSS_FAMILY_CASES), ids=list(_CROSS_FAMILY_CASES)
    )
    def test_multi_frequency_selection_evolves_every_family(self, family):
        algo_name, build_population = _CROSS_FAMILY_CASES[family]
        population = build_population()
        for agent in population:
            agent.subpopulation_id = agent.index // 4
        strategy = MultiFrequencySelection(
            population_size=8,
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
            seed=0,
        )
        mutation = _regrama_mutation()

        resets_seen = 0
        for _cycle in range(3):
            rank_population_by_subpopulation(population)
            _mark_population_dormant(population)
            _pin_population_biases(population)
            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo=algo_name,
            )
            resets_seen += _population_zeroed_biases(population)

            assert len(population) == 8
            assert len({agent.index for agent in population}) == 8
            for agent in population:
                for _network_id, network in regrama.eval_networks(agent):
                    assert all(
                        torch.isfinite(value).all()
                        for value in network.state_dict().values()
                    )

        assert resets_seen > 0

    def test_a_child_reads_the_snapshot_captured_while_its_parent_trained(self):
        # Capture happens on the parent, the reset happens on the clone.
        population = _build_single_agent_population(DQN)
        _give_population_fitness(population)
        _mark_population_dormant(population)
        strategy = TournamentSelection(
            tournament_size=2,
            elitism=True,
            population_size=8,
        )

        evolved = run_selection_and_mutation(
            strategy,
            population=population,
            mutation=_regrama_mutation(),
            env_name="Env",
            algo="DQN",
        )

        assert all(agent.grama_scores is not None for agent in evolved)


class TestRegramaTrainerWiring:
    """Every non-LLM trainer enables the capture ReGraMa depends on."""

    @staticmethod
    def assert_capture_enabled(population) -> None:
        assert population
        assert all(agent.capture_grama is True for agent in population)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_off_policy_enables_capture(self, env, population_off_policy):
        train_off_policy(
            env,
            "env_name",
            "algo",
            population_off_policy,
            DummyMemory(),
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            mutation=DummyMutations(dormant_reset_param_mut=True),
            wb=False,
            verbose=False,
        )

        self.assert_capture_enabled(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_on_policy_enables_capture(self, env, population_on_policy):
        train_on_policy(
            env,
            "env_name",
            "algo",
            population_on_policy,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            mutation=DummyMutations(dormant_reset_param_mut=True),
            wb=False,
            verbose=False,
        )

        self.assert_capture_enabled(population_on_policy)

    @pytest.mark.parametrize(("state_size", "action_size", "vect"), _FLAT_VECT)
    def test_train_offline_enables_capture(
        self,
        env,
        population_off_policy,
        memory,
        offline_init_hp,
        dummy_h5py_data,
    ):
        train_offline(
            env,
            "env_name",
            "algo",
            population_off_policy,
            memory,
            dataset=dummy_h5py_data,
            init_hp=offline_init_hp,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            mutation=DummyMutations(dormant_reset_param_mut=True),
            wb=False,
        )

        self.assert_capture_enabled(population_off_policy)

    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_bandits_enables_capture(
        self,
        bandit_env,
        population_bandit,
        bandit_memory,
    ):
        train_bandits(
            bandit_env,
            "bandit_env_name",
            "algo",
            population_bandit,
            bandit_memory,
            max_steps=10,
            episode_steps=5,
            evo_steps=5,
            eval_steps=2,
            eval_loop=1,
            mutation=DummyMutations(dormant_reset_param_mut=True),
            wb=False,
        )

        self.assert_capture_enabled(population_bandit)

    @pytest.mark.parametrize("on_policy", [False])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_off_policy_enables_capture(
        self,
        multi_env,
        population_multi_agent,
        multi_memory,
    ):
        train_multi_agent_off_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            memory=multi_memory,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            mutation=DummyMutations(dormant_reset_param_mut=True),
        )

        self.assert_capture_enabled(population_multi_agent)

    @pytest.mark.parametrize("on_policy", [True])
    @pytest.mark.parametrize(("state_size", "action_size"), _FLAT)
    def test_train_multi_agent_on_policy_enables_capture(
        self,
        multi_env,
        population_multi_agent,
    ):
        train_multi_agent_on_policy(
            multi_env,
            "env_name",
            "algo",
            pop=population_multi_agent,
            max_steps=4,
            evo_steps=4,
            eval_loop=1,
            mutation=DummyMutations(dormant_reset_param_mut=True),
        )

        self.assert_capture_enabled(population_multi_agent)

    def test_off_policy_training_captures_a_snapshot(self):
        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        agent = DQN(
            vec_env.single_observation_space,
            vec_env.single_action_space,
            batch_size=4,
            learn_step=1,
        )

        population, _fitnesses = train_off_policy(
            vec_env,
            "CartPole-v1",
            "DQN",
            [agent],
            ReplayBuffer(max_size=100, device="cpu"),
            max_steps=12,
            evo_steps=12,
            eval_steps=2,
            eval_loop=1,
            mutation=_regrama_mutation(),
            verbose=False,
        )

        assert population[0].capture_grama is True
        assert population[0].grama_scores
        assert any(entry is not None for entry in population[0].grama_scores[0])

    def test_compiled_agent_still_captures_with_a_warning(self):
        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        agent = DQN(
            vec_env.single_observation_space,
            vec_env.single_action_space,
            batch_size=4,
            learn_step=1,
        )
        agent.torch_compiler = "default"

        with pytest.warns(UserWarning, match=r"torch\.compile"):
            population, _fitnesses = train_off_policy(
                vec_env,
                "CartPole-v1",
                "DQN",
                [agent],
                ReplayBuffer(max_size=100, device="cpu"),
                max_steps=12,
                evo_steps=12,
                eval_steps=2,
                eval_loop=1,
                mutation=_regrama_mutation(),
                verbose=False,
            )

        assert population[0].capture_grama is True
        assert any(entry is not None for entry in population[0].grama_scores[0])

    def test_capture_stays_off_without_regrama(self):
        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        agent = DQN(
            vec_env.single_observation_space,
            vec_env.single_action_space,
            batch_size=4,
            learn_step=1,
        )
        mutation = Mutations(0.2, 0.0, 0.0, 0.4, 0.0, 0.4, rand_seed=0, device="cpu")

        population, _fitnesses = train_off_policy(
            vec_env,
            "CartPole-v1",
            "DQN",
            [agent],
            ReplayBuffer(max_size=100, device="cpu"),
            max_steps=12,
            evo_steps=12,
            eval_steps=2,
            eval_loop=1,
            mutation=mutation,
            verbose=False,
        )

        # No hooks are registered, so capture costs nothing when off.
        assert population[0].capture_grama is False
        assert population[0].grama_scores is None


# Normalisation between a widened layer and its consumer re-scales every unit
# together, so the fixup stands down on any network built with the default
# layer_norm=True. The shared table leaves it on, which would leave the whole
# function-preserving suite asserting nothing but that mutation still runs.
_FP_FAMILY_NET_CONFIG = {
    "encoder_config": {
        "hidden_size": [8, 8],
        "min_mlp_nodes": 7,
        "layer_norm": False,
        "activation": "ReLU",
    },
    "head_config": {
        "hidden_size": [8],
        "min_mlp_nodes": 7,
        "layer_norm": False,
        "activation": "ReLU",
    },
}


# Written out rather than derived from the shared table: that table has no
# on-policy case, and both adding one to it and giving it the unnormalised
# config above would silently widen the ReGraMa and multi-frequency suites too.
_ARCH_FAMILY_CASES = {
    "off-policy (DQN)": (
        "DQN",
        lambda: _build_single_agent_population(DQN, _FP_FAMILY_NET_CONFIG),
    ),
    "multi-agent off-policy (MADDPG)": (
        "MADDPG",
        lambda: _build_maddpg_population(_FP_FAMILY_NET_CONFIG),
    ),
    "multi-agent on-policy (IPPO)": (
        "IPPO",
        lambda: _build_ippo_population(_FP_FAMILY_NET_CONFIG),
    ),
    "bandit (NeuralUCB)": (
        "NeuralUCB",
        lambda: _build_single_agent_population(NeuralUCB, _FP_FAMILY_NET_CONFIG),
    ),
    "offline (CQN)": (
        "CQN",
        lambda: _build_single_agent_population(CQN, _FP_FAMILY_NET_CONFIG),
    ),
    "on-policy (PPO)": (
        "PPO",
        lambda: _build_single_agent_population(PPO, _FP_FAMILY_NET_CONFIG),
    ),
}


# No family has anything the fixup declines: every net config below is
# unnormalised and ReLU, and MADDPG's EvolvableMultiInput critic encoder is no
# obstacle to a latent widening, which is surgery on the head alone.
_FP_EXPECTED_DECLINES: dict[str, set[str]] = {}


def _fp_declines(recorded) -> set:
    """Return the reason keys the fixup stood down on, from recorded warnings.

    Anchored on the phrase that introduces the reason, because one reason's
    prose is a substring of another's ("a residual skip ..." inside "a SimBa
    block's residual skip ...") and a bare containment check would report both.
    """
    messages = [str(warning.message) for warning in recorded]
    return {
        reason
        for reason, prose in function_preserving.DECLINE_REASONS.items()
        if any(f"initialisation: {prose}." in message for message in messages)
    }


@contextmanager
def _no_unexpected_fp_fallback(expected=frozenset()):
    """Assert the fixup stood down only where the architecture forces it.

    A subset rather than an equality, since which mutations get sampled is up to
    the operator's RNG; the direction that matters is that nothing *else*
    silently fell back to random initialisation.
    """
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        yield
    assert _fp_declines(recorded) <= set(expected)


def _arch_mutation(seed: int = 0) -> Mutations:
    """Return an architecture-mutation-only operator."""
    return Mutations(
        no_mutation=0.0,
        architecture=1.0,
        new_layer_prob=0.5,
        parameters=0.0,
        activation=0.0,
        rl_hp=0.0,
        mutation_sd=0.1,
        rand_seed=seed,
        device="cpu",
    )


def _population_is_finite(population) -> bool:
    """Return whether every agent's policy weights are still finite."""
    for agent in population:
        policy = getattr(agent, agent.registry.policy())
        for tensor in policy.state_dict().values():
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                return False
    return True


def _architecture_fingerprint(population) -> list:
    """Return every agent's policy parameter shapes."""
    return [
        [
            tuple(tensor.shape)
            for tensor in getattr(agent, agent.registry.policy()).state_dict().values()
        ]
        for agent in population
    ]


class TestFunctionPreservingCrossFamilyEvolution:
    """Architecture mutations evolve a real population of every non-LLM family."""

    @staticmethod
    def evolve(population, strategy, cycles: int = 3):
        """Run several evaluate-select-mutate cycles and return the population."""
        mutation = _arch_mutation()
        applied = []
        for _ in range(cycles):
            _give_population_fitness(population)
            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo="algo",
            )
            applied += [agent.mut for agent in population]
        return population, applied

    @pytest.mark.parametrize(
        "family", list(_ARCH_FAMILY_CASES), ids=list(_ARCH_FAMILY_CASES)
    )
    def test_the_fixup_engages_for_every_family(self, family):
        """Additions reach the fixup rather than falling back to random init.

        Guards the net config as much as the operator: with normalisation left
        on, every addition declines, and the other assertions of this class hold
        whether or not the fixup does anything at all. Exact preservation is
        asserted per architecture in ``tests/test_hpo``; what is family-specific
        is only whether the surgery can run.
        """
        _, build_population = _ARCH_FAMILY_CASES[family]
        population = build_population()

        with _no_unexpected_fp_fallback(_FP_EXPECTED_DECLINES.get(family, frozenset())):
            _population, applied = self.evolve(
                population,
                TournamentSelection(
                    tournament_size=2, elitism=True, population_size=len(population)
                ),
            )

        assert any("add" in (mut or "") for mut in applied)

    @pytest.mark.parametrize(
        "family", list(_ARCH_FAMILY_CASES), ids=list(_ARCH_FAMILY_CASES)
    )
    def test_tournament_selection_evolves_every_family(self, family):
        _, build_population = _ARCH_FAMILY_CASES[family]
        population = build_population()
        before = _architecture_fingerprint(population)

        population, _applied = self.evolve(
            population,
            TournamentSelection(
                tournament_size=2, elitism=True, population_size=len(population)
            ),
        )

        assert len(population) == 8
        assert _population_is_finite(population)
        assert _architecture_fingerprint(population) != before

    @pytest.mark.parametrize(
        "family", list(_ARCH_FAMILY_CASES), ids=list(_ARCH_FAMILY_CASES)
    )
    def test_multi_frequency_selection_evolves_every_family(self, family):
        _, build_population = _ARCH_FAMILY_CASES[family]
        population = build_population()
        for agent in population:
            agent.subpopulation_id = agent.index // 4
        strategy = MultiFrequencySelection(
            population_size=8,
            n_subpopulations=2,
            evolution_frequency_ratios=[1, 2],
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
            seed=0,
        )
        mutation = _arch_mutation()

        for _ in range(3):
            rank_population_by_subpopulation(population)
            population = run_selection_and_mutation(
                strategy,
                population=population,
                mutation=mutation,
                env_name="Env",
                algo="algo",
            )

        assert len(population) == 8
        assert len({agent.index for agent in population}) == 8
        assert _population_is_finite(population)


class _FunctionPreservingParallelEnv(ParallelEnv):
    """Two-agent parallel env with no env-defined action overrides."""

    def __init__(self, state_dims=(4,), action_dims=5):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.agents = ["agent_0", "other_agent_0"]
        self.possible_agents = list(self.agents)
        self.metadata = {"name": "function_preserving_parallel_v0"}
        self.render_mode = None

    def reset(self, seed=None, options=None):
        return self._observations(), {agent: {} for agent in self.agents}

    def step(self, action):
        return (
            self._observations(),
            {agent: float(np.random.rand()) for agent in self.agents},
            dict.fromkeys(self.agents, False),
            dict.fromkeys(self.agents, False),
            {agent: {} for agent in self.agents},
        )

    def _observations(self):
        return {
            agent: np.random.rand(*self.state_dims).astype(np.float32)
            for agent in self.agents
        }

    def action_space(self, agent):
        return Discrete(self.action_dims)

    def observation_space(self, agent):
        return Box(0.0, 1.0, self.state_dims, dtype=np.float32)


class _FunctionPreservingBanditEnv:
    """Contextual bandit env yielding one float32 context per arm."""

    def __init__(self, context_dims=(4,), arms=2):
        self.arms = arms
        self.state_size = (arms, *context_dims)
        self.observation_space = Box(0.0, 1.0, self.state_size, dtype=np.float32)
        self.action_space = Discrete(arms)
        self.num_envs = 1

    def reset(self, seed=None, options=None):
        return self._context()

    def step(self, action):
        return self._context(), np.random.rand(1).astype(np.float32)

    def _context(self):
        return np.random.rand(*self.state_size).astype(np.float32)


class TestFunctionPreservingTrainerWiring:
    """Real agents evolve through every non-LLM trainer with the fixup active.

    Function-preserving initialisation lives entirely inside :class:`Mutations`,
    so no trainer needs wiring of its own; these runs check that each trainer
    drives the operator end to end without crashing or corrupting a population.
    They deliberately assert nothing about preservation itself -- that an
    addition leaves the network's function unchanged is asserted per algorithm
    family in ``tests/test_hpo/test_mutation.py``.
    """

    @staticmethod
    def assert_evolved(population, before) -> None:
        """Assert the population survived and its architectures actually moved."""
        assert population
        assert _population_is_finite(population)
        assert _architecture_fingerprint(population) != before

    def test_off_policy(self):
        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        population = DQN.population(
            size=2,
            observation_space=vec_env.single_observation_space,
            action_space=vec_env.single_action_space,
            net_config=_FP_FAMILY_NET_CONFIG,
            batch_size=4,
            learn_step=1,
            device="cpu",
        )
        before = _architecture_fingerprint(population)

        with _no_unexpected_fp_fallback():
            population, _fitnesses = train_off_policy(
                vec_env,
                "CartPole-v1",
                "DQN",
                population,
                ReplayBuffer(max_size=100, device="cpu"),
                max_steps=24,
                evo_steps=8,
                eval_steps=2,
                eval_loop=1,
                selection_strategy=TournamentSelection(
                    tournament_size=2, elitism=True, population_size=2
                ),
                mutation=_arch_mutation(),
                verbose=False,
            )

        self.assert_evolved(population, before)

    def test_on_policy(self):
        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        population = PPO.population(
            size=2,
            observation_space=vec_env.single_observation_space,
            action_space=vec_env.single_action_space,
            net_config=_FP_FAMILY_NET_CONFIG,
            batch_size=4,
            learn_step=8,
            device="cpu",
        )
        before = _architecture_fingerprint(population)

        with _no_unexpected_fp_fallback():
            population, _fitnesses = train_on_policy(
                vec_env,
                "CartPole-v1",
                "PPO",
                population,
                max_steps=24,
                evo_steps=8,
                eval_steps=2,
                eval_loop=1,
                selection_strategy=TournamentSelection(
                    tournament_size=2, elitism=True, population_size=2
                ),
                mutation=_arch_mutation(),
                verbose=False,
            )

        self.assert_evolved(population, before)

    def test_multi_agent_on_policy(self):
        env = make_multi_agent_vect_envs(
            _FunctionPreservingParallelEnv, num_envs=2, state_dims=(4,), action_dims=5
        )
        population = IPPO.population(
            size=2,
            observation_space=[Box(0.0, 1.0, (4,)), Box(0.0, 1.0, (4,))],
            action_space=[Discrete(5), Discrete(5)],
            agent_ids=["agent_0", "other_agent_0"],
            net_config=_FP_FAMILY_NET_CONFIG,
            device="cpu",
        )
        before = _architecture_fingerprint(population)

        with _no_unexpected_fp_fallback():
            population, _fitnesses = train_multi_agent_on_policy(
                env,
                "env_name",
                "IPPO",
                pop=population,
                max_steps=24,
                evo_steps=8,
                eval_steps=2,
                eval_loop=1,
                selection_strategy=TournamentSelection(
                    tournament_size=2, elitism=True, population_size=2
                ),
                mutation=_arch_mutation(),
                verbose=False,
            )
        env.close()

        self.assert_evolved(population, before)

    def test_multi_agent_off_policy(self):
        env = make_multi_agent_vect_envs(
            _FunctionPreservingParallelEnv, num_envs=2, state_dims=(4,), action_dims=5
        )
        population = MADDPG.population(
            size=2,
            observation_space=[Box(0.0, 1.0, (4,)), Box(0.0, 1.0, (4,))],
            action_space=[Discrete(5), Discrete(5)],
            agent_ids=["agent_0", "other_agent_0"],
            net_config=_FP_FAMILY_NET_CONFIG,
            batch_size=4,
            device="cpu",
        )
        before = _architecture_fingerprint(population)

        with _no_unexpected_fp_fallback({"multi_input"}):
            population, _fitnesses = train_multi_agent_off_policy(
                env,
                "env_name",
                "MADDPG",
                pop=population,
                memory=ReplayBuffer(max_size=100, device="cpu"),
                max_steps=24,
                evo_steps=8,
                eval_steps=2,
                eval_loop=1,
                learning_delay=0,
                selection_strategy=TournamentSelection(
                    tournament_size=2, elitism=True, population_size=2
                ),
                mutation=_arch_mutation(),
                verbose=False,
            )
        env.close()

        self.assert_evolved(population, before)

    def test_bandits(self):
        env = _FunctionPreservingBanditEnv((4,), 2)
        population = NeuralUCB.population(
            size=2,
            observation_space=generate_random_box_space((4,)),
            action_space=generate_discrete_space(2),
            net_config=_FP_FAMILY_NET_CONFIG,
            batch_size=4,
            device="cpu",
        )
        before = _architecture_fingerprint(population)

        with _no_unexpected_fp_fallback():
            population, _fitnesses = train_bandits(
                env,
                "bandit_env_name",
                "NeuralUCB",
                population,
                ReplayBuffer(max_size=100, device="cpu"),
                max_steps=24,
                episode_steps=4,
                evo_steps=8,
                eval_steps=2,
                eval_loop=1,
                selection_strategy=TournamentSelection(
                    tournament_size=2, elitism=True, population_size=2
                ),
                mutation=_arch_mutation(),
                wb=False,
                verbose=False,
            )

        self.assert_evolved(population, before)

    def test_offline(self):
        vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
        population = CQN.population(
            size=2,
            observation_space=vec_env.single_observation_space,
            action_space=vec_env.single_action_space,
            net_config=_FP_FAMILY_NET_CONFIG,
            batch_size=4,
            learn_step=1,
            device="cpu",
        )
        dataset = {
            "observations": np.random.randn(16, 4).astype(np.float32),
            "actions": np.random.randint(0, 2, size=(16, 1)),
            "rewards": np.random.randn(16).astype(np.float32),
            "terminals": np.zeros(16, dtype=bool),
        }
        before = _architecture_fingerprint(population)

        with _no_unexpected_fp_fallback():
            population, _fitnesses = train_offline(
                vec_env,
                "CartPole-v1",
                "CQN",
                population,
                ReplayBuffer(max_size=100, device="cpu"),
                dataset=dataset,
                max_steps=24,
                evo_steps=8,
                eval_steps=2,
                eval_loop=1,
                selection_strategy=TournamentSelection(
                    tournament_size=2, elitism=True, population_size=2
                ),
                mutation=_arch_mutation(),
                wb=False,
                verbose=False,
            )

        self.assert_evolved(population, before)
