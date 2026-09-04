# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the refactored Trainer abstraction.

Covers:
- Trainer base class (algo resolution from string / spec)
- LocalTrainer (construction, train delegation)
- ArenaTrainer (construction, manifest building, train delegation)
- LLM algorithm integration (DPO, GRPO) with mocked dependencies
- Multi-frequency selection wiring (construction, manifest round trip, evolution)
"""

from __future__ import annotations

import contextlib
import importlib
import warnings
from collections import Counter
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from gymnasium.spaces import Box, Discrete
from pydantic import BaseModel

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import DQN
from agilerl.algorithms.core.base import EvolvableAlgorithm
from agilerl.arena.models import BanditEnvSpec as ArenaBanditEnvSpec
from agilerl.arena.models import GymEnvSpec as ArenaEnvSpec
from agilerl.arena.models import LLMEnvSpec as ArenaLLMEnvSpec
from agilerl.arena.models.algorithms import DPOSpec, GRPOSpec, LLMAlgorithmSpec
from agilerl.builders import select_builder
from agilerl.components.replay_buffer import MultiStepReplayBuffer, ReplayBuffer
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.models import (
    MANIFEST_REGISTRY,
    DDPGSpec,
    DQNSpec,
    PPOSpec,
    TD3Spec,
)
from agilerl.models.env import GymEnvSpec
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
    TournamentSelectionSpec,
)
from agilerl.models.networks import MlpSpec, QNetworkSpec, StochasticActorSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.strategies import (
    LLMDatasetStrategy,
    LLMRolloutStrategy,
    SingleAgentOnPolicyStrategy,
    select_strategy,
)
from agilerl.training.trainer import ArenaTrainer, LocalTrainer, Trainer
from agilerl.utils.trainer_utils import (
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_tournament_from_spec,
    create_population_from_spec,
)
from agilerl.utils.utils import run_selection_and_mutation
from tests.helper_functions import (
    rank_population_by_subpopulation,
    weakest_agent_index,
)


class DummyEnv:
    """Minimal gym-like environment for unit tests.

    Supports the vectorized-env attributes that ``LocalTrainer`` and
    ``build_train_kwargs`` rely on (``single_observation_space``,
    ``single_action_space``, ``num_envs``).  Also provides a
    ``make_env`` method so it can be used directly as an environment
    spec in ``LocalTrainer._make_env()``.
    """

    def __init__(self) -> None:
        self.name = "DummyEnv-v0"
        self.observation_space = Box(low=-1, high=1, shape=(4,))
        self.action_space = Discrete(2)
        self.num_envs = 1

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def single_observation_space(self, agent=None):
        return self.observation_space

    def single_action_space(self, agent=None):
        return self.action_space

    def make_single_env(self):
        return self

    def make_env(self, **kwargs):
        return self

    def close(self):
        pass

    def __str__(self) -> str:
        return "DummyEnv"


@pytest.fixture
def env():
    return DummyEnv()


@pytest.fixture
def ppo_spec() -> PPOSpec:
    return PPOSpec(
        learn_step=128,
        net_config=StochasticActorSpec(
            encoder_config=MlpSpec(hidden_size=[64]),
            head_config=MlpSpec(hidden_size=[64]),
        ),
    )


@pytest.fixture
def ddpg_spec() -> DDPGSpec:
    return DDPGSpec()


@pytest.fixture
def training_spec() -> TrainingSpec:
    return TrainingSpec(max_steps=500, pop_size=2, evo_steps=100)


@pytest.fixture
def mutation_spec() -> MutationSpec:
    return MutationSpec(
        probabilities=MutationProbabilities(no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2),
        mutation_sd=0.05,
    )


@pytest.fixture
def tournament_spec() -> TournamentSelectionSpec:
    return TournamentSelectionSpec(tournament_size=3, elitism=True)


@pytest.fixture
def buffer_spec() -> ReplayBufferSpec:
    return ReplayBufferSpec(memory_size=5_000)


@pytest.fixture
def mock_population():
    """Return a list of mock agents that quack like EvolvableAlgorithm."""
    return [MagicMock(algo="PPO") for _ in range(2)]


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.submit_experiment.return_value = {
        "job_id": "test-123",
        "status": "PENDING",
    }
    return client


class TestBuildMutations:
    def test_none_returns_none(self):
        assert build_mutations_from_spec(None, "cpu") is None

    def test_from_spec(self, mutation_spec):
        result = build_mutations_from_spec(mutation_spec, "cpu")
        assert isinstance(result, Mutations)
        assert result.no_mut == 0.5
        assert result.parameters_mut == 0.3
        assert result.mutation_sd == 0.05


class TestBuildTournament:
    def test_none_returns_none(self, training_spec):
        assert build_tournament_from_spec(None, training_spec) is None

    def test_from_spec(self, tournament_spec, training_spec):
        result = build_tournament_from_spec(tournament_spec, training_spec)
        assert isinstance(result, TournamentSelection)
        assert result.tournament_size == 3
        assert result.elitism is True


class TestBuildReplayBuffer:
    def test_none_with_on_policy_returns_none(self, ppo_spec):
        assert build_replay_buffer_from_spec(ppo_spec, None, "cpu") is None

    def test_none_with_off_policy_creates_default(self):
        dqn_spec = DQNSpec()
        result = build_replay_buffer_from_spec(dqn_spec, None, "cpu")
        assert isinstance(result, ReplayBuffer)
        assert result.max_size == 100_000

    def test_from_spec_standard(self, buffer_spec):
        dqn_spec = DQNSpec()
        result = build_replay_buffer_from_spec(dqn_spec, buffer_spec, "cpu")
        assert isinstance(result, ReplayBuffer)
        assert result.max_size == 5_000

    def test_from_spec_n_step(self):
        spec = ReplayBufferSpec(memory_size=10_000, n_step_buffer=True)
        from agilerl.models import RainbowDQNSpec

        rainbow_spec = RainbowDQNSpec()
        result = build_replay_buffer_from_spec(rainbow_spec, spec, "cpu")
        assert isinstance(result, MultiStepReplayBuffer)
        assert result.max_size == 10_000


class TestAlgoNetSpecCls:
    def test_falls_back_to_networkspec_without_net_config_field(self):
        from agilerl.arena.models.algorithms import DPOSpec
        from agilerl.models.networks import NetworkSpec

        # LLM specs carry no ``net_config`` field, so there is no concrete
        # subclass to resolve and the base spec is used.
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = DPOSpec()
        assert trainer._algo_net_spec_cls() is NetworkSpec

    def test_resolves_concrete_spec_from_net_config_annotation(self):
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = DQNSpec()
        assert issubclass(trainer._algo_net_spec_cls(), QNetworkSpec)


class TestGetTrainingKwargs:
    @pytest.fixture
    def gym_env_spec(self):
        from agilerl.models.env import GymEnvSpec

        return GymEnvSpec(name="CartPole-v1", num_envs=1)

    def test_on_policy_has_no_memory(self, training_spec, ppo_spec, gym_env_spec):
        kwargs = select_strategy(ppo_spec).get_trainer_kwargs(
            ppo_spec, training=training_spec, env_spec=gym_env_spec, memory=None
        )
        assert "memory" not in kwargs
        assert kwargs["algo"] == "PPO"
        assert kwargs["eval_loop"] == training_spec.eval_loop

    def test_off_policy_has_memory_and_delay(self, training_spec, gym_env_spec):
        dqn_spec = DQNSpec()
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(dqn_spec).get_trainer_kwargs(
            dqn_spec, training=training_spec, env_spec=gym_env_spec, memory=buffer
        )
        assert kwargs["memory"] is buffer
        assert "learning_delay" in kwargs

    def test_env_name_forwarded(self, training_spec, ppo_spec, gym_env_spec):
        kwargs = select_strategy(ppo_spec).get_trainer_kwargs(
            ppo_spec, training=training_spec, env_spec=gym_env_spec
        )
        assert kwargs["env_name"] == "CartPole-v1"

    def test_off_policy_epsilon_forwarded(self, gym_env_spec):
        spec = DQNSpec()
        training = TrainingSpec(
            max_steps=100,
            evo_steps=50,
            pop_size=2,
            eps_start=0.5,
            eps_end=0.05,
            eps_decay=0.99,
        )
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec, training=training, env_spec=gym_env_spec, memory=buffer
        )
        assert kwargs["eps_start"] == 0.5
        assert kwargs["eps_end"] == 0.05
        assert kwargs["eps_decay"] == 0.99

    def test_off_policy_epsilon_omitted_when_none(self, training_spec, gym_env_spec):
        spec = DQNSpec()
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec, training=training_spec, env_spec=gym_env_spec, memory=buffer
        )
        assert "eps_start" not in kwargs
        assert "eps_end" not in kwargs
        assert "eps_decay" not in kwargs

    def test_off_policy_n_step_memory_forwarded(self, training_spec, gym_env_spec):
        spec = DQNSpec()
        buffer = ReplayBuffer(max_size=100, device="cpu")
        n_step_buf = MagicMock()
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec,
            training=training_spec,
            env_spec=gym_env_spec,
            memory=buffer,
            n_step_memory=n_step_buf,
        )
        assert kwargs["n_step_memory"] is n_step_buf

    def test_off_policy_no_n_step_memory_when_none(self, training_spec, gym_env_spec):
        spec = DQNSpec()
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec, training=training_spec, env_spec=gym_env_spec, memory=buffer
        )
        assert "n_step_memory" not in kwargs

    def test_bandit_episode_steps_forwarded(self, gym_env_spec):
        from agilerl.models import NeuralUCBSpec

        spec = NeuralUCBSpec()
        training = TrainingSpec(
            max_steps=100,
            evo_steps=50,
            pop_size=2,
            episode_steps=250,
        )
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec, training=training, env_spec=gym_env_spec, memory=buffer
        )
        assert kwargs["episode_steps"] == 250

    def test_multi_agent_sum_scores_forwarded(self, gym_env_spec):
        from agilerl.models import MADDPGSpec

        spec = MADDPGSpec()
        training = TrainingSpec(
            max_steps=100,
            evo_steps=50,
            pop_size=2,
            sum_scores=False,
        )
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec, training=training, env_spec=gym_env_spec, memory=buffer
        )
        assert kwargs["sum_scores"] is False

    def test_multi_agent_sum_scores_defaults_true(self, gym_env_spec):
        from agilerl.models import MADDPGSpec

        spec = MADDPGSpec()
        training = TrainingSpec(max_steps=100, evo_steps=50, pop_size=2)
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = select_strategy(spec).get_trainer_kwargs(
            spec, training=training, env_spec=gym_env_spec, memory=buffer
        )
        assert kwargs["sum_scores"] is True

    def test_on_policy_has_no_paradigm_specific_kwargs(
        self, training_spec, ppo_spec, gym_env_spec
    ):
        kwargs = select_strategy(ppo_spec).get_trainer_kwargs(
            ppo_spec, training=training_spec, env_spec=gym_env_spec
        )
        assert "memory" not in kwargs
        assert "learning_delay" not in kwargs
        assert "eps_start" not in kwargs
        assert "episode_steps" not in kwargs
        assert "sum_scores" not in kwargs
        assert "n_step_memory" not in kwargs


class TestLocalTrainerHpo:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_hpo_enables_default_mutation_and_tournament_specs(
        self, mock_create_pop, env, training_spec
    ):
        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="PPO",
            environment=env,
            training=training_spec,
            mutation=None,
            selection_strategy=None,
            hpo=True,
        )
        assert trainer.mutation_spec is not None
        assert trainer.selection_strategy_spec is not None

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_hpo_does_not_override_explicit_specs(
        self, mock_create_pop, env, training_spec, mutation_spec, tournament_spec
    ):
        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="DQN",
            environment=env,
            training=training_spec,
            mutation=mutation_spec,
            selection_strategy=tournament_spec,
            hpo=True,
        )
        assert trainer.mutation_spec is mutation_spec
        assert trainer.selection_strategy_spec is tournament_spec

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_hpo_does_not_override_a_multi_frequency_spec(
        self, mock_create_pop, env, mutation_spec
    ):
        """hpo=True must not inject tournament selection over a configured regime."""
        mock_create_pop.return_value = [MagicMock()]
        mf_spec = MultiFrequencySelectionSpec(n_subpopulations=2)
        trainer = LocalTrainer(
            algorithm="DQN",
            environment=env,
            training=TrainingSpec(pop_size=8),
            mutation=mutation_spec,
            selection_strategy=mf_spec,
            hpo=True,
        )
        assert trainer.selection_strategy_spec is mf_spec
        assert isinstance(trainer.selection_strategy, MultiFrequencySelection)


class TestLocalTrainerConstruction:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_string_algorithm(self, mock_create_pop, env, training_spec):
        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(algorithm="PPO", environment=env, training=training_spec)
        assert isinstance(trainer.algorithm_spec, PPOSpec)
        assert trainer.training_spec is training_spec

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_spec_algorithm(self, mock_create_pop, env, ppo_spec, training_spec):
        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm=ppo_spec,
            environment=env,
            training=training_spec,
        )
        assert trainer.algorithm_spec is ppo_spec
        assert trainer.training_spec.pop_size == 2

    def test_unknown_algorithm_raises(self, env, training_spec):
        with pytest.raises((ValueError, KeyError)):
            LocalTrainer(
                algorithm="UnknownAlgo", environment=env, training=training_spec
            )

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_all_optional_params(
        self, mock_create_pop, env, mutation_spec, tournament_spec, buffer_spec
    ):
        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="DQN",
            environment=env,
            training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=2),
            mutation=mutation_spec,
            selection_strategy=tournament_spec,
            replay_buffer=buffer_spec,
            device="cpu",
        )
        assert trainer.mutation_spec is mutation_spec
        assert trainer.selection_strategy_spec is tournament_spec
        assert trainer.replay_buffer_spec is buffer_spec


class TestLocalTrainerFromManifest:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_from_dict_manifest(self, mock_create_pop):
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        data = {
            "algorithm": {"name": "PPO", "learn_step": 128},
            "environment": {"name": "CartPole-v1", "num_envs": 1},
            "training": {"max_steps": 100, "evo_steps": 10, "pop_size": 2},
        }
        with patch.object(LocalTrainer, "_make_env", return_value=MagicMock()):
            trainer = LocalTrainer.from_manifest(data)

        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert trainer.env_spec.name == "CartPole-v1"
        assert trainer.training_spec.max_steps == 100

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_from_validated_manifest_instance(self, mock_create_pop):
        from agilerl.models.manifest import TrainingManifest

        mock_create_pop.return_value = [MagicMock()]
        data = {
            "algorithm": {"name": "DQN", "learn_step": 1},
            "environment": {"name": "CartPole-v1", "num_envs": 1},
            "training": {"max_steps": 50, "evo_steps": 5, "pop_size": 2},
        }
        manifest = TrainingManifest.get_validated(data, mode="python")
        with patch.object(LocalTrainer, "_make_env", return_value=MagicMock()):
            trainer = LocalTrainer.from_manifest(manifest)

        assert isinstance(trainer, LocalTrainer)
        assert trainer.algorithm_spec.name == "DQN"
        assert trainer.training_spec.max_steps == 50


class TestLocalTrainerTrain:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_train_delegates_to_fn(
        self,
        mock_create_pop,
        training_spec,
    ):
        from agilerl.models.env import GymEnvSpec

        env_spec = GymEnvSpec(name="CartPole-v1")
        mock_pop = [MagicMock()]
        mock_create_pop.return_value = mock_pop
        mock_train_fn = MagicMock(return_value=(mock_pop, [[1.0]]))
        mock_env = MagicMock()

        with (
            patch.object(
                SingleAgentOnPolicyStrategy,
                "get_training_loop",
                return_value=mock_train_fn,
            ),
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
        ):
            trainer = LocalTrainer(
                algorithm="PPO",
                environment=env_spec,
                training=training_spec,
            )
            result = trainer.train()

        mock_train_fn.assert_called_once()
        assert result == (mock_pop, [[1.0]])

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_train_warns_max_wall_seconds_ignored_for_non_multiturn(
        self,
        mock_create_pop,
    ):
        from agilerl.models.env import GymEnvSpec

        env_spec = GymEnvSpec(name="CartPole-v1")
        mock_pop = [MagicMock()]
        mock_create_pop.return_value = mock_pop
        mock_train_fn = MagicMock(return_value=(mock_pop, [[1.0]]))

        with (
            patch.object(
                SingleAgentOnPolicyStrategy,
                "get_training_loop",
                return_value=mock_train_fn,
            ),
            patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
        ):
            trainer = LocalTrainer(
                algorithm="PPO",
                environment=env_spec,
                training=TrainingSpec(
                    max_steps=500, evo_steps=100, pop_size=2, max_wall_seconds=60
                ),
            )
            with pytest.warns(UserWarning, match="max_wall_seconds"):
                trainer.train()

        assert "max_wall_seconds" not in mock_train_fn.call_args[1]


class TestArenaTrainerConstruction:
    def test_string_algorithm_and_env(self, mock_client, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        assert isinstance(trainer.algorithm_spec, PPOSpec)
        assert trainer._client is mock_client

    def test_spec_algorithm(self, mock_client, ppo_spec, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        assert trainer.algorithm_spec is ppo_spec

    def test_env_spec(self, mock_client, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1", version="v1", num_envs=8)
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        assert trainer.env_spec is env_spec

    def test_all_specs(self, mock_client, mutation_spec, tournament_spec, buffer_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment=env_spec,
            training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=3),
            mutation=mutation_spec,
            selection_strategy=tournament_spec,
            replay_buffer=buffer_spec,
            client=mock_client,
        )
        assert trainer.mutation_spec is mutation_spec
        assert trainer.selection_strategy_spec is tournament_spec
        assert trainer.replay_buffer_spec is buffer_spec
        assert trainer.training_spec.pop_size == 3

    @patch("agilerl.arena.client.ArenaClient")
    def test_auto_creates_client(self, mock_cls, training_spec):
        """If no client is passed, ArenaTrainer creates one automatically."""
        mock_cls.return_value = MagicMock()
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO", environment=env_spec, training=training_spec
        )
        assert trainer._client is not None


class TestArenaTrainerManifest:
    def test_minimal_manifest_from_string_algo_and_env(
        self, mock_client, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest["algorithm"]["name"] == "PPO"
        assert manifest["environment"]["name"] == "CartPole-v1"
        assert manifest.get("network") in (None, {})

    def test_string_based_manifest(self, mock_client, ppo_spec, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert isinstance(manifest, dict)
        assert manifest["algorithm"]["name"] == "PPO"
        assert manifest["environment"]["name"] == "CartPole-v1"
        assert manifest["training"]["max_steps"] == 500

    def test_spec_based_manifest(self, mock_client, ppo_spec):
        training = TrainingSpec(max_steps=50_000, evo_steps=500, pop_size=8)
        env_spec = ArenaEnvSpec(name="MountainCar-v0")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest["algorithm"]["name"] == "PPO"
        assert manifest["algorithm"]["learn_step"] == 128
        assert manifest["training"]["pop_size"] == 8
        assert manifest["environment"]["name"] == "MountainCar-v0"

    def test_env_spec_manifest(self, mock_client, training_spec):
        env_spec = ArenaEnvSpec(name="LunarLander-v3", version="v2", num_envs=4)
        dqn_spec = DQNSpec(
            net_config=QNetworkSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            )
        )
        trainer = ArenaTrainer(
            algorithm=dqn_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest["environment"]["name"] == "LunarLander-v3"
        assert manifest["environment"]["num_envs"] == 4

    def test_manifest_includes_mutation_and_tournament(
        self, mock_client, mutation_spec, tournament_spec, ppo_spec, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            mutation=mutation_spec,
            selection_strategy=tournament_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest["mutation"]["probabilities"]["no_mut"] == 0.5
        assert manifest["mutation"]["probabilities"]["params_mut"] == 0.3
        assert manifest["mutation"]["probabilities"]["rl_hp_mut"] == 0.2
        assert manifest["mutation"]["mutation_sd"] == 0.05
        assert manifest["tournament_selection"]["tournament_size"] == 3
        assert manifest["tournament_selection"]["elitism"] is True

    def test_manifest_includes_replay_buffer(
        self, mock_client, buffer_spec, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        dqn_spec = DQNSpec(
            net_config=QNetworkSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            )
        )
        trainer = ArenaTrainer(
            algorithm=dqn_spec,
            environment=env_spec,
            training=training_spec,
            replay_buffer=buffer_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest["replay_buffer"]["max_size"] == 5_000

    def test_manifest_network_from_algo_spec(
        self, mock_client, ppo_spec, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest["network"]["encoder_config"]["hidden_size"] == [64]
        assert manifest["network"]["head_config"]["hidden_size"] == [64]

    def test_manifest_payload_uses_spec_serializers(
        self, mock_client, ppo_spec, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest["algorithm"]["name"] == "PPO"
        assert manifest["training"]["max_steps"] == 500
        assert manifest["environment"]["name"] == "CartPole-v1"

    def test_invalid_env_type_raises(self, mock_client, training_spec):
        """Passing a non-spec environment to ArenaTrainer.to_manifest raises TypeError."""
        trainer = ArenaTrainer.__new__(ArenaTrainer)
        trainer.algorithm_spec = PPOSpec()
        trainer.env_spec = 42  # not an EnvironmentSpec or str
        trainer.training_spec = training_spec
        trainer.mutation_spec = None
        trainer.selection_strategy_spec = None
        trainer.replay_buffer_spec = None
        trainer._client = mock_client

        with pytest.raises((TypeError, Exception)):
            trainer.to_manifest()


class TestArenaTrainerTrain:
    def test_train_validates_with_arena_manifest(
        self, mock_client, ppo_spec, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        result = trainer.train()

        mock_client.submit_experiment.assert_called_once()
        payload = mock_client.submit_experiment.call_args.args[0]
        assert payload["environment"]["name"] == "CartPole-v1"
        assert payload["algorithm"]["name"] == "PPO"
        assert result["job_id"] == "test-123"

    def test_train_forwards_submit_kwargs(self, mock_client, ppo_spec, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        trainer.train(
            resource_id="arena-medium",
            num_nodes=2,
            project="GSM8K Tutorial",
            experiment_name="gsm8k-grpo",
            reward_file="reward.py",
            completion="#### 42",
        )

        kwargs = mock_client.submit_experiment.call_args.kwargs
        assert kwargs["resource_id"] == "arena-medium"
        assert kwargs["num_nodes"] == 2
        assert kwargs["project"] == "GSM8K Tutorial"
        assert kwargs["experiment_name"] == "gsm8k-grpo"
        assert kwargs["reward_file"] == "reward.py"
        assert kwargs["completion"] == "#### 42"

    def test_train_submits_manifest(self, mock_client, ppo_spec, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        result = trainer.train()

        mock_client.submit_experiment.assert_called_once()
        submitted_manifest = mock_client.submit_experiment.call_args[0][0]
        assert isinstance(submitted_manifest, dict)
        assert submitted_manifest["algorithm"]["name"] == "PPO"
        assert "mutation" in submitted_manifest
        assert "tournament_selection" in submitted_manifest
        assert result["job_id"] == "test-123"

    def test_train_submits_manifest_with_serializable_payload(
        self, mock_client, ppo_spec, training_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        trainer.train()
        submitted_manifest = mock_client.submit_experiment.call_args[0][0]
        assert submitted_manifest["algorithm"]["name"] == "PPO"
        assert submitted_manifest["training"]["max_steps"] == 500


class TestArenaTrainerDelegation:
    """Tests for ArenaTrainer methods that delegate to the underlying client."""

    def test_resume_from_checkpoint(self, mock_client, training_spec):
        mock_client.resume_experiment.return_value = {"status": "RESUMED"}
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        trainer.resume_from_checkpoint("job-42", max_steps=1000)
        mock_client.resume_experiment.assert_called_once_with("job-42", 1000)

    def test_list_experiments(self, mock_client, training_spec):
        mock_client.list_experiments.return_value = [{"name": "exp1"}]
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        result = trainer.list_experiments("my-project")
        mock_client.list_experiments.assert_called_once_with("my-project")
        assert result == [{"name": "exp1"}]

    def test_list_checkpoints(self, mock_client, training_spec):
        mock_client.list_checkpoints.return_value = [{"step": 100}]
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        result = trainer.list_checkpoints("job-42")
        mock_client.list_checkpoints.assert_called_once_with("job-42")
        assert result == [{"step": 100}]


class TestArenaTrainerFromManifest:
    """Tests for ArenaTrainer.from_manifest()."""

    def test_rejects_prevalidated_core_manifest(self):
        from agilerl.models.manifest import TrainingManifest

        data = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "CartPole-v1"},
            "training": {"max_steps": 10},
        }
        manifest = TrainingManifest.get_validated(data, mode="python")
        with pytest.raises(TypeError, match="expects a serialized manifest"):
            ArenaTrainer.from_manifest(manifest)

    def test_from_dict(self):
        from agilerl.arena.models import DQNSpec as ArenaDQNSpec

        data = {
            "algorithm": {"name": "DQN", "learn_step": 1},
            "environment": {"name": "CartPole-v1", "num_envs": 4},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
        }
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(data, client=mock_client)
        assert isinstance(trainer, ArenaTrainer)
        assert isinstance(trainer.algorithm_spec, ArenaDQNSpec)
        assert trainer.algorithm_spec.name == "DQN"
        assert trainer.env_spec.name == "CartPole-v1"
        assert trainer._client is mock_client

    @pytest.mark.parametrize("key", ["selection_strategy", "tournament_selection"])
    def test_from_manifest_carries_the_selection_section(self, mock_client, key):
        """Either spelling of the selection block reaches the trainer's spec."""
        data = {
            "algorithm": {"name": "DQN", "learn_step": 1},
            "environment": {"name": "CartPole-v1", "num_envs": 4},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
            key: {"tournament_size": 3, "elitism": False},
        }
        trainer = ArenaTrainer.from_manifest(data, client=mock_client)
        assert trainer.selection_strategy_spec.tournament_size == 3
        assert trainer.selection_strategy_spec.elitism is False

        # The payload uses the tournament_selection alias
        manifest = trainer.to_manifest()
        assert manifest["tournament_selection"]["tournament_size"] == 3
        assert "selection_strategy" not in manifest

    def test_from_manifest_train_round_trip(self, mock_client):
        """``from_manifest(...).train()`` validates and submits an Arena manifest."""
        from agilerl.arena.models import PPOSpec as ArenaPPOSpec

        data = {
            "algorithm": {"name": "PPO", "learn_step": 128},
            "environment": {"name": "CartPole-v1", "num_envs": 4},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
            "network": {
                "encoder_config": {"arch": "mlp", "hidden_size": [64]},
                "head_config": {"arch": "mlp", "hidden_size": [64]},
            },
        }
        trainer = ArenaTrainer.from_manifest(data, client=mock_client)
        assert isinstance(trainer.algorithm_spec, ArenaPPOSpec)

        result = trainer.train()

        mock_client.submit_experiment.assert_called_once()
        submitted_manifest = mock_client.submit_experiment.call_args[0][0]
        assert isinstance(submitted_manifest, dict)
        assert submitted_manifest["algorithm"]["name"] == "PPO"
        assert submitted_manifest["algorithm"]["learn_step"] == 128
        assert submitted_manifest["environment"]["name"] == "CartPole-v1"
        assert submitted_manifest["environment"]["num_envs"] == 4
        assert submitted_manifest["training"]["max_steps"] == 100
        assert submitted_manifest["training"]["pop_size"] == 2
        assert submitted_manifest["network"]["encoder_config"]["hidden_size"] == [64]
        assert submitted_manifest["network"]["head_config"]["hidden_size"] == [64]
        assert "mutation" in submitted_manifest
        assert "tournament_selection" in submitted_manifest
        assert result["job_id"] == "test-123"

    def test_from_yaml_manifest_train_round_trip(self, mock_client):
        """YAML ``from_manifest(...).train()`` round-trips through arena validation."""
        trainer = ArenaTrainer.from_manifest(
            str(Path(__file__).parents[2] / "configs/training/ppo/ppo.yaml"),
            client=mock_client,
        )
        result = trainer.train()

        mock_client.submit_experiment.assert_called_once()
        submitted_manifest = mock_client.submit_experiment.call_args[0][0]
        assert submitted_manifest["algorithm"]["name"] == "PPO"
        assert submitted_manifest["environment"]["name"] == "LunarLander-v3"
        assert submitted_manifest["training"]["pop_size"] == 4
        assert submitted_manifest["network"]["encoder_config"]["hidden_size"] == [64]
        assert result["job_id"] == "test-123"

    def test_core_specs_train_round_trip(self, mock_client, ppo_spec, training_spec):
        """Direct construction with core specs still submits a valid Arena manifest."""
        env_spec = ArenaEnvSpec(name="CartPole-v1", num_envs=8)
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )

        result = trainer.train()

        mock_client.submit_experiment.assert_called_once()
        submitted_manifest = mock_client.submit_experiment.call_args[0][0]
        assert submitted_manifest["algorithm"]["name"] == "PPO"
        assert submitted_manifest["algorithm"]["learn_step"] == 128
        assert submitted_manifest["environment"]["num_envs"] == 8
        assert submitted_manifest["training"]["max_steps"] == 500
        assert submitted_manifest["network"]["encoder_config"]["hidden_size"] == [64]
        assert result["job_id"] == "test-123"

    def test_from_yaml_file(self):
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(
            str(Path(__file__).parents[2] / "configs/training/ppo/ppo.yaml"),
            client=mock_client,
        )
        assert isinstance(trainer, ArenaTrainer)
        assert trainer.algorithm_spec.name == "PPO"

    def test_forwards_api_key(self):
        data = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "CartPole-v1"},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
        }
        with patch("agilerl.arena.auth.KeycloakOpenID"):
            trainer = ArenaTrainer.from_manifest(data, api_key="my-api-key")
        assert trainer._client._api_key == "my-api-key"


class TestAlgoRegistry:
    EXPECTED_ALGOS: ClassVar[set[str]] = {
        "PPO",
        "DQN",
        "DDPG",
        "TD3",
        "RainbowDQN",
        "CQN",
        "NeuralUCB",
        "NeuralTS",
        "IPPO",
        "MADDPG",
        "MATD3",
    }

    def test_all_algorithms_registered(self):
        available = set(MANIFEST_REGISTRY._entries)
        assert self.EXPECTED_ALGOS.issubset(available)

    def test_registry_entries_have_spec_cls(self):
        for name in self.EXPECTED_ALGOS:
            assert MANIFEST_REGISTRY.get(name) is not None


class VectorizedDummyEnv(DummyEnv):
    """DummyEnv that exposes vectorized-env *attributes* (not methods) for
    ``single_observation_space`` / ``single_action_space`` so that
    ``get_spaces_from_env`` can read them directly.
    """

    def __init__(self, *, continuous: bool = False) -> None:
        super().__init__()
        if continuous:
            self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space


class TestLocalTrainerCustomNetworks:
    """Verify that custom EvolvableMLP networks passed to algorithm specs
    are propagated to every individual in the LocalTrainer population.
    """

    OBS_DIM = 4
    DISCRETE_ACTIONS = 2
    CONTINUOUS_ACTIONS = 2
    HIDDEN: ClassVar[list[int]] = [64, 64]
    POP_SIZE = 3

    @staticmethod
    def _make_mlp(num_inputs: int, num_outputs: int):
        from agilerl.modules.mlp import EvolvableMLP

        return EvolvableMLP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            hidden_size=TestLocalTrainerCustomNetworks.HIDDEN,
        )

    @staticmethod
    def _training() -> TrainingSpec:
        return TrainingSpec(
            max_steps=100,
            evo_steps=50,
            pop_size=TestLocalTrainerCustomNetworks.POP_SIZE,
        )

    def _build_trainer(self, algo_spec, env, **networks):
        with patch.object(LocalTrainer, "_make_env", return_value=env):
            return LocalTrainer(
                algorithm=algo_spec,
                environment=env,
                training=self._training(),
                **networks,
            )

    # -- DQN (discrete, actor only) -----------------------------------------

    def test_dqn_custom_actor(self):
        from agilerl.modules.mlp import EvolvableMLP

        actor = self._make_mlp(self.OBS_DIM, self.DISCRETE_ACTIONS)
        trainer = self._build_trainer(
            DQNSpec(), VectorizedDummyEnv(), actor_network=actor
        )

        assert len(trainer.population) == self.POP_SIZE
        for agent in trainer.population:
            assert isinstance(agent.actor, EvolvableMLP)
            assert agent.actor.hidden_size == self.HIDDEN

    # -- PPO (discrete, actor + critic) -------------------------------------

    def test_ppo_custom_actor_critic(self):
        from agilerl.modules.mlp import EvolvableMLP
        from agilerl.networks import StochasticActor
        from agilerl.networks.value_networks import ValueNetwork

        actor = self._make_mlp(self.OBS_DIM, self.DISCRETE_ACTIONS)
        critic = self._make_mlp(self.OBS_DIM, 1)
        trainer = self._build_trainer(
            PPOSpec(), VectorizedDummyEnv(), actor_network=actor, critic_network=critic
        )

        assert len(trainer.population) == self.POP_SIZE
        for agent in trainer.population:
            # PPO adopts the custom MLP as the encoder of its stochastic
            # actor / value network and adds the distribution / value head.
            assert isinstance(agent.actor, StochasticActor)
            assert isinstance(agent.critic, ValueNetwork)
            assert isinstance(agent.actor.encoder, EvolvableMLP)
            assert isinstance(agent.critic.encoder, EvolvableMLP)
            assert agent.actor.encoder.hidden_size == self.HIDDEN
            assert agent.critic.encoder.hidden_size == self.HIDDEN

    # -- DDPG (continuous, actor + critic) ----------------------------------

    def test_ddpg_custom_actor_critic(self):
        from agilerl.modules.mlp import EvolvableMLP

        actor = self._make_mlp(self.OBS_DIM, self.CONTINUOUS_ACTIONS)
        critic = self._make_mlp(self.OBS_DIM + self.CONTINUOUS_ACTIONS, 1)
        trainer = self._build_trainer(
            DDPGSpec(),
            VectorizedDummyEnv(continuous=True),
            actor_network=actor,
            critic_network=critic,
        )

        assert len(trainer.population) == self.POP_SIZE
        for agent in trainer.population:
            assert isinstance(agent.actor, EvolvableMLP)
            assert isinstance(agent.critic, EvolvableMLP)
            assert agent.actor.hidden_size == self.HIDDEN
            assert agent.critic.hidden_size == self.HIDDEN

    # -- TD3 (continuous, actor + 2 critics) --------------------------------

    def test_td3_custom_actor_critics(self):
        from agilerl.modules.mlp import EvolvableMLP

        actor = self._make_mlp(self.OBS_DIM, self.CONTINUOUS_ACTIONS)
        critic_1 = self._make_mlp(self.OBS_DIM + self.CONTINUOUS_ACTIONS, 1)
        critic_2 = self._make_mlp(self.OBS_DIM + self.CONTINUOUS_ACTIONS, 1)
        trainer = self._build_trainer(
            TD3Spec(),
            VectorizedDummyEnv(continuous=True),
            actor_network=actor,
            critic_networks=[critic_1, critic_2],
        )

        assert len(trainer.population) == self.POP_SIZE
        for agent in trainer.population:
            assert isinstance(agent.actor, EvolvableMLP)
            assert isinstance(agent.critic_1, EvolvableMLP)
            assert isinstance(agent.critic_2, EvolvableMLP)
            assert agent.actor.hidden_size == self.HIDDEN
            assert agent.critic_1.hidden_size == self.HIDDEN
            assert agent.critic_2.hidden_size == self.HIDDEN

    # -- Verify deep copies (individuals don't share the same object) -------

    def test_custom_networks_are_deep_copied(self):
        actor = self._make_mlp(self.OBS_DIM, self.DISCRETE_ACTIONS)
        trainer = self._build_trainer(
            DQNSpec(), VectorizedDummyEnv(), actor_network=actor
        )

        actors = [agent.actor for agent in trainer.population]
        for i, a in enumerate(actors):
            assert a is not actor, "Population actor should be a deep copy"
            for j, b in enumerate(actors):
                if i != j:
                    assert a is not b, "Each individual should have its own copy"


_DPOSpec, _GRPOSpec = DPOSpec, GRPOSpec

_LLM_COMMON_KWARGS = {
    "update_epochs": 1,
    "lora_config": {"lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
    "max_model_len": 512,
    "use_separate_reference_adapter": False,
    "pretrained_model_name_or_path": "gpt2",
    "calc_position_embeddings": False,
}


@pytest.fixture
def dpo_spec():
    return _DPOSpec(**_LLM_COMMON_KWARGS)


@pytest.fixture
def grpo_spec():
    return _GRPOSpec(group_size=4, **_LLM_COMMON_KWARGS)


class TestLLMSpecConstruction:
    """Verify that DPOSpec / GRPOSpec can be constructed and expose the
    expected class-level attributes.
    """

    def test_dpo_spec_fields(self, dpo_spec):
        assert dpo_spec.name == "DPO"
        assert dpo_spec.env_type == "dataset"
        assert dpo_spec.objective == "preference"
        assert isinstance(dpo_spec, LLMAlgorithmSpec)
        assert dpo_spec.pretrained_model_name_or_path == "gpt2"

    def test_grpo_spec_fields(self, grpo_spec):
        assert grpo_spec.name == "GRPO"
        assert grpo_spec.env_type == "rollout"
        assert isinstance(grpo_spec, LLMAlgorithmSpec)
        assert grpo_spec.group_size == 4

    def test_dpo_training_fn(self, dpo_spec):
        from agilerl.training.llm import train_llm_dataset

        assert (
            select_strategy(dpo_spec).get_training_loop(dpo_spec) is train_llm_dataset
        )

    def test_grpo_training_fn(self, grpo_spec):
        from agilerl.training.llm import train_llm_rollout

        assert (
            select_strategy(grpo_spec).get_training_loop(grpo_spec) is train_llm_rollout
        )

    def test_dpo_model_dump_contains_expected_fields(self, dpo_spec):
        dumped = dpo_spec.model_dump(mode="python", exclude={"hp_config"})
        assert dumped["pretrained_model_name_or_path"] == "gpt2"
        assert dumped["update_epochs"] == 1
        assert dumped["beta"] == pytest.approx(0.1)  # DPO's constructor default

    def test_grpo_model_dump_contains_group_size(self, grpo_spec):
        dumped = grpo_spec.model_dump(mode="python", exclude={"hp_config"})
        assert dumped["group_size"] == 4
        assert dumped["pretrained_model_name_or_path"] == "gpt2"


class TestLLMGetTrainingKwargs:
    """Verify the LLM-specific early-return path in
    the strategy's ``training_kwargs``.
    """

    def test_llm_kwargs_defaults(self, dpo_spec):
        env_spec = MagicMock(max_reward=None)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        kwargs = select_strategy(dpo_spec).get_trainer_kwargs(
            dpo_spec, training=training, env_spec=env_spec
        )
        assert kwargs == {"evaluation_interval": 10}
        assert "max_reward" not in kwargs
        assert "num_epochs" not in kwargs

    def test_llm_kwargs_include_max_reward(self, grpo_spec):
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="preference",
            dataset="dummy",
            max_reward=5.0,
        )
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        kwargs = select_strategy(grpo_spec).get_trainer_kwargs(
            grpo_spec, training=training, env_spec=env_spec
        )
        assert kwargs["max_reward"] == 5.0

    def test_llm_kwargs_include_checkpoint_steps(self, dpo_spec):
        env_spec = MagicMock(max_reward=None)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        training.checkpoint_steps = 50
        kwargs = select_strategy(dpo_spec).get_trainer_kwargs(
            dpo_spec, training=training, env_spec=env_spec
        )
        assert kwargs["checkpoint_steps"] == 50

    def test_llm_kwargs_never_include_memory(self, dpo_spec):
        env_spec = MagicMock(max_reward=None)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        buf = MagicMock()
        kwargs = select_strategy(dpo_spec).get_trainer_kwargs(
            dpo_spec, training=training, env_spec=env_spec, memory=buf
        )
        assert "memory" not in kwargs


class TestLLMBuildAlgorithm:
    """Verify that ``LLMAlgorithmSpec.build_algorithm`` calls the algo class
    constructor with the right arguments.
    """

    def test_dpo_build_algorithm(self, dpo_spec):
        mock_algo = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None

        with (
            patch.object(
                select_builder(dpo_spec), "algo_class", return_value=mock_algo
            ),
            patch(
                "agilerl.utils.llm_utils.load_pad_token_configs",
                return_value=(None, None),
            ),
        ):
            select_builder(dpo_spec).build(dpo_spec, tokenizer=mock_tokenizer, index=0)

        mock_algo.assert_called_once()
        call_kwargs = mock_algo.call_args[1]
        assert call_kwargs["model_name"] == "gpt2"
        assert call_kwargs["pad_token_id"] == 50256
        assert call_kwargs["pad_token"] == "<|endoftext|>"
        assert call_kwargs["index"] == 0

    def test_grpo_build_algorithm(self, grpo_spec):
        mock_algo = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None

        with (
            patch.object(
                select_builder(grpo_spec), "algo_class", return_value=mock_algo
            ),
            patch(
                "agilerl.utils.llm_utils.load_pad_token_configs",
                return_value=(None, None),
            ),
        ):
            select_builder(grpo_spec).build(
                grpo_spec, tokenizer=mock_tokenizer, index=1
            )

        mock_algo.assert_called_once()
        call_kwargs = mock_algo.call_args[1]
        assert call_kwargs["index"] == 1
        assert call_kwargs["model_name"] == "gpt2"

    def test_build_algorithm_with_accelerator(self, dpo_spec):
        mock_algo = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.unk_token_id = None
        mock_accel = MagicMock()
        mock_accel.num_processes = 2

        with (
            patch.object(
                select_builder(dpo_spec), "algo_class", return_value=mock_algo
            ),
            patch(
                "agilerl.utils.llm_utils.load_pad_token_configs",
                return_value=(None, None),
            ),
        ):
            select_builder(dpo_spec).build(
                dpo_spec, tokenizer=mock_tokenizer, index=0, accelerator=mock_accel
            )

        call_kwargs = mock_algo.call_args[1]
        assert call_kwargs["accelerator"] is mock_accel
        # Unset on the spec: the algorithm derives its own micro batch size,
        # matching direct construction.
        assert "micro_batch_size_per_gpu" not in call_kwargs


class TestLLMLocalTrainer:
    """End-to-end tests for LocalTrainer with LLM algorithms.

    Since ``peft`` and ``transformers`` are not installed, we mock the
    heavy-weight components (tokenizer loading, environment creation,
    population building) and verify the wiring is correct.
    """

    POP_SIZE = 2

    def _training(self):
        return TrainingSpec(max_steps=100, evo_steps=10, pop_size=self.POP_SIZE)

    # -- Construction -------------------------------------------------------

    def test_construction_with_spec(self, dpo_spec):
        mock_pop = [MagicMock() for _ in range(self.POP_SIZE)]
        mock_env = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer.pad_token = "<|endoftext|>"

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ) as mock_create_pop,
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            from agilerl.models.env import LLMEnvSpec, LLMEnvType

            env_spec = MagicMock(spec=LLMEnvSpec)
            env_spec.env_type = LLMEnvType.DATASET
            trainer = LocalTrainer(
                algorithm=dpo_spec,
                environment=env_spec,
                training=self._training(),
            )

        assert trainer.algorithm_spec is dpo_spec
        assert trainer.tokenizer is mock_tokenizer
        mock_auto_tok.from_pretrained.assert_called_once_with("gpt2")
        mock_create_pop.assert_called_once()
        _, create_kwargs = mock_create_pop.call_args
        assert create_kwargs["tokenizer"] is mock_tokenizer
        assert create_kwargs["population_size"] == self.POP_SIZE

    def test_construction_with_grpo(self, grpo_spec):
        mock_pop = [MagicMock() for _ in range(self.POP_SIZE)]
        mock_env = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer.pad_token = "<|endoftext|>"

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            env_spec = MagicMock()
            trainer = LocalTrainer(
                algorithm=grpo_spec,
                environment=env_spec,
                training=self._training(),
            )

        assert trainer.algorithm_spec is grpo_spec
        assert trainer.tokenizer is mock_tokenizer

    # -- _make_env dispatches to LLMEnvSpec.make_dataset_env() --------------

    def test_make_env_dispatches_to_llm_env_spec(self, dpo_spec):
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        mock_env = MagicMock()
        mock_llm_env_spec = MagicMock(spec=LLMEnvSpec)
        mock_llm_env_spec.env_type = LLMEnvType.DATASET
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ) as mock_create_accel,
            patch(
                "agilerl.training.trainer.make_llm_env",
                return_value=mock_env,
            ) as mock_make_llm_env,
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=dpo_spec,
                environment=mock_llm_env_spec,
                training=self._training(),
            )

        mock_accel = mock_create_accel.return_value
        mock_make_llm_env.assert_called_once_with(
            mock_llm_env_spec,
            mock_tokenizer,
            data_batch_size_per_gpu=dpo_spec.batch_size,
            max_context_length=dpo_spec.max_model_len,
            seed=dpo_spec.seed,
            rank=mock_accel.process_index,
            world_size=mock_accel.num_processes,
        )
        assert trainer.env is mock_env

    def test_rollout_factory_gets_the_run_seed(self, grpo_spec):
        """A rollout env builds no env here; the factory carries the run seed."""
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        mock_llm_env_spec = MagicMock(spec=LLMEnvSpec)
        mock_llm_env_spec.env_type = LLMEnvType.ROLLOUT

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
            patch(
                "agilerl.training.trainer.make_rollout_env_factory",
                return_value=(MagicMock(), 5),
            ) as mock_factory,
        ):
            # The pad resolver reads real token ids off the tokenizer.
            mock_tokenizer = MagicMock()
            mock_tokenizer.eos_token_id = 0
            mock_tokenizer.pad_token_id = 1
            mock_tokenizer.unk_token_id = None
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=grpo_spec,
                environment=mock_llm_env_spec,
                training=self._training(),
            )

        factory_kwargs = mock_factory.call_args.kwargs
        assert factory_kwargs["seed"] == grpo_spec.seed
        assert factory_kwargs["max_model_len"] == grpo_spec.max_model_len
        # The env itself is built per-trajectory by the factory, not here.
        assert trainer.env is None

    # -- No replay buffer for LLM algorithms --------------------------------

    def test_no_replay_buffer_for_llm(self, dpo_spec):
        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = MagicMock(
                eos_token_id=0, eos_token="<eos>"
            )
            trainer = LocalTrainer(
                algorithm=dpo_spec,
                environment=MagicMock(),
                training=self._training(),
            )

        assert trainer.memory is None

    # -- train() delegates to correct training function ---------------------

    def test_train_delegates_to_llm_fn(self, dpo_spec):
        mock_pop = [MagicMock()]
        mock_env = MagicMock()
        mock_train_fn = MagicMock(return_value=(mock_pop, [[1.0]]))
        mock_tokenizer = MagicMock(eos_token_id=0, eos_token="<eos>")
        env_spec = MagicMock(max_reward=None)

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ),
            patch.object(
                LLMDatasetStrategy, "get_training_loop", return_value=mock_train_fn
            ),
            patch.object(LocalTrainer, "to_manifest", return_value={}),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=dpo_spec,
                environment=env_spec,
                training=self._training(),
            )
            result = trainer.train()

        mock_train_fn.assert_called_once()
        call_kwargs = mock_train_fn.call_args[1]
        assert call_kwargs["pop"] is mock_pop
        assert call_kwargs["env"] is mock_env
        assert call_kwargs["max_steps"] == 100
        assert call_kwargs["evo_steps"] == 10
        assert result == (mock_pop, [[1.0]])
        assert "tournament" not in call_kwargs
        assert call_kwargs["selection_strategy"] is trainer.selection_strategy

    # -- Missing LLM dependencies raises ImportError -----------------------

    def test_missing_llm_deps_raises(self, dpo_spec):
        with (
            patch("agilerl.training.trainer.AutoTokenizer", None),
            patch("agilerl.training.trainer.create_llm_accelerator", None),
        ):
            with pytest.raises(ImportError, match="LLM dependencies"):
                LocalTrainer(
                    algorithm=dpo_spec,
                    environment=MagicMock(),
                    training=self._training(),
                )

    def test_make_tokenizer_missing_auto_tokenizer_raises(self, dpo_spec):
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = dpo_spec
        with patch("agilerl.training.trainer.AutoTokenizer", None):
            with pytest.raises(ImportError, match="LLM dependencies"):
                trainer._make_tokenizer()

    def test_make_tokenizer_falls_back_to_the_default_chat_template(self, dpo_spec):
        from agilerl.utils.chat_template import DEFAULT_CHAT_TEMPLATE

        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = dpo_spec
        mock_tokenizer = MagicMock(
            chat_template=None,
            eos_token_id=0,
            eos_token="<eos>",
            pad_token_id=None,
            unk_token_id=None,
        )
        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.load_pad_token_configs",
                return_value=(None, None),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            tokenizer = trainer._make_tokenizer()

        assert tokenizer.chat_template == DEFAULT_CHAT_TEMPLATE

    def test_make_tokenizer_keeps_an_existing_chat_template(self, dpo_spec):
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = dpo_spec
        mock_tokenizer = MagicMock(
            chat_template="{{ 'preset' }}",
            eos_token_id=0,
            eos_token="<eos>",
            pad_token_id=None,
            unk_token_id=None,
        )
        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.load_pad_token_configs",
                return_value=(None, None),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            tokenizer = trainer._make_tokenizer()

        assert tokenizer.chat_template == "{{ 'preset' }}"


class TestTrainerBaseNotImplemented:
    def test_resolve_env_spec_raises(self):
        manifest = MagicMock()
        with pytest.raises(NotImplementedError, match="_resolve_env_spec"):
            Trainer._resolve_env_spec(manifest)

    def test_train_raises(self):
        with pytest.raises(NotImplementedError, match="train method"):
            Trainer.train(MagicMock())


class TestArenaTrainerResolveEnvSpec:
    def test_llm_environment_points_at_submit_experiment(self):
        # LLM envs have a name (the dataset alias); the real reason they are
        # rejected is that ArenaTrainer does not run them.
        manifest = MagicMock()
        manifest.environment = ArenaLLMEnvSpec(
            env_type="rollout",
            dataset="openai/gsm8k",
            reward_file_path="reward.py",
            prompt_template={"user_0": "{question}"},
        )
        with pytest.raises(ValueError, match="submit_experiment"):
            ArenaTrainer._resolve_env_spec(manifest)

    def test_gym_environment_uses_name(self):
        manifest = MagicMock()
        manifest.environment = ArenaEnvSpec(name="CartPole-v1", num_envs=4)
        result = ArenaTrainer._resolve_env_spec(manifest)
        assert result.name == "CartPole-v1"
        assert result.num_envs == 4

    def test_bandit_environment_uses_name(self):
        manifest = MagicMock()
        manifest.environment = ArenaBanditEnvSpec(name="MyBandit")
        result = ArenaTrainer._resolve_env_spec(manifest)
        assert result.name == "MyBandit"

    def test_unsupported_environment_type_raises(self):
        manifest = MagicMock()
        manifest.environment = object()
        with pytest.raises(TypeError, match="not a supported Arena environment spec"):
            ArenaTrainer._resolve_env_spec(manifest)


class TestLLMAlgoRegistry:
    """DPO/GRPO are in the contract's registry whether or not the LLM extras are
    installed: the specs carry no runtime dependencies.
    """

    def test_dpo_registered(self):
        assert MANIFEST_REGISTRY.get("DPO") is _DPOSpec

    def test_grpo_registered(self):
        assert MANIFEST_REGISTRY.get("GRPO") is _GRPOSpec

    def test_llm_specs_are_llm_algorithm_specs(self):
        assert issubclass(_DPOSpec, LLMAlgorithmSpec)
        assert issubclass(_GRPOSpec, LLMAlgorithmSpec)


# ============================================================================
# Integration tests — real env, real population, real train()
# ============================================================================


class TestLocalTrainerIntegration:
    """End-to-end integration tests that exercise the full LocalTrainer
    pipeline with **no mocks** on the critical path:

        env spec → _make_env → create_population_from_spec → train()

    Each test uses the smallest feasible configuration so it finishes in
    seconds, not minutes.  The goal is to prove that the wiring between
    specs, environments, populations, and training loops is correct for
    every training paradigm — something the unit tests (which mock most
    of these) cannot guarantee.
    """

    POP_SIZE = 2
    MAX_STEPS = 64
    EVO_STEPS = 32

    @staticmethod
    def _training(
        pop_size: int = 2,
        max_steps: int = 64,
        evo_steps: int = 32,
        **kwargs,
    ) -> TrainingSpec:
        return TrainingSpec(
            pop_size=pop_size,
            max_steps=max_steps,
            evo_steps=evo_steps,
            **kwargs,
        )

    @staticmethod
    def _mock_llm_tokenizer() -> MagicMock:
        """Tokenizer stand-in so integration tests do not hit Hugging Face."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 0
        return mock_tokenizer

    @staticmethod
    def _mock_llm_accelerator() -> MagicMock:
        """Accelerator stand-in with real DP shard ints for DatasetEnv."""
        mock_accel = MagicMock()
        mock_accel.process_index = 0
        mock_accel.num_processes = 1
        return mock_accel

    @staticmethod
    @contextlib.contextmanager
    def _llm_trainer_patches(mock_pop: list[MagicMock]):
        """Patches population, accelerator, and tokenizer loading for LLM tests."""
        mock_tokenizer = TestLocalTrainerIntegration._mock_llm_tokenizer()
        with (
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=TestLocalTrainerIntegration._mock_llm_accelerator(),
            ),
            patch(
                "agilerl.training.trainer.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            yield

    # -- On-policy: PPO + CartPole-v1 ---------------------------------------

    def test_ppo_cartpole(self):
        """PPO (on-policy) trains for one evolution step on CartPole."""
        from agilerl.models.env import GymEnvSpec

        trainer = LocalTrainer(
            algorithm=PPOSpec(learn_step=32),
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            training=self._training(),
        )
        try:
            pop, fitness = trainer.train()
            assert len(pop) == self.POP_SIZE
            assert len(fitness) >= 1
        finally:
            trainer.env.close()

    # -- Off-policy: DQN + CartPole-v1 --------------------------------------

    def test_dqn_cartpole(self):
        """DQN (off-policy) trains for one evolution step on CartPole."""
        from agilerl.models.env import GymEnvSpec
        from agilerl.models.hpo import MutationProbabilities

        trainer = LocalTrainer(
            algorithm=DQNSpec(learn_step=1),
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            training=self._training(),
            replay_buffer=ReplayBufferSpec(max_size=1_000),
            mutation=MutationSpec(
                probabilities=MutationProbabilities(
                    no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2
                ),
                mutation_sd=0.05,
            ),
        )
        try:
            pop, fitness = trainer.train()
            assert len(pop) == self.POP_SIZE
            assert len(fitness) >= 1
        finally:
            trainer.env.close()

    # -- Off-policy continuous: DDPG + Pendulum  ------------------

    def test_ddpg_continuous(self):
        """DDPG (off-policy, continuous) on Pendulum-v1."""
        from agilerl.models.env import GymEnvSpec

        trainer = LocalTrainer(
            algorithm=DDPGSpec(learn_step=1),
            environment=GymEnvSpec(name="Pendulum-v1", num_envs=1),
            training=self._training(),
            replay_buffer=ReplayBufferSpec(max_size=1_000),
        )
        try:
            pop, fitness = trainer.train()
            assert len(pop) == self.POP_SIZE
            assert len(fitness) >= 1
        finally:
            trainer.env.close()

    # -- Bandit: NeuralUCB --------------------------------------------------

    def test_neural_ucb_bandit(self, tmp_path):
        """NeuralUCB (bandit) on synthetic data."""
        import numpy as np
        import pandas as pd

        from agilerl.models import NeuralUCBSpec
        from agilerl.models.env import BanditEnvSpec

        rng = np.random.default_rng(42)
        features = pd.DataFrame(rng.standard_normal((100, 4)).astype(np.float32))
        targets = pd.DataFrame(rng.integers(0, 2, size=(100, 1)).astype(np.float32))
        feat_path = tmp_path / "features.csv"
        tgt_path = tmp_path / "targets.csv"
        features.to_csv(feat_path, index=False)
        targets.to_csv(tgt_path, index=False)

        trainer = LocalTrainer(
            algorithm=NeuralUCBSpec(learn_step=2),
            environment=BanditEnvSpec(
                features=str(feat_path),
                targets=str(tgt_path),
            ),
            training=self._training(max_steps=100, evo_steps=50, eval_steps=50),
            replay_buffer=ReplayBufferSpec(max_size=500),
        )
        pop, fitness = trainer.train()
        assert len(pop) == self.POP_SIZE
        assert len(fitness) >= 1

    # -- Multi-agent off-policy: MADDPG + simple_speaker_listener -----------

    def test_maddpg_speaker_listener(self):
        """MADDPG (multi-agent off-policy) on simple_speaker_listener."""
        from agilerl.models import MADDPGSpec
        from agilerl.models.env import GymEnvSpec
        from agilerl.models.hpo import MutationProbabilities

        trainer = LocalTrainer(
            algorithm=MADDPGSpec(learn_step=2),
            environment=GymEnvSpec(
                name="mpe2.simple_speaker_listener_v4",
                num_envs=2,
            ),
            training=self._training(max_steps=32, evo_steps=16),
            replay_buffer=ReplayBufferSpec(max_size=1_000),
            mutation=MutationSpec(
                probabilities=MutationProbabilities(
                    no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2
                ),
                mutation_sd=0.05,
            ),
        )
        try:
            pop, fitness = trainer.train()
            assert len(pop) == self.POP_SIZE
            assert len(fitness) >= 1
        finally:
            trainer.env.close()

    # -- Custom networks: DQN with custom MLP → actually trains -------------

    def test_custom_network_dqn_trains(self):
        """DQN with a user-supplied EvolvableMLP trains without error."""
        from agilerl.models.env import GymEnvSpec
        from agilerl.modules.mlp import EvolvableMLP

        class TestMLP(EvolvableMLP): ...

        actor = TestMLP(num_inputs=4, num_outputs=2, hidden_size=[32, 32])

        trainer = LocalTrainer(
            algorithm=DQNSpec(learn_step=1),
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            actor_network=actor,
            training=self._training(),
            replay_buffer=ReplayBufferSpec(max_size=1_000),
        )
        try:
            pop, fitness = trainer.train()
            assert len(pop) == self.POP_SIZE
            assert len(fitness) >= 1
            for agent in pop:
                assert isinstance(agent.actor, TestMLP)
                assert agent.actor.hidden_size == [32, 32]
        finally:
            trainer.env.close()

    # -- Offline: CQL + CartPole + dummy HDF5 --------------------------------

    def test_cql_offline_cartpole(self, tmp_path):
        """CQL (offline) on CartPole with a dummy HDF5 dataset."""
        import h5py
        import numpy as np

        from agilerl.models import CQNSpec
        from agilerl.models.env import OfflineEnvSpec

        n_samples = 50
        obs_dim = 4
        rng = np.random.default_rng(0)

        h5_path = tmp_path / "cartpole_offline.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "observations",
                data=rng.standard_normal((n_samples, obs_dim)).astype(np.float32),
            )
            f.create_dataset(
                "actions",
                data=rng.integers(0, 2, size=(n_samples, 1)).astype(np.float32),
            )
            f.create_dataset(
                "rewards",
                data=rng.standard_normal(n_samples).astype(np.float32),
            )
            f.create_dataset(
                "terminals",
                data=rng.integers(0, 2, size=n_samples).astype(np.float32),
            )

        trainer = LocalTrainer(
            algorithm=CQNSpec(learn_step=1),
            environment=OfflineEnvSpec(
                name="CartPole-v1",
                num_envs=1,
                dataset_path=str(h5_path),
            ),
            training=self._training(max_steps=64, evo_steps=32, eval_steps=32),
            replay_buffer=ReplayBufferSpec(max_size=500),
        )
        try:
            pop, fitness = trainer.train()
            assert len(pop) == self.POP_SIZE
            assert len(fitness) >= 1
        finally:
            trainer.env.close()

    # -- LLM: GRPO (reasoning) via LocalTrainer ------------------------------

    @pytest.mark.llm
    def test_grpo_rollout_train(self, tmp_path):
        """GRPO rollout: real LLMEnvSpec factory with temp dataset files.

        Tests the full LocalTrainer wiring: spec → env construction → kwargs
        assembly → dispatch to train_llm_rollout.  The finetune function
        itself is patched since running it requires CUDA agents.
        """
        try:
            import peft  # noqa: F401 -- the test needs the LLM extras

            from agilerl.models.env import LLMEnvSpec, LLMEnvType
        except ImportError:
            pytest.skip("LLM dependencies not installed")

        reward_file = tmp_path / "reward.py"
        reward_file.write_text(
            "from agilerl.llm_envs.rubrics import reward_fn_to_rubric\n"
            "def simple_reward(completion, answer, prompt, **kwargs):\n"
            "    return 1.0\n"
            "RUBRIC = reward_fn_to_rubric(simple_reward)\n"
        )

        import pandas as pd

        df = pd.DataFrame(
            {
                "question": [f"What is {i} + {i}?" for i in range(20)],
                "answer": [str(i * 2) for i in range(20)],
            }
        )
        dataset_path = tmp_path / "reasoning.parquet"
        df.to_parquet(dataset_path)

        from agilerl.arena.models.algorithms import GRPOSpec

        lora_config = {
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }
        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            dataset=str(dataset_path),
            rubric_file_path=str(reward_file),
            rubric_name="RUBRIC",
            prompt_template={"user_0": "Solve: {question}"},
        )
        algo_spec = GRPOSpec(
            pretrained_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            group_size=2,
            update_epochs=1,
            lora_config=lora_config,
            max_model_len=128,
            # Must stay under max_model_len or the rollout prompt budget is 0.
            max_output_tokens=32,
        )

        mock_pop = [MagicMock()]

        with self._llm_trainer_patches(mock_pop):
            trainer = LocalTrainer(
                algorithm=algo_spec,
                environment=env_spec,
                training=self._training(pop_size=1, max_steps=8, evo_steps=4),
            )

        from agilerl.llm_envs import RolloutHarness

        # A rollout env is built per trajectory, so the trainer holds a factory.
        assert trainer.env is None
        assert callable(trainer.env_factory)
        assert trainer.tokenizer is not None
        assert trainer.tokenizer.pad_token is not None

        rollout_env = trainer.env_factory()
        try:
            assert isinstance(rollout_env, RolloutHarness)
            assert rollout_env.max_turns == 1
        finally:
            rollout_env.close()

        mock_fn = MagicMock(return_value=None)
        with patch.object(trainer, "train_fn", mock_fn):
            trainer.train()
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["pop"] is mock_pop
            assert call_kwargs["env_factory"] is trainer.env_factory
            assert call_kwargs["max_turns"] == 1
            assert call_kwargs["max_steps"] == 8
            assert call_kwargs["evo_steps"] == 4
            assert "evaluation_interval" in call_kwargs

    # -- LLM: DPO (preference) via LocalTrainer ------------------------------

    @pytest.mark.llm
    def test_dpo_preference_train(self, tmp_path):
        """DPO preference: real LLMEnvSpec.make_dataset_env() with temp dataset files.

        Tests the full LocalTrainer wiring: spec → env construction → kwargs
        assembly → dispatch to train_llm_dataset.  The finetune function
        itself is patched since running it requires CUDA agents.
        """
        try:
            import peft  # noqa: F401 -- the test needs the LLM extras

            from agilerl.models.env import LLMEnvSpec, LLMEnvType
        except ImportError:
            pytest.skip("LLM dependencies not installed")

        import pandas as pd

        df = pd.DataFrame(
            {
                "prompt": [f"Explain concept {i}" for i in range(20)],
                "chosen": [f"Good answer for {i}" for i in range(20)],
                "rejected": [f"Bad answer for {i}" for i in range(20)],
            }
        )
        dataset_path = tmp_path / "preference.parquet"
        df.to_parquet(dataset_path)

        from agilerl.arena.models.algorithms import DPOSpec

        lora_config = {
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
        }
        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="preference",
            dataset=str(dataset_path),
        )
        algo_spec = DPOSpec(
            pretrained_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            update_epochs=1,
            lora_config=lora_config,
            max_model_len=128,
        )

        mock_pop = [MagicMock()]

        with self._llm_trainer_patches(mock_pop):
            trainer = LocalTrainer(
                algorithm=algo_spec,
                environment=env_spec,
                training=self._training(pop_size=1, max_steps=8, evo_steps=4),
            )

        from agilerl.llm_envs import DatasetEnv

        assert isinstance(trainer.env, DatasetEnv)
        assert trainer.tokenizer is not None

        mock_fn = MagicMock(return_value=None)
        with patch.object(trainer, "train_fn", mock_fn):
            trainer.train()
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["pop"] is mock_pop
            assert isinstance(call_kwargs["env"], DatasetEnv)
            assert call_kwargs["max_steps"] == 8
            assert "evaluation_interval" in call_kwargs


class TestStringEnvironmentResolution:
    """Verify that passing a plain string as the ``environment`` parameter
    produces the correct env spec and that the constructed environment
    corresponds to the requested gym / PettingZoo id.
    """

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_gym_env_from_string(self, mock_create_pop, training_spec):
        """A string environment for a single-agent algo resolves to GymEnvSpec
        and the constructed env matches the requested id.
        """
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="DQN",
            environment="CartPole-v1",
            training=training_spec,
        )
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert trainer.env_spec.name == "CartPole-v1"
        assert trainer.env is not None
        assert hasattr(trainer.env, "single_observation_space")

    @patch("agilerl.training.trainer.LocalTrainer._make_env", return_value=MagicMock())
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_pz_env_from_string(self, mock_create_pop, mock_make_env, training_spec):
        """A string environment for a multi-agent algo resolves to GymEnvSpec."""
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        # Bare "simple_spread_v3" only imports once another test has loaded mpe2.
        trainer = LocalTrainer(
            algorithm="MADDPG",
            environment="mpe2.simple_spread_v3",
            training=training_spec,
        )
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert trainer.env_spec.name == "mpe2.simple_spread_v3"

    def test_offline_string_raises(self, training_spec):
        """Offline algorithms must be given a full OfflineEnvSpec."""
        with pytest.raises(ValueError, match="Only Gym and PettingZoo"):
            LocalTrainer(
                algorithm="CQN",
                environment="CartPole-v1",
                training=training_spec,
            )

    def test_bandit_string_raises(self, training_spec):
        """Bandit algorithms must be given a full BanditEnvSpec."""
        with pytest.raises(ValueError, match="Only Gym and PettingZoo"):
            LocalTrainer(
                algorithm="NeuralUCB",
                environment="BanditEnv",
                training=training_spec,
            )

    def test_arena_trainer_string_env(self, mock_client, training_spec):
        """ArenaTrainer converts a plain string to ArenaEnvSpec."""
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment="CartPole-v1",
            client=mock_client,
            training=training_spec,
        )
        assert isinstance(trainer.env_spec, ArenaEnvSpec)
        assert trainer.env_spec.name == "CartPole-v1"

    def test_llm_string_raises(self, training_spec):
        """LLM algorithms must be given a full LLMEnvSpec."""
        spec = _DPOSpec(**_LLM_COMMON_KWARGS)
        with pytest.raises(ValueError, match="Only Gym and PettingZoo"):
            LocalTrainer(
                algorithm=spec,
                environment="SomeEnv",
                training=training_spec,
            )

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_on_policy_string_env(self, mock_create_pop, training_spec):
        """On-policy algo with a string env resolves to GymEnvSpec."""
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="PPO",
            environment="CartPole-v1",
            training=training_spec,
        )
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert trainer.env_spec.name == "CartPole-v1"


class TestGetValidatedManifest:
    def test_from_yaml_file(self):
        from agilerl.models.manifest import TrainingManifest

        manifest = TrainingManifest.get_validated(
            str(Path(__file__).parents[2] / "configs/training/ppo/ppo.yaml"),
            mode="python",
        )
        assert manifest.algorithm.name == "PPO"
        assert manifest.training.max_steps == 6_000_000

    def test_from_dict(self):
        from agilerl.models.manifest import TrainingManifest

        data = {
            "algorithm": {"name": "DQN", "learn_step": 1},
            "environment": {"name": "CartPole-v1", "num_envs": 1},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
        }
        manifest = TrainingManifest.get_validated(data, mode="python")
        assert manifest.algorithm.name == "DQN"
        assert manifest.training.max_steps == 100
        assert manifest.training.pop_size == 2


class TestLocalTrainerToManifest:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_to_manifest_returns_dict(self, mock_create_pop, training_spec):
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        env_spec = GymEnvSpec(name="CartPole-v1")
        mock_env = MagicMock()
        with patch.object(LocalTrainer, "_make_env", return_value=mock_env):
            trainer = LocalTrainer(
                algorithm="PPO", environment=env_spec, training=training_spec
            )
        manifest = trainer.to_manifest()
        assert isinstance(manifest, dict)
        assert manifest["algorithm"]["name"] == "PPO"
        assert manifest["training"]["max_steps"] == 500

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_to_manifest_includes_network_when_present(self, mock_create_pop):
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        spec = DQNSpec(
            net_config=QNetworkSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            )
        )
        env_spec = GymEnvSpec(name="CartPole-v1")
        mock_env = MagicMock()
        with patch.object(LocalTrainer, "_make_env", return_value=mock_env):
            trainer = LocalTrainer(
                algorithm=spec,
                environment=env_spec,
                training=TrainingSpec(max_steps=100, evo_steps=50, pop_size=2),
            )
        manifest = trainer.to_manifest()
        assert "network" in manifest
        assert manifest["network"]["encoder_config"]["hidden_size"] == [64]

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_to_manifest_excludes_none_sections(self, mock_create_pop, training_spec):
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        env_spec = GymEnvSpec(name="CartPole-v1")
        mock_env = MagicMock()
        with patch.object(LocalTrainer, "_make_env", return_value=mock_env):
            trainer = LocalTrainer(
                algorithm="PPO", environment=env_spec, training=training_spec
            )
        manifest = trainer.to_manifest()
        assert manifest.get("replay_buffer") is None or "replay_buffer" not in manifest


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestLocalTrainerToManifestLLM:
    """``LocalTrainer.to_manifest()`` for LLM algorithms."""

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_to_manifest_llm_network_json_shape_and_round_trip(self, mock_create_pop):
        import json

        from agilerl.arena.models.algorithms import DPOSpec
        from agilerl.models.env import LLMEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<eos>"
        lora = {
            "lora_r": 4,
            "lora_alpha": 9,
            "target_modules": ["q_proj"],
            "task_type": "CAUSAL_LM",
            "lora_dropout": 0.11,
        }
        spec = DPOSpec(
            batch_size=4,
            pretrained_model_name_or_path="test-model",
            max_model_len=333,
            lora_config=lora,
        )
        env_spec = LLMEnvSpec(
            env_type="dataset",
            objective="preference",
            dataset="data.parquet",
            columns={"prompt": "q", "chosen": "ok"},
        )
        mock_env = MagicMock()
        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            trainer = LocalTrainer(
                algorithm=spec,
                environment=env_spec,
                training=TrainingSpec(max_steps=100, evo_steps=10, pop_size=2),
            )
        manifest = trainer.to_manifest()
        json.dumps(manifest)

        net = manifest["network"]
        assert net["pretrained_model_name_or_path"] == "test-model"
        assert net["max_context_length"] == 333
        lc = net["lora_config"]
        assert lc["lora_r"] == 4
        assert lc["lora_alpha"] == 9
        assert lc["lora_dropout"] == pytest.approx(0.11)
        assert "peft_type" not in lc

        from agilerl.models.manifest import TrainingManifest

        round_trip = TrainingManifest.get_validated(manifest, mode="python")
        assert round_trip.algorithm.pretrained_model_name_or_path == "test-model"
        assert round_trip.algorithm.max_model_len == 333
        assert round_trip.algorithm.lora_config.lora_r == 4
        assert round_trip.algorithm.lora_config.lora_alpha == 9


class TestLocalTrainerTrainKwargs:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_train_kwargs_forwarded(self, mock_create_pop, training_spec):
        from agilerl.models.env import GymEnvSpec

        env_spec = GymEnvSpec(name="CartPole-v1")
        mock_pop = [MagicMock()]
        mock_create_pop.return_value = mock_pop
        mock_train_fn = MagicMock(return_value=(mock_pop, [[1.0]]))
        mock_env = MagicMock()

        with (
            patch.object(
                SingleAgentOnPolicyStrategy,
                "get_training_loop",
                return_value=mock_train_fn,
            ),
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
        ):
            trainer = LocalTrainer(
                algorithm="PPO",
                environment=env_spec,
                training=training_spec,
            )
            trainer.train(
                verbose=False,
                wb=True,
                tensorboard=True,
                tensorboard_log_dir="/tmp/tb",
                save_elite=True,
                elite_path="/tmp/elite",
                wandb_api_key="test-key",
                wandb_kwargs={"project": "test"},
                checkpoint_steps=50,
                checkpoint_path="/tmp/ckpt",
                overwrite_checkpoints=True,
            )

        call_kwargs = mock_train_fn.call_args[1]
        assert call_kwargs["verbose"] is False
        assert call_kwargs["wb"] is True
        assert call_kwargs["tensorboard"] is True
        assert call_kwargs["tensorboard_log_dir"] == "/tmp/tb"
        assert call_kwargs["save_elite"] is True
        assert call_kwargs["elite_path"] == "/tmp/elite"
        assert call_kwargs["wandb_api_key"] == "test-key"
        assert call_kwargs["wandb_kwargs"] == {"project": "test"}
        assert trainer.training_spec.checkpoint_steps == 50
        assert trainer.training_spec.checkpoint_path == "/tmp/ckpt"
        assert trainer.training_spec.overwrite_checkpoints is True


class TestGRPOSpecRollout:
    """Verify GRPOSpec returns the correct training function."""

    def test_single_turn_training_fn(self):
        from agilerl.training.llm import train_llm_rollout

        fn = select_strategy(_GRPOSpec.model_construct()).get_training_loop(
            _GRPOSpec.model_construct()
        )
        assert fn is train_llm_rollout

    def test_rollout_training_fn(self):
        from agilerl.training.llm import train_llm_rollout

        fn = select_strategy(_GRPOSpec.model_construct()).get_training_loop(
            _GRPOSpec.model_construct()
        )
        assert fn is train_llm_rollout


class TestLocalTrainerRollout:
    """Verify LocalTrainer wires rollout LLM training correctly."""

    POP_SIZE = 1

    def _training(self):
        return TrainingSpec(max_steps=100, evo_steps=10, pop_size=self.POP_SIZE)

    def test_construction_rollout(self, grpo_spec):
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        mock_pop = [MagicMock()]
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_tokenizer.pad_token_id = 50256

        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="gem:make",
            env_config={"env_id": "game:GuessTheNumber-v0-easy"},
            max_turns=5,
        )

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ),
            patch(
                "agilerl.training.trainer.make_rollout_env_factory",
                return_value=(MagicMock(), 5),
            ) as mock_factory_method,
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=grpo_spec,
                environment=env_spec,
                training=self._training(),
            )

        assert trainer._rollout is True
        assert trainer.env is None
        assert trainer.env_factory is not None
        mock_factory_method.assert_called_once()
        from agilerl.training.llm import train_llm_rollout

        assert trainer.train_fn is train_llm_rollout

    def test_train_delegates_rollout_kwargs(self, grpo_spec):
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        mock_pop = [MagicMock()]
        mock_tokenizer = MagicMock(eos_token_id=0, eos_token="<eos>", pad_token_id=0)
        mock_train_fn = MagicMock(return_value=mock_pop)
        mock_env_factory = MagicMock()

        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
            max_turns=8,
            max_reward=1.0,
        )

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ),
            patch(
                "agilerl.training.trainer.make_rollout_env_factory",
                return_value=(mock_env_factory, 8),
            ),
            patch.object(
                LLMRolloutStrategy, "get_training_loop", return_value=mock_train_fn
            ),
            patch.object(LocalTrainer, "to_manifest", return_value={}),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=grpo_spec,
                environment=env_spec,
                training=self._training(),
            )
            trainer.train()

        mock_train_fn.assert_called_once()
        call_kwargs = mock_train_fn.call_args[1]
        assert call_kwargs["env_factory"] is mock_env_factory
        assert call_kwargs["max_turns"] == 8
        assert call_kwargs["pop"] is mock_pop
        assert call_kwargs["max_steps"] == 100
        assert "env" not in call_kwargs
        # The wandb run name reads the flat env_name key from init_hp.
        assert call_kwargs["init_hp"]["env_name"] == "my_mod:make"
        assert "max_wall_seconds" not in call_kwargs

    def test_train_forwards_max_wall_seconds(self, grpo_spec):
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        mock_pop = [MagicMock()]
        mock_tokenizer = MagicMock(eos_token_id=0, eos_token="<eos>", pad_token_id=0)
        mock_train_fn = MagicMock(return_value=mock_pop)

        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.ROLLOUT,
            entrypoint="my_mod:make",
            max_turns=8,
            max_reward=1.0,
        )

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ),
            patch(
                "agilerl.training.trainer.make_rollout_env_factory",
                return_value=(MagicMock(), 8),
            ),
            patch.object(
                LLMRolloutStrategy, "get_training_loop", return_value=mock_train_fn
            ),
            patch.object(LocalTrainer, "to_manifest", return_value={}),
            patch(
                "agilerl.training.trainer.create_llm_accelerator",
                return_value=MagicMock(),
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=grpo_spec,
                environment=env_spec,
                training=TrainingSpec(
                    max_steps=100, evo_steps=10, pop_size=1, max_wall_seconds=1800
                ),
            )
            trainer.train()

        assert mock_train_fn.call_args[1]["max_wall_seconds"] == 1800


class TestImportGuardReload:
    """Module-level fallbacks via importlib reload."""

    def test_llm_fallbacks_without_deps(self):
        """AutoTokenizer and create_llm_accelerator set to None."""
        import agilerl.training.trainer as mod

        with (
            patch("agilerl.HAS_LLM_DEPENDENCIES", False),
            patch("agilerl.training.trainer.HAS_LLM_DEPENDENCIES", False),
        ):
            importlib.reload(mod)
            assert mod.AutoTokenizer is None
            assert mod.create_llm_accelerator is None

        importlib.reload(mod)


class TestMakeEnvBranches:
    """Unit tests for LocalTrainer._make_env individual branches."""

    def test_llm_rollout_returns_none(self):
        """ROLLOUT builds no env here; the factory builds one per trajectory."""
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.env_spec = MagicMock(spec=LLMEnvSpec)
        trainer.env_spec.env_type = LLMEnvType.ROLLOUT
        trainer.algorithm_spec = MagicMock()
        trainer.tokenizer = MagicMock()
        trainer.accelerator = MagicMock()

        assert trainer._make_env() is None

    def test_llm_dataset_calls_make_llm_env(self):
        """A dataset LLMEnvSpec passes algo fields into make_llm_env."""
        from agilerl.models.env import LLMEnvSpec, LLMEnvType

        mock_env = MagicMock()
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.env_spec = MagicMock(spec=LLMEnvSpec)
        trainer.env_spec.env_type = LLMEnvType.DATASET
        trainer.algorithm_spec = MagicMock(spec=LLMAlgorithmSpec)
        trainer.algorithm_spec.max_model_len = 1024
        trainer.algorithm_spec.seed = 42
        trainer.algorithm_spec.batch_size = 8
        trainer.tokenizer = MagicMock()
        trainer.accelerator = MagicMock()
        trainer.accelerator.process_index = 0
        trainer.accelerator.num_processes = 1

        with patch(
            "agilerl.training.trainer.make_llm_env",
            return_value=mock_env,
        ) as mock_make:
            result = trainer._make_env()

        assert result is mock_env
        mock_make.assert_called_once_with(
            trainer.env_spec,
            trainer.tokenizer,
            data_batch_size_per_gpu=8,
            max_context_length=1024,
            seed=42,
            rank=0,
            world_size=1,
        )

    def test_unknown_env_spec_raises(self):
        trainer = LocalTrainer.__new__(LocalTrainer)

        class OtherSpec(BaseModel):
            pass

        trainer.env_spec = OtherSpec()
        trainer.algorithm_spec = MagicMock()
        trainer.tokenizer = None
        trainer.accelerator = None

        with pytest.raises(TypeError, match="Unsupported environment spec"):
            trainer._make_env()

    def test_live_env_is_used_as_is(self):
        dummy = DummyEnv()
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.env_spec = dummy
        trainer.algorithm_spec = MagicMock()
        trainer.tokenizer = None
        trainer.accelerator = None

        assert trainer._make_env() is dummy


class TestLocalTrainerResolveEnvSpecBranches:
    """LocalTrainer._resolve_env_spec returns the manifest environment as-is."""

    def test_returns_manifest_environment(self):
        spec = GymEnvSpec(name="TestEnv-v0", num_envs=4)
        manifest = MagicMock()
        manifest.environment = spec
        assert LocalTrainer._resolve_env_spec(manifest) is spec


def test_from_manifest_infers_multiinput_when_arch_absent(tmp_path):
    """A Dict-obs env with NO arch builds an EvolvableMultiInput encoder."""
    import yaml

    from agilerl.modules.multi_input import EvolvableMultiInput
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "PPO", "learn_step": 64},
        "environment": {
            "name": "dict-obs-env",
            "num_envs": 2,
            "entrypoint": "tests.test_train._dummy_envs:DictObsEnv",
        },
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))

    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    encoder = trainer.population[0].actor.encoder
    assert isinstance(encoder, EvolvableMultiInput)


def test_from_manifest_wrong_arch_is_overridden_when_omitted(tmp_path):
    """Vector-obs env with no arch builds an EvolvableMLP."""
    import yaml

    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "PPO", "learn_step": 64},
        "environment": {"name": "CartPole-v1", "num_envs": 2},
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    assert isinstance(trainer.population[0].actor.encoder, EvolvableMLP)


@pytest.mark.parametrize(
    ("arch", "environment", "algorithm_extra", "network_extra", "assertions"),
    [
        pytest.param(
            "mlp",
            {"name": "CartPole-v1", "num_envs": 2},
            {},
            {},
            # MlpSpec defaults (dataclass MlpNetConfig would be 500 / 3)
            {"max_mlp_nodes": 256, "max_hidden_layers": 6},
            id="mlp",
        ),
        pytest.param(
            "multiinput",
            {
                "name": "dict-obs-env",
                "num_envs": 2,
                "entrypoint": "tests.test_train._dummy_envs:DictObsEnv",
            },
            {},
            {},
            # MultiInputSpec default (dataclass MultiInputNetConfig would be 16)
            {"latent_dim": 32},
            id="multiinput",
        ),
        pytest.param(
            "cnn",
            {
                "name": "image-obs-env",
                "num_envs": 2,
                "entrypoint": "tests.test_train._dummy_envs:ImageObsEnv",
            },
            {},
            {},
            # CnnSpec default (dataclass CnnNetConfig would be 16)
            {"min_channel_size": 8},
            id="cnn",
        ),
        pytest.param(
            "lstm",
            {"name": "CartPole-v1", "num_envs": 2},
            {"recurrent": True},
            {},
            # LstmSpec defaults (dataclass LstmNetConfig would be 500 / 4)
            {"max_hidden_state_size": 256, "max_layers": 6},
            id="lstm",
        ),
        pytest.param(
            "simba",
            {"name": "CartPole-v1", "num_envs": 2},
            {},
            {"simba": True},
            # SimbaSpec default (dataclass SimBaNetConfig would be 500)
            {"max_mlp_nodes": 256},
            id="simba",
        ),
    ],
)
def test_deferred_encoder_uses_spec_defaults_not_dataclass(
    tmp_path, arch, environment, algorithm_extra, network_extra, assertions
):
    """A no-arch manifest must resolve encoder HP bounds from the pydantic
    ``*Spec`` defaults, not the ``modules/configs`` dataclass defaults.

    Parametrized across every inferable arch (mlp, multiinput, cnn, lstm,
    simba) so the invariant is guarded regardless of which encoder the
    deferred path infers from the observation space / algorithm flags.
    """
    import yaml

    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "PPO", "learn_step": 64, **algorithm_extra},
        "environment": environment,
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {
            "latent_dim": 32,
            "head_config": {"hidden_size": [32]},
            **network_extra,
        },
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")

    enc = trainer.algorithm_spec.net_config.encoder_config
    assert enc.arch == arch
    for field, expected in assertions.items():
        assert getattr(enc, field) == expected


def test_resolve_deferred_net_config_noop_when_arch_present(tmp_path):
    """A raw net_config that already declares an ``arch`` is left untouched
    (nothing to infer), so ``_resolve_deferred_net_config`` returns early.
    """
    import yaml

    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "PPO", "learn_step": 64},
        "environment": {"name": "CartPole-v1", "num_envs": 2},
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {
            "latent_dim": 32,
            "arch": "mlp",
            "encoder_config": {"hidden_size": [32]},
            "head_config": {"hidden_size": [32]},
        },
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")

    raw = {"arch": "mlp", "encoder_config": {"arch": "mlp", "hidden_size": [64]}}
    trainer.algorithm_spec.net_config = raw
    trainer._resolve_deferred_net_config()
    assert trainer.algorithm_spec.net_config is raw


def test_from_manifest_multi_agent_heterogeneous_per_agent_encoders(tmp_path):
    """Heterogeneous multi-agent env with no arch: per-agent encoders inferred."""
    import yaml

    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.modules.multi_input import EvolvableMultiInput
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "IPPO", "learn_step": 64},
        "environment": {
            "name": "hetero-env",
            "num_envs": 2,
            "entrypoint": "tests.test_train._dummy_envs:HeteroParallelEnv",
        },
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))

    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    agent = trainer.population[0]
    encoders = {aid: net.encoder for aid, net in agent.actors.items()}
    assert isinstance(encoders["dict_agent"], EvolvableMultiInput)
    assert isinstance(encoders["vec_agent"], EvolvableMLP)


def test_from_manifest_multi_agent_homogeneous(tmp_path):
    """Homogeneous multi-agent env with no arch: shared MLP encoder per group."""
    import yaml

    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.training.trainer import LocalTrainer

    manifest = {
        "algorithm": {"name": "IPPO", "learn_step": 64},
        "environment": {
            "name": "mpe2.simple_spread_v3",
            "num_envs": 2,
        },
        "training": {"max_steps": 200, "evo_steps": 100, "pop_size": 1},
        "network": {"latent_dim": 32, "head_config": {"hidden_size": [32]}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(manifest))
    trainer = LocalTrainer.from_manifest(manifest=path, device="cpu")
    agent = trainer.population[0]
    # simple_spread agents (agent_0/1/2) share a prefix -> one grouped policy.
    assert len(agent.actors) >= 1
    for net in agent.actors.values():
        assert isinstance(net.encoder, EvolvableMLP)


def _make_multi_frequency_ppo_trainer(
    training: TrainingSpec | None = None,
    accelerator: object | None = None,
    selection_strategy: MultiFrequencySelectionSpec | None = None,
) -> LocalTrainer:
    """Build an eight-slot PPO LocalTrainer driven by multi-frequency selection."""
    selection_strategy = selection_strategy or MultiFrequencySelectionSpec(
        n_subpopulations=2,
        evolution_frequency_ratios=[1, 2],
        n_winners=1,
        n_survivors=1,
        n_open_for_migration=1,
        n_losers=1,
    )
    mutation = MutationSpec(
        probabilities=MutationProbabilities(no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2),
        rl_hp_selection={
            "lr": RLHyperparameter(min=1e-5, max=1e-2),
            # learn_step resizes the rollout buffer
            "learn_step": RLHyperparameter(min=64, max=2048),
        },
    )
    ppo = PPOSpec(
        learn_step=128,
        net_config=StochasticActorSpec(
            encoder_config=MlpSpec(hidden_size=[16]),
            head_config=MlpSpec(hidden_size=[16]),
        ),
    )
    with patch.object(LocalTrainer, "_make_env", return_value=VectorizedDummyEnv()):
        return LocalTrainer(
            algorithm=ppo,
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
            training=training or TrainingSpec(max_steps=200, evo_steps=50, pop_size=8),
            mutation=mutation,
            selection_strategy=selection_strategy,
            accelerator=accelerator,
        )


class TestLocalTrainerMultiFrequency:
    """LocalTrainer builds the operator, tags subpopulations and rejects bad specs."""

    def test_builds_multi_frequency_and_tags_subpopulations(self):
        trainer = _make_multi_frequency_ppo_trainer()

        assert isinstance(trainer.selection_strategy, MultiFrequencySelection)
        subpops = sorted(a.subpopulation_id for a in trainer.population)
        assert subpops == [0, 0, 0, 0, 1, 1, 1, 1]
        assert len({a.index for a in trainer.population}) == 8

    def test_accepts_multi_frequency_with_accelerator(self):
        accelerator = MagicMock()
        fake_population = [MagicMock() for _ in range(8)]
        with patch(
            "agilerl.training.trainer.create_population_from_spec",
            return_value=fake_population,
        ):
            trainer = _make_multi_frequency_ppo_trainer(accelerator=accelerator)

        assert trainer.accelerator is accelerator
        assert isinstance(trainer.selection_strategy, MultiFrequencySelection)

    def test_requires_explicit_pop_size_without_manifest(self):
        # pop_size is mandatory under MF-PBT
        with pytest.raises(ValueError, match="pop_size is required"):
            _make_multi_frequency_ppo_trainer(
                training=TrainingSpec(max_steps=200, evo_steps=50)
            )

    def test_rejects_pop_size_below_six_without_manifest(self):
        with pytest.raises(ValueError, match="population_size must be >= 6"):
            _make_multi_frequency_ppo_trainer(
                training=TrainingSpec(max_steps=200, evo_steps=50, pop_size=4)
            )

    def test_resolves_bracket_defaults_onto_the_spec(self):
        trainer = _make_multi_frequency_ppo_trainer(
            selection_strategy=MultiFrequencySelectionSpec(n_subpopulations=2)
        )

        spec = trainer.selection_strategy_spec
        assert (spec.n_winners, spec.n_survivors, spec.n_open_for_migration) == (
            1,
            0,
            1,
        )
        assert spec.n_losers == 2  # 4 - 1 - 0 - 1
        assert trainer.to_manifest()["selection_strategy"]["n_losers"] == 2


class TestLocalTrainerMultiFrequencyManifest:
    """Multi-frequency selection round-trips through the ``selection_strategy`` union."""

    def test_trainer_to_manifest_uses_the_unified_block(self):
        trainer = _make_multi_frequency_ppo_trainer()

        manifest = trainer.to_manifest()

        assert manifest["training"]["pop_size"] == 8
        assert manifest["selection_strategy"]["strategy"] == "multi_frequency"
        assert "multi_frequency_selection" not in manifest

    def test_manifest_round_trip_rebuilds_via_unified_block(self):
        trainer = _make_multi_frequency_ppo_trainer()
        manifest = trainer.to_manifest()

        with patch.object(LocalTrainer, "_make_env", return_value=VectorizedDummyEnv()):
            rebuilt = LocalTrainer.from_manifest(manifest)

        assert isinstance(rebuilt.selection_strategy, MultiFrequencySelection)
        assert rebuilt.training_spec.pop_size == 8


class TestResumeAndWarmStartAreExclusive:
    """``resume_from_checkpoint`` and ``load_weights_from`` mean different things.

    One continues a run (weights, optimizer state, and the hyperparameters they
    belong to); the other warm-starts a new one from weights alone. Passing both
    is a contradiction the docstrings already claimed was rejected.
    """

    def test_trainer_rejects_both(self, training_spec):
        with pytest.raises(ValueError, match="mutually"):
            LocalTrainer(
                algorithm="DQN",
                environment="CartPole-v1",
                training=training_spec,
                resume_from_checkpoint="a",
                load_weights_from="b",
            )

    def test_create_population_from_spec_rejects_both(self):
        with pytest.raises(ValueError, match="mutually"):
            create_population_from_spec(
                population_size=1,
                algo_spec=DQNSpec(),
                env=VectorizedDummyEnv(),
                mutation_spec=None,
                replay_buffer_spec=None,
                resume_from_checkpoint="a",
                load_weights_from="b",
            )

    def test_multi_agent_population_receives_load_weights_from(self, tmp_path):
        """The multi-agent branch forwards ``load_weights_from`` to the algorithm builder."""
        from agilerl.arena.models.algorithms import MADDPGSpec

        spec = MADDPGSpec()

        mock_build = MagicMock(return_value=MagicMock())
        with (
            patch(
                "agilerl.utils.trainer_utils.get_spaces_from_env",
                return_value=({"a": Discrete(2)}, {"a": Discrete(2)}),
            ),
            patch(
                "agilerl.builders.MultiAgentBuilder.build",
                mock_build,
            ),
        ):
            create_population_from_spec(
                population_size=1,
                algo_spec=spec,
                env=MagicMock(),
                mutation_spec=None,
                replay_buffer_spec=None,
                load_weights_from="warm-start",
            )

        builder_call = mock_build.call_args
        assert builder_call.kwargs["load_weights_from"] == "warm-start"


class TestCreatePopulationFromSpecMultiFrequency:
    """A resumed population keeps its slot layout, so subpopulation tags stay valid."""

    def test_resumed_population_keeps_slot_indices_and_evolves(self, tmp_path):
        env = VectorizedDummyEnv()
        checkpoint = tmp_path / "resume.pt"
        DQN(
            observation_space=env.observation_space,
            action_space=env.action_space,
            index=3,
        ).save_checkpoint(str(checkpoint))

        mf_spec = MultiFrequencySelectionSpec(n_subpopulations=2)
        population = create_population_from_spec(
            population_size=6,
            algo_spec=DQNSpec(),
            env=env,
            mutation_spec=None,
            replay_buffer_spec=None,
            resume_from_checkpoint=str(checkpoint),
            selection_strategy_spec=mf_spec,
        )

        assert [agent.index for agent in population] == [0, 1, 2, 3, 4, 5]
        assert [agent.subpopulation_id for agent in population] == [0, 0, 0, 1, 1, 1]

        for i, agent in enumerate(population):
            agent.fitness = [float(6 - i)]
        strategy = MultiFrequencySelection(
            population_size=6, n_subpopulations=2, evolution_frequency_ratios=[1, 2]
        )
        _elite, evolved, _indices = strategy.select(population)

        assert len({agent.index for agent in evolved}) == len(evolved)


class TestMultiFrequencyRealPopulationEvolution:
    """The operator evolves a real, trainer-built PPO population cycle after cycle."""

    def test_evolves_real_ppo_population_across_cycles(self):
        trainer = _make_multi_frequency_ppo_trainer()
        agents = list(trainer.population)

        for _cycle in range(3):
            rank_population_by_subpopulation(agents)
            # Subpopulation 0 evolves every cycle (delta 1), so its weakest member is
            # always cloned over
            doomed = weakest_agent_index(agents, subpop=0)

            agents = run_selection_and_mutation(
                trainer.selection_strategy,
                population=agents,
                mutation=trainer.mutations,
                env_name="CartPole-v1",
                algo="PPO",
            )

            assert doomed not in {a.index for a in agents}
            assert len(agents) == 8
            assert Counter(a.subpopulation_id for a in agents) == Counter({0: 4, 1: 4})
            assert len({a.index for a in agents}) == 8  # indices stay unique
            assert all(isinstance(a, EvolvableAlgorithm) for a in agents)

    def test_migration_weights_resizes_rollout_buffer(self):
        # A fast-to-slow migrant clones the external agent's networks but takes the
        # studied subpop elite's mutable HPs, including learn_step. PPO sizes its
        # rollout buffer from learn_step via a mutation hook, so the migrant's buffer
        # must be rebuilt to match the elite's learn_step, or rollout collection
        # overflows it.
        trainer = _make_multi_frequency_ppo_trainer()
        agents = list(trainer.population)
        external, elite = agents[4], agents[0]

        external.learn_step = 256
        external.mutation_hook()  # buffer sized to 256
        elite.learn_step = 512
        elite.mutation_hook()  # buffer sized to 512

        trainer.selection_strategy._sync_index(agents)
        migrant = trainer.selection_strategy._migrate_weights(external, elite, subpop=1)

        assert migrant.learn_step == 512  # took the elite's learn_step
        # Buffer capacity must follow the new learn_step (ceil(learn_step / num_envs))
        assert migrant.rollout_buffer.capacity == -(512 // -migrant.num_envs)


class TestLocalTrainerSelectionStrategyDeprecation:
    """The superseded ``tournament`` keyword still routes, loudly and unambiguously."""

    @staticmethod
    def _build(env, **kwargs):
        with patch(
            "agilerl.training.trainer.create_population_from_spec",
            return_value=[MagicMock()],
        ):
            return LocalTrainer(
                algorithm="PPO",
                environment=env,
                training=TrainingSpec(max_steps=500, pop_size=2, evo_steps=100),
                **kwargs,
            )

    def test_deprecated_tournament_kwarg_warns_and_routes(self, env, tournament_spec):
        with pytest.warns(DeprecationWarning, match="'tournament' argument"):
            trainer = self._build(env, tournament=tournament_spec)

        assert trainer.selection_strategy_spec is tournament_spec
        assert isinstance(trainer.selection_strategy, TournamentSelection)

    def test_equal_specs_from_both_spellings_route(self, env):
        """Equality, not identity: two equal specs are not a conflict."""
        current = TournamentSelectionSpec(tournament_size=3)
        deprecated = TournamentSelectionSpec(tournament_size=3)
        assert current is not deprecated

        with pytest.warns(DeprecationWarning, match="'tournament' argument"):
            trainer = self._build(
                env, selection_strategy=current, tournament=deprecated
            )

        assert trainer.selection_strategy_spec is current

    def test_conflicting_specs_raise(self, env):
        with (
            pytest.warns(DeprecationWarning, match="'tournament' argument"),
            pytest.raises(ValueError, match="conflicting selection strategies"),
        ):
            self._build(
                env,
                selection_strategy=TournamentSelectionSpec(tournament_size=2),
                tournament=TournamentSelectionSpec(tournament_size=3),
            )

    def test_conflicting_regimes_raise(self, env):
        """Two different regimes can never be configured together."""
        with (
            pytest.warns(DeprecationWarning, match="'tournament' argument"),
            pytest.raises(ValueError, match="conflicting selection strategies"),
        ):
            self._build(
                env,
                selection_strategy=MultiFrequencySelectionSpec(n_subpopulations=2),
                tournament=TournamentSelectionSpec(),
            )

    def test_explicit_tournament_none_warns_but_supplies_no_strategy(
        self, env, tournament_spec
    ):
        """The keyword's presence is deprecated, but a None value cannot conflict."""
        with pytest.warns(DeprecationWarning, match="'tournament' argument"):
            trainer = self._build(
                env, selection_strategy=tournament_spec, tournament=None
            )

        assert trainer.selection_strategy_spec is tournament_spec

    def test_unknown_kwarg_raises_type_error(self, env, tournament_spec):
        with pytest.raises(TypeError, match="tournment"):
            self._build(env, tournment=tournament_spec)

    def test_warning_points_at_the_caller(self, env, tournament_spec):
        """stacklevel must blame the user's call site, not the library."""
        with pytest.warns(DeprecationWarning, match="'tournament' argument") as record:
            self._build(env, tournament=tournament_spec)

        deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert deprecations[0].filename == __file__

    def test_arena_trainer_deprecated_tournament_kwarg_warns(
        self, mock_client, tournament_spec, training_spec
    ):
        with pytest.warns(DeprecationWarning, match="'tournament' argument"):
            trainer = ArenaTrainer(
                algorithm="PPO",
                environment="CartPole-v1",
                training=training_spec,
                client=mock_client,
                tournament=tournament_spec,
            )

        assert trainer.selection_strategy_spec is tournament_spec

    def test_arena_trainer_rejects_multi_frequency(self, mock_client, training_spec):
        with pytest.raises(ValueError, match="only supports tournament selection"):
            ArenaTrainer(
                algorithm="PPO",
                environment="CartPole-v1",
                training=training_spec,
                client=mock_client,
                selection_strategy=MultiFrequencySelectionSpec(n_subpopulations=2),
            )


class TestTrainerDeprecatedSelectionAttributes:
    @staticmethod
    def _build(selection_strategy):
        # A real env spec, so to_manifest() can serialize the environment section
        with (
            patch.object(LocalTrainer, "_make_env", return_value=VectorizedDummyEnv()),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
        ):
            return LocalTrainer(
                algorithm="PPO",
                environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
                training=TrainingSpec(max_steps=500, pop_size=8, evo_steps=100),
                selection_strategy=selection_strategy,
            )

    def test_tournament_selection_spec_warns_and_returns_the_spec(
        self, tournament_spec
    ):
        trainer = self._build(tournament_spec)

        with pytest.warns(DeprecationWarning, match="tournament_selection_spec"):
            assert trainer.tournament_selection_spec is tournament_spec

    def test_tournament_selection_warns_and_returns_the_operator(self, tournament_spec):
        trainer = self._build(tournament_spec)

        with pytest.warns(DeprecationWarning, match="tournament_selection"):
            assert isinstance(trainer.tournament_selection, TournamentSelection)

    def test_deprecated_attributes_are_none_under_multi_frequency(self):
        """Under MF-PBT there is no tournament spec or operator to return."""
        trainer = self._build(MultiFrequencySelectionSpec(n_subpopulations=2))

        with pytest.warns(DeprecationWarning, match="tournament_selection_spec"):
            assert trainer.tournament_selection_spec is None
        with pytest.warns(DeprecationWarning, match="tournament_selection"):
            assert trainer.tournament_selection is None

    def test_to_manifest_does_not_warn(self, tournament_spec):
        """to_manifest() runs on every train(), so it must read the attribute."""
        trainer = self._build(tournament_spec)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer.to_manifest()

        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


class TestManifestNetworkAttachmentGuards:
    def test_a_none_only_net_config_annotation_keeps_the_spec(self):
        from types import SimpleNamespace

        from agilerl.arena.models.algorithms import DQNSpec
        from agilerl.training.trainer import _algorithm_with_network

        class FrozenNetDQNSpec(DQNSpec):
            net_config: None = None

        algorithm = FrozenNetDQNSpec()
        manifest = SimpleNamespace(algorithm=algorithm, network=MagicMock())
        assert _algorithm_with_network(manifest) is algorithm


class TestFromManifestValidationGuard:
    def test_a_non_manifest_validation_result_is_rejected(self, monkeypatch):
        from agilerl.models import TrainingManifest

        monkeypatch.setattr(
            TrainingManifest,
            "get_validated",
            classmethod(lambda cls, manifest, mode: {"still": "a dict"}),
        )
        with pytest.raises(TypeError, match="expected TrainingManifest"):
            LocalTrainer.from_manifest({"algorithm": {"name": "DQN"}})


class TestLLMEnvConstructionGuards:
    """The dataset-LLM branch of ``LocalTrainer._make_env``."""

    def _trainer_with(self, algorithm_spec, tokenizer):
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.env_spec = ArenaLLMEnvSpec(
            env_type="dataset", objective="sft", dataset="ds"
        )
        trainer.algorithm_spec = algorithm_spec
        trainer.tokenizer = tokenizer
        trainer.accelerator = None
        return trainer

    def test_a_non_llm_algorithm_cannot_build_an_llm_env(self):
        from agilerl.arena.models.algorithms import DQNSpec

        trainer = self._trainer_with(DQNSpec(), MagicMock())
        with pytest.raises(TypeError, match="not an LLMAlgorithmSpec"):
            trainer._make_env()

    def test_a_missing_tokenizer_cannot_build_an_llm_env(self):
        trainer = self._trainer_with(LLMAlgorithmSpec.model_construct(), None)
        with pytest.raises(TypeError, match="requires a tokenizer"):
            trainer._make_env()


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestMakeTokenizerGuards:
    def test_a_non_llm_spec_gets_no_tokenizer(self):
        from agilerl.arena.models.algorithms import DQNSpec

        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = DQNSpec()
        with pytest.raises(TypeError, match="not an LLMAlgorithmSpec"):
            trainer._make_tokenizer()

    def test_a_none_tokenizer_from_transformers_is_rejected(self, monkeypatch):
        import agilerl.training.trainer as trainer_module

        class NoneTokenizer:
            @staticmethod
            def from_pretrained(name):
                return None

        monkeypatch.setattr(trainer_module, "AutoTokenizer", NoneTokenizer)
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = LLMAlgorithmSpec.model_construct(
            pretrained_model_name_or_path="stub/model"
        )
        with pytest.raises(TypeError, match="returned None"):
            trainer._make_tokenizer()


class TestDeferredNetConfigParadigmGuard:
    def test_a_spec_outside_the_narrowed_paradigms_is_left_alone(self):

        from agilerl.arena.models.algorithms import LLMAlgorithmSpec

        class NetConfigLLMSpec(LLMAlgorithmSpec):
            net_config: dict | None = None

        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = NetConfigLLMSpec.model_construct(net_config={"a": 1})
        # Has the field, but is not a single-agent or concrete multi-agent
        # spec, so the resolver leaves it untouched.
        assert trainer._resolve_deferred_net_config() is None
        assert trainer.algorithm_spec.net_config == {"a": 1}


class TestMakeRolloutFactoryGuards:
    def _trainer_with(self, algorithm_spec, tokenizer):
        trainer = LocalTrainer.__new__(LocalTrainer)
        trainer.algorithm_spec = algorithm_spec
        trainer.tokenizer = tokenizer
        return trainer

    def _rollout_spec(self):
        return ArenaLLMEnvSpec(
            env_type="rollout",
            dataset="ds",
            rubric_file_path="reward.py",
            prompt_template={"user": "{q}"},
        )

    def test_a_non_llm_algorithm_gets_no_rollout_factory(self):
        from agilerl.arena.models.algorithms import DQNSpec

        trainer = self._trainer_with(DQNSpec(), MagicMock())
        with pytest.raises(TypeError, match="not an LLMAlgorithmSpec"):
            trainer._make_rollout_factory(self._rollout_spec())

    def test_a_missing_tokenizer_gets_no_rollout_factory(self):
        trainer = self._trainer_with(LLMAlgorithmSpec.model_construct(), None)
        with pytest.raises(TypeError, match="requires a tokenizer"):
            trainer._make_rollout_factory(self._rollout_spec())
