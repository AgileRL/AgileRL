"""Tests for the refactored Trainer abstraction.

Covers:
- Trainer base class (algo resolution from string / spec)
- LocalTrainer (construction, train delegation)
- ArenaTrainer (construction, manifest building, train delegation)
- trainer_utils pure functions
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from gymnasium.spaces import Box, Discrete

from agilerl.components.replay_buffer import MultiStepReplayBuffer, ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.models import (
    ALGO_REGISTRY,
    DDPGSpec,
    DQNSpec,
    ArenaEnvSpec,
    PPOSpec,
    RLAlgorithmSpec,
    TrainingManifest,
)
from agilerl.models.networks import MlpSpec, NetworkSpec
from agilerl.models.hpo import (
    MutationProbabilities,
    MutationSpec,
    TournamentSelectionSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.training.trainer import ArenaTrainer, LocalTrainer, Trainer
from agilerl.utils.trainer_utils import (
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_tournament_from_spec,
    build_train_kwargs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyEnv:
    """Minimal gym-like environment for unit tests."""

    def __init__(self) -> None:
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

    def __str__(self) -> str:
        return "DummyEnv"


@pytest.fixture()
def env():
    return DummyEnv()


@pytest.fixture()
def ppo_spec() -> PPOSpec:
    return PPOSpec(
        learn_step=128,
        net_config=NetworkSpec(
            encoder_config=MlpSpec(hidden_size=[64]),
            head_config=MlpSpec(hidden_size=[64]),
        ),
    )


@pytest.fixture()
def ddpg_spec() -> DDPGSpec:
    return DDPGSpec()


@pytest.fixture()
def training_spec() -> TrainingSpec:
    return TrainingSpec(max_steps=500, pop_size=2, evo_steps=100)


@pytest.fixture()
def mutation_spec() -> MutationSpec:
    return MutationSpec(
        probabilities=MutationProbabilities(no_mut=0.5, params_mut=0.3, rl_hp_mut=0.2),
        mutation_sd=0.05,
    )


@pytest.fixture()
def tournament_spec() -> TournamentSelectionSpec:
    return TournamentSelectionSpec(tournament_size=3, elitism=True)


@pytest.fixture()
def buffer_spec() -> ReplayBufferSpec:
    return ReplayBufferSpec(memory_size=5_000)


@pytest.fixture()
def mock_population():
    """Return a list of mock agents that quack like EvolvableAlgorithm."""
    agents = [MagicMock(algo="PPO") for _ in range(2)]
    return agents


@pytest.fixture()
def mock_client():
    client = MagicMock()
    client.submit_job.return_value = {"job_id": "test-123", "status": "PENDING"}
    return client


# ---------------------------------------------------------------------------
# trainer_utils — build_* functions
# ---------------------------------------------------------------------------


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


class TestBuildTrainKwargs:
    def test_on_policy_has_no_memory(self, env, training_spec, ppo_spec):
        kwargs = build_train_kwargs(
            algo_spec=ppo_spec,
            env=env,
            env_name="test",
            pop=[MagicMock()],
            training=training_spec,
            tournament=None,
            mutations=None,
            memory=None,
        )
        assert "memory" not in kwargs
        assert kwargs["max_steps"] == 500

    def test_off_policy_has_memory_and_delay(self, env, training_spec):
        dqn_spec = DQNSpec()
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = build_train_kwargs(
            algo_spec=dqn_spec,
            env=env,
            env_name="test",
            pop=[MagicMock()],
            training=training_spec,
            tournament=None,
            mutations=None,
            memory=buffer,
        )
        assert kwargs["memory"] is buffer
        assert "learning_delay" in kwargs


# ---------------------------------------------------------------------------
# LocalTrainer
# ---------------------------------------------------------------------------


class TestLocalTrainerConstruction:
    def test_string_algorithm(self, env):
        trainer = LocalTrainer(algorithm="PPO", environment=env)
        assert isinstance(trainer.algorithm, PPOSpec)
        assert trainer.training.max_steps == 1_000_000  # default

    def test_spec_algorithm(self, env, ppo_spec, training_spec):
        trainer = LocalTrainer(
            algorithm=ppo_spec,
            environment=env,
            training=training_spec,
        )
        assert trainer.algorithm is ppo_spec
        assert trainer.training.pop_size == 2

    def test_unknown_algorithm_raises(self, env):
        with pytest.raises((ValueError, KeyError)):
            LocalTrainer(algorithm="UnknownAlgo", environment=env)

    def test_default_training_spec(self, env):
        trainer = LocalTrainer(algorithm="PPO", environment=env)
        assert isinstance(trainer.training, TrainingSpec)
        assert trainer.training.pop_size == 4

    def test_all_optional_params(
        self, env, mutation_spec, tournament_spec, buffer_spec
    ):
        trainer = LocalTrainer(
            algorithm="DQN",
            environment=env,
            training=TrainingSpec(max_steps=200),
            mutation=mutation_spec,
            tournament=tournament_spec,
            replay_buffer=buffer_spec,
            device="cpu",
        )
        assert trainer.mutation is mutation_spec
        assert trainer.tournament is tournament_spec
        assert trainer.replay_buffer is buffer_spec


class TestLocalTrainerTrain:
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_train_delegates_to_fn(
        self,
        mock_create_pop,
        env,
    ):
        mock_pop = [MagicMock()]
        mock_create_pop.return_value = mock_pop
        mock_train_fn = MagicMock(return_value=(mock_pop, [[1.0]]))

        trainer = LocalTrainer(
            algorithm="PPO",
            environment=env,
        )
        with patch.object(PPOSpec, "get_training_fn", return_value=mock_train_fn):
            result = trainer.train()

        mock_train_fn.assert_called_once()
        assert result == (mock_pop, [[1.0]])


# ---------------------------------------------------------------------------
# ArenaTrainer
# ---------------------------------------------------------------------------


class TestArenaTrainerConstruction:
    def test_string_algorithm_and_env(self, mock_client):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            client=mock_client,
        )
        assert isinstance(trainer.algorithm, PPOSpec)
        assert trainer._client is mock_client

    def test_spec_algorithm(self, mock_client, ppo_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        assert trainer.algorithm is ppo_spec

    def test_env_spec(self, mock_client):
        env_spec = ArenaEnvSpec(name="CartPole-v1", version="v1", num_envs=8)
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment=env_spec,
            client=mock_client,
        )
        assert trainer.environment is env_spec

    def test_default_training_spec(self, mock_client):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO", environment=env_spec, client=mock_client
        )
        assert isinstance(trainer.training, TrainingSpec)

    def test_all_specs(self, mock_client, mutation_spec, tournament_spec, buffer_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment=env_spec,
            training=TrainingSpec(max_steps=200, pop_size=3),
            mutation=mutation_spec,
            tournament=tournament_spec,
            replay_buffer=buffer_spec,
            client=mock_client,
        )
        assert trainer.mutation is mutation_spec
        assert trainer.tournament is tournament_spec
        assert trainer.replay_buffer is buffer_spec
        assert trainer.training.pop_size == 3

    @patch("agilerl.arena.client.ArenaClient")
    def test_auto_creates_client(self, mock_cls):
        """If no client is passed, ArenaTrainer creates one automatically."""
        mock_cls.return_value = MagicMock()
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(algorithm="PPO", environment=env_spec)
        assert trainer._client is not None


class TestArenaTrainerManifest:
    def test_minimal_manifest_from_string_algo_and_env(self, mock_client):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment=env_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        payload = manifest.to_payload()

        assert isinstance(manifest, TrainingManifest)
        assert manifest.network is None
        assert payload["algorithm"]["name"] == "PPO"
        assert payload["environment"]["name"] == "CartPole-v1"
        assert "network" not in payload

    def test_string_based_manifest(self, mock_client, ppo_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert isinstance(manifest, TrainingManifest)
        assert isinstance(manifest.algorithm, PPOSpec)
        assert manifest.environment.name == "CartPole-v1"
        assert manifest.training.max_steps == 1_000_000

    def test_spec_based_manifest(self, mock_client, ppo_spec):
        training = TrainingSpec(max_steps=50_000, pop_size=8)
        env_spec = ArenaEnvSpec(name="MountainCar-v0")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest.algorithm is ppo_spec
        assert manifest.algorithm.learn_step == 128
        assert manifest.training.pop_size == 8
        assert manifest.environment.name == "MountainCar-v0"

    def test_env_spec_manifest(self, mock_client):
        env_spec = ArenaEnvSpec(name="LunarLander-v3", version="v2", num_envs=4)
        dqn_spec = DQNSpec(
            net_config=NetworkSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            )
        )
        trainer = ArenaTrainer(
            algorithm=dqn_spec,
            environment=env_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest.environment is env_spec
        assert manifest.environment.num_envs == 4

    def test_manifest_includes_mutation_and_tournament(
        self, mock_client, mutation_spec, tournament_spec, ppo_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            mutation=mutation_spec,
            tournament=tournament_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest.mutation is mutation_spec
        assert manifest.tournament_selection is tournament_spec

    def test_manifest_includes_replay_buffer(self, mock_client, buffer_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        dqn_spec = DQNSpec(
            net_config=NetworkSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            )
        )
        trainer = ArenaTrainer(
            algorithm=dqn_spec,
            environment=env_spec,
            replay_buffer=buffer_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest.replay_buffer is buffer_spec

    def test_manifest_network_from_algo_spec(self, mock_client, ppo_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest.network is ppo_spec.net_config

    def test_manifest_payload_uses_spec_serializers(self, mock_client, ppo_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        payload = trainer.to_manifest().to_payload()

        assert payload["algorithm"]["name"] == "PPO"
        assert payload["training"]["name"] == "train_on_policy"
        assert payload["environment"]["name"] == "CartPole-v1"

    def test_invalid_env_type_raises(self, mock_client):
        """Passing a non-spec environment to ArenaTrainer.to_manifest raises TypeError."""
        trainer = ArenaTrainer.__new__(ArenaTrainer)
        trainer.algorithm = PPOSpec()
        trainer.environment = 42  # not an EnvironmentSpec or str
        trainer.training = TrainingSpec()
        trainer.mutation = None
        trainer.tournament = None
        trainer.replay_buffer = None
        trainer._client = mock_client

        with pytest.raises(TypeError, match="ArenaTrainer requires"):
            trainer.to_manifest()


class TestArenaTrainerTrain:
    def test_train_submits_manifest(self, mock_client, ppo_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        result = trainer.train()

        mock_client.submit_job.assert_called_once()
        submitted_manifest = mock_client.submit_job.call_args[0][0]
        assert isinstance(submitted_manifest, TrainingManifest)
        assert result["job_id"] == "test-123"

    def test_train_submits_manifest_with_serializable_payload(
        self, mock_client, ppo_spec
    ):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        trainer.train()
        submitted_manifest = mock_client.submit_job.call_args[0][0]
        payload = submitted_manifest.to_payload()
        assert payload["algorithm"]["name"] == "PPO"
        assert payload["training"]["name"] == "train_on_policy"

    def test_train_with_stream(self, mock_client, ppo_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            client=mock_client,
        )
        trainer.train(stream=True)

        _, call_kwargs = mock_client.submit_job.call_args
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Spec defaults (zero-config)
# ---------------------------------------------------------------------------


class TestSpecDefaults:
    def test_ppo_defaults(self):
        spec = PPOSpec()
        assert spec.learn_step == 2048
        assert spec.num_envs == 1
        assert spec.gamma == 0.99
        assert spec.net_config is None

    def test_dqn_defaults(self):
        spec = DQNSpec()
        assert spec.learn_step == 5

    def test_ddpg_defaults(self):
        spec = DDPGSpec()
        assert spec.learn_step == 5

    def test_rl_algorithm_spec_default_net_config(self):
        spec = RLAlgorithmSpec()
        assert spec.net_config is None
        assert spec.learn_step == 5


# ---------------------------------------------------------------------------
# ALGO_REGISTRY completeness
# ---------------------------------------------------------------------------


class TestAlgoRegistry:
    EXPECTED_ALGOS = {
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
        available = set(ALGO_REGISTRY.arena_algorithms()) | set(
            ALGO_REGISTRY.local_algorithms()
        )
        assert self.EXPECTED_ALGOS.issubset(available)

    def test_registry_entries_have_spec_cls(self):
        for name in self.EXPECTED_ALGOS:
            entry = ALGO_REGISTRY.get(name)
            assert entry.spec_cls is not None
