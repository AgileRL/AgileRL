"""Tests for the refactored Trainer abstraction.

Covers:
- Trainer base class (algo name / meta resolution)
- LocalTrainer (construction, population resolution, train delegation)
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
    ArenaTrainingManifest,
    PPOSpec,
    DDPGSpec,
    DQNSpec,
    EnvironmentSpec,
    RLAlgorithmSpec,
    AlgorithmMeta,
)
from agilerl.models.hpo import (
    MutationProbabilities,
    MutationSpec,
    TournamentSelectionSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.trainer import ArenaTrainer, LocalTrainer, Trainer
from agilerl.utils.trainer_utils import (
    build_mutations,
    build_replay_buffer,
    build_tournament,
    build_train_kwargs,
    get_algo_meta,
    get_training_fn,
    resolve_algo_name,
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

    def __str__(self) -> str:
        return "DummyEnv"


@pytest.fixture()
def env():
    return DummyEnv()


@pytest.fixture()
def ppo_spec() -> PPOSpec:
    return PPOSpec(learn_step=128)


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
# trainer_utils — resolve_algo_name / get_algo_meta
# ---------------------------------------------------------------------------


class TestResolveAlgoName:
    def test_from_string(self):
        assert resolve_algo_name("PPO") == "PPO"

    def test_from_spec(self, ppo_spec):
        assert resolve_algo_name(ppo_spec) == "PPO"

    def test_from_ddpg_spec(self, ddpg_spec):
        assert resolve_algo_name(ddpg_spec) == "DDPG"

    def test_from_instance(self):
        inst = MagicMock()
        inst.algo = "TD3"
        assert resolve_algo_name(inst) == "TD3"

    def test_unknown_spec_falls_back_to_base(self):
        """An RLAlgorithmSpec subclass not directly in the registry matches
        the base RLAlgorithmSpec entries (e.g. CQN, NeuralUCB)."""

        class CustomSpec(RLAlgorithmSpec):
            pass

        # Should match one of the base RLAlgorithmSpec entries
        name = resolve_algo_name(CustomSpec())
        assert name in ALGO_REGISTRY


class TestGetAlgoMeta:
    def test_known_algorithm(self):
        meta = get_algo_meta("DQN")
        assert meta.name == "DQN"
        assert meta.spec_cls is DQNSpec
        assert meta.requires_buffer is True

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="No registry entry.*FakeAlgo"):
            get_algo_meta("FakeAlgo")

    def test_all_registry_entries_consistent(self):
        for name, meta in ALGO_REGISTRY.items():
            assert meta.name == name
            assert isinstance(meta.train_fn_name, str)
            assert isinstance(meta.requires_buffer, bool)


# ---------------------------------------------------------------------------
# trainer_utils — build_* functions
# ---------------------------------------------------------------------------


class TestBuildMutations:
    def test_none_returns_none(self):
        assert build_mutations(None, "cpu") is None

    def test_from_spec(self, mutation_spec):
        result = build_mutations(mutation_spec, "cpu")
        assert isinstance(result, Mutations)
        assert result.no_mut == 0.5
        assert result.parameters_mut == 0.3
        assert result.mutation_sd == 0.05


class TestBuildTournament:
    def test_none_returns_none(self, training_spec):
        assert build_tournament(None, training_spec) is None

    def test_from_spec(self, tournament_spec, training_spec):
        result = build_tournament(tournament_spec, training_spec)
        assert isinstance(result, TournamentSelection)
        assert result.tournament_size == 3
        assert result.elitism is True


class TestBuildReplayBuffer:
    def test_none_with_on_policy_returns_none(self):
        meta = get_algo_meta("PPO")
        assert build_replay_buffer(None, meta, "cpu") is None

    def test_none_with_off_policy_creates_default(self):
        meta = get_algo_meta("DQN")
        result = build_replay_buffer(None, meta, "cpu")
        assert isinstance(result, ReplayBuffer)
        assert result.max_size == 100_000

    def test_from_spec_standard(self, buffer_spec):
        meta = get_algo_meta("DQN")
        result = build_replay_buffer(buffer_spec, meta, "cpu")
        assert isinstance(result, ReplayBuffer)
        assert result.max_size == 5_000

    def test_from_spec_n_step(self):
        spec = ReplayBufferSpec(memory_size=10_000, n_step_buffer=True)
        meta = get_algo_meta("RainbowDQN")
        result = build_replay_buffer(spec, meta, "cpu")
        assert isinstance(result, MultiStepReplayBuffer)
        assert result.max_size == 10_000


class TestGetTrainingFn:
    def test_returns_callable(self):
        meta = get_algo_meta("PPO")
        fn = get_training_fn(meta)
        assert callable(fn)
        assert fn.__name__ == "train_on_policy"

    def test_off_policy_fn(self):
        meta = get_algo_meta("DQN")
        fn = get_training_fn(meta)
        assert fn.__name__ == "train_off_policy"

    def test_multi_agent_fn(self):
        meta = get_algo_meta("MADDPG")
        fn = get_training_fn(meta)
        assert fn.__name__ == "train_multi_agent_off_policy"


class TestBuildTrainKwargs:
    def test_on_policy_has_no_memory(self, env, training_spec):
        meta = get_algo_meta("PPO")
        kwargs = build_train_kwargs(
            algo_meta=meta,
            algo_name="PPO",
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
        assert kwargs["wandb_kwargs"] is None

    def test_off_policy_has_memory_and_delay(self, env, training_spec):
        meta = get_algo_meta("DQN")
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = build_train_kwargs(
            algo_meta=meta,
            algo_name="DQN",
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

    def test_bandit_has_memory_no_delay(self, env, training_spec):
        meta = get_algo_meta("NeuralUCB")
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = build_train_kwargs(
            algo_meta=meta,
            algo_name="NeuralUCB",
            env=env,
            env_name="test",
            pop=[MagicMock()],
            training=training_spec,
            tournament=None,
            mutations=None,
            memory=buffer,
        )
        assert kwargs["memory"] is buffer
        assert "learning_delay" not in kwargs


# ---------------------------------------------------------------------------
# LocalTrainer
# ---------------------------------------------------------------------------


class TestLocalTrainerConstruction:
    def test_string_algorithm(self, env):
        trainer = LocalTrainer(algorithm="PPO", environment=env)
        assert trainer._algo_name == "PPO"
        assert trainer._algo_meta.name == "PPO"
        assert trainer.training.max_steps == 1_000_000  # default

    def test_spec_algorithm(self, env, ppo_spec, training_spec):
        trainer = LocalTrainer(
            algorithm=ppo_spec,
            environment=env,
            training=training_spec,
        )
        assert trainer._algo_name == "PPO"
        assert trainer.training.pop_size == 2

    def test_pre_built_population(self, env, mock_population):
        trainer = LocalTrainer(
            algorithm="PPO",
            environment=env,
            population=mock_population,
        )
        assert trainer.population is mock_population

    def test_unknown_algorithm_raises(self, env):
        with pytest.raises(ValueError, match="No registry entry"):
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
            verbose=False,
            checkpoint=50,
            checkpoint_path="/tmp/ckpt",
            save_elite=True,
            elite_path="/tmp/elite",
            wb=True,
            wandb_api_key="fake-key",
            wandb_kwargs={"project": "test"},
            swap_channels=True,
            env_name="TestEnv",
            device="cpu",
        )
        assert trainer.verbose is False
        assert trainer.checkpoint == 50
        assert trainer.swap_channels is True
        assert trainer.env_name == "TestEnv"


class TestLocalTrainerResolvePopulation:
    def test_pre_built_population_returned(self, env, mock_population):
        trainer = LocalTrainer(
            algorithm="PPO", environment=env, population=mock_population
        )
        assert trainer._resolve_population() is mock_population

    def test_instance_cloned(self, env):
        from agilerl.algorithms.core import EvolvableAlgorithm

        mock_algo = MagicMock(spec=EvolvableAlgorithm)
        mock_algo.algo = "PPO"
        clone_results = [MagicMock(), MagicMock()]
        mock_algo.clone.side_effect = clone_results

        trainer = LocalTrainer(
            algorithm=mock_algo,
            environment=env,
            training=TrainingSpec(pop_size=2),
        )
        pop = trainer._resolve_population()
        assert len(pop) == 2
        assert mock_algo.clone.call_count == 2

    @patch("agilerl.trainer.create_population_from_spec")
    def test_spec_creates_population(self, mock_create, env, ppo_spec, training_spec):
        mock_create.return_value = [MagicMock(), MagicMock()]
        trainer = LocalTrainer(
            algorithm=ppo_spec, environment=env, training=training_spec
        )
        pop = trainer._resolve_population()
        mock_create.assert_called_once_with(ppo_spec, trainer._algo_meta, env, 2, "cpu")
        assert len(pop) == 2

    @patch("agilerl.trainer.create_population_from_spec")
    def test_string_creates_default_spec(self, mock_create, env, training_spec):
        mock_create.return_value = [MagicMock(), MagicMock()]
        trainer = LocalTrainer(algorithm="PPO", environment=env, training=training_spec)
        pop = trainer._resolve_population()
        # Should have been called with a default PPOSpec instance
        call_args = mock_create.call_args
        assert isinstance(call_args[0][0], PPOSpec)
        assert len(pop) == 2


class TestLocalTrainerTrain:
    @patch("agilerl.trainer.get_training_fn")
    @patch("agilerl.trainer.build_train_kwargs")
    @patch("agilerl.trainer.build_replay_buffer")
    @patch("agilerl.trainer.build_tournament")
    @patch("agilerl.trainer.build_mutations")
    def test_train_delegates_to_fn(
        self,
        mock_build_mut,
        mock_build_tourn,
        mock_build_buf,
        mock_build_kwargs,
        mock_get_fn,
        env,
        mock_population,
    ):
        mock_train_fn = MagicMock(return_value=(mock_population, [[1.0]]))
        mock_get_fn.return_value = mock_train_fn
        mock_build_kwargs.return_value = {"env": env, "pop": mock_population}
        mock_build_mut.return_value = None
        mock_build_tourn.return_value = None
        mock_build_buf.return_value = None

        trainer = LocalTrainer(
            algorithm="PPO",
            environment=env,
            population=mock_population,
        )
        result = trainer.train()

        mock_train_fn.assert_called_once()
        assert result == (mock_population, [[1.0]])

    @patch("agilerl.trainer.get_training_fn")
    def test_train_with_mutation_and_tournament(
        self, mock_get_fn, env, mock_population, mutation_spec, tournament_spec
    ):
        mock_train_fn = MagicMock(return_value=(mock_population, [[1.0]]))
        mock_get_fn.return_value = mock_train_fn

        trainer = LocalTrainer(
            algorithm="PPO",
            environment=env,
            population=mock_population,
            training=TrainingSpec(max_steps=100, pop_size=2, evo_steps=50),
            mutation=mutation_spec,
            tournament=tournament_spec,
        )
        result = trainer.train()
        mock_train_fn.assert_called_once()
        call_kwargs = mock_train_fn.call_args[1]
        assert isinstance(call_kwargs.get("mutation"), Mutations)
        assert isinstance(call_kwargs.get("tournament"), TournamentSelection)


# ---------------------------------------------------------------------------
# ArenaTrainer
# ---------------------------------------------------------------------------


class TestArenaTrainerConstruction:
    def test_string_algorithm_and_env(self, mock_client):
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment="CartPole-v1",
            client=mock_client,
        )
        assert trainer._algo_name == "PPO"
        assert trainer._client is mock_client

    def test_spec_algorithm(self, mock_client, ppo_spec):
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment="CartPole-v1",
            client=mock_client,
        )
        assert trainer._algo_name == "PPO"

    def test_env_spec(self, mock_client):
        env_spec = EnvironmentSpec(name="CartPole-v1", version="v1", num_envs=8)
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment=env_spec,
            client=mock_client,
        )
        assert trainer.environment is env_spec

    def test_default_training_spec(self, mock_client):
        trainer = ArenaTrainer(
            algorithm="PPO", environment="CartPole-v1", client=mock_client
        )
        assert isinstance(trainer.training, TrainingSpec)

    def test_all_specs(self, mock_client, mutation_spec, tournament_spec, buffer_spec):
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment="CartPole-v1",
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
        trainer = ArenaTrainer(algorithm="PPO", environment="CartPole-v1")
        assert trainer._client is not None


class TestArenaTrainerManifest:
    def test_string_based_manifest(self, mock_client):
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment="CartPole-v1",
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert isinstance(manifest, ArenaTrainingManifest)
        assert isinstance(manifest.algorithm, PPOSpec)
        assert manifest.environment.name == "CartPole-v1"
        assert manifest.training.max_steps == 1_000_000

    def test_spec_based_manifest(self, mock_client, ppo_spec):
        training = TrainingSpec(max_steps=50_000, pop_size=8)
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment="MountainCar-v0",
            training=training,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest.algorithm is ppo_spec
        assert manifest.algorithm.learn_step == 128
        assert manifest.training.pop_size == 8
        assert manifest.environment.name == "MountainCar-v0"

    def test_env_spec_manifest(self, mock_client):
        env_spec = EnvironmentSpec(name="LunarLander-v3", version="v2", num_envs=4)
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment=env_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()

        assert manifest.environment is env_spec
        assert manifest.environment.num_envs == 4

    def test_manifest_includes_mutation_and_tournament(
        self, mock_client, mutation_spec, tournament_spec
    ):
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment="CartPole-v1",
            mutation=mutation_spec,
            tournament=tournament_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest.mutation is mutation_spec
        assert manifest.tournament_selection is tournament_spec

    def test_manifest_includes_replay_buffer(self, mock_client, buffer_spec):
        trainer = ArenaTrainer(
            algorithm="DQN",
            environment="CartPole-v1",
            replay_buffer=buffer_spec,
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest.replay_buffer is buffer_spec

    def test_manifest_network_from_algo_spec(self, mock_client, ppo_spec):
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment="CartPole-v1",
            client=mock_client,
        )
        manifest = trainer.to_manifest()
        assert manifest.network is ppo_spec.net_config

    def test_invalid_env_type_raises(self, mock_client):
        trainer = ArenaTrainer.__new__(ArenaTrainer)
        trainer.algorithm = "PPO"
        trainer.environment = 42  # not an EnvironmentSpec or str
        trainer._algo_name = "PPO"
        trainer._algo_meta = get_algo_meta("PPO")
        trainer.training = TrainingSpec()
        trainer.mutation = None
        trainer.tournament = None
        trainer.replay_buffer = None
        trainer._client = mock_client

        with pytest.raises(TypeError, match="ArenaTrainer requires"):
            trainer.to_manifest()


class TestArenaTrainerTrain:
    def test_train_submits_manifest(self, mock_client):
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment="CartPole-v1",
            client=mock_client,
        )
        result = trainer.train()

        mock_client.submit_job.assert_called_once()
        submitted_manifest = mock_client.submit_job.call_args[0][0]
        assert isinstance(submitted_manifest, ArenaTrainingManifest)
        assert result["job_id"] == "test-123"

    def test_train_with_stream(self, mock_client):
        trainer = ArenaTrainer(
            algorithm="PPO",
            environment="CartPole-v1",
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
        assert spec.net_config is not None

    def test_dqn_defaults(self):
        spec = DQNSpec()
        assert spec.learn_step == 5

    def test_ddpg_defaults(self):
        spec = DDPGSpec()
        assert spec.learn_step == 5

    def test_rl_algorithm_spec_default_net_config(self):
        spec = RLAlgorithmSpec()
        assert spec.net_config is not None
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
        assert set(ALGO_REGISTRY.keys()) == self.EXPECTED_ALGOS

    def test_on_policy_algos_dont_require_buffer(self):
        for name in ("PPO", "IPPO"):
            assert ALGO_REGISTRY[name].requires_buffer is False

    def test_off_policy_algos_require_buffer(self):
        for name in ("DQN", "DDPG", "TD3", "RainbowDQN", "MADDPG", "MATD3"):
            assert ALGO_REGISTRY[name].requires_buffer is True

    def test_bandit_and_offline_require_buffer(self):
        for name in ("NeuralUCB", "NeuralTS", "CQN"):
            assert ALGO_REGISTRY[name].requires_buffer is True
