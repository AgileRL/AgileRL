"""Tests for the refactored Trainer abstraction.

Covers:
- Trainer base class (algo resolution from string / spec)
- LocalTrainer (construction, train delegation)
- ArenaTrainer (construction, manifest building, train delegation)
- trainer_utils pure functions
"""

from __future__ import annotations

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
)
from agilerl.models.networks import MlpSpec, QNetworkSpec, StochasticActorSpec
from agilerl.models.hpo import (
    MutationProbabilities,
    MutationSpec,
    TournamentSelectionSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.training.trainer import ArenaTrainer, LocalTrainer
from agilerl.utils.trainer_utils import (
    build_mutations_from_spec,
    build_replay_buffer_from_spec,
    build_tournament_from_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture()
def env():
    return DummyEnv()


@pytest.fixture()
def ppo_spec() -> PPOSpec:
    return PPOSpec(
        learn_step=128,
        net_config=StochasticActorSpec(
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


class TestGetTrainingKwargs:
    @pytest.fixture()
    def gym_env_spec(self):
        from agilerl.models.env import GymEnvSpec

        return GymEnvSpec(name="CartPole-v1", num_envs=1)

    def test_on_policy_has_no_memory(self, training_spec, ppo_spec, gym_env_spec):
        kwargs = ppo_spec.get_training_kwargs(
            training=training_spec, env_spec=gym_env_spec, memory=None
        )
        assert "memory" not in kwargs
        assert kwargs["algo"] == "PPO"
        assert kwargs["eval_loop"] == training_spec.eval_loop

    def test_off_policy_has_memory_and_delay(self, training_spec, gym_env_spec):
        dqn_spec = DQNSpec()
        buffer = ReplayBuffer(max_size=100, device="cpu")
        kwargs = dqn_spec.get_training_kwargs(
            training=training_spec, env_spec=gym_env_spec, memory=buffer
        )
        assert kwargs["memory"] is buffer
        assert "learning_delay" in kwargs

    def test_env_name_forwarded(self, training_spec, ppo_spec, gym_env_spec):
        kwargs = ppo_spec.get_training_kwargs(
            training=training_spec, env_spec=gym_env_spec
        )
        assert kwargs["env_name"] == "CartPole-v1"


# ---------------------------------------------------------------------------
# LocalTrainer
# ---------------------------------------------------------------------------


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
        assert trainer.training_spec.population_size == 2

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
            tournament=tournament_spec,
            replay_buffer=buffer_spec,
            device="cpu",
        )
        assert trainer.mutation_spec is mutation_spec
        assert trainer.tournament_selection_spec is tournament_spec
        assert trainer.replay_buffer_spec is buffer_spec


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
            patch.object(PPOSpec, "get_training_fn", return_value=mock_train_fn),
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


# ---------------------------------------------------------------------------
# ArenaTrainer
# ---------------------------------------------------------------------------


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
            tournament=tournament_spec,
            replay_buffer=buffer_spec,
            client=mock_client,
        )
        assert trainer.mutation_spec is mutation_spec
        assert trainer.tournament_selection_spec is tournament_spec
        assert trainer.replay_buffer_spec is buffer_spec
        assert trainer.training_spec.population_size == 3

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
        assert "network" not in manifest

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
        assert manifest["training"]["population_size"] == 8
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
            tournament=tournament_spec,
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
        assert manifest["algorithm"]["net_config"]["encoder_config"]["hidden_size"] == [
            64
        ]
        assert manifest["algorithm"]["net_config"]["head_config"]["hidden_size"] == [64]

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
        trainer.tournament_selection_spec = None
        trainer.replay_buffer_spec = None
        trainer._client = mock_client

        with pytest.raises((TypeError, Exception)):
            trainer.to_manifest()


class TestArenaTrainerTrain:
    def test_train_submits_manifest(self, mock_client, ppo_spec, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        result = trainer.train()

        mock_client.submit_job.assert_called_once()
        submitted_manifest = mock_client.submit_job.call_args[0][0]
        assert isinstance(submitted_manifest, dict)
        assert submitted_manifest["algorithm"]["name"] == "PPO"
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
        submitted_manifest = mock_client.submit_job.call_args[0][0]
        assert submitted_manifest["algorithm"]["name"] == "PPO"
        assert submitted_manifest["training"]["max_steps"] == 500

    def test_train_with_stream(self, mock_client, ppo_spec, training_spec):
        env_spec = ArenaEnvSpec(name="CartPole-v1")
        trainer = ArenaTrainer(
            algorithm=ppo_spec,
            environment=env_spec,
            training=training_spec,
            client=mock_client,
        )
        trainer.train(stream=True)

        _, call_kwargs = mock_client.submit_job.call_args
        assert call_kwargs["stream"] is True


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
