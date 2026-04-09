"""Tests for the refactored Trainer abstraction.

Covers:
- Trainer base class (algo resolution from string / spec)
- LocalTrainer (construction, train delegation)
- ArenaTrainer (construction, manifest building, train delegation)
- trainer_utils pure functions
- LLM algorithm integration (DPO, GRPO) with mocked dependencies
"""

from __future__ import annotations

import types
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
    TD3Spec,
    RLAlgorithmSpec,
)
from agilerl.models.algo import LLMAlgorithmSpec
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


# ---------------------------------------------------------------------------
# Custom network injection via LocalTrainer
# ---------------------------------------------------------------------------


class VectorizedDummyEnv(DummyEnv):
    """DummyEnv that exposes vectorized-env *attributes* (not methods) for
    ``single_observation_space`` / ``single_action_space`` so that
    ``get_spaces_from_env`` can read them directly."""

    def __init__(self, *, continuous: bool = False) -> None:
        super().__init__()
        if continuous:
            self.action_space = Box(low=-1.0, high=1.0, shape=(2,))
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space


class TestLocalTrainerCustomNetworks:
    """Verify that custom EvolvableMLP networks passed to algorithm specs
    are propagated to every individual in the LocalTrainer population."""

    OBS_DIM = 4
    DISCRETE_ACTIONS = 2
    CONTINUOUS_ACTIONS = 2
    HIDDEN = [64, 64]
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

    def _build_trainer(self, algo_spec, env):
        with patch.object(LocalTrainer, "_make_env", return_value=env):
            return LocalTrainer(
                algorithm=algo_spec,
                environment=env,
                training=self._training(),
            )

    # -- DQN (discrete, actor only) -----------------------------------------

    def test_dqn_custom_actor(self):
        from agilerl.modules.mlp import EvolvableMLP

        actor = self._make_mlp(self.OBS_DIM, self.DISCRETE_ACTIONS)
        spec = DQNSpec(actor_network=actor)

        trainer = self._build_trainer(spec, VectorizedDummyEnv())

        assert len(trainer.population) == self.POP_SIZE
        for agent in trainer.population:
            assert isinstance(agent.actor, EvolvableMLP)
            assert agent.actor.hidden_size == self.HIDDEN

    # -- PPO (discrete, actor + critic) -------------------------------------

    def test_ppo_custom_actor_critic(self):
        from agilerl.modules.mlp import EvolvableMLP

        actor = self._make_mlp(self.OBS_DIM, self.DISCRETE_ACTIONS)
        critic = self._make_mlp(self.OBS_DIM, 1)
        spec = PPOSpec(actor_network=actor, critic_network=critic)

        trainer = self._build_trainer(spec, VectorizedDummyEnv())

        assert len(trainer.population) == self.POP_SIZE
        for agent in trainer.population:
            assert isinstance(agent.actor, EvolvableMLP)
            assert isinstance(agent.critic, EvolvableMLP)
            assert agent.actor.hidden_size == self.HIDDEN
            assert agent.critic.hidden_size == self.HIDDEN

    # -- DDPG (continuous, actor + critic) ----------------------------------

    def test_ddpg_custom_actor_critic(self):
        from agilerl.modules.mlp import EvolvableMLP

        actor = self._make_mlp(self.OBS_DIM, self.CONTINUOUS_ACTIONS)
        critic = self._make_mlp(self.OBS_DIM + self.CONTINUOUS_ACTIONS, 1)
        spec = DDPGSpec(actor_network=actor, critic_network=critic)

        trainer = self._build_trainer(spec, VectorizedDummyEnv(continuous=True))

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
        spec = TD3Spec(
            actor_network=actor,
            critic_networks=[critic_1, critic_2],
        )

        trainer = self._build_trainer(spec, VectorizedDummyEnv(continuous=True))

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
        spec = DQNSpec(actor_network=actor)

        trainer = self._build_trainer(spec, VectorizedDummyEnv())

        actors = [agent.actor for agent in trainer.population]
        for i, a in enumerate(actors):
            assert a is not actor, "Population actor should be a deep copy"
            for j, b in enumerate(actors):
                if i != j:
                    assert a is not b, "Each individual should have its own copy"


# ---------------------------------------------------------------------------
# LLM algorithm integration (DPO / GRPO)
# ---------------------------------------------------------------------------


try:
    from peft import LoraConfig as _LoraConfig

    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


class FakeLoraConfig:
    """Lightweight stand-in for ``peft.LoraConfig`` used when the real peft
    package is not installed."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_lora_config(**kwargs):
    """Create a LoraConfig using the real peft class when available,
    falling back to FakeLoraConfig otherwise."""
    if _HAS_PEFT:
        return _LoraConfig(
            r=kwargs.get("lora_r", 8),
            lora_alpha=kwargs.get("lora_alpha", 16),
            lora_dropout=kwargs.get("lora_dropout", 0.1),
            task_type="CAUSAL_LM",
        )
    return FakeLoraConfig(**kwargs)


def _rebuild_llm_specs():
    """Import and rebuild the LLM algorithm specs so Pydantic can resolve
    the ``LoraConfig`` forward reference against :class:`FakeLoraConfig`.

    Returns ``(DPOSpec, GRPOSpec)`` classes ready for instantiation.
    """
    import sys

    _LoraConfigCls = _LoraConfig if _HAS_PEFT else FakeLoraConfig

    if "peft" not in sys.modules:
        peft_mod = types.ModuleType("peft")
        peft_mod.LoraConfig = _LoraConfigCls
        sys.modules["peft"] = peft_mod

    from agilerl.models.algorithms.dpo import DPOSpec
    from agilerl.models.algorithms.grpo import GRPOSpec
    from agilerl.utils.algo_utils import CosineLRScheduleConfig, VLLMConfig

    ns = {"LoraConfig": _LoraConfigCls}
    LLMAlgorithmSpec.model_rebuild(_types_namespace=ns)
    DPOSpec.model_rebuild(_types_namespace=ns)
    grpo_ns = {
        **ns,
        "VLLMConfig": VLLMConfig,
        "CosineLRScheduleConfig": CosineLRScheduleConfig,
    }
    GRPOSpec.model_rebuild(_types_namespace=grpo_ns)

    return DPOSpec, GRPOSpec


_DPOSpec, _GRPOSpec = _rebuild_llm_specs()

_LLM_COMMON_KWARGS = dict(
    update_epochs=1,
    lora_config=_make_lora_config(lora_r=8, lora_alpha=16, lora_dropout=0.1),
    max_model_len=512,
    use_separate_reference_adapter=False,
    pretrained_model_name_or_path="gpt2",
    calc_position_embeddings=False,
)


@pytest.fixture()
def dpo_spec():
    return _DPOSpec(**_LLM_COMMON_KWARGS)


@pytest.fixture()
def grpo_spec():
    return _GRPOSpec(group_size=4, **_LLM_COMMON_KWARGS)


class TestLLMSpecConstruction:
    """Verify that DPOSpec / GRPOSpec can be constructed and expose the
    expected class-level attributes."""

    def test_dpo_spec_fields(self, dpo_spec):
        assert dpo_spec.name == "DPO"
        assert dpo_spec.env_type == "preference"
        assert isinstance(dpo_spec, LLMAlgorithmSpec)
        assert dpo_spec.pretrained_model_name_or_path == "gpt2"

    def test_grpo_spec_fields(self, grpo_spec):
        assert grpo_spec.name == "GRPO"
        assert grpo_spec.env_type == "reasoning"
        assert isinstance(grpo_spec, LLMAlgorithmSpec)
        assert grpo_spec.group_size == 4

    def test_dpo_training_fn(self, dpo_spec):
        from agilerl.training.train_llm import finetune_llm_preference

        assert dpo_spec.get_training_fn() is finetune_llm_preference

    def test_grpo_training_fn(self, grpo_spec):
        from agilerl.training.train_llm import finetune_llm_reasoning

        assert grpo_spec.get_training_fn() is finetune_llm_reasoning

    def test_dpo_model_dump_contains_expected_fields(self, dpo_spec):
        dumped = dpo_spec.model_dump(mode="python", exclude={"hp_config"})
        assert dumped["pretrained_model_name_or_path"] == "gpt2"
        assert dumped["update_epochs"] == 1
        assert dumped["beta"] == pytest.approx(0.001)

    def test_grpo_model_dump_contains_group_size(self, grpo_spec):
        dumped = grpo_spec.model_dump(mode="python", exclude={"hp_config"})
        assert dumped["group_size"] == 4
        assert dumped["pretrained_model_name_or_path"] == "gpt2"


class TestLLMGetTrainingKwargs:
    """Verify the LLM-specific early-return path in
    ``AlgorithmSpec.get_training_kwargs``."""

    def test_llm_kwargs_defaults(self, dpo_spec):
        env_spec = MagicMock(max_reward=None)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        kwargs = dpo_spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert kwargs == {"evaluation_interval": 10}
        assert "max_reward" not in kwargs
        assert "num_epochs" not in kwargs

    def test_llm_kwargs_include_max_reward(self, grpo_spec):
        env_spec = MagicMock(max_reward=5.0)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        kwargs = grpo_spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert kwargs["max_reward"] == 5.0

    def test_llm_kwargs_include_checkpoint_steps(self, dpo_spec):
        env_spec = MagicMock(max_reward=None)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        training.checkpoint_steps = 50
        kwargs = dpo_spec.get_training_kwargs(training=training, env_spec=env_spec)
        assert kwargs["checkpoint_steps"] == 50

    def test_llm_kwargs_never_include_memory(self, dpo_spec):
        env_spec = MagicMock(max_reward=None)
        training = TrainingSpec(max_steps=100, evo_steps=10, pop_size=2)
        buf = MagicMock()
        kwargs = dpo_spec.get_training_kwargs(
            training=training, env_spec=env_spec, memory=buf
        )
        assert "memory" not in kwargs


class TestLLMBuildAlgorithm:
    """Verify that ``LLMAlgorithmSpec.build_algorithm`` calls the algo class
    constructor with the right arguments."""

    def test_dpo_build_algorithm(self, dpo_spec):
        mock_algo = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"

        with patch.object(type(dpo_spec), "algo_class", mock_algo):
            agent = dpo_spec.build_algorithm(tokenizer=mock_tokenizer, index=0)

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

        with patch.object(type(grpo_spec), "algo_class", mock_algo):
            agent = grpo_spec.build_algorithm(tokenizer=mock_tokenizer, index=1)

        mock_algo.assert_called_once()
        call_kwargs = mock_algo.call_args[1]
        assert call_kwargs["index"] == 1
        assert call_kwargs["model_name"] == "gpt2"

    def test_build_algorithm_with_accelerator(self, dpo_spec):
        mock_algo = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 50256
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_accel = MagicMock()
        mock_accel.num_processes = 2

        with patch.object(type(dpo_spec), "algo_class", mock_algo):
            dpo_spec.build_algorithm(
                tokenizer=mock_tokenizer, index=0, accelerator=mock_accel
            )

        call_kwargs = mock_algo.call_args[1]
        assert call_kwargs["accelerator"] is mock_accel
        assert call_kwargs["micro_batch_size_per_gpu"] is not None


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

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
            ) as mock_create_pop,
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            from agilerl.models.env import LLMEnvSpec

            env_spec = MagicMock(spec=LLMEnvSpec)
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

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch.object(LocalTrainer, "_make_env", return_value=mock_env),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=mock_pop,
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

    # -- _make_env dispatches to LLMEnvSpec.make_env() ----------------------

    def test_make_env_dispatches_to_llm_env_spec(self, dpo_spec):
        from agilerl.models.env import LLMEnvSpec

        mock_env = MagicMock()
        mock_llm_env_spec = MagicMock(spec=LLMEnvSpec)
        mock_llm_env_spec.make_env.return_value = mock_env
        mock_tokenizer = MagicMock()

        with (
            patch(
                "agilerl.training.trainer.AutoTokenizer", create=True
            ) as mock_auto_tok,
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
        ):
            mock_auto_tok.from_pretrained.return_value = mock_tokenizer
            trainer = LocalTrainer(
                algorithm=dpo_spec,
                environment=mock_llm_env_spec,
                training=self._training(),
            )

        assert mock_llm_env_spec.return_raw_completions is False
        assert mock_llm_env_spec.max_context_length == dpo_spec.max_model_len
        assert mock_llm_env_spec.seed == dpo_spec.seed
        mock_llm_env_spec.make_env.assert_called_once_with(
            tokenizer=mock_tokenizer, accelerator=None
        )
        assert trainer.env is mock_env

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
            patch.object(type(dpo_spec), "get_training_fn", return_value=mock_train_fn),
            patch.object(LocalTrainer, "to_manifest", return_value={}),
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

    # -- Missing LLM dependencies raises ImportError -----------------------

    def test_missing_llm_deps_raises(self, dpo_spec):
        with patch("agilerl.training.trainer.AutoTokenizer", None):
            with pytest.raises(ImportError, match="LLM dependencies"):
                LocalTrainer(
                    algorithm=dpo_spec,
                    environment=MagicMock(),
                    training=self._training(),
                )


class TestLLMAlgoRegistry:
    """Verify that DPO/GRPO are registered once their modules are imported.

    The ``_rebuild_llm_specs()`` call at module scope force-imports the LLM
    spec modules, which triggers ``@register``.  So by the time these tests
    run the entries exist regardless of whether the real ``peft`` /
    ``transformers`` packages are installed.
    """

    def test_dpo_registered(self):
        entry = ALGO_REGISTRY.get("DPO")
        assert entry.spec_cls is _DPOSpec

    def test_grpo_registered(self):
        entry = ALGO_REGISTRY.get("GRPO")
        assert entry.spec_cls is _GRPOSpec

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

    # -- Off-policy continuous: DDPG + LunarLanderContinuous ----------------

    def test_ddpg_continuous(self):
        """DDPG (off-policy, continuous) on LunarLander."""
        from agilerl.models.env import GymEnvSpec

        trainer = LocalTrainer(
            algorithm=DDPGSpec(learn_step=1),
            environment=GymEnvSpec(name="LunarLanderContinuous-v3", num_envs=1),
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

    def test_neural_ucb_bandit(self):
        """NeuralUCB (bandit) on synthetic data."""
        import numpy as np
        import pandas as pd

        from agilerl.models.env import BanditEnvSpec
        from agilerl.models import NeuralUCBSpec

        rng = np.random.default_rng(42)
        features = pd.DataFrame(rng.standard_normal((100, 4)).astype(np.float32))
        targets = pd.DataFrame(rng.integers(0, 2, size=(100, 1)).astype(np.float32))

        trainer = LocalTrainer(
            algorithm=NeuralUCBSpec(learn_step=2),
            environment=BanditEnvSpec(features=features, targets=targets),
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
        from agilerl.models.env import PzEnvSpec
        from agilerl.models.hpo import MutationProbabilities

        trainer = LocalTrainer(
            algorithm=MADDPGSpec(learn_step=2),
            environment=PzEnvSpec(
                name="pettingzoo.mpe.simple_speaker_listener_v4",
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
            algorithm=DQNSpec(actor_network=actor, learn_step=1),
            environment=GymEnvSpec(name="CartPole-v1", num_envs=1),
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
            training=self._training(max_steps=64, evo_steps=32, eval_steps=10),
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
    def test_grpo_reasoning_train(self, tmp_path):
        """GRPO reasoning: real LLMEnvSpec.make_env() with temp dataset files.

        Tests the full LocalTrainer wiring: spec → env construction → kwargs
        assembly → dispatch to finetune_llm_reasoning.  The finetune function
        itself is patched since running it requires CUDA agents.
        """
        try:
            from peft import LoraConfig
            from agilerl.models.env import LLMEnvSpec, LLMEnvType
        except ImportError:
            pytest.skip("LLM dependencies not installed")

        reward_file = tmp_path / "reward.py"
        reward_file.write_text(
            "def simple_reward(completion, answer, prompt, **kwargs):\n    return 1.0\n"
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

        from agilerl.models.algorithms.grpo import GRPOSpec

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
        )
        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.REASONING,
            dataset_path=str(dataset_path),
            reward_file_path=str(reward_file),
            reward_fn_name="simple_reward",
            prompt_template={"user_0": "Solve: {question}"},
            data_batch_size_per_gpu=4,
        )
        algo_spec = GRPOSpec(
            pretrained_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            group_size=2,
            update_epochs=1,
            lora_config=lora_config,
            max_model_len=128,
        )

        mock_pop = [MagicMock()]

        with patch(
            "agilerl.training.trainer.create_population_from_spec",
            return_value=mock_pop,
        ):
            trainer = LocalTrainer(
                algorithm=algo_spec,
                environment=env_spec,
                training=self._training(pop_size=1, max_steps=8, evo_steps=4),
            )

        from agilerl.utils.llm_utils import ReasoningGym

        assert isinstance(trainer.env, ReasoningGym)
        assert trainer.tokenizer is not None
        assert trainer.tokenizer.pad_token is not None

        mock_fn = MagicMock(return_value=None)
        with patch.object(trainer, "train_fn", mock_fn):
            trainer.train()
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["pop"] is mock_pop
            assert isinstance(call_kwargs["env"], ReasoningGym)
            assert call_kwargs["max_steps"] == 8
            assert call_kwargs["evo_steps"] == 4
            assert "evaluation_interval" in call_kwargs

    # -- LLM: DPO (preference) via LocalTrainer ------------------------------

    @pytest.mark.llm
    def test_dpo_preference_train(self, tmp_path):
        """DPO preference: real LLMEnvSpec.make_env() with temp dataset files.

        Tests the full LocalTrainer wiring: spec → env construction → kwargs
        assembly → dispatch to finetune_llm_preference.  The finetune function
        itself is patched since running it requires CUDA agents.
        """
        try:
            from peft import LoraConfig
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

        from agilerl.models.algorithms.dpo import DPOSpec

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
        )
        env_spec = LLMEnvSpec(
            env_type=LLMEnvType.PREFERENCE,
            dataset_path=str(dataset_path),
            data_batch_size_per_gpu=4,
        )
        algo_spec = DPOSpec(
            pretrained_model_name_or_path="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            update_epochs=1,
            lora_config=lora_config,
            max_model_len=128,
        )

        mock_pop = [MagicMock()]

        with patch(
            "agilerl.training.trainer.create_population_from_spec",
            return_value=mock_pop,
        ):
            trainer = LocalTrainer(
                algorithm=algo_spec,
                environment=env_spec,
                training=self._training(pop_size=1, max_steps=8, evo_steps=4),
            )

        from agilerl.utils.llm_utils import PreferenceGym

        assert isinstance(trainer.env, PreferenceGym)
        assert trainer.tokenizer is not None

        mock_fn = MagicMock(return_value=None)
        with patch.object(trainer, "train_fn", mock_fn):
            trainer.train()
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["pop"] is mock_pop
            assert isinstance(call_kwargs["env"], PreferenceGym)
            assert call_kwargs["max_steps"] == 8
            assert "evaluation_interval" in call_kwargs


# ---------------------------------------------------------------------------
# String environment resolution
# ---------------------------------------------------------------------------


class TestStringEnvironmentResolution:
    """Verify that passing a plain string as the ``environment`` parameter
    produces the correct env spec and that the constructed environment
    corresponds to the requested gym / PettingZoo id."""

    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_gym_env_from_string(self, mock_create_pop, training_spec):
        """A string environment for a single-agent algo resolves to GymEnvSpec
        and the constructed env matches the requested id."""
        from agilerl.models.env import GymEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="DQN",
            environment="LunarLander-v3",
            training=training_spec,
        )
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert trainer.env_spec.name == "LunarLander-v3"
        assert trainer.env is not None
        assert hasattr(trainer.env, "single_observation_space")

    @patch("agilerl.training.trainer.LocalTrainer._make_env", return_value=MagicMock())
    @patch("agilerl.training.trainer.create_population_from_spec")
    def test_pz_env_from_string(self, mock_create_pop, _mock_make_env, training_spec):
        """A string environment for a multi-agent algo resolves to PzEnvSpec."""
        from agilerl.models.env import PzEnvSpec

        mock_create_pop.return_value = [MagicMock()]
        trainer = LocalTrainer(
            algorithm="MADDPG",
            environment="simple_spread_v3",
            training=training_spec,
        )
        assert isinstance(trainer.env_spec, PzEnvSpec)
        assert trainer.env_spec.name == "simple_spread_v3"

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
        _DPO, _ = _rebuild_llm_specs()
        spec = _DPO(**_LLM_COMMON_KWARGS)
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


# ---------------------------------------------------------------------------
# Trainer.get_validated_manifest — direct / dict
# ---------------------------------------------------------------------------


class TestGetValidatedManifest:
    def test_from_yaml_file(self):
        from agilerl.training.trainer import Trainer

        manifest = Trainer.get_validated_manifest("configs/training/ppo/ppo.yaml")
        assert manifest.algorithm.name == "PPO"
        assert manifest.training.max_steps == 6_000_000

    def test_from_dict(self):
        from agilerl.training.trainer import Trainer

        data = {
            "algorithm": {"name": "DQN", "learn_step": 1},
            "environment": {"name": "CartPole-v1", "num_envs": 1},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
        }
        manifest = Trainer.get_validated_manifest(data)
        assert manifest.algorithm.name == "DQN"
        assert manifest.training.max_steps == 100
        assert manifest.training.population_size == 2


# ---------------------------------------------------------------------------
# LocalTrainer.to_manifest
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LocalTrainer.train optional kwargs forwarding
# ---------------------------------------------------------------------------


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
            patch.object(PPOSpec, "get_training_fn", return_value=mock_train_fn),
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
