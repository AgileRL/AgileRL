# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for TrainingManifest parsing and Trainer.from_manifest().

Covers single-agent (on-policy, off-policy, offline, bandit), multi-agent,
and LLM training scenarios, along with all supported network architectures
(MLP, CNN, LSTM, SimBA, MultiInput).

Test manifests live under ``tests/manifests/`` as YAML files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from gymnasium import spaces
from pydantic import ValidationError

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.arena.models import GymEnvSpec as ArenaEnvSpec
from agilerl.arena.models.algorithms import (
    CQNSpec,
    DDPGSpec,
    DQNSpec,
    IPPOSpec,
    MADDPGSpec,
    MATD3Spec,
    NeuralTSSpec,
    NeuralUCBSpec,
    PPOSpec,
    RainbowDQNSpec,
    TD3Spec,
)
from agilerl.models.env import (
    BanditEnvSpec,
    GymEnvSpec,
    OfflineEnvSpec,
)
from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    MutationSpec,
    TournamentSelectionSpec,
)
from agilerl.models.manifest import TrainingManifest, from_trainer_specs
from agilerl.models.networks import (
    CnnSpec,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    SimbaSpec,
    StochasticActorSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.training.trainer import LocalTrainer

if HAS_LLM_DEPENDENCIES:
    from agilerl.arena.models.algorithms import DPOSpec, GRPOSpec
    from agilerl.models.env import LLMEnvSpec
else:
    # Placeholder names so ``@pytest.mark.parametrize`` decorators (evaluated at
    # class-body load time) don't NameError when LLM deps are absent.  The tests
    # using them are gated by ``@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES)``.
    DPOSpec = GRPOSpec = LLMEnvSpec = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Manifest loading helpers
# ---------------------------------------------------------------------------

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "training"

_TRAINING = {"max_steps": 1000, "evo_steps": 100, "pop_size": 2}


def _load(name: str) -> dict:
    """Load a YAML manifest from ``tests/manifests/<name>.yaml``."""
    with open(MANIFESTS_DIR / f"{name}.yaml") as fh:
        return yaml.safe_load(fh)


def _make_manifest(algo: dict, env: dict | None = None, **sections) -> dict:
    """Build an ad-hoc manifest dict (for one-off / error-case tests).

    The environment defaults to a valid gym section: it is now validated like
    every other section, so an empty one fails on the name a gym env requires
    rather than passing straight through.
    """
    m = {
        "algorithm": algo,
        "environment": env if env is not None else {"name": "CartPole-v1"},
        "training": sections.pop("training", _TRAINING),
    }
    m.update(sections)
    return m


def test_get_validated_json_resolves_unset_optional_sections() -> None:
    """The payload carries the sections the runtime indexes straight into.

    The runtime reads ``mutation["rl_hp_selection"]`` and
    ``tournament_selection["tournament_size"]`` without guarding them, so an
    omitted section has to arrive filled rather than empty or absent.
    """
    raw = {
        "algorithm": {
            "name": "DQN",
            "batch_size": 128,
            "lr": 0.00063,
            "tau": 0.001,
            "double": False,
        },
        "environment": {"name": "LunarLander-v3", "num_envs": 16},
        "training": {"max_steps": 1_000_000, "evo_steps": 10_000, "pop_size": 4},
    }
    out = TrainingManifest.get_validated(raw, mode="json")
    assert out["mutation"]["rl_hp_selection"] == {}
    assert out["mutation"]["probabilities"]["no_mut"] == 1.0
    assert out["mutation"]["probabilities"]["arch_mut"] == 0.0
    assert out["tournament_selection"]["tournament_size"] == 2
    assert out["tournament_selection"]["elitism"] is True
    # The encoder is inferred from the observation space, so there is no
    # architecture to fill in here.
    assert "network" not in out
    assert "selection_strategy" not in out


def test_recurrent_ppo_alias_sets_recurrent() -> None:
    raw = {
        "algorithm": {"name": "Recurrent PPO", "lr": 3e-4},
        "environment": {"name": "CartPole-v1", "num_envs": 8},
        "training": {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
    }
    validated = TrainingManifest.model_validate(raw)
    assert validated.algorithm.recurrent is True
    assert validated.algorithm.name == "Recurrent PPO"


def test_recurrent_ppo_rejects_an_explicit_false() -> None:
    with pytest.raises(ValueError, match="recurrent"):
        TrainingManifest.model_validate(
            {
                "algorithm": {
                    "name": "Recurrent PPO",
                    "recurrent": False,
                    "lr": 3e-4,
                },
                "environment": {"name": "CartPole-v1", "num_envs": 8},
                "training": {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
            }
        )


def test_get_validated_raises_on_unknown_algorithm_field() -> None:
    with pytest.raises(ValueError, match=r"algorithm\.bogus_algo_field"):
        TrainingManifest.get_validated(
            _make_manifest(
                {"name": "DQN", "lr": 3e-4, "bogus_algo_field": 1},
                env={"name": "CartPole-v1"},
            )
        )


def test_get_validated_raises_on_unknown_top_level_field() -> None:
    with pytest.raises(ValueError, match="bogus_top_level"):
        TrainingManifest.get_validated(
            _make_manifest(
                {"name": "DQN"}, env={"name": "CartPole-v1"}, bogus_top_level=123
            )
        )


def test_get_validated_accepts_known_aliased_or_excluded_fields() -> None:
    # `cudagraphs` is declared-but-excluded, and population_size / metrics_interval
    # / memory_size are validation aliases: none are unknown.
    TrainingManifest.get_validated(
        _make_manifest(
            {"name": "DQN", "lr": 3e-4, "cudagraphs": True},
            env={"name": "CartPole-v1"},
            training={
                "population_size": 4,
                "metrics_interval": 123,
                "max_steps": 1000,
            },
            replay_buffer={"memory_size": 4096},
        )
    )


def test_get_validated_accepts_the_tournament_selection_alias() -> None:
    TrainingManifest.get_validated(
        _make_manifest(
            {"name": "DQN"},
            env={"name": "CartPole-v1"},
            tournament_selection={"tournament_size": 3},
        )
    )


def test_get_validated_reports_unknown_keys_under_the_tournament_selection_alias() -> (
    None
):
    # The error names the key the manifest actually wrote, so the author can
    # find it; pydantic reports under whichever alias the document used.
    with pytest.raises(
        ValueError, match=r"tournament_selection\.tournament\.bogus_key"
    ):
        TrainingManifest.get_validated(
            _make_manifest(
                {"name": "DQN"},
                env={"name": "CartPole-v1"},
                tournament_selection={"tournament_size": 3, "bogus_key": 1},
            )
        )


def test_get_validated_reports_unknown_keys_under_the_current_selection_key() -> None:
    with pytest.raises(ValueError, match=r"selection_strategy\.tournament\.bogus_key"):
        TrainingManifest.get_validated(
            _make_manifest(
                {"name": "DQN"},
                env={"name": "CartPole-v1"},
                selection_strategy={"tournament_size": 3, "bogus_key": 1},
            )
        )


@pytest.mark.parametrize("key", ["tournament_selection", "selection_strategy"])
def test_get_validated_json_writes_the_serialized_selection_key(key: str) -> None:
    """Either spelling normalizes onto the one key the runtime indexes.

    ``selection_strategy`` is the field name; ``tournament_selection`` is its
    serialization alias, and SpecAssembler reads the payload with that name and
    no fallback, so the payload has to carry it.
    """
    out = TrainingManifest.get_validated(
        _make_manifest(
            {"name": "DQN"},
            env={"name": "CartPole-v1"},
            **{key: {"tournament_size": 3}},
        ),
        mode="json",
    )
    assert out["tournament_selection"]["tournament_size"] == 3
    assert "selection_strategy" not in out


@pytest.mark.parametrize("key", ["selection_strategy", "tournament_selection"])
def test_selection_strategy_accepts_either_manifest_key(key: str) -> None:
    manifest = TrainingManifest.get_validated(
        _make_manifest(
            {"name": "DQN"},
            env={"name": "CartPole-v1"},
            **{key: {"tournament_size": 3}},
        ),
        mode="python",
    )
    assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)
    assert manifest.selection_strategy.tournament_size == 3


def test_selection_strategy_is_none_when_unset() -> None:
    manifest = TrainingManifest.get_validated(
        _make_manifest({"name": "DQN"}, env={"name": "CartPole-v1"}), mode="python"
    )
    assert manifest.selection_strategy is None


def test_selection_strategy_is_multi_frequency_when_configured() -> None:
    manifest = TrainingManifest.get_validated(
        _make_manifest(
            {"name": "DQN"},
            env={"name": "CartPole-v1"},
            training={"max_steps": 1000, "evo_steps": 100, "pop_size": 6},
            selection_strategy={"strategy": "multi_frequency"},
        ),
        mode="python",
    )
    assert isinstance(manifest.selection_strategy, MultiFrequencySelectionSpec)


def test_selection_strategy_serializes_as_tournament_selection() -> None:
    manifest = TrainingManifest.get_validated(
        _make_manifest(
            {"name": "DQN"},
            env={"name": "CartPole-v1"},
            selection_strategy={"tournament_size": 3},
        ),
        mode="python",
    )
    assert "tournament_selection" not in TrainingManifest.model_computed_fields
    dumped = manifest.model_dump(exclude_none=True, by_alias=True)
    assert dumped["tournament_selection"]["tournament_size"] == 3


class TestNormalizeNetworkForPlatform:
    """Tests for network normalization via get_validated(mode='json')."""

    def test_arch_is_carried_on_the_encoder_not_the_top_level(self):
        # arch picks the encoder spec, and normalization moves it onto the
        # encoder it selected. The dump must keep arch on the encoder so a
        # later validate does not infer a different one.
        raw = {
            "algorithm": {"name": "DQN"},
            "environment": {"name": "CartPole-v1"},
            "training": _TRAINING,
            "network": {
                "latent_dim": 64,
                "arch": "mlp",
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        network = out["network"]
        assert "arch" not in network
        assert network["encoder_config"]["arch"] == "mlp"
        assert network["encoder_config"]["hidden_size"] == [64]

    def test_name_field_not_added(self):
        raw = {
            "algorithm": {"name": "DQN"},
            "environment": {"name": "CartPole-v1"},
            "training": _TRAINING,
            "network": {
                "latent_dim": 64,
                "arch": "mlp",
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        network = out["network"]
        assert "name" not in network

    def test_default_fields_filled_in(self):
        raw = {
            "algorithm": {"name": "DQN"},
            "environment": {"name": "CartPole-v1"},
            "training": _TRAINING,
            "network": {
                "latent_dim": 64,
                "arch": "mlp",
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        encoder = out["network"]["encoder_config"]
        assert "layer_norm" in encoder
        assert "init_layers" in encoder
        assert "output_vanish" in encoder


# ---------------------------------------------------------------------------
# Load all test manifests at module level
# ---------------------------------------------------------------------------

DQN_MANIFEST = _load("dqn")
RAINBOW_MANIFEST = _load("rainbow_dqn")
PPO_MANIFEST = _load("ppo")
PPO_CNN_MANIFEST = _load("ppo_cnn")
PPO_LSTM_MANIFEST = _load("ppo_lstm")
DDPG_MANIFEST = _load("ddpg")
DDPG_SIMBA_MANIFEST = _load("ddpg_simba")
TD3_MANIFEST = _load("td3")
CQN_MANIFEST = _load("cqn")
NEURAL_TS_MANIFEST = _load("neural_ts")
NEURAL_UCB_MANIFEST = _load("neural_ucb")
MULTI_INPUT_MANIFEST = _load("multi_input")
MADDPG_MANIFEST = _load("maddpg")
MATD3_MANIFEST = _load("matd3")
IPPO_MANIFEST = _load("ippo")
IPPO_CNN_MANIFEST = _load("ippo_cnn")


# ============================================================================
# TestTrainingManifest - manifest parsing and validation
# ============================================================================


class TestTrainingManifest:
    """Tests for the ``TrainingManifest`` Pydantic model."""

    # -- Algorithm dispatch -------------------------------------------------

    @pytest.mark.parametrize(
        ("name", "expected_cls"),
        [
            ("DQN", DQNSpec),
            ("PPO", PPOSpec),
            ("DDPG", DDPGSpec),
            ("TD3", TD3Spec),
            ("CQN", CQNSpec),
            ("RainbowDQN", RainbowDQNSpec),
            ("NeuralTS", NeuralTSSpec),
            ("NeuralUCB", NeuralUCBSpec),
            ("IPPO", IPPOSpec),
            ("MADDPG", MADDPGSpec),
            ("MATD3", MATD3Spec),
        ],
    )
    def test_algorithm_dispatch_by_name(self, name, expected_cls):
        data = _make_manifest(algo={"name": name})
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.algorithm, expected_cls)

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    @pytest.mark.parametrize(
        ("name", "extra_fields", "expected_cls"),
        [
            ("GRPO", {"group_size": 6, "temperature": 0.9}, GRPOSpec),
            ("DPO", {}, DPOSpec),
        ],
    )
    def test_algorithm_dispatch_llm(self, name, extra_fields, expected_cls):
        algo = {
            "name": name,
            "update_epochs": 1,
            "lora_config": {
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "task_type": "CAUSAL_LM",
            },
            "max_model_len": 512,
            "use_separate_reference_adapter": True,
            "pretrained_model_name_or_path": "test-model",
            "calc_position_embeddings": True,
            **extra_fields,
        }
        env = {"dataset": "d.parquet"}
        if name == "GRPO":
            # A rollout over rows needs its rubric; a dataset env rejects one.
            env.update(
                reward_file_path="reward.py",
                prompt_template={"user_0": "{question}"},
            )
        data = _make_manifest(algo=algo, env=env)
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.algorithm, expected_cls)

    def test_algorithm_missing_name_raises(self):
        data = _make_manifest(algo={"batch_size": 64})
        with pytest.raises(Exception, match="name"):
            TrainingManifest.model_validate(data)

    def test_algorithm_unknown_name_raises(self):
        data = _make_manifest(algo={"name": "NonExistentAlgo"})
        with pytest.raises(ValueError, match="NonExistentAlgo"):
            TrainingManifest.model_validate(data)

    # -- Network architecture injection -------------------------------------

    def test_network_missing_arch_defers_resolution(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            network={
                "latent_dim": 64,
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        )
        manifest = TrainingManifest.model_validate(data)
        # Deferred: no arch declared, so the network section stays untyped and
        # algorithm.net_config is unset until the trainer infers the encoder.
        assert not manifest.network_arch_is_resolvable
        assert manifest.algorithm.net_config is None

    @pytest.mark.parametrize(
        ("arch", "encoder_kwargs", "expected_encoder_cls"),
        [
            ("mlp", {"hidden_size": [64]}, MlpSpec),
            (
                "cnn",
                {"channel_size": [32], "kernel_size": [3], "stride_size": [1]},
                CnnSpec,
            ),
            ("lstm", {"hidden_state_size": 64, "num_layers": 1}, LstmSpec),
            (
                "simba",
                {"hidden_size": 128, "num_blocks": 2},
                SimbaSpec,
            ),
            (
                "multiinput",
                {"latent_dim": 32, "mlp_config": {"hidden_size": [32]}},
                MultiInputSpec,
            ),
        ],
    )
    def test_network_arch_injection(self, arch, encoder_kwargs, expected_encoder_cls):
        data = _make_manifest(
            algo={"name": "DQN"},
            network={
                "latent_dim": 64,
                "arch": arch,
                "encoder_config": encoder_kwargs,
                "head_config": {"hidden_size": [64]},
            },
        )
        manifest = TrainingManifest.model_validate(data)
        # arch selects the encoder spec rather than surviving as data: the
        # section is validated now, not carried as a raw dict.
        assert isinstance(manifest.network.encoder_config, expected_encoder_cls)
        assert manifest.network.encoder_config.arch == arch

    def test_simba_convenience_flag_stripped(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            network={
                "latent_dim": 64,
                "arch": "mlp",
                "simba": False,
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.network.encoder_config.arch == "mlp"

    # -- Field aliases ------------------------------------------------------

    def test_pop_size_alias(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            training={"max_steps": 100, "evo_steps": 10, "pop_size": 8},
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.training.pop_size == 8

    def test_memory_size_alias(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            replay_buffer={"memory_size": 50_000},
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.replay_buffer.max_size == 50_000

    # -- Optional sections --------------------------------------------------

    def test_optional_sections_default_to_none(self):
        data = _make_manifest(algo={"name": "DQN"})
        manifest = TrainingManifest.model_validate(data)
        assert manifest.network is None
        assert manifest.mutation is None
        assert manifest.replay_buffer is None
        assert manifest.selection_strategy is None

    def test_environment_required(self):
        data = {"algorithm": {"name": "DQN"}, "training": _TRAINING}
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(data)

    def test_training_section_optional(self):
        data = {
            "algorithm": {"name": "DQN"},
            "environment": {"name": "CartPole-v1"},
        }
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.training, TrainingSpec)
        assert manifest.training.max_steps == 1_000_000
        assert manifest.training.pop_size == 1
        assert manifest.training.hpo is True

    def test_environment_is_validated_not_passed_through(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1", "num_envs": 4, "custom": False},
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.environment.name == "CartPole-v1"
        assert manifest.environment.num_envs == 4

    # -- Full manifest parsing ----------------------------------------------

    def test_parse_full_dqn_manifest(self):
        manifest = TrainingManifest.model_validate(DQN_MANIFEST)

        assert isinstance(manifest.algorithm, DQNSpec)
        assert manifest.algorithm.batch_size == 128
        assert manifest.algorithm.tau == 0.001
        assert manifest.algorithm.double is False
        assert manifest.algorithm.lr == pytest.approx(6.3e-4)

        assert isinstance(manifest.mutation, MutationSpec)
        assert manifest.network.encoder_config.arch == "mlp"
        assert manifest.replay_buffer is not None
        assert manifest.replay_buffer.max_size == 100_000
        assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)
        assert isinstance(manifest.training, TrainingSpec)
        assert manifest.training.pop_size == 4
        assert manifest.training.eps_start == 1.0
        assert manifest.training.eps_end == 0.1


# ============================================================================
# TestLocalTrainerSingleAgent - Gymnasium (single-agent) scenarios
# ============================================================================


class TestLocalTrainerSingleAgent:
    """``LocalTrainer.from_manifest()`` for single-agent RL algorithms."""

    @pytest.fixture(autouse=True)
    def _patch_heavy_init(self):
        """Avoid spawning real AsyncVectorEnv subprocesses and building
        real neural-network populations; these tests only verify manifest
        parsing and trainer construction.
        """
        with (
            patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
        ):
            yield

    # -- Parametrized: all single-agent algorithms --------------------------

    @pytest.mark.parametrize(
        ("manifest", "expected_algo_cls", "expected_encoder_cls"),
        [
            (DQN_MANIFEST, DQNSpec, MlpSpec),
            (RAINBOW_MANIFEST, RainbowDQNSpec, MlpSpec),
            (PPO_MANIFEST, PPOSpec, MlpSpec),
            (PPO_CNN_MANIFEST, PPOSpec, CnnSpec),
            (PPO_LSTM_MANIFEST, PPOSpec, LstmSpec),
            (DDPG_MANIFEST, DDPGSpec, MlpSpec),
            (DDPG_SIMBA_MANIFEST, DDPGSpec, SimbaSpec),
            (TD3_MANIFEST, TD3Spec, MlpSpec),
            (MULTI_INPUT_MANIFEST, PPOSpec, MultiInputSpec),
        ],
        ids=[
            "DQN-MLP",
            "RainbowDQN-MLP",
            "PPO-MLP",
            "PPO-CNN",
            "PPO-LSTM",
            "DDPG-MLP",
            "DDPG-SimBA",
            "TD3-MLP",
            "PPO-MultiInput",
        ],
    )
    def test_from_manifest_creates_correct_types(
        self, manifest, expected_algo_cls, expected_encoder_cls
    ):
        trainer = LocalTrainer.from_manifest(manifest)

        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)

        assert trainer.algorithm_spec.net_config is not None
        assert isinstance(trainer.algorithm_spec.net_config, NetworkSpec)
        assert isinstance(
            trainer.algorithm_spec.net_config.encoder_config, expected_encoder_cls
        )

    @pytest.mark.parametrize(
        ("manifest", "expected_algo_cls"),
        [
            (NEURAL_TS_MANIFEST, NeuralTSSpec),
            (NEURAL_UCB_MANIFEST, NeuralUCBSpec),
        ],
        ids=["NeuralTS-MLP", "NeuralUCB-MLP"],
    )
    def test_bandit_from_manifest_creates_correct_types(
        self, manifest, expected_algo_cls
    ):
        trainer = LocalTrainer.from_manifest(manifest)

        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, BanditEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)

        assert trainer.algorithm_spec.net_config is not None
        assert isinstance(trainer.algorithm_spec.net_config, NetworkSpec)
        assert isinstance(trainer.algorithm_spec.net_config.encoder_config, MlpSpec)

    def test_offline_manifest_creates_offline_env_spec(self):
        trainer = LocalTrainer.from_manifest(CQN_MANIFEST)

        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, CQNSpec)
        assert isinstance(trainer.env_spec, OfflineEnvSpec)
        assert trainer.env_spec.minari_dataset_id == "cartpole-v0"
        assert isinstance(trainer.training_spec, TrainingSpec)
        assert trainer.algorithm_spec.net_config is not None
        assert isinstance(trainer.algorithm_spec.net_config.encoder_config, MlpSpec)

    # -- Algorithm-specific field checks ------------------------------------

    def test_dqn_off_policy_fields(self):
        trainer = LocalTrainer.from_manifest(DQN_MANIFEST)
        assert trainer.algorithm_spec.tau == 0.001
        assert trainer.algorithm_spec.double is False
        assert trainer.algorithm_spec.lr == pytest.approx(6.3e-4)
        assert trainer.replay_buffer_spec is not None
        assert trainer.replay_buffer_spec.max_size == 100_000
        assert trainer.training_spec.eps_start == 1.0
        assert trainer.training_spec.eps_end == 0.1

    def test_rainbow_dqn_fields(self):
        trainer = LocalTrainer.from_manifest(RAINBOW_MANIFEST)
        assert trainer.algorithm_spec.n_step == 4
        assert trainer.algorithm_spec.num_atoms == 51
        assert trainer.algorithm_spec.v_min == -200.0
        assert trainer.algorithm_spec.v_max == 200.0
        assert trainer.algorithm_spec.noise_std == 0.5

    def test_ppo_on_policy_fields(self):
        trainer = LocalTrainer.from_manifest(PPO_MANIFEST)
        assert trainer.algorithm_spec.gae_lambda == 0.95
        assert trainer.algorithm_spec.clip_coef == 0.2
        assert trainer.algorithm_spec.ent_coef == 0.01
        assert trainer.algorithm_spec.update_epochs == 4
        assert trainer.algorithm_spec.share_encoders is True
        assert trainer.replay_buffer_spec is None

    def test_ppo_recurrent_flag(self):
        trainer = LocalTrainer.from_manifest(PPO_LSTM_MANIFEST)
        assert trainer.algorithm_spec.recurrent is True

    def test_ddpg_actor_critic_lrs(self):
        trainer = LocalTrainer.from_manifest(DDPG_MANIFEST)
        assert trainer.algorithm_spec.lr_actor == 0.0003
        assert trainer.algorithm_spec.lr_critic == 0.0003
        assert trainer.algorithm_spec.O_U_noise is True
        assert trainer.algorithm_spec.share_encoders is True

    def test_td3_policy_freq(self):
        trainer = LocalTrainer.from_manifest(TD3_MANIFEST)
        assert trainer.algorithm_spec.policy_freq == 2

    def test_cqn_offline_fields(self):
        trainer = LocalTrainer.from_manifest(CQN_MANIFEST)
        assert trainer.algorithm_spec.double is True

    def test_bandit_fields(self):
        trainer = LocalTrainer.from_manifest(NEURAL_TS_MANIFEST)
        assert trainer.algorithm_spec.lamb == 1.0
        assert trainer.algorithm_spec.reg == pytest.approx(0.00625)
        assert isinstance(trainer.env_spec, BanditEnvSpec)
        assert trainer.env_spec.name == "IRIS"

    # -- Environment spec construction --------------------------------------

    def test_env_name_and_num_envs(self):
        trainer = LocalTrainer.from_manifest(DQN_MANIFEST)
        assert trainer.env_spec.name == "LunarLander-v3"
        assert trainer.env_spec.num_envs == 16

    # -- Network injection into algo spec -----------------------------------

    def test_network_injected_into_algo_net_config(self):
        trainer = LocalTrainer.from_manifest(DQN_MANIFEST)
        assert trainer.algorithm_spec.net_config is not None
        assert trainer.algorithm_spec.net_config.latent_dim == 128

    def test_no_network_section_leaves_net_config_none(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1"},
        )
        trainer = LocalTrainer.from_manifest(data)
        assert trainer.algorithm_spec.net_config is None

    # -- Mutation / tournament / replay buffer passthrough -------------------

    def test_mutation_and_tournament_forwarded(self):
        trainer = LocalTrainer.from_manifest(DQN_MANIFEST)
        assert isinstance(trainer.mutation_spec, MutationSpec)
        assert isinstance(trainer.selection_strategy_spec, TournamentSelectionSpec)

    def test_replay_buffer_forwarded(self):
        trainer = LocalTrainer.from_manifest(DQN_MANIFEST)
        assert isinstance(trainer.replay_buffer_spec, ReplayBufferSpec)
        assert trainer.replay_buffer_spec.max_size == 100_000

    def test_optional_sections_none_when_omitted(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1"},
        )
        trainer = LocalTrainer.from_manifest(data)
        assert trainer.mutation_spec is None
        assert trainer.selection_strategy_spec is None
        assert trainer.replay_buffer_spec is None

    # -- Loading from files -------------------------------------------------

    def test_from_yaml_file(self):
        trainer = LocalTrainer.from_manifest(MANIFESTS_DIR / "dqn.yaml")
        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, DQNSpec)
        assert trainer.env_spec.name == "LunarLander-v3"

    def test_from_string_path(self):
        trainer = LocalTrainer.from_manifest(str(MANIFESTS_DIR / "ppo.yaml"))
        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, PPOSpec)

    # -- Extra kwargs -------------------------------------------------------

    def test_device_kwarg_forwarded(self):
        with patch.object(LocalTrainer, "__init__", return_value=None) as init:
            LocalTrainer.from_manifest(DQN_MANIFEST, device="cuda")
        assert init.call_args.kwargs["device"] == "cuda"

    def test_hp_config_kwarg_forwarded(self):
        hp_config = object()
        with patch.object(LocalTrainer, "__init__", return_value=None) as init:
            LocalTrainer.from_manifest(DQN_MANIFEST, hp_config=hp_config)
        assert init.call_args.kwargs["hp_config"] is hp_config

    # -- Minimal manifest ---------------------------------------------------

    def test_minimal_manifest(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1", "num_envs": 1},
        )
        trainer = LocalTrainer.from_manifest(data)
        assert isinstance(trainer, LocalTrainer)
        assert trainer.algorithm_spec.batch_size == 64  # DQN's constructor default
        assert trainer.env_spec.name == "CartPole-v1"
        assert trainer.mutation_spec is None
        assert trainer.selection_strategy_spec is None
        assert trainer.replay_buffer_spec is None


# ============================================================================
# TestLocalTrainerMultiAgent - PettingZoo (multi-agent) scenarios
# ============================================================================


class TestLocalTrainerMultiAgent:
    """``LocalTrainer.from_manifest()`` for multi-agent RL algorithms."""

    @pytest.fixture(autouse=True)
    def _patch_heavy_init(self):
        """Avoid spawning real AsyncPettingZooVecEnv subprocesses."""
        with (
            patch.object(LocalTrainer, "_make_env", return_value=MagicMock()),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
        ):
            yield

    @pytest.mark.parametrize(
        ("manifest", "expected_algo_cls"),
        [
            (MADDPG_MANIFEST, MADDPGSpec),
            (MATD3_MANIFEST, MATD3Spec),
            (IPPO_MANIFEST, IPPOSpec),
            (IPPO_CNN_MANIFEST, IPPOSpec),
        ],
        ids=["MADDPG", "MATD3", "IPPO-MLP", "IPPO-CNN"],
    )
    def test_multi_agent_produces_pz_env_spec(self, manifest, expected_algo_cls):
        trainer = LocalTrainer.from_manifest(manifest)

        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)

    def test_maddpg_env_name(self):
        trainer = LocalTrainer.from_manifest(MADDPG_MANIFEST)
        assert trainer.env_spec.name == "mpe2.simple_speaker_listener_v4"
        assert trainer.env_spec.num_envs == 16

    def test_maddpg_actor_critic_lrs(self):
        trainer = LocalTrainer.from_manifest(MADDPG_MANIFEST)
        assert trainer.algorithm_spec.lr_actor == 0.0001
        assert trainer.algorithm_spec.lr_critic == 0.001
        assert trainer.algorithm_spec.O_U_noise is True

    def test_matd3_policy_freq(self):
        trainer = LocalTrainer.from_manifest(MATD3_MANIFEST)
        assert trainer.algorithm_spec.policy_freq == 2

    def test_ippo_on_policy_fields(self):
        trainer = LocalTrainer.from_manifest(IPPO_MANIFEST)
        assert trainer.algorithm_spec.gae_lambda == 0.95
        assert trainer.algorithm_spec.clip_coef == 0.2
        assert trainer.algorithm_spec.ent_coef == 0.05

    def test_ippo_cnn_encoder(self):
        trainer = LocalTrainer.from_manifest(IPPO_CNN_MANIFEST)
        assert isinstance(trainer.algorithm_spec.net_config.encoder_config, CnnSpec)
        encoder = trainer.algorithm_spec.net_config.encoder_config
        assert encoder.channel_size == [64, 64, 32]

    def test_multi_agent_network_injection(self):
        trainer = LocalTrainer.from_manifest(MADDPG_MANIFEST)
        assert trainer.algorithm_spec.net_config is not None
        assert trainer.algorithm_spec.net_config.latent_dim == 64

    def test_multi_agent_replay_buffer(self):
        trainer = LocalTrainer.from_manifest(MADDPG_MANIFEST)
        assert isinstance(trainer.replay_buffer_spec, ReplayBufferSpec)
        assert trainer.replay_buffer_spec.max_size == 100_000


# ============================================================================
# TrainingManifest - Arena bridge helpers
# ============================================================================


class TestTrainingManifestArenaBridge:
    """``from_trainer_specs`` and ``get_validated`` / ``to_payload``."""

    def test_from_trainer_specs_builds_core_manifest(self):
        ppo_spec = PPOSpec(
            learn_step=128,
            net_config=StochasticActorSpec(
                encoder_config=MlpSpec(hidden_size=[64]),
                head_config=MlpSpec(hidden_size=[64]),
            ),
        )
        manifest = from_trainer_specs(
            algorithm=ppo_spec,
            environment=GymEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=500),
        )
        assert isinstance(manifest.algorithm, PPOSpec)
        assert manifest.algorithm.net_config is not None

    def test_from_trainer_specs_accepts_arena_env_spec(self):
        ppo_spec = PPOSpec(learn_step=128)
        manifest = from_trainer_specs(
            algorithm=ppo_spec,
            environment=ArenaEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=500),
        )
        assert manifest.environment.name == "CartPole-v1"
        assert isinstance(manifest.algorithm, PPOSpec)

    def test_from_trainer_specs_passthrough_plain_dict_mutation(self):
        """Plain dict mutation hits the _coerce passthrough branch."""
        manifest = from_trainer_specs(
            algorithm=PPOSpec(learn_step=64),
            environment=ArenaEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=200),
            mutation={"mutation_sd": 0.05},
        )
        assert manifest.mutation is not None
        assert manifest.mutation.mutation_sd == 0.05

    def test_from_trainer_specs_accepts_selection_strategy(self):
        manifest = from_trainer_specs(
            algorithm=PPOSpec(learn_step=64),
            environment=GymEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=200),
            selection_strategy=TournamentSelectionSpec(tournament_size=3),
        )
        assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)
        assert manifest.selection_strategy.tournament_size == 3

    def test_from_trainer_specs_accepts_multi_frequency_selection_strategy(self):
        """Both regimes land in the one discriminated field."""
        manifest = from_trainer_specs(
            algorithm=PPOSpec(learn_step=64),
            environment=GymEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=200, pop_size=16),
            selection_strategy=MultiFrequencySelectionSpec(n_subpopulations=2),
        )
        assert isinstance(manifest.selection_strategy, MultiFrequencySelectionSpec)
        assert manifest.selection_strategy.strategy == "multi_frequency"

    def test_from_trainer_specs_deprecated_tournament_selection_kwarg_warns(self):
        spec = TournamentSelectionSpec(tournament_size=3)
        with pytest.warns(DeprecationWarning, match="'tournament_selection' argument"):
            manifest = from_trainer_specs(
                algorithm=PPOSpec(learn_step=64),
                environment=GymEnvSpec(name="CartPole-v1"),
                training=TrainingSpec(max_steps=200),
                tournament_selection=spec,
            )
        assert manifest.selection_strategy == spec

    def test_from_trainer_specs_conflicting_selection_kwargs_raise(self):
        with (
            pytest.warns(DeprecationWarning, match="'tournament_selection' argument"),
            pytest.raises(ValueError, match="conflicting selection strategies"),
        ):
            from_trainer_specs(
                algorithm=PPOSpec(learn_step=64),
                environment=GymEnvSpec(name="CartPole-v1"),
                training=TrainingSpec(max_steps=200),
                selection_strategy=TournamentSelectionSpec(tournament_size=2),
                tournament_selection=TournamentSelectionSpec(tournament_size=3),
            )

    def test_from_trainer_specs_unknown_kwarg_raises_type_error(self):
        with pytest.raises(TypeError, match="tournament_seletcion"):
            from_trainer_specs(
                algorithm=PPOSpec(learn_step=64),
                environment=GymEnvSpec(name="CartPole-v1"),
                training=TrainingSpec(max_steps=200),
                tournament_seletcion=TournamentSelectionSpec(),
            )

    def test_from_trainer_specs_coerces_a_foreign_tournament_spec(self):
        from agilerl.arena.models import (
            TournamentSelectionSpec as ArenaTournamentSelectionSpec,
        )

        manifest = from_trainer_specs(
            algorithm=PPOSpec(learn_step=64),
            environment=ArenaEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=200),
            selection_strategy=ArenaTournamentSelectionSpec(tournament_size=4),
        )
        assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)
        assert manifest.selection_strategy.tournament_size == 4

    def test_get_validated_from_dict_payload(self):
        data = {
            "algorithm": {"name": "PPO", "learn_step": 128},
            "environment": {"name": "CartPole-v1", "num_envs": 4},
            "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
            "network": {
                "encoder_config": {"arch": "mlp", "hidden_size": [64]},
                "head_config": {"arch": "mlp", "hidden_size": [64]},
            },
        }
        submission = TrainingManifest.get_validated(data)
        assert submission["algorithm"]["name"] == "PPO"
        assert submission["network"]["encoder_config"]["hidden_size"] == [64]

    def test_to_payload_from_trainer_specs_manifest(self):
        core = from_trainer_specs(
            algorithm=PPOSpec(learn_step=64),
            environment=ArenaEnvSpec(name="CartPole-v1"),
            training=TrainingSpec(max_steps=200),
        )
        submission = core.to_payload()
        assert submission["algorithm"]["name"] == "PPO"
        assert submission["environment"]["name"] == "CartPole-v1"

    def test_get_validated_from_yaml_path(self, tmp_path):
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "algorithm": {"name": "PPO", "learn_step": 32},
                    "environment": {"name": "CartPole-v1", "num_envs": 2},
                    "training": {"max_steps": 100, "evo_steps": 50, "pop_size": 2},
                    "network": {
                        "encoder_config": {"arch": "mlp", "hidden_size": [32]},
                        "head_config": {"arch": "mlp", "hidden_size": [32]},
                    },
                }
            )
        )
        submission = TrainingManifest.get_validated(manifest_path)
        assert submission["algorithm"]["learn_step"] == 32
        assert submission["environment"]["num_envs"] == 2


# ============================================================================
# Trainer.get_validated_manifest - LLM ``network`` section
# ============================================================================


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestManifestBatchSizing:
    """Both batch-sizing fields reach the algorithm from the manifest.

    An unknown algorithm key is rejected outright, so a manifest cannot set an
    optimizer-step size the spec does not carry.
    """

    @staticmethod
    def _manifest(**algo_overrides):
        return _make_manifest(
            algo={
                "name": "GRPO",
                "batch_size": 8,
                "group_size": 4,
                **algo_overrides,
            },
            env={
                "env_type": "rollout",
                "dataset": "d.parquet",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            network={
                "pretrained_model_name_or_path": "net-model",
                "max_context_length": 256,
                "lora_config": {
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
        )

    def test_both_batch_fields_survive_the_manifest(self):
        manifest = TrainingManifest.get_validated(
            self._manifest(micro_batch_size_per_gpu=2, mini_batch_size=4),
            mode="python",
        )
        assert manifest.algorithm.micro_batch_size_per_gpu == 2
        assert manifest.algorithm.mini_batch_size == 4

    def test_an_unset_mini_batch_size_stays_none(self):
        """``None`` lets the algorithm apply its own family default."""
        manifest = TrainingManifest.get_validated(self._manifest(), mode="python")
        assert manifest.algorithm.mini_batch_size is None


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestTrainerGetValidatedManifestLLMNetwork:
    """``network`` fills LLM algorithm model fields after manifest parse."""

    def test_network_fills_pretrained_and_lora_when_absent_in_algorithm(self):
        data = _make_manifest(
            algo={"name": "DPO", "batch_size": 8},
            env={
                "env_type": "dataset",
                "objective": "preference",
                "dataset": "d.parquet",
                "columns": {"prompt": "p", "chosen": "c"},
            },
            network={
                "pretrained_model_name_or_path": "net-model",
                "max_context_length": 256,
                "lora_config": {
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "v_proj"],
                },
            },
        )
        manifest = TrainingManifest.get_validated(data, mode="python")
        assert isinstance(manifest.algorithm, DPOSpec)
        assert manifest.algorithm.pretrained_model_name_or_path == "net-model"
        assert manifest.algorithm.max_model_len == 256
        assert manifest.algorithm.lora_config is not None
        assert manifest.algorithm.lora_config.lora_r == 8
        assert manifest.algorithm.lora_config.lora_alpha == 16

    def test_disagreeing_network_and_algorithm_model_ids_are_an_error(self):
        data = _make_manifest(
            algo={
                "name": "DPO",
                "batch_size": 8,
                "pretrained_model_name_or_path": "algo-model",
                "max_model_len": 128,
                "lora_config": {
                    "lora_r": 2,
                    "lora_alpha": 4,
                    "target_modules": ["k_proj"],
                    "task_type": "CAUSAL_LM",
                },
            },
            env={
                "env_type": "dataset",
                "objective": "preference",
                "dataset": "d.parquet",
                "columns": {"prompt": "p", "chosen": "c"},
            },
            network={
                "pretrained_model_name_or_path": "network-model",
                "max_context_length": 512,
                "lora_config": {
                    "lora_r": 32,
                    "lora_alpha": 64,
                    "target_modules": ["o_proj"],
                },
            },
        )
        with pytest.raises(ValueError, match="pretrained_model_name_or_path"):
            TrainingManifest.get_validated(data, mode="python")

    def test_missing_pretrained_after_merge_raises(self):
        data = _make_manifest(
            algo={"name": "DPO", "batch_size": 8},
            env={
                "env_type": "dataset",
                "objective": "preference",
                "dataset": "d.parquet",
                "columns": {"prompt": "p", "chosen": "c"},
            },
        )
        with pytest.raises(ValueError, match="pretrained_model_name_or_path"):
            TrainingManifest.get_validated(data, mode="python")


# ============================================================================
# TestLocalTrainerLLM - LLM fine-tuning scenarios
# ============================================================================


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestLocalTrainerLLM:
    """``LocalTrainer.from_manifest()`` for LLM algorithms (GRPO, DPO)."""

    @staticmethod
    def _llm_algo_base() -> dict:
        return {
            "update_epochs": 1,
            "lora_config": {
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "task_type": "CAUSAL_LM",
            },
            "max_model_len": 512,
            # Below max_model_len, so rollout prompts get a non-zero budget
            # (GRPOSpec's 1024 default would leave none at this context size).
            "max_output_tokens": 128,
            "use_separate_reference_adapter": True,
            "pretrained_model_name_or_path": "test-model",
            "calc_position_embeddings": True,
            "max_grad_norm": 0.1,
        }

    @classmethod
    def _grpo_manifest(cls) -> dict:
        """GRPO manifest with inline environment section."""
        algo = {
            "name": "GRPO",
            "batch_size": 16,
            "beta": 0.001,
            "lr": 0.000005,
            "clip_coef": 0.2,
            "group_size": 6,
            "temperature": 0.9,
            **cls._llm_algo_base(),
        }
        return _make_manifest(
            algo=algo,
            env={
                "dataset": "train.parquet",
                "rubric_file_path": "reward.py",
                "rubric_name": "combined_rewards",
                "prompt_template": {
                    "system_0": "You are a helpful assistant.",
                    "user_1": "Solve {question} to get {answer}.",
                },
                "max_reward": 10.0,
                "train_test_split": 0.8,
            },
            mutation={
                "probabilities": {"no_mut": 0.1, "rl_hp_mut": 0.6},
                "mutation_sd": 0.1,
            },
        )

    @classmethod
    def _dpo_manifest(cls) -> dict:
        """DPO manifest with inline environment section."""
        algo = {
            "name": "DPO",
            "batch_size": 16,
            "beta": 0.001,
            **cls._llm_algo_base(),
        }
        # Teacher-forced DPO never generates, so its spec has no
        # max_output_tokens; the strict manifest check rejects unknown keys.
        del algo["max_output_tokens"]
        return _make_manifest(
            algo=algo,
            env={
                "env_type": "dataset",
                "objective": "preference",
                "dataset": "dpo_data.parquet",
                "columns": {"prompt": "question", "chosen": "accepted"},
            },
            mutation={
                "probabilities": {"no_mut": 0.1, "rl_hp_mut": 0.6},
                "mutation_sd": 0.1,
            },
        )

    @pytest.fixture(autouse=True)
    def _patch_llm_trainer_init(self):
        """Patch heavy LocalTrainer init steps (env creation, population, tokenizer)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.eos_token = "<eos>"
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
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
            yield

    def test_grpo_produces_llm_env_spec(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, GRPOSpec)
        assert isinstance(trainer.env_spec, LLMEnvSpec)

    def test_grpo_env_type_injected_from_algo(self):
        """When ``env_type`` is not in the environment section, it should be
        injected from ``LLMAlgorithmSpec.env_type``.
        """
        manifest = self._grpo_manifest()
        manifest["environment"].pop("env_type", None)
        trainer = LocalTrainer.from_manifest(manifest)
        assert str(trainer.env_spec.env_type) == "rollout"

    def test_grpo_algorithm_fields(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert trainer.algorithm_spec.group_size == 6
        assert trainer.algorithm_spec.temperature == 0.9
        assert trainer.algorithm_spec.clip_coef == 0.2

    def test_grpo_env_fields(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert trainer.env_spec.dataset == "train.parquet"
        assert trainer.env_spec.rubric_file_path == "reward.py"
        assert trainer.env_spec.rubric_name == "combined_rewards"
        assert trainer.env_spec.max_reward == 10.0
        assert trainer.env_spec.train_test_split == 0.8
        assert trainer.env_spec.prompt_template == {
            "system_0": "You are a helpful assistant.",
            "user_1": "Solve {question} to get {answer}.",
        }

    def test_dpo_produces_llm_env_spec(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, DPOSpec)
        assert isinstance(trainer.env_spec, LLMEnvSpec)

    def test_dpo_env_type_dataset_with_preference_objective(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert str(trainer.env_spec.env_type) == "dataset"
        assert trainer.env_spec.objective == "preference"

    def test_dpo_env_fields(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert trainer.env_spec.dataset == "dpo_data.parquet"
        assert trainer.env_spec.columns == {
            "prompt": "question",
            "chosen": "accepted",
        }

    def test_dpo_algorithm_fields(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert trainer.algorithm_spec.beta == 0.001
        assert trainer.algorithm_spec.update_epochs == 1

    def test_llm_no_network_section(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert (
            trainer.algorithm_spec.net_config is None
            if hasattr(trainer.algorithm_spec, "net_config")
            else True
        )

    def test_llm_rollout_buffer_does_not_init_n_step_memory(self):
        data = self._grpo_manifest()
        data["replay_buffer"] = {"kind": "llm"}
        with patch(
            "agilerl.training.trainer.build_replay_buffer_from_spec",
            return_value=None,
        ):
            trainer = LocalTrainer.from_manifest(data)
        assert trainer.n_step_memory is None
        assert not isinstance(trainer.replay_buffer_spec, ReplayBufferSpec)


# ============================================================================
# TestFromConfigFiles - integration tests loading actual YAML configs
# ============================================================================

_SINGLE_AGENT_CONFIGS = [
    ("dqn/dqn.yaml", DQNSpec, MlpSpec),
    ("dqn/dqn_rainbow.yaml", RainbowDQNSpec, MlpSpec),
    ("dqn/dqn_lstm.yaml", DQNSpec, LstmSpec),
    ("ppo/ppo.yaml", PPOSpec, MlpSpec),
    ("ppo/ppo_image.yaml", PPOSpec, CnnSpec),
    ("ppo/ppo_recurrent.yaml", PPOSpec, LstmSpec),
    ("ddpg/ddpg.yaml", DDPGSpec, MlpSpec),
    ("ddpg/ddpg_lstm.yaml", DDPGSpec, LstmSpec),
    ("ddpg/ddpg_simba.yaml", DDPGSpec, SimbaSpec),
    ("td3.yaml", TD3Spec, MlpSpec),
    ("multi_input.yaml", PPOSpec, MultiInputSpec),
]

_BANDIT_CONFIGS = [
    ("bandit/neural_ts.yaml", NeuralTSSpec, MlpSpec),
    ("bandit/neural_ucb.yaml", NeuralUCBSpec, MlpSpec),
]

_OFFLINE_CONFIGS = [
    ("cqn.yaml", CQNSpec, MlpSpec),
]

_MULTI_AGENT_CONFIGS = [
    ("multi_agent/maddpg.yaml", MADDPGSpec),
    ("multi_agent/matd3.yaml", MATD3Spec),
    ("multi_agent/ippo.yaml", IPPOSpec),
    ("multi_agent/ippo_pong.yaml", IPPOSpec),
]

# Configs omit the network ``arch``, so it is inferred from the observation
# space at build time. Give the stub env a space matching the expected encoder.
_OBS_SPACE_FOR_ENCODER = {
    MlpSpec: spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    LstmSpec: spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    SimbaSpec: spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    CnnSpec: spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8),
    MultiInputSpec: spaces.Dict(
        {"vector": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)}
    ),
}


_LLM_CONFIGS = sorted((CONFIGS_DIR / "llm_finetuning").glob("*.yaml"))


@pytest.mark.parametrize("path", _LLM_CONFIGS, ids=lambda p: p.name)
def test_shipped_llm_configs_satisfy_the_contract(path) -> None:
    """Every shipped LLM config must validate against the manifest contract."""
    TrainingManifest.model_validate(yaml.safe_load(path.read_text()))


class TestFromConfigFiles:
    """Load every YAML config under ``configs/training/`` and verify that
    ``LocalTrainer.from_manifest()`` produces the expected types.
    """

    @pytest.fixture(autouse=True)
    def _patch_heavy_init(self, request):
        """Avoid spawning real environments and populations for config
        file integration tests.

        When a test parametrizes ``expected_encoder_cls``, give the stub env a
        matching observation space so the deferred obs-space -> encoder-arch
        inference resolves to the expected encoder.
        """
        env = MagicMock()
        callspec = getattr(request.node, "callspec", None)
        expected_encoder_cls = (
            callspec.params.get("expected_encoder_cls")
            if callspec is not None
            else None
        )
        obs_space = _OBS_SPACE_FOR_ENCODER.get(expected_encoder_cls)
        if obs_space is not None:
            env.single_observation_space = obs_space

        with (
            patch.object(LocalTrainer, "_make_env", return_value=env),
            patch(
                "agilerl.training.trainer.create_population_from_spec",
                return_value=[MagicMock()],
            ),
        ):
            yield

    @pytest.mark.parametrize(
        ("rel_path", "expected_algo_cls", "expected_encoder_cls"),
        _SINGLE_AGENT_CONFIGS,
        ids=[p for p, *_ in _SINGLE_AGENT_CONFIGS],
    )
    def test_single_agent_config(
        self, rel_path, expected_algo_cls, expected_encoder_cls
    ):
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        trainer = LocalTrainer.from_manifest(config_path)

        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, GymEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)
        if trainer.algorithm_spec.net_config is not None:
            assert isinstance(
                trainer.algorithm_spec.net_config.encoder_config,
                expected_encoder_cls,
            )

    @pytest.mark.parametrize(
        ("rel_path", "expected_algo_cls", "expected_encoder_cls"),
        _OFFLINE_CONFIGS,
        ids=[p for p, *_ in _OFFLINE_CONFIGS],
    )
    def test_offline_config(self, rel_path, expected_algo_cls, expected_encoder_cls):
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        trainer = LocalTrainer.from_manifest(config_path)

        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, OfflineEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)
        if trainer.algorithm_spec.net_config is not None:
            assert isinstance(
                trainer.algorithm_spec.net_config.encoder_config,
                expected_encoder_cls,
            )

    @pytest.mark.parametrize(
        ("rel_path", "expected_algo_cls", "expected_encoder_cls"),
        _BANDIT_CONFIGS,
        ids=[p for p, *_ in _BANDIT_CONFIGS],
    )
    def test_bandit_config(self, rel_path, expected_algo_cls, expected_encoder_cls):
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        trainer = LocalTrainer.from_manifest(config_path)

        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, BanditEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)
        if trainer.algorithm_spec.net_config is not None:
            assert isinstance(
                trainer.algorithm_spec.net_config.encoder_config,
                expected_encoder_cls,
            )

    @pytest.mark.parametrize(
        ("rel_path", "expected_algo_cls"),
        _MULTI_AGENT_CONFIGS,
        ids=[p for p, _ in _MULTI_AGENT_CONFIGS],
    )
    def test_multi_agent_config(self, rel_path, expected_algo_cls):
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        trainer = LocalTrainer.from_manifest(config_path)

        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, GymEnvSpec)

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    @pytest.mark.parametrize(
        ("rel_path", "expected_chunk_rows"),
        [
            ("llm_finetuning/cispo_quant_bench.yaml", 128),
            ("llm_finetuning/cispo_quant_bench_qwen.yaml", 64),
        ],
        ids=["cispo-gemma", "cispo-qwen"],
    )
    def test_llm_quant_bench_configs_validate_strictly(
        self, rel_path, expected_chunk_rows
    ):
        """Shipped configs pass strict manifest validation, and ``chunk_rows``
        survives the round trip as a spec field.
        """
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        out = TrainingManifest.get_validated(config_path)
        assert out["algorithm"]["chunk_rows"] == expected_chunk_rows

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    @pytest.mark.parametrize(
        ("rel_path", "expected_algo_cls"),
        [("grpo.yaml", GRPOSpec), ("dpo.yaml", DPOSpec)],
        ids=["grpo", "dpo"],
    )
    def test_llm_config_parses_as_manifest(self, rel_path, expected_algo_cls):
        """LLM configs need runtime-supplied fields (model path, LoRA
        config, etc.) before they fully validate.  This test merges
        those fields in, mirroring what the benchmarking script does.
        """
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as fh:
            data = yaml.safe_load(fh)

        data["algorithm"].update(
            {
                "lora_config": {
                    "lora_r": 16,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "task_type": "CAUSAL_LM",
                },
                "max_model_len": 512,
                "use_separate_reference_adapter": True,
                "pretrained_model_name_or_path": "test-model",
                "calc_position_embeddings": True,
            }
        )
        data.setdefault("environment", {})

        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.algorithm, expected_algo_cls)


class TestArchOptional:
    def test_arch_present_still_validates(self):
        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "CartPole-v1"},
            "network": {
                "arch": "mlp",
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        assert out["network"]["encoder_config"]["arch"] == "mlp"

    def test_arch_absent_keeps_network_raw(self):
        from agilerl.models.manifest import TrainingManifest as TM

        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "CartPole-v1"},
            "network": {"latent_dim": 64, "encoder_config": {"hidden_size": [64]}},
        }
        manifest = TM.model_validate(raw)
        # Deferred: the network section is kept, algorithm.net_config is not.
        assert not manifest.network_arch_is_resolvable
        assert manifest.algorithm.net_config is None

    def test_network_arch_is_resolvable(self):
        from agilerl.models.networks import network_arch_is_resolvable

        assert network_arch_is_resolvable({"arch": "mlp"})
        assert network_arch_is_resolvable({"encoder_config": {"arch": "cnn"}})
        assert not network_arch_is_resolvable({"encoder_config": {"hidden_size": [64]}})
        assert not network_arch_is_resolvable({})


class TestSimbaRecurrentConflict:
    """``simba`` and ``recurrent`` are contradictory encoder requests.

    A network cannot simultaneously be a SimBa encoder and a recurrent (LSTM)
    encoder, so setting both must raise a clear validation error rather than
    silently picking one (``_build_encoder`` checks recurrent before simba,
    which would otherwise silently ignore the ``simba`` flag).
    """

    def test_deferred_raises(self):
        """No ``arch``: ``net_config`` is a raw dict when the conflict fires."""
        raw = _make_manifest(
            algo={"name": "PPO", "recurrent": True},
            env={"name": "CartPole-v1"},
            network={"simba": True, "head_config": {"hidden_size": [64]}},
        )
        with pytest.raises(ValueError, match="cannot both be set"):
            TrainingManifest.model_validate(raw)

    def test_eager_raises(self):
        """``arch: simba`` declared: ``net_config`` is a validated ``NetworkSpec``."""
        raw = _make_manifest(
            algo={"name": "PPO", "recurrent": True},
            env={"name": "CartPole-v1"},
            network={
                "arch": "simba",
                "encoder_config": {"hidden_size": 128, "num_blocks": 2},
                "head_config": {"hidden_size": [64]},
            },
        )
        with pytest.raises(ValueError, match="cannot both be set"):
            TrainingManifest.model_validate(raw)

    def test_only_simba_validates(self):
        raw = _make_manifest(
            algo={"name": "PPO"},
            env={"name": "CartPole-v1"},
            network={"simba": True, "head_config": {"hidden_size": [64]}},
        )
        manifest = TrainingManifest.model_validate(raw)
        assert manifest.network is not None
        assert manifest.network.simba is True
        assert manifest.algorithm.net_config is None

    def test_only_recurrent_validates(self):
        raw = _make_manifest(
            algo={"name": "PPO", "recurrent": True},
            env={"name": "CartPole-v1"},
            network={"head_config": {"hidden_size": [64]}},
        )
        manifest = TrainingManifest.model_validate(raw)
        assert manifest.algorithm.recurrent is True

    def test_neither_validates(self):
        raw = _make_manifest(
            algo={"name": "PPO"},
            env={"name": "CartPole-v1"},
            network={"head_config": {"hidden_size": [64]}},
        )
        manifest = TrainingManifest.model_validate(raw)
        assert manifest.algorithm.recurrent is False


# The LLM manifests are the ones the demo (``demos/llm/demo_llm_finetuning.py``)
# runs against, and the only place the algorithm-injected ``env_type`` /
# ``objective`` are exercised straight from disk.
_LLM_CONFIG_ENV_TYPES = {
    "cispo.yaml": ("rollout", None),
    "cispo_gemma4_group5.yaml": ("rollout", None),
    "cispo_quant_bench.yaml": ("rollout", None),
    "cispo_quant_bench_qwen.yaml": ("rollout", None),
    "dpo.yaml": ("dataset", "preference"),
    "grpo.yaml": ("rollout", None),
    "grpo_env.yaml": ("rollout", None),
    "gspo.yaml": ("rollout", None),
    "ppo_llm.yaml": ("rollout", None),
    "ppo_llm_quant_bench.yaml": ("rollout", None),
    "reinforce_llm.yaml": ("rollout", None),
    "reinforce_quant_bench.yaml": ("rollout", None),
    "sft.yaml": ("dataset", "sft"),
}


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
class TestLLMConfigFiles:
    """Every manifest under ``configs/training/llm_finetuning/`` must validate.

    Building a ``LocalTrainer`` from these would download a tokenizer, so this
    stops at the manifest and env-spec layer -- which is exactly the contract
    the LLM demo relies on.
    """

    def test_every_llm_config_is_covered(self):
        """Guard against a new config slipping in unvalidated."""
        on_disk = {
            path.name for path in (CONFIGS_DIR / "llm_finetuning").glob("*.yaml")
        }
        assert on_disk == set(_LLM_CONFIG_ENV_TYPES)

    @pytest.mark.parametrize(
        ("filename", "expected"),
        sorted(_LLM_CONFIG_ENV_TYPES.items()),
        ids=sorted(_LLM_CONFIG_ENV_TYPES),
    )
    def test_llm_config_resolves_env_spec(self, filename, expected):
        expected_env_type, expected_objective = expected
        manifest = TrainingManifest.get_validated(
            CONFIGS_DIR / "llm_finetuning" / filename, mode="python"
        )
        env_spec = LocalTrainer._resolve_env_spec(manifest)

        assert isinstance(env_spec, LLMEnvSpec)
        assert str(env_spec.env_type) == expected_env_type
        assert env_spec.objective == expected_objective


def test_network_section_is_none_for_a_paradigm_base_spec():
    from agilerl.arena.models.algorithms import MultiAgentAlgorithmSpec
    from agilerl.models.manifest import _network_from_algorithm

    # The multi-agent base carries no net_config; only its concrete specs do.
    assert _network_from_algorithm(MultiAgentAlgorithmSpec()) is None
