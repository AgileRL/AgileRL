"""Tests for TrainingManifest parsing and Trainer.from_manifest().

Covers single-agent (on-policy, off-policy, offline, bandit), multi-agent,
and LLM training scenarios, along with all supported network architectures
(MLP, CNN, LSTM, SimBA, MultiInput).

Test manifests live under ``tests/manifests/`` as YAML files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.models.algorithms.cqn import CQNSpec
from agilerl.models.algorithms.ddpg import DDPGSpec
from agilerl.models.algorithms.dqn import DQNSpec
from agilerl.models.algorithms.ippo import IPPOSpec
from agilerl.models.algorithms.maddpg import MADDPGSpec
from agilerl.models.algorithms.matd3 import MATD3Spec
from agilerl.models.algorithms.neural_ts import NeuralTSSpec
from agilerl.models.algorithms.neural_ucb import NeuralUCBSpec
from agilerl.models.algorithms.ppo import PPOSpec
from agilerl.models.algorithms.rainbow_dqn import RainbowDQNSpec
from agilerl.models.algorithms.td3 import TD3Spec
from agilerl.models.env import (
    ArenaEnvSpec,
    BanditEnvSpec,
    GymEnvSpec,
    OfflineEnvSpec,
    PzEnvSpec,
)
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.models.networks import (
    CnnSpec,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    NetworkSpec,
    SimbaSpec,
)
from agilerl.models.training import ReplayBufferSpec, TrainingSpec
from agilerl.training.trainer import ArenaTrainer, LocalTrainer

if HAS_LLM_DEPENDENCIES:
    from agilerl.models.algorithms.dpo import DPOSpec
    from agilerl.models.algorithms.grpo import GRPOSpec
    from agilerl.models.env import LLMEnvSpec

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
    """Build an ad-hoc manifest dict (for one-off / error-case tests)."""
    m = {
        "algorithm": algo,
        "environment": env if env is not None else {},
        "training": sections.pop("training", _TRAINING),
    }
    m.update(sections)
    return m


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
# TestTrainingManifest – manifest parsing and validation
# ============================================================================


class TestTrainingManifest:
    """Tests for the ``TrainingManifest`` Pydantic model."""

    # -- Algorithm dispatch -------------------------------------------------

    @pytest.mark.parametrize(
        "name, expected_cls",
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
        "name, extra_fields, expected_cls",
        [
            ("GRPO", {"group_size": 6, "temperature": 0.9}, GRPOSpec),
            ("DPO", {}, DPOSpec),
        ],
    )
    def test_algorithm_dispatch_llm(self, name, extra_fields, expected_cls):
        from peft import LoraConfig

        algo = {
            "name": name,
            "update_epochs": 1,
            "lora_config": LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ),
            "max_model_len": 512,
            "use_separate_reference_adapter": True,
            "pretrained_model_name_or_path": "test-model",
            "calc_position_embeddings": True,
            **extra_fields,
        }
        data = _make_manifest(algo=algo)
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.algorithm, expected_cls)

    def test_algorithm_missing_name_raises(self):
        data = _make_manifest(algo={"batch_size": 64})
        with pytest.raises(Exception, match="name"):
            TrainingManifest.model_validate(data)

    def test_algorithm_unknown_name_raises(self):
        data = _make_manifest(algo={"name": "NonExistentAlgo"})
        with pytest.raises(KeyError, match="NonExistentAlgo"):
            TrainingManifest.model_validate(data)

    # -- Network architecture injection -------------------------------------

    @pytest.mark.parametrize(
        "arch, encoder_kwargs, expected_encoder_cls",
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
        assert isinstance(manifest.network, dict)
        assert manifest.network["encoder_config"]["arch"] == arch

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
        assert manifest.network["encoder_config"]["arch"] == "mlp"

    # -- Field aliases ------------------------------------------------------

    def test_pop_size_alias(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            training={"max_steps": 100, "evo_steps": 10, "pop_size": 8},
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.training.population_size == 8

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
        assert manifest.tournament_selection is None

    def test_environment_required(self):
        data = {"algorithm": {"name": "DQN"}, "training": _TRAINING}
        with pytest.raises(Exception):
            TrainingManifest.model_validate(data)

    def test_environment_kept_as_raw_dict(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1", "num_envs": 4, "custom": False},
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.environment, dict)
        assert manifest.environment["name"] == "CartPole-v1"

    # -- Full manifest parsing ----------------------------------------------

    def test_parse_full_dqn_manifest(self):
        manifest = TrainingManifest.model_validate(DQN_MANIFEST)

        assert isinstance(manifest.algorithm, DQNSpec)
        assert manifest.algorithm.batch_size == 128
        assert manifest.algorithm.tau == 0.001
        assert manifest.algorithm.double is False
        assert manifest.algorithm.lr == pytest.approx(6.3e-4)

        assert isinstance(manifest.mutation, MutationSpec)
        assert isinstance(manifest.network, dict)
        assert manifest.network["encoder_config"]["arch"] == "mlp"
        assert isinstance(manifest.replay_buffer, ReplayBufferSpec)
        assert manifest.replay_buffer.max_size == 100_000
        assert isinstance(manifest.tournament_selection, TournamentSelectionSpec)
        assert isinstance(manifest.training, TrainingSpec)
        assert manifest.training.population_size == 4
        assert manifest.training.eps_start == 1.0
        assert manifest.training.eps_end == 0.1


# ============================================================================
# TestLocalTrainerSingleAgent – Gymnasium (single-agent) scenarios
# ============================================================================


class TestLocalTrainerSingleAgent:
    """``LocalTrainer.from_manifest()`` for single-agent RL algorithms."""

    # -- Parametrized: all single-agent algorithms --------------------------

    @pytest.mark.parametrize(
        "manifest, expected_algo_cls, expected_encoder_cls",
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
        "manifest, expected_algo_cls",
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
        assert isinstance(trainer.tournament_selection_spec, TournamentSelectionSpec)

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
        assert trainer.tournament_selection_spec is None
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
        trainer = LocalTrainer.from_manifest(DQN_MANIFEST, device="cpu")
        assert str(trainer.device) == "cpu"

    # -- Minimal manifest ---------------------------------------------------

    def test_minimal_manifest(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1", "num_envs": 1},
        )
        trainer = LocalTrainer.from_manifest(data)
        assert isinstance(trainer, LocalTrainer)
        assert trainer.algorithm_spec.batch_size == 128  # DQN default
        assert trainer.env_spec.name == "CartPole-v1"
        assert trainer.mutation_spec is None
        assert trainer.tournament_selection_spec is None
        assert trainer.replay_buffer_spec is None


# ============================================================================
# TestLocalTrainerMultiAgent – PettingZoo (multi-agent) scenarios
# ============================================================================


class TestLocalTrainerMultiAgent:
    """``LocalTrainer.from_manifest()`` for multi-agent RL algorithms."""

    @pytest.mark.parametrize(
        "manifest, expected_algo_cls",
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
        assert isinstance(trainer.env_spec, PzEnvSpec)
        assert isinstance(trainer.training_spec, TrainingSpec)

    def test_maddpg_env_name(self):
        trainer = LocalTrainer.from_manifest(MADDPG_MANIFEST)
        assert trainer.env_spec.name == "pettingzoo.mpe.simple_speaker_listener_v4"
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
# TestLocalTrainerLLM – LLM fine-tuning scenarios
# ============================================================================


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
class TestLocalTrainerLLM:
    """``LocalTrainer.from_manifest()`` for LLM algorithms (GRPO, DPO)."""

    @staticmethod
    def _llm_algo_base() -> dict:
        from peft import LoraConfig

        return {
            "update_epochs": 1,
            "lora_config": LoraConfig(
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ),
            "max_model_len": 512,
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
                "dataset_path": "train.parquet",
                "reward_file_path": "reward.py",
                "prompt_template": {"role": "user", "content": "{question}"},
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
        return _make_manifest(
            algo=algo,
            env={
                "env_type": "preference",
                "dataset_path": "dpo_data.parquet",
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
        ):
            yield

    def test_grpo_produces_llm_env_spec(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, GRPOSpec)
        assert isinstance(trainer.env_spec, LLMEnvSpec)

    def test_grpo_env_type_injected_from_algo(self):
        """When ``env_type`` is not in the environment section, it should be
        injected from ``LLMAlgorithmSpec.env_type``."""
        manifest = self._grpo_manifest()
        manifest["environment"].pop("env_type", None)
        trainer = LocalTrainer.from_manifest(manifest)
        assert str(trainer.env_spec.env_type) == "reasoning"

    def test_grpo_algorithm_fields(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert trainer.algorithm_spec.group_size == 6
        assert trainer.algorithm_spec.temperature == 0.9
        assert trainer.algorithm_spec.clip_coef == 0.2

    def test_grpo_env_fields(self):
        trainer = LocalTrainer.from_manifest(self._grpo_manifest())
        assert trainer.env_spec.dataset_path == "train.parquet"
        assert trainer.env_spec.reward_file_path == "reward.py"
        assert trainer.env_spec.max_reward == 10.0
        assert trainer.env_spec.train_test_split == 0.8
        assert trainer.env_spec.prompt_template == {
            "role": "user",
            "content": "{question}",
        }

    def test_dpo_produces_llm_env_spec(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert isinstance(trainer, LocalTrainer)
        assert isinstance(trainer.algorithm_spec, DPOSpec)
        assert isinstance(trainer.env_spec, LLMEnvSpec)

    def test_dpo_env_type_preference(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert str(trainer.env_spec.env_type) == "preference"

    def test_dpo_env_fields(self):
        trainer = LocalTrainer.from_manifest(self._dpo_manifest())
        assert trainer.env_spec.dataset_path == "dpo_data.parquet"
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


class TestArenaTrainerFromManifest:
    """``ArenaTrainer.from_manifest()`` always produces an :class:`ArenaEnvSpec`."""

    def test_from_dict(self):
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(DQN_MANIFEST, client=mock_client)

        assert isinstance(trainer, ArenaTrainer)
        assert isinstance(trainer.algorithm_spec, DQNSpec)
        assert isinstance(trainer.env_spec, ArenaEnvSpec)
        assert trainer.env_spec.name == "LunarLander-v3"
        assert trainer.env_spec.num_envs == 16

    def test_env_defaults_version_to_latest(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1", "num_envs": 4},
        )
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(data, client=mock_client)
        assert trainer.env_spec.version == "latest"

    def test_env_version_from_manifest(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"name": "CartPole-v1", "num_envs": 4, "version": "2.1"},
        )
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(data, client=mock_client)
        assert trainer.env_spec.version == "2.1"

    def test_arena_env_missing_name_raises(self):
        data = _make_manifest(
            algo={"name": "DQN"},
            env={"num_envs": 4},
        )
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="Environment name is required"):
            ArenaTrainer.from_manifest(data, client=mock_client)

    def test_from_yaml_file(self):
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(
            MANIFESTS_DIR / "dqn.yaml", client=mock_client
        )
        assert isinstance(trainer, ArenaTrainer)
        assert isinstance(trainer.env_spec, ArenaEnvSpec)

    @pytest.mark.parametrize(
        "manifest, expected_algo_cls",
        [
            (PPO_MANIFEST, PPOSpec),
            (DDPG_MANIFEST, DDPGSpec),
            (MADDPG_MANIFEST, MADDPGSpec),
            (IPPO_MANIFEST, IPPOSpec),
        ],
        ids=["PPO", "DDPG", "MADDPG", "IPPO"],
    )
    def test_all_algorithms_produce_arena_env(self, manifest, expected_algo_cls):
        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(manifest, client=mock_client)
        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, ArenaEnvSpec)


# ============================================================================
# TestFromConfigFiles – integration tests loading actual YAML configs
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


class TestFromConfigFiles:
    """Load every YAML config under ``configs/training/`` and verify that
    ``LocalTrainer.from_manifest()`` produces the expected types."""

    @pytest.mark.parametrize(
        "rel_path, expected_algo_cls, expected_encoder_cls",
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
        "rel_path, expected_algo_cls, expected_encoder_cls",
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
        "rel_path, expected_algo_cls, expected_encoder_cls",
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
        "rel_path, expected_algo_cls",
        _MULTI_AGENT_CONFIGS,
        ids=[p for p, _ in _MULTI_AGENT_CONFIGS],
    )
    def test_multi_agent_config(self, rel_path, expected_algo_cls):
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        trainer = LocalTrainer.from_manifest(config_path)

        assert isinstance(trainer.algorithm_spec, expected_algo_cls)
        assert isinstance(trainer.env_spec, PzEnvSpec)

    @pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM deps not installed")
    @pytest.mark.parametrize(
        "rel_path, expected_algo_cls",
        [("grpo.yaml", GRPOSpec), ("dpo.yaml", DPOSpec)],
        ids=["grpo", "dpo"],
    )
    def test_llm_config_parses_as_manifest(self, rel_path, expected_algo_cls):
        """LLM configs need runtime-supplied fields (model path, LoRA
        config, etc.) before they fully validate.  This test merges
        those fields in, mirroring what the benchmarking script does."""
        from peft import LoraConfig

        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as fh:
            data = yaml.safe_load(fh)

        data["algorithm"].update(
            {
                "lora_config": LoraConfig(
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                ),
                "max_model_len": 512,
                "use_separate_reference_adapter": True,
                "pretrained_model_name_or_path": "test-model",
                "calc_position_embeddings": True,
            }
        )
        data.setdefault("environment", {})

        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.algorithm, expected_algo_cls)

    @pytest.mark.parametrize(
        "rel_path",
        [p for p, *_ in _SINGLE_AGENT_CONFIGS]
        + [p for p, *_ in _BANDIT_CONFIGS]
        + [p for p, *_ in _OFFLINE_CONFIGS]
        + [p for p, _ in _MULTI_AGENT_CONFIGS],
    )
    def test_arena_trainer_from_config(self, rel_path):
        """Every RL config can also be loaded by ``ArenaTrainer``."""
        config_path = CONFIGS_DIR / rel_path
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        mock_client = MagicMock()
        trainer = ArenaTrainer.from_manifest(config_path, client=mock_client)
        assert isinstance(trainer.env_spec, ArenaEnvSpec)
