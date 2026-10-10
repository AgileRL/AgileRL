# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena manifest models that live only in agilerl-arena."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from importlib.metadata import PackageNotFoundError
from io import StringIO
from unittest.mock import patch

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from agilerl.arena.models import (
    MANIFEST_REGISTRY,
    CheckpointExportSpec,
    ReplayBufferSpec,
    TrainingManifest,
    TrainingSpec,
)
from agilerl.arena.models.algorithms.cispo import CISPOSpec
from agilerl.arena.models.algorithms.dqn import DQNSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.algorithms.gspo import GSPOSpec
from agilerl.arena.models.algorithms.llmppo import LLMPPOSpec
from agilerl.arena.models.algorithms.ppo import PPOSpec, RecurrentPPOSpec
from agilerl.arena.models.env import GymEnvSpec, LLMEnvSpec, LLMEnvType
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.manifest import _resolve_algorithm
from agilerl.arena.models.networks import VLLMConfig
from agilerl.arena.models.registry import AlgorithmRegistry, register
from agilerl.arena.models.schema import _package_version
from agilerl.arena.models.verdict import errors, loc_to_path, read_manifest, verdict


def _manifest(**sections) -> dict:
    data = {
        "algorithm": sections.pop("algorithm", {"name": "DQN"}),
        "environment": sections.pop("environment", {"name": "CartPole-v1"}),
    }
    data.update(sections)
    return data


class TestGymEnvSpec:
    def test_defaults(self) -> None:
        spec = GymEnvSpec(name="CartPole-v1")
        assert spec.num_envs == 32
        assert spec.version is None


class TestTrainingAndReplayAliases:
    def test_population_and_memory_aliases(self) -> None:
        training = TrainingSpec.model_validate(
            {"population_size": 4, "metrics_interval": 123}
        )
        assert training.pop_size == 4
        assert training.evo_steps == 123

        replay = ReplayBufferSpec.model_validate({"memory_size": 4096})
        assert replay.max_size == 4096


class TestTrainingSpecValidators:
    def test_rejects_evo_steps_above_max_steps(self) -> None:
        with pytest.raises(
            ValueError, match=r"evo_steps .* must be less than or equal to max_steps"
        ):
            TrainingSpec(max_steps=100, evo_steps=200)

    def test_rejects_eps_start_below_eps_end(self) -> None:
        with pytest.raises(
            ValueError, match=r"eps_start .* must be greater than or equal to eps_end"
        ):
            TrainingSpec(eps_start=0.1, eps_end=0.9)

    def test_rejects_held_out_eval_with_no_pass(self) -> None:
        with pytest.raises(
            ValueError, match=r"eval_greedy false needs .*eval_samples_per_task >= 1"
        ):
            TrainingSpec(eval_greedy=False)

    def test_accepts_a_sampled_only_held_out_eval(self) -> None:
        spec = TrainingSpec(eval_samples_per_task=4, eval_greedy=False)

        assert spec.eval_samples_per_task == 4
        assert spec.eval_greedy is False


class TestTrainingSpecDefaults:
    def test_bare_training_spec_field_defaults(self) -> None:
        spec = TrainingSpec()
        assert spec.learning_delay is None
        assert spec.experience_sharing is None
        assert spec.hpo is False
        assert spec.checkpoint_export is None
        assert spec.rollout_version_stamp is None

    def test_checkpoints_store_optimizer_state_unless_disabled(self) -> None:
        assert TrainingSpec().checkpoint_optimizer is True
        assert (
            TrainingSpec.model_validate(
                {"checkpoint_optimizer": False}
            ).checkpoint_optimizer
            is False
        )

    def test_held_out_eval_defaults_to_one_uncapped_greedy_pass(self) -> None:
        spec = TrainingSpec()

        assert spec.eval_samples_per_task == 0
        assert spec.eval_greedy is True
        assert spec.eval_loop == 1
        assert spec.eval_max_concurrent_episodes is None


class TestAlgorithmLearnStepDefaults:
    def test_ppo_learn_step_default(self) -> None:
        assert PPOSpec().learn_step == 4096

    def test_recurrent_ppo_learn_step_default(self) -> None:
        assert RecurrentPPOSpec().learn_step == 8192


class TestCheckpointExportSpec:
    def test_classic_training_omits_checkpoint_export(self) -> None:
        spec = TrainingSpec()

        assert spec.checkpoint_export is None
        assert spec.effective_checkpoint_export().format == "adapter"
        assert spec.effective_checkpoint_export().trigger == "final"
        assert (
            spec.effective_checkpoint_export().should_merge(is_final=True, is_best=True)
            is False
        )

    def test_effective_checkpoint_export_returns_the_explicit_spec(self) -> None:
        spec = TrainingSpec.model_validate(
            {
                "max_steps": 100,
                "checkpoint_export": {"format": "merged", "trigger": "every"},
            }
        )

        assert spec.effective_checkpoint_export() is spec.checkpoint_export

    def test_effective_rollout_version_stamp(self) -> None:
        assert TrainingSpec().effective_rollout_version_stamp() is None
        explicit = TrainingSpec(rollout_version_stamp="publish")
        assert explicit.effective_rollout_version_stamp() == "publish"
        async_spec = TrainingSpec(
            rollout_mode="async",
            rollout_engines_per_agent=1,
        )
        assert async_spec.rollout_version_stamp is None
        assert async_spec.effective_rollout_version_stamp() == "oldest_turn"

    def test_parses_merged_every_from_training_dict(self) -> None:
        spec = TrainingSpec.model_validate(
            {
                "max_steps": 100,
                "checkpoint_export": {"format": "merged", "trigger": "every"},
            }
        )

        assert spec.checkpoint_export.format == "merged"
        assert spec.checkpoint_export.trigger == "every"
        assert (
            spec.checkpoint_export.should_merge(is_final=False, is_best=False) is True
        )

    def test_final_merges_only_on_final_save(self) -> None:
        export = CheckpointExportSpec(format="merged", trigger="final")

        assert export.should_merge(is_final=False, is_best=False) is False
        assert export.should_merge(is_final=True, is_best=False) is True

    def test_best_falls_back_to_final(self) -> None:
        export = CheckpointExportSpec(format="merged", trigger="best")

        assert export.should_merge(is_final=False, is_best=False) is False
        assert export.should_merge(is_final=False, is_best=True) is True
        assert export.should_merge(is_final=True, is_best=False) is True

    def test_on_demand_never_merges_during_train(self) -> None:
        export = CheckpointExportSpec(format="merged", trigger="on_demand")

        assert export.should_merge(is_final=True, is_best=True) is False

    def test_rejects_unknown_format(self) -> None:
        with pytest.raises(ValidationError, match="format"):
            CheckpointExportSpec(format="full", trigger="final")

    def test_rejects_unknown_trigger(self) -> None:
        with pytest.raises(ValidationError, match="trigger"):
            CheckpointExportSpec(format="merged", trigger="interval")


class TestVLLMConfigLimitMmPerPrompt:
    def test_parses_per_modality_counts(self) -> None:
        config = VLLMConfig.model_validate(
            {"limit_mm_per_prompt": {"image": 4, "video": 0}}
        )

        assert config.limit_mm_per_prompt == {"image": 4, "video": 0}

    def test_defaults_to_unset(self) -> None:
        assert VLLMConfig().limit_mm_per_prompt is None

    def test_rejects_a_negative_count(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            VLLMConfig(limit_mm_per_prompt={"image": -1})


class TestLLMEnvType:
    def test_str(self) -> None:
        assert str(LLMEnvType.ROLLOUT) == "rollout"
        assert str(LLMEnvType.DATASET) == "dataset"


class TestAlgorithmRegistry:
    def test_get_unknown_name_lists_registered(self) -> None:
        with pytest.raises(KeyError, match="No registry entry for algorithm 'NOPE'"):
            MANIFEST_REGISTRY.get("NOPE")

    def test_create_recurrent_ppo_defaults_recurrent(self) -> None:
        spec = MANIFEST_REGISTRY.create("Recurrent PPO")
        assert spec.recurrent is True
        assert spec.name == "Recurrent PPO"

    def test_create_rejects_recurrent_ppo_with_recurrent_off(self) -> None:
        with pytest.raises(ValueError, match="Recurrent PPO requires recurrent=True"):
            MANIFEST_REGISTRY.create("Recurrent PPO", recurrent=False)

    def test_override_logs_warning(self) -> None:
        with patch("agilerl.arena.models.registry.logger") as mock_logger:
            MANIFEST_REGISTRY.add("DQN", DQNSpec)
        mock_logger.warning.assert_called_once_with(
            "Overriding existing registration for algorithm %r",
            "DQN",
        )

    def test_register_decorator_uses_class_name(self) -> None:
        registry = AlgorithmRegistry()

        with patch("agilerl.arena.models.registry.MANIFEST_REGISTRY", registry):

            @register()
            class _TempSpec(DQNSpec):
                pass

        try:
            assert registry.get("_Temp") is _TempSpec
        finally:
            MANIFEST_REGISTRY._entries.pop("_Temp", None)


class TestResolveAlgorithm:
    def test_requires_name(self) -> None:
        with pytest.raises(ValueError, match="must include a 'name' field"):
            _resolve_algorithm({"lr": 1e-3})

    def test_rejects_non_mapping(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            _resolve_algorithm(42)

    def test_unknown_name_is_a_validation_error(self) -> None:
        with pytest.raises(ValueError, match="not a registered algorithm"):
            _resolve_algorithm({"name": "NOT_REAL"})

    def test_passes_through_a_spec(self) -> None:
        spec = DQNSpec()
        assert _resolve_algorithm(spec) is spec


class TestGetValidated:
    def test_loads_yaml_file(self, tmp_path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(_manifest()))

        payload = TrainingManifest.get_validated(path)
        assert payload["algorithm"]["name"] == "DQN"

    def test_python_mode_returns_model(self) -> None:
        manifest = TrainingManifest.get_validated(_manifest(), mode="python")
        assert manifest.algorithm.name == "DQN"
        assert manifest.environment.name == "CartPole-v1"

    def test_infers_offline_env_type_for_cqn(self) -> None:
        validated = TrainingManifest.get_validated(
            _manifest(algorithm={"name": "CQN"}, environment={"name": "CartPole-v1"}),
            mode="python",
        )
        assert validated.environment.env_type == "offline"

    def test_infers_bandit_env_type(self) -> None:
        validated = TrainingManifest.get_validated(
            _manifest(algorithm={"name": "NeuralUCB"}),
            mode="python",
        )
        assert validated.environment.env_type == "bandit"


class TestLLMAlgorithmSpecValidators:
    def test_valid_constrain_answer_pattern_compiles(self) -> None:
        spec = GRPOSpec(
            group_size=2,
            constrain_answer_pattern=r"<answer>.*</answer>",
        )
        assert spec.constrain_answer_pattern == r"<answer>.*</answer>"

    def test_explicit_none_constrain_answer_pattern(self) -> None:
        spec = GRPOSpec.model_validate(
            {"group_size": 2, "constrain_answer_pattern": None},
        )
        assert spec.constrain_answer_pattern is None

    def test_answer_pattern_alias_populates_constrain_answer_pattern(self) -> None:
        spec = GRPOSpec.model_validate(
            {"group_size": 2, "answer_pattern": r"<answer>.*</answer>"},
        )
        assert spec.constrain_answer_pattern == r"<answer>.*</answer>"

    def test_invalid_constrain_answer_pattern_regex(self) -> None:
        with pytest.raises(ValidationError, match="not a valid regular expression"):
            GRPOSpec(group_size=2, constrain_answer_pattern="(")

    def test_answer_continuation_requires_constrain_answer_pattern(self) -> None:
        with pytest.raises(ValidationError, match="answer_continuation requires"):
            GRPOSpec(group_size=2, answer_continuation=True)

    def test_constrain_answer_pattern_without_continuation(self) -> None:
        spec = GRPOSpec(
            group_size=2,
            constrain_answer_pattern=r"\\boxed\{\d+\}",
        )
        assert spec.constrain_answer_pattern == r"\\boxed\{\d+\}"
        assert spec.answer_continuation is False

    def test_thinking_budget_requires_max_output_tokens(self) -> None:
        with pytest.raises(ValidationError, match="requires max_output_tokens"):
            GRPOSpec(group_size=2, thinking_token_budget=8)

    def test_thinking_budget_must_leave_answer_headroom(self) -> None:
        with pytest.raises(ValidationError, match="must be less than"):
            GRPOSpec(group_size=2, thinking_token_budget=16, max_output_tokens=16)

    def test_thinking_budget_below_max_output_tokens(self) -> None:
        spec = GRPOSpec(group_size=2, thinking_token_budget=8, max_output_tokens=32)
        assert spec.thinking_token_budget == 8

    def test_dpo_thinking_budget_has_no_output_cap(self) -> None:
        from agilerl.arena.models.algorithms.dpo import DPOSpec

        with pytest.raises(ValidationError, match="requires max_output_tokens"):
            DPOSpec(thinking_token_budget=8)

    def test_sequence_packing_requires_packed_attention(self) -> None:
        with pytest.raises(ValidationError, match="use_sequence_packing requires"):
            GRPOSpec(group_size=2, use_sequence_packing=True)

    def test_sequence_packing_with_flash_attention(self) -> None:
        spec = GRPOSpec(
            group_size=2,
            use_sequence_packing=True,
            attn_implementation="flash_attention_2",
        )
        assert spec.use_sequence_packing is True

    def test_deepspeed_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GRPOSpec(group_size=2, deepspeed={"activation_checkpointing": {}})

    def test_zero_stage_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            GRPOSpec(group_size=2, zero_stage=3)

    def test_fsdp_true_coerces_to_config(self) -> None:
        from agilerl.distributed import FSDPConfig

        spec = GRPOSpec(group_size=2, fsdp=True)
        assert spec.fsdp == FSDPConfig()

    def test_fsdp_empty_dict_coerces_to_config(self) -> None:
        from agilerl.distributed import FSDPConfig

        spec = GRPOSpec(group_size=2, fsdp={})
        assert spec.fsdp == FSDPConfig()

    def test_fsdp_dict_coerces_known_fields(self) -> None:
        spec = GRPOSpec(group_size=2, fsdp={"cpu_offload": True})
        assert spec.fsdp is not None
        assert spec.fsdp.cpu_offload is True

    def test_fsdp_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Unknown fsdp keys"):
            GRPOSpec(group_size=2, fsdp={"not_a_field": True})

    def test_fsdp_none_stays_none(self) -> None:
        spec = GRPOSpec(group_size=2, fsdp=None)
        assert spec.fsdp is None

    def test_fsdp_config_passes_through(self) -> None:
        config = FSDPConfig(wrap_every_n_blocks=2)
        spec = GRPOSpec(group_size=2, fsdp=config)
        assert spec.fsdp == config

    def test_fsdp_false_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="fsdp must be null"):
            GRPOSpec(group_size=2, fsdp=False)

    def test_fsdp_serializes_dtypes_as_names(self) -> None:
        spec = GRPOSpec(group_size=2, fsdp=True)
        dumped = spec.model_dump()["fsdp"]
        assert dumped["param_dtype"] == "bfloat16"
        assert dumped["reduce_dtype"] == "float32"

    def test_model_validate_rejects_non_dict(self) -> None:
        with pytest.raises(ValidationError):
            GRPOSpec.model_validate(["not", "a", "dict"])

    def test_mini_batch_must_be_a_multiple_of_micro_batch(self) -> None:
        with pytest.raises(ValidationError, match="not a multiple"):
            GRPOSpec(group_size=2, mini_batch_size=3, micro_batch_size_per_gpu=2)

    def test_mini_batch_multiple_of_micro_batch(self) -> None:
        spec = GRPOSpec(group_size=2, mini_batch_size=4, micro_batch_size_per_gpu=2)
        assert spec.mini_batch_size == 4


class TestGRPOClipCoef:
    def test_rejects_negative_scalar(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            GRPOSpec(group_size=2, clip_coef=-0.1)

    def test_rejects_clip_coef_list(self) -> None:
        with pytest.raises(ValidationError, match="valid number"):
            GRPOSpec(group_size=2, clip_coef=[0.1, 0.2])

    def test_rejects_scalar_above_one(self) -> None:
        with pytest.raises(ValidationError, match=r"less than or equal to 1"):
            GRPOSpec(group_size=2, clip_coef=1.5)

    def test_rejects_non_numeric_clip_coef(self) -> None:
        with pytest.raises(ValidationError, match="valid number"):
            GRPOSpec(group_size=2, clip_coef="wide")


class TestGRPOSpecOffPolicyCorrections:
    @pytest.mark.parametrize("spec_cls", [GRPOSpec, CISPOSpec, GSPOSpec])
    def test_corrections_and_mean_only_advantages_are_on_by_default(
        self, spec_cls: type[GRPOSpec]
    ) -> None:
        spec = spec_cls(group_size=2)

        assert spec.off_policy_token_mask_bounds == (0.5, 5.0)
        assert spec.off_policy_sequence_mask_threshold == 0.03
        assert spec.use_bias_correction_kl is True
        assert spec.adv_norm == "mean_only"
        assert spec.top_p == 1.0

    def test_yaml_null_and_false_turn_them_off(self) -> None:
        spec = GRPOSpec.model_validate(
            yaml.safe_load(
                "group_size: 2\n"
                "off_policy_token_mask_bounds: null\n"
                "off_policy_sequence_mask_threshold: null\n"
                "use_bias_correction_kl: false\n"
                "adv_norm: mean_std\n"
            )
        )

        assert spec.off_policy_token_mask_bounds is None
        assert spec.off_policy_sequence_mask_threshold is None
        assert spec.use_bias_correction_kl is False
        assert spec.adv_norm == "mean_std"

    def test_yaml_values_parse(self) -> None:
        spec = GRPOSpec.model_validate(
            yaml.safe_load(
                "group_size: 2\n"
                "off_policy_token_mask_bounds: [0.5, 5.0]\n"
                "off_policy_sequence_mask_threshold: 0.05\n"
                "use_bias_correction_kl: true\n"
            )
        )

        assert spec.off_policy_token_mask_bounds == (0.5, 5.0)
        assert spec.off_policy_sequence_mask_threshold == 0.05
        assert spec.use_bias_correction_kl is True

    def test_rejects_a_three_value_band(self) -> None:
        with pytest.raises(ValidationError, match="at most 2 items"):
            GRPOSpec(group_size=2, off_policy_token_mask_bounds=[0.5, 1.0, 5.0])

    def test_rejects_a_negative_sequence_threshold(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            GRPOSpec(group_size=2, off_policy_sequence_mask_threshold=-0.1)


@pytest.mark.parametrize(
    ("spec_cls", "required"),
    [(GRPOSpec, {"group_size": 2}), (LLMPPOSpec, {})],
    ids=["grpo", "llmppo"],
)
class TestKlClampField:
    def test_defaults_to_ten(
        self, spec_cls: type[BaseModel], required: dict[str, int]
    ) -> None:
        spec = spec_cls.model_validate(required)

        assert spec.kl_clamp == 10.0

    @pytest.mark.parametrize(("value", "expected"), [("2.5", 2.5), ("null", None)])
    def test_yaml_value_parses(
        self,
        spec_cls: type[BaseModel],
        required: dict[str, int],
        value: str,
        expected: float | None,
    ) -> None:
        spec = spec_cls.model_validate(
            {**required, **yaml.safe_load(f"kl_clamp: {value}\n")}
        )

        assert spec.kl_clamp == expected

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_a_non_positive_bound(
        self, spec_cls: type[BaseModel], required: dict[str, int], value: float
    ) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            spec_cls.model_validate({**required, "kl_clamp": value})


class TestDPOSFTAndRainbow:
    def test_dpo_rejects_sequence_packing(self) -> None:
        from agilerl.arena.models.algorithms.dpo import DPOSpec

        assert DPOSpec().use_sequence_packing is False
        with pytest.raises(
            ValidationError, match="does not accept use_sequence_packing"
        ):
            DPOSpec(use_sequence_packing=True)

    def test_sft_rejects_sequence_packing(self) -> None:
        from agilerl.arena.models.algorithms.sft import SFTSpec

        with pytest.raises(
            ValidationError, match="does not accept use_sequence_packing"
        ):
            SFTSpec(use_sequence_packing=True)

    def test_sft_rejects_activation_offload(self) -> None:
        from agilerl.arena.models.algorithms.sft import SFTSpec

        with pytest.raises(ValidationError, match="activation_offload"):
            SFTSpec(activation_offload=True)

    def test_rainbow_accepts_a_valid_value_range(self) -> None:
        from agilerl.arena.models.algorithms.rainbow_dqn import RainbowDQNSpec

        spec = RainbowDQNSpec(v_min=-10.0, v_max=10.0)
        assert spec.v_min < spec.v_max


class TestVerdict:
    def test_loc_to_path_indexes_and_skips_internal_nodes(self) -> None:
        assert loc_to_path(("algorithm", "clip_coef", 0)) == "algorithm.clip_coef[0]"
        assert loc_to_path((0, "algorithm")) == "algorithm"
        assert loc_to_path(("algorithm", "function-after[1]")) == "algorithm"

    def test_errors_from_value_error(self) -> None:
        assert errors(ValueError("nope")) == [
            {"path": "", "message": "nope", "kind": "value_error"}
        ]

    def test_errors_from_validation_error(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            GRPOSpec(group_size=2, use_vllm=True)
        records = errors(exc_info.value)
        assert records
        assert records[0]["kind"]

    def test_read_manifest_from_stdin(self) -> None:
        with patch("sys.stdin", StringIO(yaml.safe_dump(_manifest()))):
            document = read_manifest("-")
        assert document["algorithm"]["name"] == "DQN"

    def test_valid_and_invalid_documents(self) -> None:
        assert verdict(_manifest())["ok"] is True
        failed = verdict(_manifest(algorithm={"name": "NOPE"}))
        assert failed["ok"] is False
        assert failed["errors"]


class TestSchemaPackageVersion:
    def test_source_checkout_without_distribution(self) -> None:
        with patch(
            "agilerl.arena.models.schema.distribution_version",
            side_effect=PackageNotFoundError("agilerl-arena"),
        ):
            assert _package_version() == "0+unknown"

    def test_models_skips_non_model_registry_entries(self) -> None:
        from agilerl.arena.models.manifest import TrainingManifest
        from agilerl.arena.models.schema import _models

        MANIFEST_REGISTRY.add("__not_a_model__", object)
        try:
            found = _models(TrainingManifest)
        finally:
            MANIFEST_REGISTRY._entries.pop("__not_a_model__", None)
        assert "TrainingManifest" in found

    def test_add_alias_spellings_without_properties(self) -> None:
        from agilerl.arena.models.manifest import TrainingManifest
        from agilerl.arena.models.schema import _add_alias_spellings

        schema: dict = {}
        _add_alias_spellings(schema, TrainingManifest)
        assert "properties" not in schema


class TestPackageVersionFallback:
    def test_models_init_without_distribution(self) -> None:
        import importlib

        import agilerl.arena.models as models

        with patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError("agilerl-arena"),
        ):
            importlib.reload(models)
            assert models.__version__ == "0+unknown"
        importlib.reload(models)


class TestTrainingSpec:
    def test_rejects_retired_async_rollout_key(self) -> None:
        with pytest.raises(ValidationError, match="retired training key"):
            TrainingSpec.model_validate({"async_rollout": True})

    def test_async_rollout_fills_defaults(self) -> None:
        spec = TrainingSpec(
            rollout_mode="async",
            rollout_engines_per_agent="auto",
            training_gpus_per_agent=1,
        )
        assert spec.weight_sync_backend == "nccl"
        assert spec.rollout_batch_size == 1
        assert spec.weight_sync_interval == 1

    def test_async_rollout_rejects_zero_engines(self) -> None:
        with pytest.raises(ValidationError, match="rollout_engines_per_agent"):
            TrainingSpec(
                rollout_mode="async",
                rollout_engines_per_agent=0,
                training_gpus_per_agent=1,
            )

    def test_async_rollout_rejects_batch_not_multiple_of_gpus(self) -> None:
        with pytest.raises(ValidationError, match="multiple of"):
            TrainingSpec(
                rollout_mode="async",
                rollout_engines_per_agent="auto",
                rollout_batch_size=3,
                training_gpus_per_agent=2,
            )

    def test_async_rollout_accepts_batch_not_multiple_of_engines(self) -> None:
        spec = TrainingSpec(
            rollout_mode="async",
            rollout_engines_per_agent=2,
            rollout_batch_size=3,
            training_gpus_per_agent=1,
        )

        assert spec.rollout_batch_size == 3

    def test_async_rollout_rejects_experience_sharing(self) -> None:
        with pytest.raises(ValidationError, match="experience_sharing"):
            TrainingSpec(
                rollout_mode="async",
                rollout_engines_per_agent="auto",
                training_gpus_per_agent=1,
                experience_sharing=True,
            )

    def test_n_step_none_defaults(self) -> None:
        from agilerl.arena.models.training import NStepBufferArgs, PerBufferArgs

        assert NStepBufferArgs.model_validate({"n_step": None}).n_step == 3
        assert PerBufferArgs.model_validate({"alpha": None}).alpha == 0.5


CLASSIFIEDS_SERVICE = {
    "name": "classifieds",
    "port": 9980,
    "url_env": "VWA_CLASSIFIEDS",
    "containers": [
        {"name": "db", "image": "classifieds-db:1"},
        {"name": "web", "image": "classifieds-web:1", "port": 9980},
    ],
}


class TestLLMEnvSpecEnvPods:
    def test_image_defaults_and_services_parse(self) -> None:
        from agilerl.arena.models.env import EnvServiceSpec

        spec = LLMEnvSpec(
            env_type="rollout",
            env_image="browser:1",
            env_sessions_per_host=4,
            env_session_ports={"BROWSERGYM_PORT": 8000},
            env_services=[CLASSIFIEDS_SERVICE],
        )

        assert spec.env_port == 8000
        assert spec.cpus_per_env_host == 0.01
        assert spec.env_host_ready_timeout_s == 600.0
        assert spec.env_services == [EnvServiceSpec.model_validate(CLASSIFIEDS_SERVICE)]
        assert spec.env_services[0].ready_timeout_s == 600.0
        assert spec.env_services[0].containers[0].cpu == "10m"

    @pytest.mark.parametrize(
        "field",
        [
            {"env_sessions_per_host": 2},
            {"env_session_ports": {"BROWSERGYM_PORT": 8000}},
            {"env_host_memory_limit_bytes": 1024},
            {"env_host_ready_timeout_s": 900},
            {"env_services": [CLASSIFIEDS_SERVICE]},
        ],
    )
    def test_pod_fields_need_an_image(self, field: dict) -> None:
        with pytest.raises(ValidationError, match="configure env_image Pods"):
            LLMEnvSpec(env_type="rollout", entrypoint="pkg.mod:Env", **field)

    def test_image_resolves_one_session_per_pod(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_image="browser:1")

        assert spec.env_sessions_per_host == 1
        assert spec.model_dump(mode="json")["env_sessions_per_host"] == 1

    def test_env_host_ready_timeout_s_is_kept(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_image="browser:1",
            env_host_ready_timeout_s=1800,
        )

        assert spec.env_host_ready_timeout_s == 1800.0

    def test_sessions_sharing_a_pod_need_session_ports(self) -> None:
        with pytest.raises(
            ValidationError,
            match="env_sessions_per_host=4 needs env_session_ports",
        ):
            LLMEnvSpec(
                env_type="rollout",
                env_image="browser:1",
                env_sessions_per_host=4,
            )

    @pytest.mark.parametrize(
        ("field", "name"),
        [
            ({"env_vars": {"A": "1"}}, "env_vars"),
            ({"env_sessions_per_host": 2}, "env_sessions_per_host"),
            ({"env_session_ports": {"PORT": 8000}}, "env_session_ports"),
            ({"env_host_memory_limit_bytes": 1024}, "env_host_memory_limit_bytes"),
            ({"env_host_ready_timeout_s": 900}, "env_host_ready_timeout_s"),
            ({"env_services": [CLASSIFIEDS_SERVICE]}, "env_services"),
        ],
    )
    def test_dataset_env_rejects_pod_fields(self, field: dict, name: str) -> None:
        with pytest.raises(
            ValidationError,
            match=f"{name} configure env_image Pods; a dataset environment",
        ):
            LLMEnvSpec(env_type="dataset", objective="sft", dataset="rows", **field)

    def test_rejects_a_repeated_service_name(self) -> None:
        with pytest.raises(ValidationError, match="repeats a service name"):
            LLMEnvSpec(
                env_type="rollout",
                env_image="browser:1",
                env_services=[CLASSIFIEDS_SERVICE, CLASSIFIEDS_SERVICE],
            )

    def test_rejects_an_env_name_set_twice(self) -> None:
        with pytest.raises(ValidationError, match=r"\['VWA_CLASSIFIEDS'\] set more"):
            LLMEnvSpec(
                env_type="rollout",
                env_image="browser:1",
                env_vars={"VWA_CLASSIFIEDS": "http://elsewhere"},
                env_services=[CLASSIFIEDS_SERVICE],
            )

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"name": "Classifieds"}, "string_pattern_mismatch|should match pattern"),
            ({"url_env": "1BAD"}, "should match pattern"),
            ({"replicas": 2}, "Extra inputs are not permitted"),
            (
                {"containers": [{"name": "db", "image": "db:1"}]},
                "exactly one container with a port",
            ),
            (
                {
                    "containers": [
                        {"name": "web", "image": "a:1", "port": 80},
                        {"name": "web", "image": "b:1"},
                    ],
                },
                "repeats a container name",
            ),
        ],
    )
    def test_service_spec_errors(self, change: dict, match: str) -> None:
        from agilerl.arena.models.env import EnvServiceSpec

        with pytest.raises(ValidationError, match=match):
            EnvServiceSpec.model_validate({**CLASSIFIEDS_SERVICE, **change})

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            (
                {"port": 80, "readiness_command": ["true"], "readiness_path": "/"},
                "one readiness probe",
            ),
            ({"readiness_path": "/"}, "readiness_path probes port; set port"),
            ({"memory_rootfs": "100Gi"}, "without its ENTRYPOINT; set command"),
        ],
    )
    def test_container_spec_errors(self, change: dict, match: str) -> None:
        from agilerl.arena.models.env import EnvContainerSpec

        with pytest.raises(ValidationError, match=match):
            EnvContainerSpec.model_validate({"name": "web", "image": "a:1", **change})


class TestLLMEnvSpecSurfaces:
    def test_name_from_dataset_and_urls(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        dataset = LLMEnvSpec(
            env_type="rollout",
            dataset="rows",
            rubric_file_path="rubric.py",
            prompt_template={"user_0": "{q}"},
        )
        assert dataset.name is None
        assert dataset.label == "rows"
        assert dataset.dataset_backed_rollout is True

        url = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=1)
        assert url.label == "http://env"

        urls = LLMEnvSpec(
            env_type="rollout", env_url=["http://a", "http://b"], max_turns=1
        )
        assert urls.label == "http://a"

        image = LLMEnvSpec(
            env_type="rollout",
            env_image="env:latest",
            env_vars={"BROWSERGYM_BENCHMARK": "miniwob"},
        )
        assert image.label == "env:latest"
        assert image.env_vars == {"BROWSERGYM_BENCHMARK": "miniwob"}

        with pytest.raises(ValueError, match="env_vars set container environment"):
            LLMEnvSpec(
                env_type="rollout",
                dataset="rows",
                env_vars={"BROWSERGYM_BENCHMARK": "miniwob"},
            )

        named = LLMEnvSpec(
            env_type="rollout",
            dataset="rows",
            name="countdown",
            rubric_file_path="rubric.py",
            prompt_template={"user_0": "{q}"},
        )
        assert named.name == "countdown"
        assert named.label == "countdown"

    def test_env_hosts_requires_entrypoint_or_image(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        with pytest.raises(ValidationError, match="env_hosts"):
            LLMEnvSpec(
                env_type="rollout",
                dataset="rows",
                rubric_file_path="rubric.py",
                prompt_template={"user_0": "{q}"},
                env_hosts=1,
            )

    def test_dataset_env_rejects_an_entrypoint(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        with pytest.raises(ValidationError, match="teacher-forced"):
            LLMEnvSpec(
                env_type="dataset",
                objective="sft",
                dataset="rows",
                entrypoint="mod:make",
            )

    def test_env_packages_need_a_package_list(self) -> None:
        from agilerl.arena.models.env import _check_env_packages

        with pytest.raises(ValueError, match="env_packages"):
            _check_env_packages({"uv": {"packages": []}})

        _check_env_packages({"uv": {"packages": ["agilerl"]}})
        _check_env_packages({"pip": ["agilerl"]})
        with pytest.raises(ValueError, match="must be a list"):
            _check_env_packages({"uv": "agilerl"})

    def test_env_config_env_id_is_rejected(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        with pytest.raises(ValidationError, match="cannot contain env_id"):
            LLMEnvSpec(
                env_type="rollout",
                factory="gem:make",
                entrypoint="game:GuessTheNumber-v0-easy",
                env_config={"env_id": "game:GuessTheNumber-v0-easy"},
            )

    def test_factory_without_entrypoint_is_rejected(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        with pytest.raises(ValidationError, match="factory is set but entrypoint"):
            LLMEnvSpec(env_type="rollout", factory="gem:make")

    def test_gem_rollout_sets_rollout_defaults_leaves_dataset_fields_none(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        gem = LLMEnvSpec(
            env_type="rollout",
            entrypoint="game:GuessTheNumber-v0-easy",
            factory="gem:make",
        )
        assert gem.train_test_split is None
        assert gem.response_column is None
        assert gem.rubric_name is None
        assert gem.strict_chat_template_boundary is True
        assert gem.num_envs == 1
        assert gem.action_field == "message"

    def test_dataset_backed_rollout_fills_conditional_defaults(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        rollout = LLMEnvSpec(
            env_type="rollout",
            dataset="rows",
            rubric_file_path="rubric.py",
            prompt_template={"user_0": "{q}"},
        )
        assert rollout.train_test_split == 0.9
        assert rollout.response_column is None
        assert rollout.rubric_name == "reward_fn"
        assert rollout.strict_chat_template_boundary is True
        assert rollout.num_envs == 1
        assert rollout.action_field is None

    def test_dataset_env_type_fills_split_and_response_column(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        dataset = LLMEnvSpec(
            env_type="dataset",
            objective="sft",
            dataset="rows",
        )
        assert dataset.train_test_split == 0.9
        assert dataset.response_column == "response"
        assert dataset.strict_chat_template_boundary is None
        assert dataset.rubric_name is None
        assert dataset.num_envs is None
        assert dataset.action_field is None


class TestLLMEnvSpecSegmentPromptTokens:
    def test_defaults_to_none(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=10)

        assert spec.segment_prompt_tokens is None

    def test_accepts_a_positive_token_count(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            segment_prompt_tokens=20000,
        )

        assert spec.segment_prompt_tokens == 20000

    @pytest.mark.parametrize("tokens", [0, -1])
    def test_rejects_a_non_positive_token_count(self, tokens: int) -> None:
        with pytest.raises(ValidationError, match="segment_prompt_tokens"):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                segment_prompt_tokens=tokens,
            )


class TestLLMEnvSpecSegmentMaxImages:
    def test_accepts_a_positive_image_count(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            segment_max_images=4,
        )

        assert spec.segment_max_images == 4

    def test_defaults_to_none(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=10)

        assert spec.segment_max_images is None

    @pytest.mark.parametrize("images", [0, -1])
    def test_rejects_a_non_positive_image_count(self, images: int) -> None:
        with pytest.raises(ValidationError, match="segment_max_images"):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                segment_max_images=images,
            )


class TestLLMEnvSpecRestartKeepTurns:
    def test_accepts_a_turn_count(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            segment_max_images=4,
            restart_keep_turns=2,
        )

        assert spec.restart_keep_turns == 2

    def test_accepts_a_token_limit_alone(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            segment_prompt_tokens=24000,
            restart_keep_turns=2,
        )

        assert spec.restart_keep_turns == 2

    def test_rejects_kept_turns_without_a_restart_limit(self) -> None:
        with pytest.raises(
            ValidationError,
            match="restart_keep_turns applies at a context restart; set "
            "segment_prompt_tokens or segment_max_images",
        ):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                restart_keep_turns=2,
            )

    @pytest.mark.parametrize("keep", [4, 5])
    def test_rejects_as_many_kept_turns_as_segment_images(self, keep: int) -> None:
        with pytest.raises(
            ValidationError,
            match=rf"restart_keep_turns \({keep}\) must be below "
            r"segment_max_images \(4\)",
        ):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                segment_max_images=4,
                restart_keep_turns=keep,
            )

    def test_defaults_to_zero(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=10)

        assert spec.restart_keep_turns == 0

    def test_rejects_a_negative_count(self) -> None:
        with pytest.raises(ValidationError, match="restart_keep_turns"):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                restart_keep_turns=-1,
            )


class TestLLMEnvSpecTaskFamilyField:
    def test_accepts_a_family_field_under_adaptive_sampling(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            adaptive_task_sampling=True,
            task_family_field="template_key",
            task_family_prior_strength=0.5,
        )

        assert spec.task_family_field == "template_key"
        assert spec.task_family_prior_strength == 0.5

    def test_defaults_to_no_family(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=10)

        assert spec.task_family_field is None
        assert spec.task_family_prior_strength == 1.0

    def test_rejects_a_family_field_without_adaptive_sampling(self) -> None:
        with pytest.raises(
            ValidationError,
            match="task_family_field weights rows under adaptive_task_sampling",
        ):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                task_family_field="template_key",
            )

    def test_rejects_a_prior_strength_that_is_not_positive(self) -> None:
        with pytest.raises(ValidationError, match="task_family_prior_strength"):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                task_family_prior_strength=0.0,
            )


class TestLLMEnvSpecActionErrorField:
    def test_accepts_a_field_name(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            action_error_field="error",
        )

        assert spec.action_error_field == "error"

    def test_defaults_to_empty(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=10)

        assert spec.action_error_field == ""


class TestLLMEnvSpecRestartOlderObservations:
    def test_accepts_older_turn_settings_with_two_kept_turns(self) -> None:
        spec = LLMEnvSpec(
            env_type="rollout",
            env_url="http://env",
            max_turns=10,
            segment_max_images=4,
            restart_keep_turns=2,
            restart_older_obs_field="url",
            restart_older_images=False,
        )

        assert spec.restart_older_obs_field == "url"
        assert spec.restart_older_images is False

    def test_defaults_repeat_kept_turns_in_full(self) -> None:
        spec = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=10)

        assert spec.restart_older_obs_field == ""
        assert spec.restart_older_images is True

    @pytest.mark.parametrize(
        "older",
        [{"restart_older_obs_field": "url"}, {"restart_older_images": False}],
    )
    def test_rejects_older_turn_settings_with_one_kept_turn(
        self, older: dict[str, object]
    ) -> None:
        with pytest.raises(
            ValidationError, match="set restart_keep_turns to 2 or more"
        ):
            LLMEnvSpec(
                env_type="rollout",
                env_url="http://env",
                max_turns=10,
                segment_max_images=4,
                restart_keep_turns=1,
                **older,
            )


class TestManifestHelpers:
    def test_tag_environment_passes_non_dict_through(self) -> None:
        from agilerl.arena.models.manifest import TrainingManifest

        assert TrainingManifest._tag_environment("raw") == "raw"

    def test_fold_network_mutation_passes_non_dict_through(self) -> None:
        from agilerl.arena.models.manifest import TrainingManifest

        assert TrainingManifest._fold_network_mutation_ranges("raw") == "raw"

    def test_fold_network_mutation_from_spec(self) -> None:
        from agilerl.arena.models.hpo import MutationSpec, NetworkMutationRanges
        from agilerl.arena.models.manifest import TrainingManifest
        from agilerl.arena.models.networks import MlpSpec, QNetworkSpec

        data = TrainingManifest._fold_network_mutation_ranges(
            {
                "algorithm": {"name": "DQN"},
                "environment": {"name": "CartPole-v1"},
                "network": QNetworkSpec(
                    encoder_config=MlpSpec(hidden_size=[64]),
                    head_config=MlpSpec(hidden_size=[64]),
                ),
                "mutation": MutationSpec(
                    network=NetworkMutationRanges(
                        min_latent_dim=8,
                        encoder={"min_hidden_layers": 1},
                    )
                ),
            }
        )
        assert data["network"]["min_latent_dim"] == 8

    def test_empty_deferred_network_is_omitted_from_payload(self) -> None:
        payload = TrainingManifest.get_validated({**_manifest(), "network": {}})
        assert "network" not in payload


class TestHpoHelpers:
    def test_frequency_ratio_errors(self) -> None:
        from agilerl.arena.models.hpo import resolve_frequency_ratios

        resolve_frequency_ratios(None, 3)
        with pytest.raises(ValueError, match="must have length"):
            resolve_frequency_ratios([1], 2)
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_frequency_ratios([0, 1], 2)
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_frequency_ratios([1, 1], 2)

    def test_selection_bracket_errors(self) -> None:
        from agilerl.arena.models.hpo import resolve_selection_brackets

        resolve_selection_brackets(8, 2, None, None, None, None, None)
        with pytest.raises(ValueError, match="population_size must be >= 6"):
            resolve_selection_brackets(4, 2, None, None, None, None, None)
        with pytest.raises(ValueError, match="n_subpopulations must be >= 2"):
            resolve_selection_brackets(8, 1, None, None, None, None, None)
        with pytest.raises(ValueError, match="must be divisible"):
            resolve_selection_brackets(9, 2, None, None, None, None, None)
        with pytest.raises(ValueError, match="must be >= 3"):
            resolve_selection_brackets(6, 3, None, None, None, None, None)
        with pytest.raises(ValueError, match="n_winners"):
            resolve_selection_brackets(8, 2, None, 0, 0, 1, 3)
        with pytest.raises(ValueError, match="n_survivors"):
            resolve_selection_brackets(8, 2, None, 1, -1, 1, 3)
        with pytest.raises(ValueError, match="n_open_for_migration"):
            resolve_selection_brackets(8, 2, None, 1, 0, 0, 3)
        with pytest.raises(ValueError, match="n_losers"):
            resolve_selection_brackets(8, 2, None, 1, 0, 3, 0)
        with pytest.raises(ValueError, match="must equal"):
            resolve_selection_brackets(8, 2, None, 1, 0, 1, 1)

    def test_mutation_ceiling(self) -> None:
        from types import SimpleNamespace

        from agilerl.arena.models.hpo import (
            MutationSpec,
            RLHyperparameter,
            mutation_ceiling,
        )

        assert mutation_ceiling(None, "lr", 3) == 3
        assert mutation_ceiling(MutationSpec(), "lr", 3) == 3
        spec = MutationSpec(
            rl_hp_selection={
                "learn_step": RLHyperparameter(min=1, max=10),
            }
        )
        assert mutation_ceiling(spec, "learn_step", 3) == 10
        assert mutation_ceiling(SimpleNamespace(rl_hp_selection="nope"), "lr", 3) == 3


class TestReplayBufferParse:
    def test_non_mapping_passthrough(self) -> None:
        from agilerl.arena.models.training import (
            ReplayBufferSpec,
            _parse_buffer_section,
        )

        spec = ReplayBufferSpec()
        assert _parse_buffer_section(spec) is spec
        assert _parse_buffer_section("raw") == "raw"


def _extra(cls: type) -> str | None:
    cfg = getattr(cls, "model_config", None) or {}
    extra = cfg.get("extra") if hasattr(cfg, "get") else None
    if extra is None:
        return None
    return extra if isinstance(extra, str) else getattr(extra, "value", str(extra))


def _base_models(package) -> list[type]:
    models: list[type] = []
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            try:
                if issubclass(obj, BaseModel) and obj is not BaseModel:
                    models.append(obj)
            except TypeError:
                continue
    return models


class TestPydanticModelsForbidExtra:
    def test_arena_models_forbid_extra(self) -> None:
        import agilerl.arena.inference as inference
        import agilerl.arena.models as models

        # These parse a subset of inference JSON (results, title/created_by, success).
        wire_subset = {
            "agilerl.arena.inference.agent.PredictResult",
            "agilerl.arena.inference.agent.SessionDetail",
            "agilerl.arena.inference.agent.SessionInfo",
        }
        found = (*_base_models(models), *_base_models(inference))
        assert found
        extra_by_name = {
            f"{cls.__module__}.{cls.__name__}": _extra(cls) for cls in found
        }
        missing = {name for name, extra in extra_by_name.items() if extra != "forbid"}

        assert missing == wire_subset
        assert all(extra_by_name[name] == "ignore" for name in wire_subset)
