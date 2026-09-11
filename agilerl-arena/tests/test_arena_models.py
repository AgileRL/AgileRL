# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena manifest models that live only in agilerl-arena."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from io import StringIO
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from agilerl.arena.models import (
    MANIFEST_REGISTRY,
    ReplayBufferSpec,
    TrainingManifest,
    TrainingSpec,
)
from agilerl.arena.models.algorithms.dqn import DQNSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.env import GymEnvSpec, LLMEnvType
from agilerl.arena.models.manifest import _resolve_algorithm
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
        assert spec.num_envs == 16
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


class TestLLMEnvType:
    def test_str(self) -> None:
        assert str(LLMEnvType.ROLLOUT) == "rollout"
        assert str(LLMEnvType.DATASET) == "dataset"


class TestAlgorithmRegistry:
    def test_get_unknown_name_lists_registered(self) -> None:
        with pytest.raises(KeyError, match="No registry entry for algorithm 'NOPE'"):
            MANIFEST_REGISTRY.get("NOPE")

    def test_create_applies_alias_implies(self) -> None:
        spec = MANIFEST_REGISTRY.create("Recurrent PPO")
        assert spec.recurrent is True
        assert spec.name == "Recurrent PPO"

    def test_create_rejects_conflicting_implied_field(self) -> None:
        with pytest.raises(ValueError, match="implies"):
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
    def test_valid_answer_pattern_compiles(self) -> None:
        spec = GRPOSpec(group_size=2, answer_pattern=r"<answer>.*</answer>")
        assert spec.answer_pattern == r"<answer>.*</answer>"

    def test_explicit_none_answer_pattern(self) -> None:
        spec = GRPOSpec.model_validate({"group_size": 2, "answer_pattern": None})
        assert spec.answer_pattern is None

    def test_invalid_answer_pattern_regex(self) -> None:
        with pytest.raises(ValidationError, match="not a valid regular expression"):
            GRPOSpec(group_size=2, answer_pattern="(")

    def test_answer_continuation_requires_pattern(self) -> None:
        with pytest.raises(ValidationError, match="answer_continuation requires"):
            GRPOSpec(group_size=2, answer_continuation=True)

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

    def test_deepspeed_activation_checkpointing_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="activation_checkpointing"):
            GRPOSpec(group_size=2, deepspeed={"activation_checkpointing": {}})

    def test_deepspeed_gradient_clipping_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="gradient_clipping"):
            GRPOSpec(group_size=2, deepspeed={"gradient_clipping": 1.0})

    def test_deepspeed_overrides_without_ignored_keys(self) -> None:
        spec = GRPOSpec(group_size=2, deepspeed={"train_batch_size": 8})
        assert spec.deepspeed == {"train_batch_size": 8}

    def test_mini_batch_must_be_a_multiple_of_micro_batch(self) -> None:
        with pytest.raises(ValidationError, match="not a multiple"):
            GRPOSpec(group_size=2, mini_batch_size=3, micro_batch_size_per_gpu=2)

    def test_mini_batch_multiple_of_micro_batch(self) -> None:
        spec = GRPOSpec(group_size=2, mini_batch_size=4, micro_batch_size_per_gpu=2)
        assert spec.mini_batch_size == 4


class TestGRPOClipCoef:
    def test_rejects_negative_scalar(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to zero"):
            GRPOSpec(group_size=2, clip_coef=-0.1)

    def test_rejects_pair_of_wrong_length(self) -> None:
        with pytest.raises(ValidationError, match="exactly two values"):
            GRPOSpec(group_size=2, clip_coef=[0.1])

    def test_rejects_non_numeric_pair(self) -> None:
        with pytest.raises(ValidationError, match="must be floats"):
            GRPOSpec(group_size=2, clip_coef=["a", "b"])

    def test_rejects_scalar_above_one(self) -> None:
        with pytest.raises(ValidationError, match=r"must be <= 1\.0"):
            GRPOSpec(group_size=2, clip_coef=1.5)

    def test_rejects_non_numeric_clip_coef(self) -> None:
        with pytest.raises(ValidationError, match="float or a list/tuple"):
            GRPOSpec(group_size=2, clip_coef="wide")


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


class TestLLMEnvSpecSurfaces:
    def test_name_from_dataset_and_urls(self) -> None:
        from agilerl.arena.models.env import LLMEnvSpec

        dataset = LLMEnvSpec(
            env_type="rollout",
            dataset="rows",
            rubric_file_path="rubric.py",
            prompt_template={"user_0": "{q}"},
        )
        assert dataset.name == "rows"
        assert dataset.dataset_backed_rollout is True

        url = LLMEnvSpec(env_type="rollout", env_url="http://env", max_turns=1)
        assert url.name == "http://env"

        urls = LLMEnvSpec(
            env_type="rollout", env_url=["http://a", "http://b"], max_turns=1
        )
        assert urls.name == "http://a"

        image = LLMEnvSpec(env_type="rollout", env_image="env:latest")
        assert image.name == "env:latest"

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
