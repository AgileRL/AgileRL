# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""What a manifest accepts, rejects, and resolves to."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, ValidationError

from agilerl.models import (
    MANIFEST_REGISTRY,
    DPOSpec,
    GRPOSpec,
    LLMEnvSpec,
    LLMRolloutBufferSpec,
    MultiFrequencySelectionSpec,
    SFTSpec,
    TrainingManifest,
    TrainingSpec,
    __version__,
    manifest_schema,
)
from agilerl.models.algorithms import (
    CISPOSpec,
    GSPOSpec,
    LLMPPOSpec,
    LLMREINFORCESpec,
    RolloutLLMSpec,
)
from agilerl.models.algorithms.rollout_llm import records_sampling_supports
from agilerl.models.defaults import default_verdict
from agilerl.models.env import LLMEnvType
from agilerl.models.hpo import (
    RLHyperparameter,
    require_mutation_range_inside_field,
)
from agilerl.models.manifest import mutates, strips_multimodal_towers
from agilerl.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.models.networks import LoraConfigDict
from agilerl.models.schema import (
    LORA_CONFIG_REF,
    _inline_lora_config_inplace,
    algorithm_name_if,
    dataset_backed_rollout_if,
    environment_dataset_identity_if,
    environment_dataset_type_if,
    environment_rollout_type_if,
    environment_then,
    registered_algorithm_names,
    strip_non_form_algorithm_fields,
    training_then,
)
from agilerl.models.verdict import (
    CLASSIC_RL_UNAVAILABLE_MESSAGE,
    HPO_UNAVAILABLE_MESSAGE,
    catalog_gpu_share_warnings,
    hpo_fields,
    is_classic_rl,
    verdict,
)

GRPO = {
    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
    "environment": {
        "env_type": "rollout",
        "dataset": "openai/gsm8k",
        "reward_file_path": "reward.py",
        "prompt_template": {"user_0": "{question}"},
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_context_length": 512,
    },
    "training": {"max_steps": 100},
}

DQN = {
    "algorithm": {"name": "DQN", "lr": 0.001},
    "environment": {"name": "LunarLander-v3", "num_envs": 8},
}


def built_in_run(environment: dict) -> dict:
    """GRPO on a bundled environment with a vision model and the gym's default context."""
    doc = copy.deepcopy(GRPO)
    doc["environment"] = environment
    doc["network"]["pretrained_model_name_or_path"] = "Qwen/Qwen3.5-0.8B"
    doc["network"].pop("max_context_length", None)
    return doc


def manifest(base: dict, **overrides: dict) -> dict:
    """Deep-copy *base* and merge a per-section override into it."""
    out = copy.deepcopy(base)
    for section, values in overrides.items():
        out.setdefault(section, {}).update(values)
    return out


def schema_errors(payload: dict) -> list[str]:
    """Schema violations in *payload*.

    :param payload: A manifest document.
    :return: One ``path: message`` line per violation.
    """
    errors = Draft202012Validator(manifest_schema()).iter_errors(payload)
    return [f"{list(e.path)}: {e.message}" for e in errors]


class TestAcceptance:
    def test_llm_manifest_validates(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert isinstance(validated.algorithm, GRPOSpec)
        assert isinstance(validated.environment, LLMEnvSpec)

    def test_every_registered_algorithm_is_reachable_by_name(self) -> None:
        for name in MANIFEST_REGISTRY.names():
            assert MANIFEST_REGISTRY.get(name).__name__.endswith("Spec")

    @pytest.mark.parametrize("name", ["LLMPPO", "LLMREINFORCE"])
    def test_llm_policy_gradient_manifests_take_advantage_granularity(
        self, name: str
    ) -> None:
        # The constructors accept the field; the manifest must be able to set it.
        doc = manifest(GRPO, algorithm={"advantage_granularity": "token"})
        doc["algorithm"] = {
            k: v for k, v in doc["algorithm"].items() if k != "group_size"
        }
        doc["algorithm"]["name"] = name
        validated = TrainingManifest.model_validate(doc)
        assert validated.algorithm.advantage_granularity == "token"


class TestUnknownKeys:
    def test_unknown_algorithm_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"definitely_not_a_field": 1})
            )

    def test_unknown_training_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(manifest(GRPO, training={"nope": 1}))

    def test_unknown_top_level_section_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate({**GRPO, "nope": {}})

    def test_unknown_algorithm_without_env_type_is_a_validation_error(self) -> None:
        # env_type is inferred from the algorithm, so a misspelled name reaches
        # the registry from the inference path first.
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "GRPOO"}
        doc["environment"] = {"dataset": "openai/gsm8k"}
        with pytest.raises(ValidationError, match="not a registered algorithm"):
            TrainingManifest.model_validate(doc)

    def test_unknown_environment_key_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(GRPO, environment={"use_vectorized_env": True})
            )

    def test_peft_lora_r_is_an_error(self) -> None:
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(
                manifest(GRPO, network={"lora_config": {"r": 8}})
            )

    def test_vllm_enforce_eager_is_an_error(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"vllm_config\.enforce_eager\n  Extra inputs are not permitted",
        ):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"vllm_config": {"enforce_eager": True}})
            )


class TestResolvedMemoryFields:
    def test_manifest_without_them_validates(self) -> None:
        validated = TrainingManifest.get_validated(
            manifest(GRPO, algorithm={"fsdp": {"shard_group_size": 2}}), mode="python"
        )

        assert validated.algorithm.micro_batch_size_per_gpu is None
        assert validated.algorithm.fsdp.reshard_after_forward is None
        assert validated.algorithm.fsdp.activation_offload is None

    def test_picked_values_validate_on_the_model(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={
                "micro_batch_size_per_gpu": 2,
                "fsdp": {"reshard_after_forward": False, "activation_offload": False},
            },
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.micro_batch_size_per_gpu == 2
        assert validated.algorithm.fsdp.reshard_after_forward is False

    def test_picked_shard_frozen_tp_validates_on_the_model(self) -> None:
        doc = manifest(GRPO, algorithm={"fsdp": {"tp": 2, "shard_frozen_tp": False}})

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.fsdp.shard_frozen_tp is False
        assert validated.algorithm.fsdp.frozen_tp_resident is True

    def test_picked_save_moe_collectives_validates_on_the_model(self) -> None:
        doc = manifest(GRPO, algorithm={"fsdp": {"save_moe_collectives": True}})

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.fsdp.save_moe_collectives is True

    @pytest.mark.parametrize(
        ("algorithm", "dotted"),
        [
            ({"micro_batch_size_per_gpu": 1}, "micro_batch_size_per_gpu"),
            ({"micro_batch_tokens_per_gpu": 4096}, "micro_batch_tokens_per_gpu"),
            ({"gradient_checkpointing": False}, "gradient_checkpointing"),
            ({"activation_offload": True}, "activation_offload"),
            ({"routed_expert_chunk_mib": 512}, "routed_expert_chunk_mib"),
            ({"fsdp": {"reshard_after_forward": True}}, "fsdp.reshard_after_forward"),
            ({"fsdp": {"optim_cpu_offload": True}}, "fsdp.optim_cpu_offload"),
            (
                {"fsdp": {"routed_expert_chunk_mib": 512}},
                "fsdp.routed_expert_chunk_mib",
            ),
            ({"fsdp": {"activation_offload": False}}, "fsdp.activation_offload"),
            ({"fsdp": {"cpu_offload": True}}, "fsdp.cpu_offload"),
            ({"fsdp": {"shard_frozen_tp": False}}, "fsdp.shard_frozen_tp"),
            ({"fsdp": {"save_moe_collectives": True}}, "fsdp.save_moe_collectives"),
            (
                {"fsdp": {"checkpoint_every_n_blocks": 2}},
                "fsdp.checkpoint_every_n_blocks",
            ),
        ],
    )
    def test_a_pinned_field_is_refused(self, algorithm: dict, dotted: str) -> None:
        with pytest.raises(ValueError, match=rf"algorithm\.{dotted} cannot be set"):
            TrainingManifest.get_validated(manifest(GRPO, algorithm=algorithm))

    def test_every_pinned_field_is_named(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={
                "micro_batch_size_per_gpu": 1,
                "fsdp": {"optim_cpu_offload": True},
            },
        )

        with pytest.raises(
            ValueError,
            match=r"algorithm\.micro_batch_size_per_gpu, "
            r"algorithm\.fsdp\.optim_cpu_offload cannot be set",
        ):
            TrainingManifest.get_validated(doc)

    def test_a_section_without_an_algorithm_mapping_is_left_to_validation(
        self,
    ) -> None:
        with pytest.raises(ValidationError):
            TrainingManifest.get_validated(manifest(GRPO) | {"algorithm": None})


class TestMalformedSections:
    # Shape errors must surface as ValidationError like any other bad input.
    # A raw TypeError would escape verdict() and crash the CLI / submit path.

    def test_non_mapping_algorithm_section_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            TrainingManifest.model_validate(
                {**copy.deepcopy(GRPO), "algorithm": "GRPO"}
            )

    def test_wrong_typed_clip_coef_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="valid number"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"clip_coef": "abc"})
            )

    def test_non_numeric_clip_coef_pair_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="valid number"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"clip_coef": [None, 0.2]})
            )

    def test_a_field_named_self_is_a_validation_error(self) -> None:
        # `self` must not collide with the registry's create() parameters.
        with pytest.raises(ValidationError, match="not permitted"):
            TrainingManifest.model_validate(manifest(GRPO, algorithm={"self": 1}))

    def test_hpo_range_with_min_above_max_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to max"):
            TrainingManifest.model_validate(
                manifest(
                    GRPO,
                    mutation={"rl_hp_selection": {"lr": {"min": 1.0, "max": 0.0001}}},
                )
            )


class TestResolution:
    def test_evo_steps_defaults_to_the_algorithms_cadence(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert validated.training.evo_steps == GRPOSpec.default_evo_steps

    def test_evo_steps_is_never_longer_than_the_run(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, training={"max_steps": 10})
        )
        assert validated.training.evo_steps == 10

    def test_explicit_evo_steps_wins(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, training={"evo_steps": 50})
        )
        assert validated.training.evo_steps == 50

    def test_network_is_lifted_onto_the_llm_algorithm(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert (
            validated.algorithm.pretrained_model_name_or_path
            == "Qwen/Qwen2.5-0.5B-Instruct"
        )
        assert validated.algorithm.max_model_len == 512

    def test_colocated_rollout_fills_default_vllm_config(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert validated.training.rollout_mode == "colocated"
        assert validated.algorithm.vllm_config is not None
        assert validated.algorithm.vllm_config.max_num_seqs is None
        assert validated.algorithm.vllm_config.gpu_memory_utilization == 0.9

    @pytest.mark.parametrize(
        "vllm_config",
        [None, {"max_num_seqs": 8}, {"gpu_memory_utilization": 0.5}],
        ids=["no_vllm_config", "share_unset", "share_set"],
    )
    def test_catalogue_engine_share_wins_over_the_manifest(
        self, vllm_config: dict | None
    ) -> None:
        # Arrange
        hub_id = "zai-org/GLM-5.3-Flash"
        doc = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": hub_id,
                "max_context_length": 4096,
            },
        )
        if vllm_config is not None:
            doc["algorithm"]["vllm_config"] = vllm_config

        # Act
        validated = TrainingManifest.model_validate(doc)

        # Assert
        assert SUPPORTED_MODEL_INFO[hub_id].rollout_gpu_memory_utilization == 0.85
        assert validated.algorithm.vllm_config.gpu_memory_utilization == 0.85

    def test_a_replaced_catalog_share_warns(self) -> None:
        # Arrange
        hub_id = "zai-org/GLM-5.3-Flash"
        doc = manifest(
            GRPO_ON_ARENA,
            network={
                "pretrained_model_name_or_path": hub_id,
                "max_context_length": 4096,
            },
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.5}},
        )

        # Act
        record = verdict(doc)

        # Assert
        assert record["ok"], record
        assert (
            "algorithm.vllm_config.gpu_memory_utilization is 0.5 in the manifest; "
            "Arena uses the catalog share 0.85."
        ) in record["warnings"]

    def test_an_unset_or_matching_catalog_share_does_not_warn(self) -> None:
        # Arrange
        hub_id = "zai-org/GLM-5.3-Flash"
        unset = manifest(
            GRPO_ON_ARENA,
            network={
                "pretrained_model_name_or_path": hub_id,
                "max_context_length": 4096,
            },
        )
        matching = manifest(
            unset,
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.85}},
        )

        # Act
        unset_record = verdict(unset)
        matching_record = verdict(matching)

        # Assert
        assert unset_record["ok"], unset_record
        assert matching_record["ok"], matching_record
        assert not any(
            "catalog share" in warning for warning in unset_record["warnings"]
        )
        assert not any(
            "catalog share" in warning for warning in matching_record["warnings"]
        )

    def test_a_manifest_share_without_a_catalog_share_does_not_warn(self) -> None:
        # Arrange
        doc = manifest(
            GRPO_ON_ARENA,
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.5}},
        )

        # Act
        record = verdict(doc)

        # Assert
        assert record["ok"], record
        assert (
            record["payload"]["algorithm"]["vllm_config"]["gpu_memory_utilization"]
            == 0.5
        )
        assert not any("catalog share" in warning for warning in record["warnings"])

    def test_catalog_share_warning_ignores_a_section_that_is_not_a_mapping(
        self,
    ) -> None:
        # Arrange
        rollout = TrainingManifest.model_validate(GRPO_ON_ARENA)

        # Act / Assert
        assert catalog_gpu_share_warnings(rollout, {"algorithm": "GRPO"}) == []

    def test_manifest_engine_share_is_kept_without_a_catalogue_share(self) -> None:
        # Arrange
        doc = manifest(GRPO, algorithm={"vllm_config": {"gpu_memory_utilization": 0.5}})

        # Act
        validated = TrainingManifest.model_validate(doc)

        # Assert
        assert validated.algorithm.vllm_config.gpu_memory_utilization == 0.5

    def test_empty_mutation_is_a_noop(self) -> None:
        validated = TrainingManifest.model_validate(manifest(GRPO, mutation={}))
        assert not mutates(validated.mutation)
        assert "mutation" not in validated.to_payload()

    def test_dormant_threshold_is_accepted_and_omitted_from_the_payload(self) -> None:
        payload = TrainingManifest.get_validated(
            manifest(
                GRPO,
                mutation={"mutation_sd": 0.2, "dormant_threshold": 0.05},
            )
        )
        assert payload["mutation"]["mutation_sd"] == 0.2
        assert "dormant_threshold" not in payload["mutation"]
        spec = TrainingManifest.model_validate(
            manifest(GRPO, mutation={"dormant_threshold": 0.05})
        )
        assert spec.mutation.dormant_threshold == 0.05

    def test_llm_model_id_on_algorithm_only_round_trips(self) -> None:
        raw = {
            "algorithm": {
                "name": "GRPO",
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "group_size": 4,
                "batch_size": 2,
            },
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
            "training": {"max_steps": 100},
        }
        payload = TrainingManifest.model_validate(raw).to_payload()
        assert "network" not in payload
        reloaded = TrainingManifest.model_validate(payload)
        assert (
            reloaded.algorithm.pretrained_model_name_or_path
            == "Qwen/Qwen2.5-0.5B-Instruct"
        )

    def test_mf_pbt_brackets_resolve_against_the_population(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                training={"pop_size": 8},
                selection_strategy={"strategy": "multi_frequency"},
            )
        )
        strategy = validated.selection_strategy
        assert isinstance(strategy, MultiFrequencySelectionSpec)
        assert (
            strategy.n_winners
            + strategy.n_survivors
            + strategy.n_open_for_migration
            + strategy.n_losers
            == 4
        )


class TestNothingIsStrippedInTransit:
    @pytest.mark.parametrize(
        ("field", "value", "extra"),
        [
            ("use_separate_reference_adapter", True, {}),
            ("micro_batch_size_per_gpu", 4, {}),
        ],
    )
    def test_llm_fields_survive_a_round_trip(
        self, field: str, value: object, extra: dict
    ) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={field: value, **extra})
        ).to_payload()
        assert payload["algorithm"][field] == value

    def test_max_output_tokens_survives(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={"max_output_tokens": 64})
        ).to_payload()
        assert payload["algorithm"]["max_output_tokens"] == 64

    def test_reward_fn_name_survives(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, environment={"reward_fn_name": "combined_rewards"})
        ).to_payload()
        assert payload["environment"]["rubric_name"] == "combined_rewards"

    def test_unknown_vllm_config_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(
                manifest(
                    GRPO,
                    algorithm={
                        "vllm_config": {
                            "max_num_seqs": 4,
                            "reasoning_parser": "x",
                        }
                    },
                )
            )

    def test_explicit_vllm_engine_args_are_kept(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                algorithm={"vllm_engine_args": {"reasoning_parser": "gemma4"}},
            )
        )
        assert validated.algorithm.vllm_engine_args == {"reasoning_parser": "gemma4"}

    def test_an_unsupported_reasoning_parser_is_rejected(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"reasoning_parser='mistral' is not a supported reasoning parser",
        ):
            TrainingManifest.model_validate(
                manifest(
                    GRPO,
                    algorithm={"vllm_engine_args": {"reasoning_parser": "mistral"}},
                )
            )

    def test_a_payload_revalidates(self) -> None:
        payload = TrainingManifest.model_validate(GRPO).to_payload()
        assert TrainingManifest.model_validate(payload).algorithm.name == "GRPO"

    def test_explicit_false_reference_adapter_survives(self) -> None:
        payload = TrainingManifest.get_validated(
            manifest(GRPO, algorithm={"use_separate_reference_adapter": False}),
            mode="json",
        )
        assert payload["algorithm"]["use_separate_reference_adapter"] is False

    def test_reference_update_interval_survives(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, algorithm={"reference_update_interval": 25})
        ).to_payload()
        assert payload["algorithm"]["reference_update_interval"] == 25

    def test_reference_update_interval_defaults_to_unset(self) -> None:
        validated = TrainingManifest.model_validate(manifest(GRPO))
        assert validated.algorithm.reference_update_interval is None

    def test_reference_update_interval_rejects_zero(self) -> None:
        with pytest.raises(ValidationError, match="reference_update_interval"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"reference_update_interval": 0})
            )

    def test_dpo_reference_update_interval(self) -> None:
        assert DPOSpec().reference_update_interval is None
        assert DPOSpec(reference_update_interval=25).reference_update_interval == 25
        with pytest.raises(ValidationError, match="reference_update_interval"):
            DPOSpec(reference_update_interval=0)


class TestEnvironment:
    def test_every_environment_branch_can_be_versioned(self) -> None:
        # A custom environment's version selects which uploaded archive runs.
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO, environment={"env_type": "rollout", "dataset": "x", "version": 3}
            )
        )
        assert validated.environment.version == 3


class TestCrossSectionConsistency:
    def test_llm_algorithm_requires_a_pretrained_network(self) -> None:
        broken = copy.deepcopy(GRPO)
        broken["network"] = {"latent_dim": 32}
        with pytest.raises(ValidationError, match="pretrained_model_name_or_path"):
            TrainingManifest.model_validate(broken)

    def test_an_unsupported_api_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="apiVersion"):
            TrainingManifest.model_validate({**GRPO, "apiVersion": "agilerl.com/v99"})

    def test_async_rollout_defaults_a_missing_buffer_to_the_llm_rollout_buffer(
        self,
    ) -> None:
        payload = manifest(
            GRPO,
            environment={
                "env_type": "rollout",
                "factory": "gem:make",
                "entrypoint": "game:Sudoku-v0",
                "dataset": None,
                "reward_file_path": None,
                "prompt_template": None,
            },
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
        )

        validated = TrainingManifest.model_validate(payload)

        assert validated.replay_buffer == LLMRolloutBufferSpec()
        assert validated.to_payload()["replay_buffer"]["kind"] == "llm"

    def test_async_rollout_accepts_the_llm_rollout_buffer(self) -> None:
        payload = manifest(
            GRPO,
            environment={
                "env_type": "rollout",
                "factory": "gem:make",
                "entrypoint": "game:Sudoku-v0",
                "dataset": None,
                "reward_file_path": None,
                "prompt_template": None,
            },
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )

        validated = TrainingManifest.model_validate(payload)

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)

    def test_an_anchored_constrain_answer_pattern_is_rejected(self) -> None:
        broken = manifest(
            GRPO,
            algorithm={"constrain_answer_pattern": r"^<answer>[A-D]</answer>$"},
        )
        with pytest.raises(ValidationError, match="carries an anchor"):
            TrainingManifest.model_validate(broken)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("use_sequence_packing", True), ("attn_implementation", "flex_attention")],
    )
    def test_packing_and_attention_backend_are_not_manifest_fields(
        self, field: str, value: object
    ) -> None:
        broken = manifest(GRPO, algorithm={field: value})

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TrainingManifest.model_validate(broken)


def group_replay_manifest(**replay_buffer: object) -> dict:
    """An async GRPO manifest with group replay on the LLM buffer."""
    doc = manifest(
        GRPO,
        training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
    )
    doc["replay_buffer"] = {"kind": "llm", "group_replay_max_age": 10, **replay_buffer}
    return doc


class TestRecordsSamplingSupports:
    def test_top_k_sampling_records_supports(self) -> None:
        assert records_sampling_supports(1.0, 20, 1.0, 0.05, r"\\boxed\{\d+\}")

    def test_unfiltered_sampling_records_none(self) -> None:
        assert not records_sampling_supports(0.9, 0, 1.0, 0.0, None)

    def test_greedy_top_k_sampling_records_none(self) -> None:
        assert not records_sampling_supports(0.0, 20, 1.0, 0.0, None)


class TestRolloutLLMSpecSamplingSupport:
    @pytest.mark.parametrize(
        "sampling",
        [
            {},
            {"top_k": 20, "top_p": 0.95, "min_p": 0.05},
            {"top_k": 0, "top_p": 1.0, "min_p": 0.0},
            {"top_k": 20, "constrain_answer_pattern": r"\\boxed\{\d+\}"},
        ],
    )
    def test_sampling_with_a_reported_support_validates(self, sampling: dict) -> None:
        validated = TrainingManifest.model_validate(manifest(GRPO, algorithm=sampling))

        assert validated.algorithm.top_k == sampling.get("top_k", 50)

    @pytest.mark.parametrize(
        ("sampling", "narrowed"),
        [
            ({"top_k": 0, "top_p": 0.9}, "top_p"),
            ({"top_k": 0, "min_p": 0.1}, "min_p"),
            (
                {"top_k": 0, "constrain_answer_pattern": r"\\boxed\{\d+\}"},
                "constrain_answer_pattern",
            ),
        ],
    )
    def test_a_narrowed_distribution_without_a_reported_support_is_rejected(
        self, sampling: dict, narrowed: str
    ) -> None:
        with pytest.raises(
            ValidationError, match=f"{narrowed} narrows the distribution rollout"
        ):
            TrainingManifest.model_validate(manifest(GRPO, algorithm=sampling))


def speculative_manifest(model: str, speculative_config: object, **algorithm) -> dict:
    """GRPO manifest for *model* with ``vllm_engine_args.speculative_config``."""
    return manifest(
        GRPO,
        algorithm={
            "vllm_engine_args": {"speculative_config": speculative_config},
            **algorithm,
        },
        network={"pretrained_model_name_or_path": model},
    )


class TestRolloutLLMSpecCheckSpeculativeSampling:
    def test_drafting_without_min_p_on_a_text_model_validates(self) -> None:
        doc = speculative_manifest("Qwen/Qwen2.5-0.5B-Instruct", {"method": "ngram"})

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.vllm_engine_args["speculative_config"] == {
            "method": "ngram"
        }

    def test_null_speculative_config_validates_with_min_p(self) -> None:
        doc = speculative_manifest(
            "Qwen/Qwen2.5-0.5B-Instruct", None, top_k=20, min_p=0.05
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.min_p == 0.05

    def test_null_speculative_config_validates_on_a_vision_model(self) -> None:
        doc = speculative_manifest("google/gemma-4-E4B-it", None)

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.vllm_engine_args["speculative_config"] is None

    def test_drafting_with_min_p_is_rejected(self) -> None:
        doc = speculative_manifest(
            "Qwen/Qwen2.5-0.5B-Instruct", {"method": "ngram"}, top_k=20, min_p=0.05
        )

        with pytest.raises(
            ValidationError,
            match=r"vLLM rejects every request that sets min_p on a drafting engine\. "
            r"Set vllm_engine_args\.speculative_config to null",
        ):
            TrainingManifest.model_validate(doc)

    def test_drafting_on_a_vision_model_is_rejected(self) -> None:
        doc = speculative_manifest("google/gemma-4-E4B-it", {"method": "ngram"})

        with pytest.raises(
            ValidationError,
            match=r"sets logit_bias on a drafting engine; google/gemma-4-E4B-it "
            r"rollouts mask the tokenizer's image token",
        ):
            TrainingManifest.model_validate(doc)


class TestTrainingManifestMutatedSamplingSupport:
    def test_top_p_mutation_validates_with_a_reported_support(self) -> None:
        doc = manifest(
            GRPO,
            mutation={"rl_hp_selection": {"top_p": {"min": 0.8, "max": 1.0}}},
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.mutation.rl_hp_selection["top_p"].min == 0.8

    def test_top_p_mutation_without_a_reported_support_is_rejected(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"top_k": 0},
            mutation={"rl_hp_selection": {"top_p": {"min": 0.8, "max": 1.0}}},
        )

        with pytest.raises(
            ValidationError,
            match=r"mutation\.rl_hp_selection can reach sampling rollout cannot "
            r"score: top_p narrows",
        ):
            TrainingManifest.model_validate(doc)

    @pytest.mark.parametrize("top_k", [0, 20])
    def test_temperature_mutation_on_the_open_floor_is_rejected(
        self, top_k: int
    ) -> None:
        doc = manifest(
            GRPO,
            algorithm={"top_k": top_k},
            mutation={"rl_hp_selection": {"temperature": {"min": 0.0, "max": 1.2}}},
        )

        with pytest.raises(ValidationError, match=r"min 0.0 must be > 0.0"):
            TrainingManifest.model_validate(doc)

    def test_temperature_mutation_above_the_field_ceiling_is_rejected(self) -> None:
        doc = manifest(
            GRPO,
            mutation={"rl_hp_selection": {"temperature": {"min": 0.2, "max": 3.0}}},
        )

        with pytest.raises(ValidationError, match=r"max 3.0 is above 2.0"):
            TrainingManifest.model_validate(doc)


class TestMutationRangeInsideField:
    def test_a_window_inside_the_field_is_accepted(self) -> None:
        bounds = RLHyperparameter(min=0.2, max=1.0)

        require_mutation_range_inside_field(
            bounds, GRPOSpec.model_fields["temperature"]
        )

    def test_a_manifest_keeps_a_window_inside_the_field(self) -> None:
        doc = manifest(
            GRPO, mutation={"rl_hp_selection": {"lr": {"min": 1e-4, "max": 0.01}}}
        )

        validated = TrainingManifest.model_validate(doc)

        lr = validated.mutation.rl_hp_selection["lr"]
        assert lr.min == 1e-4
        assert lr.max == 0.01

    def test_an_open_floor_and_a_ceiling_are_both_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"min 0.0 must be > 0.0"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=0.0, max=0.5), GRPOSpec.model_fields["lr"]
            )

        with pytest.raises(ValueError, match=r"max 5.0 is above 1.0"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=1e-4, max=5.0), GRPOSpec.model_fields["lr"]
            )

    def test_an_inclusive_window_below_or_above_the_field_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"min -1.0 is below 0.0"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=-1.0, max=1.0),
                GRPOSpec.model_fields["clip_coef"],
            )

        with pytest.raises(ValueError, match=r"max 9.0 is above 1.0"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=0.0, max=9.0),
                GRPOSpec.model_fields["clip_coef"],
            )

    def test_an_open_integer_window_on_either_end_is_rejected(self) -> None:
        class Sample(BaseModel):
            value: int = Field(gt=0, lt=10)

        field = Sample.model_fields["value"]

        require_mutation_range_inside_field(RLHyperparameter(min=1, max=9), field)
        with pytest.raises(ValueError, match=r"min 0.0 must be > 0"):
            require_mutation_range_inside_field(RLHyperparameter(min=0, max=9), field)
        with pytest.raises(ValueError, match=r"max 10.0 must be < 10"):
            require_mutation_range_inside_field(RLHyperparameter(min=1, max=10), field)

    def test_an_optional_float_rejects_either_open_end(self) -> None:
        class Sample(BaseModel):
            value: float | None = Field(default=None, gt=0.0, lt=1.0)

        field = Sample.model_fields["value"]

        with pytest.raises(ValueError, match=r"min 0.0 must be > 0.0"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=0.0, max=0.5), field
            )
        with pytest.raises(ValueError, match=r"max 1.0 must be < 1.0"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=0.5, max=1.0), field
            )

    def test_an_optional_int_rejects_a_min_below_the_floor(self) -> None:
        class Sample(BaseModel):
            value: int | None = Field(default=None, ge=1)

        with pytest.raises(ValueError, match=r"min 0.0 is below 1"):
            require_mutation_range_inside_field(
                RLHyperparameter(min=0, max=8), Sample.model_fields["value"]
            )

    def test_a_window_that_misses_the_field_names_the_limits(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"range \[2.0, 3.0\] leaves the field limits \(> 0.0 and <= 1.0\)",
        ):
            require_mutation_range_inside_field(
                RLHyperparameter(min=2, max=3), GRPOSpec.model_fields["lr"]
            )

    def test_a_manifest_rejects_a_learning_rate_above_the_field(self) -> None:
        doc = manifest(
            GRPO, mutation={"rl_hp_selection": {"lr": {"min": 1e-4, "max": 5}}}
        )

        with pytest.raises(ValidationError, match=r"max 5.0 is above 1.0"):
            TrainingManifest.model_validate(doc)

    def test_a_manifest_rejects_a_window_that_misses_the_field(self) -> None:
        doc = manifest(GRPO, mutation={"rl_hp_selection": {"lr": {"min": 2, "max": 3}}})

        with pytest.raises(ValidationError, match="leaves the field limits"):
            TrainingManifest.model_validate(doc)

    def test_an_aliased_learning_rate_uses_the_actor_field_limit(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "LLMPPO"}
        doc["mutation"] = {"rl_hp_selection": {"lr": {"min": 1e-8, "max": 5}}}

        with pytest.raises(ValidationError, match=r"max 5.0 is above 1.0"):
            TrainingManifest.model_validate(doc)

    def test_an_output_token_window_below_one_is_rejected(self) -> None:
        doc = manifest(
            GRPO,
            mutation={"rl_hp_selection": {"max_output_tokens": {"min": 0, "max": 64}}},
        )

        with pytest.raises(ValidationError, match=r"min 0.0 is below 1"):
            TrainingManifest.model_validate(doc)


def env_rollout(
    environment: dict, model: str = "Qwen/Qwen3.5-0.8B", **overrides: dict
) -> dict:
    """GRPO over an env_url rollout serving *model*."""
    doc = manifest(GRPO, **overrides)
    doc["network"]["pretrained_model_name_or_path"] = model
    doc["environment"] = {
        "env_type": "rollout",
        "env_url": "http://env",
        "max_turns": 1,
        **environment,
    }
    return doc


def resolved_strip(doc: dict) -> bool:
    validated = TrainingManifest.model_validate(doc)
    assert isinstance(validated.algorithm, RolloutLLMSpec)
    assert isinstance(validated.environment, LLMEnvSpec)
    return strips_multimodal_towers(
        validated.algorithm, validated.environment.text_only
    )


class TestStripMultimodalTowersResolution:
    def test_text_env_on_a_vl_model_strips_the_towers(self) -> None:
        doc = env_rollout({"modality": "text"})

        assert resolved_strip(doc) is True

    def test_async_text_env_without_vllm_config_strips_the_towers(self) -> None:
        # Arrange
        doc = env_rollout(
            {"modality": "text", "max_turns": 20},
            model="Qwen/Qwen3.6-35B-A3B",
            training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
            replay_buffer={"kind": "llm"},
        )

        # Act
        payload = TrainingManifest.get_validated(doc)

        # Assert
        assert payload["training"]["rollout_mode"] == "async"
        assert resolved_strip(doc) is True
        assert "max_num_seqs" not in payload["algorithm"]["vllm_config"]
        assert payload["algorithm"]["vllm_config"]["gpu_memory_utilization"] == 0.9

    def test_dataset_rollout_on_a_vl_model_strips_the_towers(self) -> None:
        doc = manifest(
            GRPO, network={"pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B"}
        )

        assert resolved_strip(doc) is True

    def test_image_fields_without_modality_keep_the_towers(self) -> None:
        doc = env_rollout({"screenshot_field": "screenshot"})

        assert resolved_strip(doc) is False

    @pytest.mark.parametrize("modality", [None, "vision"])
    def test_unknown_or_vision_env_keeps_the_towers(self, modality: str | None) -> None:
        doc = env_rollout({"modality": modality})

        assert resolved_strip(doc) is False

    def test_vision_model_strips_by_its_modalities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        hub_id = "Qwen/Qwen3.5-0.8B"
        monkeypatch.setitem(
            SUPPORTED_MODEL_INFO,
            hub_id,
            SUPPORTED_MODEL_INFO[hub_id].model_copy(
                update={"language_model_prefix": None}
            ),
        )
        doc = env_rollout({"modality": "text"}, model=hub_id)

        # Act
        strip = resolved_strip(doc)

        # Assert
        assert strip is True

    def test_text_only_model_is_left_unstripped(self) -> None:
        doc = env_rollout({"modality": "text"}, model="Qwen/Qwen2.5-0.5B-Instruct")

        assert resolved_strip(doc) is False

    def test_strip_multimodal_towers_pin_is_rejected(self) -> None:
        doc = env_rollout(
            {"modality": "text"},
            algorithm={"vllm_config": {"strip_multimodal_towers": False}},
        )

        with pytest.raises(ValidationError, match="strip_multimodal_towers"):
            TrainingManifest.model_validate(doc)

    def test_text_modality_rejects_image_fields(self) -> None:
        doc = env_rollout({"modality": "text", "screenshot_field": "screenshot"})

        with pytest.raises(ValidationError, match="modality text means the env sends"):
            TrainingManifest.model_validate(doc)

    def test_vision_modality_rejects_a_dataset_rollout(self) -> None:
        doc = manifest(GRPO, environment={"modality": "vision"})

        with pytest.raises(ValidationError, match="modality vision needs image"):
            TrainingManifest.model_validate(doc)


class TestAsyncStopSequences:
    def test_colocated_rollout_keeps_stop_sequences(self) -> None:
        doc = env_rollout(
            {"modality": "text"},
            algorithm={"vllm_config": {"stop_sequences": ["</answer>"]}},
        )

        payload = TrainingManifest.get_validated(doc)

        assert payload["algorithm"]["vllm_config"]["stop_sequences"] == ["</answer>"]

    def test_async_rollout_rejects_stop_sequences(self) -> None:
        doc = env_rollout(
            {"modality": "text"},
            algorithm={"vllm_config": {"stop_sequences": ["</answer>"]}},
            training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
            replay_buffer={"kind": "llm"},
        )

        with pytest.raises(ValidationError, match="stop_sequences is not supported"):
            TrainingManifest.model_validate(doc)


class TestCpuOffloadColocated:
    def test_colocated_rollout_accepts_full_cpu_offload(self) -> None:
        doc = manifest(GRPO, algorithm={"fsdp": {"cpu_offload": True}})

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.fsdp is not None
        assert validated.algorithm.fsdp.cpu_offload is True

    def test_async_rollout_rejects_full_cpu_offload(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"fsdp": {"cpu_offload": True}},
            training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
            replay_buffer={"kind": "llm"},
        )

        with pytest.raises(ValidationError, match=r"fsdp\.cpu_offload needs colocated"):
            TrainingManifest.model_validate(doc)

    def test_sft_accepts_full_cpu_offload(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "SFT", "fsdp": {"cpu_offload": True}}
        doc["environment"] = {"env_type": "dataset", "objective": "sft", "dataset": "d"}

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.fsdp is not None
        assert validated.algorithm.fsdp.cpu_offload is True

    def test_async_rollout_accepts_optimizer_cpu_offload(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"fsdp": {"optim_cpu_offload": True}},
            training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
            replay_buffer={"kind": "llm"},
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.fsdp is not None
        assert validated.algorithm.fsdp.optim_cpu_offload is True


class TestRolloutEngines:
    def test_rollout_algorithm_accepts_rollout_engines(self) -> None:
        doc = manifest(GRPO, training={"rollout_engines_per_agent": 2})

        validated = TrainingManifest.model_validate(doc)

        assert validated.training.rollout_engines_per_agent == 2

    @pytest.mark.parametrize(
        ("name", "objective"), [("SFT", "sft"), ("DPO", "preference")]
    )
    def test_dataset_algorithm_accepts_no_rollout_engines(
        self, name: str, objective: str
    ) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": name}
        doc["environment"] = {
            "env_type": "dataset",
            "objective": objective,
            "dataset": "d",
        }

        validated = TrainingManifest.model_validate(doc)

        assert validated.training.rollout_engines_per_agent == 0

    @pytest.mark.parametrize(
        ("name", "objective"), [("SFT", "sft"), ("DPO", "preference")]
    )
    @pytest.mark.parametrize("engines", [2, "auto"])
    def test_dataset_algorithm_rejects_rollout_engines(
        self, name: str, objective: str, engines: int | str
    ) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": name}
        doc["environment"] = {
            "env_type": "dataset",
            "objective": objective,
            "dataset": "d",
        }
        doc["training"] = {
            **doc.get("training", {}),
            "rollout_engines_per_agent": engines,
        }

        with pytest.raises(
            ValidationError,
            match=f"rollout_engines_per_agent={engines} requires async rollout",
        ):
            TrainingManifest.model_validate(doc)


class TestGroupReplay:
    def test_async_grpo_accepts_group_replay_without_max_reward(self) -> None:
        validated = TrainingManifest.model_validate(
            group_replay_manifest(group_replay_max_age=4)
        )

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.group_replay_max_age == 4
        assert validated.to_payload()["replay_buffer"]["group_replay_max_age"] == 4

    def test_group_replay_is_on_by_default(self) -> None:
        spec = LLMRolloutBufferSpec()

        assert spec.group_replay_max_age == 20

    def test_negative_max_age_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="group_replay_max_age"):
            TrainingManifest.model_validate(
                group_replay_manifest(group_replay_max_age=-1)
            )

    def test_cispo_accepts_group_replay(self) -> None:
        doc = group_replay_manifest()
        doc["algorithm"]["name"] = "CISPO"

        replay_buffer = TrainingManifest.model_validate(doc).replay_buffer
        assert isinstance(replay_buffer, LLMRolloutBufferSpec)
        assert replay_buffer.group_replay_max_age == 10

    def test_zero_max_age_turns_group_replay_off(self) -> None:
        validated = TrainingManifest.model_validate(
            group_replay_manifest(group_replay_max_age=0)
        )

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.group_replay_max_age == 0

    @pytest.mark.parametrize(
        "algorithm",
        [
            {"name": "LLMPPO", "batch_size": 2},
            {"name": "GRPO", "batch_size": 2, "group_size": 1},
        ],
        ids=["llmppo", "grpo-group-size-1"],
    )
    def test_default_buffer_accepts_one_completion_groups(
        self, algorithm: dict
    ) -> None:
        # Group replay leaves a one-completion group alone, so the default applies.
        doc = group_replay_manifest()
        doc["algorithm"] = algorithm
        doc["replay_buffer"] = {"kind": "llm"}

        validated = TrainingManifest.model_validate(doc)

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.group_replay_max_age == 20

    def test_accepts_group_replay_without_rollout_mode(self) -> None:
        # The ray trainer drops training.rollout_mode before validating.
        doc = group_replay_manifest()
        del doc["training"]["rollout_mode"]
        del doc["training"]["rollout_engines_per_agent"]

        replay_buffer = TrainingManifest.model_validate(doc).replay_buffer

        assert isinstance(replay_buffer, LLMRolloutBufferSpec)
        assert replay_buffer.group_replay_max_age == 10


class TestDropUninformativeGroups:
    def test_tied_groups_are_dropped_by_default(self) -> None:
        spec = LLMRolloutBufferSpec()

        assert spec.drop_uninformative_groups is True

    def test_dropping_can_be_turned_off(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
                replay_buffer={"kind": "llm", "drop_uninformative_groups": False},
            )
        )

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.drop_uninformative_groups is False


def async_grpo(batch_size: int, **training: object) -> dict:
    """An async GRPO manifest with one rollout engine."""
    return manifest(
        GRPO,
        algorithm={"batch_size": batch_size},
        training={"rollout_mode": "async", "rollout_engines_per_agent": 1, **training},
        replay_buffer={"kind": "llm"},
    )


class TestAsyncRolloutDefaults:
    def test_rollout_batch_size_stays_unset_for_the_driver(self) -> None:
        validated = TrainingManifest.model_validate(
            async_grpo(64, training_gpus_per_agent=4)
        )

        assert validated.training.rollout_batch_size is None

    def test_payload_omits_split_and_cycle_the_author_left_unset(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"batch_size": 8},
            training={"rollout_mode": "async"},
            replay_buffer={"kind": "llm"},
        )

        training = TrainingManifest.model_validate(doc).to_payload()["training"]

        assert "training_gpus_per_agent" not in training
        assert "rollout_engines_per_agent" not in training
        assert "rollout_batch_size" not in training

    def test_payload_keeps_split_and_cycle_the_author_set(self) -> None:
        doc = async_grpo(
            8,
            training_gpus_per_agent=2,
            rollout_engines_per_agent="auto",
            rollout_batch_size=4,
        )

        training = TrainingManifest.model_validate(doc).to_payload()["training"]

        assert training["training_gpus_per_agent"] == 2
        assert training["rollout_engines_per_agent"] == "auto"
        assert training["rollout_batch_size"] == 4

    def test_explicit_rollout_batch_size_is_kept(self) -> None:
        validated = TrainingManifest.model_validate(
            async_grpo(64, training_gpus_per_agent=4, rollout_batch_size=16)
        )

        assert validated.training.rollout_batch_size == 16

    def test_version_lag_defaults_to_two(self) -> None:
        validated = TrainingManifest.model_validate(async_grpo(8))

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.max_rollout_version_lag == 2

    def test_zero_version_lag_is_accepted_as_strict_on_policy(self) -> None:
        doc = async_grpo(8)
        doc["replay_buffer"] = {"kind": "llm", "max_rollout_version_lag": 0}

        validated = TrainingManifest.model_validate(doc)

        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.max_rollout_version_lag == 0

    def test_null_version_lag_is_rejected(self) -> None:
        doc = async_grpo(8)
        doc["replay_buffer"] = {"kind": "llm", "max_rollout_version_lag": None}

        with pytest.raises(ValidationError, match="max_rollout_version_lag"):
            TrainingManifest.model_validate(doc)


class TestEpochDrivenTraining:
    """A run may state its schedule in epochs; the trainer converts to steps."""

    @pytest.mark.parametrize(
        ("name", "objective"),
        [("SFT", "sft"), ("DPO", "preference")],
    )
    def test_dataset_llm_rejects_num_epochs_and_max_steps(
        self, name: str, objective: str
    ) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": name}
        doc["environment"] = {
            "env_type": "dataset",
            "objective": objective,
            "dataset": "d",
        }
        doc["training"] = {"num_epochs": 1, "max_steps": 50}

        with pytest.raises(ValidationError, match="not used by LLM runs"):
            TrainingManifest.model_validate(doc)

    @pytest.mark.parametrize(
        ("name", "objective"),
        [("SFT", "sft"), ("DPO", "preference")],
    )
    def test_dataset_llm_rejects_num_epochs(self, name: str, objective: str) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": name}
        doc["environment"] = {
            "env_type": "dataset",
            "objective": objective,
            "dataset": "d",
        }
        doc["training"] = {"num_epochs": 1}

        with pytest.raises(ValidationError, match="not used by LLM runs"):
            TrainingManifest.model_validate(doc)

    @pytest.mark.parametrize(
        ("name", "objective"),
        [("SFT", "sft"), ("DPO", "preference")],
    )
    def test_dataset_llm_accepts_only_max_steps(
        self, name: str, objective: str
    ) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": name}
        doc["environment"] = {
            "env_type": "dataset",
            "objective": objective,
            "dataset": "d",
        }
        doc["training"] = {"max_steps": 50}

        validated = TrainingManifest.model_validate(doc)
        payload = validated.to_payload()

        assert validated.training.max_steps == 50
        assert validated.training.num_epochs is None
        assert payload["training"]["max_steps"] == 50
        TrainingManifest.model_validate(payload)

    def test_dataset_backed_grpo_without_steps_round_trips(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["training"] = {}

        payload = TrainingManifest.model_validate(doc).to_payload()

        assert "max_steps" not in payload["training"]
        TrainingManifest.model_validate(payload)

    def test_llm_rejects_evo_epochs(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["training"] = {"num_epochs": 2, "evo_epochs": 1}

        with pytest.raises(
            ValidationError,
            match=r"training\.num_epochs and training\.evo_epochs are not used",
        ):
            TrainingManifest.model_validate(doc)

    def test_grpo_rejects_num_epochs_and_max_steps(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"training\.num_epochs are not used by LLM runs; set "
            r"training\.max_steps and training\.evo_steps instead",
        ):
            TrainingManifest.model_validate(
                manifest(GRPO, training={"num_epochs": 1, "max_steps": 50})
            )

    def test_evo_epochs_alone_is_rejected(self) -> None:
        # The trainer only converts epochs for an epoch-driven loop, so the
        # interval would otherwise be silently dropped.
        with pytest.raises(ValidationError, match="evo_epochs needs"):
            TrainingSpec.model_validate({"evo_epochs": 1})

    def test_an_llm_run_with_evo_epochs_is_told_to_use_evo_steps(self) -> None:
        with pytest.raises(
            ValidationError,
            match=r"training\.evo_epochs are not used by LLM runs; set "
            r"training\.max_steps and training\.evo_steps instead",
        ):
            TrainingManifest.model_validate(manifest(GRPO, training={"evo_epochs": 1}))

    def test_a_step_driven_run_still_gets_its_evo_steps_default(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)
        assert validated.training.evo_steps is not None

    def test_zero_evo_steps_is_rejected(self) -> None:
        # The trainer's next-event maths is `steps % evo_steps`, so a zero
        # interval is a ZeroDivisionError.
        with pytest.raises(ValidationError, match="evo_steps"):
            TrainingManifest.model_validate(manifest(GRPO, training={"evo_steps": 0}))


class TestReplayBufferKind:
    """``kind`` is ``llm``, the async rollout deque."""

    def test_an_omitted_kind_is_the_rollout_deque(self) -> None:
        validated = TrainingManifest.model_validate(
            {**GRPO, "replay_buffer": {"max_size": 64}}
        )
        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert validated.replay_buffer.max_size == 64

    def test_llm_kind_selects_the_rollout_deque(self) -> None:
        validated = TrainingManifest.model_validate(
            {**GRPO, "replay_buffer": {"kind": "llm", "max_size": 64}}
        )
        assert isinstance(validated.replay_buffer, LLMRolloutBufferSpec)
        assert "kind" not in LLMRolloutBufferSpec.model_fields
        assert validated.replay_buffer.model_dump()["kind"] == "llm"

    def test_llm_rollout_buffer_spec_rejects_kind_as_a_constructor_arg(self) -> None:
        with pytest.raises(ValidationError):
            LLMRolloutBufferSpec(kind="llm")

    @pytest.mark.parametrize("kind", ["classic", "rl", "nope"])
    def test_any_other_kind_is_rejected(self, kind: str) -> None:
        with pytest.raises(ValidationError, match=r"replay_buffer\.kind must be 'llm'"):
            TrainingManifest.model_validate(
                {**GRPO, "replay_buffer": {"kind": kind, "max_size": 64}}
            )


class TestCli:
    def _run(self, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "agilerl.cli", "manifest", *args],
            capture_output=True,
            text=True,
            input=stdin,
            check=False,
        )

    def test_reads_a_manifest_from_stdin(self, tmp_path) -> None:
        document = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "max_context_length": 1024,
            },
        )

        result = self._run("validate", "-", "--json", stdin=json.dumps(document))

        assert result.returncode == 0
        assert json.loads(result.stdout)["ok"] is True

    def test_json_reports_an_invalid_manifest_without_failing(self) -> None:
        # The verdict is data. A non-zero exit has to mean the tool broke, or a
        # caller cannot tell a bad manifest from a broken validator.
        broken = manifest(GRPO, algorithm={"name": "GPRO"})
        result = self._run("validate", "-", "--json", stdin=json.dumps(broken))
        assert result.returncode == 0
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert report["errors"][0]["path"] == "algorithm"
        assert "not a registered algorithm" in report["errors"][0]["message"]

    def test_json_reports_a_non_mapping_algorithm_section(self) -> None:
        # A malformed section shape must surface as a verdict, never as an
        # escaped TypeError traceback with exit 1 and empty stdout.
        broken = {**copy.deepcopy(GRPO), "algorithm": "just-a-string"}
        result = self._run("validate", "-", "--json", stdin=json.dumps(broken))
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["ok"] is False
        assert any("must be a mapping" in e["message"] for e in report["errors"])
        assert "Traceback" not in result.stderr

    def test_json_reports_the_package_version(self) -> None:
        result = self._run("validate", "-", "--json", stdin=json.dumps(GRPO))
        assert json.loads(result.stdout)["manifest_version"] == __version__

    def test_an_unreadable_manifest_exits_differently(self) -> None:
        result = self._run("validate", "/does/not/exist.yaml", "--json")
        assert result.returncode == 2
        assert result.stdout == ""

    def test_error_paths_name_the_field_a_submitter_wrote(self) -> None:
        broken = manifest(GRPO, algorithm={"vllm_config": {"max_num_seqs": 0}})
        result = self._run("validate", "-", "--json", stdin=json.dumps(broken))
        paths = [e["path"] for e in json.loads(result.stdout)["errors"]]
        assert any(p.endswith("vllm_config.max_num_seqs") for p in paths), paths
        assert not any("function-after" in p for p in paths), paths


class TestSchema:
    def test_schema_states_the_version_that_produced_it(self) -> None:
        assert manifest_schema()["x-manifest-version"] == __version__

    def test_schema_is_valid_json_schema(self) -> None:
        Draft202012Validator.check_schema(manifest_schema())

    def test_an_alias_spelling_is_accepted_but_not_drawn_twice(self) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "LLMPPO"
        )
        assert variant["properties"]["lr"]["x-ui-alias-of"] == "lr_actor"
        assert "x-ui-alias-of" not in variant["properties"]["lr_actor"]

    def test_to_payload_omits_unset_llm_image_observation_keys(self) -> None:
        payload = TrainingManifest.model_validate(GRPO).to_payload()
        env = payload["environment"]
        assert "screenshot_field" not in env
        assert "goal_images_field" not in env

    def test_to_payload_keeps_set_llm_image_observation_keys(self) -> None:
        source = manifest(
            GRPO,
            environment={
                "env_type": "rollout",
                "env_url": "http://localhost",
                "max_turns": 5,
                "screenshot_field": "page",
                "goal_images_field": "task_images",
            },
        )
        env = source["environment"]
        env.pop("dataset", None)
        env.pop("reward_file_path", None)
        env.pop("prompt_template", None)

        payload = TrainingManifest.model_validate(source).to_payload()
        assert payload["environment"]["screenshot_field"] == "page"
        assert payload["environment"]["goal_images_field"] == "task_images"

    @pytest.mark.parametrize("name", ["AgentWorldModel", "VisualWebArena"])
    def test_name_only_built_in_submits_only_authored_environment_keys(
        self, name: str
    ) -> None:
        doc = built_in_run({"name": name, "env_type": "rollout"})

        payload = TrainingManifest.model_validate(doc).to_payload()

        assert payload["environment"] == {"env_type": "rollout", "name": name}
        assert not schema_errors(payload)

    def test_name_only_built_in_keeps_environment_fields_the_manifest_sets(
        self,
    ) -> None:
        authored = {
            "name": "VisualWebArena",
            "env_type": "rollout",
            "num_envs": 4,
            "max_turns": 5,
            "segment_prompt_tokens": 1000,
            "restart_keep_turns": 1,
            "action_error_field": "error",
            "repeat_system_prompt": True,
            "strict_chat_template_boundary": False,
        }
        doc = built_in_run(authored)

        payload = TrainingManifest.model_validate(doc).to_payload()

        assert payload["environment"] == authored

    def test_a_built_in_with_a_source_still_submits_environment_defaults(self) -> None:
        doc = built_in_run(
            {"name": "AgentWorldModel", "env_type": "rollout", "env_image": "awm:1"}
        )

        environment = TrainingManifest.model_validate(doc).to_payload()["environment"]

        assert environment["num_envs"] == 1
        assert environment["strict_chat_template_boundary"] is True
        assert environment["env_services"] == []
        assert environment["restart_keep_turns"] == 0
        assert environment["action_error_field"] == ""
        assert environment["repeat_system_prompt"] is False
        assert environment["env_image"] == "awm:1"

    def test_validating_a_manifest_instance_returns_it_unchanged(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(GRPO)

        # Act
        again = TrainingManifest.model_validate(manifest)

        # Assert
        assert again.to_payload() == manifest.to_payload()

    def test_max_model_len_publishes_no_default(self) -> None:
        # Arrange
        grpo = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )

        # Act
        field = grpo["properties"]["max_model_len"]

        # Assert
        assert "default" not in field

    def test_marked_fields_carry_their_visibility(self) -> None:
        # Arrange
        schema = manifest_schema()
        grpo = next(
            v
            for v in schema["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )
        defs = schema["$defs"]

        # Act
        tiers = {
            "batch_size": grpo["properties"]["batch_size"].get("x-arena-visibility"),
            "rollout_version_stamp": defs["TrainingSpec"]["properties"][
                "rollout_version_stamp"
            ].get("x-arena-visibility"),
            "env_port": defs["LLMEnvSpec"]["properties"]["env_port"].get(
                "x-arena-visibility"
            ),
            "env_vars": defs["LLMEnvSpec"]["properties"]["env_vars"].get(
                "x-arena-visibility"
            ),
        }

        # Assert
        assert tiers == {
            "batch_size": "auto",
            "rollout_version_stamp": "hidden",
            "env_port": "advanced",
            "env_vars": None,
        }

    def test_the_payload_we_submit_satisfies_the_schema_we_publish(self) -> None:
        # The schema is what the UI form validates against; a payload it rejects
        # means the form and the models disagree.
        payload = TrainingManifest.model_validate(GRPO).to_payload()
        assert not schema_errors(payload)

    def test_optional_scalars_are_one_input_not_a_type_choice(self) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )
        field = variant["properties"]["mini_batch_size"]
        assert "anyOf" not in field
        assert field["type"] == ["integer", "null"]
        assert field["minimum"] == 1

    def test_free_form_objects_ask_for_a_json_editor(self) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )
        assert variant["properties"]["vllm_engine_args"]["x-ui-widget"] == "json"

    @pytest.mark.parametrize("title", ["GRPO", "GSPO", "CISPO", "LLMPPO"])
    def test_trainer_offload_is_not_on_the_form(self, title: str) -> None:
        variant = next(
            v
            for v in manifest_schema()["properties"]["algorithm"]["oneOf"]
            if v["title"] == title
        )
        assert "offload_trainer_during_rollout" not in variant["properties"]

    def test_a_model_with_fields_is_never_a_json_blob(self) -> None:
        schema = manifest_schema()
        blobs: list[str] = []

        def check(node: object, path: str) -> None:
            if isinstance(node, list):
                for i, item in enumerate(node):
                    check(item, f"{path}[{i}]")
                return
            if not isinstance(node, dict):
                return
            if node.get("x-ui-widget") == "json" and node.get("properties"):
                blobs.append(path)
            for key, value in node.items():
                check(value, f"{path}.{key}")

        check(schema, "$")
        assert not blobs, blobs

    def test_aliased_spellings_are_declared(self) -> None:
        schema = manifest_schema()
        assert "metrics_interval" in schema["$defs"]["TrainingSpecLLM"]["properties"]
        assert "memory_size" in schema["$defs"]["LLMRolloutBufferSpec"]["properties"]

    def test_schema_offers_each_algorithm_once(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        titles = [v["title"] for v in variants]
        assert len(titles) == len(set(titles))
        accepted = {n for v in variants for n in v["properties"]["name"]["enum"]}
        assert accepted == set(MANIFEST_REGISTRY.names())

    def test_an_llm_payload_has_no_mutation_or_selection_section(self) -> None:
        # Arrange
        stored = manifest(
            GRPO,
            mutation={
                "name": "Mutations",
                "probabilities": {
                    "no_mut": 1.0,
                    "arch_mut": 0.0,
                    "new_layer": 0.0,
                    "params_mut": 0.0,
                    "act_mut": 0.0,
                    "rl_hp_mut": 0.0,
                },
                "rl_hp_selection": {},
            },
            tournament_selection={"name": "TournamentSelection"},
        )

        # Act
        payloads = [
            TrainingManifest.model_validate(source).to_payload()
            for source in (GRPO, stored)
        ]

        # Assert
        for payload in payloads:
            assert {"mutation", "tournament_selection"}.isdisjoint(payload)

    def test_an_llm_run_keeps_its_reference_adapter(self) -> None:
        # The algorithm classes default this on for every LLM paradigm but SFT.
        # to_payload() emits it either way, and the runtime only setdefaults, so
        # a wrong default here silently disables the reference adapter.
        for name in ("GRPO", "GSPO", "CISPO"):
            payload = TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"name": name})
            ).to_payload()
            assert payload["algorithm"]["use_separate_reference_adapter"] is True

    def test_sft_has_no_reference_adapter(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "SFT"}
        doc["environment"] = {"env_type": "dataset", "objective": "sft", "dataset": "d"}
        payload = TrainingManifest.model_validate(doc).to_payload()
        assert payload["algorithm"]["use_separate_reference_adapter"] is False

    def test_mutating_a_field_the_algorithm_lacks_is_rejected(self) -> None:
        doc = manifest(
            GRPO, mutation={"rl_hp_selection": {"theta": {"min": 0.1, "max": 1.0}}}
        )
        with pytest.raises(ValidationError, match="no such hyperparameter"):
            TrainingManifest.model_validate(doc)

    def test_a_near_miss_names_the_field_it_meant(self) -> None:
        doc = manifest(
            GRPO, mutation={"rl_hp_selection": {"betas": {"min": 0.1, "max": 1.0}}}
        )
        with pytest.raises(ValidationError, match="Did you mean 'beta'"):
            TrainingManifest.model_validate(doc)

    def test_mutating_a_non_numeric_field_is_rejected(self) -> None:
        # The field exists, so a name check alone would let this through and
        # then apply a grow factor to a flag.
        doc = manifest(
            GRPO,
            mutation={
                "rl_hp_selection": {
                    "use_separate_reference_adapter": {"min": 0, "max": 1}
                }
            },
        )
        with pytest.raises(ValidationError, match="not a number"):
            TrainingManifest.model_validate(doc)

    def test_mutating_a_list_or_scalar_union_is_rejected(self) -> None:
        doc = manifest(
            GRPO,
            mutation={"rl_hp_selection": {"clip_coef": {"min": 0.05, "max": 0.35}}},
        )
        assert TrainingManifest.model_validate(doc)

    def test_an_aliased_spelling_may_be_mutated(self) -> None:
        # Most manifests spell LLMPPO's lr_actor as lr; rejecting the alias
        # would fail documents that run today.
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "LLMPPO"}
        doc["mutation"] = {"rl_hp_selection": {"lr": {"min": 1e-7, "max": 1e-5}}}
        assert TrainingManifest.model_validate(doc)

    def test_schema_publishes_no_hpo_ranges(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        assert not [v["title"] for v in variants if "x-hpo-ranges" in v]

    def test_every_field_carries_help_text(self) -> None:
        # The schema is what renders the UI form, so an undescribed field ships
        # as a labelled box with no explanation.
        schema = manifest_schema()
        undescribed: list[str] = []

        def check(node: dict, path: str) -> None:
            for name, field in (node.get("properties") or {}).items():
                if "description" not in field and not field.get("deprecated"):
                    undescribed.append(f"{path}.{name}")

        check(schema, "")
        for variant in schema["properties"]["algorithm"]["oneOf"]:
            check(variant, f"algorithm[{variant['title']}]")
        for name, definition in schema["$defs"].items():
            check(definition, name)

        assert not undescribed

    def test_trainer_only_fields_are_deprecated_but_still_accepted(self) -> None:
        schema = manifest_schema()
        training = schema["$defs"]["TrainingSpecLLM"]["properties"]
        for name in ("mutation", "tournament_selection", "selection_strategy"):
            assert schema["properties"][name] == {"deprecated": True}
        hidden = {"pop_size", "population_size", "hpo", "experience_sharing"}
        hidden |= {"eval_steps", "num_epochs", "evo_epochs", "learning_delay"}
        for name in hidden:
            assert training[name] == {"deprecated": True}

        doc = manifest(
            GRPO,
            mutation={"probabilities": {"no_mut": 1.0}},
            tournament_selection={"tournament_size": 2},
            training={"pop_size": 1, "hpo": False, "experience_sharing": False},
        )
        validated = TrainingManifest.model_validate(doc)
        assert validated.to_payload()["training"]["pop_size"] == 1
        assert validated.selection_strategy.tournament_size == 2

    def test_schema_declares_the_manifest_sections(self) -> None:
        properties = manifest_schema()["properties"]
        assert {
            "algorithm",
            "environment",
            "training",
            "network",
            "replay_buffer",
        } <= set(properties)

    def _applied_training_defaults(self, instance: dict) -> dict[str, object]:
        applied: dict[str, object] = {}
        for cond in manifest_schema().get("allOf") or []:
            if_schema = cond.get("if")
            then = cond.get("then")
            if not isinstance(if_schema, dict) or not isinstance(then, dict):
                continue
            if not Draft202012Validator(if_schema).is_valid(instance):
                continue
            training_props = (
                then.get("properties", {}).get("training", {}).get("properties", {})
            )
            for name, field in training_props.items():
                if "default" in field:
                    applied[name] = field["default"]
        return applied

    def test_training_conditionals_hang_off_the_root_schema(self) -> None:
        schema = manifest_schema()
        training = schema["properties"]["training"]
        assert "allOf" not in training

        conditionals = [
            item
            for item in schema.get("allOf") or []
            if isinstance(item, dict) and "if" in item and "then" in item
        ]
        assert conditionals
        for cond in conditionals:
            if_clause = cond["if"]
            then_props = cond["then"]["properties"]
            if "training" in then_props:
                if "allOf" in if_clause:
                    assert any(
                        part.get("required") == ["algorithm"]
                        for part in if_clause["allOf"]
                    )
                else:
                    assert if_clause["required"] == ["algorithm"]
                    assert if_clause["properties"]["algorithm"]["required"] == ["name"]
            elif "environment" in then_props:
                if "allOf" in if_clause:
                    assert any(
                        part.get("required") == ["environment"]
                        for part in if_clause["allOf"]
                    )
                else:
                    assert if_clause["required"] == ["environment"]
            else:
                msg = f"unexpected conditional then keys: {sorted(then_props)}"
                raise AssertionError(msg)

    def test_training_defaults_follow_spec_types(self) -> None:
        grpo = self._applied_training_defaults({"algorithm": {"name": "GRPO"}})
        assert grpo["reporting_interval"] == 1
        assert grpo["max_steps"] == 200
        assert "experience_sharing" not in grpo

        sft = self._applied_training_defaults({"algorithm": {"name": "SFT"}})
        assert "num_epochs" not in sft
        assert sft["reporting_interval"] == 1
        assert sft["max_steps"] is None

    @pytest.mark.parametrize("name", ["GRPO", "LLMPPO", "LLMREINFORCE"])
    def test_rollout_llm_runs_default_the_async_version_stamp(self, name) -> None:
        applied = self._applied_training_defaults({"algorithm": {"name": name}})

        assert applied["rollout_version_stamp"] == "oldest_turn"

    @pytest.mark.parametrize("name", ["SFT", "DPO"])
    def test_dataset_llm_runs_leave_the_version_stamp_unset(self, name) -> None:
        applied = self._applied_training_defaults({"algorithm": {"name": name}})

        assert "rollout_version_stamp" not in applied

    def test_training_spec_model_defaults_match_schema(self) -> None:
        spec = TrainingSpec()
        assert spec.experience_sharing is None
        assert spec.evaluation_interval is None
        assert spec.hpo is False
        assert spec.checkpoint_export is None
        assert spec.rollout_version_stamp is None

    def test_llm_algorithms_get_evo_steps_twenty_from_schema(self) -> None:
        grpo = self._applied_training_defaults({"algorithm": {"name": "GRPO"}})
        assert grpo["evo_steps"] == 20
        assert grpo["reporting_interval"] == 1

    def test_dataset_backed_rollout_leaves_max_steps_unset(self) -> None:
        expected_if = dataset_backed_rollout_if()
        conditionals = manifest_schema().get("allOf") or []
        match = next(
            c
            for c in conditionals
            if c.get("if") == expected_if
            and c.get("then")
            == {
                "properties": {
                    "training": {
                        "properties": {
                            "max_steps": {"default": None},
                        }
                    }
                }
            }
        )
        assert match is not None

        dataset_rollout = {
            "algorithm": {"name": "GRPO"},
            "environment": {
                "env_type": "rollout",
                "dataset": "openai/gsm8k",
                "reward_file_path": "reward.py",
                "prompt_template": {"user_0": "{question}"},
            },
        }
        applied = self._applied_training_defaults(dataset_rollout)
        assert "num_epochs" not in applied
        assert applied["max_steps"] is None
        assert applied["evo_steps"] == 20

        for name in ("LLMPPO", "LLMREINFORCE"):
            applied = self._applied_training_defaults(
                {**dataset_rollout, "algorithm": {"name": name}}
            )
            assert "num_epochs" not in applied
            assert applied["max_steps"] is None

        generative_rollout = {
            "algorithm": {"name": "GRPO"},
            "environment": {
                "env_type": "rollout",
                "factory": "my_env:build",
                "entrypoint": "task",
            },
        }
        generative_defaults = self._applied_training_defaults(generative_rollout)
        assert "num_epochs" not in generative_defaults

    def test_rollout_llm_max_steps_conditional_covers_grpo_family(self) -> None:
        def rollout_llm(spec: type) -> bool:
            return issubclass(spec, RolloutLLMSpec)

        rollout_if = algorithm_name_if(registered_algorithm_names(rollout_llm))
        conditionals = manifest_schema().get("allOf") or []
        max_steps_conds = [
            c
            for c in conditionals
            if c.get("then") == training_then({"max_steps": {"default": 200}})
            and c.get("if") == rollout_if
        ]
        assert len(max_steps_conds) == 1
        for name in ("GRPO", "GSPO", "CISPO", "LLMPPO", "LLMREINFORCE"):
            applied = self._applied_training_defaults({"algorithm": {"name": name}})
            assert applied["max_steps"] == 200

    def test_dataset_llm_max_steps_conditional_is_null(self) -> None:
        def dataset_llm(spec: type) -> bool:
            return spec.env_type == LLMEnvType.DATASET

        dataset_if = algorithm_name_if(registered_algorithm_names(dataset_llm))
        conditionals = manifest_schema().get("allOf") or []
        null_max_steps = [
            c
            for c in conditionals
            if c.get("if") == dataset_if
            and c.get("then") == training_then({"max_steps": {"default": None}})
        ]
        assert len(null_max_steps) == 1
        for name in ("DPO", "SFT"):
            applied = self._applied_training_defaults({"algorithm": {"name": name}})
            assert applied["max_steps"] is None

    def test_llm_evaluation_interval_schema_default(self) -> None:
        llm = self._applied_training_defaults({"algorithm": {"name": "GRPO"}})
        assert llm["evaluation_interval"] == 10

    def test_grpo_group_size_schema_default(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        grpo = next(v for v in variants if v["title"] == "GRPO")
        assert grpo["properties"]["group_size"]["default"] == 8

    def _applied_environment_defaults(self, instance: dict) -> dict[str, object]:
        applied: dict[str, object] = {}
        for cond in manifest_schema().get("allOf") or []:
            if_schema = cond.get("if")
            then = cond.get("then")
            if not isinstance(if_schema, dict) or not isinstance(then, dict):
                continue
            if not Draft202012Validator(if_schema).is_valid(instance):
                continue
            env_props = (
                then.get("properties", {}).get("environment", {}).get("properties", {})
            )
            for name, field in env_props.items():
                if "default" in field:
                    applied[name] = field["default"]
        return applied

    def _llm_rollout_branch_properties(self) -> dict:
        schema = manifest_schema()
        llm = schema["$defs"]["LLMEnvSpec"]
        for branch in llm.get("anyOf", llm.get("oneOf", [llm])):
            props = branch.get("properties", {})
            env_type = props.get("env_type", {})
            if (
                env_type.get("const") == "rollout"
                or env_type.get("default") == "rollout"
            ):
                return props
        return llm.get("properties", {})

    def test_llm_env_schema_conditionals_publish_dataset_defaults(self) -> None:
        conditionals = manifest_schema().get("allOf") or []
        split_cond = next(
            c
            for c in conditionals
            if c.get("if") == environment_dataset_identity_if()
            and c.get("then")
            == environment_then({"train_test_split": {"default": 0.9}})
        )
        assert split_cond is not None

        response_column_cond = next(
            c
            for c in conditionals
            if c.get("if") == environment_dataset_type_if()
            and c.get("then")
            == environment_then({"response_column": {"default": "response"}})
        )
        assert response_column_cond is not None

        rollout_cond = next(
            c
            for c in conditionals
            if c.get("if") == environment_rollout_type_if()
            and c.get("then")
            == environment_then(
                {
                    "num_envs": {"default": 1},
                    "strict_chat_template_boundary": {"default": True},
                }
            )
        )
        assert rollout_cond is not None

        gem_rollout = {
            "environment": {
                "env_type": "rollout",
                "entrypoint": "game:Env",
            }
        }
        assert self._applied_environment_defaults(gem_rollout) == {
            "num_envs": 1,
            "strict_chat_template_boundary": True,
        }

        dataset_rollout = {
            "environment": {
                "env_type": "rollout",
                "dataset": "rows",
                "prompt_template": {"user_0": "{question}"},
            }
        }
        applied = self._applied_environment_defaults(dataset_rollout)
        assert applied["train_test_split"] == 0.9
        assert applied["strict_chat_template_boundary"] is True
        assert "response_column" not in applied
        assert "rubric_name" not in applied
        assert applied["num_envs"] == 1

        dataset_sft = {
            "environment": {
                "env_type": "dataset",
                "objective": "sft",
                "dataset": "rows",
            }
        }
        applied_sft = self._applied_environment_defaults(dataset_sft)
        assert applied_sft["train_test_split"] == 0.9
        assert applied_sft["response_column"] == "response"
        assert "strict_chat_template_boundary" not in applied_sft
        assert "rubric_name" not in applied_sft
        assert "num_envs" not in applied_sft

    def test_gem_shaped_llm_rollout_schema_lacks_dataset_field_defaults(self) -> None:
        props = self._llm_rollout_branch_properties()
        for name in (
            "train_test_split",
            "response_column",
            "rubric_name",
            "strict_chat_template_boundary",
            "action_field",
            "screenshot_field",
            "goal_images_field",
        ):
            field = props.get(name, {})
            assert "default" not in field, name

    def test_training_section_refs_the_llm_training_def(self) -> None:
        schema = manifest_schema()
        training = schema["properties"]["training"]
        assert "anyOf" not in training
        assert training["$ref"] == "#/$defs/TrainingSpecLLM"

        training_spec = schema["$defs"]["TrainingSpec"]["properties"]
        assert "training_gpus_per_agent" not in training_spec

    def test_llm_training_gpus_per_agent_schema_default(self) -> None:
        schema = manifest_schema()
        llm_training = schema["$defs"]["TrainingSpecLLM"]["properties"]
        assert llm_training["training_gpus_per_agent"]["default"] == 1

        for name in ("GRPO", "SFT", "DPO", "CISPO"):
            applied = self._applied_training_defaults({"algorithm": {"name": name}})
            assert "training_gpus_per_agent" not in applied

    def test_served_lora_config_inlines_rank_alpha_dropout(self) -> None:
        schema = manifest_schema()
        network_lora = schema["$defs"]["FinetuningNetworkSpec"]["properties"][
            "lora_config"
        ]
        assert "$ref" not in network_lora
        properties = network_lora["properties"]
        assert properties["lora_r"]["default"] == 1
        assert properties["lora_r"]["title"] == "Rank"
        assert properties["lora_alpha"]["default"] == 32
        assert properties["lora_dropout"]["default"] == 0.0

        grpo = next(
            v
            for v in schema["properties"]["algorithm"]["oneOf"]
            if v["title"] == "GRPO"
        )
        algo_lora = grpo["properties"]["lora_config"]
        assert algo_lora["properties"]["lora_r"]["default"] == 1
        assert "lora_alpha" in algo_lora["properties"]
        assert "lora_dropout" in algo_lora["properties"]

    def test_inlines_lora_config_when_field_is_a_bare_ref(self) -> None:
        schema = {
            "properties": {
                "lora_config": {
                    "$ref": LORA_CONFIG_REF,
                    "description": "LoRA",
                }
            }
        }
        lora_def = {
            "properties": {"lora_r": {"type": "integer"}},
            "additionalProperties": False,
            "required": ["lora_r"],
        }

        _inline_lora_config_inplace(schema, lora_def)

        field = schema["properties"]["lora_config"]
        assert field["properties"]["lora_r"] == {"type": "integer"}
        assert field["required"] == ["lora_r"]
        assert field["additionalProperties"] is False
        assert field["description"] == "LoRA"
        assert "$ref" not in field

    def test_leaves_non_ref_lora_config_unchanged(self) -> None:
        original = {"type": "object", "properties": {"lora_r": {"type": "integer"}}}
        schema = {"properties": {"lora_config": original}}

        _inline_lora_config_inplace(schema, {"properties": {}})

        assert schema["properties"]["lora_config"] is original

    def test_unset_llm_training_gpus_per_agent_is_omitted_from_the_payload(
        self,
    ) -> None:
        payload = TrainingManifest.model_validate(GRPO).to_payload()
        assert "training_gpus_per_agent" not in payload["training"]

    def test_explicit_llm_training_gpus_per_agent_is_kept_in_the_payload(self) -> None:
        payload = TrainingManifest.model_validate(
            manifest(GRPO, training={"training_gpus_per_agent": 2})
        ).to_payload()
        assert payload["training"]["training_gpus_per_agent"] == 2

    def test_sft_algorithm_schema_omits_beta_dpo_keeps_it(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        sft = next(v for v in variants if v["title"] == "SFT")
        dpo = next(v for v in variants if v["title"] == "DPO")
        assert "beta" not in sft.get("properties", {})
        assert dpo["properties"]["beta"]["default"] == 0.1

    def test_dataset_llm_algorithm_schema_omits_answer_continuation(self) -> None:
        variants = manifest_schema()["properties"]["algorithm"]["oneOf"]
        sft = next(v for v in variants if v["title"] == "SFT")
        dpo = next(v for v in variants if v["title"] == "DPO")
        for variant in (sft, dpo):
            assert "answer_continuation" not in variant.get("properties", {})

        grpo = next(v for v in variants if v["title"] == "GRPO")
        assert grpo["properties"]["answer_continuation"]["default"] is False
        assert "answer_pattern" not in grpo.get("properties", {})
        assert "constrain_answer_pattern" in grpo["properties"]

    def test_dataset_schema_drops_answer_continuation_from_required(self) -> None:
        schema = {
            "properties": {"answer_continuation": {}, "beta": {}, "name": {}},
            "required": ["answer_continuation", "name"],
        }

        strip_non_form_algorithm_fields(schema, SFTSpec)

        assert "answer_continuation" not in schema["properties"]
        assert "beta" not in schema["properties"]
        assert schema["required"] == ["name"]


class TestPublishedSchemaAcceptsManifests:
    @pytest.mark.parametrize(
        "name", ["GRPO", "CISPO", "GSPO", "LLMPPO", "LLMREINFORCE"]
    )
    def test_default_manifest_and_its_payload_validate(self, name: str) -> None:
        # Arrange
        record = default_verdict(
            {
                "algorithm": {"name": name},
                "environment": copy.deepcopy(GRPO["environment"]),
                "network": {"pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B"},
            },
            user_view=True,
        )
        template = record["payload"]

        # Act
        payload = TrainingManifest.model_validate(template).to_payload()

        # Assert
        assert record["ok"] is True
        assert schema_errors(template) == []
        assert schema_errors(payload) == []

    def test_llm_payload_leaves_out_hidden_training_defaults(self) -> None:
        payload = TrainingManifest.model_validate(GRPO).to_payload()

        assert "pop_size" not in payload["training"]
        assert "hpo" not in payload["training"]

    def test_llm_payload_keeps_a_hidden_training_key_the_manifest_sets(self) -> None:
        doc = manifest(GRPO, training={"pop_size": 1})

        payload = TrainingManifest.model_validate(doc).to_payload()

        assert payload["training"]["pop_size"] == 1
        assert schema_errors(payload) == []

    def test_1x_manifest_keys_validate_as_deprecated(self) -> None:
        # Arrange
        doc = TrainingManifest.model_validate(GRPO).to_payload()
        doc["training"].update(
            {"pop_size": 1, "hpo": False, "evo_epochs": 1, "learning_delay": 0}
        )
        doc["mutation"] = {"probabilities": {"no_mut": 1.0}}
        doc["tournament_selection"] = {"tournament_size": 2, "elitism": True}

        # Act
        errors = schema_errors(doc)

        # Assert
        assert errors == []

    def test_a_misspelt_key_still_fails(self) -> None:
        doc = TrainingManifest.model_validate(GRPO).to_payload()
        doc["training"]["pop_sise"] = 1

        errors = schema_errors(doc)

        assert errors
        assert all("'pop_sise' was unexpected" in error for error in errors)


class TestStripNonFormAlgorithmFields:
    def test_removes_answer_continuation_from_required_on_dataset_specs(self) -> None:
        schema = {
            "properties": {"answer_continuation": {"type": "boolean"}},
            "required": ["name", "answer_continuation"],
        }

        strip_non_form_algorithm_fields(schema, DPOSpec)

        assert "answer_continuation" not in schema["properties"]
        assert "answer_continuation" not in schema["required"]

    def test_removes_answer_pattern_alias_from_the_form(self) -> None:
        schema = {
            "properties": {
                "constrain_answer_pattern": {"type": "string"},
                "answer_pattern": {"x-ui-alias-of": "constrain_answer_pattern"},
            },
            "required": ["name", "answer_pattern"],
        }

        strip_non_form_algorithm_fields(schema, GRPOSpec)

        assert "answer_pattern" not in schema["properties"]
        assert "answer_pattern" not in schema["required"]
        assert "constrain_answer_pattern" in schema["properties"]


class TestRolloutSamplingFields:
    def test_grpo_defaults_match_the_algorithm_ctor(self) -> None:
        assert GRPOSpec().group_size == 8
        spec = GRPOSpec.model_construct(group_size=4)
        assert spec.top_p == 1.0
        assert spec.top_k == 50
        assert spec.min_p == 0.0
        assert spec.repetition_penalty == 1.0
        assert spec.loss_type == "grpo"

    def test_llm_ppo_and_reinforce_default_top_p_to_one(self) -> None:
        from agilerl.models.algorithms import LLMPPOSpec, LLMREINFORCESpec

        assert LLMPPOSpec.model_construct().top_p == 1.0
        assert LLMREINFORCESpec.model_construct().top_p == 1.0

    def test_gspo_and_cispo_pin_loss_type(self) -> None:
        from agilerl.models.algorithms import CISPOSpec, GSPOSpec

        assert GSPOSpec.model_construct(group_size=4).loss_type == "gspo"
        assert CISPOSpec.model_construct(group_size=4).loss_type == "cispo"
        with pytest.raises(ValidationError):
            GSPOSpec(group_size=4, loss_type="grpo")


class TestExpertLoraReferenceAdapter:
    def test_expert_lora_keeps_the_reference_adapter(self) -> None:
        moe = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "lora_config": {},
            },
        )

        validated = TrainingManifest.model_validate(moe)

        assert validated.algorithm.lora_config.target_parameters
        assert validated.algorithm.use_separate_reference_adapter is True

    def test_explicit_reference_adapter_with_experts_validates(self) -> None:
        moe = manifest(
            GRPO,
            algorithm={"use_separate_reference_adapter": True},
            network={
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "lora_config": {},
            },
        )

        validated = TrainingManifest.model_validate(moe)

        assert validated.algorithm.use_separate_reference_adapter is True

    def test_no_experts_keeps_the_reference_adapter(self) -> None:
        moe = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "lora_config": {"target_parameters": []},
            },
        )

        validated = TrainingManifest.model_validate(moe)

        assert validated.algorithm.use_separate_reference_adapter is True


class TestLoraConfigInBothSections:
    def test_same_moe_lora_in_both_sections_validates(self) -> None:
        # Arrange
        doc = manifest(
            GRPO,
            algorithm={"lora_config": {"lora_r": 8}},
            network={
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "lora_config": {"lora_r": 8},
            },
        )

        # Act
        validated = TrainingManifest.model_validate(doc)

        # Assert
        expected = sorted(SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].parameters)
        assert validated.algorithm.lora_config.target_parameters == expected

    def test_different_moe_targets_in_the_two_sections_are_rejected(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"lora_config": {"lora_r": 8}},
            network={
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "lora_config": {"lora_r": 8, "target_parameters": []},
            },
        )

        with pytest.raises(
            ValidationError, match=r"algorithm\.lora_config=.* disagrees"
        ):
            TrainingManifest.model_validate(doc)


class TestAlgorithmOnlyLoraCatalog:
    @pytest.mark.parametrize(
        ("hub_id", "lora", "match"),
        [
            (
                "Qwen/Qwen2.5-0.5B-Instruct",
                {"lora_r": 256},
                r"LoRA rank 256 .*Qwen2.5-0.5B-Instruct \(128\)",
            ),
            (
                "openai/gpt-oss-20b",
                {"target_parameters": ["not.a.path"]},
                r"Unknown LoRA target_parameters \['not.a.path'\]",
            ),
        ],
    )
    def test_rejects_a_block_the_network_catalog_would_reject(
        self, hub_id: str, lora: dict[str, object], match: str
    ) -> None:
        doc = manifest(
            GRPO,
            algorithm={"lora_config": lora},
            network={"pretrained_model_name_or_path": hub_id},
        )

        with pytest.raises(ValidationError, match=match):
            TrainingManifest.model_validate(doc)


class TestToPayloadModelFields:
    def test_model_fields_are_only_in_the_network_section(self) -> None:
        # Arrange
        doc = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "max_context_length": 2048,
                "lora_config": {"lora_r": 8},
            },
        )

        # Act
        payload = TrainingManifest.model_validate(doc).to_payload()

        # Assert
        for field in ("pretrained_model_name_or_path", "max_model_len", "lora_config"):
            assert field not in payload["algorithm"]
        assert payload["network"]["pretrained_model_name_or_path"] == (
            "Qwen/Qwen3.5-0.8B"
        )
        assert payload["network"]["max_context_length"] == 2048
        assert payload["network"]["lora_config"]["lora_r"] == 8

    def test_algorithm_only_lora_moves_to_the_network(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"lora_config": {"lora_r": 8}},
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "max_context_length": 2048,
            },
        )

        payload = TrainingManifest.model_validate(doc).to_payload()

        assert "lora_config" not in payload["algorithm"]
        assert payload["network"]["lora_config"]["lora_r"] == 8

    @pytest.mark.parametrize(
        ("algorithm", "network"),
        [
            ({}, {}),
            ({"lora_config": None}, {}),
            ({}, {"lora_config": None}),
            ({"lora_config": None}, {"lora_config": None}),
        ],
        ids=["absent", "algorithm-null", "network-null", "both-null"],
    )
    def test_an_unset_or_null_lora_trains_the_catalog_default(
        self, algorithm: dict[str, object], network: dict[str, object]
    ) -> None:
        # Arrange
        hub_id = "openai/gpt-oss-20b"
        doc = manifest(
            GRPO,
            algorithm=algorithm,
            network={
                "pretrained_model_name_or_path": hub_id,
                "max_context_length": 2048,
                **network,
            },
        )

        # Act
        validated = TrainingManifest.model_validate(doc)
        payload = validated.to_payload()

        # Assert
        expected = sorted(SUPPORTED_MODEL_INFO[hub_id].parameters)
        assert validated.algorithm.lora_config.lora_r == 16
        assert validated.algorithm.lora_config.target_parameters == expected
        assert payload["network"]["lora_config"]["target_parameters"] == expected
        assert "lora_config" not in payload["algorithm"]

    def test_a_network_lora_wins_over_an_algorithm_null(self) -> None:
        doc = manifest(
            GRPO,
            algorithm={"lora_config": None},
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "max_context_length": 2048,
                "lora_config": {"lora_r": 8},
            },
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.lora_config.lora_r == 8
        assert validated.to_payload()["network"]["lora_config"]["lora_r"] == 8

    def test_payload_revalidates_to_the_same_algorithm(self) -> None:
        # Arrange
        doc = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "openai/gpt-oss-20b",
                "max_context_length": 2048,
                "lora_config": {"lora_r": 8},
            },
        )
        validated = TrainingManifest.model_validate(doc)

        # Act
        revalidated = TrainingManifest.model_validate(validated.to_payload())

        # Assert
        assert revalidated.algorithm.lora_config == validated.algorithm.lora_config
        assert revalidated.algorithm.max_model_len == 2048
        assert revalidated.algorithm.pretrained_model_name_or_path == (
            "openai/gpt-oss-20b"
        )


STORE_WEIGHTS = "s3://arena-tenant/models/qwen3.5-0.8b/"


class TestWeightsPath:
    def test_network_weights_path_lifts_and_dumps_on_the_network(self) -> None:
        # Arrange
        doc = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "weights_path": STORE_WEIGHTS,
            },
        )

        # Act
        validated = TrainingManifest.model_validate(doc)
        payload = validated.to_payload()

        # Assert
        assert validated.algorithm.pretrained_model_name_or_path == (
            "Qwen/Qwen3.5-0.8B"
        )
        assert validated.algorithm.weights_path == STORE_WEIGHTS
        assert validated.algorithm.load_path == STORE_WEIGHTS
        assert payload["network"]["weights_path"] == STORE_WEIGHTS
        assert "weights_path" not in payload["algorithm"]

    def test_algorithm_only_weights_path_moves_to_the_network(self) -> None:
        # Arrange
        doc = manifest(GRPO, algorithm={"weights_path": STORE_WEIGHTS})

        # Act
        payload = TrainingManifest.model_validate(doc).to_payload()

        # Assert
        assert payload["network"]["weights_path"] == STORE_WEIGHTS
        assert "weights_path" not in payload["algorithm"]

    def test_without_weights_path_the_model_id_loads(self) -> None:
        # Act
        validated = TrainingManifest.model_validate(GRPO)

        # Assert
        assert validated.algorithm.weights_path is None
        assert validated.algorithm.load_path == "Qwen/Qwen2.5-0.5B-Instruct"
        assert "weights_path" not in validated.to_payload()["network"]

    def test_disagreeing_weights_paths_raise(self) -> None:
        # Arrange
        doc = manifest(
            GRPO,
            algorithm={"weights_path": "s3://arena-tenant/models/other/"},
            network={"weights_path": STORE_WEIGHTS},
        )

        # Act / Assert
        with pytest.raises(ValidationError, match=r"algorithm\.weights_path="):
            TrainingManifest.model_validate(doc)


class TestTrainingSpecWeightSyncBackend:
    def test_the_removed_key_is_rejected(self) -> None:
        broken = manifest(GRPO, training={"weight_sync_backend": "nccl"})

        with pytest.raises(ValidationError, match="weight_sync_backend"):
            TrainingManifest.model_validate(broken)


def thinking_budget_manifest(model: str, **algorithm: object) -> dict:
    """GRPO manifest for *model* with a 64-token thinking budget."""
    return manifest(
        GRPO,
        algorithm={"max_output_tokens": 128, "thinking_token_budget": 64, **algorithm},
        network={"pretrained_model_name_or_path": model},
    )


class TestThinkingBudgetParser:
    @pytest.mark.parametrize(
        "model",
        [
            "google/gemma-4-E4B-it",
            "Qwen/Qwen3-1.7B",
            "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        ],
    )
    def test_accepts_a_budget_for_a_model_with_a_reasoning_parser(
        self, model: str
    ) -> None:
        validated = TrainingManifest.model_validate(thinking_budget_manifest(model))

        assert validated.algorithm.thinking_token_budget == 64

    def test_rejects_a_budget_for_a_model_without_a_reasoning_parser(self) -> None:
        model = "Qwen/Qwen2.5-0.5B-Instruct"

        with pytest.raises(
            ValidationError,
            match=(
                r"thinking_token_budget=64 needs a vLLM reasoning parser with "
                r"reasoning start and end tokens, and "
                rf"{re.escape(model)} has none\. Unset thinking_token_budget, or "
                r"set algorithm\.vllm_engine_args\.reasoning_parser to one that "
                r"has them: deepseek_r1, deepseek_v4, gemma4, glm47, mimo, "
                r"nemotron_v3, qwen3\."
            ),
        ):
            TrainingManifest.model_validate(thinking_budget_manifest(model))

    def test_rejects_a_budget_for_gpt_oss_whose_parser_cannot_enforce_it(
        self,
    ) -> None:
        with pytest.raises(
            ValidationError, match=r"and reasoning parser openai_gptoss has none\."
        ):
            TrainingManifest.model_validate(
                thinking_budget_manifest("openai/gpt-oss-20b")
            )

    def test_rejects_a_manifest_reasoning_parser_without_reasoning_tokens(
        self,
    ) -> None:
        doc = thinking_budget_manifest(
            "openai/gpt-oss-20b", vllm_engine_args={"reasoning_parser": "openai_gptoss"}
        )

        with pytest.raises(
            ValidationError, match=r"and reasoning parser openai_gptoss has none\."
        ):
            TrainingManifest.model_validate(doc)

    def test_accepts_a_budget_with_a_manifest_reasoning_parser(self) -> None:
        doc = thinking_budget_manifest(
            "Qwen/Qwen2.5-0.5B-Instruct",
            vllm_engine_args={"reasoning_parser": "deepseek_r1"},
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.thinking_token_budget == 64

    def test_leaves_an_unsupported_model_to_the_runtime(self) -> None:
        validated = TrainingManifest.model_validate(
            thinking_budget_manifest("org/untracked")
        )

        assert validated.algorithm.thinking_token_budget == 64

    def test_accepts_a_model_without_a_parser_when_no_budget_is_set(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)

        assert validated.algorithm.thinking_token_budget is None


def parallel_manifest(model: str, tp: int, ep: int) -> dict:
    """GRPO manifest for *model* with the given tensor and expert parallel sizes."""
    return manifest(
        GRPO,
        algorithm={"fsdp": {"tp": tp, "ep": ep}},
        network={"pretrained_model_name_or_path": model},
    )


class TestExpertParallelCoversTensorParallel:
    @pytest.mark.parametrize(("tp", "ep"), [(2, 2), (2, 4), (1, 2)])
    def test_accepts_ep_at_or_above_tp_for_a_moe_model(self, tp: int, ep: int) -> None:
        validated = TrainingManifest.model_validate(
            parallel_manifest("openai/gpt-oss-20b", tp, ep)
        )

        assert (validated.algorithm.fsdp.tp, validated.algorithm.fsdp.ep) == (tp, ep)

    def test_accepts_tp_without_ep_for_a_dense_model(self) -> None:
        validated = TrainingManifest.model_validate(
            parallel_manifest("Qwen/Qwen3-4B", 2, 1)
        )

        assert validated.algorithm.fsdp.tp == 2

    def test_rejects_ep_below_tp_for_a_moe_model(self) -> None:
        with pytest.raises(
            ValidationError,
            match=(
                r"algorithm\.fsdp\.ep=1 is below tp=2: openai/gpt-oss-20b is a "
                r"mixture-of-experts model, and tensor parallel does not split "
                r"routed experts\. Set ep >= tp\."
            ),
        ):
            TrainingManifest.model_validate(
                parallel_manifest("openai/gpt-oss-20b", 2, 1)
            )

    def test_leaves_an_unsupported_model_to_the_trainer(self) -> None:
        validated = TrainingManifest.model_validate(
            parallel_manifest("org/untracked", 2, 1)
        )

        assert validated.algorithm.fsdp.tp == 2


class TestLoraDropoutResolution:
    @pytest.mark.parametrize(
        "spec_cls",
        [GRPOSpec, GSPOSpec, CISPOSpec, LLMPPOSpec, LLMREINFORCESpec, SFTSpec, DPOSpec],
    )
    def test_llm_algorithms_default_to_no_dropout(self, spec_cls) -> None:
        spec = spec_cls(lora_config={"lora_r": 8})

        assert spec.lora_config.lora_dropout == 0.0

    @pytest.mark.parametrize("spec_cls", [GRPOSpec, SFTSpec, DPOSpec])
    def test_explicit_dropout_is_kept(self, spec_cls) -> None:
        spec = spec_cls(lora_config=LoraConfigDict(lora_dropout=0.1))

        assert spec.lora_config.lora_dropout == 0.1

    def test_network_lora_lifted_onto_a_rollout_algorithm_has_no_dropout(self) -> None:
        # Arrange
        doc = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
                "lora_config": {"lora_r": 8},
            },
        )

        # Act
        validated = TrainingManifest.model_validate(doc)

        # Assert
        assert validated.algorithm.lora_config.lora_dropout == 0.0
        assert validated.to_payload()["network"]["lora_config"]["lora_dropout"] == 0.0

    def test_network_lora_lifted_onto_sft_has_no_dropout(self) -> None:
        doc = copy.deepcopy(GRPO)
        doc["algorithm"] = {"name": "SFT"}
        doc["environment"] = {"env_type": "dataset", "objective": "sft", "dataset": "d"}
        doc["network"]["lora_config"] = {"lora_r": 8}

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.lora_config.lora_dropout == 0.0

    def test_explicit_network_dropout_on_a_rollout_algorithm_is_kept(self) -> None:
        doc = manifest(
            GRPO,
            network={
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
                "lora_config": {"lora_dropout": 0.1},
            },
        )

        validated = TrainingManifest.model_validate(doc)

        assert validated.algorithm.lora_config.lora_dropout == 0.1


def _algorithm_variant(title: str) -> dict:
    return next(
        variant
        for variant in manifest_schema()["properties"]["algorithm"]["oneOf"]
        if variant["title"] == title
    )


def _schema_types(prop: dict) -> set[str]:
    raw = prop.get("type")
    found = set(raw if isinstance(raw, list) else [raw] if raw else [])
    for branch in prop.get("anyOf", []):
        branch_type = branch.get("type")
        if isinstance(branch_type, str):
            found.add(branch_type)
    return found


def _schema_keys(prop: dict) -> set[str]:
    keys = set(prop)
    for branch in prop.get("anyOf", []):
        keys.update(branch)
    return keys


# Lengths and counts scale with the job. Return support and the seed have no
# fixed range. The trainer settings listed here have no ceiling the code
# enforces, so a static maximum would reject a saved run.
NO_STATIC_MAXIMUM = frozenset(
    {
        "adv_clip_range",
        "adv_filter_eps",
        "batch_size",
        "critic_warmup_steps",
        "group_size",
        "kl_clamp",
        "max_grad_norm",
        "max_model_len",
        "max_output_tokens",
        "max_row_tokens",
        "micro_batch_size_per_gpu",
        "micro_batch_tokens_per_gpu",
        "min_output_tokens",
        "mini_batch_size",
        "nll_alpha",
        "reference_update_interval",
        "repetition_penalty",
        "routed_expert_chunk_mib",
        "seed",
        "thinking_token_budget",
        "top_k",
        "vllm_importance_sampling_cap",
        "vllm_max_logprob_gap",
    }
)
NO_STATIC_MINIMUM = frozenset({"seed"})


class TestAlgorithmHpSchemaBounds:
    @pytest.mark.parametrize(
        ("title", "field", "bounds"),
        [
            (
                "GRPO",
                "temperature",
                {"exclusiveMinimum": 0.0, "maximum": 2.0, "default": 0.9},
            ),
            (
                "GSPO",
                "temperature",
                {"exclusiveMinimum": 0.0, "maximum": 2.0, "default": 0.9},
            ),
            (
                "CISPO",
                "temperature",
                {"exclusiveMinimum": 0.0, "maximum": 2.0, "default": 0.9},
            ),
            (
                "LLMPPO",
                "temperature",
                {"exclusiveMinimum": 0.0, "maximum": 2.0, "default": 1.0},
            ),
            (
                "LLMREINFORCE",
                "temperature",
                {"exclusiveMinimum": 0.0, "maximum": 2.0, "default": 1.0},
            ),
            (
                "GRPO",
                "top_p",
                {"exclusiveMinimum": 0.0, "maximum": 1.0, "default": 1.0},
            ),
            ("GRPO", "top_k", {"minimum": 0, "default": 50}),
            ("GRPO", "min_p", {"minimum": 0.0, "maximum": 1.0, "default": 0.0}),
            ("GRPO", "group_size", {"minimum": 1, "default": 8}),
            (
                "GRPO",
                "kl_clamp",
                {"exclusiveMinimum": 0.0, "default": 10.0},
            ),
            (
                "LLMPPO",
                "kl_clamp",
                {"exclusiveMinimum": 0.0, "default": 10.0},
            ),
            ("LLMPPO", "critic_warmup_steps", {"minimum": 0, "default": 0}),
            (
                "LLMREINFORCE",
                "top_p",
                {"exclusiveMinimum": 0.0, "maximum": 1.0, "default": 1.0},
            ),
            (
                "LLMPPO",
                "top_p",
                {"exclusiveMinimum": 0.0, "maximum": 1.0, "default": 1.0},
            ),
            ("GRPO", "update_epochs", {"minimum": 1, "maximum": 10, "default": 1}),
            (
                "GRPO",
                "repetition_penalty",
                {"exclusiveMinimum": 0.0, "default": 1.0},
            ),
            ("DPO", "nll_alpha", {"minimum": 0.0, "default": 1.0}),
            ("LLMREINFORCE", "top_k", {"minimum": 0, "default": 50}),
            ("DPO", "reference_update_interval", {"minimum": 1, "default": None}),
        ],
    )
    def test_served_schema_states_the_bound(
        self, title: str, field: str, bounds: dict
    ) -> None:
        prop = _algorithm_variant(title)["properties"][field]
        keys = _schema_keys(prop)

        for key, value in bounds.items():
            assert prop[key] == value
        if "maximum" not in bounds and "exclusiveMaximum" not in bounds:
            assert "maximum" not in keys
            assert "exclusiveMaximum" not in keys
        if "exclusiveMinimum" in bounds:
            assert "minimum" not in keys

    @pytest.mark.parametrize(
        "title", ["GRPO", "GSPO", "CISPO", "LLMPPO", "LLMREINFORCE"]
    )
    def test_max_output_tokens_min_is_one_and_has_no_static_max(
        self, title: str
    ) -> None:
        prop = _algorithm_variant(title)["properties"]["max_output_tokens"]

        assert prop["minimum"] == 1
        assert prop["default"] is None
        assert prop["type"] == ["integer", "null"]
        assert "maximum" not in prop
        assert "exclusiveMaximum" not in prop

    @pytest.mark.parametrize(
        "title",
        ["GRPO", "GSPO", "CISPO", "LLMPPO", "LLMREINFORCE", "DPO", "SFT"],
    )
    def test_constrain_answer_pattern_stays_with_default_none(self, title: str) -> None:
        prop = _algorithm_variant(title)["properties"]["constrain_answer_pattern"]

        assert prop["default"] is None
        assert prop["type"] == ["string", "null"]

    @pytest.mark.parametrize(
        "title", ["GRPO", "GSPO", "CISPO", "LLMPPO", "LLMREINFORCE"]
    )
    def test_mini_batch_size_stays_with_default_none(self, title: str) -> None:
        prop = _algorithm_variant(title)["properties"]["mini_batch_size"]

        assert prop["default"] is None
        assert prop["minimum"] == 1
        assert prop["type"] == ["integer", "null"]

    @pytest.mark.parametrize(
        "title", ["GRPO", "GSPO", "CISPO", "LLMPPO", "LLMREINFORCE"]
    )
    def test_sampling_fields_are_on_the_schema(self, title: str) -> None:
        props = _algorithm_variant(title)["properties"]

        assert props["top_p"]["default"] == 1.0
        assert props["top_k"]["default"] == 50
        assert props["min_p"]["default"] == 0.0
        assert props["min_output_tokens"]["default"] is None
        assert props["reference_update_interval"]["default"] is None
        assert props["reference_update_interval"]["minimum"] == 1

    def test_llmppo_schema_has_no_group_size(self) -> None:
        assert "group_size" not in _algorithm_variant("LLMPPO")["properties"]

    def test_grpo_schema_includes_group_size(self) -> None:
        assert _algorithm_variant("GRPO")["properties"]["group_size"]["default"] == 8

    def test_numeric_hparams_are_bounded_on_both_sides(self) -> None:
        missing: list[str] = []
        for variant in manifest_schema()["properties"]["algorithm"]["oneOf"]:
            for name, prop in variant["properties"].items():
                if not _schema_types(prop) & {"number", "integer"}:
                    continue
                keys = _schema_keys(prop)
                if name not in NO_STATIC_MINIMUM and not (
                    {"minimum", "exclusiveMinimum"} & keys
                ):
                    missing.append(f"{variant['title']}.{name} has no minimum")
                if name not in NO_STATIC_MAXIMUM and not (
                    {"maximum", "exclusiveMaximum"} & keys
                ):
                    missing.append(f"{variant['title']}.{name} has no maximum")

        assert missing == []

    @pytest.mark.parametrize(
        ("spec_cls", "kwargs"),
        [
            (GRPOSpec, {"temperature": 0.0}),
            (GRPOSpec, {"temperature": -0.1}),
            (GRPOSpec, {"temperature": 2.1}),
            (GRPOSpec, {"top_p": 0.0}),
            (GRPOSpec, {"lr": 0.0}),
            (GRPOSpec, {"kl_clamp": 0.0}),
            (GRPOSpec, {"repetition_penalty": 0.0}),
            (GRPOSpec, {"vllm_importance_sampling_cap": 0.0}),
            (GRPOSpec, {"update_epochs": 11}),
            (GRPOSpec, {"max_output_tokens": 0}),
            (LLMPPOSpec, {"temperature": 0.0}),
            (LLMPPOSpec, {"temperature": 3.0}),
            (LLMREINFORCESpec, {"temperature": 0.0}),
            (LLMREINFORCESpec, {"temperature": -1.0}),
        ],
    )
    def test_out_of_range_hyperparameters_are_rejected(
        self, spec_cls: type, kwargs: dict
    ) -> None:
        with pytest.raises(ValidationError):
            spec_cls(**kwargs)

    def test_values_above_the_old_ui_caps_are_kept(self) -> None:
        grpo = GRPOSpec(
            kl_clamp=1000.0,
            repetition_penalty=10.0,
            max_grad_norm=1000.0,
            vllm_importance_sampling_cap=1000.0,
            vllm_max_logprob_gap=1000.0,
            update_epochs=10,
        )
        dpo = DPOSpec(nll_alpha=50.0)

        assert grpo.kl_clamp == 1000.0
        assert grpo.repetition_penalty == 10.0
        assert grpo.max_grad_norm == 1000.0
        assert grpo.vllm_importance_sampling_cap == 1000.0
        assert grpo.vllm_max_logprob_gap == 1000.0
        assert grpo.update_epochs == 10
        assert dpo.nll_alpha == 50.0

    def test_sampling_penalties_match_vllm(self) -> None:
        props = manifest_schema()["$defs"]["VLLMConfig"]["properties"]

        assert props["presence_penalty"]["minimum"] == -2.0
        assert props["presence_penalty"]["maximum"] == 2.0
        assert props["frequency_penalty"]["minimum"] == -2.0
        assert props["frequency_penalty"]["maximum"] == 2.0

    def test_max_output_tokens_below_the_model_context_is_kept(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                algorithm={"max_output_tokens": 2048},
                network={"max_context_length": 4096},
            )
        )

        assert validated.algorithm.max_output_tokens == 2048
        assert validated.algorithm.max_model_len == 4096

    def test_max_output_tokens_at_the_model_context_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="context length \\(512\\)"):
            TrainingManifest.model_validate(
                manifest(GRPO, algorithm={"max_output_tokens": 512})
            )

    def test_omitted_max_output_tokens_skips_the_context_check(self) -> None:
        validated = TrainingManifest.model_validate(GRPO)

        assert validated.algorithm.max_output_tokens is None


HPO_MUTATION = {"probabilities": {"no_mut": 0.5, "rl_hp_mut": 0.5}}
NO_OP_PROBABILITIES = {
    "no_mut": 1.0,
    "arch_mut": 0.0,
    "new_layer": 0.0,
    "params_mut": 0.0,
    "act_mut": 0.0,
    "rl_hp_mut": 0.0,
}


class TestHpoFields:
    def test_hpo_flag_and_mutation_are_both_reported(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, training={"hpo": True}, mutation=HPO_MUTATION)
        )

        assert hpo_fields(validated) == ["training.hpo", "mutation"]

    def test_hpo_flag_alone_is_reported(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, training={"hpo": True})
        )

        assert hpo_fields(validated) == ["training.hpo"]

    def test_mutating_section_alone_is_reported(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, mutation=HPO_MUTATION)
        )

        assert hpo_fields(validated) == ["mutation"]

    def test_rl_hp_selection_is_reported(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                mutation={
                    "probabilities": NO_OP_PROBABILITIES,
                    "rl_hp_selection": {"lr": {"min": 1e-4, "max": 1e-2}},
                },
            )
        )

        assert hpo_fields(validated) == ["mutation"]

    @pytest.mark.parametrize(
        "mutation",
        [None, {}, {"probabilities": NO_OP_PROBABILITIES}],
        ids=["absent", "empty", "no-op"],
    )
    def test_no_hpo_reports_nothing(self, mutation: dict | None) -> None:
        document = manifest(GRPO, training={"hpo": False})
        if mutation is not None:
            document["mutation"] = mutation

        validated = TrainingManifest.model_validate(document)

        assert hpo_fields(validated) == []

    def test_llm_population_is_reported(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, training={"pop_size": 2})
        )

        assert hpo_fields(validated) == ["training.pop_size"]

    def test_llm_hpo_mutation_and_population_are_all_reported(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(GRPO, training={"hpo": True, "pop_size": 2}, mutation=HPO_MUTATION)
        )

        assert hpo_fields(validated) == [
            "training.hpo",
            "mutation",
            "training.pop_size",
        ]

    def test_llm_single_agent_with_no_op_mutation_reports_nothing(self) -> None:
        validated = TrainingManifest.model_validate(
            manifest(
                GRPO,
                training={"hpo": False, "pop_size": 1},
                mutation={"probabilities": NO_OP_PROBABILITIES, "rl_hp_selection": {}},
            )
        )

        assert hpo_fields(validated) == []


# Arena takes a context from the catalog model's options; 512 is not one.
GRPO_ON_ARENA = manifest(GRPO, network={"max_context_length": 1024})


class TestVerdictHpo:
    def test_hpo_manifest_is_invalid(self) -> None:
        result = verdict(
            manifest(GRPO_ON_ARENA, training={"hpo": True}, mutation=HPO_MUTATION)
        )

        assert result["ok"] is False
        assert [error["path"] for error in result["errors"]] == [
            "training.hpo",
            "mutation",
        ]
        assert {error["message"] for error in result["errors"]} == {
            HPO_UNAVAILABLE_MESSAGE
        }

    def test_manifest_without_hpo_is_valid(self) -> None:
        result = verdict(GRPO_ON_ARENA)

        assert result["ok"] is True
        assert "errors" not in result

    def test_llm_population_is_invalid(self) -> None:
        result = verdict(manifest(GRPO_ON_ARENA, training={"pop_size": 2}))

        assert result["ok"] is False
        assert result["errors"] == [
            {
                "path": "training.pop_size",
                "message": HPO_UNAVAILABLE_MESSAGE,
                "kind": "hpo_unavailable",
            }
        ]
        assert "HPO and populations aren't available for LLM training on Arena" in (
            HPO_UNAVAILABLE_MESSAGE
        )


class TestVerdictArenaOptions:
    def test_stored_verdict_rejects_a_batch_size_outside_the_options(self) -> None:
        result = verdict(manifest(GRPO_ON_ARENA, algorithm={"batch_size": 6}))

        assert result["ok"] is False
        assert [error["path"] for error in result["errors"]] == ["algorithm.batch_size"]

    def test_stored_verdict_rejects_a_context_outside_the_options(self) -> None:
        result = verdict(GRPO)

        assert result["ok"] is False
        assert [error["path"] for error in result["errors"]] == [
            "network.max_context_length"
        ]

    def test_warns_when_the_context_is_the_filled_default(self) -> None:
        document = copy.deepcopy(GRPO)
        del document["network"]["max_context_length"]

        result = verdict(document)

        assert result["ok"] is True, result
        assert any("default context of 2048 tokens" in w for w in result["warnings"])

    def test_filled_default_leaves_room_for_the_output_cap(self) -> None:
        document = copy.deepcopy(GRPO)
        del document["network"]["max_context_length"]
        document["algorithm"]["max_output_tokens"] = 4096

        result = verdict(document)

        assert result["ok"] is True, result
        assert any("default context of 8192 tokens" in w for w in result["warnings"])

    def test_filled_default_reads_a_quoted_output_cap(self) -> None:
        document = copy.deepcopy(GRPO)
        del document["network"]["max_context_length"]
        document["algorithm"]["max_output_tokens"] = "4096"

        result = verdict(document)

        assert result["ok"] is True, result
        assert any("default context of 8192 tokens" in w for w in result["warnings"])

    def test_filled_default_leaves_room_for_answer_continuation(self) -> None:
        document = copy.deepcopy(GRPO)
        del document["network"]["max_context_length"]
        document["algorithm"]["max_output_tokens"] = 2048
        document["algorithm"]["answer_continuation"] = True
        document["algorithm"]["constrain_answer_pattern"] = r"\d+"

        result = verdict(document)

        assert result["ok"] is True, result
        assert any("default context of 8192 tokens" in w for w in result["warnings"])

    def test_no_context_warning_when_the_manifest_sets_it(self) -> None:
        result = verdict(GRPO_ON_ARENA)

        assert not any("default context" in w for w in result["warnings"])


class TestVerdictClassicRl:
    def test_classic_manifest_is_invalid(self) -> None:
        result = verdict(DQN)

        assert result["ok"] is False
        assert result["errors"] == [
            {
                "path": "algorithm.name",
                "message": CLASSIC_RL_UNAVAILABLE_MESSAGE,
                "kind": "classic_rl_unavailable",
            }
        ]

    @pytest.mark.parametrize(
        ("name", "classic"),
        [
            ("PPO", True),
            ("Rainbow DQN", True),
            ("MADDPG", True),
            ("GRPO", False),
            ("SFT", False),
            ("Supervised", False),
        ],
    )
    def test_is_classic_rl(self, name: str, classic: bool) -> None:
        assert is_classic_rl(name) is classic


class TestVerdictClassicOnlyKeys:
    IGNORED = (
        "training.learning_delay, training.sum_scores ignored; removed in a "
        "future agilerl release."
    )

    def test_dropped_keys_are_reported_once_as_a_verdict_warning(self, recwarn) -> None:
        result = verdict(
            manifest(GRPO_ON_ARENA, training={"learning_delay": 0, "sum_scores": True})
        )

        assert result["ok"] is True
        assert result["warnings"].count(self.IGNORED) == 1
        assert len(recwarn) == 0

    def test_manifest_without_them_has_no_such_warning(self) -> None:
        assert not any("ignored" in warning for warning in verdict(GRPO)["warnings"])


class TestTrainingManifestAsyncRolloutDefaults:
    @staticmethod
    def grpo_manifest(training: dict[str, Any]) -> dict[str, Any]:
        return {
            "algorithm": {"name": "GRPO"},
            "environment": {"name": "env", "env_type": "rollout", "env_image": "env:1"},
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B",
                "max_context_length": 1024,
            },
            "training": training,
        }

    def test_async_run_gets_llm_buffer_and_oldest_turn_stamp(self) -> None:
        # Act
        payload = TrainingManifest.model_validate(self.grpo_manifest({})).to_payload()

        # Assert
        assert payload["replay_buffer"]["kind"] == "llm"
        assert payload["training"]["rollout_version_stamp"] == "oldest_turn"

    def test_keeps_an_authored_stamp(self) -> None:
        # Act
        payload = TrainingManifest.model_validate(
            self.grpo_manifest({"rollout_version_stamp": "publish"})
        ).to_payload()

        # Assert
        assert payload["training"]["rollout_version_stamp"] == "publish"

    def test_colocated_run_gets_neither(self) -> None:
        # Act
        payload = TrainingManifest.model_validate(
            self.grpo_manifest({"rollout_mode": "colocated"})
        ).to_payload()

        # Assert
        assert "replay_buffer" not in payload
        assert "rollout_version_stamp" not in payload["training"]
