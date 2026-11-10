# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Which Ray runs generate on async rollout engines, how many they start, their cycle size, whether they draft and how they sync LoRA."""

from typing import Any

import pytest

from agilerl.arena.memory.resolve.rollout import (
    MERGED_SYNC_BUCKET_BYTES,
    MULTI_TURN_SPECULATIVE_CONFIG,
    apply_run_rollout_mode,
    async_rollout_algorithm,
    checkpoint_quantized,
    default_rollout_batch_size,
    drafts_by_default,
    drafts_with_mtp,
    merged_lora_sync_blockers,
    merged_sync_bucket_bytes,
    rollout_engines_blocker,
    run_rollout_engines,
    runs_dual_stream_lora,
)
from agilerl.arena.memory.specs import GiB, MiB
from agilerl.arena.models.algorithms import (
    DPOSpec,
    DQNSpec,
    GRPOSpec,
    LLMPPOSpec,
    LLMREINFORCESpec,
    SFTSpec,
)
from agilerl.arena.models.algorithms.base import AlgorithmSpec
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

# quantization_config of openai/gpt-oss-20b on the Hub.
GPT_OSS_MXFP4 = {
    "modules_to_not_convert": [
        "model.layers.*.self_attn",
        "model.layers.*.mlp.router",
        "model.embed_tokens",
        "lm_head",
    ],
    "quant_method": "mxfp4",
}


class TestAsyncRolloutAlgorithm:
    @pytest.mark.parametrize("algorithm", [GRPOSpec, LLMPPOSpec, LLMREINFORCESpec])
    def test_rollout_llm_algorithms_run_async(
        self, algorithm: type[AlgorithmSpec]
    ) -> None:
        assert async_rollout_algorithm(algorithm) is True

    @pytest.mark.parametrize("algorithm", [SFTSpec, DPOSpec, DQNSpec])
    def test_dataset_and_classic_algorithms_do_not(
        self, algorithm: type[AlgorithmSpec]
    ) -> None:
        assert async_rollout_algorithm(algorithm) is False


class TestRunRolloutEngines:
    @pytest.mark.parametrize("configured", [2, "auto"])
    def test_an_async_run_keeps_a_set_count(self, configured: int | str) -> None:
        assert run_rollout_engines(True, configured) == configured

    def test_an_async_run_reads_zero_as_auto(self) -> None:
        assert run_rollout_engines(True, 0) == "auto"

    @pytest.mark.parametrize("configured", [0, 2])
    def test_a_run_without_rollout_keeps_the_written_count(
        self, configured: int
    ) -> None:
        assert run_rollout_engines(False, configured) == configured


class TestRolloutEnginesBlocker:
    @pytest.mark.parametrize("configured", [0, 2, "auto"])
    def test_an_async_run_takes_any_engine_count(self, configured: int | str) -> None:
        assert rollout_engines_blocker(True, configured) is None

    def test_a_run_without_rollout_takes_no_engines(self) -> None:
        assert rollout_engines_blocker(False, 0) is None

    @pytest.mark.parametrize("configured", [2, "auto"])
    def test_a_run_without_rollout_rejects_engines(self, configured: int | str) -> None:
        assert rollout_engines_blocker(False, configured) == (
            f"rollout_engines_per_agent={configured} requires async rollout."
        )


class TestDefaultRolloutBatchSize:
    def test_a_batch_every_trainer_and_engine_divides_is_kept(self) -> None:
        assert default_rollout_batch_size(64, 4, 2) == 64

    def test_rounds_up_to_the_trainer_count(self) -> None:
        assert default_rollout_batch_size(6, 4, 1) == 8

    def test_rounds_up_to_both_trainers_and_engines(self) -> None:
        assert default_rollout_batch_size(10, 2, 3) == 12


class TestApplyRunRolloutMode:
    def test_an_async_run_sets_async_and_auto_engines(self) -> None:
        # Arrange
        training: dict[str, Any] = {"rollout_mode": "colocated", "max_steps": 4}

        # Act
        apply_run_rollout_mode(training, True)

        # Assert
        assert training == {
            "rollout_mode": "async",
            "rollout_engines_per_agent": "auto",
            "max_steps": 4,
        }

    def test_an_async_run_keeps_its_engine_count(self) -> None:
        # Arrange
        training: dict[str, Any] = {"rollout_engines_per_agent": 3}

        # Act
        apply_run_rollout_mode(training, True)

        # Assert
        assert training == {"rollout_mode": "async", "rollout_engines_per_agent": 3}

    def test_a_run_without_rollout_drops_the_rollout_mode(self) -> None:
        # Arrange
        training: dict[str, Any] = {"rollout_mode": "async", "max_steps": 4}

        # Act
        apply_run_rollout_mode(training, False)

        # Assert
        assert training == {"max_steps": 4}


class TestDraftsByDefault:
    def test_multi_turn_attention_only_engine_drafts(self) -> None:
        assert drafts_by_default(True, False, 0.0, False) is True

    def test_single_turn_engine_does_not_draft(self) -> None:
        assert drafts_by_default(False, False, 0.0, False) is False

    def test_hybrid_ssm_engine_does_not_draft(self) -> None:
        assert drafts_by_default(True, True, 0.0, False) is False

    def test_min_p_engine_does_not_draft(self) -> None:
        assert drafts_by_default(True, False, 0.05, False) is False

    def test_logit_bias_engine_does_not_draft(self) -> None:
        assert drafts_by_default(True, False, 0.0, True) is False


class TestRunsDualStreamLora:
    def test_dense_engine_runs_two_lora_streams(self) -> None:
        assert runs_dual_stream_lora(False) is True

    def test_moe_engine_runs_one_lora_stream(self) -> None:
        assert runs_dual_stream_lora(True) is False


class TestMergedLoraSyncBlockers:
    @pytest.mark.parametrize("quantization", [None, "none", "None"])
    def test_unquantized_engine_takes_merged_weights(
        self, quantization: str | None
    ) -> None:
        assert merged_lora_sync_blockers(quantization, False, False, False) == []

    def test_quantized_engine_keeps_serving_the_adapter(self) -> None:
        assert merged_lora_sync_blockers("fp8", False, False, False) == [
            "vllm_config.quantization is 'fp8'"
        ]

    def test_quantized_checkpoint_keeps_serving_the_adapter(self) -> None:
        assert merged_lora_sync_blockers(None, True, False, False) == [
            "the checkpoint's config.json ships a quantization config"
        ]

    def test_tower_lora_keeps_serving_the_adapter(self) -> None:
        assert merged_lora_sync_blockers(None, False, True, False) == [
            "lora_config.tower_target_modules targets the vision tower"
        ]

    def test_mtp_drafting_keeps_serving_the_adapter(self) -> None:
        assert merged_lora_sync_blockers(None, False, False, True) == [
            "vllm_engine_args.speculative_config drafts with MTP"
        ]

    def test_lists_every_blocker(self) -> None:
        assert merged_lora_sync_blockers("fp8", True, True, True) == [
            "vllm_config.quantization is 'fp8'",
            "the checkpoint's config.json ships a quantization config",
            "lora_config.tower_target_modules targets the vision tower",
            "vllm_engine_args.speculative_config drafts with MTP",
        ]


class TestCheckpointQuantized:
    def test_catalogued_bf16_checkpoint_is_unquantized(self) -> None:
        config = SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config

        assert checkpoint_quantized(config) is False

    def test_hub_mxfp4_checkpoint_is_quantized(self) -> None:
        config = {
            **SUPPORTED_MODEL_INFO["openai/gpt-oss-20b"].config,
            "quantization_config": GPT_OSS_MXFP4,
        }

        assert checkpoint_quantized(config) is True

    def test_compressed_tensors_checkpoint_is_quantized(self) -> None:
        config = {"compression_config": {"quant_method": "compressed-tensors"}}

        assert checkpoint_quantized(config) is True

    def test_multimodal_text_config_quantization_counts(self) -> None:
        config = {"text_config": {"quantization_config": {"quant_method": "fp8"}}}

        assert checkpoint_quantized(config) is True

    def test_null_quantization_config_is_unquantized(self) -> None:
        assert checkpoint_quantized({"quantization_config": None}) is False


class TestDraftsWithMtp:
    @pytest.mark.parametrize("method", ["mtp", "qwen3_5_mtp", "deepseek_mtp"])
    def test_mtp_methods_draft_with_mtp(self, method: str) -> None:
        assert drafts_with_mtp({"method": method, "num_speculative_tokens": 1}) is True

    @pytest.mark.parametrize("method", [None, "draft_model"])
    def test_draft_model_left_to_vllm_counts_as_mtp(self, method: str | None) -> None:
        config = {
            "method": method,
            "model": "org/Model-MTP-Draft",
            "num_speculative_tokens": 1,
        }

        assert drafts_with_mtp(config) is True

    def test_ngram_drafting_is_not_mtp(self) -> None:
        assert drafts_with_mtp(MULTI_TURN_SPECULATIVE_CONFIG) is False

    def test_ngram_model_without_method_is_not_mtp(self) -> None:
        assert drafts_with_mtp({"model": "ngram", "num_speculative_tokens": 4}) is False

    def test_no_drafting_is_not_mtp(self) -> None:
        assert drafts_with_mtp(None) is False


class TestMergedSyncBucketBytes:
    def test_small_tensors_stream_in_default_rounds(self) -> None:
        assert merged_sync_bucket_bytes(64 * GiB, 4 * MiB) == MERGED_SYNC_BUCKET_BYTES

    def test_a_tensor_over_one_round_sizes_the_buffer(self) -> None:
        assert merged_sync_bucket_bytes(64 * GiB, GiB) == GiB

    def test_buffer_never_exceeds_the_merged_weights(self) -> None:
        assert merged_sync_bucket_bytes(100 * MiB, 10 * MiB) == 100 * MiB
