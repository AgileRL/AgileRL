# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Rollout engine count and engine defaults picked from the allocation."""

import copy

from agilerl.arena.memory.estimator import estimate_generation
from agilerl.arena.memory.generation import KV_MARGIN_BYTES
from agilerl.arena.memory.manifest import (
    MANIFEST_WEIGHT_VARIANTS,
    generation_settings_from_manifest,
    lookup_gpu,
    validated_run_manifest,
)
from agilerl.arena.memory.resolve.engine import (
    resolve_engine_fast_path,
    resolve_rollout_engines_per_agent,
    servable_engine_widths,
)
from agilerl.arena.memory.specs import ModelArch, ModelSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

GEMMA_4_E4B = "google/gemma-4-E4B-it"
# Gemma 4 E4B's text tower leaves a 0.95 L4 engine under 1 GiB for the
# fast-path defaults and the KV cache.
GEMMA_ENGINE = {
    "algorithm": {
        "name": "GRPO",
        "group_size": 4,
        "batch_size": 8,
        "vllm_config": {
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.95,
            "tensor_parallel_size": 1,
        },
    },
    "environment": {
        "env_type": "rollout",
        "env_url": "http://env",
        "modality": "text",
        "max_turns": 2,
    },
    "network": {
        "pretrained_model_name_or_path": GEMMA_4_E4B,
        "max_context_length": 2048,
        "lora_config": {
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    },
    "training": {
        "max_steps": 10,
        "rollout_mode": "async",
        "rollout_engines_per_agent": 1,
        "rollout_batch_size": 4,
    },
    "replay_buffer": {"kind": "llm"},
}


def gemma_engine(**algorithm: object) -> TrainingManifest:
    doc = copy.deepcopy(GEMMA_ENGINE)
    doc["algorithm"].update(algorithm)
    return validated_run_manifest(TrainingManifest.model_validate(doc))


def gemma_model() -> ModelSpec:
    info = SUPPORTED_MODEL_INFO[GEMMA_4_E4B]
    return ModelSpec(
        model_id=GEMMA_4_E4B,
        arch=ModelArch.from_hf_config(info.config),
        n_params=info.inspected["num_params"],
        variants=MANIFEST_WEIGHT_VARIANTS,
    )


class TestResolveEngineFastPath:
    def test_engine_without_kv_room_pins_the_defaults_off(self) -> None:
        # Arrange
        manifest = gemma_engine()
        l4 = lookup_gpu("L4").device_spec()

        # Act
        spec = resolve_engine_fast_path(manifest, l4)

        # Assert
        assert spec is manifest.algorithm
        assert spec.vllm_engine_args == {
            "speculative_config": None,
            "lora_dual_stream": False,
        }
        assert spec.vllm_config.max_num_batched_tokens == 2048
        kv = next(
            c
            for c in estimate_generation(
                gemma_model(), l4, generation_settings_from_manifest(manifest)
            ).components
            if c.key == "kv_cache"
        )
        assert (
            kv.detail["startup_pool"]
            >= kv.detail["full_sequence_bytes"] + KV_MARGIN_BYTES
        )

    def test_engine_with_kv_room_keeps_the_defaults(self) -> None:
        # Arrange
        manifest = gemma_engine()

        # Act
        spec = resolve_engine_fast_path(manifest, lookup_gpu("A100-80").device_spec())

        # Assert
        assert spec is None
        assert manifest.algorithm.vllm_engine_args == {}
        assert manifest.algorithm.vllm_config.max_num_batched_tokens is None

    def test_pinned_engine_settings_are_kept(self) -> None:
        # Arrange
        manifest = gemma_engine(
            vllm_engine_args={"speculative_config": None},
            vllm_config={
                **GEMMA_ENGINE["algorithm"]["vllm_config"],
                "max_num_batched_tokens": 4096,
            },
        )

        # Act
        spec = resolve_engine_fast_path(manifest, lookup_gpu("L4").device_spec())

        # Assert
        assert spec.vllm_engine_args == {
            "speculative_config": None,
            "lora_dual_stream": False,
        }
        assert spec.vllm_config.max_num_batched_tokens == 4096

    def test_step_without_chunked_prefill_holds_a_full_prompt(self) -> None:
        # Arrange
        doc = copy.deepcopy(GEMMA_ENGINE)
        doc["algorithm"]["vllm_engine_args"] = {"enable_chunked_prefill": False}
        doc["network"]["max_context_length"] = 4096
        manifest = validated_run_manifest(TrainingManifest.model_validate(doc))

        # Act
        spec = resolve_engine_fast_path(manifest, lookup_gpu("L4").device_spec())

        # Assert
        assert spec.vllm_config.max_num_batched_tokens == 4096

    def test_algorithm_without_rollouts_is_left_unchanged(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(
            {
                "algorithm": {"name": "SFT", "batch_size": 8},
                "environment": {"dataset": "openai/gsm8k"},
                "network": {
                    "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "max_context_length": 512,
                },
                "training": {"max_steps": 100},
            }
        )

        # Act
        spec = resolve_engine_fast_path(manifest, lookup_gpu("L4").device_spec())

        # Assert
        assert spec is None

    def test_unset_engine_width_is_left_unchanged(self) -> None:
        # Arrange
        manifest = gemma_engine(
            vllm_config={"max_num_seqs": 16, "gpu_memory_utilization": 0.95}
        )

        # Act
        spec = resolve_engine_fast_path(manifest, lookup_gpu("L4").device_spec())

        # Assert
        assert spec is None
        assert manifest.algorithm.vllm_engine_args == {}


class TestResolveRolloutEnginesPerAgent:
    def test_auto_splits_the_surplus_across_members(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            "auto",
            total_gpus=8,
            pop_size=2,
            training_gpus_per_agent=1,
            gpus_per_engine=1,
        )

        # Assert
        assert engines == 3

    def test_auto_rounds_a_partial_engine_down(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            "auto",
            total_gpus=8,
            pop_size=2,
            training_gpus_per_agent=1,
            gpus_per_engine=2,
        )

        # Assert
        assert engines == 1

    def test_set_count_is_kept(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            2,
            total_gpus=8,
            pop_size=1,
            training_gpus_per_agent=1,
            gpus_per_engine=1,
        )

        # Assert
        assert engines == 2

    def test_auto_without_a_spare_engine_slot_is_zero(self) -> None:
        # Act
        engines = resolve_rollout_engines_per_agent(
            "auto",
            total_gpus=4,
            pop_size=2,
            training_gpus_per_agent=2,
            gpus_per_engine=1,
        )

        # Assert
        assert engines == 0


def rollout_manifest(hub_id: str, context: int) -> TrainingManifest:
    """Async CISPO rollout manifest on ``hub_id`` with the engine width omitted."""
    return validated_run_manifest(
        TrainingManifest.model_validate(
            {
                "algorithm": {
                    "name": "CISPO",
                    "batch_size": 64,
                    "group_size": 16,
                    "max_output_tokens": 2048,
                },
                "environment": {
                    "env_type": "rollout",
                    "env_image": "env:latest",
                    "modality": "text",
                    "max_turns": 20,
                },
                "network": {
                    "pretrained_model_name_or_path": hub_id,
                    "max_context_length": context,
                },
                "training": {"rollout_mode": "async", "pop_size": 1},
                "replay_buffer": {"kind": "llm"},
            }
        )
    )


class TestServableEngineWidths:
    def test_moe_model_skips_widths_without_room_or_an_even_head_split(self) -> None:
        # Arrange: Qwen3.6-35B-A3B's weights fill one 48 GB L40S, and its
        # attention and linear-attention heads do not split 3, 5, 6 or 7 ways.
        manifest = rollout_manifest("Qwen/Qwen3.6-35B-A3B", 32768)
        l40s = lookup_gpu("L40S").device_spec()

        # Act
        widths = servable_engine_widths(manifest, l40s, 7)

        # Assert
        assert widths == [2, 4]

    def test_dense_model_widths_split_its_heads_evenly(self) -> None:
        # Arrange: Qwen3-4B has 32 attention heads and 8 KV heads.
        manifest = rollout_manifest("Qwen/Qwen3-4B", 16384)
        h100 = lookup_gpu("H100-80").device_spec()

        # Act
        widths = servable_engine_widths(manifest, h100, 7)

        # Assert
        assert widths == [1, 2, 4]

    def test_widths_stop_at_the_widest_asked_for(self) -> None:
        # Arrange
        manifest = rollout_manifest("Qwen/Qwen3-4B", 16384)
        h100 = lookup_gpu("H100-80").device_spec()

        # Act
        widths = servable_engine_widths(manifest, h100, 1)

        # Assert
        assert widths == [1]
