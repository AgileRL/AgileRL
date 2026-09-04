# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The manifest -> estimator bridge.

The whole point of the bridge is that the gate reads the *same* validated
document the submission carries, so most tests here assert one direction:
write a manifest field, see it land on the estimator's settings.
"""

import copy
import json
from pathlib import Path

import pytest

from agilerl.arena.memory.manifest import (
    device_spec_from_resource_class,
    estimate_manifest,
    generation_knobs_from_manifest,
    lookup_gpu,
    run_config_from_manifest,
    training_knobs_from_manifest,
)
from agilerl.arena.models.manifest import TrainingManifest

TINY_CONFIG = json.loads(
    (Path(__file__).parent / "assets" / "tiny_llm" / "config.json").read_text()
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
    "algorithm": {"name": "DQN"},
    "environment": {"name": "LunarLander-v3", "num_envs": 8},
    "training": {"max_steps": 1000, "pop_size": 2},
}

L4_TIER = {"name": "l4-1x", "gpu_type": "NVIDIA L4", "num_gpus": 1, "ram_gb": 32}


def manifest(**overrides: dict) -> dict:
    out = copy.deepcopy(GRPO)
    for section, values in overrides.items():
        out.setdefault(section, {}).update(values)
    return out


class TestTrainingKnobs:
    def test_knobs_come_from_the_manifest(self):
        knobs = training_knobs_from_manifest(
            manifest(
                algorithm={
                    "beta": 0.01,
                    "micro_batch_size_per_gpu": 2,
                    "chunk_rows": 256,
                    "quantization": "nf4",
                    "attn_implementation": "eager",
                    "gradient_checkpointing": False,
                    "use_separate_reference_adapter": False,
                },
                network={
                    "lora_config": {"lora_r": 32},
                    "max_context_length": 2048,
                },
            )
        )
        assert knobs.algorithm == "grpo"
        assert knobs.batch_size == 2
        assert knobs.group_size == 4
        # The update carries prompts x group_size completion rows, exactly as
        # the trainer chunks it.
        assert knobs.trajectories_per_update == 8
        assert knobs.micro_batch_size_per_gpu == 2
        assert knobs.max_model_len == 2048  # lifted from network section
        assert knobs.lora_rank == 32
        assert knobs.beta == 0.01
        assert knobs.chunk_rows == 256
        assert knobs.quantization == "nf4"
        assert knobs.attn_implementation == "eager"
        assert knobs.gradient_checkpointing is False
        assert knobs.use_separate_reference_adapter is False

    def test_defaults_match_the_algorithm_constructors(self):
        knobs = training_knobs_from_manifest(GRPO)
        assert knobs.lora_rank == 16  # LoraConfigDict default
        assert knobs.quantization == "none"
        assert knobs.attn_implementation == "auto"
        assert knobs.distributed == "none"

    def test_quantization_spellings(self):
        for given, expected in (
            ({"load_in_4bit": True}, "nf4"),
            ({"load_in_8bit": True}, "int8"),
            ("int8", "int8"),
            ("NF4", "nf4"),
        ):
            knobs = training_knobs_from_manifest(
                manifest(algorithm={"quantization": given})
            )
            assert knobs.quantization == expected, given

    def test_attention_only_lora_scope(self):
        knobs = training_knobs_from_manifest(
            manifest(
                network={
                    "lora_config": {"target_modules": ["q_proj", "k_proj", "v_proj"]}
                }
            )
        )
        assert knobs.lora_target_scope == "attention-only"

    def test_data_parallel_training_implies_deepspeed(self):
        knobs = training_knobs_from_manifest(
            manifest(training={"training_gpus_per_agent": 2})
        )
        assert knobs.distributed == "deepspeed"

    def test_a_classic_rl_manifest_is_refused(self):
        with pytest.raises(ValueError, match="LLM fine-tuning"):
            training_knobs_from_manifest(DQN)


class TestGenerationKnobs:
    def test_colocated_defaults_are_the_resolved_engine_config(self):
        # Manifest validation fills the colocated vLLM config in; the bridge
        # must read that resolution, not invent its own.
        knobs = generation_knobs_from_manifest(GRPO)
        assert knobs.gpu_memory_utilization == 0.9
        assert knobs.max_num_seqs == 16
        assert knobs.max_model_len == 512
        assert knobs.concurrent_requests == 8  # batch_size x group_size

    def test_explicit_engine_config_wins(self):
        knobs = generation_knobs_from_manifest(
            manifest(
                algorithm={
                    "vllm_config": {
                        "gpu_memory_utilization": 0.4,
                        "max_num_seqs": 4,
                        "kv_cache_dtype": "fp8_e4m3",
                        "enforce_eager": True,
                        "dtype": "float16",
                    }
                }
            )
        )
        assert knobs.gpu_memory_utilization == 0.4
        assert knobs.max_num_seqs == 4
        assert knobs.kv_cache_dtype == "fp8"
        assert knobs.enforce_eager is True
        assert knobs.weight_dtype == "fp16"

    def test_engine_quantization_becomes_a_weight_variant(self):
        doc = manifest(algorithm={"vllm_config": {"quantization": "awq"}})
        config = run_config_from_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert config.generation.weight_variant == "engine"
        assert config.model.variant("engine").quantization == "awq"


class TestDeviceFromResourceClass:
    def test_known_tier(self):
        device = device_spec_from_resource_class(L4_TIER)
        assert device.name == "NVIDIA L4"
        assert device.total_bytes == 24 * 1024**3
        assert device.supports_fp8  # Ada, cc 8.9

    def test_gpu_type_spellings(self):
        assert lookup_gpu("a100 80gb").total_gib == 80
        assert lookup_gpu("NVIDIA A100-SXM4-40GB").total_gib == 40
        assert lookup_gpu("L40S").total_gib == 48
        assert lookup_gpu("l4").name == "NVIDIA L4"
        assert lookup_gpu("Tesla T4").cc_major == 7  # pre-Ampere: no flash, no fp8

    def test_unknown_gpu_needs_an_explicit_size(self):
        tier = {"name": "exotic", "gpu_type": "TPU v5e"}
        with pytest.raises(ValueError, match="Unknown gpu_type"):
            device_spec_from_resource_class(tier)
        device = device_spec_from_resource_class(tier, gpu_memory_gib=16)
        assert device.total_bytes == 16 * 1024**3

    def test_cpu_tier_is_refused(self):
        with pytest.raises(ValueError, match="no GPU"):
            device_spec_from_resource_class({"name": "cpu-only", "gpu_type": None})


class TestRunConfig:
    def test_colocated_by_default(self):
        config = run_config_from_manifest(
            GRPO, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert config.colocated
        assert config.model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_async_rollout_disaggregates(self):
        doc = manifest(
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )
        config = run_config_from_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert not config.colocated
        assert config.gen_device is not None

    def test_estimate_manifest_produces_both_bars(self):
        estimate = estimate_manifest(
            GRPO, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert estimate.training.components
        assert estimate.generation.components

    def test_validated_manifest_object_is_accepted(self):
        validated = TrainingManifest.model_validate(copy.deepcopy(GRPO))
        knobs = training_knobs_from_manifest(validated)
        assert knobs.algorithm == "grpo"


class TestDistributedAndOrchestration:
    def test_zero_stage_and_gpu_count_come_from_the_manifest(self):
        knobs = training_knobs_from_manifest(
            manifest(
                algorithm={"zero_stage": 3},
                training={"training_gpus_per_agent": 4},
            )
        )
        assert knobs.distributed == "deepspeed"
        assert knobs.zero_stage == 3
        assert knobs.n_training_gpus == 4
        assert knobs.dp_world_size == 4

    def test_arena_runs_are_orchestrated(self):
        config = run_config_from_manifest(
            GRPO, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert config.orchestrated


class TestEnginelessAlgorithms:
    def test_sft_manifest_gets_an_empty_generation_bar(self):
        doc = {
            "algorithm": {"name": "SFT"},
            "environment": {"dataset": "openai/gsm8k"},
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
            },
            "training": {"max_steps": 100},
        }
        estimate = estimate_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert estimate.generation.components == ()
        assert estimate.training.components
