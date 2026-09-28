# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Manifest to estimator: a written field appears on the estimator settings."""

import copy
import json
from pathlib import Path

import pytest

from agilerl.arena.memory.manifest import (
    device_spec_from_resource_class,
    estimate_manifest,
    generation_settings_from_manifest,
    lookup_gpu,
    run_config_from_manifest,
    training_settings_from_manifest,
)
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.manifest import TrainingManifest

TINY_CONFIG = json.loads(
    (Path(__file__).parent / "assets" / "tiny_llm" / "config.json").read_text()
)

GRPO = {
    "algorithm": {
        "name": "GRPO",
        "group_size": 4,
        "batch_size": 2,
        "micro_batch_size_per_gpu": 1,
    },
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


class TestTrainingSettings:
    def test_settings_come_from_the_manifest(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "beta": 0.01,
                    "micro_batch_size_per_gpu": 1,
                    "chunk_rows": 256,
                    "attn_implementation": "sdpa",
                    "gradient_checkpointing": False,
                    "use_separate_reference_adapter": False,
                },
                network={
                    "lora_config": {"lora_r": 32},
                    "max_context_length": 2048,
                },
            )
        )
        assert settings.algorithm == "grpo"
        assert settings.group_size == 4
        # Update size is prompts x group_size, matching the trainer chunk.
        assert settings.trajectories_per_update == 8
        assert settings.max_model_len == 2048  # lifted from network section
        assert settings.lora_rank == 32
        assert settings.beta == 0.01
        assert settings.chunk_rows == 256
        assert settings.gradient_checkpointing is False
        assert settings.use_separate_reference_adapter is False

    def test_defaults_match_the_algorithm_constructors(self):
        settings = training_settings_from_manifest(GRPO)
        assert settings.lora_rank == 16  # LoraConfigDict default
        assert settings.n_training_gpus == 1

    @pytest.mark.parametrize(
        "given",
        ["nf4", "int8", "n4f", {"load_in_4bit": True}, {"load_in_8bit": True}],
    )
    def test_trainer_quantization_is_refused(self, given):
        with pytest.raises(ValueError, match="Trainer quantization"):
            training_settings_from_manifest(manifest(algorithm={"quantization": given}))

    def test_attention_only_lora_scope(self):
        settings = training_settings_from_manifest(
            manifest(
                network={
                    "lora_config": {"target_modules": ["q_proj", "k_proj", "v_proj"]}
                }
            )
        )
        assert settings.lora_target_scope == "attention-only"

    def test_packed_expert_targets_select_contracted_dispatch(self):
        settings = training_settings_from_manifest(
            manifest(
                network={
                    "lora_config": {
                        "target_parameters": [
                            "mixer.experts.up_proj",
                            "mixer.experts.down_proj",
                        ]
                    }
                }
            )
        )

        assert settings.lora_packed_target_matrices == 2
        assert settings.packed_moe_dispatch == "contracted"

    def test_a_classic_rl_manifest_is_refused(self):
        with pytest.raises(ValueError, match="LLM fine-tuning"):
            training_settings_from_manifest(DQN)

    def test_eager_attn_backend_is_refused(self):
        with pytest.raises(ValueError, match="flash-style backends only"):
            training_settings_from_manifest(
                manifest(algorithm={"attn_implementation": "eager"})
            )

    @pytest.mark.parametrize(
        "algorithm",
        [
            {"use_liger_loss": True, "importance_sampling_level": "turn"},
            {"use_liger_loss": True, "importance_sampling_level": "trajectory"},
            {"use_liger_loss": True, "loss_type": "gspo"},
        ],
    )
    def test_unbounded_liger_loss_is_refused(self, algorithm):
        with pytest.raises(ValueError, match="chunked token-level"):
            training_settings_from_manifest(manifest(algorithm=algorithm))

    def test_token_level_liger_loss_passes(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "use_liger_loss": True,
                    "importance_sampling_level": "token",
                }
            )
        )
        assert settings.algorithm == "grpo"


class TestGenerationSettings:
    def test_defaults_are_the_resolved_engine_config(self):
        # Read the validated vLLM config, not a second set of defaults.
        settings = generation_settings_from_manifest(GRPO)
        assert settings.gpu_memory_utilization == 0.9
        assert settings.max_num_seqs == 16
        assert settings.max_model_len == 512
        assert settings.concurrent_requests == 8  # batch_size x group_size

    def test_explicit_engine_config_wins(self):
        settings = generation_settings_from_manifest(
            manifest(
                algorithm={
                    "vllm_config": {
                        "gpu_memory_utilization": 0.4,
                        "max_num_seqs": 4,
                        "enforce_eager": True,
                        "dtype": "float16",
                        "sleep_mode": True,
                    }
                }
            )
        )
        assert settings.gpu_memory_utilization == 0.4
        assert settings.max_num_seqs == 4
        assert settings.enforce_eager is True
        assert settings.weight_dtype == "fp16"

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3", "int8"])
    def test_kv_cache_quantization_is_refused(self, kv_cache_dtype):
        doc = manifest(algorithm={"vllm_config": {"kv_cache_dtype": kv_cache_dtype}})
        with pytest.raises(ValueError, match="unquantized KV"):
            generation_settings_from_manifest(doc)

    def test_rollout_without_an_engine_config_is_refused(self):
        doc = manifest(
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )
        with pytest.raises(ValueError, match="no vllm_config"):
            generation_settings_from_manifest(doc)

    def test_engine_serving_another_checkpoint_is_refused(self):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "vllm_model_name_or_path": "other/model",
                    "sleep_mode": True,
                }
            }
        )
        with pytest.raises(ValueError, match="different checkpoint"):
            generation_settings_from_manifest(doc)

    def test_engine_serving_the_trainer_checkpoint_passes(self):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "vllm_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "sleep_mode": True,
                }
            }
        )
        settings = generation_settings_from_manifest(doc)
        assert settings.max_num_seqs == 8

    def test_tensor_parallel_engine_is_refused(self):
        doc = manifest(
            algorithm={"vllm_config": {"tensor_parallel_size": 2, "sleep_mode": True}}
        )
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            generation_settings_from_manifest(doc)

    def test_verbatim_engine_args_warn(self):
        doc = manifest(
            algorithm={"vllm_engine_args": {"enable_chunked_prefill": False}}
        )
        with pytest.warns(UserWarning, match="verbatim"):
            generation_settings_from_manifest(doc)

    def test_strip_true_selects_the_engine_variant(self):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "strip_multimodal_towers": True,
                    "sleep_mode": True,
                }
            },
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )
        config = run_config_from_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert config.generation.weight_variant == "engine"
        assert config.model.variant("engine").stripped_multimodal is True

    def test_partial_tower_list_warns_and_sizes_towers_kept(self):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "strip_multimodal_towers": ["vision_tower"],
                    "sleep_mode": True,
                }
            }
        )
        with pytest.warns(UserWarning, match="strips all towers or none"):
            settings = generation_settings_from_manifest(doc)
        assert settings.weight_variant == "base"

    @pytest.mark.parametrize(
        "quantization",
        ["awq", "gptq", "fp8", "compressed-tensors", "bitsandbytes", "int8"],
    )
    def test_engine_quantization_is_refused(self, quantization):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "quantization": quantization,
                    "sleep_mode": True,
                }
            }
        )
        with pytest.raises(ValueError, match="Engine quantization"):
            generation_settings_from_manifest(doc)

    def test_unrecognized_engine_dtype_is_refused(self):
        doc = manifest(algorithm={"vllm_config": {"dtype": "int4", "sleep_mode": True}})
        with pytest.raises(ValueError, match="Unknown vLLM dtype"):
            generation_settings_from_manifest(doc)


class TestDeviceFromResourceClass:
    def test_known_tier(self):
        device = device_spec_from_resource_class(L4_TIER)
        assert device.name == "NVIDIA L4"
        assert device.total_bytes == 24 * 1024**3

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
    def test_non_async_rollout_is_refused(self):
        with pytest.raises(ValueError, match="async rollout only"):
            run_config_from_manifest(
                GRPO, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
            )

    def test_async_rollout_disaggregates(self):
        doc = manifest(
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.9}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )
        config = run_config_from_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert config.gen_device is not None
        assert config.model.model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_estimate_manifest_produces_both_bars(self):
        doc = manifest(
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.9}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )
        estimate = estimate_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert estimate.training.components
        assert estimate.generation.components

    def test_validated_manifest_object_is_accepted(self):
        validated = TrainingManifest.model_validate(copy.deepcopy(GRPO))
        settings = training_settings_from_manifest(validated)
        assert settings.algorithm == "grpo"


class TestDistributedAndOrchestration:
    def test_fsdp_config_and_gpu_count_come_from_the_manifest(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={"fsdp": {"reshard_after_forward": False}},
                training={"training_gpus_per_agent": 4},
            )
        )
        assert settings.fsdp == FSDPConfig(reshard_after_forward=False)
        assert settings.n_training_gpus == 4

    def test_arena_runs_are_orchestrated(self):
        doc = manifest(
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.9}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )
        config = run_config_from_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )
        assert config.orchestrated

    def test_single_row_micro_batch_passes(self):
        settings = training_settings_from_manifest(GRPO)
        assert settings.trajectories == 8  # batch_size x group_size, one shard

    def test_flat_data_parallel_has_no_fsdp_config(self):
        assert training_settings_from_manifest(GRPO).fsdp is None

    def test_single_row_resolved_from_defaults_passes(self):
        settings = training_settings_from_manifest(
            manifest(algorithm={"batch_size": 1, "micro_batch_size_per_gpu": None})
        )
        assert settings.trajectories == 4

    @pytest.mark.parametrize(
        "doc",
        [
            # Explicit.
            manifest(algorithm={"micro_batch_size_per_gpu": 5}),
            # Unset on one GPU resolves to the per-rank batch.
            manifest(algorithm={"batch_size": 8, "micro_batch_size_per_gpu": None}),
            # mini_batch_size set alone becomes the micro-batch.
            manifest(
                algorithm={
                    "micro_batch_size_per_gpu": None,
                    "mini_batch_size": 3,
                },
                training={"training_gpus_per_agent": 2},
            ),
            # Unset on two GPUs resolves to batch_size // n_training_gpus.
            manifest(
                algorithm={
                    "batch_size": 8,
                    "micro_batch_size_per_gpu": None,
                },
                training={"training_gpus_per_agent": 2},
            ),
        ],
    )
    def test_non_single_row_micro_batch_is_refused(self, doc):
        with pytest.raises(ValueError, match="single-row micro-batches"):
            training_settings_from_manifest(doc)


class TestEnginelessAlgorithms:
    def test_sft_manifest_gets_an_empty_generation_bar(self):
        doc = {
            "algorithm": {"name": "SFT", "micro_batch_size_per_gpu": 1},
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

    def test_sft_training_settings_map(self):
        doc = {
            "algorithm": {"name": "SFT", "micro_batch_size_per_gpu": 1},
            "environment": {"dataset": "openai/gsm8k"},
            "network": {
                "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "max_context_length": 512,
            },
            "training": {"max_steps": 100},
        }
        settings = training_settings_from_manifest(doc)
        assert settings.algorithm == "sft"
