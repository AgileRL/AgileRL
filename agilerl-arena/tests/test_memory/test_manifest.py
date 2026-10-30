# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Manifest to estimator: a written field appears on the estimator settings."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agilerl.arena.memory.estimator import estimate_run, estimate_training
from agilerl.arena.memory.manifest import (
    check_fp8_moe_backend_for_device,
    device_spec_from_resource_class,
    lookup_gpu,
    marlin_pads_non_gated_experts,
    resolve_training_memory,
    training_settings_from_manifest,
    training_settings_from_spec,
    validated_run_manifest,
)
from agilerl.arena.memory.manifest import (
    estimate_manifest as _estimate_manifest,
)
from agilerl.arena.memory.manifest import (
    generation_settings_from_manifest as _generation_settings_from_manifest,
)
from agilerl.arena.memory.manifest import (
    run_config_from_manifest as _run_config_from_manifest,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB, ModelArch, ModelSpec
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

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

ENGINE = {"vllm_config": {"gpu_memory_utilization": 0.85}}

ASYNC_ROLLOUT = {
    "training": {"rollout_mode": "async", "rollout_engines_per_agent": 1},
    "replay_buffer": {"kind": "llm"},
}

DQN = {
    "algorithm": {"name": "DQN"},
    "environment": {"name": "LunarLander-v3", "num_envs": 8},
    "training": {"max_steps": 1000, "pop_size": 2},
}

L4_TIER = {"name": "l4-1x", "gpu_type": "NVIDIA L4", "num_gpus": 1, "ram_gb": 32}
A100_TIER = {
    "name": "a100-80-1x",
    "gpu_type": "NVIDIA A100-SXM4-80GB",
    "num_gpus": 1,
    "ram_gb": 128,
}


def manifest(**overrides: dict) -> dict:
    out = copy.deepcopy(GRPO)
    for section, values in overrides.items():
        out.setdefault(section, {}).update(values)
    return out


def _engine_width(doc: TrainingManifest | dict) -> int:
    """Width these cases size. An omitted manifest width is one GPU."""
    validated = (
        doc
        if isinstance(doc, TrainingManifest)
        else TrainingManifest.model_validate(doc)
    )
    vllm = getattr(validated.algorithm, "vllm_config", None)
    if vllm is not None and vllm.tensor_parallel_size is not None:
        return vllm.tensor_parallel_size
    return 1


def generation_settings_from_manifest(doc, tensor_parallel_size=None):
    width = _engine_width(doc) if tensor_parallel_size is None else tensor_parallel_size
    return _generation_settings_from_manifest(doc, tensor_parallel_size=width)


def run_config_from_manifest(doc, device, model_config, **kwargs):
    if kwargs.get("tensor_parallel_size") is None:
        kwargs["tensor_parallel_size"] = _engine_width(doc)
    return _run_config_from_manifest(doc, device, model_config, **kwargs)


def estimate_manifest(doc, device, model_config, **kwargs):
    if kwargs.get("tensor_parallel_size") is None:
        kwargs["tensor_parallel_size"] = _engine_width(doc)
    return _estimate_manifest(doc, device, model_config, **kwargs)


class TestTrainingSettings:
    def test_settings_come_from_the_manifest(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "beta": 0.01,
                    "micro_batch_size_per_gpu": 1,
                    "use_separate_reference_adapter": False,
                },
                network={
                    "lora_config": {"lora_r": 32, "lora_dropout": 0.05},
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
        assert settings.lora_dropout == 0.05
        assert settings.beta == 0.01
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

    def test_full_cpu_offload_is_refused(self):
        with pytest.raises(ValueError, match=r"fsdp\.cpu_offload needs colocated"):
            training_settings_from_manifest(
                manifest(algorithm={"fsdp": {"cpu_offload": True}})
            )

    def test_optimizer_cpu_offload_is_sized(self):
        settings = training_settings_from_manifest(
            manifest(algorithm={"fsdp": {"optim_cpu_offload": True}})
        )

        assert settings.fsdp is not None
        assert settings.fsdp.optim_cpu_offload is True

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
                    "pretrained_model_name_or_path": "ibm-granite/granite-3.1-3b-a800m-instruct",
                    "lora_config": {
                        "target_parameters": [
                            "block_sparse_moe.experts.gate_up_proj",
                            "block_sparse_moe.experts.down_proj",
                        ]
                    },
                }
            )
        )

        assert settings.lora_packed_target_matrices == 2
        assert settings.packed_moe_dispatch == "contracted"

    def test_repeated_packed_expert_targets_count_once(self):
        settings = training_settings_from_manifest(
            manifest(
                network={
                    "pretrained_model_name_or_path": "ibm-granite/granite-3.1-3b-a800m-instruct",
                    "lora_config": {
                        "target_parameters": [
                            "block_sparse_moe.experts.gate_up_proj",
                            "block_sparse_moe.experts.gate_up_proj",
                            "block_sparse_moe.experts.down_proj",
                        ]
                    },
                }
            )
        )

        assert settings.lora_packed_target_matrices == 2

    def test_packed_expert_targets_reach_the_engine(self):
        doc = manifest(
            network={
                "pretrained_model_name_or_path": "ibm-granite/granite-3.1-3b-a800m-instruct",
                "lora_config": {
                    "target_parameters": [
                        "block_sparse_moe.experts.gate_up_proj",
                        "block_sparse_moe.experts.down_proj",
                    ]
                },
            }
        )

        assert generation_settings_from_manifest(doc).lora_packed_target_matrices == 2

    def test_llmppo_gradient_forward_holds_actor_and_critic_rows(self):
        # Arrange
        doc = manifest()
        doc["algorithm"] = {
            "name": "LLMPPO",
            "batch_size": 2,
            "micro_batch_size_per_gpu": 1,
        }

        # Act
        settings = training_settings_from_manifest(doc)

        # Assert
        assert settings.algorithm == "ppo"
        assert settings.grad_forward_rows == 2

    def test_llmppo_fuse_key_is_unknown(self):
        doc = manifest()
        doc["algorithm"] = {
            "name": "LLMPPO",
            "batch_size": 2,
            "fuse_actor_critic_pass": True,
        }

        with pytest.raises(ValidationError, match="fuse_actor_critic_pass"):
            training_settings_from_manifest(doc)

    def test_top_k_sampling_records_supports_as_wide_as_top_k(self):
        settings = training_settings_from_manifest(
            manifest(algorithm={"top_k": 20, "temperature": 1.0})
        )

        assert settings.sampling_support_width == 20
        assert settings.support_scored_loss is True

    def test_unfiltered_sampling_records_no_supports(self):
        settings = training_settings_from_manifest(manifest(algorithm={"top_k": 0}))

        assert settings.sampling_support_width == 0

    def test_llmppo_liger_loss_scores_the_full_vocabulary(self):
        # Arrange
        doc = manifest()
        doc["algorithm"] = {
            "name": "LLMPPO",
            "batch_size": 2,
            "micro_batch_size_per_gpu": 1,
            "top_k": 20,
        }

        # Act
        settings = training_settings_from_manifest(doc)

        # Assert
        assert settings.sampling_support_width == 20
        assert settings.support_scored_loss is False

    @pytest.mark.parametrize("total_gib", [4, 2], ids=["fits", "tight"])
    def test_run_config_always_fuses_the_actor_critic_pass(
        self, total_gib: float
    ) -> None:
        # Arrange: 4 GiB holds the fused peak; 2 GiB does not.
        doc = manifest()
        algorithm: dict[str, object] = {
            "name": "LLMPPO",
            "batch_size": 2,
            "micro_batch_size_per_gpu": 1,
            "vllm_config": {"tensor_parallel_size": 1},
        }
        doc["algorithm"] = algorithm
        doc["training"]["rollout_mode"] = "async"
        doc["training"]["rollout_engines_per_agent"] = 1
        doc["replay_buffer"] = {"kind": "llm"}
        device = DeviceSpec(total_bytes=int(total_gib * GiB), name="NVIDIA L4")
        model_config = SUPPORTED_MODEL_INFO["Qwen/Qwen2.5-0.5B-Instruct"].config

        # Act
        config = run_config_from_manifest(doc, device, model_config)

        # Assert
        assert config.training.grad_forward_rows == 2
        peak = estimate_run(config).training
        at_settings = estimate_training(
            config.model,
            device,
            config.training,
            orchestrated=True,
        )
        assert peak.total_bytes == at_settings.total_bytes

    def test_llmppo_gradient_rows_do_not_depend_on_the_device(self) -> None:
        # Arrange
        doc = manifest()
        doc["algorithm"] = {
            "name": "LLMPPO",
            "batch_size": 2,
            "micro_batch_size_per_gpu": 1,
            "vllm_config": {"tensor_parallel_size": 1},
        }
        doc["training"]["rollout_mode"] = "async"
        doc["training"]["rollout_engines_per_agent"] = 1
        doc["replay_buffer"] = {"kind": "llm"}
        model_config = SUPPORTED_MODEL_INFO["Qwen/Qwen2.5-0.5B-Instruct"].config
        devices = (
            DeviceSpec(total_bytes=int(24 * GiB), name="NVIDIA L4"),
            DeviceSpec(total_bytes=int(1 * GiB), name="NVIDIA L4"),
        )

        # Act
        rows = []
        for device in devices:
            config = run_config_from_manifest(doc, device, model_config)

            # Assert
            assert config.training.grad_forward_rows == 2
            rows.append(config.training.grad_forward_rows)

        assert rows == [2, 2]

    def test_host_memory_settings_come_from_the_training_section(self):
        default = training_settings_from_manifest(GRPO)
        overridden = training_settings_from_manifest(
            manifest(training={"checkpoint_optimizer": False})
        )

        assert default.checkpoint_optimizer is True
        assert overridden.checkpoint_optimizer is False

    def test_rollout_runs_size_async_with_rollout_mode_unset(self):
        settings = training_settings_from_manifest(GRPO)

        assert settings.async_rollout is True

    def test_a_classic_rl_manifest_is_refused(self):
        with pytest.raises(ValueError, match="LLM fine-tuning"):
            training_settings_from_manifest(DQN)

    def test_rollout_algorithm_packs_on_a_dense_model(self):
        settings = training_settings_from_manifest(manifest())

        assert settings.sequence_packing is True

    def test_llmppo_trains_padded_under_flash_attention_2(self):
        settings = training_settings_from_manifest(llmppo())

        assert settings.sequence_packing is False

    def test_hybrid_model_without_boundary_reset_trains_padded(self):
        settings = training_settings_from_manifest(
            manifest(
                network={
                    "pretrained_model_name_or_path": "ibm-granite/granite-4.0-h-tiny"
                }
            )
        )

        assert settings.sequence_packing is False

    def test_hybrid_model_with_boundary_reset_packs(self):
        settings = training_settings_from_manifest(
            manifest(network={"pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B"})
        )

        assert settings.sequence_packing is True

    def test_model_outside_the_catalog_packs_as_dense(self):
        settings = training_settings_from_manifest(
            manifest(network={"pretrained_model_name_or_path": "org/unlisted"})
        )

        assert settings.sequence_packing is True


def llmppo(**algorithm: object) -> dict:
    """An LLMPPO manifest on the single-turn GRPO base."""
    doc = manifest()
    doc["algorithm"] = {
        "name": "LLMPPO",
        "batch_size": 2,
        "micro_batch_size_per_gpu": 1,
        **algorithm,
    }
    return doc


class TestTrainingSettingsLigerLoss:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("vllm_importance_sampling_correction", False),
            ("old_logprobs_source", "trainer"),
        ],
    )
    def test_sampling_logprob_options_are_rejected(self, key: str, value: object):
        with pytest.raises(
            ValidationError, match=f"{key}\n  Extra inputs are not permitted"
        ):
            TrainingManifest.model_validate(manifest(algorithm={key: value}))

    def test_token_level_prices_token_tiles(self):
        settings = training_settings_from_manifest(
            llmppo(importance_sampling_level="token")
        )

        assert settings.whole_sequence_loss is False
        assert settings.liger_weight_grad is True

    @pytest.mark.parametrize("name", ["LLMPPO", "LLMREINFORCE"])
    @pytest.mark.parametrize("level", ["turn", "trajectory"])
    def test_above_token_level_runs_the_standard_path(self, name: str, level: str):
        # Arrange: sampling log-probs weight each token, which pooled ratios cannot.
        doc = llmppo(importance_sampling_level=level, top_k=20)
        doc["algorithm"]["name"] = name

        # Act
        settings = training_settings_from_manifest(doc)

        # Assert
        assert settings.whole_sequence_loss is False
        assert settings.liger_weight_grad is False
        assert settings.support_scored_loss is True

    @pytest.mark.parametrize(
        ("multi_turn", "head_grad"), [(True, False), (False, True)]
    )
    def test_auto_level_pools_turns_only_on_multi_turn_episodes(
        self, multi_turn: bool, head_grad: bool
    ):
        # Arrange
        spec = TrainingManifest.model_validate(llmppo()).algorithm

        # Act
        settings = training_settings_from_spec(spec, 1, 1, multi_turn=multi_turn)

        # Assert
        assert settings.whole_sequence_loss is False
        assert settings.liger_weight_grad is head_grad

    @pytest.mark.parametrize(
        ("algorithm", "head_grad"),
        [
            ({"importance_sampling_level": "trajectory"}, False),
            ({"loss_type": "gspo"}, False),
            ({"importance_sampling_level": "turn"}, False),
            ({"importance_sampling_level": "token"}, True),
        ],
        ids=["trajectory", "gspo", "turn", "token"],
    )
    def test_grpo_liger_runs_only_at_token_level(
        self, algorithm: dict, head_grad: bool
    ):
        # Arrange: unmasked, so only the sampling log-probs weight tokens.
        doc = manifest(
            algorithm={
                "top_k": 0,
                "off_policy_token_mask_bounds": None,
                "off_policy_sequence_mask_threshold": None,
                **algorithm,
            }
        )

        # Act
        settings = training_settings_from_manifest(doc)

        # Assert
        assert settings.whole_sequence_loss is False
        assert settings.liger_weight_grad is head_grad


class TestLigerSequenceLoss:
    def test_default_gspo_manifest_returns_a_phase_breakdown(self):
        # Arrange
        doc = manifest(algorithm={"name": "GSPO", **ENGINE}, **ASYNC_ROLLOUT)

        # Act
        settings = training_settings_from_manifest(doc)
        estimate = estimate_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )

        # Assert: sampling log-probs keep GSPO on the standard path, which
        # holds no sequence tile and no head grad.
        assert settings.whole_sequence_loss is False
        assert settings.liger_weight_grad is False
        assert estimate.training.total_bytes > 0


def screenshots(**environment: object) -> dict:
    """GRPO on a remote env whose observations carry page screenshots."""
    out = manifest(
        algorithm={"vllm_config": {"limit_mm_per_prompt": {"image": 3}}},
        network={"pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B"},
        training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
        replay_buffer={"kind": "llm"},
    )
    out["environment"] = {
        "env_type": "rollout",
        "env_image": "browsergym-vwa:latest",
        "env_port": 8000,
        "max_turns": 10,
        "screenshot_field": "screenshot",
        **environment,
    }
    return out


class TestImageInputs:
    def test_screenshots_render_at_the_default_viewport(self):
        settings = training_settings_from_manifest(screenshots())

        assert (settings.image_width, settings.image_height) == (1280, 720)
        assert settings.images_per_row == 3

    def test_segment_max_images_bounds_the_row(self):
        settings = training_settings_from_manifest(
            screenshots(segment_max_images=2, segment_prompt_tokens=400)
        )

        assert settings.images_per_row == 2

    def test_segment_limits_train_segment_rows(self):
        segmented = training_settings_from_manifest(
            screenshots(segment_max_images=2, segment_prompt_tokens=400)
        )
        whole = training_settings_from_manifest(screenshots())

        assert segmented.segmented_rows
        assert not whole.segmented_rows

    def test_viewport_env_vars_set_the_image_size(self):
        env_vars = {
            "BROWSERGYM_VIEWPORT_WIDTH": "1280",
            "BROWSERGYM_VIEWPORT_HEIGHT": "2048",
        }

        settings = training_settings_from_manifest(screenshots(env_vars=env_vars))

        assert (settings.image_width, settings.image_height) == (1280, 2048)

    def test_goal_images_alone_count_as_images(self):
        settings = training_settings_from_manifest(
            screenshots(screenshot_field="", goal_images_field="goal_images")
        )

        assert (settings.images_per_row, settings.image_width) == (3, 1280)

    def test_tower_target_modules_train_the_vision_tower(self):
        trained = screenshots()
        trained["network"]["lora_config"] = {
            "target_modules": ["q_proj"],
            "tower_target_modules": ["qkv"],
        }

        settings = training_settings_from_manifest(trained)

        assert settings.vision_tower_lora
        assert not training_settings_from_manifest(screenshots()).vision_tower_lora

    def test_text_environment_has_no_images(self):
        settings = training_settings_from_manifest(GRPO)

        assert settings.images_per_update == 0
        assert (settings.image_width, settings.image_height) == (0, 0)

    def test_qwen_vwa_estimate_holds_the_prefetched_pixels(self):
        # Arrange: 2 prompts x 4 rows over 1 GPU, 3 screenshots of 44 x 80
        # Qwen patches of 1536 fp32 values each.
        config = SUPPORTED_MODEL_INFO["Qwen/Qwen3.5-0.8B"].config
        device = DeviceSpec(total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB")
        text = screenshots(screenshot_field="")

        # Act
        with_images = estimate_manifest(screenshots(), device, config).training
        without = estimate_manifest(text, device, config).training

        # Assert
        held = [
            next(c.n_bytes for c in breakdown.host if c.key == "prefetched_rollout")
            for breakdown in (with_images, without)
        ]
        assert held[0] - held[1] == 8 * 3 * 44 * 80 * 1536 * 4


def segmented(segment_prompt_tokens: int | None, **algorithm: object) -> dict:
    """GRPO on a multi-turn env that restarts context at *segment_prompt_tokens*."""
    out = manifest(algorithm={"max_output_tokens": 128, **algorithm})
    out["environment"] = {
        "env_type": "rollout",
        "env_url": "http://env",
        "max_turns": 10,
        "segment_prompt_tokens": segment_prompt_tokens,
    }
    return out


class TestMaxRowTokens:
    def test_segment_rows_size_the_training_estimate(self):
        settings = training_settings_from_manifest(segmented(300))

        assert settings.max_model_len == 300 + 128

    def test_answer_continuation_adds_a_second_generation(self):
        settings = training_settings_from_manifest(
            segmented(
                200,
                answer_continuation=True,
                constrain_answer_pattern=r"\d+",
            )
        )

        assert settings.max_model_len == 200 + 2 * 128

    def test_rows_never_exceed_the_context(self):
        settings = training_settings_from_manifest(segmented(500))

        assert settings.max_model_len == 512

    def test_matching_explicit_value_is_kept(self):
        validated = TrainingManifest.model_validate(segmented(300, max_row_tokens=428))

        assert validated.algorithm.max_row_tokens == 428

    def test_disagreeing_explicit_value_is_refused(self):
        with pytest.raises(ValueError, match=r"max_row_tokens=400 disagrees with 428"):
            TrainingManifest.model_validate(segmented(300, max_row_tokens=400))

    def test_unsegmented_env_sizes_rows_to_the_context(self):
        validated = TrainingManifest.model_validate(segmented(None))

        assert validated.algorithm.max_row_tokens is None
        assert training_settings_from_manifest(segmented(None)).max_model_len == 512


class TestGenerationSettings:
    def test_family_catalog_sets_the_batched_tokens(self):
        settings = generation_settings_from_manifest(
            manifest(
                network={
                    "pretrained_model_name_or_path": (
                        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
                    )
                }
            )
        )

        assert settings.max_num_batched_tokens == 8192

    def test_manifest_batched_tokens_beat_the_family_default(self):
        settings = generation_settings_from_manifest(
            manifest(
                algorithm={"vllm_config": {"max_num_batched_tokens": 4096}},
                network={
                    "pretrained_model_name_or_path": (
                        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
                    )
                },
            )
        )

        assert settings.max_num_batched_tokens == 4096

    def test_family_without_a_default_leaves_batched_tokens_to_vllm(self):
        settings = generation_settings_from_manifest(GRPO)

        assert settings.max_num_batched_tokens is None

    def test_defaults_are_the_resolved_engine_config(self):
        # Defaults come from the validated vLLM config.
        settings = generation_settings_from_manifest(GRPO)
        assert settings.gpu_memory_utilization == 0.9
        assert settings.max_num_seqs == 16
        assert settings.max_model_len == 512
        # One prompt on one engine x group_size.
        assert settings.concurrent_requests == 4

    def test_explicit_engine_config_wins(self):
        settings = generation_settings_from_manifest(
            manifest(
                algorithm={
                    "vllm_config": {
                        "gpu_memory_utilization": 0.4,
                        "max_num_seqs": 4,
                        "dtype": "float16",
                    }
                }
            )
        )
        assert settings.gpu_memory_utilization == 0.4
        assert settings.max_num_seqs == 4
        assert settings.weight_dtype == "fp16"

    def test_fp32_engine_dtype(self):
        settings = generation_settings_from_manifest(
            manifest(algorithm={"vllm_config": {"dtype": "float32"}})
        )

        assert settings.weight_dtype == "fp32"

    @pytest.mark.parametrize(
        ("engines", "rollout_batch_size", "expected"),
        [(2, 6, 12), ("auto", 6, 24), (2, 7, 16)],
    )
    def test_async_engine_without_max_num_seqs_runs_one_per_episode(
        self, engines, rollout_batch_size, expected
    ):
        # Arrange: group size 4. "auto" counts as one engine.
        doc = manifest(
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.85}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": engines,
                "rollout_batch_size": rollout_batch_size,
            },
            replay_buffer={"kind": "llm"},
        )

        # Act
        settings = generation_settings_from_manifest(doc)

        # Assert
        assert settings.max_num_seqs == expected

    def test_async_engine_without_max_num_seqs_sizes_the_mutable_group_ceiling(
        self,
    ):
        # Arrange: group size 4 can mutate up to 12; 2 engines share 6 prompts.
        doc = manifest(
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.85}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 2,
                "rollout_batch_size": 6,
            },
            replay_buffer={"kind": "llm"},
            mutation={"rl_hp_selection": {"group_size": {"min": 2, "max": 12}}},
        )

        # Act
        settings = generation_settings_from_manifest(doc)

        # Assert
        assert settings.max_num_seqs == 3 * 12

    @pytest.mark.parametrize(
        ("engines", "rollout_batch_size", "expected"),
        [(2, 6, 12), ("auto", 6, 24), (2, 7, 16)],
    )
    def test_concurrent_requests_are_one_engine_share_of_the_job_batch(
        self, engines, rollout_batch_size, expected
    ):
        # Arrange: group size 4, max_num_seqs set. "auto" counts as one engine.
        doc = manifest(
            algorithm={"vllm_config": {"max_num_seqs": 64}},
            training={
                "rollout_engines_per_agent": engines,
                "rollout_batch_size": rollout_batch_size,
            },
        )

        # Act
        settings = generation_settings_from_manifest(doc)

        # Assert
        assert settings.concurrent_requests == expected

    def test_concurrent_requests_size_the_mutable_group_ceiling(self):
        # Arrange: one prompt per engine; group size 4 can mutate up to 12.
        doc = manifest(
            mutation={"rl_hp_selection": {"group_size": {"min": 2, "max": 12}}},
        )

        # Act
        settings = generation_settings_from_manifest(doc)

        # Assert
        assert settings.concurrent_requests == 12

    def test_engine_image_limit_reaches_the_settings(self):
        doc = manifest(algorithm={"vllm_config": {"limit_mm_per_prompt": {"image": 4}}})

        settings = generation_settings_from_manifest(doc)

        assert settings.max_images_per_prompt == 4

    @pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3", "int8"])
    def test_kv_cache_quantization_is_refused(self, kv_cache_dtype):
        doc = manifest(algorithm={"vllm_config": {"kv_cache_dtype": kv_cache_dtype}})
        with pytest.raises(ValueError, match="unquantized KV"):
            generation_settings_from_manifest(doc)

    def test_engine_serving_another_checkpoint_is_refused(self):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "vllm_model_name_or_path": "other/model",
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
                }
            }
        )
        settings = generation_settings_from_manifest(doc)
        # One async prompt of group 4.
        assert settings.max_num_seqs == 4

    def test_tensor_parallel_engine_is_sized(self):
        doc = manifest(algorithm={"vllm_config": {"tensor_parallel_size": 4}})

        settings = generation_settings_from_manifest(doc)

        assert settings.tensor_parallel_size == 4

    def test_verbatim_engine_args_warn(self):
        doc = manifest(
            algorithm={"vllm_engine_args": {"enable_chunked_prefill": False}}
        )
        with pytest.warns(UserWarning, match="verbatim"):
            generation_settings_from_manifest(doc)

    def test_text_env_on_a_vl_model_sizes_the_engine_variant(self):
        doc = manifest(network={"pretrained_model_name_or_path": "Qwen/Qwen3.5-0.8B"})

        settings = generation_settings_from_manifest(doc)

        assert settings.weight_variant == "engine"

    def test_async_text_env_without_vllm_config_sizes_the_tower_free_engine(self):
        # Arrange
        hub_id = "Qwen/Qwen3.6-35B-A3B"
        doc = manifest(
            environment={
                "env_type": "rollout",
                "env_url": "http://env",
                "max_turns": 20,
                "modality": "text",
            },
            network={"pretrained_model_name_or_path": hub_id},
            training={"rollout_mode": "async", "rollout_engines_per_agent": 1},
            replay_buffer={"kind": "llm"},
        )
        for key in ("dataset", "reward_file_path", "prompt_template"):
            doc["environment"].pop(key)
        device = device_spec_from_resource_class(A100_TIER)
        info = SUPPORTED_MODEL_INFO[hub_id]

        # Act
        stripped = estimate_manifest(doc, device, info.config, n_params=info.num_params)
        doc["environment"].pop("modality")
        kept = estimate_manifest(doc, device, info.config, n_params=info.num_params)

        # Assert
        weights = {
            name: next(
                c.n_bytes for c in estimate.generation.components if c.key == "weights"
            )
            for name, estimate in (("stripped", stripped), ("kept", kept))
        }
        assert weights["stripped"] < weights["kept"]

    def test_kept_towers_still_declare_the_engine_variant(self):
        doc = manifest(
            algorithm={"vllm_config": {}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )

        config = run_config_from_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )

        assert config.generation.weight_variant == "base"
        assert config.model.variant("engine").stripped_multimodal is True

    @pytest.mark.parametrize(
        "quantization",
        ["awq", "gptq", "compressed-tensors", "bitsandbytes", "int8"],
    )
    def test_engine_quantization_is_refused(self, quantization):
        doc = manifest(
            algorithm={
                "vllm_config": {
                    "quantization": quantization,
                }
            }
        )
        with pytest.raises(ValueError, match="Engine quantization"):
            generation_settings_from_manifest(doc)

    def test_fp8_engine_weights_are_sized(self):
        doc = manifest(algorithm={"vllm_config": {"quantization": "fp8"}})

        settings = generation_settings_from_manifest(doc)

        assert settings.weight_quantization == "fp8"

    def test_unrecognized_engine_dtype_is_refused(self):
        doc = manifest(algorithm={"vllm_config": {"dtype": "int4"}})
        with pytest.raises(ValueError, match="Unknown vLLM dtype"):
            generation_settings_from_manifest(doc)


class TestDeviceFromResourceClass:
    def test_known_tier(self):
        device = device_spec_from_resource_class(L4_TIER)
        assert device.name == "NVIDIA L4"
        assert device.total_bytes == int(22.03 * 1024**3)
        assert device.compute_capability == (8, 9)

    @pytest.mark.parametrize(
        ("gpu_type", "cuda_gib"),
        [("NVIDIA H100 80GB HBM3", 79.11), ("a100 80gb", 79.25), ("NVIDIA L4", 22.03)],
    )
    def test_measured_cards_carry_their_cuda_visible_total(self, gpu_type, cuda_gib):
        assert lookup_gpu(gpu_type).total_gib == cuda_gib

    def test_gpu_type_spellings(self):
        assert lookup_gpu("a100 80gb").total_gib == 79.25
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

    def test_an_explicit_size_overrides_a_known_gpu(self):
        device = device_spec_from_resource_class(L4_TIER, gpu_memory_gib=16)

        assert device.name == "NVIDIA L4"
        assert device.total_bytes == 16 * 1024**3

    def test_cpu_tier_is_refused(self):
        with pytest.raises(ValueError, match="no GPU"):
            device_spec_from_resource_class({"name": "cpu-only", "gpu_type": None})


class TestLlmSpec:
    def test_an_unmapped_llm_algorithm_is_refused(self, monkeypatch):
        monkeypatch.setattr("agilerl.arena.memory.manifest.ALGORITHM_NAMES", {})

        with pytest.raises(ValueError, match="No memory model"):
            training_settings_from_manifest(GRPO)

    def test_a_manifest_without_a_model_is_refused(self):
        parsed = TrainingManifest.model_validate(
            manifest(
                algorithm={"vllm_config": {"gpu_memory_utilization": 0.9}},
                training={
                    "rollout_mode": "async",
                    "rollout_engines_per_agent": 1,
                    "rollout_batch_size": 4,
                },
                replay_buffer={"kind": "llm"},
            )
        )
        algo = parsed.algorithm.model_copy(
            update={"pretrained_model_name_or_path": None}
        )
        broken = parsed.model_copy(update={"algorithm": algo})

        with pytest.raises(ValueError, match="no pretrained model"):
            run_config_from_manifest(
                broken, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
            )


class TestValidatedRunManifest:
    def test_unset_rollout_mode_runs_async_on_auto_engines(self):
        doc = TrainingManifest.model_validate(manifest(replay_buffer={"kind": "llm"}))

        run = validated_run_manifest(doc)

        assert run.training.rollout_mode == "async"
        assert run.training.rollout_engines_per_agent == "auto"

    def test_non_generating_run_is_returned_as_written(self):
        doc = TrainingManifest.model_validate(DQN)

        run = validated_run_manifest(doc)

        assert run is doc

    def test_full_cpu_offload_with_unset_rollout_mode_is_refused(self):
        doc = TrainingManifest.model_validate(
            manifest(
                algorithm={"fsdp": {"cpu_offload": True}},
                replay_buffer={"kind": "llm"},
            )
        )

        with pytest.raises(ValidationError, match=r"fsdp\.cpu_offload needs colocated"):
            validated_run_manifest(doc)


class TestRunConfig:
    def test_unset_rollout_mode_is_sized_async(self):
        config = run_config_from_manifest(
            GRPO, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )

        assert config.training.async_rollout is True

    def test_explicit_colocated_rollout_is_refused(self):
        doc = manifest(training={"rollout_mode": "colocated"})

        with pytest.raises(ValueError, match="async rollout only"):
            run_config_from_manifest(
                doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
            )

    def test_liger_supported_model_type_sizes_the_liger_layers(self):
        config = run_config_from_manifest(
            GRPO, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )

        assert config.model.arch.fp32_norm_inputs is False
        assert config.model.arch.stock_gated_mlp is False

    def test_liger_unsupported_model_type_sizes_the_stock_norm_and_mlp(self):
        config = run_config_from_manifest(
            GRPO,
            device_spec_from_resource_class(L4_TIER),
            {**TINY_CONFIG, "model_type": "granitemoehybrid"},
        )

        assert config.model.arch.fp32_norm_inputs is True
        assert config.model.arch.stock_gated_mlp is True

    def test_async_lora_rank_snaps_to_a_vllm_accepted_value(self):
        doc = manifest(
            algorithm={
                "lora_config": {"lora_r": 20},
                "vllm_config": {"tensor_parallel_size": 1},
            },
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
                "rollout_batch_size": 1,
            },
            replay_buffer={"kind": "llm"},
        )

        settings = _generation_settings_from_manifest(doc)

        assert settings.max_lora_rank == 32

    def test_unknown_model_prices_the_rank_batch(self) -> None:
        # Arrange: 8 prompts, group 4, on 2 trainer GPUs. The id is outside the catalog.
        doc = manifest(
            algorithm={
                "batch_size": 8,
                "micro_batch_size_per_gpu": None,
                "vllm_config": {"tensor_parallel_size": 1},
            },
            network={"pretrained_model_name_or_path": "missing/not-a-model"},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
                "rollout_batch_size": 2,
                "training_gpus_per_agent": 2,
            },
            replay_buffer={"kind": "llm"},
        )
        device = device_spec_from_resource_class(L4_TIER)

        # Act
        config = run_config_from_manifest(doc, device, TINY_CONFIG)

        # Assert
        assert config.training.micro_batch_size == 16

    def test_omitted_micro_batch_is_searched_on_the_catalog_model(self) -> None:
        # Arrange: 8 prompts, group 4, at 8192 tokens on one L4. The run sizes
        # the forward on the catalog checkpoint whatever config is passed.
        doc = manifest(
            algorithm={
                "batch_size": 8,
                "micro_batch_size_per_gpu": None,
                "vllm_config": {"tensor_parallel_size": 1},
            },
            network={"max_context_length": 8192},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
                "rollout_batch_size": 2,
            },
            replay_buffer={"kind": "llm"},
        )
        device = device_spec_from_resource_class(L4_TIER)
        catalog = SUPPORTED_MODEL_INFO["Qwen/Qwen2.5-0.5B-Instruct"].config

        # Act
        tiny = run_config_from_manifest(doc, device, TINY_CONFIG)
        listed = run_config_from_manifest(doc, device, catalog)

        # Assert
        assert tiny.training.micro_batch_size == 8
        assert listed.training.micro_batch_size == 8

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


class TestCheckFp8MoeBackendForDevice:
    @pytest.mark.parametrize(
        "quantization",
        ["fp8", "fp8_per_block", "fp8_per_tensor", "fp8_per_channel"],
    )
    def test_fp8_family_without_marlin_fails_below_sm89(
        self, quantization: str
    ) -> None:
        with pytest.raises(ValueError, match="moe_backend='marlin'"):
            check_fp8_moe_backend_for_device(quantization, None, True, 8, 0)

    def test_marlin_passes_below_sm89(self) -> None:
        check_fp8_moe_backend_for_device("fp8_per_block", "marlin", True, 8, 0)

    @pytest.mark.parametrize("cc", [(9, 0), (8, 9)])
    def test_native_fp8_hardware_skips(self, cc: tuple[int, int]) -> None:
        check_fp8_moe_backend_for_device("fp8", None, True, *cc)

    def test_dense_models_skip(self) -> None:
        check_fp8_moe_backend_for_device("fp8", None, False, 8, 0)

    @pytest.mark.parametrize("quantization", [None, "awq", "mxfp8"])
    def test_non_fp8_quantization_skips(self, quantization: str | None) -> None:
        check_fp8_moe_backend_for_device(quantization, None, True, 8, 0)


LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


class TestMarlinPadsNonGatedExperts:
    def _lightning(self) -> ModelArch:
        return ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[LIGHTNING].config)

    @pytest.mark.parametrize("width", [2, 4, 8])
    def test_misaligned_relu2_expert_shard_pads(self, width: int) -> None:
        assert marlin_pads_non_gated_experts(
            "fp8", "marlin", self._lightning(), width, False
        )

    def test_aligned_expert_width_does_not_pad(self) -> None:
        # Lightning's 1856-wide experts are a multiple of 64 on one GPU.
        assert not marlin_pads_non_gated_experts(
            "fp8", "marlin", self._lightning(), 1, False
        )

    def test_engine_expert_parallel_keeps_experts_whole(self) -> None:
        assert not marlin_pads_non_gated_experts(
            "fp8", "marlin", self._lightning(), 2, True
        )

    def test_gated_experts_do_not_pad(self) -> None:
        arch = ModelArch.from_hf_config(
            SUPPORTED_MODEL_INFO["Qwen/Qwen3.6-35B-A3B"].config
        )

        assert not marlin_pads_non_gated_experts("fp8", "marlin", arch, 8, False)

    @pytest.mark.parametrize(
        ("quantization", "moe_backend"),
        [(None, "marlin"), ("awq", "marlin"), ("fp8", "triton"), ("fp8", None)],
    )
    def test_other_quantization_or_backend_does_not_pad(
        self, quantization: str | None, moe_backend: str | None
    ) -> None:
        assert not marlin_pads_non_gated_experts(
            quantization, moe_backend, self._lightning(), 2, False
        )


class TestRunConfigFp8MoeBackend:
    def _doc(self, quantization: str, moe_backend: str | None = None) -> dict:
        algorithm: dict[str, object] = {"vllm_config": {"quantization": quantization}}
        if moe_backend is not None:
            algorithm["vllm_engine_args"] = {"moe_backend": moe_backend}
        return manifest(
            algorithm=algorithm,
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )

    def _moe_config(self) -> dict:
        return dict(TINY_CONFIG, num_experts=8, num_experts_per_tok=2)

    def test_fp8_without_marlin_fails_on_a100(self) -> None:
        with pytest.raises(ValueError, match="moe_backend='marlin'"):
            run_config_from_manifest(
                self._doc("fp8"),
                device_spec_from_resource_class(A100_TIER),
                self._moe_config(),
            )

    def test_fp8_with_marlin_sizes_on_a100(self) -> None:
        config = run_config_from_manifest(
            self._doc("fp8", "marlin"),
            device_spec_from_resource_class(A100_TIER),
            self._moe_config(),
        )

        assert config.generation.weight_quantization == "fp8"

    def test_misaligned_non_gated_marlin_sizes_bf16(self) -> None:
        # 96-wide relu2 experts are not a multiple of Marlin's 64-row tile.
        config = dict(self._moe_config(), hidden_act="relu2", moe_intermediate_size=96)

        run = run_config_from_manifest(
            self._doc("fp8", "marlin"),
            device_spec_from_resource_class(A100_TIER),
            config,
        )

        assert run.generation.weight_quantization == "none"

    def test_fp8_per_block_without_marlin_reports_marlin_first(self) -> None:
        with pytest.raises(ValueError, match="moe_backend='marlin'"):
            run_config_from_manifest(
                self._doc("fp8_per_block"),
                device_spec_from_resource_class(A100_TIER),
                self._moe_config(),
            )


class TestDistributedAndOrchestration:
    def test_fsdp_config_and_gpu_count_come_from_the_manifest(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "fsdp": {"reshard_after_forward": False},
                    "batch_size": 4,
                },
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
        assert settings.micro_batch_size == 1

    def test_flat_data_parallel_has_no_fsdp_config(self):
        assert training_settings_from_manifest(GRPO).fsdp is None

    def test_single_row_resolved_from_defaults_passes(self):
        settings = training_settings_from_manifest(
            manifest(algorithm={"batch_size": 1, "micro_batch_size_per_gpu": None})
        )
        assert settings.trajectories == 4
        assert settings.micro_batch_size == 4


class TestMicroBatchRows:
    def test_unset_micro_batch_on_one_gpu_is_the_rank_trajectory_count(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "batch_size": 16,
                    "group_size": 8,
                    "micro_batch_size_per_gpu": None,
                }
            )
        )

        assert settings.micro_batch_size == 128

    def test_unset_micro_batch_folds_tensor_parallel(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "batch_size": 16,
                    "group_size": 8,
                    "micro_batch_size_per_gpu": None,
                    "fsdp": {"tp": 2},
                },
                training={"training_gpus_per_agent": 4},
            )
        )

        assert settings.micro_batch_size == 64

    def test_explicit_micro_batch_is_kept(self):
        settings = training_settings_from_manifest(
            manifest(
                algorithm={
                    "batch_size": 8,
                    "group_size": 4,
                    "micro_batch_size_per_gpu": 4,
                }
            )
        )

        assert settings.micro_batch_size == 4

    def test_micro_batch_larger_than_the_rank_fails_divisibility(self):
        with pytest.raises(ValueError, match="divisible by micro_batch_size_per_gpu"):
            training_settings_from_manifest(
                manifest(
                    algorithm={
                        "batch_size": 2,
                        "group_size": 4,
                        "micro_batch_size_per_gpu": 32,
                    }
                )
            )

    def test_batch_must_divide_by_the_data_parallel_size(self):
        with pytest.raises(ValueError, match="divisible by the data-parallel"):
            training_settings_from_manifest(
                manifest(
                    algorithm={"batch_size": 3, "micro_batch_size_per_gpu": None},
                    training={"training_gpus_per_agent": 2},
                )
            )

    def test_mini_batch_must_divide_the_rank_batch(self):
        with pytest.raises(ValueError, match="divisible by mini_batch_size"):
            training_settings_from_manifest(
                manifest(
                    algorithm={
                        "batch_size": 2,
                        "group_size": 4,
                        "micro_batch_size_per_gpu": None,
                        "mini_batch_size": 3,
                    },
                    training={"training_gpus_per_agent": 2},
                )
            )

    def test_mini_batch_must_divide_by_the_micro_batch(self):
        with pytest.raises(ValueError, match="divisible by micro_batch_size_per_gpu"):
            training_settings_from_manifest(
                {
                    "algorithm": {
                        "name": "SFT",
                        "batch_size": 8,
                        "micro_batch_size_per_gpu": 3,
                    },
                    "environment": {"dataset": "openai/gsm8k"},
                    "network": {
                        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                        "max_context_length": 512,
                    },
                    "training": {"max_steps": 100},
                }
            )

    def test_tensor_parallel_must_divide_the_trainer_world(self):
        with pytest.raises(ValueError, match="divisible by tp"):
            training_settings_from_manifest(
                manifest(
                    algorithm={"fsdp": {"tp": 3}},
                    training={"training_gpus_per_agent": 2},
                )
            )

    def test_sft_unset_micro_batch_is_the_rank_batch(self):
        settings = training_settings_from_manifest(
            {
                "algorithm": {
                    "name": "SFT",
                    "batch_size": 8,
                    "micro_batch_size_per_gpu": None,
                },
                "environment": {"dataset": "openai/gsm8k"},
                "network": {
                    "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "max_context_length": 512,
                },
                "training": {"max_steps": 100},
            }
        )

        assert settings.micro_batch_size == 8

    def test_dpo_micro_batch_keeps_two_graphs(self):
        settings = training_settings_from_manifest(
            {
                "algorithm": {
                    "name": "DPO",
                    "batch_size": 8,
                    "micro_batch_size_per_gpu": 4,
                },
                "environment": {"dataset": "openai/gsm8k"},
                "network": {
                    "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "max_context_length": 512,
                },
                "training": {"max_steps": 100},
            }
        )

        assert settings.algorithm == "dpo"
        assert settings.micro_batch_size == 4
        assert settings.grad_graph_rows == 2

    def test_dpo_liger_loss_holds_whole_sequences(self):
        settings = training_settings_from_manifest(
            {
                "algorithm": {"name": "DPO", "batch_size": 8},
                "environment": {"dataset": "openai/gsm8k"},
                "network": {
                    "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
                    "max_context_length": 512,
                },
                "training": {"max_steps": 100},
            }
        )

        assert settings.whole_sequence_loss is True
        assert settings.liger_weight_grad is False

    def test_sequence_packing_is_sized_as_padded_rows(self):
        doc = manifest(
            algorithm={"vllm_config": {"gpu_memory_utilization": 0.3}},
            training={
                "rollout_mode": "async",
                "rollout_engines_per_agent": 1,
            },
            replay_buffer={"kind": "llm"},
        )

        settings = training_settings_from_manifest(doc)
        estimate = estimate_manifest(
            doc, device_spec_from_resource_class(L4_TIER), TINY_CONFIG
        )

        assert settings.sequence_packing is True
        assert settings.micro_batch_size == 1
        assert any("padded" in warning for warning in estimate.training.warnings)


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


class TestResolveTrainingMemory:
    def model(self) -> ModelSpec:
        config = SUPPORTED_MODEL_INFO["Qwen/Qwen2.5-0.5B-Instruct"].config
        return ModelSpec(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            arch=ModelArch.from_hf_config(config),
        )

    def test_roomy_device_resolves_micro_batch_and_turns_checkpointing_off(self):
        # Arrange
        doc = manifest()
        del doc["algorithm"]["micro_batch_size_per_gpu"]
        validated = TrainingManifest.model_validate(doc)
        device = DeviceSpec(total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB")

        # Act
        resolved = resolve_training_memory(validated, device, self.model())

        # Assert
        assert resolved.algorithm.micro_batch_size_per_gpu >= 1
        assert resolved.algorithm.gradient_checkpointing is False
        assert validated.algorithm.micro_batch_size_per_gpu is None
        assert validated.algorithm.gradient_checkpointing is None

    def test_set_values_are_kept(self):
        # Arrange
        doc = manifest(algorithm={"gradient_checkpointing": True})
        validated = TrainingManifest.model_validate(doc)
        device = DeviceSpec(total_bytes=80 * GiB, name="NVIDIA A100-SXM4-80GB")

        # Act
        resolved = resolve_training_memory(validated, device, self.model())

        # Assert
        assert resolved.algorithm.micro_batch_size_per_gpu == 1
        assert resolved.algorithm.gradient_checkpointing is True

    def test_unset_checkpointing_sizes_as_on(self):
        validated = TrainingManifest.model_validate(manifest())

        settings = training_settings_from_manifest(validated)

        assert settings.gradient_checkpointing is True
