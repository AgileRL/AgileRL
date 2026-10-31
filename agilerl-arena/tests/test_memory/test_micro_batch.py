# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Batch shapes, and the micro-batch the training-memory search picks for a manifest."""

import pytest

from agilerl.arena.memory.estimator import estimate_training
from agilerl.arena.memory.manifest import (
    run_config_from_manifest,
    training_memory_from_manifest,
    training_settings_from_manifest,
)
from agilerl.arena.memory.resolve.batch import (
    catalog_model,
    engine_group_size,
    resolve_micro_batch_size_per_gpu,
    resolve_mini_batch_size,
)
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

HUB_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _manifest(micro: int | None = None, mini: int | None = None) -> dict:
    algorithm: dict[str, object] = {
        "name": "GRPO",
        "batch_size": 4,
        "group_size": 4,
        "vllm_config": {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.3,
        },
    }
    if micro is not None:
        algorithm["micro_batch_size_per_gpu"] = micro
    if mini is not None:
        algorithm["mini_batch_size"] = mini
    return {
        "algorithm": algorithm,
        "environment": {"env_type": "rollout", "env_image": "env:latest"},
        "network": {
            "pretrained_model_name_or_path": HUB_ID,
            "max_context_length": 128,
        },
        "training": {
            "max_steps": 10,
            "rollout_mode": "async",
            "rollout_engines_per_agent": 1,
            "rollout_batch_size": 1,
            "pop_size": 1,
        },
        "replay_buffer": {"kind": "llm"},
    }


def _llmppo_manifest() -> dict:
    raw = _manifest()
    raw["algorithm"]["name"] = "LLMPPO"
    del raw["algorithm"]["group_size"]
    raw["algorithm"]["batch_size"] = 16
    raw["network"]["max_context_length"] = 512
    raw["network"]["lora_config"] = {"lora_dropout": 0.05}
    return raw


def _device(total_gib: float) -> DeviceSpec:
    return DeviceSpec(total_bytes=int(total_gib * GiB), name="NVIDIA L4")


def _plan(raw: dict, total_gib: float) -> TrainingMemoryPlan:
    manifest = TrainingManifest.model_validate(raw)
    return training_memory_from_manifest(
        manifest, catalog_model(manifest.algorithm), _device(total_gib)
    )


class TestEngineGroupSize:
    def test_fixed_group_size_is_kept(self) -> None:
        manifest = TrainingManifest.model_validate(_manifest())

        assert engine_group_size(manifest.algorithm, manifest.mutation) == 4

    def test_mutable_group_size_is_raised_to_its_ceiling(self) -> None:
        # Arrange
        raw = _manifest()
        raw["mutation"] = {"rl_hp_selection": {"group_size": {"min": 2, "max": 12}}}
        manifest = TrainingManifest.model_validate(raw)

        # Act
        size = engine_group_size(manifest.algorithm, manifest.mutation)

        # Assert
        assert size == 12


class TestResolveMiniBatchSize:
    def test_pinned_mini_is_kept(self) -> None:
        resolved = resolve_mini_batch_size(6, 16)

        assert resolved == 6

    def test_unset_mini_is_the_per_rank_count(self) -> None:
        resolved = resolve_mini_batch_size(None, 16)

        assert resolved == 16


class TestResolveMicroBatchSizePerGpu:
    def test_set_micro_batch_is_kept(self) -> None:
        manifest = TrainingManifest.model_validate(_manifest(micro=2))

        chosen = resolve_micro_batch_size_per_gpu(manifest.algorithm, 1)

        assert chosen == 2

    def test_unset_micro_batch_is_the_rank_trajectory_count(self) -> None:
        # Arrange: 16 prompts on 4 GPUs, 4 completions each.
        raw = _manifest()
        raw["algorithm"]["batch_size"] = 16
        raw["training"]["training_gpus_per_agent"] = 4
        raw["training"]["rollout_batch_size"] = 4
        manifest = TrainingManifest.model_validate(raw)

        # Act
        chosen = resolve_micro_batch_size_per_gpu(manifest.algorithm, 4)

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_pinned_mini_batch_is_the_micro_batch(self) -> None:
        manifest = TrainingManifest.model_validate(_manifest(mini=4))

        chosen = resolve_micro_batch_size_per_gpu(manifest.algorithm, 1)

        assert chosen == 4

    def test_training_settings_without_a_device_use_the_rank_batch(self) -> None:
        settings = training_settings_from_manifest(_manifest())

        assert settings.micro_batch_size == 16

    def test_no_splitting_mini_batch_raises(self) -> None:
        # Arrange: one rank holds 4 trajectories; mini-batch 3 divides none.
        raw = _manifest(mini=3)
        raw["algorithm"]["batch_size"] = 2
        raw["training"]["training_gpus_per_agent"] = 2
        raw["training"]["rollout_batch_size"] = 2
        manifest = TrainingManifest.model_validate(raw)

        # Act / Assert
        with pytest.raises(ValueError, match="divisible by mini_batch_size"):
            resolve_micro_batch_size_per_gpu(manifest.algorithm, 2)


class TestTrainingMemoryFromManifest:
    def test_roomy_device_packs_the_whole_mini_batch_without_checkpointing(
        self,
    ) -> None:
        plan = _plan(_manifest(), 24)

        assert plan.micro_batch_tokens_per_gpu == 16 * 128
        assert plan.micro_batch_size_per_gpu == 16
        assert plan.gradient_checkpointing is False

    def test_tight_device_checkpoints_before_shrinking_the_micro_batch(self) -> None:
        plan = _plan(_manifest(), 3.37)

        assert plan.micro_batch_tokens_per_gpu == 16 * 128
        assert plan.gradient_checkpointing is True

    def test_full_vocabulary_scoring_shrinks_the_micro_batch(self) -> None:
        # Arrange: unfiltered sampling scores the full vocabulary.
        raw = _manifest()
        raw["algorithm"]["top_k"] = 0

        # Act
        plan = _plan(raw, 3.37)

        # Assert
        assert plan.micro_batch_tokens_per_gpu == 1024
        assert plan.micro_batch_size_per_gpu == 8
        assert plan.gradient_checkpointing is True

    def test_pinned_mini_batch_caps_the_micro_batch(self) -> None:
        plan = _plan(_manifest(mini=4), 24)

        assert plan.micro_batch_tokens_per_gpu == 4 * 128
        assert plan.micro_batch_size_per_gpu == 4

    def test_pinned_micro_batch_is_kept(self) -> None:
        plan = _plan(_manifest(micro=2), 24)

        assert plan.micro_batch_size_per_gpu == 2
        assert plan.micro_batch_tokens_per_gpu == 2 * 128

    def test_packed_budget_steps_by_1024_tokens_not_by_rows(self) -> None:
        # Arrange: 1536-token rows, so a row multiple is not a 1024 multiple.
        raw = _manifest()
        raw["network"]["max_context_length"] = 1536

        # Act
        plan = _plan(raw, 3.8)

        # Assert
        assert plan.micro_batch_tokens_per_gpu == 10 * 1024
        assert plan.micro_batch_size_per_gpu == 4

    def test_tight_device_floors_at_the_longest_row(self) -> None:
        raw = _manifest()
        raw["network"]["max_context_length"] = 1536

        plan = _plan(raw, 3.2)

        assert plan.micro_batch_tokens_per_gpu == 1536
        assert plan.micro_batch_size_per_gpu == 1

    def test_llmppo_budget_prices_the_actor_and_critic_copies(self) -> None:
        # Arrange: the same learn under GRPO, which packs one copy.
        raw = _llmppo_manifest()
        grpo = _llmppo_manifest()
        grpo["algorithm"].update(name="GRPO", batch_size=4, group_size=4)

        # Act
        plan = _plan(raw, 5)
        grpo_plan = _plan(grpo, 5)

        # Assert
        assert plan.micro_batch_tokens_per_gpu == 7 * 1024
        assert grpo_plan.micro_batch_tokens_per_gpu == 8 * 1024
        assert plan.gradient_checkpointing is True

    def test_manifest_is_not_written(self) -> None:
        manifest = TrainingManifest.model_validate(_manifest())

        training_memory_from_manifest(
            manifest, catalog_model(manifest.algorithm), _device(24)
        )

        assert manifest.algorithm.micro_batch_size_per_gpu is None
        assert manifest.algorithm.gradient_checkpointing is None

    def test_raises_when_nothing_fits(self) -> None:
        with pytest.raises(TrainingMemoryUnfitError, match="at the lightest settings"):
            _plan(_manifest(), 2)


class TestTrainingMemoryFromManifestFsdpChoice:
    def test_roomy_single_gpu_trains_ddp(self) -> None:
        plan = _plan(_manifest(), 24)

        assert plan.fsdp is None

    def test_omitted_fsdp_plans_like_an_empty_fsdp_block(self) -> None:
        # Arrange: at 3 GiB the FSDP plan beats the DDP one on one L4.
        searched = _manifest()
        searched["algorithm"]["fsdp"] = {}

        # Act
        omitted_plan = _plan(_manifest(), 3.0)
        searched_plan = _plan(searched, 3.0)

        # Assert
        assert omitted_plan.fsdp is not None
        assert omitted_plan.fsdp == searched_plan.fsdp
        assert (
            omitted_plan.micro_batch_tokens_per_gpu,
            omitted_plan.gradient_checkpointing,
        ) == (
            searched_plan.micro_batch_tokens_per_gpu,
            searched_plan.gradient_checkpointing,
        )


class TestRunConfigFromManifest:
    def test_measures_the_resolved_plan(self) -> None:
        # Arrange
        raw = _manifest()
        raw["algorithm"]["top_k"] = 0
        caller = TrainingManifest.model_validate(raw)
        model_config = SUPPORTED_MODEL_INFO[HUB_ID].config

        # Act
        config = run_config_from_manifest(caller, _device(3.37), model_config)

        # Assert
        assert config.training.micro_batch_size == 8
        assert config.training.micro_batch_tokens == 1024
        assert config.training.gradient_checkpointing is True
        assert caller.algorithm.micro_batch_size_per_gpu is None

    def test_omitted_fsdp_is_sized_at_the_planned_fsdp(self) -> None:
        caller = TrainingManifest.model_validate(_manifest())
        model_config = SUPPORTED_MODEL_INFO[HUB_ID].config

        config = run_config_from_manifest(caller, _device(3.0), model_config)

        assert config.training.fsdp == _plan(_manifest(), 3.0).fsdp
        assert config.training.fsdp is not None
        assert caller.algorithm.fsdp is None

    def test_padded_llmppo_fuses_policy_and_value_rows(self) -> None:
        config = run_config_from_manifest(
            _llmppo_manifest(), _device(5), SUPPORTED_MODEL_INFO[HUB_ID].config
        )

        assert config.training.micro_batch_size == 8
        assert config.training.grad_forward_rows == 16

    def test_unfit_manifest_is_sized_at_the_lightest_plan(self) -> None:
        # Arrange
        model_config = SUPPORTED_MODEL_INFO[HUB_ID].config

        # Act
        config = run_config_from_manifest(_manifest(), _device(2), model_config)
        breakdown = estimate_training(
            config.model, config.train_device, config.training, orchestrated=True
        )

        # Assert
        assert config.training.micro_batch_size == 1
        assert config.training.micro_batch_tokens == 128
        assert config.training.gradient_checkpointing is True
        assert not breakdown.fits
