# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""An omitted micro-batch is the largest divisor whose training estimate fits."""

from types import SimpleNamespace

import pytest

from agilerl.arena.memory.estimator import estimate_training
from agilerl.arena.memory.manifest import (
    run_config_from_manifest,
    training_settings_from_manifest,
)
from agilerl.arena.memory.resolve.batch import (
    resolve_micro_batch_size_per_gpu,
    resolve_mini_batch_size,
)
from agilerl.arena.memory.specs import DeviceSpec, GiB, ModelArch, ModelSpec
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


def _device(total_gib: float) -> DeviceSpec:
    return DeviceSpec(total_bytes=int(total_gib * GiB), name="NVIDIA L4")


def _resolve(manifest: TrainingManifest, device: DeviceSpec | None, model=None) -> int:
    spec = manifest.algorithm
    n_training_gpus = manifest.training.training_gpus_per_agent
    settings = None
    if device is not None:
        settings = training_settings_from_manifest(manifest, micro_batch_size=1)
    return resolve_micro_batch_size_per_gpu(
        spec, n_training_gpus, device, model, settings
    )


class TestResolveMiniBatchSize:
    def test_pinned_mini_is_kept(self) -> None:
        resolved = resolve_mini_batch_size(6, 16)

        assert resolved == 6

    def test_unset_mini_is_the_per_rank_count(self) -> None:
        resolved = resolve_mini_batch_size(None, 16)

        assert resolved == 16


class TestResolveMicroBatchSizePerGpu:
    def test_largest_divisor_is_kept_when_it_fits(self) -> None:
        # Arrange: one rank holds 16 trajectories.
        manifest = TrainingManifest.model_validate(_manifest())
        device = _device(24)

        # Act
        chosen = _resolve(manifest, device)

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_wide_micro_batch_shrinks_to_one_that_fits(self) -> None:
        # Arrange: unfiltered sampling scores the full vocabulary, so 16
        # trajectories do not fit; 8 does. The standard loss path keeps the
        # full tiles that make 16 oversized.
        raw = _manifest()
        raw["algorithm"]["top_k"] = 0
        raw["algorithm"]["use_liger_loss"] = False
        manifest = TrainingManifest.model_validate(raw)
        device = _device(4)

        # Act
        chosen = _resolve(manifest, device)

        # Assert
        assert chosen == 8
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_no_fitting_divisor_is_one(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_manifest())
        device = _device(1)

        # Act
        chosen = _resolve(manifest, device)

        # Assert
        assert chosen == 1
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_explicit_micro_batch_is_unchanged(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_manifest(micro=2))
        device = _device(1)

        # Act
        chosen = _resolve(manifest, device)

        # Assert
        assert chosen == 2
        assert manifest.algorithm.micro_batch_size_per_gpu == 2

    def test_pinned_mini_batch_caps_the_micro_batch(self) -> None:
        # Arrange: mini 4 must stay a multiple of the micro-batch.
        manifest = TrainingManifest.model_validate(_manifest(mini=4))
        device = _device(24)

        # Act
        chosen = _resolve(manifest, device)

        # Assert
        assert chosen == 4
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_no_device_returns_the_rank_trajectory_count(self) -> None:
        # Arrange: 16 prompts on 4 GPUs, 4 completions each.
        raw = _manifest()
        raw["algorithm"]["batch_size"] = 16
        raw["training"]["training_gpus_per_agent"] = 4
        raw["training"]["rollout_batch_size"] = 4
        manifest = TrainingManifest.model_validate(raw)

        # Act
        chosen = _resolve(manifest, None)

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_unknown_model_returns_the_rank_trajectory_count(self) -> None:
        # Arrange
        raw = _manifest()
        raw["network"]["pretrained_model_name_or_path"] = "missing/not-a-model"
        manifest = TrainingManifest.model_validate(raw)

        # Act
        chosen = _resolve(manifest, _device(24))

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_unknown_model_with_a_passed_spec_does_not_search(self) -> None:
        # Arrange: a passed spec would fit; the id is outside the catalog.
        raw = _manifest()
        raw["network"]["pretrained_model_name_or_path"] = "missing/not-a-model"
        manifest = TrainingManifest.model_validate(raw)
        info = SUPPORTED_MODEL_INFO[HUB_ID]
        supplied = ModelSpec(
            model_id="missing/not-a-model",
            arch=ModelArch.from_hf_config(info.config),
            n_params=1,
        )

        # Act
        chosen = _resolve(manifest, _device(24), supplied)

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_device_without_settings_raises(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_manifest())

        # Act / Assert
        with pytest.raises(ValueError, match="needs training settings"):
            resolve_micro_batch_size_per_gpu(
                manifest.algorithm,
                manifest.training.training_gpus_per_agent,
                _device(24),
            )

    def test_training_settings_without_a_device_uses_the_rank_batch(self) -> None:
        # Arrange: the same manifest's fit is 16; the reader has no device.
        raw = _manifest()

        # Act
        settings = training_settings_from_manifest(raw)

        # Assert
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
            _resolve(manifest, _device(24))

        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_no_device_pinned_mini_is_the_micro_batch(self) -> None:
        # Arrange: mini 4 divides 16 trajectories. The search does not run.
        manifest = TrainingManifest.model_validate(_manifest(mini=4))

        # Act
        chosen = _resolve(manifest, None)

        # Assert
        assert chosen == 4
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_caller_model_is_what_the_search_estimates(self, monkeypatch) -> None:
        # Arrange: catalog weights for this id do not carry n_params == 1.
        manifest = TrainingManifest.model_validate(_manifest())
        device = _device(24)
        info = SUPPORTED_MODEL_INFO[HUB_ID]
        supplied = ModelSpec(
            model_id=HUB_ID,
            arch=ModelArch.from_hf_config(info.config),
            n_params=1,
        )

        def fits_supplied_model(model, _device, _settings, orchestrated=False):
            return SimpleNamespace(
                fits_with_buffer=model.n_params == 1 and orchestrated is True
            )

        monkeypatch.setattr(
            "agilerl.arena.memory.resolve.batch.estimate_training",
            fits_supplied_model,
        )

        # Act
        chosen = _resolve(manifest, device, supplied)

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_unset_llmppo_keeps_a_split_micro_when_fusion_does_not_fit(self) -> None:
        # Arrange: 16 fused rows are over 5 GiB; the split peak fits.
        raw = _manifest()
        raw["algorithm"]["name"] = "LLMPPO"
        del raw["algorithm"]["group_size"]
        raw["algorithm"]["batch_size"] = 16
        raw["network"]["max_context_length"] = 512
        manifest = TrainingManifest.model_validate(raw)
        device = _device(5)
        model_config = SUPPORTED_MODEL_INFO[HUB_ID].config

        # Act
        chosen = _resolve(manifest, device)
        config = run_config_from_manifest(
            TrainingManifest.model_validate(raw), device, model_config
        )

        # Assert
        assert chosen == 16
        assert manifest.algorithm.micro_batch_size_per_gpu is None
        assert config.training.micro_batch_size == 16
        assert config.training.fuse_actor_critic_pass is False
        peak = estimate_training(
            config.model, device, config.training, orchestrated=True
        )
        fused = estimate_training(
            config.model,
            device,
            config.training.model_copy(update={"fuse_actor_critic_pass": True}),
            orchestrated=True,
        )
        assert peak.total_bytes != fused.total_bytes

    def test_unset_llmppo_skips_a_micro_that_only_fits_split(self) -> None:
        # Arrange: 16 fused rows are inside 5.5 GiB and outside the buffer.
        raw = _manifest()
        raw["algorithm"]["name"] = "LLMPPO"
        del raw["algorithm"]["group_size"]
        raw["algorithm"]["batch_size"] = 16
        raw["network"]["max_context_length"] = 512
        manifest = TrainingManifest.model_validate(raw)
        device = _device(5.5)

        # Act
        chosen = _resolve(manifest, device)

        # Assert
        assert chosen == 8
        assert manifest.algorithm.micro_batch_size_per_gpu is None

    def test_run_config_measures_the_resolved_size(self) -> None:
        # Arrange
        raw = _manifest()
        raw["algorithm"]["top_k"] = 0
        raw["algorithm"]["use_liger_loss"] = False
        caller = TrainingManifest.model_validate(raw)
        device = _device(4)
        model_config = SUPPORTED_MODEL_INFO[HUB_ID].config

        # Act
        config = run_config_from_manifest(caller, device, model_config)
        copy = TrainingManifest.model_validate(raw)
        chosen = _resolve(copy, device, config.model)

        # Assert
        assert config.training.micro_batch_size == chosen
        assert chosen == 8
        assert caller.algorithm.micro_batch_size_per_gpu is None
        assert copy.algorithm.micro_batch_size_per_gpu is None
