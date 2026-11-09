# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Omitted engine tensor parallel and ``max_num_seqs`` are chosen from the allocation."""

from __future__ import annotations

import pytest

from agilerl.arena.memory.manifest import (
    generation_settings_from_manifest,
    lookup_gpu,
)
from agilerl.arena.memory.resolve.engine import resolve_engine_tensor_parallel
from agilerl.arena.memory.specs import DeviceSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.networks import VLLMConfig, default_rollout_vllm_config


def _qwen_manifest(
    tensor_parallel_size: int | None = None,
    hub_id: str = "Qwen/Qwen3.6-35B-A3B",
) -> dict:
    vllm_config: dict[str, int | float] = {
        "gpu_memory_utilization": 0.95,
        "max_num_seqs": 16,
    }
    if tensor_parallel_size is not None:
        vllm_config["tensor_parallel_size"] = tensor_parallel_size
    return {
        "algorithm": {
            "name": "GRPO",
            "batch_size": 16,
            "group_size": 8,
            "vllm_config": vllm_config,
        },
        "environment": {"env_type": "rollout", "env_image": "env:latest"},
        "network": {
            "pretrained_model_name_or_path": hub_id,
            "max_context_length": 32768,
        },
        "training": {
            "rollout_mode": "async",
            "rollout_engines_per_agent": "auto",
            "training_gpus_per_agent": 4,
            "pop_size": 1,
            "rollout_batch_size": 4,
        },
        "replay_buffer": {"kind": "llm"},
    }


def _a100() -> DeviceSpec:
    gpu = lookup_gpu("nvidia-a100-80gb")
    assert gpu is not None
    return gpu.device_spec()


class TestResolveEngineTensorParallel:
    def test_qwen36_32k_on_eight_a100s_uses_width_2(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_qwen_manifest())
        device = _a100()

        # Act
        width = resolve_engine_tensor_parallel(manifest, 8, device, 8)

        # Assert
        assert width == 2
        assert isinstance(manifest.algorithm, RolloutLLMSpec)
        assert manifest.algorithm.vllm_config is not None
        assert manifest.algorithm.vllm_config.tensor_parallel_size == 2

    @pytest.mark.parametrize(("context", "expected"), [(4096, 1), (4608, 2)])
    def test_a_width_must_clear_demand_by_the_kv_margin(
        self, context: int, expected: int
    ) -> None:
        # Arrange: TP=1 predicts a 1.92 GiB attention pool on four leftover
        # A100s. 16 sequences at 4096 tokens want 1.29 GiB; at 4608 they want
        # 1.61 GiB, inside the pool but not by the 512 MiB margin. 16
        # prompts keep every engine at max_num_seqs.
        raw = _qwen_manifest()
        raw["network"]["max_context_length"] = context
        raw["training"]["rollout_batch_size"] = 16
        manifest = TrainingManifest.model_validate(raw)

        # Act
        width = resolve_engine_tensor_parallel(manifest, 8, _a100(), 8)

        # Assert
        assert width == expected

    @pytest.mark.parametrize("width", [1, 4])
    def test_explicit_width_is_left_unchanged(self, width: int) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(
            _qwen_manifest(tensor_parallel_size=width)
        )
        device = _a100()

        # Act
        resolved = resolve_engine_tensor_parallel(manifest, 8, device, 8)

        # Assert
        assert resolved is None
        assert isinstance(manifest.algorithm, RolloutLLMSpec)
        assert manifest.algorithm.vllm_config is not None
        assert manifest.algorithm.vllm_config.tensor_parallel_size == width

    def test_missing_vllm_config_object_returns_none(self) -> None:
        manifest = TrainingManifest.model_validate(_qwen_manifest())
        assert isinstance(manifest.algorithm, RolloutLLMSpec)
        manifest.algorithm = manifest.algorithm.model_copy(update={"vllm_config": None})

        assert resolve_engine_tensor_parallel(manifest, 8, _a100(), 8) is None

    def test_more_engines_than_leftover_gpus_returns_none(self) -> None:
        raw = _qwen_manifest()
        raw["training"]["training_gpus_per_agent"] = 1
        raw["training"]["rollout_engines_per_agent"] = 8
        manifest = TrainingManifest.model_validate(raw)

        assert resolve_engine_tensor_parallel(manifest, 8, _a100(), 8) is None

    def test_stripped_towers_still_resolve(self) -> None:
        raw = _qwen_manifest()
        raw["environment"]["modality"] = "text"
        manifest = TrainingManifest.model_validate(raw)

        width = resolve_engine_tensor_parallel(manifest, 8, _a100(), 8)

        assert width == 2

    def test_no_width_that_can_serve_returns_none(self) -> None:
        manifest = TrainingManifest.model_validate(_qwen_manifest())
        tiny = DeviceSpec(total_bytes=1, name="tiny")

        assert resolve_engine_tensor_parallel(manifest, 8, tiny, 8) is None

    def test_unknown_hub_id_returns_none(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(
            _qwen_manifest(hub_id="missing/not-a-supported-model")
        )
        device = _a100()

        # Act
        resolved = resolve_engine_tensor_parallel(manifest, 8, device, 8)

        # Assert
        assert resolved is None
        assert isinstance(manifest.algorithm, RolloutLLMSpec)
        assert manifest.algorithm.vllm_config is not None
        assert manifest.algorithm.vllm_config.tensor_parallel_size is None
        assert (
            "tensor_parallel_size"
            not in manifest.algorithm.vllm_config.model_fields_set
        )

    def test_illegal_mamba_width_is_skipped(self) -> None:
        # leftover 6 includes TP 3; Nemotron Nano's 8 Mamba groups reject that.
        raw = _qwen_manifest(hub_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
        raw["training"]["training_gpus_per_agent"] = 2
        manifest = TrainingManifest.model_validate(raw)
        device = _a100()

        width = resolve_engine_tensor_parallel(manifest, 8, device, 8)

        assert width == 1
        assert isinstance(manifest.algorithm, RolloutLLMSpec)
        assert manifest.algorithm.vllm_config is not None
        assert manifest.algorithm.vllm_config.tensor_parallel_size == 1

    def test_manifest_without_vllm_config_resolves_on_eight_a100s(self) -> None:
        # Arrange
        raw = _qwen_manifest()
        raw["algorithm"].pop("vllm_config")
        raw["training"].pop("rollout_mode")
        manifest = TrainingManifest.model_validate(raw)
        device = _a100()
        assert isinstance(manifest.algorithm, RolloutLLMSpec)
        vllm_config = manifest.algorithm.vllm_config
        assert vllm_config is not None
        assert vllm_config.gpu_memory_utilization == 0.9
        assert vllm_config.max_num_seqs is None
        assert vllm_config.tensor_parallel_size is None
        assert "tensor_parallel_size" not in vllm_config.model_fields_set

        # Act
        width = resolve_engine_tensor_parallel(manifest, 8, device, 8)

        # Assert
        assert width == 2
        assert manifest.algorithm.vllm_config is not None
        assert manifest.algorithm.vllm_config.tensor_parallel_size == 2


def _qwen3_4b_manifest(
    context: int = 1536,
    trainers: int = 4,
    engines: int = 4,
    vllm_config: dict[str, int | float] | None = None,
) -> dict:
    """GRPO on Qwen3-4B: 64 prompts x 8, no rollout_batch_size or vllm_config."""
    algorithm: dict[str, object] = {"name": "GRPO", "batch_size": 64, "group_size": 8}
    if vllm_config is not None:
        algorithm["vllm_config"] = vllm_config
    return {
        "algorithm": algorithm,
        "environment": {"env_type": "rollout", "env_image": "env:latest"},
        "network": {
            "pretrained_model_name_or_path": "Qwen/Qwen3-4B",
            "max_context_length": context,
        },
        "training": {
            "rollout_mode": "async",
            "rollout_engines_per_agent": engines,
            "training_gpus_per_agent": trainers,
        },
        "replay_buffer": {"kind": "llm"},
    }


def _engine_max_num_seqs(raw: dict, engines: int) -> int:
    manifest = TrainingManifest.model_validate(raw)
    settings = generation_settings_from_manifest(
        manifest, tensor_parallel_size=1, rollout_engines=engines
    )
    return settings.max_num_seqs


class TestOmittedAsyncMaxNumSeqs:
    def test_long_context_engine_runs_every_episode(self) -> None:
        # 64 groups over 4 engines x group 8 at 16k tokens: 128 episodes per
        # engine, more than one A100-80's KV pool holds at full length.
        assert _engine_max_num_seqs(_qwen3_4b_manifest(context=16384), 4) == 128

    def test_prompts_round_up_per_engine(self) -> None:
        raw = _qwen3_4b_manifest(trainers=2, engines=4)
        raw["training"]["rollout_batch_size"] = 6

        assert _engine_max_num_seqs(raw, 4) == 2 * 8

    def test_mutated_group_size_counts_at_its_ceiling(self) -> None:
        raw = _qwen3_4b_manifest()
        raw["mutation"] = {
            "rl_hp_selection": {"group_size": {"min": 4, "max": 16}},
        }

        assert _engine_max_num_seqs(raw, 4) == 16 * 16

    def test_explicit_max_num_seqs_is_kept(self) -> None:
        raw = _qwen3_4b_manifest(context=16384, vllm_config={"max_num_seqs": 12})

        assert _engine_max_num_seqs(raw, 4) == 12


class TestDefaultRolloutVllmConfig:
    def test_leaves_tensor_parallel_size_unset(self) -> None:
        # Act
        config = default_rollout_vllm_config()

        # Assert
        assert config.tensor_parallel_size is None
        assert "tensor_parallel_size" not in config.model_fields_set
        assert VLLMConfig().tensor_parallel_size is None

    def test_leaves_max_num_seqs_unset(self) -> None:
        # Act
        config = default_rollout_vllm_config()

        # Assert
        assert config.max_num_seqs is None
        assert "max_num_seqs" not in config.model_dump(exclude_none=True)


class TestGenerationSettingsFromManifest:
    def test_unset_width_without_a_candidate_is_refused(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_qwen_manifest())

        # Act / Assert
        with pytest.raises(ValueError, match="tensor_parallel_size is unset"):
            generation_settings_from_manifest(manifest)

    def test_scores_the_width_it_is_given(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_qwen_manifest())

        # Act
        settings = generation_settings_from_manifest(manifest, tensor_parallel_size=2)

        # Assert
        assert settings.tensor_parallel_size == 2
        assert manifest.algorithm.vllm_config is not None
        assert manifest.algorithm.vllm_config.tensor_parallel_size is None


class TestToPayload:
    def test_omits_an_unset_width(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(_qwen_manifest())

        # Act
        payload = manifest.to_payload()

        # Assert
        assert "tensor_parallel_size" not in payload["algorithm"]["vllm_config"]

    def test_keeps_an_explicit_width_of_one(self) -> None:
        # Arrange
        manifest = TrainingManifest.model_validate(
            _qwen_manifest(tensor_parallel_size=1)
        )

        # Act
        payload = manifest.to_payload()

        # Assert
        assert payload["algorithm"]["vllm_config"]["tensor_parallel_size"] == 1
