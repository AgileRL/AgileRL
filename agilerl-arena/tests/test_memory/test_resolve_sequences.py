# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Omitted vLLM ``max_num_seqs`` and ``gpu_memory_utilization`` resolve by engine placement."""

import pytest

from agilerl.arena.memory.resolve.sequences import (
    resolve_gpu_memory_utilization,
    resolve_max_num_seqs,
    rollout_engine_gpu_memory_utilization,
)

EXPERT_LORA_MODELS = [
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "zai-org/GLM-5.3-Flash",
    "XiaomiMiMo/MiMo-V2.6-Flash-RL",
]


class TestResolveMaxNumSeqs:
    def test_rollout_engine_runs_every_episode(self) -> None:
        assert resolve_max_num_seqs(None, 16, 8, False) == 128

    def test_colocated_engine_decodes_sixteen(self) -> None:
        assert resolve_max_num_seqs(None, 16, 8, True) == 16

    @pytest.mark.parametrize("colocated", [True, False])
    def test_explicit_value_is_kept(self, colocated: bool) -> None:
        assert resolve_max_num_seqs(4, 16, 8, colocated) == 4


class TestRolloutEngineGpuMemoryUtilization:
    @pytest.mark.parametrize("hub_id", EXPERT_LORA_MODELS)
    def test_catalogue_value_wins(self, hub_id: str) -> None:
        assert rollout_engine_gpu_memory_utilization(hub_id) == 0.85

    @pytest.mark.parametrize("hub_id", ["Qwen/Qwen3-4B", "org/not-catalogued", None])
    def test_other_models_take_vllm_default(self, hub_id: str | None) -> None:
        assert rollout_engine_gpu_memory_utilization(hub_id) == 0.9


class TestResolveGpuMemoryUtilization:
    def test_rollout_engine_takes_vllm_default(self) -> None:
        assert resolve_gpu_memory_utilization(None, False, "Qwen/Qwen3-4B") == 0.9

    @pytest.mark.parametrize("hub_id", EXPERT_LORA_MODELS)
    def test_rollout_engine_takes_the_catalogue_value(self, hub_id: str) -> None:
        assert resolve_gpu_memory_utilization(None, False, hub_id) == 0.85

    @pytest.mark.parametrize("hub_id", ["Qwen/Qwen3-4B", EXPERT_LORA_MODELS[0]])
    def test_colocated_engine_leaves_room_for_the_trainer(self, hub_id: str) -> None:
        assert resolve_gpu_memory_utilization(None, True, hub_id) == 0.3

    @pytest.mark.parametrize("colocated", [True, False])
    @pytest.mark.parametrize("hub_id", ["Qwen/Qwen3-4B", EXPERT_LORA_MODELS[0]])
    def test_explicit_value_is_kept(self, colocated: bool, hub_id: str) -> None:
        assert resolve_gpu_memory_utilization(0.9, colocated, hub_id) == 0.9
