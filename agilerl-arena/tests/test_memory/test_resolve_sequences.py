# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Omitted vLLM ``max_num_seqs`` and ``gpu_memory_utilization`` resolve by engine placement."""

import pytest

from agilerl.arena.memory.resolve.sequences import (
    resolve_gpu_memory_utilization,
    resolve_max_num_seqs,
)


class TestResolveMaxNumSeqs:
    def test_rollout_engine_runs_every_episode(self) -> None:
        assert resolve_max_num_seqs(None, 16, 8, False) == 128

    def test_colocated_engine_decodes_sixteen(self) -> None:
        assert resolve_max_num_seqs(None, 16, 8, True) == 16

    @pytest.mark.parametrize("colocated", [True, False])
    def test_explicit_value_is_kept(self, colocated: bool) -> None:
        assert resolve_max_num_seqs(4, 16, 8, colocated) == 4


class TestResolveGpuMemoryUtilization:
    def test_rollout_engine_takes_vllm_default(self) -> None:
        assert resolve_gpu_memory_utilization(None, False) == 0.9

    def test_colocated_engine_leaves_room_for_the_trainer(self) -> None:
        assert resolve_gpu_memory_utilization(None, True) == 0.3

    @pytest.mark.parametrize("colocated", [True, False])
    def test_explicit_value_is_kept(self, colocated: bool) -> None:
        assert resolve_gpu_memory_utilization(0.5, colocated) == 0.5
