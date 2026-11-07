# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""One async weight sync is priced from its bytes, rounds and link."""

from __future__ import annotations

import pytest

from agilerl.arena.memory.lora_formulas import (
    lora_sync_buffer_bytes,
    merged_lora_sync_bytes,
)
from agilerl.arena.memory.resolve import weight_sync
from agilerl.arena.memory.resolve.weight_sync import (
    ADAPTER_SYNC_SECONDS,
    LINK_EGRESS_BYTES_PER_SECOND,
    MERGED_SYNC_BARRIER_SECONDS,
    MERGED_SYNC_ENGINE_ROUND_SECONDS,
    MERGED_SYNC_PAUSE_SECONDS,
    MERGED_SYNC_TRAINER_ROUND_SECONDS,
    MULTI_TURN_MERGED_SYNC_ENGINE_ROUND_SECONDS,
    NVLINK_BROADCAST_BYTES_PER_SECOND,
    adapter_sync_seconds,
    merged_sync_seconds,
)
from agilerl.arena.memory.specs import ModelArch, ModelSpec
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


def _qwen3_4b() -> ModelSpec:
    hub_id = "Qwen/Qwen3-4B"
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(SUPPORTED_MODEL_INFO[hub_id].config),
    )


ENGINE_BOUND_FIXED_SECONDS = (
    MERGED_SYNC_PAUSE_SECONDS
    + MERGED_SYNC_TRAINER_ROUND_SECONDS
    + 28 * MERGED_SYNC_ENGINE_ROUND_SECONDS
    + MERGED_SYNC_BARRIER_SECONDS
)
"""Qwen3-4B's 28 rounds when every engine round outlasts the next merge, without the bytes."""


class TestMergedSyncSeconds:
    def test_engine_bound_rounds_cost_the_engine_side_after_the_first_merge(
        self,
    ) -> None:
        # Arrange: Qwen3-4B all-linear moves 7.27 GB, 28 rounds of 256 MiB.
        model = _qwen3_4b()
        total, _ = merged_lora_sync_bytes(model.arch, "all-linear", 0, 2)

        # Act
        seconds = merged_sync_seconds(model, "all-linear", 0, 4, 1, False, "cuda_ipc")

        # Assert
        assert total == pytest.approx(7.27e9, rel=0.01)
        assert seconds == pytest.approx(
            ENGINE_BOUND_FIXED_SECONDS
            + 4 * total / LINK_EGRESS_BYTES_PER_SECOND["cuda_ipc"]
        )

    def test_trainer_bound_rounds_cost_the_merge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a 50 ms merge outlasts every engine round over NVLink.
        model = _qwen3_4b()
        total, _ = merged_lora_sync_bytes(model.arch, "all-linear", 0, 2)
        monkeypatch.setattr(weight_sync, "MERGED_SYNC_TRAINER_ROUND_SECONDS", 0.05)

        # Act
        seconds = merged_sync_seconds(model, "all-linear", 0, 4, 1, False, "cuda_ipc")

        # Assert: 28 merges, then the last round's receive.
        assert seconds == pytest.approx(
            MERGED_SYNC_PAUSE_SECONDS
            + 28 * 0.05
            + MERGED_SYNC_ENGINE_ROUND_SECONDS
            + 4 * total / LINK_EGRESS_BYTES_PER_SECOND["cuda_ipc"] / 28
            + MERGED_SYNC_BARRIER_SECONDS
        )

    def test_each_engine_reads_once_and_broadcasts_over_nvlink(self) -> None:
        # Arrange
        model = _qwen3_4b()
        total, _ = merged_lora_sync_bytes(model.arch, "all-linear", 0, 2)
        fixed = ENGINE_BOUND_FIXED_SECONDS

        # Act
        tp2 = merged_sync_seconds(model, "all-linear", 0, 4, 2, False, "tcp")
        tp8 = merged_sync_seconds(model, "all-linear", 0, 4, 8, False, "tcp")

        # Assert
        assert tp8 == pytest.approx(tp2)
        assert tp2 - fixed == pytest.approx(
            4 * total / LINK_EGRESS_BYTES_PER_SECOND["tcp"]
            + total / NVLINK_BROADCAST_BYTES_PER_SECOND
        )

    def test_multi_turn_rounds_cost_more(self) -> None:
        # Arrange
        model = _qwen3_4b()

        # Act
        single = merged_sync_seconds(model, "all-linear", 0, 4, 1, False, "cuda_ipc")
        multi = merged_sync_seconds(model, "all-linear", 0, 4, 1, True, "cuda_ipc")

        # Assert
        assert multi - single == pytest.approx(
            28
            * (
                MULTI_TURN_MERGED_SYNC_ENGINE_ROUND_SECONDS
                - MERGED_SYNC_ENGINE_ROUND_SECONDS
            )
        )

    def test_tcp_takes_over_a_minute_for_four_engines(self) -> None:
        # Act
        seconds = merged_sync_seconds(_qwen3_4b(), "all-linear", 0, 4, 1, False, "tcp")

        # Assert: 4 x 7.27 GB at 0.4 GB/s.
        assert seconds > 70


class TestAdapterSyncSeconds:
    def test_nvlink_costs_the_engine_rpcs(self) -> None:
        # Act
        seconds = adapter_sync_seconds(
            _qwen3_4b(), 16, "all-linear", 0, 4, 1, "cuda_ipc"
        )

        # Assert: 4 x 66 MB is under 1 ms at 275 GB/s.
        assert seconds == pytest.approx(ADAPTER_SYNC_SECONDS, abs=1e-3)

    def test_every_engine_rank_reads_the_adapter(self) -> None:
        # Arrange
        model = _qwen3_4b()
        adapter = lora_sync_buffer_bytes(model.arch, 16, "all-linear", 0, 2)

        # Act
        seconds = adapter_sync_seconds(model, 16, "all-linear", 0, 2, 2, "tcp")

        # Assert
        assert seconds == pytest.approx(
            ADAPTER_SYNC_SECONDS + 4 * adapter / LINK_EGRESS_BYTES_PER_SECOND["tcp"]
        )

    def test_scales_with_tensor_parallel_width(self) -> None:
        # Arrange
        model = _qwen3_4b()
        adapter = lora_sync_buffer_bytes(model.arch, 16, "all-linear", 0, 2)

        # Act
        one_tp4 = adapter_sync_seconds(model, 16, "all-linear", 0, 1, 4, "tcp")
        one_tp8 = adapter_sync_seconds(model, 16, "all-linear", 0, 1, 8, "tcp")

        # Assert
        assert one_tp8 - one_tp4 == pytest.approx(
            4 * adapter / LINK_EGRESS_BYTES_PER_SECOND["tcp"]
        )
