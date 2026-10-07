# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``agilerl.utils.learn_profiler``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from agilerl.arena.models.profiling import ProfilingConfig
from agilerl.utils.learn_profiler import LearnProfiler, default_profile_ranks


@pytest.fixture
def fake_cuda_memory(monkeypatch):
    """Stand in for the CUDA allocator: history recording, snapshot dump and stats."""
    recorded: list[dict] = []
    gib = 1024**3

    def dump_snapshot(path: str) -> None:
        Path(path).write_bytes(b"snapshot")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_record_memory_history",
        lambda **kwargs: recorded.append(kwargs),
    )
    monkeypatch.setattr(torch.cuda.memory, "_dump_snapshot", dump_snapshot)
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {
            "allocated_bytes.all.current": 2 * gib,
            "active_bytes.all.current": 2 * gib,
            "reserved_bytes.all.current": 3 * gib,
            "inactive_split_bytes.all.current": gib // 2,
            "num_alloc_retries": 4,
        },
    )
    return recorded


def run_learn_calls(profiler: LearnProfiler, calls: int, micro_batches: int) -> None:
    """Drive ``calls`` learn calls of ``micro_batches`` tiny CPU matmuls each."""
    for _ in range(calls):
        with profiler.learn_step():
            for _ in range(micro_batches):
                with profiler.micro_batch():
                    torch.ones(4, 4) @ torch.ones(4, 4)


def raise_in_learn_step(profiler: LearnProfiler, error: Exception) -> None:
    """Raise ``error`` inside one ``learn_step``."""
    with profiler.learn_step():
        raise error


class TestDefaultProfileRanks:
    def test_picks_rank_zero_and_first_rank_of_second_shard_group(self) -> None:
        assert default_profile_ranks(16, 8) == frozenset({0, 8})

    def test_unsharded_world_picks_rank_zero(self) -> None:
        assert default_profile_ranks(16, None) == frozenset({0})

    def test_single_shard_group_picks_rank_zero(self) -> None:
        assert default_profile_ranks(8, 8) == frozenset({0})


class TestLearnProfilerInit:
    def test_records_allocator_history_when_snapshots_are_on(
        self, tmp_path, fake_cuda_memory
    ) -> None:
        config = ProfilingConfig(
            output_dir=str(tmp_path),
            memory_snapshot_on_oom=True,
            memory_history_max_entries=500,
        )

        LearnProfiler(config, rank=0, world_size=1, shard_group_size=None)

        assert fake_cuda_memory == [{"max_entries": 500, "stacks": "python"}]

    def test_snapshots_without_cuda_raise(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        config = ProfilingConfig(output_dir=str(tmp_path), memory_snapshot_on_oom=True)

        with pytest.raises(ValueError, match="memory_snapshot_on_oom needs CUDA"):
            LearnProfiler(config, rank=0, world_size=1, shard_group_size=None)


class TestLearnProfilerLearnStep:
    def test_oom_writes_snapshot_logs_summary_and_reraises(
        self, tmp_path, fake_cuda_memory, caplog
    ) -> None:
        # Arrange
        config = ProfilingConfig(output_dir=str(tmp_path), memory_snapshot_on_oom=True)
        profiler = LearnProfiler(config, rank=3, world_size=16, shard_group_size=8)
        with profiler.learn_step():
            pass

        # Act
        with (
            caplog.at_level(logging.ERROR, logger="agilerl.utils.learn_profiler"),
            pytest.raises(torch.OutOfMemoryError, match="expert GEMM"),
        ):
            raise_in_learn_step(profiler, torch.OutOfMemoryError("expert GEMM"))

        # Assert
        snapshot = tmp_path / "rank3_learn2_oom_snapshot.pickle"
        assert snapshot.read_bytes() == b"snapshot"
        assert (
            "allocated=2.00 GiB active=2.00 GiB reserved=3.00 GiB "
            "inactive_split=0.50 GiB alloc_retries=4"
        ) in caplog.text

    def test_other_errors_propagate_without_snapshot(
        self, tmp_path, fake_cuda_memory
    ) -> None:
        config = ProfilingConfig(output_dir=str(tmp_path), memory_snapshot_on_oom=True)
        profiler = LearnProfiler(config, rank=0, world_size=1, shard_group_size=None)

        with pytest.raises(RuntimeError, match="not finite"):
            raise_in_learn_step(profiler, RuntimeError("loss not finite"))

        assert list(tmp_path.iterdir()) == []

    def test_oom_with_snapshots_off_propagates_without_snapshot(self, tmp_path) -> None:
        profiler = LearnProfiler(None, rank=0, world_size=1, shard_group_size=None)

        with pytest.raises(torch.OutOfMemoryError):
            raise_in_learn_step(profiler, torch.OutOfMemoryError("out of memory"))

        assert profiler.learn_calls == 1
        assert list(tmp_path.iterdir()) == []


class TestLearnProfilerMicroBatch:
    def test_traces_first_micro_batch_of_the_configured_learn_call(
        self, tmp_path, caplog
    ) -> None:
        # Arrange
        config = ProfilingConfig(output_dir=str(tmp_path), torch_profile_step=2)
        profiler = LearnProfiler(config, rank=0, world_size=1, shard_group_size=None)

        # Act
        with caplog.at_level(logging.INFO, logger="agilerl.utils.learn_profiler"):
            run_learn_calls(profiler, calls=3, micro_batches=2)

        # Assert
        assert [path.name for path in tmp_path.iterdir()] == ["rank0_learn2_trace.json"]
        trace = json.loads((tmp_path / "rank0_learn2_trace.json").read_text())
        assert any(event.get("name") == "aten::mm" for event in trace["traceEvents"])
        assert caplog.text.count("Wrote torch.profiler trace of learn call 2") == 1
        assert "aten::mm" in caplog.text

    def test_rank_outside_default_ranks_writes_no_trace(self, tmp_path) -> None:
        config = ProfilingConfig(output_dir=str(tmp_path), torch_profile_step=1)
        profiler = LearnProfiler(config, rank=5, world_size=16, shard_group_size=8)

        run_learn_calls(profiler, calls=1, micro_batches=1)

        assert list(tmp_path.iterdir()) == []

    def test_explicit_profile_ranks_select_the_traced_rank(self, tmp_path) -> None:
        config = ProfilingConfig(
            output_dir=str(tmp_path), torch_profile_step=1, profile_ranks=[5]
        )
        profiler = LearnProfiler(config, rank=5, world_size=16, shard_group_size=8)

        run_learn_calls(profiler, calls=1, micro_batches=1)

        assert [path.name for path in tmp_path.iterdir()] == ["rank5_learn1_trace.json"]

    def test_no_config_traces_nothing(self, tmp_path) -> None:
        profiler = LearnProfiler(None, rank=0, world_size=1, shard_group_size=None)

        run_learn_calls(profiler, calls=2, micro_batches=2)

        assert profiler.learn_calls == 2
        assert list(tmp_path.iterdir()) == []
