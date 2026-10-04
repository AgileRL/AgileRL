# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA memory snapshots and torch.profiler traces for LLM ``learn`` calls."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.profiler import ProfilerActivity, profile

from agilerl.arena.models.profiling import ProfilingConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


def default_profile_ranks(
    world_size: int, shard_group_size: int | None
) -> frozenset[int]:
    """Rank 0, plus the first rank of the second FSDP shard group when one exists.

    :param world_size: Trainer ranks.
    :type world_size: int
    :param shard_group_size: Ranks per weight-shard group; ``None`` is the world.
    :type shard_group_size: int | None
    :return: Ranks that record the profiler trace.
    :rtype: frozenset[int]
    """
    if shard_group_size is None or shard_group_size >= world_size:
        return frozenset({0})
    return frozenset({0, shard_group_size})


class LearnProfiler:
    """Dump a CUDA allocator snapshot on OOM and trace one learn micro-batch."""

    def __init__(
        self,
        config: ProfilingConfig | None,
        *,
        rank: int,
        world_size: int,
        shard_group_size: int | None,
    ) -> None:
        """Start CUDA allocator history when snapshots are on.

        :param config: Profiling settings; ``None`` profiles nothing.
        :type config: ProfilingConfig | None
        :param rank: This process's trainer rank.
        :type rank: int
        :param world_size: Trainer ranks.
        :type world_size: int
        :param shard_group_size: Ranks per FSDP weight-shard group; ``None`` is the world.
        :type shard_group_size: int | None
        :raises ValueError: If memory snapshots are requested without CUDA.
        """
        self.rank = rank
        self.learn_calls = 0
        self.snapshot_dir: Path | None = None
        self.trace_dir: Path | None = None
        self.trace_step: int | None = None
        self.trace_with_stack = False
        self.pending_trace: Path | None = None
        if config is None or config.output_dir is None:
            return
        output_dir = Path(config.output_dir)
        if config.memory_snapshot_on_oom:
            if not torch.cuda.is_available():
                msg = "ProfilingConfig.memory_snapshot_on_oom needs CUDA"
                raise ValueError(msg)
            torch.cuda.memory._record_memory_history(
                max_entries=config.memory_history_max_entries,
                stacks="python",
            )
            self.snapshot_dir = output_dir
        trace_ranks = (
            frozenset(config.profile_ranks)
            if config.profile_ranks is not None
            else default_profile_ranks(world_size, shard_group_size)
        )
        if config.torch_profile_step is not None and rank in trace_ranks:
            self.trace_dir = output_dir
            self.trace_step = config.torch_profile_step
            self.trace_with_stack = config.torch_profile_with_stack

    @contextmanager
    def learn_step(self) -> Iterator[None]:
        """Count one learn call; on CUDA OOM dump the allocator snapshot and re-raise."""
        self.learn_calls += 1
        if self.trace_dir is not None and self.learn_calls == self.trace_step:
            self.pending_trace = (
                self.trace_dir / f"rank{self.rank}_learn{self.learn_calls}_trace.json"
            )
        try:
            yield
        except torch.OutOfMemoryError:
            if self.snapshot_dir is not None:
                self._dump_memory_snapshot(self.snapshot_dir)
            raise
        finally:
            self.pending_trace = None

    @contextmanager
    def micro_batch(self) -> Iterator[None]:
        """Trace this micro-batch when it is the first one of the profiled learn call."""
        path = self.pending_trace
        if path is None:
            yield
            return
        self.pending_trace = None
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=False,
            with_stack=self.trace_with_stack,
        ) as prof:
            yield
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))
        logger.info(
            "Wrote torch.profiler trace of learn call %d on rank %d to %s\n%s",
            self.learn_calls,
            self.rank,
            path,
            prof.key_averages().table(sort_by="device_time_total", row_limit=30),
        )

    def _dump_memory_snapshot(self, directory: Path) -> None:
        """Write the CUDA allocator snapshot and log a one-line memory summary.

        :param directory: Directory the snapshot is written to.
        :type directory: Path
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = (
            directory / f"rank{self.rank}_learn{self.learn_calls}_oom_snapshot.pickle"
        )
        torch.cuda.memory._dump_snapshot(str(path))
        stats = torch.cuda.memory_stats()
        gib = 1024**3
        logger.error(
            "CUDA OOM in learn call %d on rank %d, snapshot at %s: "
            "allocated=%.2f GiB active=%.2f GiB reserved=%.2f GiB "
            "inactive_split=%.2f GiB alloc_retries=%d",
            self.learn_calls,
            self.rank,
            path,
            stats["allocated_bytes.all.current"] / gib,
            stats["active_bytes.all.current"] / gib,
            stats["reserved_bytes.all.current"] / gib,
            stats["inactive_split_bytes.all.current"] / gib,
            stats["num_alloc_retries"],
        )
