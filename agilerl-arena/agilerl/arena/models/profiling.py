# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Opt-in learn-step profiling settings for LLM training (torch-free)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field


@dataclass
class ProfilingConfig:
    """Settings for CUDA memory snapshots and a one-micro-batch profiler trace in ``learn``."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    output_dir: Annotated[
        str | None,
        Field(
            description=(
                "Directory each trainer rank writes snapshots and traces to. "
                "Required when any profiling is on."
            ),
        ),
    ] = None
    memory_snapshot_on_oom: Annotated[
        bool,
        Field(
            description=(
                "Record CUDA allocator history on every trainer rank and dump a "
                "snapshot when learn runs out of GPU memory."
            ),
        ),
    ] = False
    memory_history_max_entries: Annotated[
        int,
        Field(
            ge=1,
            description="Allocator events kept in the recorded history ring buffer.",
        ),
    ] = 100_000
    torch_profile_step: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Learn call, counted from 1 on each trainer process, whose first "
                "micro-batch runs under torch.profiler. None traces nothing."
            ),
        ),
    ] = None
    torch_profile_with_stack: Annotated[
        bool,
        Field(description="Record Python stacks in the profiler trace."),
    ] = False
    profile_ranks: Annotated[
        list[int] | None,
        Field(
            description=(
                "Trainer ranks that record the profiler trace. None picks rank 0 "
                "and the first rank of the second FSDP shard group."
            ),
        ),
    ] = None

    def __post_init__(self) -> None:
        if self.memory_history_max_entries < 1:
            msg = "ProfilingConfig.memory_history_max_entries must be >= 1"
            raise ValueError(msg)
        if self.torch_profile_step is not None and self.torch_profile_step < 1:
            msg = "ProfilingConfig.torch_profile_step must be >= 1"
            raise ValueError(msg)
        if self.profile_ranks is not None and any(
            rank < 0 for rank in self.profile_ranks
        ):
            msg = "ProfilingConfig.profile_ranks must be non-negative"
            raise ValueError(msg)
        enabled = self.memory_snapshot_on_oom or self.torch_profile_step is not None
        if enabled and self.output_dir is None:
            msg = (
                "ProfilingConfig.output_dir is required when memory_snapshot_on_oom "
                "or torch_profile_step is set"
            )
            raise ValueError(msg)
