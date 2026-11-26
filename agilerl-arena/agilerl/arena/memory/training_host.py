# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Host RAM of one training process: its runtime, its offloads, and the bytes plans share."""

from __future__ import annotations

import agilerl.arena.memory.formulas as formulas
from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown


def process_host_components(
    offloaded_activations: float,
    offloaded_params: float,
) -> tuple[MemoryComponent, ...]:
    """The trainer process's runtime and the tensors it offloads."""
    return (
        MemoryComponent(
            key="trainer_process",
            label="Trainer process",
            n_bytes=formulas.TRAINER_PROCESS_HOST_BYTES,
            note="Python, torch and the CUDA runtime.",
        ),
        MemoryComponent(
            key="activation_offload",
            label="Offloaded activations (pinned)",
            n_bytes=int(offloaded_activations),
            note="Tensors one micro-batch's forward saves for backward.",
        ),
        MemoryComponent(
            key="parameter_offload",
            label="Offloaded FSDP shard (pinned)",
            n_bytes=int(offloaded_params),
            note="Sharded parameters CPUOffloadPolicy keeps on host.",
        ),
    )


def fixed_host_bytes(breakdown: PhaseBreakdown) -> int:
    """Host bytes of a training breakdown that no memory rung, checkpointing or micro-batch changes.

    The rest are set by the trainer count, the layout and the run's settings.

    :param breakdown: Training breakdown.
    :return: Bytes every plan at the same trainer count and layout holds on host.
    """
    searched = {
        "optimizer_offload",
        "activation_offload",
        "parameter_offload",
        "vision_row_staging",
    }
    return sum(part.n_bytes for part in breakdown.host if part.key not in searched)


def layout_free_host_bytes(breakdown: PhaseBreakdown) -> int:
    """Host bytes of a training breakdown that no layout, split or memory setting changes.

    The trainer process and the checkpoint snapshot, which holds the gathered
    trained adapters whatever the sharding.

    :param breakdown: Training breakdown.
    :return: Bytes every config of the run holds on host.
    """
    kept = {"trainer_process", "checkpoint_snapshot"}
    return sum(part.n_bytes for part in breakdown.host if part.key in kept)
