# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Learn-time constants per GPU class for the training-memory search."""

from __future__ import annotations

from dataclasses import dataclass

from agilerl.arena.memory.specs import DeviceSpec


@dataclass(frozen=True)
class LearnCosts:
    """Learn-time constants measured on one GPU class.

    Token costs are in tokens of uncheckpointed learn work on that GPU.
    """

    # Recomputing one checkpointed token.
    recompute_cost: float
    # Fixed cost of one micro-batch: launches, hooks and host work.
    micro_batch_overhead_tokens: float
    # Frozen-base all-gather of one micro-batch resharded after forward.
    reshard_gather_tokens: float
    # One routed-expert row chunk in one MoE layer: its kernel launches and the
    # expert-LoRA recompute's autograd pass in backward.
    routed_chunk_tokens: float


SM80_LEARN_COSTS = LearnCosts(
    recompute_cost=0.30,
    micro_batch_overhead_tokens=730,
    reshard_gather_tokens=1320,
    # granite-3.1-3b-a800m GRPO LoRA under DDP, GC off: 64 MiB chunks
    # (15 per layer) learn 12% slower than 512 MiB chunks (2 per layer).
    routed_chunk_tokens=12,
)
"""A100-80; also used for GPUs with no measured entry."""

SM90_LEARN_COSTS = LearnCosts(
    recompute_cost=0.255,
    micro_batch_overhead_tokens=7000,
    reshard_gather_tokens=1170,
    # The A100 chunk time at this GPU's token rate.
    routed_chunk_tokens=15,
)
"""H100 80GB HBM3: a micro-batch's host work is about 0.5 s, so fewer, larger micro-batches win."""

LEARN_COSTS_BY_CAPABILITY: dict[tuple[int, int], LearnCosts] = {
    (8, 0): SM80_LEARN_COSTS,
    (9, 0): SM90_LEARN_COSTS,
}


def learn_costs(device: DeviceSpec) -> LearnCosts:
    """Learn-time constants for a training GPU.

    :param device: The training GPU.
    :return: The entry for its compute capability, or the A100 entry when none is measured.
    """
    if device.compute_capability is None:
        return SM80_LEARN_COSTS
    return LEARN_COSTS_BY_CAPABILITY.get(device.compute_capability, SM80_LEARN_COSTS)
