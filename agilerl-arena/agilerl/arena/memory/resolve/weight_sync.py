# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Seconds one async weight sync takes, per sync mode and trainer-to-engine link."""

from __future__ import annotations

import math
from typing import Literal

from agilerl.arena.memory.formulas import merged_lora_sync_bytes
from agilerl.arena.memory.lora_formulas import lora_sync_buffer_bytes
from agilerl.arena.memory.resolve.rollout import merged_sync_bucket_bytes
from agilerl.arena.memory.specs import DTYPE_BYTES, LoraTargetScope, ModelSpec

SyncLink = Literal["cuda_ipc", "rdma", "tcp"]
"""How NIXL reads trainer rank 0's send buffer: NVLink or PCIe peer access, GPUDirect RDMA, or UCX over TCP."""

LINK_EGRESS_BYTES_PER_SECOND: dict[SyncLink, float] = {
    # NIXL READs out of one A100 trainer GPU over NVSwitch, 4-7 engines at once.
    "cuda_ipc": 275e9,
    # One 100 Gb/s NIC; not measured.
    "rdma": 12.5e9,
    # UCX software-emulated GET over TCP, measured on pod loopback.
    "tcp": 0.4e9,
}
"""Bytes per second all engines together read out of trainer rank 0."""

# TP rank 0's NCCL broadcast of the received buffer to its engine's other TP
# ranks over NVLink; not measured, below A100 and H100 NVSwitch bus bandwidth.
NVLINK_BROADCAST_BYTES_PER_SECOND = 100e9

# One merged round's control loop, trainer side, barrier and engine side
# together: 22 ms on A100 and H100 single-turn runs, 45 ms while multi-turn
# engines answer between turns. The split is an estimate pending a GPU
# measurement of each side: the trainer side (RPC to every rank, gather and
# merge) and the engine side (RPC to the engine and its workers, load) take
# 10 ms each and the barrier 2 ms; multi-turn engines add their 23 ms to the
# engine side.
MERGED_SYNC_TRAINER_ROUND_SECONDS = 0.010
MERGED_SYNC_ENGINE_ROUND_SECONDS = 0.010
MULTI_TURN_MERGED_SYNC_ENGINE_ROUND_SECONDS = 0.033
MERGED_SYNC_BARRIER_SECONDS = 0.002
# Pausing busy engines before a merged sync's first round: about 50 ms on
# H100 countdown.
MERGED_SYNC_PAUSE_SECONDS = 0.05
# An adapter sync's engine RPCs while the engines generate: 0.14 s median
# on H100 countdown.
ADAPTER_SYNC_SECONDS = 0.14


def engine_receive_seconds(
    nbytes: int, engines: int, engine_width: int, link: SyncLink
) -> float:
    """Seconds for each engine's TP rank 0 to READ ``nbytes`` and broadcast them to its other ranks.

    :param nbytes: Bytes every engine receives.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param link: Link the engines read trainer rank 0 over.
    :return: The bytes over the link, then over NVLink inside each engine.
    """
    wire = engines * nbytes / LINK_EGRESS_BYTES_PER_SECOND[link]
    if engine_width == 1:
        return wire
    return wire + nbytes / NVLINK_BROADCAST_BYTES_PER_SECOND


def merged_sync_seconds(
    model: ModelSpec,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    engines: int,
    engine_width: int,
    multi_turn: bool,
    link: SyncLink,
) -> float:
    """Seconds a merged-LoRA sync blocks the trainer and pauses the engines.

    Each engine's TP rank 0 reads every merged tensor, in rounds of
    :func:`merged_sync_bucket_bytes`, and broadcasts it to the engine's other
    ranks; every rank shards it on its own GPU. The engines receive round
    ``k`` while the trainers merge round ``k + 1``, so after the first merge
    each round costs its slower side. The last round's merge ends with a
    barrier.

    :param model: Model the trainers and engines run.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param multi_turn: Episodes run more than one turn.
    :param link: Link the engines read trainer rank 0 over.
    :return: The pause, the first merge, the slower side of every later
        round, the last receive and the barrier.
    """
    total, largest = merged_lora_sync_bytes(
        model.arch, scope, n_packed_matrices, DTYPE_BYTES[model.checkpoint_dtype]
    )
    rounds = math.ceil(total / merged_sync_bucket_bytes(total, largest))
    trainer_round = MERGED_SYNC_TRAINER_ROUND_SECONDS
    engine_round = (
        MULTI_TURN_MERGED_SYNC_ENGINE_ROUND_SECONDS
        if multi_turn
        else MERGED_SYNC_ENGINE_ROUND_SECONDS
    ) + engine_receive_seconds(total, engines, engine_width, link) / rounds
    return (
        MERGED_SYNC_PAUSE_SECONDS
        + trainer_round
        + (rounds - 1) * max(trainer_round, engine_round)
        + engine_round
        + MERGED_SYNC_BARRIER_SECONDS
    )


def adapter_sync_seconds(
    model: ModelSpec,
    rank: int,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    engines: int,
    engine_width: int,
    link: SyncLink,
) -> float:
    """Seconds an adapter sync blocks the trainer; the engines keep generating.

    Every engine rank reads the whole adapter.

    :param model: Model the trainers and engines run.
    :param rank: LoRA rank.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param link: Link the engines read trainer rank 0 over.
    :return: The engine RPCs and the bytes over the link.
    """
    adapter = lora_sync_buffer_bytes(
        model.arch, rank, scope, n_packed_matrices, DTYPE_BYTES[model.checkpoint_dtype]
    )
    wire = engines * engine_width * adapter / LINK_EGRESS_BYTES_PER_SECOND[link]
    return ADAPTER_SYNC_SECONDS + wire
