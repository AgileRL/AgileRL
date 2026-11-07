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

# One merged round's control loop: the trainer RPC to every rank, merge and
# barrier, then each engine's load. 22 ms on A100 and H100 single-turn runs;
# 45 ms while multi-turn engines answer between turns.
MERGED_SYNC_ROUND_SECONDS = 0.022
MULTI_TURN_MERGED_SYNC_ROUND_SECONDS = 0.045
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
    ranks; every rank shards it on its own GPU.

    :param model: Model the trainers and engines run.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param multi_turn: Episodes run more than one turn.
    :param link: Link the engines read trainer rank 0 over.
    :return: The pause, every round's control loop and :func:`engine_receive_seconds`.
    """
    total, largest = merged_lora_sync_bytes(
        model.arch, scope, n_packed_matrices, DTYPE_BYTES[model.checkpoint_dtype]
    )
    rounds = math.ceil(total / merged_sync_bucket_bytes(total, largest))
    per_round = (
        MULTI_TURN_MERGED_SYNC_ROUND_SECONDS
        if multi_turn
        else MERGED_SYNC_ROUND_SECONDS
    )
    return (
        MERGED_SYNC_PAUSE_SECONDS
        + rounds * per_round
        + engine_receive_seconds(total, engines, engine_width, link)
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

    Each engine's TP rank 0 reads the whole adapter and broadcasts it to the
    engine's other ranks.

    :param model: Model the trainers and engines run.
    :param rank: LoRA rank.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param link: Link the engines read trainer rank 0 over.
    :return: The engine RPCs and :func:`engine_receive_seconds`.
    """
    adapter = lora_sync_buffer_bytes(
        model.arch, rank, scope, n_packed_matrices, DTYPE_BYTES[model.checkpoint_dtype]
    )
    return ADAPTER_SYNC_SECONDS + engine_receive_seconds(
        adapter, engines, engine_width, link
    )
