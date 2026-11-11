# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Seconds one async weight sync takes, per sync mode and trainer-to-engine link."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agilerl.arena.memory.lora_formulas import (
    NcclReshardLayout,
    lora_adapter_tensor_count,
    lora_sync_buffer_bytes,
    merged_lora_sync_tensor_bytes,
    nccl_reshard_move,
    nccl_reshard_sync_linears,
)
from agilerl.arena.memory.manifest import GPU_CATALOGUE
from agilerl.arena.memory.resolve.rollout import merged_sync_bucket_bytes
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GiB,
    LoraTargetScope,
    ModelSpec,
)

WeightSyncTransport = Literal["nixl", "nccl_reshard"]
"""How async rollout copies trainer weights onto the engines, as the planner picks it."""

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

# TP rank 0's NCCL broadcast of a received round to its engine's other TP
# ranks, then vLLM's load, as a share of the GPU's all-reduce bus bandwidth:
# 0.43 on A100 and 0.25-0.44 on H100, merged NIXL at TP 2 on Qwen3.5-2B to
# Qwen3.8-27B.
ENGINE_LOAD_ALL_REDUCE_SHARE = 0.4

# A merged round's fixed costs: the trainer side's RPC to every rank and DDP
# merge, and the engine side's RPC to the engine and its workers. Fitted with
# the pause and barrier below to merged NIXL syncs on GuessTheNumber (2 turns):
# Qwen3.5-2B, 4B, 9B, Qwen3.8-27B and Qwen3.6-35B-A3B on 8x H100 (4 trainers,
# 2 engines at TP 2, 11-256 rounds), within 11% of each, and Qwen3.5-4B on
# 4x A100 (2 trainers, 1 engine at TP 2), 6% low.
MERGED_SYNC_TRAINER_ROUND_SECONDS = 0.0055
MERGED_SYNC_ENGINE_ROUND_SECONDS = 0.0065
MERGED_SYNC_BARRIER_SECONDS = 0.002
# Pausing busy engines before a merged sync's first round: about 50 ms on
# H100 countdown.
MERGED_SYNC_PAUSE_SECONDS = 0.05
# Gathering one FSDP-sharded weight whole on every trainer rank before it
# merges: Qwen3.5-9B and Qwen3.6-35B-A3B merged NIXL syncs on 4 H100 trainers.
FSDP_GATHER_SECONDS_PER_TENSOR = 0.00126
# An adapter sync's engine RPCs, and the engines' swap of the adapter into
# the served slot: DDP adapter syncs of Qwen3.5-2B (0.109 s), 4B (0.141 s) and
# Qwen3.8-27B (0.328 s) on 8x H100, 2 engines at TP 2, within 2% of each.
ADAPTER_SYNC_SECONDS = 0.072
ADAPTER_SWAP_BYTES_PER_SECOND = 0.91e9
# Gathering one FSDP-sharded LoRA factor whole: Qwen3.5-9B (0.294 s) and
# Qwen3.6-35B-A3B (0.512 s) adapter syncs on 4 H100 trainers, priced 22% high
# and 30% low.
ADAPTER_FSDP_GATHER_SECONDS_PER_TENSOR = 0.000385


def engine_receive_seconds(
    nbytes: int, engines: int, engine_width: int, device: DeviceSpec, link: SyncLink
) -> float:
    """Seconds for each engine's TP rank 0 to READ ``nbytes``, broadcast them to its other ranks and load them.

    :param nbytes: Bytes every engine receives.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param device: One GPU of the engine.
    :param link: Link the engines read trainer rank 0 over.
    :return: The bytes over the link, then at
        :data:`ENGINE_LOAD_ALL_REDUCE_SHARE` of the GPU's all-reduce bus
        bandwidth inside each engine.
    :raises ValueError: When a wider engine's GPU has no all-reduce bandwidth.
    """
    wire = engines * nbytes / LINK_EGRESS_BYTES_PER_SECOND[link]
    if engine_width == 1:
        return wire
    bus_bandwidth = device.all_reduce_bytes_per_second
    if bus_bandwidth is None:
        msg = (
            f"{device.name or 'This GPU'} has no all-reduce bandwidth in the GPU "
            "catalogue, so a tensor-parallel engine's merged sync cannot be priced."
        )
        raise ValueError(msg)
    return wire + nbytes / (ENGINE_LOAD_ALL_REDUCE_SHARE * bus_bandwidth)


def merged_sync_rounds(sizes: list[int]) -> int:
    """Rounds of :func:`merged_sync_bucket_bytes` the trainer packs ``sizes`` into, in order; no tensor straddles two.

    :param sizes: Bytes of each pushed tensor, in push order.
    :return: Round count.
    """
    bucket = merged_sync_bucket_bytes(sum(sizes), max(sizes))
    rounds, filled = 1, 0
    for size in sizes:
        if filled and filled + size > bucket:
            rounds += 1
            filled = 0
        filled += size
    return rounds


def merged_sync_seconds(
    model: ModelSpec,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    engines: int,
    engine_width: int,
    trainer_fsdp: bool,
    device: DeviceSpec,
    link: SyncLink,
) -> float:
    """Seconds a merged-LoRA sync blocks the trainer and pauses the engines.

    Each engine's TP rank 0 reads every merged tensor, in the
    :func:`merged_sync_rounds` the trainer packs them into, and broadcasts it
    to the engine's other ranks; every rank shards it on its own GPU. The
    engines receive round ``k`` while the trainers merge round ``k + 1``, so
    after the first merge each round costs its slower side. The last round's
    merge ends with a barrier.

    :param model: Model the trainers and engines run.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param trainer_fsdp: Trainers shard weights with FSDP, so each merged
        weight is gathered first.
    :param device: One GPU of the engine.
    :param link: Link the engines read trainer rank 0 over.
    :return: The pause, the first merge, the slower side of every later
        round, the last receive and the barrier.
    """
    sizes = merged_lora_sync_tensor_bytes(
        model.arch, scope, n_packed_matrices, DTYPE_BYTES[model.checkpoint_dtype]
    )
    rounds = merged_sync_rounds(sizes)
    trainer_round = MERGED_SYNC_TRAINER_ROUND_SECONDS
    if trainer_fsdp:
        trainer_round += FSDP_GATHER_SECONDS_PER_TENSOR * len(sizes) / rounds
    engine_round = (
        MERGED_SYNC_ENGINE_ROUND_SECONDS
        + engine_receive_seconds(sum(sizes), engines, engine_width, device, link)
        / rounds
    )
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
    trainer_fsdp: bool,
    link: SyncLink,
) -> float:
    """Seconds an adapter sync blocks the trainer; the engines keep generating.

    Every engine rank reads the whole adapter, and every engine swaps it in.

    :param model: Model the trainers and engines run.
    :param rank: LoRA rank.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param trainer_fsdp: Trainers shard the adapter with FSDP, so each LoRA
        factor is gathered first.
    :param link: Link the engines read trainer rank 0 over.
    :return: The engine RPCs, the gathers, the swap and the bytes over the link.
    """
    adapter = lora_sync_buffer_bytes(
        model.arch, rank, scope, n_packed_matrices, DTYPE_BYTES[model.checkpoint_dtype]
    )
    wire = engines * engine_width * adapter / LINK_EGRESS_BYTES_PER_SECOND[link]
    gathers = (
        ADAPTER_FSDP_GATHER_SECONDS_PER_TENSOR
        * lora_adapter_tensor_count(model.arch, scope, n_packed_matrices)
        if trainer_fsdp
        else 0.0
    )
    return (
        ADAPTER_SYNC_SECONDS + gathers + adapter / ADAPTER_SWAP_BYTES_PER_SECOND + wire
    )


@dataclass(frozen=True)
class NcclReshardCosts:
    """``nccl_reshard`` refit rates on one NVLink GPU class.

    Bulk terms are seconds per GiB of dense FFN and decoder attention
    parameters. The control term is a full refit's fixed cost; misc parameters
    move at ``misc_bytes_per_second`` per engine.
    """

    # Per GiB per engine, not per parameter.
    bulk_engine_seconds_per_gib: float
    # Per GiB per engine, divided by the engine's TP width.
    bulk_rank_seconds_per_gib: float
    control_seconds: float
    misc_bytes_per_second: float


SM80_NCCL_RESHARD_COSTS = NcclReshardCosts(
    # One ungrouped reshard of a 0.984 GiB FFN block, socket NCCL: 1 engine at
    # TP 2 takes 4.0 ms, 2 engines at TP 1 take 9.8 ms.
    bulk_engine_seconds_per_gib=0.00315,
    bulk_rank_seconds_per_gib=0.00183,
    # Fit to Qwen3.5-0.8B (0.1485 s) and Qwen3.5-9B (0.306 s) full refits on
    # 4x A100, 2 DDP trainers, 1 engine TP 2, after subtracting the bulk term.
    # How both scale past one engine is not measured.
    control_seconds=0.1303,
    misc_bytes_per_second=2.385e10,
)
"""A100-SXM4-80GB."""

SM90_NCCL_RESHARD_COSTS = NcclReshardCosts(
    # One ungrouped reshard of a 0.984 GiB FFN block over NVSwitch, least
    # squares on relative error: 1 engine at TP 2 takes 2.74 ms (fit -11%),
    # 2 engines at TP 2 4.50 ms (+9%), 1 engine at TP 4 1.88 ms (0%).
    bulk_engine_seconds_per_gib=0.00133,
    bulk_rank_seconds_per_gib=0.00231,
    # Fit to Qwen3.5-2B (0.111 s), 4B (0.179 s) and Qwen3.8-27B (0.675 s) full
    # refits on 8x H100, 4 DDP trainers, 2 engines at TP 2, after subtracting
    # the bulk term: within 2% of each.
    control_seconds=0.0734,
    misc_bytes_per_second=5.135e10,
)
"""H100 80GB HBM3. H200 has the same NVLink and NVSwitch generation."""

SM80_NCCL_RESHARD_GPU = dict(GPU_CATALOGUE)["A100-80"].device_spec()
"""The GPU :data:`SM80_NCCL_RESHARD_COSTS` was measured on."""

NCCL_RESHARD_COSTS_BY_CAPABILITY: dict[tuple[int, int], NcclReshardCosts] = {
    (8, 0): SM80_NCCL_RESHARD_COSTS,
    (9, 0): SM90_NCCL_RESHARD_COSTS,
}
"""Measured ``nccl_reshard`` rates by compute capability."""


def nccl_reshard_costs(device: DeviceSpec) -> NcclReshardCosts | None:
    """``nccl_reshard`` rates for a GPU.

    An exact compute-capability row wins. Hopper and newer with no row use
    the SM90 entry. Below Hopper, the SM80 bulk and misc rates scale by the
    device's all-reduce bus bandwidth over A100's; the control term stays.

    :param device: The training and rollout GPU.
    :return: The rates, or ``None`` off NVLink or when a GPU below Hopper
        has no all-reduce bandwidth.
    """
    # nccl.m2n has only run over NVLink; a failed refit crashes the run.
    if not device.nvlink:
        return None
    capability = device.compute_capability
    measured = (
        None if capability is None else NCCL_RESHARD_COSTS_BY_CAPABILITY.get(capability)
    )
    if measured is not None:
        return measured
    # No measured row: Hopper and newer use the SM90 rates.
    if capability is not None and capability >= (9, 0):
        return NCCL_RESHARD_COSTS_BY_CAPABILITY[(9, 0)]
    bandwidth = device.all_reduce_bytes_per_second
    reference = SM80_NCCL_RESHARD_GPU.all_reduce_bytes_per_second
    if bandwidth is None or reference is None:
        return None
    slowdown = reference / bandwidth
    return NcclReshardCosts(
        bulk_engine_seconds_per_gib=SM80_NCCL_RESHARD_COSTS.bulk_engine_seconds_per_gib
        * slowdown,
        bulk_rank_seconds_per_gib=SM80_NCCL_RESHARD_COSTS.bulk_rank_seconds_per_gib
        * slowdown,
        control_seconds=SM80_NCCL_RESHARD_COSTS.control_seconds,
        misc_bytes_per_second=SM80_NCCL_RESHARD_COSTS.misc_bytes_per_second / slowdown,
    )


def nccl_reshard_bulk_seconds(
    bulk_bytes: int, engines: int, engine_width: int, costs: NcclReshardCosts
) -> float:
    """Seconds the bulk parameters of one ``nccl_reshard`` refit take.

    :param bulk_bytes: Bytes of the dense FFN and decoder attention parameters.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param costs: Rates of the GPU class.
    :return: The microbenchmark bandwidth term, with no control cost.
    """
    per_engine = (
        costs.bulk_engine_seconds_per_gib
        + costs.bulk_rank_seconds_per_gib / engine_width
    )
    return engines * bulk_bytes / GiB * per_engine


def nccl_reshard_sync_seconds(
    model: ModelSpec,
    scope: LoraTargetScope,
    engines: int,
    engine_width: int,
    trainer_shards: int,
    trainer_fsdp: bool,
    costs: NcclReshardCosts,
) -> float:
    """Seconds one merged-LoRA ``nccl_reshard`` refit takes.

    Bulk parameters move at :func:`nccl_reshard_bulk_seconds`. Misc parameters
    move at ``costs.misc_bytes_per_second`` per engine, each gathered whole
    first under FSDP. The control term does not scale with the model.

    :param model: Model the trainers and engines run.
    :param scope: Adapter target scope.
    :param engines: Rollout engines per member.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param trainer_shards: FSDP shards that each hold a slice of a bulk parameter.
    :param trainer_fsdp: Trainers shard weights with FSDP.
    :param costs: Rates of the GPU class.
    :return: Control plus bulk plus misc.
    """
    layout = NcclReshardLayout(trainer_shards, engine_width)
    move = nccl_reshard_move(
        model.arch, scope, layout, DTYPE_BYTES[model.checkpoint_dtype]
    )
    gathers = 0.0
    if trainer_fsdp:
        _, misc = nccl_reshard_sync_linears(model.arch, scope, layout)
        gathers = FSDP_GATHER_SECONDS_PER_TENSOR * sum(linear.count for linear in misc)
    return (
        costs.control_seconds
        + nccl_reshard_bulk_seconds(move.bulk_bytes, engines, engine_width, costs)
        + engines * move.misc_bytes / costs.misc_bytes_per_second
        + gathers
    )
