# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Frozen base weights and caching-allocator slack on one training GPU."""

from __future__ import annotations

import agilerl.arena.memory.formulas as formulas
import agilerl.arena.memory.fsdp_gather as fsdp_gather
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    ModelArch,
    ModelSpec,
    TrainingSettings,
)
from agilerl.arena.models.fsdp import FSDPConfig


def trainer_base_bytes(
    model: ModelSpec,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
    activations: float,
) -> float:
    """Frozen base weights on one training GPU at the activation peak.

    :param model: Training-side model spec.
    :param settings: Training settings (dtype, FSDP).
    :param counts: Analytic parameter split.
    :param shards: GPUs one dense weight copy is sharded across.
    :param activations: Activation peak bytes.
    :return: Base-weight bytes on one GPU.
    """
    full = formulas.weight_bytes(counts, settings.weight_dtype, model.variant("base"))
    fsdp = settings.fsdp
    if fsdp is None:
        return float(full)
    arch = model.arch
    token, ep_params, expert_fsdp_params, sharded_params, tp_resident = (
        _fsdp_base_groups(arch, counts, fsdp, settings.shard_group_gpus)
    )
    gathered = fsdp_gather.fsdp_gathered_params(
        counts,
        arch,
        n_gpus=shards,
        reshard_after_forward=fsdp.reshard_after_forward,
        tensor_parallel=fsdp.tp,
        expert_parallel=fsdp.ep,
        expert_gpus=settings.expert_shard_gpus,
        frozen_tp_resident=fsdp.frozen_tp_resident,
        expert_lora=settings.lora_packed_target_matrices > 0,
    )
    byte = DTYPE_BYTES[settings.weight_dtype]
    resident = (
        0.0
        if fsdp.cpu_offload
        else _fsdp_shard_bytes(settings, sharded_params, expert_fsdp_params, shards)
    )
    # FSDP-free experts and resident TP shards are not FSDP parameters, so
    # cpu offload does not take them.
    resident += ep_params * byte / fsdp.ep + tp_resident * byte
    resident += formulas.mamba_shared_group_rows_params(arch, fsdp.tp) * byte
    # An untied lm_head and per-layer embeddings are gathered outside the
    # decoder backward, so they add only what they exceed activations by.
    side = 0
    if fsdp.reshard_after_forward:
        side = min(
            fsdp_gather.fsdp_resharded_side_params(
                counts, arch, fsdp.tp, fsdp.frozen_tp_resident
            ),
            gathered,
        )
    side_at_peak = max(side * byte - activations, 0.0)
    token_shards = fsdp.tp if arch.tensor_parallel_tied_embedding else 1
    return (
        token * byte / token_shards + resident + (gathered - side) * byte + side_at_peak
    )


def fsdp_forward_gather_bytes(
    arch: ModelArch,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
) -> tuple[float, float]:
    """Retained all-gather output and cpu-offload staging live in a decoder forward.

    :param arch: Training-side architecture.
    :param settings: Training settings (dtype, FSDP).
    :param counts: Analytic parameter split.
    :param shards: GPUs one dense weight copy is sharded across.
    :return: Retained all-gather output bytes and host-to-device staging bytes.
    """
    fsdp = settings.fsdp
    if fsdp is None or shards <= 1:
        return 0.0, 0.0
    byte = DTYPE_BYTES[settings.weight_dtype]
    retained = fsdp_gather.fsdp_retained_gather_params(
        counts,
        arch,
        n_gpus=shards,
        tensor_parallel=fsdp.tp,
        expert_parallel=fsdp.ep,
        expert_gpus=settings.expert_shard_gpus,
        frozen_tp_resident=fsdp.frozen_tp_resident,
    )
    # CPUOffloadPolicy copies the prefetched unit's shard to the GPU before
    # its all-gather (FSDPParam.all_gather_inputs).
    staging = retained * byte / shards if fsdp.cpu_offload else 0.0
    return retained * byte, staging


def fsdp_cpu_offload_host_bytes(
    model: ModelSpec,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
) -> float:
    """FSDP parameter shard pinned on host when cpu_offload is on.

    :param model: Training-side model spec.
    :param settings: Training settings (dtype, FSDP).
    :param counts: Analytic parameter split.
    :param shards: GPUs one dense weight copy is sharded across.
    :return: Host bytes of the offloaded shard, or 0.
    """
    fsdp = settings.fsdp
    if fsdp is None or not fsdp.cpu_offload:
        return 0.0
    _, _, expert_fsdp_params, sharded_params, _ = _fsdp_base_groups(
        model.arch, counts, fsdp, settings.shard_group_gpus
    )
    return _fsdp_shard_bytes(settings, sharded_params, expert_fsdp_params, shards)


def _fsdp_base_groups(
    arch: ModelArch,
    counts: formulas.ParamCounts,
    fsdp: FSDPConfig,
    shard_group_gpus: int,
) -> tuple[int, int, int, int, int]:
    """Token, FSDP-free expert, FSDP-sharded expert, FSDP-sharded dense, and resident TP counts.

    Each rank holds 1/tp of the tensor-parallel weights. FSDP shards that
    share like any dense weight, unless frozen TP shards stay resident.
    """
    token = min(formulas.token_embedding_params(arch), max(counts.total, 0))
    tp_params = (
        0 if fsdp.tp <= 1 else formulas.tensor_parallel_params(counts, arch, fsdp.tp)
    )
    routed = (
        arch.moe_layers * formulas.routed_expert_params_per_layer(arch)
        if fsdp.ep > 1
        else 0
    )
    # Experts on an EP group that fills the shard group skip FSDP; otherwise
    # leftover FSDP shards them, so EP times leftover covers the shard group.
    ep_params = routed if fsdp.ep >= shard_group_gpus else 0
    tp_local = tp_params // fsdp.tp
    tp_resident = tp_local if fsdp.frozen_tp_resident else 0
    sharded_params = max(counts.total - token - tp_params - routed, 0) + (
        tp_local - tp_resident
    )
    return token, ep_params, routed - ep_params, sharded_params, tp_resident


def _fsdp_shard_bytes(
    settings: TrainingSettings,
    sharded_params: int,
    expert_fsdp_params: int,
    shards: int,
) -> float:
    """Bytes of this rank's FSDP shards: dense over ``shards``, experts over the shard group."""
    byte = DTYPE_BYTES[settings.weight_dtype]
    return (
        sharded_params * byte / shards
        + expert_fsdp_params * byte / settings.expert_shard_gpus
    )


def trainer_allocator_slack(
    arch: ModelArch,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
    torch_side: float,
) -> tuple[float, float]:
    """Compute-stream reserve and FSDP all-gather pool on one training GPU.

    :param arch: Training-side architecture.
    :param settings: Training settings (dtype, FSDP).
    :param counts: Analytic parameter split.
    :param shards: GPUs one dense weight copy is sharded across.
    :param torch_side: Bytes the trainer allocates through torch.
    :return: Compute-stream reserve bytes and all-gather pool bytes.
    """
    fsdp = settings.fsdp
    sharded = fsdp is not None and shards > 1
    reserve = formulas.allocator_reserve_bytes(
        torch_side,
        n_ranks=shards,
        sharded=sharded,
        mixed_sizes=arch.fp32_norm_inputs or (sharded and fsdp.ep > 1),
    )
    if fsdp is None or not sharded:
        return reserve, 0.0
    pool = fsdp_gather.fsdp_gather_pool_params(
        counts,
        arch,
        fsdp.ep,
        settings.expert_shard_gpus,
        fsdp.tp,
        fsdp.frozen_tp_resident,
    )
    return reserve, pool * DTYPE_BYTES[settings.weight_dtype]
