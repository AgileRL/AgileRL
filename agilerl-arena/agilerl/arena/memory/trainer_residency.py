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
    :param shards: GPUs one weight copy is sharded across.
    :param activations: Activation peak bytes.
    :return: Base-weight bytes on one GPU.
    """
    full = formulas.weight_bytes(counts, settings.weight_dtype, model.variant("base"))
    fsdp = settings.fsdp
    if fsdp is None:
        return float(full)
    arch = model.arch
    token = min(formulas.token_embedding_params(arch), max(counts.total, 0))
    tp_params = 0 if fsdp.tp <= 1 else formulas.tensor_parallel_params(counts, arch)
    # Experts on an EP group that fills the shard group skip FSDP.
    ep_params = (
        arch.moe_layers * formulas.routed_expert_params_per_layer(arch)
        if fsdp.ep > 1 and fsdp.ep >= shards
        else 0
    )
    sharded_params = max(counts.total - token - tp_params - ep_params, 0)
    gathered = fsdp_gather.fsdp_gathered_params(
        counts,
        arch,
        n_gpus=shards,
        reshard_after_forward=fsdp.reshard_after_forward,
        tensor_parallel=fsdp.tp,
        expert_parallel=fsdp.ep,
    )
    byte = DTYPE_BYTES[settings.weight_dtype]
    resident = 0.0 if fsdp.cpu_offload else sharded_params * byte / shards
    # TP linears and FSDP-free experts are not FSDP parameters, so cpu
    # offload does not take them.
    resident += tp_params * byte / fsdp.tp + ep_params * byte / fsdp.ep
    # An untied lm_head and per-layer embeddings are gathered outside the
    # decoder backward, so they add only what they exceed activations by.
    side = 0
    if fsdp.reshard_after_forward:
        side = min(fsdp_gather.fsdp_resharded_side_params(counts, arch), gathered)
    side_at_peak = max(side * byte - activations, 0.0)
    return token * byte + resident + (gathered - side) * byte + side_at_peak


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
    :param shards: GPUs one weight copy is sharded across.
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
        shards,
        fsdp.tp,
    )
    return reserve, pool * DTYPE_BYTES[settings.weight_dtype]
