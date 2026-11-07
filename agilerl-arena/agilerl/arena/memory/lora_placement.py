# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""LoRA tensor placement and split expert-LoRA recompute bytes."""

from __future__ import annotations

from collections.abc import Collection

from agilerl.arena.memory.lora_formulas import decoder_lora_linears, tower_lora_linears
from agilerl.arena.memory.specs import (
    LoraLinear,
    LoraTargetScope,
    ModelArch,
    PackedMoeDispatch,
)


def split_moe_lora_recompute_bytes(
    arch: ModelArch,
    grad_rows: int,
    seq_len: int,
    packed_matrices: int,
    dispatch: PackedMoeDispatch,
    act_bytes: float,
    lora_rank: int,
) -> int:
    """Split packed-expert LoRA tensors live during a checkpointed backward.

    The delta runs at the activation dtype and is added into the expert
    output in place, so each targeted matrix keeps only ``x @ A``: ``r``
    elements per routed token. Zero when dispatch is materialized (PEFT
    builds ``W_eff``) or nothing targets packed experts.
    """
    if dispatch != "contracted" or not packed_matrices or not arch.is_moe:
        return 0
    routed = grad_rows * seq_len * (arch.n_experts_per_tok or 1)
    return int(routed * lora_rank * packed_matrices * act_bytes)


def _bin_lora_numel(numel: int, threshold: int | None) -> tuple[int, int]:
    """Split one LoRA matrix into (replicated, sharded) by FSDP's threshold."""
    if numel <= 0:
        return 0, 0
    if threshold is None or numel < threshold:
        return numel, 0
    return 0, numel


def _add_lora_pair(
    replicated: int,
    sharded: int,
    in_features: int,
    out_features: int,
    rank: int,
    copies: int,
    threshold: int | None,
    in_tp: int = 1,
    out_tp: int = 1,
) -> tuple[int, int]:
    """Add both factors of one wrapped linear into the replicated/sharded totals.

    ``in_tp`` / ``out_tp`` divide LoRA A / LoRA B when tensor parallel splits
    them. FSDP bins each factor by its global size.
    """
    for features, tp in ((in_features, in_tp), (out_features, out_tp)):
        kept, split = _bin_lora_numel(rank * features, threshold)
        replicated += kept // tp * copies
        sharded += split // tp * copies
    return replicated, sharded


def _linear_lora_placement(
    linears: list[LoraLinear], rank: int, threshold: int | None, tp: int
) -> tuple[int, int]:
    """(replicated, sharded) elements of every wrapped factor on one TP rank."""
    replicated, sharded = 0, 0
    for linear in linears:
        replicated, sharded = _add_lora_pair(
            replicated,
            sharded,
            linear.in_features,
            linear.out_features,
            rank,
            linear.count,
            threshold,
            in_tp=tp if linear.split == "in" else 1,
            out_tp=tp if linear.split == "out" else 1,
        )
    return replicated, sharded


def lora_tensor_placement(
    arch: ModelArch,
    rank: int,
    scope: LoraTargetScope,
    packed_matrices: int,
    threshold: int | None,
    tensor_parallel: int = 1,
    tower_targets: Collection[str] = (),
) -> tuple[int, int]:
    """(replicated elements, sharded elements) for one LoRA adapter on one rank.

    ``threshold`` is FSDP's parameter persistence threshold: a matrix with
    fewer elements stays replicated. ``None`` is flat data parallel, so
    every tensor is replicated. ``tensor_parallel`` is the degree the
    trainer's plan shards by: each rank keeps ``1/tp`` of the LoRA factor
    :func:`decoder_lora_linears` and :attr:`ModelArch.tower_lora_linears`
    mark split. ``tower_targets`` adds the tower and connector adapters
    those module names wrap. At ``tensor_parallel=1`` the two counts sum to
    :func:`lora_param_count`, :func:`packed_lora_param_count` and
    :func:`tower_lora_param_count`.
    """
    linears = decoder_lora_linears(arch, scope, tensor_parallel)
    linears += tower_lora_linears(arch, tower_targets)
    replicated, sharded = _linear_lora_placement(
        linears, rank, threshold, tensor_parallel
    )
    n_experts = arch.n_experts
    if packed_matrices and n_experts is not None and n_experts > 1:
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        # PEFT stacks every expert's factor into one ``[E * r, features]`` tensor.
        # A gated stack's fused ``gate_up_proj`` is ``2 * inter`` wide.
        widths = [expert_inter] * packed_matrices
        if arch.gated_mlp:
            widths[0] = 2 * expert_inter
        for width in widths:
            replicated, sharded = _add_lora_pair(
                replicated,
                sharded,
                arch.expert_width,
                width,
                rank * n_experts,
                arch.moe_layers,
                threshold,
            )
    return replicated, sharded
