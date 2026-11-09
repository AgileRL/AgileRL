# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""LoRA tensor placement and split expert-LoRA recompute bytes."""

from __future__ import annotations

from agilerl.arena.memory.formulas import lora_param_count, packed_lora_param_count
from agilerl.arena.memory.specs import LoraTargetScope, ModelArch, PackedMoeDispatch


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
) -> tuple[int, int]:
    """Add both factors of one wrapped linear into the replicated/sharded totals."""
    for features in (in_features, out_features):
        kept, split = _bin_lora_numel(rank * features, threshold)
        replicated += kept * copies
        sharded += split * copies
    return replicated, sharded


def lora_tensor_placement(
    arch: ModelArch,
    rank: int,
    scope: LoraTargetScope,
    packed_matrices: int,
    threshold: int | None,
) -> tuple[int, int]:
    """(replicated elements, sharded elements) for one LoRA adapter.

    ``threshold`` is FSDP's parameter persistence threshold: a matrix with
    fewer elements stays replicated. ``None`` is flat data parallel, so
    every tensor is replicated. The two counts sum to
    :func:`lora_param_count` plus :func:`packed_lora_param_count`.
    """
    h = arch.hidden_size
    q_dim = arch.n_heads * arch.head_dim
    kv_dim = arch.n_kv_heads * arch.head_dim
    replicated, sharded = _add_lora_pair(
        0, 0, h, q_dim, rank, arch.attention_layers, threshold
    )
    replicated, sharded = _add_lora_pair(
        replicated, sharded, h, kv_dim, rank, arch.attention_layers, threshold
    )
    replicated, sharded = _add_lora_pair(
        replicated, sharded, h, kv_dim, rank, arch.attention_layers, threshold
    )
    replicated, sharded = _add_lora_pair(
        replicated, sharded, q_dim, h, rank, arch.attention_layers, threshold
    )
    if scope == "all-linear":
        inter = arch.intermediate_size
        if arch.gated_mlp:
            replicated, sharded = _add_lora_pair(
                replicated, sharded, h, inter, rank, arch.mlp_layers, threshold
            )
            replicated, sharded = _add_lora_pair(
                replicated, sharded, h, inter, rank, arch.mlp_layers, threshold
            )
        else:
            replicated, sharded = _add_lora_pair(
                replicated, sharded, h, inter, rank, arch.mlp_layers, threshold
            )
        replicated, sharded = _add_lora_pair(
            replicated, sharded, inter, h, rank, arch.mlp_layers, threshold
        )
        if arch.n_mamba_layers:
            d_inner = arch.mamba_n_heads * arch.mamba_d_head
            in_out = d_inner + arch.mamba_conv_dim + arch.mamba_n_heads
            replicated, sharded = _add_lora_pair(
                replicated,
                sharded,
                h,
                in_out,
                rank,
                arch.n_mamba_layers,
                threshold,
            )
            replicated, sharded = _add_lora_pair(
                replicated, sharded, d_inner, h, rank, arch.n_mamba_layers, threshold
            )
    n_experts = arch.n_experts
    if packed_matrices and n_experts is not None and n_experts > 1:
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        # PEFT stacks every expert's factor into one ``[E * r, features]`` tensor.
        replicated, sharded = _add_lora_pair(
            replicated,
            sharded,
            arch.expert_width,
            expert_inter,
            rank * n_experts,
            arch.moe_layers * packed_matrices,
            threshold,
        )
    target = lora_param_count(arch, rank, scope) + packed_lora_param_count(
        arch, rank, packed_matrices
    )
    # ``global_head_dim`` scales the attention total in ``lora_param_count``;
    # the residual counts as replicated so both totals match.
    replicated += target - (replicated + sharded)
    return replicated, sharded
