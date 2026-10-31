# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Parameter elements FSDP gathers per decoder block, side unit and prefetch window."""

from __future__ import annotations

from agilerl.arena.memory.formulas import (
    ParamCounts,
    attention_params_per_layer,
    mamba_params_per_layer,
    mlp_params_per_layer,
    moe_params_per_layer,
    routed_expert_params_per_layer,
    tensor_parallel_params,
    token_embedding_params,
)
from agilerl.arena.memory.specs import LayerKind, ModelArch
from agilerl.arena.models.fsdp import PREFETCH_UNITS


def block_moe_params_per_layer(arch: ModelArch, expert_parallel: int = 1) -> int:
    """MoE parameters of one layer inside its block's FSDP unit.

    Expert-parallel routed experts are their own unit, so the block keeps
    only the router, shared experts and latent projections.
    """
    experts, router = moe_params_per_layer(arch)
    if expert_parallel > 1:
        experts -= routed_expert_params_per_layer(arch)
    return experts + router


def largest_block_params(
    counts: ParamCounts, arch: ModelArch, expert_parallel: int = 1
) -> int:
    """Parameters in the widest decoder block FSDP wraps."""
    if arch.block_exclusive_layers:
        mlp = mlp_params_per_layer(arch) if arch.mlp_layers else 0
        return int(
            max(
                attention_params_per_layer(arch),
                mlp,
                block_moe_params_per_layer(arch, expert_parallel),
                mamba_params_per_layer(arch),
            )
        )
    body = counts.attention + counts.mlp + counts.moe_experts + counts.norms
    body += counts.mamba
    if expert_parallel > 1:
        body -= arch.moe_layers * routed_expert_params_per_layer(arch)
    return int(body / max(arch.n_layers, 1))


def layer_params(arch: ModelArch, kind: LayerKind, expert_parallel: int = 1) -> int:
    """Parameters of one layer of a block-exclusive stack."""
    if kind == "attention":
        return attention_params_per_layer(arch)
    if kind == "mamba":
        return mamba_params_per_layer(arch)
    if kind == "mlp":
        return mlp_params_per_layer(arch)
    return block_moe_params_per_layer(arch, expert_parallel)


def prefetch_window_params(
    counts: ParamCounts,
    arch: ModelArch,
    live_units: int,
    tensor_parallel: int = 1,
    expert_parallel: int = 1,
) -> int:
    """Decoder parameters gathered at once: ``live_units`` neighbouring blocks.

    Each decoder block is one FSDP unit. With the layer order known, the
    window is the widest run of neighbouring blocks, so a large block next
    to small ones is counted once. Otherwise every block is the widest one.
    Layers with every linear on the TP mesh gather nothing.
    """
    linears_on_tp = tensor_parallel > 1 and arch.tensor_parallel_plan
    if not arch.layer_kinds:
        block = largest_block_params(counts, arch, expert_parallel)
        if linears_on_tp:
            # A uniform stack's linears are on the TP mesh. Norms and MoE stay gathered.
            moe = arch.moe_layers * block_moe_params_per_layer(arch, expert_parallel)
            block = int((counts.norms + moe) / max(arch.n_layers, 1))
        return live_units * block
    tp_kinds = ("attention", "mamba", "mlp") if linears_on_tp else ()
    sizes = [
        0 if kind in tp_kinds else layer_params(arch, kind, expert_parallel)
        for kind in arch.layer_kinds
    ]
    live_units = min(live_units, len(sizes))
    return max(
        sum(sizes[start : start + live_units])
        for start in range(len(sizes) - live_units + 1)
    )


def fsdp_gathered_params(
    counts: ParamCounts,
    arch: ModelArch,
    *,
    n_gpus: int,
    reshard_after_forward: bool,
    tensor_parallel: int = 1,
    expert_parallel: int = 1,
) -> int:
    """Unsharded parameter elements held besides the resident shard.

    Excludes the replicated token embeddings. ``reshard_after_forward``
    keeps a prefetch window of decoder units, plus an untied ``lm_head``,
    per-layer embeddings, and the root unit (tower parameters outside the
    encoder blocks, unattributed parameters) at their own gather. Otherwise
    every sharded parameter stays gathered. A single GPU gathers into its own
    buffers too, so the unsharded copy sits beside the one-rank shard.

    Expert-parallel routed experts are not in the block units. A rank
    gathers one MoE layer of its own experts across the shard-group ranks
    that hold the same experts, and nothing when ``ep`` fills the group.
    """
    token = min(token_embedding_params(arch), max(counts.total, 0))
    tp_params = 0 if tensor_parallel <= 1 else tensor_parallel_params(counts, arch)
    sharded = max(counts.total - token - tp_params, 0)
    expert_layer = 0
    if expert_parallel > 1:
        routed = routed_expert_params_per_layer(arch)
        sharded = max(sharded - arch.moe_layers * routed, 0)
        if n_gpus > expert_parallel:
            expert_layer = routed // expert_parallel
    if not reshard_after_forward:
        return sharded + arch.moe_layers * expert_layer
    live = min(max(1, arch.n_layers), 1 + PREFETCH_UNITS)
    window = prefetch_window_params(
        counts, arch, live, tensor_parallel, expert_parallel
    )
    head, extra_embed, root = _side_units(counts, arch, tensor_parallel)
    return min(sharded, window + head + extra_embed + root) + expert_layer


def _side_units(
    counts: ParamCounts, arch: ModelArch, tensor_parallel: int = 1
) -> tuple[int, int, int]:
    """Untied ``lm_head``, per-layer embedding, and root-unit elements.

    The root unit holds tower parameters outside the encoder blocks and
    parameters no group accounts for. An untied head on the TP mesh is not
    an FSDP unit.
    """
    head = 0 if arch.tied_embeddings or tensor_parallel > 1 else counts.lm_head
    extra_embed = max(counts.embedding - token_embedding_params(arch), 0)
    tower_root = max(counts.multimodal_towers - arch.multimodal_tower_block_params, 0)
    return head, extra_embed, tower_root + max(counts.unattributed, 0)


def fsdp_resharded_side_params(counts: ParamCounts, arch: ModelArch) -> int:
    """Gathered elements of the untied ``lm_head`` and per-layer embeddings.

    :param counts: Parameter counts of the model.
    :param arch: Model architecture.
    :return: Gathered elements of the two side units.
    """
    # Each side unit reshards after its own forward or backward, so neither
    # is live during the decoder backward; the root unit stays gathered.
    head, extra_embed, _root = _side_units(counts, arch)
    return head + extra_embed


def fsdp_gather_pool_params(
    counts: ParamCounts,
    arch: ModelArch,
    ep: int,
    shards: int,
    tensor_parallel: int = 1,
) -> int:
    """Elements FSDP's all-gather stream keeps reserved for the whole step.

    :param counts: Parameter counts of the model.
    :param arch: Model architecture.
    :param ep: Expert-parallel width.
    :param shards: FSDP shard count.
    :param tensor_parallel: Training TP width; TP-planned linears gather nothing.
    :return: Elements of the widest set of gathered outputs in flight together.
    """
    # The all-gather stream's cache keeps its high-water mark out of the
    # compute stream's reach: a prefetch window of decoder units, or a side
    # unit beside its neighbour (root and per-layer embeddings before the
    # first decoder unit, an untied lm_head after the last).
    linears_on_tp = tensor_parallel > 1 and arch.tensor_parallel_plan
    tp_kinds = ("attention", "mamba", "mlp") if linears_on_tp else ()
    if arch.layer_kinds:
        # Layers fully on the TP mesh never enter the stream, so the window
        # spans the next gathered layers.
        kinds = [kind for kind in arch.layer_kinds if kind not in tp_kinds]
        sizes = [layer_params(arch, kind) for kind in kinds] or [0]
        routed_layers = [kind == "moe" for kind in kinds] or [False]
    else:
        block = largest_block_params(counts, arch)
        if linears_on_tp:
            block = int(
                (counts.norms + arch.moe_layers * sum(moe_params_per_layer(arch)))
                / max(arch.n_layers, 1)
            )
        sizes = [block] * arch.n_layers
        routed_layers = [arch.is_moe] * arch.n_layers
    # Routed experts leave the block units under expert parallel; a rank
    # gathers its 1/ep share only when there are FSDP ranks beyond ep.
    if ep > 1 and arch.is_moe:
        inter = arch.expert_intermediate_size or arch.intermediate_size
        matrices = 3 if arch.gated_mlp else 2
        routed = arch.n_experts * matrices * arch.expert_width * inter
        local = routed // ep if shards > ep else 0
        sizes = [
            size - routed + local if is_routed else size
            for size, is_routed in zip(sizes, routed_layers, strict=True)
        ]
    live = min(len(sizes), 1 + PREFETCH_UNITS)
    window = max(
        sum(sizes[start : start + live]) for start in range(len(sizes) - live + 1)
    )
    head, extra_embed, root = _side_units(counts, arch, tensor_parallel)
    return max(window, max(extra_embed, root) + sizes[0], head + sizes[-1])
