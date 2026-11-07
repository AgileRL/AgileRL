# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""LoRA adapter parameter counts: the linears PEFT wraps and their sizes.

Pure functions of :mod:`agilerl.arena.memory.specs` and
:mod:`agilerl.arena.memory.formulas`. No torch, no I/O.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from agilerl.arena.memory.formulas import (
    KDA_LINEAR_NAMES,
    attention_linear_names,
    attention_linears,
    kda_linears,
)
from agilerl.arena.memory.specs import (
    LoraLinear,
    LoraSplit,
    LoraTargetScope,
    ModelArch,
)


def _attention_lora_widths(
    arch: ModelArch,
) -> tuple[tuple[int, int, int, int], ...]:
    """(head dim, KV heads, layers, layers with k/v) for each attention layer type.

    Full-attention layers use ``global_head_dim`` and ``global_n_kv_heads``
    when they differ from the sliding layers'. KV-shared layers have no k/v
    projection.
    """
    layers = arch.attention_layers
    shared = min(arch.n_kv_shared_layers, layers)
    full_head_dim = arch.global_head_dim or arch.head_dim
    full_kv_heads = arch.global_n_kv_heads or arch.n_kv_heads
    if full_head_dim == arch.head_dim and full_kv_heads == arch.n_kv_heads:
        return ((arch.head_dim, arch.n_kv_heads, layers, layers - shared),)
    full_share = (
        1.0 - arch.sliding_window_layer_fraction if arch.sliding_window else 1.0
    )
    full = round(layers * full_share)
    full_shared = round(shared * full_share)
    sliding_shared = shared - full_shared
    return (
        (
            arch.head_dim,
            arch.n_kv_heads,
            layers - full,
            layers - full - sliding_shared,
        ),
        (full_head_dim, full_kv_heads, full, full - full_shared),
    )


def _attention_lora_linears(
    arch: ModelArch, tensor_parallel: int, excluded: Collection[str]
) -> list[LoraLinear]:
    """Attention q/k/v/o; an output-gated ``q_proj`` doubles its query rows."""
    if arch.low_rank_attention:
        kv_split = "out" if arch.n_kv_heads % tensor_parallel == 0 else "none"
        # Tensor parallel keeps the down-projections whole, splits the
        # up-projections by head and the output projection by input column.
        # MLA's kv_b_proj expands the latent per query head.
        splits: dict[str, LoraSplit] = {
            "q_a_proj": "none",
            "kv_a_proj_with_mqa": "none",
            "q_proj": "out",
            "q_b_proj": "out",
            "kv_b_proj": "out",
            "kv_proj": kv_split,
            "k_proj": kv_split,
            "v_proj": kv_split,
            "o_proj": "in",
            "o_b_proj": "in",
        }
        return [
            LoraLinear(in_features, out_features, arch.attention_layers, splits[name])
            for name, (in_features, out_features) in zip(
                attention_linear_names(arch), attention_linears(arch), strict=True
            )
            if name not in excluded
        ]
    h = arch.hidden_size
    q_rows = 2 if arch.attention_output_gate else 1
    linears = []
    for head_dim, kv_heads, layers, kv_layers in _attention_lora_widths(arch):
        kv_split = "out" if kv_heads % tensor_parallel == 0 else "none"
        v_dim = arch.v_head_dim or head_dim
        named = (
            ("q_proj", LoraLinear(h, q_rows * arch.n_heads * head_dim, layers, "out")),
            ("k_proj", LoraLinear(h, kv_heads * head_dim, kv_layers, kv_split)),
            ("v_proj", LoraLinear(h, kv_heads * v_dim, kv_layers, kv_split)),
            ("o_proj", LoraLinear(arch.n_heads * v_dim, h, layers, "in")),
        )
        linears += [linear for name, linear in named if name not in excluded]
    return linears


def _mlp_lora_linears(arch: ModelArch, inter: int, count: int) -> list[LoraLinear]:
    """Gate/up and down linears of ``count`` MLPs ``inter`` wide."""
    h = arch.hidden_size
    if arch.fused_gate_up_mlp:
        gate_up = LoraLinear(h, 2 * inter, count, "out")
    else:
        gate_up = LoraLinear(h, inter, (2 if arch.gated_mlp else 1) * count, "out")
    return [gate_up, LoraLinear(inter, h, count, "in")]


def _recurrent_lora_linears(
    arch: ModelArch, excluded: Collection[str]
) -> list[LoraLinear]:
    """Mamba-2 ``in_proj``/``out_proj``, or the gated-delta-net or KDA projections."""
    h = arch.hidden_size
    layers = arch.n_mamba_layers
    if arch.mamba_kda:
        # Tensor parallel splits every KDA projection by head except the
        # low-rank gate down-projections, and ``o_proj`` by input column.
        kda_splits: dict[str, LoraSplit] = {
            "q_proj": "out",
            "k_proj": "out",
            "v_proj": "out",
            "f_a_proj": "none",
            "f_b_proj": "out",
            "b_proj": "out",
            "g_a_proj": "none",
            "g_b_proj": "out",
            "o_proj": "in",
        }
        return [
            LoraLinear(in_features, out_features, layers, kda_splits[name])
            for name, (in_features, out_features) in zip(
                KDA_LINEAR_NAMES, kda_linears(arch), strict=True
            )
            if name not in excluded
        ]
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    out_proj = LoraLinear(d_inner, h, layers, "in")
    if arch.mamba_gated_delta:
        # in_proj_qkv, in_proj_z, then in_proj_b and in_proj_a (one row per head).
        return [
            LoraLinear(h, arch.mamba_conv_dim, layers, "out"),
            LoraLinear(h, d_inner, layers, "out"),
            LoraLinear(h, arch.mamba_n_heads, 2 * layers, "out"),
            out_proj,
        ]
    in_proj = d_inner + arch.mamba_conv_dim + arch.mamba_n_heads
    return [LoraLinear(h, in_proj, layers, "out"), out_proj]


def _per_layer_input_lora_linears(arch: ModelArch) -> list[LoraLinear]:
    """Per-layer-input gate and projection, and the model-level projection."""
    width = arch.per_layer_input_dim
    if not width:
        return []
    h = arch.hidden_size
    return [
        LoraLinear(h, width, arch.n_layers),
        LoraLinear(width, h, arch.n_layers),
        LoraLinear(h, arch.n_layers * width, 1),
    ]


def decoder_lora_linears(
    arch: ModelArch, scope: LoraTargetScope, tensor_parallel: int = 1
) -> list[LoraLinear]:
    """Decoder linears one LoRA adapter wraps, with the factor tensor parallel splits.

    ``all-linear`` wraps attention q/k/v/o, dense MLP matrices, shared
    experts, recurrent-mixer projections and per-layer-input projections.
    The embedding, ``lm_head``, packed routed experts and
    :attr:`ModelArch.lora_excluded_modules` stay unwrapped. Key and value
    LoRA B split only when the key/value heads divide ``tensor_parallel``.

    :param arch: Decoder geometry.
    :param scope: LoRA target scope.
    :param tensor_parallel: Training tensor-parallel degree.
    :return: Wrapped linears.
    """
    return _decoder_lora_linears(
        arch, scope, tensor_parallel, arch.lora_excluded_modules
    )


def _decoder_lora_linears(
    arch: ModelArch,
    scope: LoraTargetScope,
    tensor_parallel: int,
    excluded: Collection[str],
) -> list[LoraLinear]:
    """:func:`decoder_lora_linears` with the module names in ``excluded`` left unwrapped."""
    linears = _attention_lora_linears(arch, tensor_parallel, excluded)
    if scope != "all-linear":
        return linears
    linears += _mlp_lora_linears(arch, arch.intermediate_size, arch.mlp_layers)
    if arch.is_moe and arch.n_shared_experts:
        shared = (
            arch.shared_expert_intermediate_size
            or arch.expert_intermediate_size
            or arch.intermediate_size
        )
        linears += _mlp_lora_linears(
            arch, shared, arch.moe_layers * arch.n_shared_experts
        )
    if arch.n_mamba_layers:
        linears += _recurrent_lora_linears(arch, excluded)
    return linears + _per_layer_input_lora_linears(arch)


def tower_lora_linears(arch: ModelArch, targets: Collection[str]) -> list[LoraLinear]:
    """Tower and connector linears whose module name is one of ``targets``."""
    return [linear for linear in arch.tower_lora_linears if linear.target in targets]


def _lora_linear_param_count(linears: Sequence[LoraLinear], rank: int) -> int:
    """LoRA parameters of ``linears``: ``rank * (in + out)`` per wrapped linear."""
    return rank * sum(
        (linear.in_features + linear.out_features) * linear.count for linear in linears
    )


def lora_param_count(
    arch: ModelArch, rank: int, scope: LoraTargetScope = "all-linear"
) -> int:
    """LoRA adapter parameters of the decoder linears :func:`decoder_lora_linears` lists."""
    return _lora_linear_param_count(decoder_lora_linears(arch, scope), rank)


def engine_lora_param_count(arch: ModelArch, rank: int) -> int:
    """Parameters of one vLLM LoRA slot: every decoder linear vLLM wraps.

    vLLM wraps its LoRA-capable layers whatever the adapter targets, so
    :attr:`ModelArch.lora_excluded_modules` still take slot space.
    """
    return _lora_linear_param_count(
        _decoder_lora_linears(arch, "all-linear", 1, ()), rank
    )


def tower_lora_param_count(arch: ModelArch, rank: int, targets: Collection[str]) -> int:
    """LoRA adapter parameters on the tower and connector linears ``targets`` name."""
    return _lora_linear_param_count(tower_lora_linears(arch, targets), rank)


def packed_lora_param_count(arch: ModelArch, rank: int, n_matrices: int) -> int:
    """Adapter parameters for PEFT ``target_parameters`` on packed experts.

    A packed ``(n_experts, out, in)`` Parameter has no ``nn.Linear`` to wrap,
    so PEFT decomposes it per expert: every targeted matrix adds
    ``n_experts * rank * (in + out)`` on each MoE layer. A gated stack packs
    gate and up into one ``gate_up_proj`` of ``out = 2 * inter``, and the
    targets include it.
    """
    n_experts = arch.n_experts
    if not n_matrices or n_experts is None or n_experts <= 1:
        return 0
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    widths = n_matrices * (arch.expert_width + expert_inter)
    if arch.gated_mlp:
        widths += expert_inter
    return int(arch.moe_layers * n_experts * rank * widths)


def merged_lora_sync_bytes(
    arch: ModelArch,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    dtype_bytes: float,
) -> tuple[int, int]:
    """Bytes of the merged weights a merged-LoRA sync pushes, and of the largest one.

    The pushed weights are the linears
    :func:`decoder_lora_linears` lists,
    without biases. Packed experts are pushed whole: ``gate_up_proj`` and
    ``down_proj`` are two targeted parameters, and one target is charged
    as ``gate_up_proj``, the larger.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param dtype_bytes: Bytes per element of the engines' weight dtype.
    :return: Total bytes, and bytes of the largest pushed tensor.
    """
    sizes = [
        (linear.in_features * linear.out_features, linear.count)
        for linear in decoder_lora_linears(arch, scope)
    ]
    if n_packed_matrices and arch.n_experts and arch.n_experts > 1:
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        down = arch.n_experts * arch.expert_width * expert_inter
        gate_up = (2 if arch.gated_mlp else 1) * down
        sizes.append((gate_up, arch.moe_layers))
        if n_packed_matrices > 1:
            sizes.append((down, arch.moe_layers))
    total = sum(size * count for size, count in sizes if count)
    largest = max((size for size, count in sizes if count), default=0)
    return int(total * dtype_bytes), int(largest * dtype_bytes)


def lora_sync_buffer_bytes(
    arch: ModelArch,
    rank: int,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    dtype_bytes: float,
) -> int:
    """Bytes of one unsharded policy adapter in a NIXL weight-sync buffer.

    :param arch: Model architecture.
    :param rank: LoRA rank.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param dtype_bytes: Bytes per element of the engines' LoRA dtype.
    :return: Buffer bytes.
    """
    params = lora_param_count(arch, rank, scope) + packed_lora_param_count(
        arch, rank, n_packed_matrices
    )
    return int(params * dtype_bytes)


def engine_expert_lora_param_count(
    arch: ModelArch, rank: int, tensor_parallel: int, packed_3d: bool
) -> int:
    """Per-rank parameters of one vLLM fused-MoE LoRA slot.

    :param arch: Model architecture.
    :param rank: Engine LoRA rank.
    :param tensor_parallel: Engine tensor-parallel width.
    :param packed_3d: The engine uses the 3D expert layout.
    :return: Slot parameters on one rank; 0 for a dense model.
    """
    n_experts = arch.n_experts
    if n_experts is None or n_experts <= 1:
        return 0
    h = arch.expert_width
    # vLLM wraps every fused-MoE layer once LoRA is on, whatever the adapter
    # targets: w13 and w2 A/B stacks over all experts, intermediate dim TP-sharded.
    inter = (arch.expert_intermediate_size or arch.intermediate_size) // tensor_parallel
    # The 3D layout packs gate and up into one w13 slice; the 2D layout keeps
    # one slice each when the MLP is gated.
    if packed_3d:
        per_expert = 2 * h + 3 * inter
    else:
        slices = 2 if arch.gated_mlp else 1
        per_expert = slices * h + h + (slices + 1) * inter
    return int(arch.moe_layers * n_experts * rank * per_expert)


def received_expert_lora_b_param_count(arch: ModelArch, rank: int) -> int:
    """Parameters of the expert ``lora_B`` stacks vLLM copies out of a received adapter.

    Loading a packed-expert adapter whose tensors already sit on the GPU,
    vLLM's 3D layout permutes each ``lora_B`` to ``(experts, out, rank)``
    with a contiguous copy. The loaded adapter keeps the copy, unsharded,
    on every engine rank.

    :param arch: Model architecture.
    :param rank: Adapter LoRA rank.
    :return: Copied parameters on one engine rank; 0 for a dense model.
    """
    n_experts = arch.n_experts
    if n_experts is None or n_experts <= 1:
        return 0
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    gate_up = (2 if arch.gated_mlp else 1) * expert_inter
    return int(arch.moe_layers * n_experts * rank * (gate_up + arch.expert_width))
