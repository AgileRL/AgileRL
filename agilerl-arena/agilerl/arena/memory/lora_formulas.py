# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""LoRA adapter parameter counts: the linears PEFT wraps and their sizes.

Pure functions of :mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from agilerl.arena.memory.specs import LoraLinear, LoraTargetScope, ModelArch


def _attention_lora_widths(arch: ModelArch) -> tuple[tuple[int, int, int], ...]:
    """(head dim, layers, layers with k/v) for each attention head width.

    Full-attention layers use ``global_head_dim`` when it differs from the
    sliding layers' ``head_dim``. KV-shared layers have no k/v projection.
    """
    layers = arch.attention_layers
    shared = min(arch.n_kv_shared_layers, layers)
    if not arch.global_head_dim or arch.global_head_dim == arch.head_dim:
        return ((arch.head_dim, layers, layers - shared),)
    full_share = (
        1.0 - arch.sliding_window_layer_fraction if arch.sliding_window else 1.0
    )
    full = round(layers * full_share)
    full_shared = round(shared * full_share)
    sliding_shared = shared - full_shared
    return (
        (arch.head_dim, layers - full, layers - full - sliding_shared),
        (arch.global_head_dim, full, full - full_shared),
    )


def _attention_lora_linears(arch: ModelArch, tensor_parallel: int) -> list[LoraLinear]:
    """Attention q/k/v/o; an output-gated ``q_proj`` doubles its query rows."""
    h = arch.hidden_size
    kv_split = "out" if arch.n_kv_heads % tensor_parallel == 0 else "none"
    q_rows = 2 if arch.attention_output_gate else 1
    linears = []
    for head_dim, layers, kv_layers in _attention_lora_widths(arch):
        q_dim = arch.n_heads * head_dim
        kv_dim = arch.n_kv_heads * head_dim
        linears += [
            LoraLinear(h, q_rows * q_dim, layers, "out"),
            LoraLinear(h, kv_dim, 2 * kv_layers, kv_split),
            LoraLinear(q_dim, h, layers, "in"),
        ]
    return linears


def _mlp_lora_linears(arch: ModelArch, inter: int, count: int) -> list[LoraLinear]:
    """Gate/up and down linears of ``count`` MLPs ``inter`` wide."""
    h = arch.hidden_size
    if arch.fused_gate_up_mlp:
        gate_up = LoraLinear(h, 2 * inter, count, "out")
    else:
        gate_up = LoraLinear(h, inter, (2 if arch.gated_mlp else 1) * count, "out")
    return [gate_up, LoraLinear(inter, h, count, "in")]


def _recurrent_lora_linears(arch: ModelArch) -> list[LoraLinear]:
    """Mamba-2 ``in_proj``/``out_proj``, or the gated-delta-net projections."""
    h = arch.hidden_size
    layers = arch.n_mamba_layers
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
    The embedding, ``lm_head`` and packed routed experts stay unwrapped. Key
    and value LoRA B split only when the key/value heads divide
    ``tensor_parallel``.

    :param arch: Decoder geometry.
    :param scope: LoRA target scope.
    :param tensor_parallel: Training tensor-parallel degree.
    :return: Wrapped linears.
    """
    linears = _attention_lora_linears(arch, tensor_parallel)
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
        linears += _recurrent_lora_linears(arch)
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


def tower_lora_param_count(arch: ModelArch, rank: int, targets: Collection[str]) -> int:
    """LoRA adapter parameters on the tower and connector linears ``targets`` name."""
    return _lora_linear_param_count(tower_lora_linears(arch, targets), rank)


def packed_lora_param_count(arch: ModelArch, rank: int, n_matrices: int) -> int:
    """Adapter parameters for PEFT ``target_parameters`` on packed experts.

    A packed ``(n_experts, out, in)`` Parameter has no ``nn.Linear`` to wrap,
    so PEFT decomposes it per expert: every targeted matrix adds
    ``n_experts * rank * (in + out)`` on each MoE layer.
    """
    n_experts = arch.n_experts
    if not n_matrices or n_experts is None or n_experts <= 1:
        return 0
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    per_matrix = n_experts * rank * (arch.expert_width + expert_inter)
    return int(arch.moe_layers * n_matrices * per_matrix)


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
