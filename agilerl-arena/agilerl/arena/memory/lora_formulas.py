# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""LoRA adapter parameter counts: the linears PEFT wraps and their sizes.

Pure functions of :mod:`agilerl.arena.memory.specs` and
:mod:`agilerl.arena.memory.formulas`. No torch, no I/O.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import NamedTuple

from agilerl.arena.memory.formulas import (
    ADAPTER_BYTES_PER_PARAM,
    KDA_LINEAR_NAMES,
    NCCL_RESHARD_MISC_BUCKET_BYTES,
    attention_linear_names,
    attention_linears,
    kda_linears,
)
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    LoraLinear,
    LoraSplit,
    LoraTargetScope,
    ModelArch,
)


class NcclReshardLayout(NamedTuple):
    """The trainer sharding and engine width a ``nccl_reshard`` refit plans its bulk/misc split for."""

    # Trainer ranks one weight's rows are sharded across; 1 without FSDP.
    trainer_shards: int
    # Engine ``tensor_parallel_size`` the bulk weights reshard into.
    engine_tensor_parallel: int


class NcclReshardMove(NamedTuple):
    """Bytes a merged-LoRA ``nccl_reshard`` refit sends, split the way the plan sends them.

    Bulk parameters each move in one ungrouped ``m2n.reshard``. Misc parameters
    are packed into 512 MiB buckets, merged on rank 0 and broadcast.
    """

    bulk_bytes: int
    misc_bytes: int


class NcclReshardRefitBytes(NamedTuple):
    """Transient trainer bytes of one merged-LoRA ``nccl_reshard`` refit on the main rank.

    The bulk phase ends before the misc phase starts. Bulk: ``bulk_group``
    is the largest bulk block's merged local shards at the sync dtype and
    ``bulk_fp32_temp`` that block's fp32 merge temp of its largest local
    shard. Misc: ``misc_bucket`` is the reused broadcast bucket plus any
    weight too large for it, ``misc_module`` the largest misc module gathered
    whole (base weight and fp32 LoRA factors), and ``misc_fp32_copy`` the
    fp32 copy of its base the merge accumulates into.
    """

    misc_bucket: int
    misc_module: int
    misc_fp32_copy: int
    bulk_group: int
    bulk_fp32_temp: int

    @property
    def bulk_phase(self) -> int:
        """Peak extra bytes while bulk blocks reshard."""
        return self.bulk_group + self.bulk_fp32_temp

    @property
    def misc_phase(self) -> int:
        """Peak extra bytes while misc buckets broadcast."""
        return self.misc_bucket + self.misc_module + self.misc_fp32_copy

    @property
    def total(self) -> int:
        """Peak extra bytes of the refit."""
        return max(self.bulk_phase, self.misc_phase)


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
    return [
        linear
        for _, named in _attention_projection_types(arch, tensor_parallel)
        for name, linear in named
        if name not in excluded
    ]


def _attention_projection_types(
    arch: ModelArch, tensor_parallel: int
) -> list[tuple[int, tuple[tuple[str, LoraLinear], ...]]]:
    """KV heads and named projections of each attention layer type."""
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
        named_linears = tuple(
            (
                name,
                LoraLinear(
                    in_features, out_features, arch.attention_layers, splits[name]
                ),
            )
            for name, (in_features, out_features) in zip(
                attention_linear_names(arch), attention_linears(arch), strict=True
            )
        )
        return [(arch.n_kv_heads, named_linears)]
    h = arch.hidden_size
    q_rows = 2 if arch.attention_output_gate else 1
    types = []
    for head_dim, kv_heads, layers, kv_layers in _attention_lora_widths(arch):
        kv_split = "out" if kv_heads % tensor_parallel == 0 else "none"
        v_dim = arch.v_head_dim or head_dim
        named = (
            ("q_proj", LoraLinear(h, q_rows * arch.n_heads * head_dim, layers, "out")),
            ("k_proj", LoraLinear(h, kv_heads * head_dim, kv_layers, kv_split)),
            ("v_proj", LoraLinear(h, kv_heads * v_dim, kv_layers, kv_split)),
            ("o_proj", LoraLinear(arch.n_heads * v_dim, h, layers, "in")),
        )
        types.append((kv_heads, named))
    return types


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


def merged_lora_sync_tensor_bytes(
    arch: ModelArch,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    dtype_bytes: float,
) -> list[int]:
    """Bytes of each merged weight a merged-LoRA sync pushes, layer by layer.

    The pushed weights are the linears :func:`decoder_lora_linears` lists,
    without biases, each spread evenly over the decoder layers. Packed
    experts are pushed whole: ``gate_up_proj`` and ``down_proj`` are two
    targeted parameters, and one target is charged as ``gate_up_proj``, the
    larger.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param dtype_bytes: Bytes per element of the engines' weight dtype.
    :return: One entry per pushed tensor, in checkpoint layer order.
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
    layers = arch.n_layers
    return [
        int(size * dtype_bytes)
        for layer in range(layers)
        for size, count in sizes
        for _ in range(count * (layer + 1) // layers - count * layer // layers)
    ]


def merged_lora_sync_bytes(
    arch: ModelArch,
    scope: LoraTargetScope,
    n_packed_matrices: int,
    dtype_bytes: float,
) -> tuple[int, int]:
    """Bytes of the merged weights a merged-LoRA sync pushes, and of the largest one.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :param dtype_bytes: Bytes per element of the engines' weight dtype.
    :return: Total bytes of :func:`merged_lora_sync_tensor_bytes`, and of the
        largest pushed tensor.
    """
    sizes = merged_lora_sync_tensor_bytes(arch, scope, n_packed_matrices, dtype_bytes)
    return sum(sizes), max(sizes, default=0)


def _reshards_evenly(
    linear: LoraLinear, row_parallel: bool, layout: NcclReshardLayout
) -> bool:
    """Whether trainer row shards and the engine's TP slices of ``linear`` divide evenly."""
    engine_dim = linear.in_features if row_parallel else linear.out_features
    return (
        linear.out_features % layout.trainer_shards == 0
        and engine_dim % layout.engine_tensor_parallel == 0
    )


def _attention_reshard_split(
    adapted: dict[str, LoraLinear], kv_heads: int, layout: NcclReshardLayout
) -> tuple[list[tuple[int, int]], list[LoraLinear]]:
    """Bulk ``(elements, layers)`` and misc linears of one attention projection type."""
    tp = layout.engine_tensor_parallel
    qkv = [adapted[name] for name in ("q_proj", "k_proj", "v_proj") if name in adapted]
    o_proj = adapted.get("o_proj")
    misc = [
        linear
        for name, linear in adapted.items()
        if name not in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    block: list[tuple[int, int]] = []
    if (
        len(qkv) == 3
        and qkv[1].count > 0
        and kv_heads >= tp
        and kv_heads % tp == 0
        and all(_reshards_evenly(linear, False, layout) for linear in qkv)
    ):
        q_proj = qkv[0]
        layers = qkv[1].count
        block += [(linear.in_features * linear.out_features, layers) for linear in qkv]
        misc.append(
            LoraLinear(
                q_proj.in_features,
                q_proj.out_features,
                q_proj.count - layers,
                q_proj.split,
            )
        )
    else:
        misc += qkv
    if o_proj is not None and _reshards_evenly(o_proj, True, layout):
        block.append((o_proj.in_features * o_proj.out_features, o_proj.count))
    elif o_proj is not None:
        misc.append(o_proj)
    return block, misc


def _mlp_reshard_split(
    arch: ModelArch,
    gate_up: LoraLinear,
    down: LoraLinear,
    layout: NcclReshardLayout,
) -> tuple[list[tuple[int, int]], list[LoraLinear]]:
    """Bulk ``(elements, layers)`` and misc linears of the dense MLP."""
    block: list[tuple[int, int]] = []
    misc: list[LoraLinear] = []
    if not arch.fused_gate_up_mlp and _reshards_evenly(gate_up, False, layout):
        gates = 2 if arch.gated_mlp else 1
        # ``gate_up.count`` is gate and up together, one of each per MLP layer.
        layers = gate_up.count // gates
        block += [(gate_up.in_features * gate_up.out_features, layers)] * gates
    else:
        misc.append(gate_up)
    if _reshards_evenly(down, True, layout):
        block.append((down.in_features * down.out_features, down.count))
    else:
        misc.append(down)
    return block, misc


def _nccl_reshard_parts(
    arch: ModelArch, scope: LoraTargetScope, layout: NcclReshardLayout
) -> tuple[list[list[tuple[int, int]]], list[LoraLinear]]:
    """Bulk blocks as ``(elements, layers)`` and the misc linears of one refit."""
    excluded = arch.lora_excluded_modules
    blocks: list[list[tuple[int, int]]] = []
    planned: list[LoraLinear] = []
    misc: list[LoraLinear] = []
    for kv_heads, named in _attention_projection_types(arch, 1):
        adapted = {name: linear for name, linear in named if name not in excluded}
        planned += adapted.values()
        block, block_misc = _attention_reshard_split(adapted, kv_heads, layout)
        misc += block_misc
        if block:
            blocks.append(block)
    if scope == "all-linear" and arch.mlp_layers:
        gate_up, down = _mlp_lora_linears(arch, arch.intermediate_size, arch.mlp_layers)
        planned += [gate_up, down]
        block, block_misc = _mlp_reshard_split(arch, gate_up, down, layout)
        misc += block_misc
        if block:
            blocks.append(block)
    others = decoder_lora_linears(arch, scope)
    for linear in planned:
        others.remove(linear)
    return blocks, [linear for linear in misc + others if linear.count]


def nccl_reshard_move(
    arch: ModelArch,
    scope: LoraTargetScope,
    layout: NcclReshardLayout,
    dtype_bytes: float,
) -> NcclReshardMove:
    """Bulk and misc bytes of one merged-LoRA ``nccl_reshard`` refit.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param layout: Trainer sharding and engine width.
    :param dtype_bytes: Bytes per element of the sync dtype.
    :return: Bulk bytes and misc bytes.
    """
    blocks, misc = _nccl_reshard_parts(arch, scope, layout)
    bulk_bytes = 0
    for block in blocks:
        for elements, layers in block:
            bulk_bytes += int(elements * layers * dtype_bytes)
    misc_bytes = int(
        sum(
            linear.in_features * linear.out_features * linear.count * dtype_bytes
            for linear in misc
        )
    )
    return NcclReshardMove(bulk_bytes, misc_bytes)


def nccl_reshard_sync_linears(
    arch: ModelArch, scope: LoraTargetScope, layout: NcclReshardLayout
) -> tuple[list[list[int]], list[LoraLinear]]:
    """Bulk blocks and misc linears of a merged-LoRA ``nccl_reshard`` refit.

    Bulk weights are decoder ``self_attn`` q/k/v/o and dense MLP
    gate/up/down whose trainer row shards and engine TP slices divide
    evenly: each trainer rank merges its own rows and reshards them one
    attention or MLP block at a time. vLLM loads ``qkv_proj`` and
    ``gate_up_proj`` whole, so their linears are bulk together or misc
    together; ``qkv_proj`` also needs all of q/k/v adapted and a KV head
    count the engine width divides. KV-shared layers have no k/v, so their
    ``q_proj`` is misc. A checkpoint's fused ``gate_up_proj`` is misc while
    ``down_proj`` stays bulk. Every other linear :func:`decoder_lora_linears`
    lists is misc, gathered whole and broadcast from rank 0.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param layout: Trainer sharding and engine width.
    :return: Weight elements of each tensor in one bulk block, one entry per
        block shape, and the misc linears.
    """
    blocks, misc = _nccl_reshard_parts(arch, scope, layout)
    return [[elements for elements, _ in block] for block in blocks], misc


def _misc_buckets(misc: Sequence[LoraLinear], dtype_bytes: float) -> tuple[int, int]:
    """Bytes of the reused misc bucket, and of the largest weight too large for it."""
    sizes = [
        (int(linear.in_features * linear.out_features * dtype_bytes), linear.count)
        for linear in misc
    ]
    packed = sum(
        size * count for size, count in sizes if size <= NCCL_RESHARD_MISC_BUCKET_BYTES
    )
    oversize = max(
        (size for size, _ in sizes if size > NCCL_RESHARD_MISC_BUCKET_BYTES), default=0
    )
    return min(packed, NCCL_RESHARD_MISC_BUCKET_BYTES), oversize


def nccl_reshard_receive_bucket_bytes(
    arch: ModelArch,
    scope: LoraTargetScope,
    layout: NcclReshardLayout,
    dtype_bytes: float,
) -> int:
    """Bytes of the largest misc bucket an engine GPU receives in a merged-LoRA ``nccl_reshard`` refit.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param layout: Trainer sharding and engine width.
    :param dtype_bytes: Bytes per element of the sync dtype.
    :return: One bucket of up to
        :data:`~agilerl.arena.memory.formulas.NCCL_RESHARD_MISC_BUCKET_BYTES`,
        or the largest misc weight when it is larger.
    """
    _, misc = nccl_reshard_sync_linears(arch, scope, layout)
    return max(_misc_buckets(misc, dtype_bytes))


def nccl_reshard_refit_bytes(
    arch: ModelArch,
    scope: LoraTargetScope,
    rank: int,
    layout: NcclReshardLayout,
    dtype_bytes: float,
) -> NcclReshardRefitBytes:
    """Transient main-rank trainer bytes of one merged-LoRA ``nccl_reshard`` refit.

    Bulk phase: a block stages its merged local shards at ``dtype_bytes``
    beside one fp32 merge temp of its largest local shard; the largest block
    is charged. Misc phase, after it: rank 0 keeps the reused bucket, a
    weight too large for it takes a buffer of its own, and the largest misc
    module is gathered whole (base at ``dtype_bytes``, fp32 LoRA factors)
    beside the fp32 copy of its base the merge accumulates into.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param rank: LoRA rank.
    :param layout: Trainer sharding and engine width.
    :param dtype_bytes: Bytes per element of the trainer base weights and the sync dtype.
    :return: The refit's terms.
    """
    blocks, misc = nccl_reshard_sync_linears(arch, scope, layout)
    shards = layout.trainer_shards
    fp32 = DTYPE_BYTES["fp32"]
    misc_module, misc_fp32_copy = max(
        (
            (
                int(
                    linear.in_features * linear.out_features * dtype_bytes
                    + rank
                    * (linear.in_features + linear.out_features)
                    * ADAPTER_BYTES_PER_PARAM
                ),
                int(linear.in_features * linear.out_features * fp32),
            )
            for linear in misc
        ),
        key=sum,
        default=(0, 0),
    )
    bulk_group, bulk_fp32_temp = max(
        (
            (
                int(sum(n // shards for n in block) * dtype_bytes),
                int(max(block) // shards * fp32),
            )
            for block in blocks
        ),
        key=sum,
        default=(0, 0),
    )
    return NcclReshardRefitBytes(
        misc_bucket=sum(_misc_buckets(misc, dtype_bytes)),
        misc_module=misc_module,
        misc_fp32_copy=misc_fp32_copy,
        bulk_group=bulk_group,
        bulk_fp32_temp=bulk_fp32_temp,
    )


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


def lora_adapter_tensor_count(
    arch: ModelArch, scope: LoraTargetScope, n_packed_matrices: int
) -> int:
    """LoRA A and B tensors of one policy adapter: two per wrapped linear and per packed-expert target per MoE layer.

    :param arch: Model architecture.
    :param scope: Adapter target scope.
    :param n_packed_matrices: PEFT ``target_parameters`` on packed experts.
    :return: Tensor count.
    """
    targets = sum(linear.count for linear in decoder_lora_linears(arch, scope))
    if arch.n_experts and arch.n_experts > 1:
        targets += n_packed_matrices * arch.moe_layers
    return 2 * targets


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
