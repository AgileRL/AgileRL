# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form memory terms for LLM RL training and generation.

Every function is a pure function of the specs in
:mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from agilerl.arena.memory.resolve.chunk import resolve_chunk_rows
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    BlockKind,
    LoraTargetScope,
    MiB,
    ModelArch,
    TrainingSettings,
    WeightDtype,
    WeightVariant,
)

# Upper bound on the CUDA-graph pool the vLLM engine captures. It cancels in
# the generation total: subtracted from the KV budget, added back as residency.
CUDA_GRAPH_POOL_BYTES = 256 * MiB
# Extra non-KV memory vLLM reserves with 4-token n-gram drafting: its CUDA-graph
# estimate covers the draft-width batches, and the profile runs the rejection
# sampler. Takes 0.52-0.61 GiB of KV pool on Qwen3-4B, TP 1, A100-80.
SPECULATIVE_DRAFTING_BYTES = 640 * MiB
# Extra non-KV memory vLLM reserves when LoRA layers run on two CUDA streams
# (``VLLM_LORA_ENABLE_DUAL_STREAM``), mostly a larger CUDA-graph estimate. Takes
# 0.5-1.1 GiB of KV pool (median 0.66) on Qwen3-4B, Granite 3.1 MoE and
# Granite 4.0-H, bf16 and fp8, TP 1, A100-80 and H100.
LORA_DUAL_STREAM_BYTES = 768 * MiB
# vLLM's non-torch memory (kernel workspaces, loaded kernels, NCCL), as a share
# of the unsharded engine weights at the checkpoint dtype. It sits inside the
# gpu_memory_utilization budget and every rank holds it whole: it does not
# shrink with TP or fp8 weights.
ENGINE_RUNTIME_FRACTION = 0.036
# Bytes per LoRA parameter under flat data parallel. PEFT
# ``autocast_adapter_dtype=True`` stores adapters and their gradients in
# fp32. AdamW keeps two moments per parameter in the parameter's dtype.
ADAPTER_BYTES_PER_PARAM = 4.0
ADAM_MOMENTS_PER_PARAM = 2
# Widest [rows, features] activation of one routed-expert row chunk, one fp32
# expert-parallel combine chunk, and one grouped LoRA GEMM output when no FSDP
# config sets one. Each chunk adds a fixed set of small kernel launches.
DEFAULT_ROUTED_CHUNK_BYTES = 64 * MiB
# Budgets an unset ``FSDPConfig.routed_expert_chunk_mib`` picks from, largest
# first. ROUTED_CHUNK_LIVE_FRACTION is not valid above 512 MiB.
ROUTED_CHUNK_MIB_CHOICES = (512, 256, 128, 64)
# Share of one chunk budget each chunked buffer adds to the MoE backward peak.
ROUTED_CHUNK_LIVE_FRACTION = 0.45
# Chunk-sized buffers the sorted path's contracted expert LoRA keeps live in
# the MoE backward: the up-projection output, its gradient and their casts.
CONTRACTED_CHUNK_LIVE_MULTIPLE = 4
# ``[routed rows, expert width]`` buffers expert parallel adds to the busiest
# rank's MoE backward peak: the send rows held for the comm stream, the
# received rows, the saved combine rows, and the gradients through both
# all-to-alls.
EP_EXCHANGE_BUFFERS = 4
# The same when uncheckpointed MoE blocks recompute their expert LoRA in
# backward. granite-3.1-3b-a800m and granite-4.0-h-tiny, A100 80GB, GC off,
# 18k-34k packed tokens: the granite-3.1 peaks sit a flat ~1.6 GiB above the
# saved tensors at every budget.
RECOMPUTED_CHUNK_LIVE_MULTIPLE = 7

# Share of a block's checkpointed-backward peak (:func:`block_recompute_bytes`)
# that a block left out of checkpointing keeps from its forward: the backward
# peak also holds a gradient that only the block in backward carries. Qwen3-4B
# LoRA r16 on A100 80GB (FA2 packed, 12k and 15k-token budgets, frozen base
# sharded) kept 0.90; 0.91 puts the per-token slope 1.4% above it. Fit on that
# one dense model only; other families are extrapolated.
STORED_BLOCK_SHARE = 0.91

# PyTorch caching-allocator slack: reserved above allocated.
# Charged on the torch-side subtotal. Training only: generation sits in
# vLLM's CuMem pool, reserved up front at ``gpu_memory_utilization``.
ALLOCATOR_RESERVE_FRACTION = 0.07
# Flat data parallel keeps an all-reduce buffer.
ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL = 0.054
# FSDP compute stream; the all-gather stream's pool is charged separately
# (:func:`agilerl.arena.memory.fsdp_gather.fsdp_gather_pool_params`). Blocks without fp32 temporaries free
# bf16 tensors the next activations reuse, so little stays free.
ALLOCATOR_RESERVE_FRACTION_FSDP = 0.02
# FSDP compute stream when freed blocks do not fit the next tensors: stock
# fp32 RMSNorm temporaries leave hidden-sized holes, and EP dispatch buffers
# change size with routing.
ALLOCATOR_RESERVE_FRACTION_FSDP_MIXED_SIZES = 0.064
# Largest share by which a measured peak has exceeded the estimate. A GPU
# recommendation keeps this share of usable memory free.
MAX_UNDERPREDICTION = 0.089


def recommendation_fits(predicted_bytes: int, usable_bytes: int) -> bool:
    """Whether a predicted peak still fits after the underprediction buffer."""
    return predicted_bytes <= int(usable_bytes * (1 - MAX_UNDERPREDICTION))


def allocator_reserve_bytes(
    allocated_bytes: float,
    n_ranks: int = 1,
    sharded: bool = False,
    mixed_sizes: bool = False,
) -> float:
    """Caching-allocator reserved-but-unallocated bytes on the compute stream.

    :param mixed_sizes: The step frees tensors the next activations do not
        fit (fp32 norm temporaries, expert-parallel dispatch buffers).
    """
    if sharded and mixed_sizes:
        fraction = ALLOCATOR_RESERVE_FRACTION_FSDP_MIXED_SIZES
    elif sharded:
        fraction = ALLOCATOR_RESERVE_FRACTION_FSDP
    elif n_ranks > 1:
        fraction = ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL
    else:
        fraction = ALLOCATOR_RESERVE_FRACTION
    return max(allocated_bytes, 0.0) * fraction


@dataclass(frozen=True)
class ParamCounts:
    """Parameter counts split by matrix group.

    LoRA targeting acts on a subset of groups. ``moe_experts`` holds the
    fused 3D expert parameters transformers stores per layer; PEFT wraps
    only ``nn.Linear``.
    """

    embedding: int
    lm_head: int
    attention: int
    mlp: int
    norms: int
    multimodal_towers: int
    moe_experts: int = 0
    # Mamba-2 mixer parameters (in/out projections, conv, gate norm).
    mamba: int = 0
    # Signed residual vs the checkpoint parameter count, held at the
    # checkpoint dtype.
    unattributed: int = 0

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.lm_head
            + self.attention
            + self.mlp
            + self.norms
            + self.multimodal_towers
            + self.moe_experts
            + self.mamba
            + self.unattributed
        )


def geometry_gap_warning(counts: ParamCounts) -> str | None:
    """Warn when parsed geometry leaves a large share of parameters unattributed.

    :param counts: Analytic parameter split from :func:`param_counts`.
    :return: Warning text, or ``None`` when the gap is small.
    """
    gap_warn_fraction = 0.02
    if counts.total <= 0:
        return None
    gap = counts.unattributed / counts.total
    if gap < gap_warn_fraction:
        return None
    return (
        f"{gap:.1%} of this checkpoint's parameters are not accounted for by "
        "the parsed geometry. Weight bytes are exact (checkpoint metadata); "
        "activation, KV and LoRA terms come from the geometry, so they are low."
    )


def tp_local(n: int, tp: int) -> int:
    """How many of ``n`` equal pieces one tensor-parallel rank holds.

    A count that does not divide ``tp`` stays whole. Key/value heads and
    Mamba groups that cannot be split are replicated.
    """
    if tp <= 1 or n % tp != 0:
        return n
    return n // tp


def tensor_parallel_params(counts: ParamCounts, arch: ModelArch) -> int:
    """Parameters placed on the tensor-parallel mesh.

    FSDP ignores these DTensors. An untied ``lm_head`` is in the set for every
    family; attention, dense MLP, and Mamba only for families with a training
    TP plan. Norms, token embeddings, the MoE router, and packed experts stay
    on the FSDP mesh.
    """
    head = 0 if arch.tied_embeddings else counts.lm_head
    if not arch.tensor_parallel_plan:
        return head
    _, router = moe_params_per_layer(arch)
    dense_mlp = counts.mlp - arch.moe_layers * router
    return counts.attention + dense_mlp + counts.mamba + head


def qkv_width_per_rank(arch: ModelArch, tp: int) -> int:
    """q+k+v elements one tensor-parallel rank materialises.

    Query heads always split. Key/value heads split only when they divide
    ``tp``; otherwise every rank keeps them.
    """
    head = arch.head_dim
    if arch.global_head_dim and arch.global_head_dim > head:
        head = arch.global_head_dim
    q = tp_local(arch.n_heads, tp) * head
    kv_heads = tp_local(arch.n_kv_heads, tp)
    return q + 2 * kv_heads * head


def attention_params_per_layer(arch: ModelArch) -> int:
    """q/k/v/o parameters of one attention block."""
    h, dh = arch.hidden_size, arch.head_dim
    q_dim = arch.n_heads * dh
    kv_dim = arch.n_kv_heads * dh
    attn_per_layer = h * q_dim + 2 * h * kv_dim + q_dim * h
    if arch.attn_bias:
        attn_per_layer += q_dim + 2 * kv_dim
    if arch.global_head_dim and arch.global_head_dim != dh:
        # Full-attention layers project to a wider head; scale the whole
        # attention block by the layer-averaged width.
        attn_per_layer = int(attn_per_layer * arch.mean_qkv_dim / (q_dim + 2 * kv_dim))
    return attn_per_layer


def mamba_params_per_layer(arch: ModelArch) -> int:
    """Parameters of one Mamba-2 mixer, from the same geometry as its state."""
    if not arch.n_mamba_layers:
        return 0
    h = arch.hidden_size
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    in_proj = h * (d_inner + arch.mamba_conv_dim + arch.mamba_n_heads)
    out_proj = d_inner * h
    conv = arch.mamba_conv_dim * (arch.mamba_d_conv + 1)
    return int(in_proj + out_proj + conv + d_inner + 3 * arch.mamba_n_heads)


def mlp_params_per_layer(arch: ModelArch) -> int:
    """Dense MLP parameters of one FFN block."""
    matrices = 3 if arch.gated_mlp else 2
    return int(
        matrices * arch.hidden_size * arch.intermediate_size * arch.mlp_width_factor
    )


def routed_expert_params_per_layer(arch: ModelArch) -> int:
    """Packed routed-expert parameters of one MoE block, the part expert parallel splits."""
    n_experts = arch.n_experts
    if n_experts is None or n_experts <= 1:
        return 0
    mlp_matrices = 3 if arch.gated_mlp else 2
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    return n_experts * mlp_matrices * arch.expert_width * expert_inter


def moe_params_per_layer(arch: ModelArch) -> tuple[int, int]:
    """(expert parameters, router parameters) of one MoE block.

    Latent MoE adds the hidden <-> latent projections to the expert count.
    """
    n_experts = arch.n_experts
    if n_experts is None or n_experts <= 1:
        return 0, 0
    h = arch.hidden_size
    mlp_matrices = 3 if arch.gated_mlp else 2
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    shared_inter = arch.shared_expert_intermediate_size or expert_inter
    experts = routed_expert_params_per_layer(arch)
    experts += arch.n_shared_experts * mlp_matrices * h * shared_inter
    if arch.moe_latent_size:
        experts += 2 * h * arch.moe_latent_size
    return experts, h * n_experts


def param_counts(arch: ModelArch, n_params: int | None = None) -> ParamCounts:
    """Analytic parameter counts from the architecture geometry.

    Each group is charged only on the layers that hold it: the full stack
    for a uniform model, the declared subsets for hybrids and
    block-exclusive layouts.

    :param n_params: Checkpoint total. The difference against the analytic
        sum goes into :attr:`ParamCounts.unattributed`.
    """
    h = arch.hidden_size
    experts_per_layer, router_per_layer = moe_params_per_layer(arch)
    mlp_per_layer = mlp_params_per_layer(arch)

    embedding = arch.vocab_size * h
    # Per-layer embeddings: one vector per layer per vocabulary entry.
    embedding += arch.per_layer_input_vocab * arch.n_layers * arch.per_layer_input_dim
    lm_head = 0 if arch.tied_embeddings else arch.vocab_size * h
    norms = arch.n_layers * 2 * h + h
    counts = ParamCounts(
        embedding=embedding,
        lm_head=lm_head,
        attention=arch.attention_layers * attention_params_per_layer(arch),
        mlp=arch.mlp_layers * mlp_per_layer + arch.moe_layers * router_per_layer,
        norms=norms,
        multimodal_towers=arch.multimodal_tower_params,
        moe_experts=arch.moe_layers * experts_per_layer,
        mamba=arch.n_mamba_layers * mamba_params_per_layer(arch),
    )
    if n_params is None:
        return counts
    return replace(counts, unattributed=n_params - counts.total)


def lora_param_count(
    arch: ModelArch, rank: int, scope: LoraTargetScope = "all-linear"
) -> int:
    """LoRA adapter parameters for the given rank and target scope.

    ``all-linear`` wraps every decoder ``nn.Linear``: attention q/k/v/o,
    dense MLP matrices, and Mamba in/out projections. The embedding and
    ``lm_head`` stay unwrapped. Each wrapped ``(out, in)`` linear adds
    ``rank * (in + out)``. Charged on the same layer subsets as
    :func:`param_counts`. Fused MoE experts live outside ``nn.Linear``,
    so a routed stack adds an MLP term only for its dense MLP blocks.
    """
    h, dh = arch.hidden_size, arch.head_dim
    q_dim = arch.n_heads * dh
    kv_dim = arch.n_kv_heads * dh
    attn = (h + q_dim) + 2 * (h + kv_dim) + (q_dim + h)
    if arch.global_head_dim and arch.global_head_dim != dh:
        # Full-attention layers adapt a wider head; scale the attention
        # adapters by the layer-averaged width.
        attn = int(attn * arch.mean_qkv_dim / (q_dim + 2 * kv_dim))
    total = attn * arch.attention_layers
    if scope != "all-linear":
        return rank * total
    mlp_matrices = 3 if arch.gated_mlp else 2
    total += arch.mlp_layers * mlp_matrices * (h + arch.intermediate_size)
    if arch.n_mamba_layers:
        d_inner = arch.mamba_n_heads * arch.mamba_d_head
        in_proj = h + d_inner + arch.mamba_conv_dim + arch.mamba_n_heads
        out_proj = d_inner + h
        total += arch.n_mamba_layers * (in_proj + out_proj)
    return rank * total


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


def weight_bytes(
    counts: ParamCounts,
    dtype: WeightDtype,
    variant: WeightVariant,
) -> int:
    """Realised base-weight bytes for one loaded copy.

    Analytic at the checkpoint dtype over the reconciled counts.
    ``stripped_multimodal`` drops the counted towers only.
    """
    towers = 0 if variant.stripped_multimodal else counts.multimodal_towers
    return int((counts.total - counts.multimodal_towers + towers) * DTYPE_BYTES[dtype])


def engine_weight_bytes(
    counts: ParamCounts,
    dtype: WeightDtype,
    variant: WeightVariant,
    quantization: Literal["none", "fp8"],
) -> int:
    """Base-weight bytes of one unsharded vLLM engine copy.

    vLLM's online fp8 stores linear, expert and Mamba projection weights at one
    byte. Embeddings, the LM head, norms and multimodal towers stay at ``dtype``.
    """
    full = weight_bytes(counts, dtype, variant)
    if quantization == "none":
        return full
    quantized = counts.attention + counts.mlp + counts.moe_experts + counts.mamba
    return int(full - quantized * (DTYPE_BYTES[dtype] - 1))


def block_kinds(arch: ModelArch) -> tuple[BlockKind, ...]:
    """Layer kinds a block-exclusive stack holds."""
    kinds: list[BlockKind] = []
    if arch.attention_layers:
        kinds.append("attention")
    if arch.mlp_layers or (arch.is_moe and arch.moe_layers):
        kinds.append("ffn")
    if arch.n_mamba_layers:
        kinds.append("mamba")
    return tuple(kinds)


def checkpointed_blocks(
    n_blocks: int, gradient_checkpointing: bool, every_n_blocks: int
) -> int:
    """Blocks activation checkpointing wraps: the first, then every ``every_n_blocks``-th.

    The remaining blocks keep their full activations until their backward.

    :param n_blocks: Blocks checkpointing may wrap.
    :param gradient_checkpointing: Activation checkpointing is on.
    :param every_n_blocks: Checkpoint stride over those blocks.
    :return: Checkpointed blocks.
    """
    if not gradient_checkpointing:
        return 0
    return -(-n_blocks // every_n_blocks)


def training_checkpointed_blocks(arch: ModelArch, settings: TrainingSettings) -> int:
    """Blocks the trainer checkpoints at the resolved stride.

    :param arch: Model architecture.
    :param settings: Training settings with checkpointing resolved.
    :return: Checkpointed blocks.
    """
    return checkpointed_blocks(
        arch.n_layers,
        settings.gradient_checkpointing is True,
        settings.checkpoint_every_n_blocks,
    )


def lora_linears(
    arch: ModelArch,
    scope: LoraTargetScope = "all-linear",
    tensor_parallel: int = 1,
    width_factor: float = 1.0,
) -> dict[BlockKind, list[tuple[int, int]]]:
    """``(in_features, out_features)`` of the linears PEFT wraps, per layer kind, in forward order.

    Routed experts live in fused 3D parameters (see ParamCounts.moe_experts),
    so the FFN kind tracks dense MLP blocks.

    :param arch: Decoder geometry.
    :param scope: LoRA target scope.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param width_factor: MLP intermediate width relative to ``intermediate_size``.
    :return: Linears of the attention, FFN and recurrent kinds; empty when unwrapped.
    """
    h = arch.hidden_size
    head_dim = max(arch.head_dim, arch.global_head_dim or 0)
    head_out = tp_local(arch.n_heads, tensor_parallel) * head_dim
    kv_out = tp_local(arch.n_kv_heads, tensor_parallel) * head_dim
    linears: dict[BlockKind, list[tuple[int, int]]] = {
        "attention": [(h, head_out), (h, kv_out), (h, kv_out), (head_out, h)],
        "ffn": [],
        "mamba": [],
    }
    if scope != "all-linear":
        return linears
    if arch.mlp_layers:
        inter = tp_local(int(arch.intermediate_size * width_factor), tensor_parallel)
        linears["ffn"] = [(h, inter)] * (2 if arch.gated_mlp else 1) + [(inter, h)]
    if arch.n_mamba_layers:
        heads_out = arch.mamba_n_heads * arch.mamba_d_head
        # Gated-delta-net LoRA wraps four residual-stream in_proj linears and
        # out_proj; Mamba-2 LoRA wraps in_proj only.
        linears["mamba"] = (
            [(h, heads_out)] * 4 + [(heads_out, h)]
            if arch.mamba_gated_delta
            else [(h, heads_out + arch.mamba_conv_dim + arch.mamba_n_heads)]
        )
    return linears


def lora_input_cast_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    scope: LoraTargetScope = "all-linear",
    gradient_checkpointing: bool = True,
    layer: BlockKind | None = None,
    tensor_parallel: int = 1,
) -> int:
    """fp32 LoRA inputs and outputs of an fp32 adapter compute over bf16 activations.

    Each wrapped linear casts its input to fp32
    (``BaseTunerLayer._cast_input_dtype``) and ``lora_A`` keeps that copy
    for backward. ``lora_B`` returns fp32 and is scaled out of place before
    the cast back to bf16, so two fp32 outputs are live while it runs.

    Under gradient checkpointing this is the recomputed block's peak: the
    casts kept so far plus one linear's two fp32 outputs, at whichever
    linear makes that largest. ``layer`` picks one layer kind; otherwise the
    block runs every kind it holds (the widest of a block-exclusive stack).
    A mixer on a stack whose layers carry an MLP beside it is followed by
    that MLP. Without checkpointing it is every layer's kept casts, which
    stay live until that layer's backward.
    """
    tokens_bytes = rows * seq_len * DTYPE_BYTES["fp32"]
    if not gradient_checkpointing:
        linears = lora_linears(arch, scope, tensor_parallel, arch.mlp_width_factor)
        layers = {
            "attention": arch.attention_layers,
            "ffn": arch.mlp_layers,
            "mamba": arch.n_mamba_layers,
        }
        kept = sum(
            layers[kind] * sum(width for width, _ in kind_linears)
            for kind, kind_linears in linears.items()
        )
        return int(tokens_bytes * kept)
    by_kind = lora_linears(arch, scope, tensor_parallel, arch.peak_mlp_width_factor)
    attention = by_kind["attention"] if arch.attention_layers else []
    mlp, mamba = by_kind["ffn"], by_kind["mamba"]
    if layer is not None:
        block = by_kind[layer]
        if layer != "ffn" and not arch.block_exclusive_layers:
            block = block + mlp
        candidates = [block]
    elif arch.block_exclusive_layers:
        candidates = [attention, mlp, mamba]
    elif arch.sublayer_transients:
        candidates = [attention + mlp, mamba + mlp]
    else:
        candidates = [attention + mamba + mlp]
    peak = 0
    for block in candidates:
        kept = 0
        for width, out in block:
            kept += width
            peak = max(peak, kept + 2 * out)
    return int(tokens_bytes * peak)


def lora_fp32_output_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    scope: LoraTargetScope = "all-linear",
    tensor_parallel: int = 1,
) -> int:
    """The two fp32 ``lora_B`` outputs of the widest wrapped linear.

    Live beside the kept casts while a layer that is not recomputed runs
    forward or backward.
    """
    linears = lora_linears(arch, scope, tensor_parallel, arch.peak_mlp_width_factor)
    present = {
        "attention": arch.attention_layers,
        "ffn": arch.mlp_layers,
        "mamba": arch.n_mamba_layers,
    }
    widest = max(
        (out for kind, block in linears.items() if present[kind] for _, out in block),
        default=0,
    )
    return int(rows * seq_len * 2 * widest * DTYPE_BYTES["fp32"])


def lora_only_stored_input_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int = 1,
) -> int:
    """Mean activation-dtype bytes per stored block that only ``lora_A`` keeps.

    That is the MLP down-projection input; the frozen projection saves none.
    A stored block drops it when ``lora_A`` keeps an fp32 cast instead, or
    when the MLP carries no adapter. The other wrapped inputs (normed rows,
    attention output) stay saved by the norm and attention ops: Qwen3-4B
    with fp32 adapter compute on H100, uncheckpointed, 6k tokens.
    """
    ffn = lora_linears(arch, "all-linear", tensor_parallel, arch.mlp_width_factor)[
        "ffn"
    ]
    if not ffn:
        return 0
    down_in = ffn[-1][0]
    return int(rows * seq_len * down_in * act_bytes * arch.mlp_layers / arch.n_layers)


# Expert-intermediate widths one gpt-oss routed row peaks at, forward and
# backward: gate_up, the clamped gate, its sigmoid, the GLU, up + 1 and the
# gated output stay saved.
CLAMPED_SWIGLU_FORWARD_INTERS = 8
CLAMPED_SWIGLU_BACKWARD_INTERS = 9


def routed_expert_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float, backward: bool = False
) -> int:
    """Live routed-expert tensors in one MoE block on the grouped-GEMM path.

    Per routed token (``tokens x top_k``): the gathered input and the expert
    output (:attr:`ModelArch.expert_width` each), gate/up (``2 x inter``) and
    the activation (``inter``). A checkpointed backward adds the activation's
    gradient. Nothing outlives the block.

    ``arch.chunked_routed_experts`` runs experts in row chunks and adds
    each chunk into one fp32 ``[tokens, expert_width]`` layer output, whose
    fp32 gradient backward holds. Backward keeps the gathered input, the
    expert output, the up projection and the activation for every chunk; a
    no-grad pass keeps only the gathered input. The per-chunk gradient
    buffers are :func:`routed_chunk_bytes`.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    h = arch.expert_width
    inter = arch.expert_intermediate_size or arch.intermediate_size
    if arch.transposed_experts:
        inters = (
            CLAMPED_SWIGLU_BACKWARD_INTERS
            if backward
            else CLAMPED_SWIGLU_FORWARD_INTERS
        )
        return int(routed * (2 * h + inters * inter) * act_bytes)
    if not arch.chunked_routed_experts:
        return int(routed * (2 * h + (4 if backward else 3) * inter) * act_bytes)
    width = 2 * h + (3 if arch.gated_mlp else 2) * inter if backward else h
    combined = rows * seq_len * h * DTYPE_BYTES["fp32"]
    return int(routed * width * act_bytes + combined)


def saved_router_topk_bytes(arch: ModelArch, rows: int, seq_len: int) -> int:
    """Router top-k outputs every checkpointed MoE block saves for backward.

    Selective checkpointing keeps each router's ``(tokens, top_k)`` scores
    and int64 expert indices on device until that block's backward, so
    recompute replays the forward routing. Activation offload does not
    move them. Scores are charged at fp32, the widest a router picks on.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    slot_bytes = DTYPE_BYTES["fp32"] + 8
    return int(routed * arch.moe_layers * slot_bytes)


def ep_exchange_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    expert_parallel: int,
    tensor_parallel: int = 1,
) -> int:
    """Token-exchange buffers of one expert-parallel MoE block at its backward peak.

    A rank sends its routed rows and receives about as many for its own
    experts. Tensor-parallel ranks share their tokens, so each routes 1/tp.
    """
    if not arch.is_moe or expert_parallel <= 1:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1) // tensor_parallel
    return int(EP_EXCHANGE_BUFFERS * routed * arch.expert_width * act_bytes)


def routed_chunk_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    chunk_bytes: int,
    expert_lora: bool,
    contracted: bool,
    ep: int,
    recomputed: bool = False,
) -> int:
    """Backward bytes of one MoE block that grow with the routed-expert chunk budget.

    Each buffer is capped by the rows it chunks. The chunked routed path
    (expert LoRA on ``arch.chunked_routed_experts``) and the expert-parallel
    fp32 combine each add :data:`ROUTED_CHUNK_LIVE_FRACTION` of a chunk. The
    sorted path's contracted LoRA holds
    :data:`CONTRACTED_CHUNK_LIVE_MULTIPLE` whole chunks, or
    :data:`RECOMPUTED_CHUNK_LIVE_MULTIPLE` when the experts recompute in
    backward; transposed experts (``arch.transposed_experts``) hold as many
    buffers over every routed row.

    :param chunk_bytes: Widest ``[rows, features]`` activation of one chunk.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param contracted: Expert LoRA runs split (``packed_moe_dispatch="contracted"``).
    :param ep: Expert-parallel degree.
    :param recomputed: Some MoE block runs outside checkpointing, so its
        expert LoRA recomputes the up-projection in backward.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    inter = arch.expert_intermediate_size or arch.intermediate_size
    widest_row = max(arch.expert_width, inter) * act_bytes
    fractional = 0.0
    whole = 0.0
    if expert_lora and arch.chunked_routed_experts:
        fractional += min(chunk_bytes, routed * widest_row)
    elif expert_lora and contracted:
        multiple = (
            RECOMPUTED_CHUNK_LIVE_MULTIPLE
            if recomputed
            else CONTRACTED_CHUNK_LIVE_MULTIPLE
        )
        held = routed * widest_row
        if not arch.transposed_experts:
            held = min(chunk_bytes, held)
        whole += multiple * held
    if ep > 1:
        fractional += min(chunk_bytes, routed * arch.expert_width * 4)
    return int(fractional * ROUTED_CHUNK_LIVE_FRACTION + whole)


def mamba_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    unfused_scan: bool = False,
    tensor_parallel: int = 1,
) -> int:
    """Live tensors of one Mamba-2 layer's chunked scan in the trainer.

    Per token: the ``in_proj`` output, the conv1d output, the gated scan
    input and an fp32 copy of the residual stream. Per chunk: the fp32 SSM
    state ``(heads, d_head, d_state)`` and the ``chunk x chunk`` CB matrix
    per group. A checkpointed backward adds the gated-norm gradient, fp32
    gradients of the ``in_proj`` output and the residual, and two more
    state buffers.

    ``unfused_scan`` adds the backward of HF's unfused path (separate conv,
    chunk scan, gated norm and output projection, the SSM inputs upcast to
    fp32), :data:`UNFUSED_MAMBA_SCAN_BYTES` per token per inner channel.
    """
    if not arch.n_mamba_layers:
        return 0
    tokens = rows * seq_len
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    groups = tp_local(arch.mamba_n_groups, tensor_parallel)
    d_inner = heads * arch.mamba_d_head
    conv_dim = arch.mamba_conv_dim
    if heads != arch.mamba_n_heads and arch.mamba_n_heads:
        conv_dim = arch.mamba_conv_dim * heads // arch.mamba_n_heads
    in_proj_out = d_inner + conv_dim + heads
    chunks = rows * -(-seq_len // arch.mamba_chunk_size)
    state = chunks * heads * arch.mamba_d_head * arch.mamba_d_state * 4
    cb = chunks * groups * arch.mamba_chunk_size**2 * 4
    per_token = (in_proj_out + arch.mamba_conv_dim + d_inner) * act_bytes
    per_token += arch.hidden_size * 4
    total = tokens * per_token + state + cb
    if backward:
        per_token_grads = d_inner * act_bytes + (in_proj_out + arch.hidden_size) * 4
        total += tokens * per_token_grads + 2 * state
    if unfused_scan and backward:
        total += tokens * d_inner * UNFUSED_MAMBA_SCAN_BYTES
    return int(total)


def gated_delta_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
) -> int:
    """Live tensors of one gated-delta-net layer's fla chunk kernel in the trainer.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param backward: Whether the layer runs a checkpointed backward.
    :return: Peak bytes of the kernel's tensors; zero without gated-delta layers.
    """
    if not (arch.n_mamba_layers and arch.mamba_gated_delta):
        return 0
    heads = arch.mamba_n_heads
    key = heads * arch.mamba_d_state
    value = heads * arch.mamba_d_head
    state = heads * arch.mamba_d_state * arch.mamba_d_head / GATED_DELTA_CHUNK
    # Per token: conv1d output, contiguous q/k/v at the value-head count,
    # l2-normed q and k, gate z, WY w and u, updated values, output, the
    # intra-chunk matrix, and one (d_state, d_head) state per head per chunk.
    width = (
        arch.mamba_conv_dim + 5 * key + 5 * value + heads * GATED_DELTA_CHUNK + state
    )
    # A checkpointed backward also keeps the in_proj_qkv output and its gate.
    if backward:
        width += arch.mamba_conv_dim + value
    return int(rows * seq_len * width * act_bytes)


def gated_delta_saved_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
) -> int:
    """Tensors a recomputed gated-delta-net layer keeps for its own backward.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :return: Saved bytes; zero without gated-delta layers.
    """
    if not (arch.n_mamba_layers and arch.mamba_gated_delta):
        return 0
    heads = arch.mamba_n_heads
    key = heads * arch.mamba_d_state
    value = heads * arch.mamba_d_head
    # Per token: in_proj_qkv output, l2-normed q and k at the value-head
    # count, contiguous v, the intra-chunk matrix, and the gated RMSNorm's
    # scaled input and output plus its fp32 input, gate and SiLU.
    width = arch.mamba_conv_dim + 2 * key + 3 * value + heads * GATED_DELTA_CHUNK
    return int(rows * seq_len * (width * act_bytes + 3 * value * 4))


def _block_mlp_recompute(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    engine_forward: bool,
    tensor_parallel: int,
    routed_split: int = 1,
) -> tuple[int, int]:
    """Per-token MLP width and routed-expert bytes of one block recompute.

    :param routed_split: Ranks that share one sequence's routed tokens
        (TP ranks under expert parallel).
    :return: ``(mlp_width, routed_bytes)``. ``mlp_width`` is elements per
        token. ``routed_bytes`` is the grouped-GEMM workspace; zero on the
        engine path and on dense models.
    """
    mlp_matrices = 3 if arch.gated_mlp else 2
    routed = 0
    if arch.is_moe:
        inter = arch.expert_intermediate_size or arch.intermediate_size
        shared_inter = arch.shared_expert_intermediate_size or inter
        # The always-on shared expert runs at its own (possibly wider) width.
        active_width = arch.n_shared_experts * shared_inter
        if engine_forward:
            active_width += (arch.n_experts_per_tok or 1) * inter
        else:
            routed = routed_expert_bytes(
                arch, rows, -(-seq_len // routed_split), act_bytes, backward
            )
    else:
        active_width = tp_local(
            int(arch.intermediate_size * arch.peak_mlp_width_factor), tensor_parallel
        )
    # Gated backward: gate, up, fused output, and one gradient, plus the
    # activated gate when the MLP is not fused.
    mlp_count = mlp_matrices
    if backward and arch.gated_mlp:
        mlp_count += 2 if arch.stock_gated_mlp else 1
    mlp = mlp_count * active_width
    if backward and not arch.gated_mlp:
        # Activation plus its gradient for each of the two matrices.
        mlp *= 2
    return mlp, routed


def block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    engine_forward: bool = False,
    layer: BlockKind | None = None,
    unfused_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    routed_split: int = 1,
) -> int:
    """Peak transient activations of one transformer block.

    The forward peak holds four residual copies, qkv, and the MLP outputs.
    A checkpointed backward holds two residual copies, qkv, and the gated
    MLP: gate, up, the fused output, and one gradient. A Q/K-norm model
    (``arch.qk_norm``) also holds one fp32 copy of Q. Flash-style backends
    tile the score matrix. A stock RMSNorm (``arch.fp32_norm_inputs``)
    keeps an fp32 copy of each norm input on the backward.

    In the trainer, routed experts run the grouped-GEMM path
    (:func:`routed_expert_bytes`), Mamba mixers their chunked scan
    (:func:`mamba_block_bytes`) and gated-delta-net mixers the fla chunk
    kernel (:func:`gated_delta_block_bytes`). When
    :attr:`ModelArch.sublayer_transients` holds, a block peaks in one
    attention, FFN or recurrent sub-layer: ``layer`` picks one, else the
    widest counts; otherwise a block holds all of them. On a stack whose
    layers carry an FFN beside the mixer, a checkpointed backward charges
    the mixer's saved tensors and that FFN together; a gated-delta-net
    layer peaks at the larger of that and its kernel's own backward
    (:func:`gated_delta_saved_bytes`).
    ``engine_forward=True`` sizes routed experts as the active FFN width
    (vLLM's in-kernel routing).
    """
    h = arch.hidden_size
    # One block's transient peaks at the widest block: checkpointed
    # recompute and the engine's layer-at-a-time forward run block by block.
    # The residual stays full width. Query/MLP/Mamba projections narrow on
    # the training mesh; the engine path sizes its own tensor parallel.
    residual = (2 if backward else 4) * h
    qkv_dim = (
        arch.peak_qkv_dim
        if engine_forward
        else qkv_width_per_rank(arch, tensor_parallel)
    )
    mlp, routed = _block_mlp_recompute(
        arch,
        rows,
        seq_len,
        act_bytes,
        backward,
        engine_forward,
        1 if engine_forward else tensor_parallel,
        routed_split,
    )
    ple = arch.n_layers * arch.per_layer_input_dim
    if backward:
        ple *= 2
    tokens = rows * seq_len
    if engine_forward:
        return int(tokens * (4 * h + qkv_dim + mlp + ple) * act_bytes)
    if arch.mamba_gated_delta:
        mamba = tokens * residual * act_bytes + gated_delta_block_bytes(
            arch, rows, seq_len, act_bytes, backward
        )
    else:
        mamba = mamba_block_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            backward,
            unfused_scan=unfused_mamba_scan,
            tensor_parallel=1 if engine_forward else tensor_parallel,
        )
    # fp32 RMSNorm inputs saved for backward, one per norm in the block.
    norm_input = tokens * h * 4 if backward and arch.fp32_norm_inputs else 0
    q_heads = (
        arch.n_heads if engine_forward else tp_local(arch.n_heads, tensor_parallel)
    )
    q_norm = tokens * q_heads * arch.head_dim * 4 if backward and arch.qk_norm else 0

    def sublayer_peak() -> int:
        ffn = tokens * (mlp + ple) * act_bytes + routed + norm_input
        # Both recurrent terms hold the layer's residual copy.
        by_layer = {
            "attention": tokens * (residual + qkv_dim) * act_bytes
            + norm_input
            + q_norm,
            "ffn": tokens * residual * act_bytes + ffn,
            "mamba": mamba + norm_input,
        }
        if backward and not arch.block_exclusive_layers:
            # Recompute rebuilds the whole layer, so the mixer's saved
            # tensors stay live through the backward of the FFN beside it.
            # The recurrent kernel's own backward runs once the FFN's are freed.
            by_layer["attention"] += ffn
            mamba_saved = (
                tokens * residual * act_bytes
                + gated_delta_saved_bytes(arch, rows, seq_len, act_bytes)
                + norm_input
            )
            by_layer["mamba"] = max(by_layer["mamba"], mamba_saved + ffn)
        if layer is not None:
            return int(by_layer[layer])
        return int(max(by_layer.values()))

    if arch.sublayer_transients:
        return sublayer_peak()
    return int(
        tokens * (residual + qkv_dim + mlp + ple) * act_bytes
        + routed
        + mamba
        + arch.norms_per_block * norm_input
        + q_norm
    )


def mean_block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    unfused_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    routed_split: int = 1,
) -> int:
    """Checkpointed-backward peak of one block, averaged over the stack's layers.

    Each uncheckpointed layer keeps its own mixer's tensors, so a stack whose
    every layer runs one mixer beside its FFN weights the attention and
    recurrent :func:`block_recompute_bytes` by their layer counts: Qwen3.5
    LoRA on A100 80GB, GC off and every second or fourth block checkpointed,
    16k-32k packed tokens. Other stacks take the widest block.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param unfused_mamba_scan: Mamba layers run HF's unfused path.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param routed_split: Ranks that share one sequence's routed tokens.
    :return: Mean bytes of one block.
    """

    def peak(layer: BlockKind | None) -> int:
        return block_recompute_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            backward=True,
            layer=layer,
            unfused_mamba_scan=unfused_mamba_scan,
            tensor_parallel=tensor_parallel,
            routed_split=routed_split,
        )

    if not arch.sublayer_transients or arch.block_exclusive_layers:
        return peak(None)
    return int(
        (
            arch.attention_layers * peak("attention")
            + arch.n_mamba_layers * peak("mamba")
        )
        / arch.n_layers
    )


def prices_stored_tensors(arch: ModelArch) -> bool:
    """Whether :func:`stored_block_bytes` prices this model's uncheckpointed blocks.

    :param arch: Model architecture.
    :return: True for a MoE stack whose every layer holds a mixer and the
        experts, on the self-routing expert LoRA path.
    """
    return (
        arch.is_moe
        and arch.routed_expert_recompute
        and not arch.block_exclusive_layers
        and not arch.mamba_gated_delta
    )


def stored_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    expert_lora: bool,
    mamba_lora: bool,
    expert_parallel: int = 1,
    routed_split: int = 1,
) -> int:
    """Tensors one uncheckpointed block keeps for its backward, averaged over layer kinds.

    Per token: each RMSNorm's input (fp32 under the stock module) and the
    normed rows the experts read. An attention layer adds the normed rows its
    LoRA reads, q, k, v and the attention output. A Mamba-2 layer adds the
    fused kernel's ``in_proj`` and scan outputs, and with LoRA on ``in_proj``
    and ``out_proj`` their inputs. The router keeps an fp32 score and an
    int64 index per routed slot. Expert LoRA recomputes the experts in
    backward and keeps another score and index per slot; frozen experts keep
    gate, up and the expert output. A shared expert keeps gate and up, plus
    the activated gate under the stock MLP. Expert parallel keeps one
    ``expert_width`` row per routed slot for the combine's router-weight
    product, and with expert LoRA four: the received rows the recompute
    reads, the combine rows, and two more the measured peaks show.

    :param arch: Model architecture; :func:`prices_stored_tensors` holds.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param mamba_lora: Mamba ``in_proj`` and ``out_proj`` carry LoRA.
    :param expert_parallel: Expert-parallel degree.
    :param routed_split: Ranks that share one sequence's routed tokens
        (TP ranks under expert parallel).
    :return: Mean saved bytes of one block.
    """
    h = arch.hidden_size
    tokens = rows * seq_len
    norm_bytes = DTYPE_BYTES["fp32"] if arch.fp32_norm_inputs else act_bytes
    per_token = arch.norms_per_block * h * norm_bytes + h * act_bytes
    slots = arch.n_experts_per_tok or 1
    slot_bytes = DTYPE_BYTES["fp32"] + 8
    inter = arch.expert_intermediate_size or arch.intermediate_size
    if expert_lora:
        per_token += 2 * slots * slot_bytes
    else:
        per_token += slots * (slot_bytes + (2 * inter + arch.expert_width) * act_bytes)
    if expert_parallel > 1:
        # granite-3.1-3b-a800m GRPO LoRA, ep=2, GC off, 2x A100 80GB: peaks
        # grow 3.70 GiB per 1k packed tokens at 8k-16k, 3.9 rows per slot
        # above the same model without expert parallel.
        ep_rows = 4 if expert_lora else 1
        per_token += ep_rows * slots * arch.expert_width * act_bytes / routed_split
    shared_inter = arch.shared_expert_intermediate_size or inter
    shared_matrices = 3 if arch.stock_gated_mlp else 2
    per_token += arch.n_shared_experts * shared_matrices * shared_inter * act_bytes
    q_dim = arch.n_heads * arch.head_dim
    attention = (h + arch.peak_qkv_dim + q_dim) * act_bytes
    if arch.qk_norm:
        attention += q_dim * DTYPE_BYTES["fp32"]
    mamba = mamba_saved_width(arch, mamba_lora) * act_bytes
    mixers = arch.attention_layers * attention + arch.n_mamba_layers * mamba
    return int(tokens * (per_token + mixers / arch.n_layers))


def mamba_saved_width(arch: ModelArch, lora: bool) -> int:
    """Elements per token one Mamba-2 layer on the fused split-scan kernel saves for backward.

    The kernel keeps its ``in_proj`` output and its scan output. LoRA on
    ``in_proj`` and ``out_proj`` adds their inputs: the normed hidden rows and
    the gated-norm output.

    :param arch: Model architecture.
    :param lora: ``in_proj`` and ``out_proj`` carry LoRA.
    :return: Saved elements per token; zero without Mamba layers.
    """
    if not arch.n_mamba_layers:
        return 0
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    width = 2 * d_inner + arch.mamba_conv_dim + arch.mamba_n_heads
    if lora:
        width += arch.hidden_size + d_inner
    return width


def nograd_forward_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    unfused_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    routed_split: int = 1,
) -> int:
    """Peak transient memory of the fused no-grad logprob forward.

    Nothing is saved for backward: one block's transients plus the
    full-sequence hidden state handed to the fused logprob kernel.
    """
    block = block_recompute_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        unfused_mamba_scan=unfused_mamba_scan,
        tensor_parallel=tensor_parallel,
        routed_split=routed_split,
    )
    hidden_out = rows * seq_len * arch.hidden_size * act_bytes
    return int(block + hidden_out)


def support_rows_bytes(
    arch: ModelArch, chunk_rows: int, width: int, act_bytes: float
) -> int:
    """Peak of one support-scored logprob chunk: its tokens' gathered lm_head rows.

    A chunk takes enough tokens for ``width`` head rows each to fill one vocab
    tile, or one token when its rows alone exceed the tile. Rows gather in the
    head's dtype; below fp32 the upcast holds both copies.

    :param chunk_rows: Tokens in one vocab tile.
    :param width: Widest sampling support, in token ids.
    :param act_bytes: Bytes per lm_head element.
    """
    tokens = max(1, chunk_rows * arch.vocab_size // (width * arch.hidden_size))
    elements = tokens * width * arch.hidden_size
    return int(elements * (act_bytes + 4 if act_bytes < 4 else act_bytes))


def logit_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
) -> tuple[float, float]:
    """Loss-instant logits and no-grad head bytes."""
    # Two fp32 tiles are live at the loss instant: recomputed logits plus
    # that tile's gradient. Tiles are fixed-size, so a fused critic row adds
    # none. The head matmul reads the lm_head in its own dtype, so no copy of
    # it is live.
    logit_rows = resolve_chunk_rows(arch.vocab_size)
    if settings.sampling_support_width:
        support_rows = support_rows_bytes(
            arch, logit_rows, settings.sampling_support_width, act_bytes
        )
        return support_rows, support_rows
    return logit_rows * arch.vocab_size * 8, logit_rows * arch.vocab_size * 4


# Extra backward bytes per token per Mamba inner channel when the mixer runs
# HF's unfused path: separate conv, chunk scan, gated norm and output
# projection, with the SSM inputs upcast to fp32.
UNFUSED_MAMBA_SCAN_BYTES = 28

# Chunk length of fla's chunked gated-delta-rule kernel.
GATED_DELTA_CHUNK = 64

# Per-process job overhead charged when ``orchestrated`` is set.
JOB_OVERHEAD_BYTES = 50 * MiB

# Non-torch device memory a one-rank trainer holds beyond the CUDA context
# and job overhead. Multi-rank trainers charge :func:`agilerl.arena.memory.nccl.trainer_nccl_bytes`.
TRAINER_LIB_OVERHEAD_BYTES = 436 * MiB


def lora_dropout_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    adapter_bytes: float,
    scope: LoraTargetScope = "all-linear",
    gradient_checkpointing: bool = True,
    layer: BlockKind | None = None,
    tensor_parallel: int = 1,
) -> int:
    """Dropped-out adapter inputs and their masks that autograd keeps.

    With ``lora_dropout > 0`` PEFT feeds each wrapped linear's adapter a
    dropped-out copy of its input at the adapter dtype and keeps a one-byte
    mask for backward. Under gradient checkpointing one block's copies are
    live at once; otherwise every layer's are. When
    :attr:`ModelArch.sublayer_transients` holds, a block's peak sits in one
    attention, MLP or recurrent sub-layer: ``layer`` picks one, else the
    widest counts. A mixer on a stack whose layers carry an MLP beside it
    adds that MLP. A parallel mixer stack charges all three in one block.
    """
    h = arch.hidden_size
    # q/k/v read the residual stream, o reads the concatenated heads.
    head_out = tp_local(
        arch.n_heads * max(arch.head_dim, arch.global_head_dim or 0), tensor_parallel
    )
    attention = 3 * h + head_out
    mlp = 0
    if scope == "all-linear" and arch.mlp_layers:
        inter = tp_local(
            int(arch.intermediate_size * arch.peak_mlp_width_factor), tensor_parallel
        )
        mlp = (2 * h if arch.gated_mlp else h) + inter
    mamba = 0
    if scope == "all-linear" and arch.n_mamba_layers:
        # Gated-delta-net LoRA wraps four residual-stream in_proj linears and
        # out_proj; Mamba-2 LoRA wraps in_proj only.
        mamba = (
            4 * h + arch.mamba_n_heads * arch.mamba_d_head
            if arch.mamba_gated_delta
            else h
        )
    if layer is not None:
        width = {"attention": attention, "ffn": mlp, "mamba": mamba}[layer]
        if layer != "ffn" and not arch.block_exclusive_layers:
            width += mlp
    elif arch.block_exclusive_layers:
        width = max(attention, mlp, mamba)
    elif arch.sublayer_transients:
        width = max(attention, mamba) + mlp
    else:
        width = attention + mlp + mamba
    per_block = rows * seq_len * width * (adapter_bytes + 1)
    if gradient_checkpointing:
        return int(per_block)
    return int(per_block * arch.n_layers)


def token_embedding_params(arch: ModelArch) -> int:
    """Token-embedding elements replicated on every FSDP rank."""
    return arch.vocab_size * arch.hidden_size
