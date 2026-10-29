# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form memory terms for LLM RL training and generation.

Every function is a pure function of the specs in
:mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agilerl.arena.memory.resolve.chunk import resolve_chunk_rows
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    BlockKind,
    LayerKind,
    LoraTargetScope,
    MiB,
    ModelArch,
    TrainingSettings,
    WeightDtype,
    WeightVariant,
)

# Upper bound on the CUDA-graph pool the vLLM engine captures unless
# enforce_eager. It cancels in the generation total: subtracted from the KV
# budget, added back as residency.
CUDA_GRAPH_POOL_BYTES = 256 * MiB
# vLLM's non-torch memory (kernel workspaces, loaded kernels, NCCL), as a share
# of the unsharded engine weights. It sits inside the gpu_memory_utilization
# budget and every rank holds it whole: it does not shrink with TP.
ENGINE_RUNTIME_FRACTION = 0.036
# Bytes per LoRA parameter under flat data parallel. PEFT
# ``autocast_adapter_dtype=True`` stores adapters and their gradients in
# fp32. AdamW keeps two moments per parameter in the parameter's dtype.
ADAPTER_BYTES_PER_PARAM = 4.0
ADAM_MOMENTS_PER_PARAM = 2
# Foreach AdamW builds one fp32 temporary per trained parameter in step();
# the fused kernel builds none.
FOREACH_ADAM_WORKSPACE_BYTES_PER_PARAM = 4.0
# Routed-expert chunk budget when no FSDP config sets one.
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

# Bytes per [row, vocab] element of a Liger loss chunk that holds one whole
# sequence: the bf16 logits plus fp32 upcast, log-softmax and gradient.
LIGER_SEQUENCE_LOGIT_BYTES = 14

# PyTorch caching-allocator slack: reserved above allocated.
# Charged on the torch-side subtotal. Training only: generation sits in
# vLLM's CuMem pool, reserved up front at ``gpu_memory_utilization``.
ALLOCATOR_RESERVE_FRACTION = 0.07
# Flat data parallel keeps an all-reduce buffer.
ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL = 0.054
# FSDP compute stream; the all-gather stream's pool is charged separately
# (:func:`fsdp_gather_pool_params`). Blocks without fp32 temporaries free
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


def kv_heads_per_rank(n_kv_heads: int, tensor_parallel: int) -> int:
    """KV heads stored on one vLLM rank.

    vLLM divides the heads by the tensor-parallel size. Fewer heads than
    ranks are replicated, so each rank keeps at least one.
    """
    return max(1, n_kv_heads // tensor_parallel)


def kv_storing_layers(arch: ModelArch) -> int:
    """Attention layers that keep their own K/V."""
    return arch.attention_layers - min(arch.n_kv_shared_layers, arch.attention_layers)


def kv_layer_head_dims(arch: ModelArch) -> tuple[float, int, int]:
    """(windowed layer fraction, sliding head dim, full-attention head dim).

    Full-attention layers may store a wider head. With no sliding window
    every storing layer uses the full-attention width.
    """
    window_dim = arch.head_dim
    full_dim = arch.global_head_dim or arch.head_dim
    if arch.sliding_window is None:
        return 0.0, window_dim, full_dim
    return arch.sliding_window_layer_fraction, window_dim, full_dim


def kv_cache_bytes_per_token(arch: ModelArch, weight_dtype: WeightDtype) -> float:
    """K + V bytes per cached token position across all layers.

    Layers that share an earlier layer's KV store nothing of their own.
    Hybrid Mamba layers contribute recurrent state only, constant in
    context length and counted by :func:`mamba_state_bytes`.

    Storing layers are charged at their own head width: sliding layers at
    :attr:`ModelArch.head_dim`, full-attention layers at
    :attr:`ModelArch.global_head_dim` when that is wider.
    """
    kv_bytes = DTYPE_BYTES[weight_dtype]
    windowed, window_dim, full_dim = kv_layer_head_dims(arch)
    mean_head_dim = windowed * window_dim + (1.0 - windowed) * full_dim
    return 2 * kv_storing_layers(arch) * arch.n_kv_heads * mean_head_dim * kv_bytes


# vLLM's default KV block size.
KV_BLOCK_SIZE_DEFAULT = 16
# vLLM's max_num_batched_tokens when the engine config is built with no usage context.
VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS = 2048
# Scheduler steps in flight under vLLM's async scheduling, each up to
# max_num_batched_tokens; sliding-window layers keep their blocks meanwhile.
VLLM_ASYNC_CONCURRENT_BATCHES = 2
# Alignment the hybrid KV block size is rounded up to.
MAMBA_KERNEL_BLOCK_ALIGNMENT = 16
# Recurrent-state blocks resident per sequence. vLLM prefix caching
# resolves every hybrid to align mode, which caps residency at two.
MAMBA_STATE_BLOCKS_PER_SEQ = 2
# Align-mode prefill checkpoint block a Mamba-2 mixer adds per sequence.
MAMBA2_PREFILL_CHECKPOINT_BLOCKS = 1


def mamba_page_bytes(arch: ModelArch, state_dtype: WeightDtype = "bf16") -> int:
    """One recurrent layer's state: the conv window plus the SSM state.

    The conv window uses ``state_dtype``; the SSM state uses
    :attr:`ModelArch.mamba_ssm_state_dtype`.
    """
    if not arch.is_hybrid_ssm:
        return 0
    conv = arch.mamba_conv_dim * max(arch.mamba_d_conv - 1, 0)
    recurrent = arch.mamba_n_heads * arch.mamba_d_head * arch.mamba_d_state
    return int(
        conv * DTYPE_BYTES[state_dtype]
        + recurrent * DTYPE_BYTES[arch.mamba_ssm_state_dtype]
    )


def shares_hybrid_pool(arch: ModelArch) -> bool:
    """Whether Mamba state and attention KV share one paged vLLM pool."""
    return arch.is_hybrid_ssm and arch.n_kv_heads > 0 and kv_storing_layers(arch) > 0


def attention_token_bytes(
    arch: ModelArch, weight_dtype: WeightDtype = "bf16", tensor_parallel: int = 1
) -> int:
    """K + V bytes one full-attention layer stores per token on one vLLM rank."""
    _, _, full_dim = kv_layer_head_dims(arch)
    heads = kv_heads_per_rank(arch.n_kv_heads, tensor_parallel)
    return int(2 * heads * full_dim * DTYPE_BYTES[weight_dtype])


def aligned_kv_block_size(
    arch: ModelArch,
    weight_dtype: WeightDtype = "bf16",
    tensor_parallel: int = 1,
) -> int:
    """Block size vLLM adopts so an attention page holds a Mamba page.

    A hybrid model puts both caches in one block pool, so the page sizes
    must match. vLLM raises the attention block size until one rank's
    attention page is at least as large as its Mamba page, then pads the
    Mamba page up to it. The block size is then hundreds or thousands of
    tokens, and every sequence's KV rounds up to that page.
    """
    if not shares_hybrid_pool(arch):
        return KV_BLOCK_SIZE_DEFAULT
    per_token_per_layer = attention_token_bytes(arch, weight_dtype, tensor_parallel)
    mamba_page = mamba_page_per_rank(arch, weight_dtype, tensor_parallel)
    tokens = -(-mamba_page // per_token_per_layer)
    alignment = MAMBA_KERNEL_BLOCK_ALIGNMENT
    return max(KV_BLOCK_SIZE_DEFAULT, -(-tokens // alignment) * alignment)


def hybrid_page_bytes(
    arch: ModelArch, weight_dtype: WeightDtype = "bf16", tensor_parallel: int = 1
) -> int:
    """One layer's page in a hybrid pool on one rank; Mamba pages pad up to it."""
    return aligned_kv_block_size(
        arch, weight_dtype, tensor_parallel
    ) * attention_token_bytes(arch, weight_dtype, tensor_parallel)


def hybrid_kv_groups(arch: ModelArch) -> tuple[int, int, int]:
    """(layers per group, attention groups, Mamba groups) of a hybrid KV pool.

    vLLM splits each layer type into groups the size of the smaller type, or
    of the larger when it is under 1.5x the smaller, and pads the last group
    of each type. One pool block holds one page for every layer of a group,
    and a request takes blocks from every group.
    """
    attention = kv_storing_layers(arch)
    mamba = arch.n_mamba_layers
    size = min(attention, mamba)
    if max(attention, mamba) < 1.5 * size:
        size = max(attention, mamba)
    return size, -(-attention // size), -(-mamba // size)


def mamba_state_blocks_per_seq(arch: ModelArch) -> int:
    """Recurrent-state blocks one sequence holds in each Mamba group."""
    if arch.mamba_gated_delta:
        return MAMBA_STATE_BLOCKS_PER_SEQ
    return MAMBA_STATE_BLOCKS_PER_SEQ + MAMBA2_PREFILL_CHECKPOINT_BLOCKS


def mamba_page_per_rank(
    arch: ModelArch, state_dtype: WeightDtype = "bf16", tensor_parallel: int = 1
) -> int:
    """One Mamba layer's state on one vLLM rank.

    vLLM shards the conv window and the SSM state by the tensor-parallel
    size. Heads must divide that size. Groups that do not divide it are
    replicated up to one group per rank, which vLLM only allows when there
    is a single group.
    """
    if tensor_parallel == 1 or not arch.is_hybrid_ssm:
        return mamba_page_bytes(arch, state_dtype)
    heads = arch.mamba_n_heads
    groups = arch.mamba_n_groups
    if heads % tensor_parallel != 0:
        msg = (
            f"mamba_num_heads ({heads}) must be divisible by "
            f"tensor_parallel_size ({tensor_parallel})"
        )
        raise ValueError(msg)
    if groups % tensor_parallel != 0 and groups != 1:
        msg = (
            f"mamba n_groups ({groups}) must divide tensor_parallel_size "
            f"({tensor_parallel}), or be 1"
        )
        raise ValueError(msg)
    if groups % tensor_parallel != 0:
        groups = tensor_parallel
    width = heads * arch.mamba_d_head
    conv_dim = (width + 2 * groups * arch.mamba_d_state) // tensor_parallel
    recurrent = (heads // tensor_parallel) * arch.mamba_d_head * arch.mamba_d_state
    return int(
        conv_dim * max(arch.mamba_d_conv - 1, 0) * DTYPE_BYTES[state_dtype]
        + recurrent * DTYPE_BYTES[arch.mamba_ssm_state_dtype]
    )


def mamba_state_bytes(
    arch: ModelArch,
    concurrency: int,
    state_dtype: WeightDtype = "bf16",
    tensor_parallel: int = 1,
) -> int:
    """Recurrent-state cache for a hybrid model's Mamba layers.

    Two tensors per recurrent layer per slot, matching vLLM's
    ``mamba2_state_shape``: a causal-conv window of ``(conv_dim, d_conv - 1)``
    and a recurrent state of ``(n_heads, d_head, d_state)``. Both are
    constant in context length, so this term scales with concurrency.
    ``tensor_parallel`` applies :func:`mamba_page_per_rank`. Beside attention
    each layer is charged the padded :func:`hybrid_page_bytes`, over the
    padded layer groups of :func:`hybrid_kv_groups`.
    """
    if not arch.is_hybrid_ssm:
        return 0
    blocks = concurrency * mamba_state_blocks_per_seq(arch)
    if not shares_hybrid_pool(arch):
        per_layer = mamba_page_per_rank(arch, state_dtype, tensor_parallel)
        return int(arch.n_mamba_layers * per_layer * blocks)
    size, _, mamba_groups = hybrid_kv_groups(arch)
    page = hybrid_page_bytes(arch, state_dtype, tensor_parallel)
    return int(mamba_groups * size * page * blocks)


def hybrid_kv_demand_bytes(
    arch: ModelArch,
    weight_dtype: WeightDtype,
    concurrency: int,
    seq_len: int,
    tensor_parallel: int = 1,
) -> int:
    """Attention KV one rank of a hybrid pool holds for every sequence at ``seq_len``.

    Each sequence takes whole blocks of :func:`aligned_kv_block_size` tokens
    from every attention group of :func:`hybrid_kv_groups`.
    """
    block = aligned_kv_block_size(arch, weight_dtype, tensor_parallel)
    size, attention_groups, _ = hybrid_kv_groups(arch)
    page = hybrid_page_bytes(arch, weight_dtype, tensor_parallel)
    return int(concurrency * attention_groups * size * -(-seq_len // block) * page)


def kv_cache_demand_bytes(
    arch: ModelArch,
    weight_dtype: WeightDtype,
    concurrency: int,
    seq_len: int,
    in_flight_tokens: int,
) -> int:
    """Worst-case KV bytes: every concurrent sequence at full context.

    Sliding-window layers hold, per sequence, the window less one token plus
    the scheduler's in-flight tokens, in whole blocks plus one, at the sliding
    head width; full-attention layers keep the full sequence at
    :attr:`ModelArch.global_head_dim`.

    :param in_flight_tokens: Tokens scheduled but not yet freed, vLLM's
        ``max_in_flight_tokens``.
    """
    kv_bytes = DTYPE_BYTES[weight_dtype]
    layer_kv = 2 * kv_storing_layers(arch) * arch.n_kv_heads * kv_bytes
    windowed, window_dim, full_dim = kv_layer_head_dims(arch)
    if arch.sliding_window is None:
        return int(concurrency * seq_len * layer_kv * full_dim)
    held = min(arch.sliding_window - 1 + in_flight_tokens, seq_len)
    window_len = (-(-held // KV_BLOCK_SIZE_DEFAULT) + 1) * KV_BLOCK_SIZE_DEFAULT
    return int(
        concurrency
        * layer_kv
        * (windowed * window_len * window_dim + (1.0 - windowed) * seq_len * full_dim)
    )


def resolve_max_num_batched_tokens(
    max_num_seqs: int, max_model_len: int, explicit: int | None = None
) -> int:
    """Scheduler-step token cap the async rollout engine starts vLLM with.

    Unset takes vLLM's default for an engine built with no usage context.
    Either value is capped at every sequence at full context.
    """
    requested = VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS if explicit is None else explicit
    return min(requested, max_num_seqs * max_model_len)


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


def lora_input_cast_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    scope: LoraTargetScope = "all-linear",
    gradient_checkpointing: bool = True,
    layer: BlockKind | None = None,
    tensor_parallel: int = 1,
) -> int:
    """The fp32 copy PEFT makes of a wrapped linear's input.

    ``get_peft_model`` defaults to ``autocast_adapter_dtype=True``, so the
    adapters are fp32 while the base is bf16, and each wrapped linear casts
    its input to fp32 (``BaseTunerLayer._cast_input_dtype``).

    Under gradient checkpointing the casts stay bounded: the block is
    recomputed as one unit, so the peak is the widest single cast (MLP
    down-projection on the intermediate). ``layer`` restricts it to one
    layer kind of a block-exclusive stack.
    """
    h = arch.hidden_size
    # in_features of the linears PEFT wraps: q/k/v and the MLP gate/up read
    # the residual stream, o reads the concatenated head output, down reads
    # the intermediate. Routed experts live in fused 3D parameters
    # (see ParamCounts.moe_experts), so the MLP term tracks dense blocks.
    if gradient_checkpointing:
        attn_out = tp_local(
            arch.n_heads * max(arch.head_dim, arch.global_head_dim or 0),
            tensor_parallel,
        )
        by_layer = {"attention": [h, attn_out], "ffn": [h], "mamba": [h]}
        if scope == "all-linear" and arch.mlp_layers:
            by_layer["ffn"].append(
                tp_local(
                    int(arch.intermediate_size * arch.peak_mlp_width_factor),
                    tensor_parallel,
                )
            )
        widths = (
            by_layer[layer]
            if layer is not None
            else [w for kind in ("attention", "ffn") for w in by_layer[kind]]
        )
        return int(rows * seq_len * max(widths) * DTYPE_BYTES["fp32"])
    # Without checkpointing every layer's casts stay live until that layer's
    # backward, so they accumulate.
    qkv = (
        arch.mean_qkv_dim
        if tensor_parallel <= 1
        else qkv_width_per_rank(arch, tensor_parallel)
    )
    width = 3 * h + qkv
    if scope == "all-linear" and arch.mlp_layers:
        inter = tp_local(
            int(arch.intermediate_size * arch.mlp_width_factor), tensor_parallel
        )
        width += (2 * h + inter) if arch.gated_mlp else (h + inter)
    return int(rows * seq_len * width * arch.n_layers * DTYPE_BYTES["fp32"])


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

    ``arch.chunked_routed_experts`` runs experts in row chunks and writes
    each chunk into the layer output. Backward keeps the gathered input, the
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
        width = 2 * h + inters * inter
    elif not arch.chunked_routed_experts:
        width = 2 * h + (4 if backward else 3) * inter
    elif backward:
        width = 2 * h + (3 if arch.gated_mlp else 2) * inter
    else:
        width = h
    return int(routed * width * act_bytes)


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
) -> int:
    """Backward bytes of one MoE block that grow with the routed-expert chunk budget.

    Each buffer is capped by the rows it chunks. The chunked routed path
    (expert LoRA on ``arch.chunked_routed_experts``) and the expert-parallel
    fp32 combine each add :data:`ROUTED_CHUNK_LIVE_FRACTION` of a chunk. The
    sorted path's contracted LoRA holds
    :data:`CONTRACTED_CHUNK_LIVE_MULTIPLE` whole chunks; transposed experts
    (``arch.transposed_experts``) hold as many buffers over every routed row.

    :param chunk_bytes: Widest ``[rows, features]`` activation of one chunk.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param contracted: Expert LoRA runs split (``packed_moe_dispatch="contracted"``).
    :param ep: Expert-parallel degree.
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
        held = routed * widest_row
        if not arch.transposed_experts:
            held = min(chunk_bytes, held)
        whole += CONTRACTED_CHUNK_LIVE_MULTIPLE * held
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
) -> tuple[float, int, float]:
    """Loss-instant logits, the hoisted fp32 head, and no-grad head bytes."""
    # Two fp32 tiles are live at the loss instant: recomputed logits plus
    # that tile's gradient. Tiles are fixed-size, so a fused critic row adds none.
    logit_rows = resolve_chunk_rows(arch.vocab_size, settings.chunk_rows)
    # The chunk loop hoists an fp32 lm_head copy when the head runs below fp32.
    fused_head = int(arch.vocab_size * arch.hidden_size * 4) if act_bytes < 4 else 0
    logits = logit_rows * arch.vocab_size * 8 + fused_head
    nograd_logits = logit_rows * arch.vocab_size * 4 + fused_head
    if settings.sampling_support_width:
        support_rows = support_rows_bytes(
            arch, logit_rows, settings.sampling_support_width, act_bytes
        )
        nograd_logits = support_rows
        if settings.support_scored_loss:
            logits = support_rows
            fused_head = 0
    return logits, fused_head, nograd_logits


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
    wrap_every_n_blocks: int,
    tensor_parallel: int = 1,
    expert_parallel: int = 1,
) -> int:
    """Decoder parameters gathered at once: ``live_units`` neighbouring FSDP units.

    With the layer order known, the window is the widest run of neighbouring
    units, so a large block next to small ones is counted once. Otherwise
    every unit is the widest block. Layers with every linear on the TP mesh
    gather nothing.
    """
    linears_on_tp = tensor_parallel > 1 and arch.tensor_parallel_plan
    if not arch.layer_kinds:
        block = largest_block_params(counts, arch, expert_parallel)
        if linears_on_tp:
            # A uniform stack's linears are on the TP mesh. Norms and MoE stay gathered.
            moe = arch.moe_layers * block_moe_params_per_layer(arch, expert_parallel)
            block = int((counts.norms + moe) / max(arch.n_layers, 1))
        return live_units * wrap_every_n_blocks * block
    tp_kinds = ("attention", "mamba", "mlp") if linears_on_tp else ()
    sizes = [
        0 if kind in tp_kinds else layer_params(arch, kind, expert_parallel)
        for kind in arch.layer_kinds
    ]
    units = [
        sum(sizes[start : start + wrap_every_n_blocks])
        for start in range(0, len(sizes), wrap_every_n_blocks)
    ]
    live_units = min(live_units, len(units))
    return max(
        sum(units[start : start + live_units])
        for start in range(len(units) - live_units + 1)
    )


def fsdp_gathered_params(
    counts: ParamCounts,
    arch: ModelArch,
    *,
    n_gpus: int,
    reshard_after_forward: bool,
    prefetch_units: int,
    wrap_every_n_blocks: int,
    cpu_offload: bool,
    tensor_parallel: int = 1,
    expert_parallel: int = 1,
) -> int:
    """Unsharded parameter elements held besides the resident shard.

    Excludes the replicated token embeddings. ``reshard_after_forward``
    keeps a prefetch window of decoder units, plus an untied ``lm_head``,
    per-layer embeddings, and the root unit (tower parameters outside the
    encoder blocks, unattributed parameters) at their own gather. Otherwise
    every sharded parameter stays gathered. A single GPU with resident parameters
    has no second copy.

    Expert-parallel routed experts are not in the block units. A rank
    gathers one MoE layer of its own experts across the shard-group ranks
    that hold the same experts, and nothing when ``ep`` fills the group.
    """
    token = min(token_embedding_params(arch), max(counts.total, 0))
    tp_params = 0 if tensor_parallel <= 1 else tensor_parallel_params(counts, arch)
    sharded = max(counts.total - token - tp_params, 0)
    if n_gpus <= 1 and not cpu_offload:
        return 0
    expert_layer = 0
    if expert_parallel > 1:
        routed = routed_expert_params_per_layer(arch)
        sharded = max(sharded - arch.moe_layers * routed, 0)
        if n_gpus > expert_parallel:
            expert_layer = routed // expert_parallel
    if not reshard_after_forward:
        return sharded + arch.moe_layers * expert_layer
    n_units = max(1, -(-arch.n_layers // wrap_every_n_blocks))
    live = min(n_units, 1 + prefetch_units)
    window = prefetch_window_params(
        counts, arch, live, wrap_every_n_blocks, tensor_parallel, expert_parallel
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
    wrap_every_n_blocks: int,
    prefetch_units: int,
    ep: int,
    shards: int,
    tensor_parallel: int = 1,
) -> int:
    """Elements FSDP's all-gather stream keeps reserved for the whole step.

    :param counts: Parameter counts of the model.
    :param arch: Model architecture.
    :param wrap_every_n_blocks: Decoder blocks per FSDP unit.
    :param prefetch_units: Units FSDP gathers ahead of the running one.
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
    units = [
        sum(sizes[start : start + wrap_every_n_blocks])
        for start in range(0, len(sizes), wrap_every_n_blocks)
    ]
    live = min(len(units), 1 + prefetch_units)
    window = max(
        sum(units[start : start + live]) for start in range(len(units) - live + 1)
    )
    head, extra_embed, root = _side_units(counts, arch, tensor_parallel)
    return max(window, max(extra_embed, root) + units[0], head + units[-1])
