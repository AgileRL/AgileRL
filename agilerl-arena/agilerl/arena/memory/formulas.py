# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form memory terms for LLM RL training and generation.

Every function is a pure function of the specs in
:mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    BlockKind,
    LayerKind,
    LoraTargetScope,
    MiB,
    ModelArch,
    PackedMoeDispatch,
    WeightDtype,
    WeightVariant,
)

# Upper bound on the CUDA-graph pool the vLLM engine captures unless
# enforce_eager. It cancels in the generation total: subtracted from the KV
# budget, added back as residency.
CUDA_GRAPH_POOL_BYTES = 256 * MiB
# Bytes per LoRA parameter under flat data parallel. PEFT
# ``autocast_adapter_dtype=True`` stores adapters and their gradients in
# fp32. AdamW keeps two fp32 moments per parameter.
ADAPTER_BYTES_PER_PARAM = 4.0
ADAM_BYTES_PER_PARAM = 8.0
# Foreach AdamW builds one fp32 temporary per trained parameter in step();
# the fused kernel builds none.
FOREACH_ADAM_WORKSPACE_BYTES_PER_PARAM = 4.0
# Routed-expert chunk budget when no FSDP config sets one.
DEFAULT_ROUTED_CHUNK_BYTES = 64 * MiB
# Budgets an unset ``FSDPConfig.routed_expert_chunk_mib`` picks from, largest first.
# Super-VL VWA learn steps on H100 80GB ran 2.2% faster at 512 MiB than at 256
# (busiest rank +0.69 GiB). That H100 rise is far above the A100 fit below, so
# larger chunks are left out: the estimate would pass them on H100 and miss the peak.
ROUTED_CHUNK_MIB_CHOICES = (512, 256, 128, 64)
# Share of one chunk budget each chunked buffer adds to the MoE backward peak
# (Super-VL, EP 8, expert LoRA). A100 80GB peaks rise 0.37 GiB per GiB of chunk
# over 64-1024 MiB; the H100 80GB VWA busiest rank rose 0.69 GiB from 256 to
# 512 MiB. 0.45 keeps the estimate at or above the H100 peak at 512 MiB.
ROUTED_CHUNK_LIVE_FRACTION = 0.45
# ``[routed rows, expert width]`` buffers expert parallel adds to the busiest
# rank's MoE backward peak: the send rows held for the comm stream, the
# received rows, the saved combine rows, and the gradients through both
# all-to-alls. Fitted to Lightning 30B-A3B EP 2, EP 4 and EP 2 x TP 2 torch
# allocations on A100 80GB at 16k tokens with NCCL cuMem off.
EP_EXCHANGE_BUFFERS = 4

# Bytes per [row, vocab] element of a Liger loss chunk that holds one whole
# sequence: the bf16 logits plus fp32 upcast, log-softmax and gradient.
# LLMPPO trajectory-level Liger on A100 80GB, Qwen3-4B and Qwen2.5-0.5B at 8k
# and 16k tokens, measured 13.2 to 13.9.
LIGER_SEQUENCE_LOGIT_BYTES = 14

# PyTorch caching-allocator slack: reserved above allocated.
# Charged on the torch-side subtotal. Training only: generation sits in
# vLLM's CuMem pool, reserved up front at ``gpu_memory_utilization``.
ALLOCATOR_RESERVE_FRACTION = 0.07
# Flat data parallel keeps an all-reduce buffer.
ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL = 0.054
# FSDP's all-gather stream keeps its own block pool, which the compute stream
# cannot reuse; it is charged separately as the gather window. The compute
# stream then fragments by one of two shares: Liger-fused attention and MLP
# blocks, or blocks with stock norms and MLP, Mamba scans or routed experts.
# Fit on A100 80GB allocator snapshots and the A100 and L4 memory-drift rows;
# 3.4% also keeps the Super-VL busiest rank inside the runtime headroom.
ALLOCATOR_RESERVE_FRACTION_FSDP = 0.005
ALLOCATOR_RESERVE_FRACTION_FSDP_IRREGULAR = 0.034
# Largest share by which a measured peak exceeded the estimate on the 4-GPU
# grids. A GPU recommendation keeps this share of usable memory free.
MAX_UNDERPREDICTION = 0.089


def recommendation_fits(predicted_bytes: int, usable_bytes: int) -> bool:
    """Whether a predicted peak still fits after the underprediction buffer."""
    return predicted_bytes <= int(usable_bytes * (1 - MAX_UNDERPREDICTION))


def allocator_reserve_bytes(
    allocated_bytes: float,
    n_ranks: int = 1,
    sharded: bool = False,
    irregular_blocks: bool = False,
    gather_pool_bytes: float = 0.0,
) -> float:
    """Caching-allocator reserved-but-unallocated bytes.

    :param irregular_blocks: Blocks run stock norms and MLP, Mamba scans or
        routed experts; applies when ``sharded``.
    :param gather_pool_bytes: FSDP all-gather stream pool; applies when ``sharded``.
    """
    if sharded:
        fraction = (
            ALLOCATOR_RESERVE_FRACTION_FSDP_IRREGULAR
            if irregular_blocks
            else ALLOCATOR_RESERVE_FRACTION_FSDP
        )
        return max(allocated_bytes, 0.0) * fraction + gather_pool_bytes
    if n_ranks > 1:
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
# Alignment the hybrid KV block size is rounded up to.
MAMBA_KERNEL_BLOCK_ALIGNMENT = 32
# Recurrent-state blocks resident per sequence. vLLM prefix caching
# resolves every hybrid to align mode, which caps residency at two.
MAMBA_STATE_BLOCKS_PER_SEQ = 2


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


def aligned_kv_block_size(
    arch: ModelArch,
    weight_dtype: WeightDtype = "bf16",
) -> int:
    """Block size vLLM adopts so an attention page holds a Mamba page.

    A hybrid model puts both caches in one block pool, so the page sizes
    must match. vLLM raises the attention block size until its page is at
    least as large as a Mamba page, then pads the Mamba page up to it.
    The block size is then hundreds or thousands of tokens, and every
    sequence's KV rounds up to that page.
    """
    if not arch.is_hybrid_ssm or not arch.attention_layers:
        return KV_BLOCK_SIZE_DEFAULT
    per_token_per_layer = kv_cache_bytes_per_token(arch, weight_dtype) / (
        arch.attention_layers
    )
    if per_token_per_layer <= 0:
        return KV_BLOCK_SIZE_DEFAULT
    tokens = -(-mamba_page_bytes(arch, weight_dtype) // int(per_token_per_layer))
    alignment = MAMBA_KERNEL_BLOCK_ALIGNMENT
    return max(KV_BLOCK_SIZE_DEFAULT, -(-tokens // alignment) * alignment)


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
    ``tensor_parallel`` applies :func:`mamba_page_per_rank`.
    """
    if not arch.is_hybrid_ssm:
        return 0
    per_layer = mamba_page_per_rank(arch, state_dtype, tensor_parallel)
    return int(
        arch.n_mamba_layers * per_layer * concurrency * MAMBA_STATE_BLOCKS_PER_SEQ
    )


def kv_cache_demand_bytes(
    arch: ModelArch,
    weight_dtype: WeightDtype,
    concurrency: int,
    seq_len: int,
) -> int:
    """Worst-case KV bytes: every concurrent sequence at full context.

    Sliding-window layers cap per-sequence growth at the window, at the
    sliding head width; full-attention layers keep the full sequence at
    :attr:`ModelArch.global_head_dim`.
    """
    kv_bytes = DTYPE_BYTES[weight_dtype]
    layer_kv = 2 * kv_storing_layers(arch) * arch.n_kv_heads * kv_bytes
    windowed, window_dim, full_dim = kv_layer_head_dims(arch)
    if arch.sliding_window is None:
        return int(concurrency * seq_len * layer_kv * full_dim)
    window_len = min(seq_len, arch.sliding_window)
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


def routed_expert_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    lora_chunk_bytes: int | None = None,
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

    Other families with expert LoRA (``lora_chunk_bytes`` set) scatter row
    chunks with the full autograd graph. Their backward keeps the gathered
    input and its gradient, which the chunks' gradients fill, for every routed
    row; and for one chunk the expert output, its gradient, autograd's
    gradient of the weighted output, gate/up and the activation.

    :param lora_chunk_bytes: Routed-expert chunk budget when packed experts
        carry LoRA.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    h = arch.expert_width
    inter = arch.expert_intermediate_size or arch.intermediate_size
    if not arch.chunked_routed_experts and backward and lora_chunk_bytes is not None:
        chunk_rows = min(
            routed, max(1, int(lora_chunk_bytes // (max(h, inter) * act_bytes)))
        )
        chunk_width = 3 * h + (4 if arch.gated_mlp else 2) * inter
        return int((2 * routed * h + chunk_rows * chunk_width) * act_bytes)
    if not arch.chunked_routed_experts:
        width = 2 * h + (4 if backward else 3) * inter
    elif backward:
        width = 2 * h + (3 if arch.gated_mlp else 2) * inter
    else:
        width = h
    return int(routed * width * act_bytes)


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
    ep: int,
) -> int:
    """Backward bytes of one MoE block that grow with the routed-expert chunk budget.

    Each buffer is capped by the rows it chunks. The chunked routed path
    (expert LoRA on ``arch.chunked_routed_experts``) and the expert-parallel
    fp32 combine each add :data:`ROUTED_CHUNK_LIVE_FRACTION` of a chunk.
    Other families' chunk buffers are in :func:`routed_expert_bytes`.

    :param chunk_bytes: Widest ``[rows, features]`` activation of one chunk.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param ep: Expert-parallel degree.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    inter = arch.expert_intermediate_size or arch.intermediate_size
    widest_row = max(arch.expert_width, inter) * act_bytes
    fractional = 0.0
    if expert_lora and arch.chunked_routed_experts:
        fractional += min(chunk_bytes, routed * widest_row)
    if ep > 1:
        fractional += min(chunk_bytes, routed * arch.expert_width * 4)
    return int(fractional * ROUTED_CHUNK_LIVE_FRACTION)


def mamba_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    eager_scan: bool = False,
    tensor_parallel: int = 1,
) -> int:
    """Live tensors of one Mamba-2 layer's chunked scan in the trainer.

    Per token: the ``in_proj`` output, the conv1d output, the gated scan
    input and an fp32 copy of the residual stream. Per chunk: the fp32 SSM
    state ``(heads, d_head, d_state)`` and the ``chunk x chunk`` CB matrix
    per group. A checkpointed backward adds the gated-norm gradient, fp32
    gradients of the ``in_proj`` output and the residual, and two more
    state buffers.

    ``eager_scan`` adds HF's eager ``torch_forward`` scan, which builds an
    fp32 broadcast product of ``tokens x chunk x heads x max(d_state,
    d_head)`` before each contraction.
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
    if eager_scan:
        width = max(arch.mamba_d_state, arch.mamba_d_head)
        total += tokens * EAGER_MAMBA_SCAN_CHUNK * heads * width * 4
    return int(total)


def _block_mlp_recompute(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    engine_forward: bool,
    tensor_parallel: int,
    lora_chunk_bytes: int | None = None,
) -> tuple[int, int]:
    """Per-token MLP width and routed-expert bytes of one block recompute.

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
                arch, rows, seq_len, act_bytes, backward, lora_chunk_bytes
            )
    else:
        active_width = tp_local(
            int(arch.intermediate_size * arch.peak_mlp_width_factor), tensor_parallel
        )
    # Gated backward: gate, up, fused output, and one gradient. The stock
    # unfused MLP also keeps the activation output and two more gradients.
    mlp_count = mlp_matrices + 1 if backward and arch.gated_mlp else mlp_matrices
    if backward and arch.gated_mlp and arch.stock_hf_modules:
        mlp_count += 2
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
    eager_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    lora_chunk_bytes: int | None = None,
) -> int:
    """Peak transient activations of one transformer block.

    The forward peak holds four residual copies, qkv, and the MLP outputs.
    A checkpointed backward holds two residual copies, qkv, and the gated
    MLP: gate, up, the fused output, and one gradient. A Q/K-norm model
    (``arch.qk_norm``) also holds one fp32 copy of Q. Flash-style backends
    tile the score matrix. A stock RMSNorm (``arch.stock_hf_modules``)
    keeps an fp32 copy of each norm input on the backward.

    In the trainer, routed experts run the grouped-GEMM path
    (:func:`routed_expert_bytes`) and Mamba mixers their chunked scan
    (:func:`mamba_block_bytes`). A block-exclusive stack's block is one
    attention, FFN or Mamba layer: ``layer`` picks one, else the widest
    counts; otherwise a block holds all of them. ``engine_forward=True``
    sizes routed experts as the active FFN width (vLLM's in-kernel routing).
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
        lora_chunk_bytes,
    )
    ple = arch.n_layers * arch.per_layer_input_dim
    if backward:
        ple *= 2
    tokens = rows * seq_len
    if engine_forward:
        return int(tokens * (4 * h + qkv_dim + mlp + ple) * act_bytes)
    mamba = mamba_block_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        backward,
        eager_scan=eager_mamba_scan,
        tensor_parallel=1 if engine_forward else tensor_parallel,
    )
    # fp32 RMSNorm inputs saved for backward, one per norm in the block.
    norm_input = tokens * h * 4 if backward and arch.stock_hf_modules else 0
    q_heads = (
        arch.n_heads if engine_forward else tp_local(arch.n_heads, tensor_parallel)
    )
    q_norm = tokens * q_heads * arch.head_dim * 4 if backward and arch.qk_norm else 0
    if arch.block_exclusive_layers:
        # The scan term already holds the Mamba layer's residual copy.
        by_layer = {
            "attention": tokens * (residual + qkv_dim) * act_bytes
            + norm_input
            + q_norm,
            "ffn": tokens * (residual + mlp + ple) * act_bytes + routed + norm_input,
            "mamba": mamba + norm_input,
        }
        if layer is not None:
            return int(by_layer[layer])
        return int(max(by_layer.values()))
    return int(
        tokens * (residual + qkv_dim + mlp + ple) * act_bytes
        + routed
        + mamba
        + 2 * norm_input
        + q_norm
    )


def nograd_forward_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    eager_mamba_scan: bool = False,
    tensor_parallel: int = 1,
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
        eager_mamba_scan=eager_mamba_scan,
        tensor_parallel=tensor_parallel,
    )
    hidden_out = rows * seq_len * arch.hidden_size * act_bytes
    return int(block + hidden_out)


# Chunk length of AgileRL's eager Mamba scan.
EAGER_MAMBA_SCAN_CHUNK = 64

# Per-process job overhead charged when ``orchestrated`` is set.
JOB_OVERHEAD_BYTES = 50 * MiB

# Non-torch device memory a one-rank trainer holds beyond the CUDA context
# and job overhead. Multi-rank trainers charge :func:`trainer_nccl_bytes`.
TRAINER_LIB_OVERHEAD_BYTES = 436 * MiB

# NCCL bytes as fractions of the device's 4-rank on-node group bytes, fitted
# on Qwen3-4B FSDP, HSDP and TP over one and two A100 and H100 nodes with
# NCCL cuMem off. A group whose ranks share a node holds NVLink buffers that
# grow with its size; a group across nodes holds few socket buffers. Each
# multi-rank trainer also pays a fixed base once.
NCCL_BASE_FRACTION = 0.14
NCCL_CROSS_NODE_FRACTION = 0.06
# All-to-all buffers expert parallel's own group holds beside its group bytes,
# on one node and across nodes. Measured on Lightning 30B-A3B EP 2 and EP 4
# over one and two A100 nodes.
NCCL_ALL_TO_ALL_ON_NODE_FRACTION = 1.1
NCCL_ALL_TO_ALL_CROSS_NODE_FRACTION = 0.54
# Non-torch memory a tensor-parallel trainer holds beyond its process groups,
# the same on A100 and H100.
TP_RUNTIME_BYTES = 512 * MiB


def nccl_on_node_fraction(group_size: int) -> float:
    """On-node group bytes against a 4-rank group, measured at 2, 4 and 8 ranks."""
    if group_size <= 2:
        return 0.65
    if group_size <= 4:
        return 1.0
    return 1.13


def trainer_nccl_bytes(
    group_bytes: int,
    n_gpus: int,
    shard_gpus: int,
    expert_parallel: int = 1,
    tensor_parallel: int = 1,
    gpus_per_node: int | None = None,
) -> int:
    """NCCL buffers on one trainer rank, summed over the process groups it joins.

    Ranks fill nodes in rank order. The world group joins every rank. HSDP
    adds its shard group (consecutive ranks) and its replicate group (one
    rank per shard). EP and TP each add their own group of consecutive ranks,
    plus the group of ranks holding the same shard when that is more than one
    rank (two groups under HSDP: the split and its flattened replicas). EP's
    own group also holds all-to-all buffers.

    :param group_bytes: The device's 4-rank on-node group bytes.
    :param gpus_per_node: GPUs on one node; ``None`` keeps every rank on one node.
    """
    node = n_gpus if gpus_per_node is None else gpus_per_node

    def group(size: int, span: int) -> float:
        if span <= node:
            return nccl_on_node_fraction(size)
        return NCCL_CROSS_NODE_FRACTION

    replicate = n_gpus // shard_gpus
    fraction = NCCL_BASE_FRACTION + group(n_gpus, n_gpus)
    if replicate > 1:
        fraction += group(shard_gpus, shard_gpus) + group(replicate, n_gpus)
    all_to_all = (
        NCCL_ALL_TO_ALL_ON_NODE_FRACTION
        if expert_parallel <= node
        else NCCL_ALL_TO_ALL_CROSS_NODE_FRACTION
    )
    for degree, own_group in (
        (expert_parallel, group(expert_parallel, expert_parallel) + all_to_all),
        (tensor_parallel, group(tensor_parallel, tensor_parallel)),
    ):
        if degree <= 1:
            continue
        fraction += own_group
        outer = shard_gpus // degree
        if outer > 1:
            fraction += group(outer, shard_gpus)
            if replicate > 1:
                fraction += group(outer * replicate, n_gpus)
    return int(group_bytes * fraction)


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
    live at once; otherwise every layer's are. A block-exclusive stack's
    block is one attention, MLP or Mamba layer: ``layer`` picks one, else
    the widest counts. A parallel mixer stack charges all three in one block.
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
    # Mamba LoRA wraps in_proj (the residual); out_proj is never wrapped.
    mamba = h if scope == "all-linear" and arch.n_mamba_layers else 0
    if layer is not None:
        width = {"attention": attention, "ffn": mlp, "mamba": mamba}[layer]
    elif arch.block_exclusive_layers:
        width = max(attention, mlp, mamba)
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
    moe = arch.moe_layers * block_moe_params_per_layer(arch, expert_parallel)
    _, router = moe_params_per_layer(arch)
    body = counts.attention + counts.mlp - arch.moe_layers * router + moe
    body += counts.norms + counts.mamba
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
    stream_peak: bool = False,
) -> int:
    """Decoder parameters gathered at once: ``live_units`` neighbouring FSDP units.

    With the layer order known, the window is the widest run of neighbouring
    units, so a large block next to small ones is counted once. Otherwise
    every unit is the widest block.

    :param stream_peak: Size the all-gather stream's own peak. Layers with
        every linear on the TP mesh gather nothing there, so the window spans
        the next gathered layers (two MoE layers of a TP-planned hybrid).
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
    if stream_peak:
        sizes = [
            layer_params(arch, kind, expert_parallel)
            for kind in arch.layer_kinds
            if kind not in tp_kinds
        ] or [0]
    else:
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
    stream_peak: bool = False,
) -> int:
    """Unsharded parameter elements held besides the resident shard.

    Excludes the replicated token embeddings. ``reshard_after_forward``
    keeps a prefetch window of decoder units, plus an untied ``lm_head`` and
    root leftovers (multimodal towers, per-layer embeddings). Otherwise every
    sharded parameter stays gathered. A single GPU with resident parameters
    has no second copy.

    Expert-parallel routed experts are not in the block units. A rank
    gathers one MoE layer of its own experts across the shard-group ranks
    that hold the same experts, and nothing when ``ep`` fills the group.

    :param stream_peak: Size the all-gather stream's own peak
        (:func:`prefetch_window_params`).
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
        counts,
        arch,
        live,
        wrap_every_n_blocks,
        tensor_parallel,
        expert_parallel,
        stream_peak,
    )
    # An untied head on the TP mesh is already in ``tp_params``.
    head = 0 if arch.tied_embeddings or tensor_parallel > 1 else counts.lm_head
    extra_embed = max(counts.embedding - token_embedding_params(arch), 0)
    leftover = extra_embed + counts.multimodal_towers + max(counts.unattributed, 0)
    return min(sharded, window + head + leftover) + expert_layer


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

    ``threshold`` is FSDP's ``param_persistence_threshold``: a matrix with
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
