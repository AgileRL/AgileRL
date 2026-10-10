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

# PyTorch caching-allocator slack: reserved above allocated.
# Charged on the torch-side subtotal. Training only: generation sits in
# vLLM's CuMem pool, reserved up front at ``gpu_memory_utilization``.
ALLOCATOR_RESERVE_FRACTION = 0.07
# Flat data parallel keeps an all-reduce buffer. FSDP pools all-gather and
# reduce-scatter blocks per stream, so it reserves a larger share.
ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL = 0.054
ALLOCATOR_RESERVE_FRACTION_FSDP = 0.069
# Largest share by which a measured peak exceeded the estimate on the 4-GPU
# grids. A GPU recommendation keeps this share of usable memory free.
MAX_UNDERPREDICTION = 0.089
# Share of a training GPU's CUDA capacity (torch ``total_memory``) a runtime
# memory pick leaves free. The estimate is in NVML device-used bytes, which
# also count the driver's reservation that ``total_memory`` leaves out
# (0.85 GiB on A100 80GB). On Super-VL learn steps (A100 80GB, EP 8, rows up
# to 30.9k tokens) the busiest rank peaked 0.87 GiB over the estimate and EP
# routing spread the ranks 1.5 GiB; at 2.5% that rank keeps about 2 GiB free
# for a step routed more unevenly.
TRAINING_HEADROOM_FRACTION = 0.025


def recommendation_fits(predicted_bytes: int, usable_bytes: int) -> bool:
    """Whether a predicted peak still fits after the underprediction buffer."""
    return predicted_bytes <= int(usable_bytes * (1 - MAX_UNDERPREDICTION))


def allocator_reserve_bytes(
    allocated_bytes: float, n_ranks: int = 1, sharded: bool = False
) -> float:
    """Caching-allocator reserved-but-unallocated bytes."""
    if sharded:
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
    experts = n_experts * mlp_matrices * arch.expert_width * expert_inter
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


def mamba_state_bytes(
    arch: ModelArch,
    concurrency: int,
    state_dtype: WeightDtype = "bf16",
) -> int:
    """Recurrent-state cache for a hybrid model's Mamba layers.

    Two tensors per recurrent layer per slot, matching vLLM's
    ``mamba2_state_shape``: a causal-conv window of ``(conv_dim, d_conv - 1)``
    and a recurrent state of ``(n_heads, d_head, d_state)``. Both are
    constant in context length, so this term scales with concurrency.
    """
    if not arch.is_hybrid_ssm:
        return 0
    per_layer = mamba_page_bytes(arch, state_dtype)
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


# Bounds on fused-logprob chunk rows. Floor: ``lm_head`` is re-read once per
# chunk. Ceiling: the fp32 logit tile grows linearly with rows.
FUSED_CHUNK_ROWS_MIN = 128
FUSED_CHUNK_ROWS_MAX = 4096


def resolve_chunk_rows(vocab_size: int, explicit: int | None = None) -> int:
    """Fused-logprob chunk rows for a 256 MiB fp32 logit tile, clamped.

    ``explicit`` is returned as-is; it is range-checked at construction.
    """
    if explicit is not None:
        return explicit
    rows = 256 * MiB // max(1, vocab_size * 4)
    return max(FUSED_CHUNK_ROWS_MIN, min(FUSED_CHUNK_ROWS_MAX, rows))


def resolve_max_num_batched_tokens(
    max_num_seqs: int, max_model_len: int, explicit: int | None = None
) -> int:
    """Scheduler-step token cap for vLLM.

    An explicit value is returned as-is. Otherwise
    ``min(max_num_seqs * max_model_len, max(max_model_len, max_num_seqs * 8192))``.
    The 8192-per-sequence cap bounds compile tensors at long context.
    """
    if explicit is not None:
        return explicit
    return min(
        max_num_seqs * max_model_len,
        max(max_model_len, max_num_seqs * 8192),
    )


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
        attn_out = arch.n_heads * max(arch.head_dim, arch.global_head_dim or 0)
        by_layer = {"attention": [h, attn_out], "ffn": [h], "mamba": [h]}
        if scope == "all-linear" and arch.mlp_layers:
            by_layer["ffn"].append(
                int(arch.intermediate_size * arch.peak_mlp_width_factor)
            )
        widths = (
            by_layer[layer]
            if layer is not None
            else [w for kind in ("attention", "ffn") for w in by_layer[kind]]
        )
        return int(rows * seq_len * max(widths) * DTYPE_BYTES["fp32"])
    # Without checkpointing every layer's casts stay live until that layer's
    # backward, so they accumulate.
    width = 3 * h + arch.mean_qkv_dim
    if scope == "all-linear" and arch.mlp_layers:
        inter = int(arch.intermediate_size * arch.mlp_width_factor)
        width += (2 * h + inter) if arch.gated_mlp else (h + inter)
    return int(rows * seq_len * width * arch.n_layers * DTYPE_BYTES["fp32"])


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
    if not arch.chunked_routed_experts:
        width = 2 * h + (4 if backward else 3) * inter
    elif backward:
        width = 2 * h + (3 if arch.gated_mlp else 2) * inter
    else:
        width = h
    return int(routed * width * act_bytes)


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
    sorted path's contracted LoRA up-projection holds one whole chunk output.

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
        whole += min(chunk_bytes, routed * widest_row)
    if ep > 1:
        fractional += min(chunk_bytes, routed * arch.expert_width * 4)
    return int(fractional * ROUTED_CHUNK_LIVE_FRACTION + whole)


def mamba_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    eager_scan: bool = False,
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
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    in_proj_out = d_inner + arch.mamba_conv_dim + arch.mamba_n_heads
    chunks = rows * -(-seq_len // arch.mamba_chunk_size)
    state = chunks * arch.mamba_n_heads * arch.mamba_d_head * arch.mamba_d_state * 4
    cb = chunks * arch.mamba_n_groups * arch.mamba_chunk_size**2 * 4
    per_token = (in_proj_out + arch.mamba_conv_dim + d_inner) * act_bytes
    per_token += arch.hidden_size * 4
    total = tokens * per_token + state + cb
    if backward:
        per_token_grads = d_inner * act_bytes + (in_proj_out + arch.hidden_size) * 4
        total += tokens * per_token_grads + 2 * state
    if eager_scan:
        width = max(arch.mamba_d_state, arch.mamba_d_head)
        total += tokens * EAGER_MAMBA_SCAN_CHUNK * arch.mamba_n_heads * width * 4
    return int(total)


def _block_mlp_recompute(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    engine_forward: bool,
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
            routed = routed_expert_bytes(arch, rows, seq_len, act_bytes, backward)
    else:
        active_width = int(arch.intermediate_size * arch.peak_mlp_width_factor)
    # Gated backward: gate, up, fused output, and one gradient.
    mlp_count = mlp_matrices + 1 if backward and arch.gated_mlp else mlp_matrices
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
) -> int:
    """Peak transient activations of one transformer block.

    The forward peak holds four residual copies, qkv, and the MLP outputs.
    A checkpointed backward holds two residual copies, qkv, and the gated
    MLP: gate, up, the fused output, and one gradient. A Q/K-norm model
    (``arch.qk_norm``) also holds one fp32 copy of Q. Flash-style backends
    tile the score matrix. A stock RMSNorm (``arch.fp32_norm_inputs``)
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
    residual = (2 if backward else 4) * h
    qkv_dim = arch.peak_qkv_dim
    mlp, routed = _block_mlp_recompute(
        arch, rows, seq_len, act_bytes, backward, engine_forward
    )
    ple = arch.n_layers * arch.per_layer_input_dim
    if backward:
        ple *= 2
    tokens = rows * seq_len
    if engine_forward:
        return int(tokens * (4 * h + qkv_dim + mlp + ple) * act_bytes)
    mamba = mamba_block_bytes(
        arch, rows, seq_len, act_bytes, backward, eager_scan=eager_mamba_scan
    )
    # fp32 RMSNorm inputs saved for backward, one per norm in the block.
    norm_input = tokens * h * 4 if backward and arch.fp32_norm_inputs else 0
    q_norm = (
        tokens * arch.n_heads * arch.head_dim * 4 if backward and arch.qk_norm else 0
    )
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
) -> int:
    """Peak transient memory of the fused no-grad logprob forward.

    Nothing is saved for backward: one block's transients plus the
    full-sequence hidden state handed to the fused logprob kernel.
    """
    block = block_recompute_bytes(
        arch, rows, seq_len, act_bytes, eager_mamba_scan=eager_mamba_scan
    )
    hidden_out = rows * seq_len * arch.hidden_size * act_bytes
    return int(block + hidden_out)


# Chunk length of AgileRL's eager Mamba scan.
EAGER_MAMBA_SCAN_CHUNK = 64

# Per-process job overhead charged when ``orchestrated`` is set.
JOB_OVERHEAD_BYTES = 50 * MiB

# Non-torch device memory a one-rank trainer holds beyond the CUDA context
# and job overhead. Multi-rank trainers charge ``DeviceSpec.nccl_bytes``.
TRAINER_LIB_OVERHEAD_BYTES = 436 * MiB


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
    attention = 3 * h + arch.n_heads * max(arch.head_dim, arch.global_head_dim or 0)
    mlp = 0
    if scope == "all-linear" and arch.mlp_layers:
        inter = int(arch.intermediate_size * arch.peak_mlp_width_factor)
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


def largest_block_params(counts: ParamCounts, arch: ModelArch) -> int:
    """Parameters in the widest decoder block FSDP wraps."""
    if arch.block_exclusive_layers:
        experts_per_layer, router_per_layer = moe_params_per_layer(arch)
        mlp = mlp_params_per_layer(arch) if arch.mlp_layers else 0
        return int(
            max(
                attention_params_per_layer(arch),
                mlp,
                experts_per_layer + router_per_layer,
                mamba_params_per_layer(arch),
            )
        )
    body = counts.attention + counts.mlp + counts.moe_experts + counts.norms
    body += counts.mamba
    return int(body / max(arch.n_layers, 1))


def layer_params(arch: ModelArch, kind: LayerKind) -> int:
    """Parameters of one layer of a block-exclusive stack."""
    if kind == "attention":
        return attention_params_per_layer(arch)
    if kind == "mamba":
        return mamba_params_per_layer(arch)
    if kind == "mlp":
        return mlp_params_per_layer(arch)
    experts, router = moe_params_per_layer(arch)
    return experts + router


def prefetch_window_params(
    counts: ParamCounts, arch: ModelArch, live_units: int, wrap_every_n_blocks: int
) -> int:
    """Decoder parameters gathered at once: ``live_units`` neighbouring FSDP units.

    With the layer order known, the window is the widest run of neighbouring
    units, so a large block next to small ones is counted once. Otherwise
    every unit is the widest block.
    """
    if not arch.layer_kinds:
        return live_units * wrap_every_n_blocks * largest_block_params(counts, arch)
    sizes = [layer_params(arch, kind) for kind in arch.layer_kinds]
    units = [
        sum(sizes[start : start + wrap_every_n_blocks])
        for start in range(0, len(sizes), wrap_every_n_blocks)
    ]
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
) -> int:
    """Unsharded parameter elements held besides the resident shard.

    Excludes the replicated token embeddings. ``reshard_after_forward``
    keeps a prefetch window of decoder units, plus an untied ``lm_head`` and
    root leftovers (multimodal towers, per-layer embeddings). Otherwise every
    sharded parameter stays gathered. A single GPU with resident parameters
    has no second copy.
    """
    token = min(token_embedding_params(arch), max(counts.total, 0))
    sharded = max(counts.total - token, 0)
    if n_gpus <= 1 and not cpu_offload:
        return 0
    if not reshard_after_forward:
        return sharded
    n_units = max(1, -(-arch.n_layers // wrap_every_n_blocks))
    live = min(n_units, 1 + prefetch_units)
    window = prefetch_window_params(counts, arch, live, wrap_every_n_blocks)
    head = 0 if arch.tied_embeddings else counts.lm_head
    extra_embed = max(counts.embedding - token_embedding_params(arch), 0)
    leftover = extra_embed + counts.multimodal_towers + max(counts.unattributed, 0)
    return min(sharded, window + head + leftover)


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
