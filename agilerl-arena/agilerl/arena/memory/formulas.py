# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form memory terms for LLM RL training and generation.

Every function here is a pure function of the specs in
:mod:`agilerl.arena.memory.specs` — no torch, no I/O. The formulas encode the
shape of each memory component as implemented by the framework's LLM
training and generation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    GiB,
    KVCacheDtype,
    LoraTargetScope,
    MiB,
    ModelArch,
    PackedMoeDispatch,
    WeightDtype,
    WeightVariant,
)

#: Realised bytes per quantized parameter, including quantization scales.
#: nf4 with double quantization stores 4 bits + a fp8 absmax per 64-block and
#: a second-level fp32 scale per 256 blocks (~0.53 B/param); bnb int8 keeps
#: outlier columns in fp16 (~1.06 B/param typical). Profiled realised sizes
#: per variant take precedence over these analytic defaults.
QUANTIZED_BYTES_PER_PARAM: dict[str, float] = {
    "nf4": 0.53,
    "int8": 1.06,
    "awq": 0.56,
    "gptq": 0.56,
}

#: Default CUDA-graph pool the vLLM engine captures unless enforce_eager.
#: Subtracted from the KV budget and added back as residency, so it cancels
#: in the generation total. Scales with captured batch sizes (tracking
#: ``max_num_seqs``), not context. 256 MiB is a ceiling on the 40-93 MiB
#: vLLM 0.26 reports.
CUDA_GRAPH_POOL_BYTES = 256 * MiB
#: Bytes per LoRA parameter as actually stored. PEFT's ``get_peft_model``
#: defaults to ``autocast_adapter_dtype=True``, keeping adapter weights fp32
#: even when the base is bf16/fp16 — so gradients and AdamW moments are fp32
#: too (confirmed by rank-sweep measurement on an L4).
ADAPTER_BYTES_PER_PARAM = 4.0

#: What a vLLM process holds on the device *outside* its
#: ``gpu_memory_utilization`` budget, beyond the CUDA context: library
#: workspaces plus engine structures, asleep or awake. The sleeping-engine
#: floor rises with depth and sequence slots: sleep level 1 hands back
#: weights and the KV pool but not per-sequence scheduler state.
ENGINE_OVERHEAD_BASE_BYTES = 512 * MiB
ENGINE_OVERHEAD_PER_LAYER_BYTES = 8 * MiB
#: Per engine sequence slot (``max_num_seqs``).
ENGINE_OVERHEAD_PER_SEQ_BYTES = 8 * MiB

#: Overhead for an awake engine. Distinct from the sleeping-floor formula: the
#: rig samples with the engine asleep, and substituting that formula
#: over-predicts generation on every Qwen.
ENGINE_PROCESS_OVERHEAD_BYTES = 700 * MiB

#: Slack the PyTorch caching allocator holds reserved but not allocated.
#: Charged on the torch-side subtotal. Training only: generation sits in
#: vLLM's CuMem pool, reserved up front at ``gpu_memory_utilization``.
#: 0.07 is the median of reserved/allocated over 184 stored training
#: points (1.069 median). Measured from torch's own counters.
ALLOCATOR_RESERVE_FRACTION = 0.07


def allocator_reserve_bytes(allocated_bytes: float) -> float:
    """Caching-allocator slack over a torch-side allocation subtotal."""
    return max(allocated_bytes, 0.0) * ALLOCATOR_RESERVE_FRACTION


@dataclass(frozen=True)
class ParamCounts:
    """Parameter counts split by matrix group.

    The split matters because quantization, LoRA targeting, and k-bit
    upcasting all act on different groups. ``moe_experts`` is separate from
    ``mlp`` because transformers 5.x stores expert weights as one fused 3D
    parameter per layer, not per-expert ``nn.Linear`` — and both bitsandbytes
    and PEFT dispatch on ``nn.Linear``, so neither reaches them.
    """

    embedding: int
    lm_head: int
    attention: int
    mlp: int
    norms: int
    multimodal_towers: int
    moe_experts: int = 0
    #: Mamba-2 mixer parameters (in/out projections, conv, gate norm). Their
    #: in/out projections are ``nn.Linear`` so bitsandbytes could reach them,
    #: but they are kept out of :attr:`quantizable` — over-charging a
    #: quantized hybrid is the safe direction and no quantized hybrid is
    #: measured.
    mamba: int = 0
    #: Signed reconciliation against the checkpoint's own parameter count: what
    #: the geometry does not account for. A count of nothing in particular, so
    #: it is neither quantizable nor LoRA-targetable, and is held at the
    #: checkpoint dtype.
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

    @property
    def quantizable(self) -> int:
        """Params bnb quantizes: every linear except ``lm_head`` (which the
        framework force-keeps unquantized for exact fused logprobs) and the
        MoE experts (see :attr:`moe_experts`).
        """
        return self.attention + self.mlp


def engine_process_overhead_bytes(arch: ModelArch, max_num_seqs: int = 1) -> int:
    """The vLLM process's own device footprint, outside its budget and beyond
    the CUDA context.
    """
    return int(
        ENGINE_OVERHEAD_BASE_BYTES
        + ENGINE_OVERHEAD_PER_LAYER_BYTES * arch.n_layers
        + ENGINE_OVERHEAD_PER_SEQ_BYTES * max_num_seqs
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
    """Parameters of one Mamba-2 mixer, from the same geometry as its state.

    The packed input projection carries z and x (``2 * d_inner``), B and C
    (``2 * n_groups * d_state`` — together ``conv_dim`` covers x, B and C)
    and the per-head dt; the output projection maps ``d_inner`` back to
    hidden. The causal conv, its bias, the gated RMSNorm and the per-head
    A/D/dt_bias vectors are the remainder — verified tensor-for-tensor
    against Nemotron 3.5's checkpoint (in_proj 10304 x 2688 et al.).
    """
    if not arch.n_mamba_layers:
        return 0
    h = arch.hidden_size
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    in_proj = h * (d_inner + arch.mamba_conv_dim + arch.mamba_n_heads)
    out_proj = d_inner * h
    conv = arch.mamba_conv_dim * (arch.mamba_d_conv + 1)
    return int(in_proj + out_proj + conv + d_inner + 3 * arch.mamba_n_heads)


def moe_params_per_layer(arch: ModelArch) -> tuple[int, int]:
    """(expert parameters, router parameters) of one MoE block."""
    if not arch.is_moe:
        return 0, 0
    assert arch.n_experts is not None
    h = arch.hidden_size
    mlp_matrices = 3 if arch.gated_mlp else 2
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    shared_inter = arch.shared_expert_intermediate_size or expert_inter
    experts = arch.n_experts * mlp_matrices * h * expert_inter
    experts += arch.n_shared_experts * mlp_matrices * h * shared_inter
    return experts, h * arch.n_experts


def param_counts(arch: ModelArch, n_params: int | None = None) -> ParamCounts:
    """Analytic parameter counts from the architecture geometry.

    ``n_params`` is the checkpoint's own total (safetensors metadata). When
    given, the difference against the analytic sum lands in
    :attr:`ParamCounts.unattributed` so weight bytes come out exact even where
    the geometry is incomplete.

    Each group is charged on the layers that hold it: every layer for a
    standard stack, the declared subsets for hybrids and block-exclusive
    layouts (a Nemotron-H layer is one of Mamba, attention, MLP or MoE, so
    charging attention on all 52 would invent 46 blocks).
    """
    h = arch.hidden_size
    mlp_matrices = 3 if arch.gated_mlp else 2
    experts_per_layer, router_per_layer = moe_params_per_layer(arch)
    mlp_per_layer = int(
        mlp_matrices * h * arch.intermediate_size * arch.mlp_width_factor
    )

    embedding = arch.vocab_size * h
    # Per-Layer Embeddings are a second, larger table: one vector per layer
    # per vocabulary entry. On Gemma 4 E2B that is 262144 x 35 x 256 = 2.35B
    # parameters, against 1.5B for everything else -- the reason a "2B
    # effective" model is nothing like 2B on disk.
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

    ``all-linear`` (the framework's default target) wraps every decoder
    linear — q/k/v/o and the MLP matrices — but never the embedding or
    ``lm_head``. Each wrapped ``(out, in)`` linear adds ``rank * (in + out)``.
    """
    h, dh = arch.hidden_size, arch.head_dim
    q_dim = arch.n_heads * dh
    kv_dim = arch.n_kv_heads * dh
    attn = (h + q_dim) + 2 * (h + kv_dim) + (q_dim + h)
    per_layer = attn
    # On a MoE, ``all-linear`` degenerates to attention-only: PEFT resolves the
    # target list by walking ``nn.Linear`` modules, and the experts are not
    # any. See :attr:`ParamCounts.moe_experts`.
    if scope == "all-linear" and not arch.is_moe:
        mlp_matrices = 3 if arch.gated_mlp else 2
        per_layer += mlp_matrices * (h + arch.intermediate_size)
    return rank * per_layer * arch.n_layers


def packed_lora_param_count(arch: ModelArch, rank: int, n_matrices: int) -> int:
    """Adapter parameters for PEFT ``target_parameters`` on packed experts.

    A packed ``(n_experts, out, in)`` Parameter has no ``nn.Linear`` to wrap,
    so PEFT decomposes it per expert: every targeted matrix adds
    ``n_experts * rank * (in + out)`` on each MoE layer. Verified against the
    live model on Nemotron 3.5 Lightning (two targeted matrices, 128 experts,
    23 MoE layers, r=8: 214M parameters, held at the checkpoint dtype).
    """
    if not n_matrices or not arch.is_moe:
        return 0
    assert arch.n_experts is not None
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    per_matrix = arch.n_experts * rank * (arch.hidden_size + expert_inter)
    return int(arch.moe_layers * n_matrices * per_matrix)


def kbit_upcast_bytes(
    counts: ParamCounts, dtype: WeightDtype, tied_embeddings: bool
) -> int:
    """Bytes ``prepare_model_for_kbit_training`` adds *after* the load.

    k-bit preparation upcasts ``lm_head`` and the norms from the checkpoint
    dtype to fp32; on a tied model ``lm_head`` is the embedding table, so the
    upcast hits the biggest unquantized block in the model. A profiled
    ``realised_bytes`` measures the load, not the prep, so this delta applies
    on top of it — skipping that is how a realised nf4 size under-charges a
    tied 0.5B by ~270 MiB.
    """
    delta = DTYPE_BYTES["fp32"] - DTYPE_BYTES[dtype]
    upcast = counts.embedding if tied_embeddings else counts.lm_head
    return int((upcast + counts.norms) * delta)


def weight_bytes(
    counts: ParamCounts,
    dtype: WeightDtype,
    variant: WeightVariant,
    kbit_prepared: bool = False,
    tied_embeddings: bool = False,
) -> int:
    """Realised base-weight bytes for one loaded copy.

    Uses the profiled ``realised_bytes`` when the variant carries one;
    otherwise analytic: quantizable groups at the quantized rate, embeddings
    at the checkpoint dtype, and — under k-bit training preparation —
    ``lm_head`` and norms upcast to fp32. A realised size for a quantized
    variant is the loaded model, so k-bit preparation's upcasts are added
    on top of it rather than assumed inside it.

    ``stripped_multimodal`` drops only the counted towers. Where a checkpoint's
    towers are part of :attr:`ParamCounts.unattributed` instead, stripping
    under-credits the saving; profile the stripped variant to get it exact.

    ``tied_embeddings`` matters only under ``kbit_prepared``. PEFT's
    ``prepare_model_for_kbit_training`` upcasts ``lm_head`` to fp32, and on a
    tied model ``lm_head`` is the embedding table — so the upcast hits a
    block this function otherwise charges at the checkpoint dtype, and
    ``counts.lm_head`` is 0 precisely because it is tied.
    """
    if variant.realised_bytes is not None:
        prep = (
            kbit_upcast_bytes(counts, dtype, tied_embeddings)
            if kbit_prepared and variant.quantization != "none"
            else 0
        )
        return variant.realised_bytes + prep

    towers = 0 if variant.stripped_multimodal else counts.multimodal_towers
    base_bytes = DTYPE_BYTES[dtype]
    if variant.quantization == "none":
        return int((counts.total - counts.multimodal_towers + towers) * base_bytes)

    q_bytes = QUANTIZED_BYTES_PER_PARAM[variant.quantization]
    held_out_bytes = 4.0 if kbit_prepared else base_bytes
    # A tied lm_head is the embedding table, so k-bit prep upcasts that block
    # rather than a separate one.
    embedding_bytes = (
        held_out_bytes if (kbit_prepared and tied_embeddings) else base_bytes
    )
    return int(
        counts.quantizable * q_bytes
        + counts.embedding * embedding_bytes
        + (counts.lm_head + counts.norms) * held_out_bytes
        + towers * base_bytes
        + counts.moe_experts * base_bytes
        + counts.unattributed * base_bytes
    )


def kv_cache_bytes_per_token(
    arch: ModelArch, kv_dtype: KVCacheDtype, weight_dtype: WeightDtype
) -> float:
    """K + V bytes per cached token position across all layers.

    Layers that share an earlier layer's KV store nothing of their own, so
    only the remainder counts -- 15 of Gemma 4 E2B's 35. Neither do the Mamba
    layers of a hybrid model, whose state is constant in context length and
    counted by :func:`mamba_state_bytes` instead; Nemotron-Nano-9B-v2 keeps
    only 4 attention layers out of 56.
    """
    kv_bytes = (
        DTYPE_BYTES[weight_dtype] if kv_dtype == "auto" else DTYPE_BYTES[kv_dtype]
    )
    storing_layers = arch.attention_layers - min(
        arch.n_kv_shared_layers, arch.attention_layers
    )
    return 2 * storing_layers * arch.n_kv_heads * arch.head_dim * kv_bytes


#: State blocks vLLM keeps per sequence, by ``mamba_cache_mode``.
#: ``none`` is one state per running sequence. ``all`` keeps one per
#: KV block (``ceil(max_model_len / block_size)``). ``align`` caches the
#: last token of a scheduler step when it hits a block boundary, which
#: bounds residency at two.
MAMBA_STATE_BLOCKS: dict[str, int] = {"none": 1, "align": 2}
#: vLLM's default KV block size, which ``all`` mode divides context by.
KV_BLOCK_SIZE_DEFAULT = 16
#: Granularity the aligned block size is rounded up to, from the attention
#: backend's supported kernel block sizes
#: (``kernel_block_alignment_size`` in vLLM's ``platforms/interface.py``).
MAMBA_KERNEL_BLOCK_ALIGNMENT = 32


def mamba_state_blocks_per_seq(
    mode: str, max_model_len: int, block_size: int = KV_BLOCK_SIZE_DEFAULT
) -> int:
    """State blocks resident per sequence under a given cache mode."""
    if mode == "all":
        return -(-max_model_len // max(block_size, 1))
    return MAMBA_STATE_BLOCKS.get(mode, 1)


def mamba_page_bytes(arch: ModelArch, state_dtype: WeightDtype = "bf16") -> int:
    """One recurrent layer's state: the conv window plus the SSM state.

    The two halves do not share a dtype. The conv window follows the model,
    but the recurrent state is resolved per model by vLLM and is often fp32 --
    which doubles the page, and with it the aligned block size. Confirmed
    against ``get_mamba_state_dtype_from_config``: Nemotron-H returns
    ``(bfloat16, float32)``, Falcon-H1 ``(bfloat16, bfloat16)``.
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
    kv_dtype: KVCacheDtype = "auto",
    weight_dtype: WeightDtype = "bf16",
    granularity: int = MAMBA_KERNEL_BLOCK_ALIGNMENT,
) -> int:
    """Block size vLLM adopts so an attention page holds a Mamba page.

    A hybrid model puts both caches in one block pool, which requires the two
    page sizes to match. vLLM raises the *attention* block size until its page
    is at least as large as a Mamba page, then pads the Mamba page up to it
    exactly. The consequence is easy to miss: the block size stops being 16 and
    becomes hundreds or thousands of tokens, so every sequence's KV rounds up
    to a much coarser page.

    Reproduces vLLM's own choice on Falcon-H1 -- 1,594,368 bytes of state per
    layer over 1,024 bytes of KV per token is 1,557 tokens, rounded up to
    **1,568** -- which is the figure it logs.
    """
    if not arch.is_hybrid_ssm or not arch.attention_layers:
        return KV_BLOCK_SIZE_DEFAULT
    per_token_per_layer = kv_cache_bytes_per_token(arch, kv_dtype, weight_dtype) / (
        arch.attention_layers
    )
    if per_token_per_layer <= 0:
        return KV_BLOCK_SIZE_DEFAULT
    tokens = -(-mamba_page_bytes(arch, weight_dtype) // int(per_token_per_layer))
    return max(KV_BLOCK_SIZE_DEFAULT, -(-tokens // granularity) * granularity)


def mamba_state_bytes(
    arch: ModelArch,
    concurrency: int,
    mode: str = "none",
    max_model_len: int = 0,
    block_size: int = KV_BLOCK_SIZE_DEFAULT,
    state_dtype: WeightDtype = "bf16",
) -> int:
    """Recurrent-state cache for a hybrid model's Mamba layers.

    Two tensors per recurrent layer per slot, following vLLM's
    ``mamba2_state_shape``: a causal-conv window of ``(conv_dim, d_conv - 1)``
    and a recurrent state of ``(n_heads, d_head, d_state)``. Both are constant
    in context length -- that is the point of the architecture — so this term
    scales with concurrency and the cache mode, never with ``seq_len``.
    """
    if not arch.is_hybrid_ssm:
        return 0
    per_layer = mamba_page_bytes(arch, state_dtype)
    blocks = mamba_state_blocks_per_seq(mode, max_model_len, block_size)
    return int(arch.n_mamba_layers * per_layer * concurrency * blocks)


def kv_cache_demand_bytes(
    arch: ModelArch,
    kv_dtype: KVCacheDtype,
    weight_dtype: WeightDtype,
    concurrency: int,
    seq_len: int,
) -> int:
    """Worst-case KV bytes the workload asks for: every concurrent sequence
    at full context. Sliding-window layers cap their per-sequence growth at
    the window size.
    """
    per_token = kv_cache_bytes_per_token(arch, kv_dtype, weight_dtype)
    if arch.sliding_window is None:
        return int(concurrency * seq_len * per_token)
    windowed = arch.sliding_window_layer_fraction
    effective = windowed * min(seq_len, arch.sliding_window) + (1 - windowed) * seq_len
    return int(concurrency * effective * per_token)


#: Bounds the fused-logprob chunk auto-tune, and an explicit ``chunk_rows``.
#: Floor: ``lm_head`` is re-read once per chunk, so traffic scales as
#: ``1/chunk_rows`` (8.1 TiB at 1 vs 32 GiB at 256 on Qwen2.5-0.5B / 32k).
#: Ceiling: the fp32 tile grows linearly; at vocab 152k a 4096-row tile is
#: 2.4 GiB held twice. The cap stops a throughput setting from becoming
#: the largest allocation in the step.
FUSED_CHUNK_ROWS_MIN = 128
FUSED_CHUNK_ROWS_MAX = 4096


def resolve_chunk_rows(vocab_size: int, explicit: int | None = None) -> int:
    """The framework's fused-logprob chunk auto-tune: rows sized to a
    256 MiB fp32 logit workspace, clamped to
    ``[FUSED_CHUNK_ROWS_MIN, FUSED_CHUNK_ROWS_MAX]``.

    Deliberately a fixed *memory* budget, and so a function of vocabulary
    alone. The speed gain from a larger chunk instead scales with hidden
    size, because what a chunk amortises is one ``lm_head`` re-read of
    ``vocab x hidden``: on an L4 at 32k tokens, raising the default to 4096
    is worth 15% of step time on Qwen2.5-0.5B and 22% on Qwen2.5-7B, for a
    workspace 9x larger. Sizing the default off that gain would make the
    footprint scale with the model twice over, so the trade is left to the
    caller and surfaced by the advice engine, which knows the headroom.

    An explicit value passes through unchanged — it is range-checked at
    construction instead, so a bad setting is rejected rather than silently
    rewritten.
    """
    if explicit is not None:
        return explicit
    rows = 256 * MiB // max(1, vocab_size * 4)
    return max(FUSED_CHUNK_ROWS_MIN, min(FUSED_CHUNK_ROWS_MAX, rows))


def fused_head_upcast_bytes(arch: ModelArch) -> int:
    """The fp32 ``(vocab, hidden)`` lm_head copy the fused-logprob path hoists.

    ``fp32_lm_head_operands`` (``agilerl/algorithms/core/llm_ops/
    fused_logprobs.py``) upcasts the head *once per chunk loop* rather than
    once per chunk, so under ``cast_logprobs_to_fp32`` (the framework
    default) a full fp32 copy of the head is live for the whole loop —
    forward and backward alike. On a tied Qwen2.5-0.5B that is
    151936 x 896 x 4 = 519 MiB.

    Zero when the head is already fp32 — which k-bit preparation makes true
    persistently (``kbit_upcast_bytes`` charges that instead), and an fp32
    checkpoint gets for free. The caller gates on those.
    """
    return int(arch.vocab_size * arch.hidden_size * 4)


def resolve_max_num_batched_tokens(
    max_num_seqs: int, max_model_len: int, explicit: int | None = None
) -> int:
    """The framework's resolution rule for vLLM's scheduler-step token cap
    (deliberately below ``max_num_seqs * max_model_len``, which drives
    multi-GiB compile/profile tensors at long context).
    """
    if explicit is not None:
        return explicit
    return min(
        max_num_seqs * max_model_len,
        max(max_model_len, max_num_seqs * 8192),
    )


def resolve_attn_implementation(
    requested: str = "auto", flash_attn_installed: bool = False
) -> str:
    """The framework's backend resolution.

    ``auto`` picks FlashAttention-2 only when the ``flash_attn`` package is
    importable. It is not part of the ``llm`` extra, so a stock install
    resolves to SDPA.
    """
    if requested != "auto":
        return requested
    return "flash_attention_2" if flash_attn_installed else "sdpa"


def materializes_attention_scores(attn_implementation: str) -> bool:
    """Whether the backend builds a ``rows x heads x S x S`` score matrix.

    Only ``eager`` does. SDPA, FlashAttention-2 and flex_attention do not,
    including on windowed models.
    """
    return attn_implementation == "eager"


def activation_hidden_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float
) -> int:
    """Checkpoint-boundary activations saved for backward: one hidden state
    per layer, ``rows x seq_len x hidden x n_layers``.
    """
    return int(rows * seq_len * arch.hidden_size * arch.n_layers * act_bytes)


def lora_input_cast_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    scope: LoraTargetScope = "all-linear",
    gradient_checkpointing: bool = True,
) -> int:
    """The fp32 copy PEFT makes of a wrapped linear's input.

    ``get_peft_model`` defaults to ``autocast_adapter_dtype=True``, so the
    adapters are fp32 while the base is bf16, and each wrapped linear casts
    its input to fp32 (``BaseTunerLayer._cast_input_dtype``).

    Under gradient checkpointing the casts **do not accumulate**: the block
    is recomputed and differentiated as one unit, and the instant holds the
    *widest single* cast rather than the sum over its wrapped linears. Two
    allocator timelines pin that to the byte, and both land on the MLP
    down-projection reading the intermediate:

        model         rows x seq   widest input   this term   observed
        SmolLM2-1.7B     8 x 4096       8192       1024 MiB   1024 MiB
        Gemma 4 E2B      4 x 4096      12288        768 MiB    768 MiB

    Observed is the timeline's ``adapters`` excursion, which carries the
    casts because PEFT allocates them inside the LoRA layer. Summing the
    block's seven wrapped inputs instead — what this charged until the
    traces were read — over-predicts SmolLM2's worst corner by 2 GiB, and
    that surplus was masked for a long time by an under-read allocator
    reserve (:data:`ALLOCATOR_RESERVE_FRACTION`); the two were measured
    together and corrected together.
    """
    h = arch.hidden_size
    # In_features of the linears PEFT wraps: q/k/v and the MLP's gate/up read
    # the residual stream, o reads the concatenated head output, down reads
    # the intermediate. ``all-linear`` degenerates to attention-only on a MoE,
    # whose experts are not nn.Linear -- see ParamCounts.moe_experts.
    if gradient_checkpointing:
        attn_out = arch.n_heads * max(arch.head_dim, arch.global_head_dim or 0)
        widths = [h, attn_out]
        if scope == "all-linear" and not arch.is_moe:
            widths.append(int(arch.intermediate_size * arch.peak_mlp_width_factor))
        return int(rows * seq_len * max(widths) * DTYPE_BYTES["fp32"])
    # No checkpointing: nothing is recomputed, so every layer's casts are
    # saved from its forward all the way to its backward and they do
    # accumulate. Unmeasured -- the corpus is checkpointed throughout -- and
    # deliberately the conservative reading.
    width = 3 * h + arch.mean_qkv_dim
    if scope == "all-linear" and not arch.is_moe:
        inter = int(arch.intermediate_size * arch.mlp_width_factor)
        width += (2 * h + inter) if arch.gated_mlp else (h + inter)
    return int(rows * seq_len * width * arch.n_layers * DTYPE_BYTES["fp32"])


def moe_dispatch_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float, backward: bool = False
) -> int:
    """Gather/scatter buffers the fused-expert forward builds per layer.

    ``OlmoeExperts.forward`` (and its siblings) routes by materialising
    copies: a gather of ``tokens x top_k x hidden``, a zeros accumulator, and
    an int64 one-hot mask over the *total* expert count while everything else
    scales with the active count.
    """
    if not arch.is_moe:
        return 0
    tokens = rows * seq_len
    top_k = (arch.n_experts_per_tok or 1) + arch.n_shared_experts
    gather = tokens * top_k * arch.hidden_size * act_bytes
    if backward:
        gather *= 2
    accumulator = tokens * arch.hidden_size * act_bytes
    mask = tokens * top_k * (arch.n_experts or 0) * 8
    return int(gather + accumulator + mask)


#: Extra expert-gather blocks resident at once, on top of the one block
#: :func:`moe_dispatch_bytes` already charges. Checkpointing frees dense
#: intermediates at each boundary; gathered expert copies stay. 4 is the
#: largest value that does not raise corpus error across three measured
#: MoEs (leave-one-model-out held-out error 4.01%). All three route
#: top_k=8, so the gather shape is not identified.
MOE_GATHER_RESIDENT_BLOCKS = 4


#: bitsandbytes backward workspace per gradient token. Paired bf16/nf4
#: runs put the slope at 40 KiB/token + 1.5 bytes per (n_layers x
#: hidden) per token (R^2 0.982). Shape is established; constants are
#: provisional.
BNB_WORKSPACE_BYTES_PER_TOKEN = 40 * 1024
BNB_WORKSPACE_BYTES_PER_LAYER_HIDDEN_TOKEN = 1.5


def bnb_backward_workspace_bytes(arch: ModelArch, rows: int, seq_len: int) -> int:
    """Transient workspace the bitsandbytes backward holds per gradient token.

    Zero unquantized: this is what makes the bf16/nf4 trade invert.
    """
    return int(
        rows
        * seq_len
        * (
            BNB_WORKSPACE_BYTES_PER_TOKEN
            + BNB_WORKSPACE_BYTES_PER_LAYER_HIDDEN_TOKEN
            * arch.n_layers
            * arch.hidden_size
        )
    )


def moe_resident_gather_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float
) -> int:
    """Expert gather copies that outlive the block that made them.

    Training only: the generation path never reaches the backward that keeps
    them alive.
    """
    if not arch.is_moe:
        return 0
    top_k = (arch.n_experts_per_tok or 1) + arch.n_shared_experts
    return int(
        MOE_GATHER_RESIDENT_BLOCKS
        * rows
        * seq_len
        * top_k
        * arch.hidden_size
        * act_bytes
    )


def block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    flash_attention: bool = True,
    backward: bool = False,
    engine_forward: bool = False,
) -> int:
    """Peak transient activations of one transformer block.

    Live tensors at the MLP peak: residual stream copies, qkv projections,
    attention output, and the gated-MLP intermediates. Without FlashAttention
    the math path additionally materialises the ``heads x S x S`` score
    matrix.

    ``backward=True`` is the checkpointed-recompute case, where the block is
    replayed and then differentiated: each gated-MLP intermediate then has a
    gradient live alongside it while the residual-stream and qkv tensors do
    not, so the MLP term doubles and the rest does not.

    ``engine_forward=True`` is the vLLM case: the engine runs its own
    kernels, not the trainer's. Transformers routes MoE tokens via gather,
    accumulator and a one-hot mask; vLLM's fused-MoE triton kernel routes
    in-kernel and never materialises those copies, so they are not charged
    here. Residual-stream copies stay at the trainer's ``4h`` per token.
    """
    h = arch.hidden_size
    mlp_matrices = 3 if arch.gated_mlp else 2
    if arch.is_moe:
        inter = arch.expert_intermediate_size or arch.intermediate_size
        shared_inter = arch.shared_expert_intermediate_size or inter
        # Active FFN width per token: the routed experts it selects plus the
        # always-on shared expert at its own (possibly wider) intermediate.
        active_width = (
            arch.n_experts_per_tok or 1
        ) * inter + arch.n_shared_experts * shared_inter
    else:
        active_width = int(arch.intermediate_size * arch.peak_mlp_width_factor)
    # One block's transient peaks at the *widest* block, not the mean one:
    # both the checkpointed recompute and the engine's layer-at-a-time
    # forward run block by block, so a Gemma 4 double-wide-MLP global
    # block is what binds. Identical to the means on uniform stacks.
    qkv_dim = arch.peak_qkv_dim
    mlp = mlp_matrices * active_width
    ple = arch.n_layers * arch.per_layer_input_dim
    if backward:
        mlp *= 2
        ple *= 2
    per_token = 4 * h + qkv_dim + mlp + ple
    peak = rows * seq_len * per_token * act_bytes
    if not engine_forward:
        peak += moe_dispatch_bytes(arch, rows, seq_len, act_bytes, backward)
    if not flash_attention:
        # The math path holds three score matrices, not one: the pre-softmax
        # scores, the softmax output saved for backward, and that output's
        # gradient during it (backend A/B on Gemma 4 E2B measured 3.33x S^2).
        peak += 3 * rows * arch.n_heads * seq_len * seq_len * act_bytes
    return int(peak)


def nograd_forward_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    flash_attention: bool = True,
) -> int:
    """Peak transient memory of the fused no-grad logprob forward (reference
    and old-logprob computation). Nothing is saved for backward, so the peak
    is one block's transients plus the full-sequence hidden state handed to
    the fused logprob kernel.
    """
    block = block_recompute_bytes(
        arch, rows, seq_len, act_bytes, flash_attention=flash_attention
    )
    hidden_out = rows * seq_len * arch.hidden_size * act_bytes
    return int(block + hidden_out)


def optimizer_bytes_per_trainable_param(distributed: str) -> float:
    """AdamW state per trainable (LoRA) parameter.

    Both moments live in the parameter dtype, and LoRA parameters are fp32
    (see :data:`ADAPTER_BYTES_PER_PARAM`). DeepSpeed keeps an fp32 master
    copy on top. This is the *unsharded* figure; ZeRO stages 2 and 3 divide
    it across the data-parallel group in the estimator.
    """
    moments = 2 * ADAPTER_BYTES_PER_PARAM
    if distributed == "deepspeed":
        return moments + 4.0
    return moments


#: Device memory the Ray worker plumbing costs a GPU actor over the same
#: workload as a plain process: measured at ~50 MiB on an L4 (agilerl-ray
#: worker against a bare run, same image). This is the *explained* part; an
#: orchestrated agilerl-ray job was once observed peaking 3.2 GiB above the
#: bare-process equivalent, which is unexplained, sits in the orchestration
#: path rather than this model, and is deliberately not charged here.
RAY_ACTOR_OVERHEAD_BYTES = 50 * MiB

#: Non-torch device memory the trainer process holds beyond the CUDA
#: context: cuBLAS/cuDNN workspaces and attention autotuning. Measured at
#: 583 MiB (SFT sweep, Qwen2.5-0.5B, A100), constant across batch and
#: rank. Charged only when the run starts no vLLM engine: colocated
#: engine-floor constants already include the trainer libraries.
TRAINER_LIB_OVERHEAD_BYTES = 583 * MiB


#: ZeRO-3 backward surcharge on a packed-MoE hybrid, per token of every
#: completion row the rank holds. Fitted on Nemotron 3.5 Lightning 30B-A3B
#: (A100-80): worlds 4 and 2 anchor, world 3 held out within +1.6/-2.2 GiB.
#: Carrier is per-row state that survives each micro-step, plus the
#: gathered leaf's backward (re-gathered weights, PEFT effective copies,
#: dispatch intermediates). One model, one device: provisional.
ZERO3_PACKED_MOE_BACKWARD_BYTES_PER_ROW_TOKEN = int(0.1499 * MiB)

#: ZeRO-3's persistent device state beyond the sharded weights and the
#: two-block gather working set: partition/allgather caches, reduce buckets,
#: and the saved effective-weight copies standing at the bound instant.
#: Measured as the row-free intercept of the same residual; ZeRO-2 on the
#: same model shows none of it (residual within +-1.5 GiB with no term at
#: all).
ZERO3_PERSISTENT_STATE_BYTES = int(5.61 * GiB)


def zero3_packed_moe_backward_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    packed_matrices: int,
    dispatch: PackedMoeDispatch = "materialized",
) -> int:
    """ZeRO-3 backward surcharge for materialized packed-expert adapters.

    ``rows`` is every completion row the rank holds for the update, not the
    gradient micro-batch: the measured surcharge doubled when the per-rank
    rows did (world 2's 8 against world 4's 4) at a fixed micro-batch of
    one. Zero when the contracted dispatch is assumed (the planned fix
    computes the low-rank delta by contraction and never forms the effective
    weight), when nothing targets the packed experts, or off ZeRO-3 —
    ZeRO-2 measured no surcharge on the same model.
    """
    if dispatch != "materialized" or not packed_matrices or not arch.is_moe:
        return 0
    return int(rows * seq_len * ZERO3_PACKED_MOE_BACKWARD_BYTES_PER_ROW_TOKEN)


def zero3_persistent_state_bytes(
    arch: ModelArch,
    packed_matrices: int,
    dispatch: PackedMoeDispatch,
    act_bytes: float,
) -> int:
    """ZeRO-3's persistent buffers and standing effective-weight copies.

    Under the contracted-dispatch assumption the effective-weight copies
    (one per targeted packed matrix at the widest block) drop out; the
    partition caches and reduce buckets remain.
    """
    total = ZERO3_PERSISTENT_STATE_BYTES
    if dispatch != "materialized" and packed_matrices and arch.is_moe:
        assert arch.n_experts is not None
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        w_eff = int(packed_matrices * arch.n_experts * expert_inter * arch.hidden_size)
        total = max(0, total - int(w_eff * act_bytes))
    return total


def zero3_gather_bytes(counts: ParamCounts, arch: ModelArch, dtype: str) -> int:
    """ZeRO-3's per-GPU gather working set, analytic and unmeasured.

    Stage 3 shards every parameter — the frozen base included, which for a
    LoRA run is where the memory is — and all-gathers each module's weights
    just-in-time for its forward and backward. With prefetch the working set
    is roughly two modules in flight: the one executing and the one being
    gathered. The largest gathered blocks are the embedding table and the
    *widest* decoder layer, so twice the larger of the two is charged. The
    widest layer matters on block-exclusive MoE hybrids: a Nemotron-H MoE
    block is gathered as one leaf (the framework marks the sparse block a
    ZeRO-3 leaf module so expert iteration cannot desynchronise the ranks'
    allgather order), and at 1.3B parameters it is more than double the
    mean layer. No ZeRO-3 measurement backs this yet; the estimator says so
    in its warnings.
    """
    if arch.block_exclusive_layers:
        experts_per_layer, router_per_layer = moe_params_per_layer(arch)
        block_candidates = [
            float(attention_params_per_layer(arch)),
            float(experts_per_layer + router_per_layer),
            float(mamba_params_per_layer(arch)),
        ]
        per_layer = max(block_candidates)
    else:
        per_layer = (
            counts.attention + counts.mlp + counts.moe_experts + counts.norms
        ) / max(arch.n_layers, 1)
    largest = max(float(counts.embedding + counts.lm_head), per_layer)
    return int(2 * largest * DTYPE_BYTES[dtype])
