"""Closed-form memory terms for LLM RL training and generation.

Every function here is a pure function of the specs in
:mod:`agilerl.memory.specs` — no torch, no I/O. The formulas encode the
*shape* of each memory component as implemented by the framework's actual
code paths (fused chunked logprobs, gradient checkpointing, LoRA-only
training, colocated vLLM); per-(model, device) calibration corrects the
constants, not the shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from agilerl.memory.specs import (
    DTYPE_BYTES,
    KVCacheDtype,
    LoraTargetScope,
    MiB,
    ModelArch,
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

#: Fused-logprob workspace budget used by the chunk auto-tune: chunk_rows is
#: sized so the transient fp32 logit tile stays under 256 MiB.
FUSED_LOGIT_WORKSPACE_BYTES = 256 * MiB
FUSED_CHUNK_ROWS_MIN = 128
FUSED_CHUNK_ROWS_MAX = 4096

#: Default CUDA-graph pool the vLLM engine captures unless enforce_eager.
CUDA_GRAPH_POOL_BYTES = 2 * 1024**3
#: CUDA context + driver reserve per process that owns the device.
CUDA_CONTEXT_BYTES = int(0.75 * 1024**3)
#: Bytes per LoRA parameter as actually stored. PEFT's ``get_peft_model``
#: defaults to ``autocast_adapter_dtype=True``, keeping adapter weights fp32
#: even when the base is bf16/fp16 — so gradients and AdamW moments are fp32
#: too. Measured on an L4: raising rank 8 -> 64 on Qwen2.5-0.5B costs ~20
#: bytes per added parameter, which only resolves as 4 (actor) + 4
#: (reference) + 4 (grad) + 8 (two moments) with fp32 adapters.
ADAPTER_BYTES_PER_PARAM = 4.0

#: What a sleeping (level 1) vLLM engine leaves on the device beyond the CUDA
#: context: engine structures that survive the sleep. Measured at roughly
#: 0.2-0.5 GiB on vLLM 0.23; calibration refines it per (model, device).
SLEEPING_ENGINE_RESIDUAL_BYTES = int(0.25 * 1024**3)


@dataclass(frozen=True)
class ParamCounts:
    """Parameter counts split by matrix group.

    The split matters because quantization, LoRA targeting, and k-bit
    upcasting all act on different groups.
    """

    embedding: int
    lm_head: int
    attention: int
    mlp: int
    norms: int
    multimodal_towers: int

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.lm_head
            + self.attention
            + self.mlp
            + self.norms
            + self.multimodal_towers
        )

    @property
    def quantizable(self) -> int:
        """Params bnb quantizes: every linear except ``lm_head`` (which the
        framework force-keeps unquantized for exact fused logprobs).
        """
        return self.attention + self.mlp


def param_counts(arch: ModelArch) -> ParamCounts:
    """Analytic parameter counts from the architecture geometry."""
    h, dh = arch.hidden_size, arch.head_dim
    q_dim = arch.n_heads * dh
    kv_dim = arch.n_kv_heads * dh
    attn_per_layer = h * q_dim + 2 * h * kv_dim + q_dim * h
    if arch.attn_bias:
        attn_per_layer += q_dim + 2 * kv_dim

    mlp_matrices = 3 if arch.gated_mlp else 2
    if arch.is_moe:
        assert arch.n_experts is not None
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        n_ffn = arch.n_experts + arch.n_shared_experts
        mlp_per_layer = n_ffn * mlp_matrices * h * expert_inter
        mlp_per_layer += h * arch.n_experts  # router
    else:
        mlp_per_layer = mlp_matrices * h * arch.intermediate_size

    embedding = arch.vocab_size * h
    lm_head = 0 if arch.tied_embeddings else arch.vocab_size * h
    norms = arch.n_layers * 2 * h + h
    return ParamCounts(
        embedding=embedding,
        lm_head=lm_head,
        attention=arch.n_layers * attn_per_layer,
        mlp=arch.n_layers * mlp_per_layer,
        norms=norms,
        multimodal_towers=arch.multimodal_tower_params,
    )


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
    if scope == "all-linear":
        inter = (
            arch.expert_intermediate_size or arch.intermediate_size
            if arch.is_moe
            else arch.intermediate_size
        )
        n_ffn = (arch.n_experts or 0) + arch.n_shared_experts if arch.is_moe else 1
        mlp_matrices = 3 if arch.gated_mlp else 2
        per_layer += n_ffn * mlp_matrices * (h + inter)
    return rank * per_layer * arch.n_layers


def weight_bytes(
    counts: ParamCounts,
    dtype: WeightDtype,
    variant: WeightVariant,
    kbit_prepared: bool = False,
) -> int:
    """Realised base-weight bytes for one loaded copy.

    Uses the profiled ``realised_bytes`` when the variant carries one;
    otherwise analytic: quantizable groups at the quantized rate, embeddings
    at the checkpoint dtype, and — under k-bit training preparation —
    ``lm_head`` and norms upcast to fp32.
    """
    if variant.realised_bytes is not None:
        return variant.realised_bytes

    towers = 0 if variant.stripped_multimodal else counts.multimodal_towers
    base_bytes = DTYPE_BYTES[dtype]
    if variant.quantization == "none":
        return int(
            (
                counts.embedding
                + counts.lm_head
                + counts.attention
                + counts.mlp
                + counts.norms
                + towers
            )
            * base_bytes
        )

    q_bytes = QUANTIZED_BYTES_PER_PARAM[variant.quantization]
    held_out_bytes = 4.0 if kbit_prepared else base_bytes
    return int(
        counts.quantizable * q_bytes
        + counts.embedding * base_bytes
        + (counts.lm_head + counts.norms) * held_out_bytes
        + towers * base_bytes
    )


def kv_cache_bytes_per_token(
    arch: ModelArch, kv_dtype: KVCacheDtype, weight_dtype: WeightDtype
) -> float:
    """K + V bytes per cached token position across all layers."""
    kv_bytes = (
        DTYPE_BYTES[weight_dtype] if kv_dtype == "auto" else DTYPE_BYTES[kv_dtype]
    )
    return 2 * arch.n_layers * arch.n_kv_heads * arch.head_dim * kv_bytes


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


def resolve_chunk_rows(vocab_size: int, explicit: int | None = None) -> int:
    """Mirror of the framework's fused-logprob chunk auto-tune: rows sized to
    a 256 MiB fp32 logit workspace, clamped to [128, 4096].
    """
    if explicit is not None:
        return explicit
    rows = FUSED_LOGIT_WORKSPACE_BYTES // (vocab_size * 4)
    return max(FUSED_CHUNK_ROWS_MIN, min(FUSED_CHUNK_ROWS_MAX, rows))


def logit_workspace_bytes(vocab_size: int, chunk_rows: int | None = None) -> int:
    """Transient fp32 logit tile of the fused chunked logprob path. The full
    ``(B, S, V)`` slab is never built; this tile is the only vocab-sized
    allocation in either direction of the pass.
    """
    return resolve_chunk_rows(vocab_size, chunk_rows) * vocab_size * 4


def resolve_max_num_batched_tokens(
    max_num_seqs: int, max_model_len: int, explicit: int | None = None
) -> int:
    """Mirror of the framework's resolution rule for vLLM's scheduler-step
    token cap (deliberately below ``max_num_seqs * max_model_len``, which
    drives multi-GiB compile/profile tensors at long context).
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
    """Mirror of the framework's backend resolution.

    ``auto`` picks FlashAttention-2 only when the ``flash_attn`` package is
    importable. It is not part of the ``llm`` extra, so a stock install
    resolves to SDPA — which matters, because SDPA is the one backend that
    can still materialise the score matrix.
    """
    if requested != "auto":
        return requested
    return "flash_attention_2" if flash_attn_installed else "sdpa"


def materializes_attention_scores(attn_implementation: str, arch: ModelArch) -> bool:
    """Whether the backend builds a ``rows x heads x S x S`` score matrix.

    The trap is SDPA: its flash kernel is only chosen when no explicit
    attention mask is passed. Sliding-window models pass one, so SDPA falls
    back to the math backend and materialises the full S x S scores and bias
    — the difference between O(S) and O(S^2) activation memory, and the
    reason long-context runs on windowed models OOM where dense ones do not.

    FlashAttention-2 and PyTorch's FlexAttention both stay O(S): FlexAttention
    expresses the window as block sparsity instead of a dense mask, which is
    why it is the recommended backend for windowed models without the
    ``flash_attn`` package.
    """
    if attn_implementation == "eager":
        return True
    if attn_implementation in ("flash_attention_2", "flex_attention"):
        return False
    return arch.sliding_window is not None


def activation_hidden_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float
) -> int:
    """Checkpoint-boundary activations saved for backward: one hidden state
    per layer, ``rows x seq_len x hidden x n_layers``.
    """
    return int(rows * seq_len * arch.hidden_size * arch.n_layers * act_bytes)


def block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    flash_attention: bool = True,
) -> int:
    """Peak transient activations of recomputing one transformer block during
    backward (gradient checkpointing). Live tensors at the MLP peak: residual
    stream copies, qkv projections, attention output, and the gated-MLP
    intermediates. Without FlashAttention the SDPA math path additionally
    materialises the ``heads x S x S`` score matrix.
    """
    h = arch.hidden_size
    inter = (
        arch.expert_intermediate_size or arch.intermediate_size
        if arch.is_moe
        else arch.intermediate_size
    )
    if arch.is_moe:
        active_ffn = (arch.n_experts_per_tok or 1) + arch.n_shared_experts
    else:
        active_ffn = 1
    qkv_dim = (arch.n_heads + 2 * arch.n_kv_heads) * arch.head_dim
    per_token = 4 * h + qkv_dim + 3 * inter * active_ffn
    peak = rows * seq_len * per_token * act_bytes
    if not flash_attention:
        peak += rows * arch.n_heads * seq_len * seq_len * act_bytes
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


def optimizer_bytes_per_trainable_param(
    weight_dtype: WeightDtype, distributed: str
) -> float:
    """AdamW state per trainable (LoRA) parameter.

    Both moments live in the parameter dtype, and LoRA parameters are fp32
    (see :data:`ADAPTER_BYTES_PER_PARAM`). DeepSpeed keeps an fp32 master
    copy on top.
    """
    moments = 2 * ADAPTER_BYTES_PER_PARAM
    if distributed == "deepspeed":
        return moments + 4.0
    return moments


def grad_bytes_per_trainable_param(
    weight_dtype: WeightDtype, distributed: str
) -> float:
    """Gradient buffer per trainable parameter — fp32, mirroring the fp32
    adapter parameters it accumulates into.
    """
    return ADAPTER_BYTES_PER_PARAM
