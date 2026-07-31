# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
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
#: Deprecated as a modelling constant: the context is device-dependent by a
#: factor of two (A100 501 MiB vs L4 226 MiB, both measured). Use
#: ``DeviceSpec.context_bytes``. Retained only so older fixtures, whose
#: residuals were derived against it, still replay.
CUDA_CONTEXT_BYTES = int(0.75 * 1024**3)
#: Bytes per LoRA parameter as actually stored. PEFT's ``get_peft_model``
#: defaults to ``autocast_adapter_dtype=True``, keeping adapter weights fp32
#: even when the base is bf16/fp16 — so gradients and AdamW moments are fp32
#: too. Measured on an L4: raising rank 8 -> 64 on Qwen2.5-0.5B costs ~20
#: bytes per added parameter, which only resolves as 4 (actor) + 4
#: (reference) + 4 (grad) + 8 (two moments) with fp32 adapters.
ADAPTER_BYTES_PER_PARAM = 4.0

#: What a vLLM process holds on the device *outside* its
#: ``gpu_memory_utilization`` budget, beyond the bare CUDA context: CUDA
#: library workspaces (cuBLAS/NCCL/flashinfer handles) plus engine
#: structures. Present whether the engine is asleep or awake.
#:
#: Two independent derivations agree, which is why they are one constant
#: rather than two. From the sleeping floors of all ten fixtures, net of each
#: device's own context: A100 1224 - 501 = 723 MiB, L4 913 - 226 = 687. And
#: from the median shortfall of the 198 *generation* points, where the engine
#: is awake: A100 713 MiB, L4 596. Near enough device-independent, unlike the
#: context itself.
#:
#: Charging it to the sleeping trainer but not to generation was worth a
#: uniform +5.0% under-prediction on the generation phase.
ENGINE_PROCESS_OVERHEAD_BYTES = 700 * 1024**2

#: Deprecated alias: this was only ever the sleeping case of the above.
SLEEPING_ENGINE_RESIDUAL_BYTES = ENGINE_PROCESS_OVERHEAD_BYTES


@dataclass(frozen=True)
class ParamCounts:
    """Parameter counts split by matrix group.

    The split matters because quantization, LoRA targeting, and k-bit
    upcasting all act on different groups.

    ``moe_experts`` is separate from ``mlp`` because transformers 5.x stores
    expert weights as one fused 3D parameter per layer
    (``layers.N.experts.gate_up_proj``) rather than per-expert ``nn.Linear``.
    Both bitsandbytes and PEFT dispatch on ``nn.Linear``, so neither reaches
    them: an MoE cannot be bnb-quantized and ``all-linear`` LoRA wraps only
    its attention. Measured on transformers 5.14 by counting parameters
    inside ``nn.Linear`` on a meta-device instantiation:

    ==========================  ======  ==============  ============
    model                       total   in ``nn.Linear``  quantizable
    ==========================  ======  ==============  ============
    google/gemma-4-26B-A4B-it   25.81B  2.94B           11.4%
    allenai/OLMoE-1B-7B-0924    6.92B   0.37B           5.4%
    ==========================  ======  ==============  ============

    So "26B at nf4" is 22.9B still in bf16 (45.7 GB) and will not load on a
    40 GB A100 — which is exactly what it did, mid-load, at 38.8 GB resident.
    """

    embedding: int
    lm_head: int
    attention: int
    mlp: int
    norms: int
    multimodal_towers: int
    moe_experts: int = 0

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
        )

    @property
    def quantizable(self) -> int:
        """Params bnb quantizes: every linear except ``lm_head`` (which the
        framework force-keeps unquantized for exact fused logprobs) and the
        MoE experts (see :attr:`moe_experts`).
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
    if arch.global_head_dim and arch.global_head_dim != dh:
        # Full-attention layers project to a wider head; scale the whole
        # attention block by the layer-averaged width.
        attn_per_layer = int(attn_per_layer * arch.mean_qkv_dim / (q_dim + 2 * kv_dim))

    mlp_matrices = 3 if arch.gated_mlp else 2
    experts_per_layer = 0
    if arch.is_moe:
        assert arch.n_experts is not None
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        n_ffn = arch.n_experts + arch.n_shared_experts
        experts_per_layer = n_ffn * mlp_matrices * h * expert_inter
        mlp_per_layer = h * arch.n_experts  # router
    else:
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
    return ParamCounts(
        embedding=embedding,
        lm_head=lm_head,
        attention=arch.n_layers * attn_per_layer,
        mlp=arch.n_layers * mlp_per_layer,
        norms=norms,
        multimodal_towers=arch.multimodal_tower_params,
        moe_experts=arch.n_layers * experts_per_layer,
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
    # On a MoE, ``all-linear`` degenerates to attention-only: PEFT resolves the
    # target list by walking ``nn.Linear`` modules, and the experts are not
    # any. See :attr:`ParamCounts.moe_experts`.
    if scope == "all-linear" and not arch.is_moe:
        mlp_matrices = 3 if arch.gated_mlp else 2
        per_layer += mlp_matrices * (h + arch.intermediate_size)
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
                + counts.moe_experts
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
        + counts.moe_experts * base_bytes
    )


def kv_cache_bytes_per_token(
    arch: ModelArch, kv_dtype: KVCacheDtype, weight_dtype: WeightDtype
) -> float:
    """K + V bytes per cached token position across all layers.

    Layers that share an earlier layer's KV store nothing of their own, so
    only the remainder counts -- 15 of Gemma 4 E2B's 35.
    """
    kv_bytes = (
        DTYPE_BYTES[weight_dtype] if kv_dtype == "auto" else DTYPE_BYTES[kv_dtype]
    )
    storing_layers = arch.n_layers - min(arch.n_kv_shared_layers, arch.n_layers)
    return 2 * storing_layers * arch.n_kv_heads * arch.head_dim * kv_bytes


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

    ``eager`` always does. SDPA does too whenever the model passes an explicit
    mask, which windowed models do — the flash kernel cannot take one, so the
    dispatch falls back to the math backend.

    Measured on Gemma 4 E2B (28 of 35 layers windowed, 8 heads) at seq 4096,
    micro-batch 8: the allocator snapshot holds **2560 MiB** under
    ``sdpa_attention_forward``, directly beneath a
    ``create_sliding_window_causal_mask`` frame, against a quadratic term of
    ``8 x 8 x 4096^2 x 2 = 2048 MiB``.

    An earlier A/B concluded the opposite (sdpa 14.78 GiB vs flex_attention
    15.03) and this function returned ``eager``-only on the strength of it.
    That comparison was run before completion length was pinned, so its
    sequences were far shorter than the 4096 budget and the quadratic term it
    was looking for had barely formed. Pinning is what made the term visible;
    the earlier reading was measuring a workload it did not intend to.
    """
    if attn_implementation == "eager":
        return True
    # flex_attention and flash_attention_2 tile the score matrix and never
    # build it; SDPA only avoids it when no explicit mask forces the math path.
    return attn_implementation == "sdpa" and arch.sliding_window is not None


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
    """fp32 copies PEFT makes of every wrapped linear's input.

    ``get_peft_model`` defaults to ``autocast_adapter_dtype=True``, so the
    adapters are fp32 while the base is bf16. Each wrapped linear then runs
    ``x.to(fp32)`` on its input (``BaseTunerLayer._cast_input_dtype``),
    allocating a fresh tensor twice the size of the bf16 original — and these
    stay live for the backward pass.

    It is not a rounding error. Measured on Qwen2.5-0.5B at seq 4096 /
    micro-batch 8 / rank 64, A/B-ing the cast off via the same switch PEFT's
    own context manager flips: trainer peak 6826 -> 5530 MiB, a **19%**
    saving. That is memory spent for nothing, since the adapters could
    simply be bf16.

    Under gradient checkpointing only the block being recomputed holds its
    casts, so this is one layer's worth; without it, every layer's survive.
    The one-layer form predicts 1280 MiB against 1296 measured, 1.2% out.
    """
    h = arch.hidden_size
    # Sum of in_features over the linears PEFT wraps: q/k/v read the residual
    # stream, o reads the concatenated head output, and the MLP's gate/up read
    # the stream while down reads the intermediate. ``all-linear`` degenerates
    # to attention-only on a MoE, whose experts are not nn.Linear -- see
    # ParamCounts.moe_experts.
    width = 3 * h + arch.mean_qkv_dim
    if scope == "all-linear" and not arch.is_moe:
        inter = int(arch.intermediate_size * arch.mlp_width_factor)
        width += (2 * h + inter) if arch.gated_mlp else (h + inter)
    layers = 1 if gradient_checkpointing else arch.n_layers
    return int(rows * seq_len * width * layers * DTYPE_BYTES["fp32"])


def moe_dispatch_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float, backward: bool = False
) -> int:
    """Gather/scatter buffers the fused-expert forward builds per layer.

    ``OlmoeExperts.forward`` (and its siblings) routes by *materialising*
    copies, not by indexing in place::

        final_hidden_states = torch.zeros_like(hidden_states)   # accumulator
        expert_mask = F.one_hot(top_k_index, num_classes=...)   # int64
        current_state = hidden_states[token_idx]                # gather

    The gather copies each token once per expert it was routed to, so the
    total is ``tokens x top_k x hidden`` — eight times the residual stream on
    OLMoE, and nothing to do with the expert *weights* the rest of the model
    already accounts for.

    Measured on OLMoE-1B-7B at rows=4, seq=2048: gather 256 MiB, accumulator
    34, mask 34, against a 334 MiB unexplained activation gap.

    The one-hot mask is built under ``no_grad`` and is int64 regardless of
    the model dtype, which makes it scale with the *total* expert count while
    everything else here scales with the active count.
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


def block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    flash_attention: bool = True,
    backward: bool = False,
) -> int:
    """Peak transient activations of one transformer block.

    Live tensors at the MLP peak: residual stream copies, qkv projections,
    attention output, and the gated-MLP intermediates. Without FlashAttention
    the SDPA math path additionally materialises the ``heads x S x S`` score
    matrix.

    ``backward=True`` is the checkpointed-recompute case, where the block is
    replayed and then differentiated. Each gated-MLP intermediate then has a
    gradient live alongside it, while the residual-stream and qkv tensors do
    not — they are consumed in place or shared with the checkpoint boundary.
    So the MLP term doubles and the rest does not.

    That is a derivation, not a fitted constant, and the two agree. Sweeping
    a *flat* whole-block multiplier over all 198 stored training points
    minimised at 1.75; the MLP share of a Qwen2.5 block is 75.0-75.5%, which
    implies 1.750-1.755 by this rule. Qwen is five of the ten fixtures, so
    the flat optimum was fitting a Qwen-weighted average of what is really a
    per-model quantity — Gemma 4 E2B and OLMoE imply 1.61 and 1.63.
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
    qkv_dim = arch.mean_qkv_dim
    mlp = 3 * int(inter * arch.mlp_width_factor) * active_ffn
    ple = arch.n_layers * arch.per_layer_input_dim
    if backward:
        mlp *= 2
        ple *= 2
    per_token = 4 * h + qkv_dim + mlp + ple
    peak = rows * seq_len * per_token * act_bytes
    peak += moe_dispatch_bytes(arch, rows, seq_len, act_bytes, backward)
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
