# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Generation-phase peak-memory terms for the vLLM engine bar."""

from __future__ import annotations

import agilerl.arena.memory.formulas as formulas
import agilerl.arena.memory.vision as vision
from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GenerationSettings,
    GiB,
    ModelArch,
    ModelSpec,
)


def engine_terms(
    model: ModelSpec, settings: GenerationSettings
) -> tuple[dict[str, int], int, int]:
    """Engine-budget terms, scheduler-step token cap, and sampler buffer bytes.

    :param model: Engine-side model spec.
    :param settings: Generation settings (dtype, batching, LoRA slots).
    :return: Budget terms, max batched tokens, and sampler buffer bytes.
    """
    arch = model.arch
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    counts = formulas.param_counts(arch, model.n_params)
    variant = model.variant(settings.weight_variant)
    batched_tokens = formulas.resolve_max_num_batched_tokens(
        settings.max_num_seqs, settings.max_model_len, settings.max_num_batched_tokens
    )
    tp = settings.tensor_parallel_size
    # Sampler buffers: per-sequence fp32 logits and probs over the vocab.
    # Tensor parallel shards the lm_head, so each rank holds 1/tp of the vocab.
    sampler = 2 * settings.max_num_seqs * arch.vocab_size * 4 // tp
    weight_elems = formulas.weight_bytes(counts, settings.weight_dtype, variant)
    weight_byte = (
        1.0
        if settings.weight_quantization == "fp8"
        else DTYPE_BYTES[settings.weight_dtype]
    )
    unsharded_weights = weight_elems / DTYPE_BYTES[settings.weight_dtype] * weight_byte
    weights = int(unsharded_weights / tp)
    decoder_startup = (
        formulas.block_recompute_bytes(
            arch,
            1,
            batched_tokens,
            act_bytes,
            engine_forward=True,
        )
        + sampler
    )
    # A compiling engine's profiling pass holds a copy of the per-layer
    # embedding table; an eager engine does not. vLLM shards the table over TP.
    compile_copy = (
        0
        if settings.enforce_eager
        else int(
            max(counts.embedding - formulas.token_embedding_params(arch), 0)
            * act_bytes
            / tp
        )
    )
    terms = {
        "weights": weights,
        "startup_peak": _with_encoder_profile(
            model, settings, batched_tokens, decoder_startup
        )
        + compile_copy,
        "cuda_graphs": 0 if settings.enforce_eager else formulas.CUDA_GRAPH_POOL_BYTES,
        "lora_slots": int(
            settings.max_loras
            * formulas.lora_param_count(arch, settings.max_lora_rank)
            * act_bytes
            / tp
        )
        + settings.max_loras
        * formulas.engine_expert_lora_param_count(
            arch,
            settings.max_lora_rank,
            tp,
            packed_3d=arch.vllm_3d_moe_lora or settings.lora_packed_target_matrices > 0,
        )
        * act_bytes,
        "engine_runtime": int(formulas.ENGINE_RUNTIME_FRACTION * unsharded_weights),
        "kv_demand": _kv_demand_per_rank(arch, settings),
        "mamba_state": formulas.mamba_state_bytes(
            arch,
            settings.max_num_seqs,
            state_dtype=settings.weight_dtype,
            tensor_parallel=tp,
        ),
    }
    return terms, batched_tokens, sampler


def _with_encoder_profile(
    model: ModelSpec,
    settings: GenerationSettings,
    batched_tokens: int,
    decoder_bytes: int,
) -> int:
    """Activation peak of a decoder pass that follows vLLM's maximum-image encode.

    :param batched_tokens: The engine's resolved scheduler-step token cap.
    :param decoder_bytes: Decoder activation and sampler bytes of the pass.
    """
    if model.variant(settings.weight_variant).stripped_multimodal:
        return decoder_bytes
    items = vision.encoder_profile_items(
        model.arch,
        batched_tokens,
        settings.max_num_seqs,
        settings.max_model_len,
        settings.max_images_per_prompt,
    )
    encoder, outputs = vision.encoder_profile_bytes(
        model.arch,
        DTYPE_BYTES[settings.weight_dtype],
        settings.tensor_parallel_size,
        items,
    )
    return max(decoder_bytes + outputs, encoder)


def _kv_demand_per_rank(arch: ModelArch, settings: GenerationSettings) -> int:
    """KV bytes on one engine rank, replicating heads when TP exceeds them."""
    if formulas.shares_hybrid_pool(arch):
        return formulas.hybrid_kv_demand_bytes(
            arch,
            settings.weight_dtype,
            settings.concurrency,
            settings.max_model_len,
            settings.tensor_parallel_size,
        )
    batched_tokens = formulas.resolve_max_num_batched_tokens(
        settings.max_num_seqs, settings.max_model_len, settings.max_num_batched_tokens
    )
    demand = formulas.kv_cache_demand_bytes(
        arch,
        settings.weight_dtype,
        settings.concurrency,
        settings.max_model_len,
        formulas.VLLM_ASYNC_CONCURRENT_BATCHES * batched_tokens,
    )
    if arch.n_kv_heads == 0:
        return demand
    return (
        demand
        * formulas.kv_heads_per_rank(arch.n_kv_heads, settings.tensor_parallel_size)
        // arch.n_kv_heads
    )


def generation_warnings(
    model: ModelSpec,
    settings: GenerationSettings,
    *,
    kv_pool: int,
    kv_demand: int,
    startup_pool: int,
    full_sequence: int,
) -> list[str]:
    """User-facing generation-phase warnings."""
    arch = model.arch
    warnings: list[str] = []
    gap = formulas.geometry_gap_warning(formulas.param_counts(arch, model.n_params))
    if gap:
        warnings.append(gap)
    if arch.is_moe:
        warnings.append(
            "MoE serving peaks can exceed this bar by a few percent: vLLM's "
            "fused-MoE kernel holds chunked intermediate caches beyond this bar."
        )
    if arch.multimodal_tower_params or arch.per_layer_input_dim:
        warnings.append(
            "Multimodal engine residency is independent of "
            "gpu_memory_utilization; construction can peak above this bar."
        )
    if arch.is_hybrid_ssm:
        block_size = formulas.aligned_kv_block_size(
            arch, settings.weight_dtype, settings.tensor_parallel_size
        )
        warnings.append(
            f"Hybrid state-space model: {arch.n_mamba_layers} recurrent layers "
            f"and {arch.attention_layers} attention. Only the attention layers "
            "hold a KV cache; the recurrent state is constant in context length."
        )
        warnings.append(
            f"vLLM raises the attention block size to {block_size} tokens "
            "so an attention page holds a Mamba page, then pads the Mamba "
            "page to match. Every sequence's KV therefore rounds up to that "
            "page, which dominates the cache for short sequences."
        )
    if startup_pool <= 0:
        warnings.append(
            "gpu_memory_utilization budget is consumed by weights and engine "
            "overhead before any KV cache: vLLM will fail at init. Raise "
            "gpu_memory_utilization or shrink the model/variant."
        )
    elif startup_pool < full_sequence:
        warnings.append(
            f"KV pool ({startup_pool / GiB:.1f} GiB) cannot hold one "
            f"{settings.max_model_len}-token sequence ({full_sequence / GiB:.1f} GiB): "
            "vLLM will fail at init. Raise gpu_memory_utilization or lower "
            "max_model_len."
        )
    elif kv_demand > kv_pool:
        warnings.append(
            f"Worst-case KV demand ({kv_demand / GiB:.1f} GiB for "
            f"{settings.concurrency} sequences at {settings.max_model_len} tokens) "
            f"exceeds the KV pool ({kv_pool / GiB:.1f} GiB): vLLM will "
            "preempt and recompute, a throughput cliff."
        )
    return warnings


def estimate_generation(
    model: ModelSpec,
    device: DeviceSpec,
    settings: GenerationSettings,
    orchestrated: bool = False,
) -> PhaseBreakdown:
    """Peak generation-phase memory on the inference device.

    vLLM self-limits to ``gpu_memory_utilization * total_bytes``. The KV pool
    is what remains after weights, prefill activations, CUDA graphs, and LoRA
    slots. The device-level bar adds CUDA context.
    """
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    terms, batched_tokens, sampler = engine_terms(model, settings)
    kv_pool, startup_pool, full_sequence = _generation_pools(
        model, settings, terms, device
    )
    warnings = generation_warnings(
        model,
        settings,
        kv_pool=kv_pool,
        kv_demand=terms["kv_demand"],
        startup_pool=startup_pool,
        full_sequence=full_sequence,
    )
    runtime_tokens = min(settings.concurrency * settings.prompt_len, batched_tokens)
    job_overhead = formulas.JOB_OVERHEAD_BYTES if orchestrated else 0
    runtime_activation = _with_encoder_profile(
        model,
        settings,
        batched_tokens,
        formulas.block_recompute_bytes(
            model.arch,
            1,
            runtime_tokens,
            act_bytes,
            engine_forward=True,
        )
        + sampler,
    )
    return PhaseBreakdown(
        phase="generation",
        components=_kv_bar_components(
            settings,
            terms,
            kv_pool,
            startup_pool,
            full_sequence,
            runtime_activation,
            runtime_tokens,
            batched_tokens,
            sampler,
        )
        + _residency_bar_components(model, device, terms, job_overhead),
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.generation_usable_bytes,
        warnings=tuple(warnings),
    )


def _generation_pools(
    model: ModelSpec,
    settings: GenerationSettings,
    terms: dict[str, int],
    device: DeviceSpec,
) -> tuple[int, int, int]:
    """KV pool, start-up pool, and one-sequence demand of the engine bar."""
    budget = int(settings.gpu_memory_utilization * device.total_bytes)
    non_kv = (
        terms["weights"]
        + terms["startup_peak"]
        + terms["cuda_graphs"]
        + terms["lora_slots"]
        + terms["mamba_state"]
        + terms["engine_runtime"]
    )
    pinned = settings.kv_cache_memory_bytes
    kv_pool = pinned if pinned is not None else max(budget - non_kv, 0)
    # vLLM holds Mamba state in the KV pool and checks room for one request at start-up.
    startup_pool = (
        pinned
        if pinned is not None
        else max(budget - (non_kv - terms["mamba_state"]), 0)
    )
    full_sequence = terms[
        "kv_demand"
    ] // settings.concurrency + formulas.mamba_state_bytes(
        model.arch,
        1,
        state_dtype=settings.weight_dtype,
        tensor_parallel=settings.tensor_parallel_size,
    )
    return kv_pool, startup_pool, full_sequence


def _kv_bar_components(
    settings: GenerationSettings,
    terms: dict[str, int],
    kv_pool: int,
    startup_pool: int,
    full_sequence: int,
    runtime_activation: int,
    runtime_tokens: int,
    batched_tokens: int,
    sampler: int,
) -> tuple[MemoryComponent, ...]:
    """Weights, KV pool, and prefill components of the generation bar."""
    return (
        MemoryComponent(
            key="weights",
            label="Model weights (engine copy)",
            n_bytes=max(int(terms["weights"]), 0),
        ),
        MemoryComponent(
            key="kv_cache",
            label="KV cache pool",
            n_bytes=max(int(kv_pool), 0),
            detail={
                "pool": int(kv_pool),
                "worst_case_demand": int(terms["kv_demand"]),
                "startup_pool": int(startup_pool),
                "full_sequence_bytes": int(full_sequence),
            },
            note=(
                "Pinned via kv_cache_memory_bytes."
                if settings.kv_cache_memory_bytes is not None
                else "Sized by vLLM from the gpu_memory_utilization budget."
            ),
        ),
        MemoryComponent(
            key="activation_peak",
            label="Prefill & sampling buffers",
            # The caching allocator keeps the start-up profile's peak reserved.
            n_bytes=max(int(runtime_activation), int(terms["startup_peak"]), 0),
            detail={
                "runtime_tokens": int(runtime_tokens),
                "sampler_buffers": int(sampler),
                "startup_peak": int(terms["startup_peak"]),
            },
            note=(
                f"Serving peak over {runtime_tokens} tokens. The start-up "
                f"scheduler step of {batched_tokens} tokens is larger and "
                "sizes the KV pool."
            ),
        ),
    )


def _residency_bar_components(
    model: ModelSpec,
    device: DeviceSpec,
    terms: dict[str, int],
    job_overhead: int,
) -> tuple[MemoryComponent, ...]:
    """CUDA-graph, Mamba, LoRA, and overhead components of the generation bar."""
    arch = model.arch
    return (
        MemoryComponent(
            key="cuda_graphs",
            label="CUDA graphs",
            n_bytes=max(int(terms["cuda_graphs"]), 0),
            note="Set enforce_eager=True to skip capture (throughput cost).",
        ),
        MemoryComponent(
            key="mamba_state",
            label="Mamba recurrent state",
            n_bytes=max(int(terms["mamba_state"]), 0),
            detail={
                "recurrent_layers": arch.n_mamba_layers,
                "state_blocks_per_seq": formulas.mamba_state_blocks_per_seq(arch),
            },
            note=(
                f"{arch.n_mamba_layers} recurrent layers against "
                f"{arch.attention_layers} attention. Constant in context "
                "length."
                if arch.is_hybrid_ssm
                else None
            ),
        ),
        MemoryComponent(
            key="lora_slots",
            label="LoRA adapter slots",
            n_bytes=max(int(terms["lora_slots"]), 0),
        ),
        MemoryComponent(
            key="overhead",
            label="Overhead (CUDA context, runtime)",
            n_bytes=max(
                int(device.context_bytes + job_overhead + terms["engine_runtime"]), 0
            ),
            detail={
                "cuda_context": int(device.context_bytes),
                "job_overhead": int(job_overhead),
                "engine_runtime": int(terms["engine_runtime"]),
            },
        ),
    )


def generation_can_serve(breakdown: PhaseBreakdown, kv_margin_bytes: int = 0) -> bool:
    """Whether this generation bar can serve the advertised context.

    :param breakdown: Generation-phase breakdown.
    :param kv_margin_bytes: Pool bytes demand must leave free.
    :return: True when vLLM starts and its KV pool covers worst-case demand.
    """
    # ``fits`` only means vLLM starts; demand above the pool at
    # max_model_len means preemption.
    if breakdown.phase != "generation" or not breakdown.fits:
        return False
    kv = next((c for c in breakdown.components if c.key == "kv_cache"), None)
    if kv is None:
        return False
    pool = kv.detail["pool"]
    demand = kv.detail["worst_case_demand"]
    return pool > 0 and demand + kv_margin_bytes <= pool
