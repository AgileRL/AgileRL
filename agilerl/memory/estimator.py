# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form peak-memory estimation for LLM RL training and generation.

Produces one stacked-bar breakdown per phase. Training and generation are
always uncoupled in the framework (colocated runs alternate via vLLM
sleep/wake plus trainer CPU-offload), so the two phases are independent
peaks — never summed — even on a single device.

Component keys are stable identifiers consumed by the Arena widget and the
CLI preflight; change them only with a schema-version bump.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agilerl.memory import formulas
from agilerl.memory.calibration import (
    ModelProfile,
    generation_basis,
    training_basis,
)
from agilerl.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GenerationKnobs,
    GiB,
    ModelSpec,
    RunConfig,
    TrainingKnobs,
)

PhaseName = Literal["training", "generation"]
CalibrationSource = Literal["same_device", "other_device", "none"]


class MemoryComponent(BaseModel):
    """One segment of a phase's stacked bar."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    bytes_: int = Field(alias="bytes", serialization_alias="bytes")
    #: Sub-lines for the tooltip (e.g. optimizer-state split), in bytes.
    detail: dict[str, int] = Field(default_factory=dict)
    note: str | None = None


class PhaseBreakdown(BaseModel):
    """Predicted peak for one phase on one device."""

    model_config = ConfigDict(frozen=True)

    phase: PhaseName
    components: tuple[MemoryComponent, ...]
    device_total_bytes: int
    device_usable_bytes: int
    #: Where the fitted constants came from. ``same_device`` is the accuracy
    #: claim; ``other_device`` means they were measured on different hardware
    #: and carry a wider, less predictable band; ``none`` is the bare
    #: analytic core.
    calibration_source: CalibrationSource = "none"
    warnings: tuple[str, ...] = ()

    @property
    def calibrated(self) -> bool:
        return self.calibration_source == "same_device"

    @property
    def total_bytes(self) -> int:
        return sum(c.bytes_ for c in self.components)

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.device_usable_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.device_usable_bytes - self.total_bytes


class RunEstimate(BaseModel):
    """The two independent bars for a sizing question."""

    model_config = ConfigDict(frozen=True)

    training: PhaseBreakdown
    generation: PhaseBreakdown

    @property
    def fits(self) -> bool:
        return self.training.fits and self.generation.fits


def _component(
    key: str,
    label: str,
    n_bytes: float,
    detail: dict[str, float] | None = None,
    note: str | None = None,
) -> MemoryComponent:
    return MemoryComponent(
        key=key,
        label=label,
        bytes=max(int(n_bytes), 0),
        detail={k: int(v) for k, v in (detail or {}).items()},
        note=note,
    )


def _apply_calibration(
    profile: ModelProfile | None,
    phase: PhaseName,
    basis: dict[str, float],
    device: DeviceSpec,
    warnings: list[str],
) -> tuple[float, CalibrationSource]:
    """Fitted correction and its provenance, appending the matching warning."""
    if profile is None or getattr(profile, phase).n_points == 0:
        warnings.append(
            "Uncalibrated estimate: no profiled constants for this "
            "(model, device); expect a wider error band."
        )
        return 0.0, "none"
    correction = getattr(profile, phase).fit.correction_bytes(basis)
    if profile.measured_on(device):
        return correction, "same_device"
    warnings.append(
        f"Constants were measured on "
        f"{profile.device.name if profile.device else 'another device'}, "
        f"not {device.name or 'this device'}: expect a wider band than a "
        "same-device profile."
    )
    return correction, "other_device"


def _engine_reservation_bytes(colocated: bool, profile: ModelProfile | None) -> float:
    """What the sleeping engine keeps resident during colocated training:
    the profile's measured residual when available, else the analytic
    constant.
    """
    if not colocated:
        return 0.0
    measured = profile.sleeping_engine_residual_bytes if profile else None
    return float(
        measured if measured is not None else formulas.ENGINE_PROCESS_OVERHEAD_BYTES
    )


def _engine_terms(
    model: ModelSpec, device: DeviceSpec, knobs: GenerationKnobs
) -> tuple[dict[str, int], int, int]:
    """Engine-budget terms shared by :func:`estimate_generation` and its
    inversion :func:`recommend_engine_budget`, plus the resolved scheduler-step
    token cap and the sampler buffer bytes.
    """
    arch = model.arch
    act_bytes = DTYPE_BYTES[knobs.weight_dtype]
    counts = formulas.param_counts(arch)
    variant = model.variant(knobs.weight_variant)
    batched_tokens = formulas.resolve_max_num_batched_tokens(
        knobs.max_num_seqs, knobs.max_model_len, knobs.max_num_batched_tokens
    )
    # Prefill transients for one scheduler step plus sampler buffers
    # (max_num_seqs x vocab logits and probs).
    sampler = 2 * knobs.max_num_seqs * arch.vocab_size * 4
    terms = {
        "weights": formulas.weight_bytes(counts, knobs.weight_dtype, variant),
        "startup_profiling_peak": formulas.block_recompute_bytes(
            arch, 1, batched_tokens, act_bytes, device.has_flash_attention
        )
        + sampler,
        "cuda_graphs": 0 if knobs.enforce_eager else formulas.CUDA_GRAPH_POOL_BYTES,
        "lora_slots": int(
            knobs.max_loras
            * formulas.lora_param_count(arch, knobs.max_lora_rank)
            * act_bytes
        ),
        "kv_demand": formulas.kv_cache_demand_bytes(
            arch,
            knobs.kv_cache_dtype,
            knobs.weight_dtype,
            knobs.concurrency,
            knobs.max_model_len,
        ),
    }
    return terms, batched_tokens, sampler


def estimate_training(
    model: ModelSpec,
    device: DeviceSpec,
    knobs: TrainingKnobs,
    trainer_variant: str = "base",
    colocated: bool = False,
    profile: ModelProfile | None = None,
) -> PhaseBreakdown:
    """Peak training-phase memory breakdown on the training device.

    When ``colocated``, the bar includes what the *sleeping* vLLM engine
    leaves resident while the trainer runs. Measured on vLLM 0.23, sleep
    level 1 releases essentially everything — the engine drops from its full
    ``gpu_memory_utilization`` footprint to a few hundred MiB of context and
    engine structures — so this is a small constant, not a
    utilization-scaled reservation.
    """
    arch = model.arch
    counts = formulas.param_counts(arch)
    act_bytes = DTYPE_BYTES[knobs.weight_dtype]
    s = knobs.max_model_len
    warnings: list[str] = []

    attn_impl = formulas.resolve_attn_implementation(
        knobs.attn_implementation, device.flash_attn_installed
    )
    # Whether attention builds an S x S matrix depends on the backend and the
    # model's masking, not on what the device is capable of.
    flash_like = (
        device.has_flash_attention
        and not formulas.materializes_attention_scores(attn_impl, arch)
    )
    if knobs.beta == 0.0 and knobs.algorithm != "sft":
        warnings.append(
            "Assumes the reference forward is skipped at beta=0. The fused "
            "no-grad pass currently builds that row unconditionally, so an "
            "unpatched run pays for one extra row of activations."
        )
    if not flash_like:
        warnings.append(
            "attn_implementation='eager' materialises a rows x heads x S x S "
            "score matrix, which dominates activation memory at long "
            "context. Use sdpa or flex_attention."
        )

    variant = model.variant(trainer_variant)
    kbit = knobs.quantization != "none"
    if kbit and arch.is_moe:
        warnings.append(
            f"Quantization barely helps this MoE: {counts.moe_experts / 1e9:.1f}B "
            f"of {counts.total / 1e9:.1f}B parameters are fused expert tensors, "
            "which bitsandbytes cannot reach, so they stay at the base dtype. "
            "Only the attention and router matrices shrink."
        )
    if kbit and not arch.is_moe:
        # bitsandbytes' MatMul4Bit.backward workspace is not yet a modelled
        # term (see fixtures/pending/README.md), so the estimator
        # under-predicts quantized training at long context — warn rather
        # than mislead.
        warnings.append(
            "Quantized training is under-predicted at long context. nf4 saves "
            "weight bytes but costs activation bytes, and the two cross over: "
            "measured on Qwen2.5-0.5B, nf4 was 100 MiB cheaper at 512 tokens "
            "per micro-batch and 2.2 GiB dearer at 32k. Treat this bar as a "
            "lower bound."
        )
    base = formulas.weight_bytes(
        counts, knobs.weight_dtype, variant, kbit_prepared=kbit
    )

    lora_params = formulas.lora_param_count(
        arch, knobs.lora_rank, knobs.lora_target_scope
    )
    adapter_bytes_per_param = formulas.ADAPTER_BYTES_PER_PARAM
    # PPOValueHead is Linear(hidden -> 1), held in modules_to_save so it is
    # trained alongside the adapters.
    value_head_params = arch.hidden_size + 1 if knobs.uses_critic else 0
    adapters = (
        lora_params * knobs.n_resident_adapters + value_head_params
    ) * adapter_bytes_per_param

    trained_params = lora_params * knobs.n_trained_adapters + value_head_params
    # Gradients are fp32, mirroring the fp32 adapter parameters they
    # accumulate into.
    grads = trained_params * formulas.ADAPTER_BYTES_PER_PARAM
    opt_state = trained_params * formulas.optimizer_bytes_per_trainable_param(
        knobs.distributed
    )

    # Gradient pass: checkpoint boundaries + one block's recompute peak +
    # the (rows, S, H) hidden state the fused logprob function saves.
    grad_rows = knobs.grad_rows
    recompute = formulas.block_recompute_bytes(
        arch, grad_rows, s, act_bytes, flash_like, backward=True
    )
    if knobs.gradient_checkpointing:
        saved = formulas.activation_hidden_bytes(arch, grad_rows, s, act_bytes)
    else:
        # Without checkpointing every block's intermediates are saved.
        saved = recompute * arch.n_layers
        warnings.append(
            "gradient_checkpointing=False saves every block's activations; "
            "expect roughly n_layers x the checkpointed footprint."
        )
    if knobs.activation_offload:
        saved = 0  # backward-saved tensors live in pinned host RAM
    loss_hidden = grad_rows * s * arch.hidden_size * act_bytes
    # PEFT casts every wrapped linear's input to the fp32 adapter dtype, and
    # those copies stay live for the backward pass. Worth 19% of the trainer
    # peak at one measured corner, and unmodelled until now.
    lora_casts = formulas.lora_input_cast_bytes(
        arch,
        grad_rows,
        s,
        knobs.lora_target_scope,
        knobs.gradient_checkpointing,
    )
    # The fused chunked path holds vocab-width tiles bounded by ``chunk_rows``,
    # never a (B, S, V) slab. Two are live together under autograd: the matmul
    # emits one in the activation dtype and the log-softmax reduction casts a
    # second to fp32, and the graph holds both for the recompute. The no-grad
    # logprob pass builds no graph, so the cast frees its input and only the
    # fp32 tile stands.
    #
    # Measured at Qwen2.5-0.5B / L4 / seq 512, mb 8, group 4, rank 64, where
    # chunk_rows auto-tunes to 441 against a 151936 vocab: the trace holds
    # tiles of exactly 255.6 MiB (441 x 151936 x 4) and 127.8 MiB
    # (441 x 151936 x 2), and logits_workspace peaks at 399 MiB against the
    # 383 MiB this predicts.
    logit_rows = formulas.resolve_chunk_rows(arch.vocab_size, knobs.chunk_rows)
    logit_tile = logit_rows * arch.vocab_size * 4
    logits = logit_rows * arch.vocab_size * (act_bytes + 4)

    # No-grad logprob pass: actor + reference (+ value) rows fused into one
    # forward — a wider batch, but nothing saved for backward. Micro-batched
    # by the same per-GPU row cap as the gradient pass.
    nograd_rows = knobs.grad_rows * knobs.n_adapter_rows
    nograd_pass = formulas.nograd_forward_bytes(
        arch, nograd_rows, s, act_bytes, flash_like
    )

    # A training step peaks at one of three distinct instants, and different
    # tensors are live at each. Summing the activation peak and the logit
    # workspace assumes they coexist; allocator snapshots say they do not —
    # the logit tiles measured 0 at the peak on all three architectures
    # profiled (Qwen2.5-0.5B, Gemma 4 E2B, OLMoE), against a ~512 MiB
    # prediction. By then the loss backward has freed them and the block
    # recompute is at its widest.
    # Gradients belong to the instant, not to every instant. The no-grad
    # logprob pass runs after ``zero_grad(set_to_none=True)`` has freed them
    # and before backward re-creates them, so nothing gradient-shaped is
    # resident there; the optimizer step is the mirror image, every transient
    # freed and only the persistent tensors left. Timeline of the allocator
    # trace at the corner above, trainer side: no-grad 1308 MiB, gradient
    # forward 1452, backward 1510, optimizer step 1756 -- the step binds, and
    # it cannot win a maximum that adds gradients to all four alike.
    backward_peak = grads + saved + recompute + loss_hidden + lora_casts
    loss_peak = grads + saved + loss_hidden + lora_casts + logits
    nograd_peak = nograd_pass + logit_tile
    optimizer_peak = grads
    peak = max(backward_peak, loss_peak, nograd_peak, optimizer_peak)

    # Report the split that belongs to whichever instant binds, so the bar
    # shows what is actually resident at the peak rather than a union.
    if peak == backward_peak:
        activations, logits, live_grads = backward_peak - grads, 0, grads
    elif peak == loss_peak:
        activations, logits, live_grads = loss_peak - logits - grads, logits, grads
    elif peak == nograd_peak:
        activations, logits, live_grads = nograd_pass, logit_tile, 0
    else:
        activations, logits, live_grads = 0, 0, grads

    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages — (rows, S) fp32-ish, megabytes at most.
    rollout = 6 * knobs.batch_size * knobs.group_size * s * 4

    correction, source = _apply_calibration(
        profile, "training", training_basis(model, knobs), device, warnings
    )
    engine_residual = _engine_reservation_bytes(colocated, profile)

    components = (
        _component(
            "base_weights",
            "Base weights (frozen)",
            base,
            note=(
                "Trainer copy; offloaded to CPU during rollout when colocated."
                if colocated
                else None
            ),
        ),
        _component(
            "adapters",
            "LoRA adapters",
            adapters,
            detail={
                "actor": lora_params * adapter_bytes_per_param,
                "reference": (
                    lora_params * adapter_bytes_per_param
                    if knobs.n_resident_adapters > 1
                    else 0
                ),
                "value_head": value_head_params * adapter_bytes_per_param,
            },
            note=(
                "The reference is a frozen adapter copy, not a second model. "
                "beta does not change this footprint."
            ),
        ),
        _component(
            "grads_optimizer",
            "Gradients + optimizer state",
            live_grads + opt_state,
            detail={"gradients": live_grads, "adamw_state": opt_state},
            note="LoRA-only training: scales with adapter params, not the base.",
        ),
        _component(
            "activations",
            "Activations",
            activations,
            detail={
                "backward_peak": backward_peak,
                "loss_peak": loss_peak,
                "nograd_peak": nograd_peak,
                "optimizer_peak": optimizer_peak,
                "checkpoint_boundaries": saved,
                "block_recompute": recompute,
                "loss_hidden_state": loss_hidden,
                "lora_fp32_input_casts": lora_casts,
            },
            note=(
                "Peak of the gradient micro-batch pass vs the fused no-grad "
                "logprob pass (actor+reference rows)."
            ),
        ),
        _component(
            "logits_workspace",
            "Logit workspace (chunked)",
            logits,
            note=(
                "The fused logprob path tiles the lm_head matmul; the full "
                "batch x seq x vocab slab is never materialised."
            ),
        ),
        _component(
            "vllm_residual",
            "Sleeping engine residual",
            engine_residual,
            note=(
                "vLLM sleep level 1 releases its weights and KV pool back to "
                "the device (measured: full footprint down to a few hundred "
                "MiB), so the trainer gets the GPU back. Only context and "
                "engine structures stay resident."
                if colocated
                else "Not colocated."
            ),
        ),
        _component(
            "overhead",
            "Overhead & calibration",
            device.context_bytes + rollout + correction,
            detail={
                "cuda_context": device.context_bytes,
                "rollout_tensors": rollout,
                "calibration_correction": correction,
            },
        ),
    )

    return PhaseBreakdown(
        phase="training",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        calibration_source=source,
        warnings=tuple(warnings),
    )


def estimate_generation(
    model: ModelSpec,
    device: DeviceSpec,
    knobs: GenerationKnobs,
    train_knobs: TrainingKnobs | None = None,
    colocated: bool = False,
    profile: ModelProfile | None = None,
) -> PhaseBreakdown:
    """Peak generation-phase memory breakdown on the inference device.

    vLLM self-limits to ``gpu_memory_utilization * total_bytes``; within that
    budget the KV pool is what remains after weights, prefill activations,
    CUDA graphs, and LoRA slots. The device-level bar adds what lives outside
    the engine budget: CUDA context and — when colocated — the trainer state
    that stays resident while the base is offloaded.
    """
    arch = model.arch
    act_bytes = DTYPE_BYTES[knobs.weight_dtype]
    warnings: list[str] = []

    budget = int(knobs.gpu_memory_utilization * device.total_bytes)
    terms, batched_tokens, sampler = _engine_terms(model, device, knobs)
    weights = terms["weights"]
    # What vLLM's start-up profiling run measures: a full scheduler step of
    # max_num_batched_tokens. This sizes the KV pool but is *transient* — it
    # is not resident while the engine serves.
    profiling_peak = terms["startup_profiling_peak"]
    graphs = terms["cuda_graphs"]
    lora_slots = terms["lora_slots"]
    kv_demand = terms["kv_demand"]

    if knobs.kv_cache_dtype == "fp8" and not device.supports_fp8:
        warnings.append(
            "fp8 KV cache requires compute capability >= 8.9; this device "
            "does not support it."
        )

    non_kv = weights + profiling_peak + graphs + lora_slots
    if knobs.kv_cache_memory_bytes is not None:
        kv_pool = knobs.kv_cache_memory_bytes
    else:
        kv_pool = budget - non_kv
    if kv_pool <= 0:
        warnings.append(
            "gpu_memory_utilization budget is consumed by weights and engine "
            "overhead before any KV cache: vLLM will fail at init. Raise "
            "gpu_memory_utilization or shrink the model/variant."
        )
        kv_pool = 0

    if kv_pool and kv_demand > kv_pool:
        warnings.append(
            f"Worst-case KV demand ({kv_demand / GiB:.1f} GiB for "
            f"{knobs.concurrency} sequences at {knobs.max_model_len} tokens) "
            f"exceeds the KV pool ({kv_pool / GiB:.1f} GiB): vLLM will "
            "preempt and recompute — a throughput cliff, not an OOM."
        )

    # Resident peak is not the whole budget. vLLM sizes the KV pool as
    # "budget minus the profiling peak", and that peak is transient — so what
    # actually stays on the device is weights + KV + graphs + the activations
    # of the real workload, which is smaller than the profiled step whenever
    # max_num_batched_tokens exceeds what the workload submits. Measured on an
    # L4: a 4096-context engine sits ~1.9 GiB *below* the same engine at 512
    # context, purely because its larger profiling step bought a smaller pool.
    runtime_tokens = min(knobs.concurrency * knobs.prompt_len, batched_tokens)
    runtime_activation = (
        formulas.block_recompute_bytes(
            arch, 1, runtime_tokens, act_bytes, device.has_flash_attention
        )
        + sampler
    )

    # Trainer state that stays on the GPU while the base is offloaded to CPU
    # for the rollout (optimizer state is created on-device and never moved).
    trainer_residual = 0.0
    if colocated and train_knobs is not None:
        lora_params = formulas.lora_param_count(
            model.arch, train_knobs.lora_rank, train_knobs.lora_target_scope
        )
        trainer_residual = (
            lora_params
            * train_knobs.n_trained_adapters
            * formulas.optimizer_bytes_per_trainable_param(train_knobs.distributed)
        )

    correction, source = _apply_calibration(
        profile, "generation", generation_basis(model, knobs), device, warnings
    )

    components = (
        _component(
            "weights",
            "Model weights (engine copy)",
            weights,
            note="vLLM owns its own base copy across the sleep/wake cycle.",
        ),
        _component(
            "kv_cache",
            "KV cache pool",
            kv_pool,
            detail={
                "pool": kv_pool,
                "worst_case_demand": kv_demand,
            },
            note=(
                "Pinned via kv_cache_memory_bytes."
                if knobs.kv_cache_memory_bytes is not None
                else "Sized by vLLM from the gpu_memory_utilization budget."
            ),
        ),
        _component(
            "activation_peak",
            "Prefill & sampling buffers",
            runtime_activation,
            detail={
                "runtime_tokens": runtime_tokens,
                "sampler_buffers": sampler,
                "startup_profiling_peak": profiling_peak,
            },
            note=(
                f"Serving peak over {runtime_tokens} tokens. vLLM's start-up "
                f"profiling step of {batched_tokens} tokens is larger and "
                "sizes the KV pool, but is not resident afterwards."
            ),
        ),
        _component(
            "cuda_graphs",
            "CUDA graphs",
            graphs,
            note="Set enforce_eager=True to skip capture (throughput cost).",
        ),
        _component("lora_slots", "LoRA adapter slots", lora_slots),
        _component(
            "trainer_residual",
            "Offloaded trainer residual",
            trainer_residual,
            note=(
                "Optimizer state stays on-device while the trainer base sits "
                "in host RAM during rollout."
                if colocated
                else "Not colocated."
            ),
        ),
        _component(
            "overhead",
            "Overhead & calibration",
            device.context_bytes + formulas.ENGINE_PROCESS_OVERHEAD_BYTES + correction,
            detail={
                "cuda_context": device.context_bytes,
                "engine_process_overhead": formulas.ENGINE_PROCESS_OVERHEAD_BYTES,
                "calibration_correction": correction,
            },
        ),
    )

    return PhaseBreakdown(
        phase="generation",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        calibration_source=source,
        warnings=tuple(warnings),
    )


def recommend_engine_budget(
    model: ModelSpec,
    device: DeviceSpec,
    knobs: GenerationKnobs,
) -> tuple[float, dict[str, int]]:
    """Smallest ``gpu_memory_utilization`` that serves the workload's KV demand.

    The generation model is invertible. vLLM sizes its KV pool as the budget
    minus what the engine needs for weights, the start-up profiling step,
    CUDA graphs and LoRA slots, so requiring the pool to cover worst-case
    demand fixes the budget:

        budget >= weights + profiling peak + graphs + LoRA + KV demand

    Below that the engine still starts but preempts and recomputes under
    load — a throughput cliff rather than an OOM — which is why guessing
    this number is worth avoiding. Returns the fraction and the terms it is
    built from; a fraction above 1.0 means the workload cannot be served on
    this device at this context and concurrency.
    """
    terms, _, _ = _engine_terms(model, device, knobs)
    return sum(terms.values()) / device.total_bytes, terms


def estimate_run(config: RunConfig, profile: ModelProfile | None = None) -> RunEstimate:
    """Estimate both phases for a run configuration."""
    model = config.model
    if profile is not None:
        model = profile.apply_realised_weights(model)
    training = estimate_training(
        model,
        config.train_device,
        config.training,
        trainer_variant=config.trainer_weight_variant,
        colocated=config.colocated,
        profile=profile,
    )
    generation = estimate_generation(
        model,
        config.generation_device,
        config.generation,
        train_knobs=config.training,
        colocated=config.colocated,
        profile=profile,
    )
    return RunEstimate(training=training, generation=generation)
