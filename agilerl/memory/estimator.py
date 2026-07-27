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
    calibrated: bool = False
    warnings: tuple[str, ...] = ()

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


def estimate_training(
    model: ModelSpec,
    device: DeviceSpec,
    knobs: TrainingKnobs,
    trainer_variant: str = "base",
    colocated: bool = False,
    profile: ModelProfile | None = None,
    colocated_engine_reservation_bytes: int = 0,
) -> PhaseBreakdown:
    """Peak training-phase memory breakdown on the training device.

    ``colocated_engine_reservation_bytes`` is what the *sleeping* vLLM engine
    leaves resident on the device while the trainer runs. Measured on vLLM
    0.23, sleep level 1 releases essentially everything — the engine drops
    from its full ``gpu_memory_utilization`` footprint to a few hundred MiB of
    context and engine structures — so this is a small constant, not a
    utilization-scaled reservation. It is a parameter rather than a fitted
    term so a profile is not pinned to one utilization.
    """
    arch = model.arch
    counts = formulas.param_counts(arch)
    act_bytes = DTYPE_BYTES[knobs.weight_dtype]
    s = knobs.max_model_len
    warnings: list[str] = []

    variant = model.variant(trainer_variant)
    kbit = knobs.quantization != "none"
    base = formulas.weight_bytes(
        counts, knobs.weight_dtype, variant, kbit_prepared=kbit
    )

    lora_params = formulas.lora_param_count(
        arch, knobs.lora_rank, knobs.lora_target_scope
    )
    adapter_bytes_per_param = 4.0 if kbit else act_bytes
    value_head_params = arch.hidden_size if knobs.algorithm == "ppo" else 0
    adapters = (
        lora_params * knobs.n_resident_adapters + value_head_params
    ) * adapter_bytes_per_param

    trained_params = lora_params * knobs.n_trained_adapters + value_head_params
    grads = trained_params * formulas.grad_bytes_per_trainable_param(
        knobs.weight_dtype, knobs.distributed
    )
    opt_state = trained_params * formulas.optimizer_bytes_per_trainable_param(
        knobs.weight_dtype, knobs.distributed
    )

    # Gradient pass: checkpoint boundaries + one block's recompute peak +
    # the (rows, S, H) hidden state the fused logprob function saves.
    grad_rows = knobs.grad_rows
    if knobs.gradient_checkpointing:
        saved = formulas.activation_hidden_bytes(arch, grad_rows, s, act_bytes)
        recompute = formulas.block_recompute_bytes(
            arch, grad_rows, s, act_bytes, device.has_flash_attention
        )
    else:
        # Without checkpointing every block's intermediates are saved.
        recompute = formulas.block_recompute_bytes(
            arch, grad_rows, s, act_bytes, device.has_flash_attention
        )
        saved = recompute * arch.n_layers
        warnings.append(
            "gradient_checkpointing=False saves every block's activations; "
            "expect roughly n_layers x the checkpointed footprint."
        )
    if knobs.activation_offload:
        saved = 0  # backward-saved tensors live in pinned host RAM
    loss_hidden = grad_rows * s * arch.hidden_size * act_bytes
    grad_pass = saved + recompute + loss_hidden

    # No-grad logprob pass: actor + reference (+ value) rows fused into one
    # forward — a wider batch, but nothing saved for backward. Micro-batched
    # by the same per-GPU row cap as the gradient pass.
    nograd_rows = knobs.grad_rows * knobs.n_adapter_rows
    nograd_pass = formulas.nograd_forward_bytes(
        arch, nograd_rows, s, act_bytes, device.has_flash_attention
    )

    activations = max(grad_pass, nograd_pass)

    # The fused chunked path holds at most two vocab-width tiles (logits and
    # their gradient) regardless of batch or sequence length.
    logits = 2 * formulas.logit_workspace_bytes(arch.vocab_size, knobs.chunk_rows)

    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages — (rows, S) fp32-ish, megabytes at most.
    rollout = 6 * knobs.batch_size * knobs.group_size * s * 4

    correction = 0.0
    calibrated = False
    if profile is not None:
        basis = training_basis(model, knobs)
        correction = profile.training.fit.correction_bytes(basis)
        calibrated = profile.training.n_points > 0

    engine_residual = float(colocated_engine_reservation_bytes) if colocated else 0.0

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
            grads + opt_state,
            detail={"gradients": grads, "adamw_state": opt_state},
            note="LoRA-only training: scales with adapter params, not the base.",
        ),
        _component(
            "activations",
            "Activations",
            activations,
            detail={
                "grad_pass": grad_pass,
                "checkpoint_boundaries": saved,
                "block_recompute": recompute,
                "loss_hidden_state": loss_hidden,
                "nograd_logprob_pass": nograd_pass,
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
            formulas.CUDA_CONTEXT_BYTES + rollout + correction,
            detail={
                "cuda_context": formulas.CUDA_CONTEXT_BYTES,
                "rollout_tensors": rollout,
                "calibration_correction": correction,
            },
        ),
    )

    if not calibrated:
        warnings.append(
            "Uncalibrated estimate: no profiled constants for this "
            "(model, device); expect a wider error band."
        )

    return PhaseBreakdown(
        phase="training",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        calibrated=calibrated,
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
    counts = formulas.param_counts(arch)
    act_bytes = DTYPE_BYTES[knobs.weight_dtype]
    warnings: list[str] = []

    budget = int(knobs.gpu_memory_utilization * device.total_bytes)
    variant = model.variant(knobs.weight_variant)
    weights = formulas.weight_bytes(counts, knobs.weight_dtype, variant)

    batched_tokens = formulas.resolve_max_num_batched_tokens(
        knobs.max_num_seqs, knobs.max_model_len, knobs.max_num_batched_tokens
    )
    # Prefill transients for one scheduler step plus sampler buffers
    # (max_num_seqs x vocab logits and probs).
    prefill = formulas.block_recompute_bytes(
        arch, 1, batched_tokens, act_bytes, device.has_flash_attention
    )
    sampler = 2 * knobs.max_num_seqs * arch.vocab_size * 4
    activation_peak = prefill + sampler

    graphs = 0 if knobs.enforce_eager else formulas.CUDA_GRAPH_POOL_BYTES
    lora_slots = (
        knobs.max_loras
        * formulas.lora_param_count(arch, knobs.max_lora_rank)
        * act_bytes
    )

    if knobs.kv_cache_dtype == "fp8" and not device.supports_fp8:
        warnings.append(
            "fp8 KV cache requires compute capability >= 8.9; this device "
            "does not support it."
        )

    non_kv = weights + activation_peak + graphs + lora_slots
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

    kv_demand = formulas.kv_cache_demand_bytes(
        arch,
        knobs.kv_cache_dtype,
        knobs.weight_dtype,
        knobs.concurrency,
        knobs.max_model_len,
    )
    if kv_pool and kv_demand > kv_pool:
        warnings.append(
            f"Worst-case KV demand ({kv_demand / GiB:.1f} GiB for "
            f"{knobs.concurrency} sequences at {knobs.max_model_len} tokens) "
            f"exceeds the KV pool ({kv_pool / GiB:.1f} GiB): vLLM will "
            "preempt and recompute — a throughput cliff, not an OOM."
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
            * formulas.optimizer_bytes_per_trainable_param(
                train_knobs.weight_dtype, train_knobs.distributed
            )
        )

    correction = 0.0
    calibrated = False
    if profile is not None:
        basis = generation_basis(model, knobs)
        correction = profile.generation.fit.correction_bytes(basis)
        calibrated = profile.generation.n_points > 0

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
            activation_peak,
            detail={
                "prefill_transients": prefill,
                "sampler_buffers": sampler,
            },
            note=f"One scheduler step of {batched_tokens} batched tokens.",
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
            formulas.CUDA_CONTEXT_BYTES + correction,
            detail={
                "cuda_context": formulas.CUDA_CONTEXT_BYTES,
                "calibration_correction": correction,
            },
        ),
    )

    if not calibrated:
        warnings.append(
            "Uncalibrated estimate: no profiled constants for this "
            "(model, device); expect a wider error band."
        )

    return PhaseBreakdown(
        phase="generation",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        calibrated=calibrated,
        warnings=tuple(warnings),
    )


def estimate_run(config: RunConfig, profile: ModelProfile | None = None) -> RunEstimate:
    """Estimate both phases for a run configuration."""
    model = config.model
    if profile is not None:
        model = profile.apply_realised_weights(model)
    engine_reservation = 0
    if config.colocated:
        measured = profile.sleeping_engine_residual_bytes if profile else None
        engine_reservation = (
            measured
            if measured is not None
            else formulas.SLEEPING_ENGINE_RESIDUAL_BYTES
        )
    training = estimate_training(
        model,
        config.train_device,
        config.training,
        trainer_variant=config.trainer_weight_variant,
        colocated=config.colocated,
        profile=profile,
        colocated_engine_reservation_bytes=engine_reservation,
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
