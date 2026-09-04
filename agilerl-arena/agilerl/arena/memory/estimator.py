# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form peak-memory estimation for LLM RL training and generation.

Produces one stacked-bar breakdown per phase. Training and generation are
always uncoupled in the framework (colocated runs alternate via vLLM
sleep/wake plus trainer CPU-offload), so the two phases are independent
peaks — never summed — even on a single device.

Component keys are stable identifiers consumed by the Arena widget and the
CLI; change them only with a schema-version bump.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GenerationKnobs,
    GiB,
    ModelArch,
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


def _engine_reservation_bytes(
    colocated: bool, arch: ModelArch, max_num_seqs: int
) -> float:
    """What the sleeping engine keeps resident during colocated training.

    ``max_num_seqs`` is the *engine's* slot count, which the training settings do
    not carry; ``group_size`` is what the framework configures it from and what
    every stored measurement was taken under.
    """
    if not colocated:
        return 0.0
    return float(formulas.engine_process_overhead_bytes(arch, max_num_seqs))


#: Share of a checkpoint the geometry may leave unattributed before it is worth
#: saying so. The six conventional checkpoints validated against sit at 0.006%;
#: Gemma 4's per-layer-embedding and tower blocks reach 3.7-5.0%.
GEOMETRY_GAP_WARN_FRACTION = 0.02

#: Longest context any stored measurement covers. The corpus's corner plan
#: runs 512 and 4096 with 1024/2048 held out, so every point above this is
#: the model extrapolating: each activation term is linear in context and
#: stays linear here, which is a claim no measurement has checked past 4096.
#: The one 32k sweep that ever existed was taken under a since-rewritten
#: fused-loss kernel and was purged with the rest of that vintage.
VALIDATED_MAX_CONTEXT = 4096


def _context_extrapolation_warning(max_model_len: int) -> str | None:
    """Say so when a run is sized beyond anything the corpus measures."""
    if max_model_len <= VALIDATED_MAX_CONTEXT:
        return None
    return (
        f"Context {max_model_len} is beyond the {VALIDATED_MAX_CONTEXT}-token "
        "ceiling of every stored measurement, so this bar is an "
        "extrapolation: the activation, KV and workspace terms are linear in "
        "context and are applied linearly here, unchecked. Treat the margin "
        "as indicative, and prefer headroom over a tight fit."
    )


def _geometry_gap_warning(counts: formulas.ParamCounts) -> str | None:
    """Flag a checkpoint whose parameters the geometry does not account for."""
    if counts.total <= 0:
        return None
    gap = counts.unattributed / counts.total
    if gap < GEOMETRY_GAP_WARN_FRACTION:
        return None
    return (
        f"{gap:.1%} of this checkpoint's parameters are not accounted for by "
        "the parsed geometry. Weight bytes are exact (taken from checkpoint "
        "metadata), but activation, KV and LoRA terms are derived from the "
        "geometry and are correspondingly low."
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
    counts = formulas.param_counts(arch, model.n_params)
    variant = model.variant(knobs.weight_variant)
    batched_tokens = formulas.resolve_max_num_batched_tokens(
        knobs.max_num_seqs, knobs.max_model_len, knobs.max_num_batched_tokens
    )
    # Prefill transients for one scheduler step plus sampler buffers
    # (max_num_seqs x vocab logits and probs).
    sampler = 2 * knobs.max_num_seqs * arch.vocab_size * 4
    terms = {
        "weights": formulas.weight_bytes(counts, knobs.weight_dtype, variant),
        # engine_forward: vLLM's kernels, not the trainer's — no MoE dispatch
        # copies (fused-MoE routes in-kernel) and a fused-residual stream
        # (add_rms_norm holds 2h per token where the trainer holds 4h).
        "startup_profiling_peak": formulas.block_recompute_bytes(
            arch,
            1,
            batched_tokens,
            act_bytes,
            device.has_flash_attention,
            engine_forward=True,
        )
        + sampler,
        "cuda_graphs": 0 if knobs.enforce_eager else formulas.CUDA_GRAPH_POOL_BYTES,
        "lora_slots": int(
            knobs.max_loras *
            formulas.lora_param_count(arch, knobs.max_lora_rank) *
            act_bytes
        ),
        "kv_demand": formulas.kv_cache_demand_bytes(
            arch,
            knobs.kv_cache_dtype,
            knobs.weight_dtype,
            knobs.concurrency,
            knobs.max_model_len,
        ),
        # Hybrid SSM only: constant per sequence, so it sits inside the engine
        # budget alongside the KV pool rather than scaling with context. The
        # block size is the *aligned* one, which for a hybrid is set by the
        # Mamba page rather than by vLLM's default 16.
        "mamba_state": formulas.mamba_state_bytes(
            arch,
            knobs.max_num_seqs,
            knobs.mamba_cache_mode,
            knobs.max_model_len,
            formulas.aligned_kv_block_size(
                arch, knobs.kv_cache_dtype, knobs.weight_dtype
            ),
            state_dtype=knobs.weight_dtype,
        ),
    }
    return terms, batched_tokens, sampler


def estimate_training(
    model: ModelSpec,
    device: DeviceSpec,
    knobs: TrainingKnobs,
    trainer_variant: str = "base",
    colocated: bool = False,
    orchestrated: bool = False,
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
    counts = formulas.param_counts(arch, model.n_params)
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
        and not formulas.materializes_attention_scores(attn_impl)
    )
    if knobs.beta == 0.0 and knobs.algorithm not in ("sft", "dpo"):
        warnings.append(
            "Assumes the reference forward is skipped at beta=0. The fused "
            "no-grad pass currently builds that row unconditionally, so an "
            "unpatched run pays for one extra row of activations."
        )
    if knobs.algorithm == "ppo":
        warnings.append(
            "The ppo training terms follow the code path (critic adapter, "
            "value head, fused rows) but its one measured sweep never "
            "exercised the batch axes — PPO samples one completion per "
            "prompt, and the rig sent one prompt — so multi-row updates "
            "carry no measured ground truth yet."
        )
    if knobs.distributed == "deepspeed" and knobs.dp_world_size > 1:
        if knobs.zero_stage == 3:
            warnings.append(
                f"DeepSpeed ZeRO-3 over {knobs.dp_world_size} GPUs: the "
                "backward surcharge and persistent-state terms are measured "
                "on one packed-MoE hybrid (Nemotron 3.5 Lightning 30B-A3B, "
                "A100-80) and are provisional beyond that family."
            )
        else:
            warnings.append(
                f"DeepSpeed ZeRO-{knobs.zero_stage} over "
                f"{knobs.dp_world_size} GPUs: the sharding terms are "
                "analytic; the one measured exemplar (Nemotron 30B-A3B, "
                "2 ranks) sat within 2% of this bar."
            )
    if knobs.lora_packed_target_matrices and knobs.packed_moe_dispatch == "contracted":
        warnings.append(
            "packed_moe_dispatch='contracted' assumes the contracted "
            "adapter path (no materialized effective weights, chunked "
            "dispatch), which is a projection — no measurement backs it "
            "until those kernels land and are swept."
        )
    if not flash_like:
        warnings.append(
            "attn_implementation='eager' materialises a rows x heads x S x S "
            "score matrix, which dominates activation memory at long "
            "context. Use sdpa or flex_attention."
        )

    extrapolated = _context_extrapolation_warning(s)
    if extrapolated:
        warnings.append(extrapolated)

    gap = _geometry_gap_warning(counts)
    if gap:
        warnings.append(gap)

    variant = model.variant(trainer_variant)
    kbit = knobs.quantization != "none"
    if kbit and arch.is_moe:
        warnings.append(
            f"Quantization barely helps this MoE: {counts.moe_experts / 1e9:.1f}B "
            f"of {counts.total / 1e9:.1f}B parameters are fused expert tensors, "
            "which bitsandbytes cannot reach, so they stay at the base dtype. "
            "Only the attention and router matrices shrink."
        )
    if kbit and arch.tied_embeddings and counts.embedding > counts.quantizable:
        # The tied lm_head is upcast to fp32 by k-bit preparation, and on a
        # wide-vocab model that block is bigger than everything nf4 can reach.
        warnings.append(
            "Quantizing this checkpoint may cost memory rather than save it: "
            f"{counts.embedding / 1e9:.1f}B of its parameters are a tied "
            "embedding that k-bit preparation upcasts to fp32, against "
            f"{counts.quantizable / 1e9:.1f}B that nf4 can reach. Measured on "
            "Gemma 4 E4B, nf4 was 0.9-2.5 GiB *dearer* at every context from "
            "512 to 8192 tokens."
        )
    base = formulas.weight_bytes(
        counts,
        knobs.weight_dtype,
        variant,
        kbit_prepared=kbit,
        tied_embeddings=arch.tied_embeddings,
    )
    # ZeRO-3 shards every parameter across the data-parallel group — for a
    # LoRA run the frozen base is where the memory is — and gathers modules
    # just-in-time, so each GPU holds its shard plus a gather working set.
    shards = knobs.dp_world_size
    if knobs.distributed == "deepspeed" and knobs.zero_stage == 3 and shards > 1:
        base = base / shards + formulas.zero3_gather_bytes(
            counts, arch, knobs.weight_dtype
        )

    lora_params = formulas.lora_param_count(
        arch, knobs.lora_rank, knobs.lora_target_scope
    )
    packed_lora_params = formulas.packed_lora_param_count(
        arch, knobs.lora_rank, knobs.lora_packed_target_matrices
    )
    adapter_bytes_per_param = formulas.ADAPTER_BYTES_PER_PARAM
    # PPOValueHead is Linear(hidden -> 1), held in modules_to_save so it is
    # trained alongside the adapters.
    value_head_params = arch.hidden_size + 1 if knobs.uses_critic else 0
    # Packed target_parameters adapters are held at the checkpoint dtype (the
    # live model measures 2 bytes/param), and ZeRO-3 shards them with the
    # base: 0.41 GiB logical measured as 0.10 GiB resident at world 4.
    packed_shards = (
        shards if knobs.distributed == "deepspeed" and knobs.zero_stage == 3 else 1
    )
    adapters = (
        lora_params * knobs.n_resident_adapters + value_head_params
    ) * adapter_bytes_per_param + (
        packed_lora_params *
        knobs.n_resident_adapters *
        DTYPE_BYTES[knobs.weight_dtype] /
        packed_shards
    )

    trained_params = (
        lora_params + packed_lora_params
    ) * knobs.n_trained_adapters + value_head_params
    # Gradients are fp32, mirroring the fp32 adapter parameters they
    # accumulate into. ZeRO stages 2 and 3 shard gradient storage and
    # optimizer state (moments + fp32 master) across the group; the transient
    # full-bucket gradient before reduce-scatter is adapter-sized and ignored.
    grads = trained_params * formulas.ADAPTER_BYTES_PER_PARAM / shards
    # The framework's resolved DeepSpeed config pins offload_optimizer to
    # CPU unconditionally, so under DeepSpeed the AdamW state costs the
    # device nothing (measured: 0 CUDA bytes, 0.66 GB host, on the 218M
    # trainable-parameter Nemotron run).
    opt_state = (
        0.0
        if knobs.distributed == "deepspeed"
        else trained_params *
        formulas.optimizer_bytes_per_trainable_param(knobs.distributed) /
        shards
    )

    # Gradient pass: checkpoint boundaries + one block's recompute peak +
    # the (rows, S, H) hidden state the fused logprob function saves.
    # ``graph_rows`` covers the rows whose autograd graphs are live together:
    # DPO holds the chosen and rejected graphs until one backward completes,
    # so its saved-side terms carry twice the micro-batch while the per-block
    # recompute transient still runs one graph at a time.
    grad_rows = knobs.grad_rows
    graph_rows = knobs.grad_graph_rows
    recompute = formulas.block_recompute_bytes(
        arch, grad_rows, s, act_bytes, flash_like, backward=True
    )
    if knobs.gradient_checkpointing:
        saved = formulas.activation_hidden_bytes(arch, graph_rows, s, act_bytes)
        saved += formulas.moe_resident_gather_bytes(arch, graph_rows, s, act_bytes)
    else:
        # Without checkpointing every block's intermediates are saved.
        saved = recompute * arch.n_layers * (graph_rows // max(grad_rows, 1))
        warnings.append(
            "gradient_checkpointing=False saves every block's activations; "
            "expect roughly n_layers x the checkpointed footprint."
        )
    if knobs.activation_offload:
        saved = 0  # backward-saved tensors live in pinned host RAM
    # bitsandbytes' backward workspace exists only under quantization, and it
    # is what makes nf4 stop paying at long context. It is transient rather
    # than offloadable, so it survives activation_offload.
    saved += formulas.bnb_backward_workspace_bytes(arch, grad_rows, s) if kbit else 0
    loss_hidden = graph_rows * s * arch.hidden_size * act_bytes
    # PEFT casts every wrapped linear's input to fp32; those copies stay
    # live for backward. Under ``_amp_ctx`` the original forward disables
    # them, but checkpoint recompute restores the flag — so one block's
    # casts reappear for ``grad_rows``. SFT's loss path stays outside
    # ``_amp_ctx`` and keeps its forward casts. See
    # :attr:`TrainingKnobs.lora_casts_recompute_only`.
    cast_rows = grad_rows if knobs.lora_casts_recompute_only else graph_rows
    lora_casts = formulas.lora_input_cast_bytes(
        arch,
        cast_rows,
        s,
        knobs.lora_target_scope,
        knobs.gradient_checkpointing,
    )
    loss_lora_casts = 0 if knobs.lora_casts_recompute_only else lora_casts
    # The fused chunked path holds vocab-width tiles bounded by
    # ``chunk_rows``, never a (B, S, V) slab. Two fp32 tiles are live at
    # the loss instant (recomputed logits plus that tile's gradient),
    # alongside the hoisted fp32 lm_head copy. Measured at the bound:
    # 519 + n x 256 MiB on Qwen2.5-0.5B; 2,069 vs 2,048 on Gemma-4-E2B.
    logit_rows = formulas.resolve_chunk_rows(arch.vocab_size, knobs.chunk_rows)
    logit_tile = logit_rows * arch.vocab_size * 4
    # The head copy exists only while the head is not already fp32: k-bit
    # preparation upcasts it persistently (charged in the weights), and an
    # fp32 run has nothing to upcast.
    fused_head = (
        formulas.fused_head_upcast_bytes(arch) if not kbit and act_bytes < 4 else 0
    )
    logits = logit_rows * arch.vocab_size * 8 + fused_head

    # No-grad logprob pass: actor + reference (+ value) rows fused into one
    # forward — a wider batch, but nothing saved for backward. Micro-batched
    # by the same per-GPU row cap as the gradient pass. SFT has no such pass
    # at all, so the instant does not exist for it.
    nograd_rows = knobs.grad_rows * knobs.n_adapter_rows
    nograd_pass = (
        formulas.nograd_forward_bytes(arch, nograd_rows, s, act_bytes, flash_like)
        if knobs.has_nograd_pass
        else 0
    )

    # A training step peaks at one of three instants; summing activation
    # and logit workspace assumes they coexist, and snapshots say they do
    # not. Gradients belong to the instant: the no-grad pass runs after
    # ``zero_grad(set_to_none=True)``, the optimizer step after transients
    # are freed. Take the max of the four instants, not their sum.
    backward_peak = grads + saved + recompute + loss_hidden + lora_casts
    loss_peak = grads + saved + loss_hidden + loss_lora_casts + logits
    # The no-grad fused loop hoists the same fp32 head copy; only one fp32
    # tile stands per chunk (nothing is saved for backward).
    nograd_peak = (
        (nograd_pass + logit_tile + fused_head) if knobs.has_nograd_pass else 0
    )
    optimizer_peak = grads
    peak = max(backward_peak, loss_peak, nograd_peak, optimizer_peak)

    # Report the split that belongs to whichever instant binds, so the bar
    # shows what is actually resident at the peak rather than a union.
    if peak == backward_peak:
        activations, logits, live_grads = backward_peak - grads, 0, grads
    elif peak == loss_peak:
        activations, logits, live_grads = loss_peak - logits - grads, logits, grads
    elif peak == nograd_peak:
        activations, logits, live_grads = nograd_pass, logit_tile + fused_head, 0
    else:
        activations, logits, live_grads = 0, 0, grads

    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages — (rows, S) fp32-ish, megabytes at most. ``trajectories`` is
    # this rank's share of the update, so data parallelism shards it.
    rollout = 6 * knobs.trajectories * s * 4

    zero3_active = (
        knobs.distributed == "deepspeed"
        and knobs.zero_stage == 3
        and knobs.dp_world_size > 1
    )
    zero3_backward = (
        formulas.zero3_packed_moe_backward_bytes(
            arch,
            knobs.trajectories,
            s,
            knobs.lora_packed_target_matrices,
            knobs.packed_moe_dispatch,
        )
        if zero3_active
        else 0
    )
    zero3_persistent = (
        formulas.zero3_persistent_state_bytes(
            arch,
            knobs.lora_packed_target_matrices,
            knobs.packed_moe_dispatch,
            act_bytes,
        )
        if zero3_active
        else 0
    )

    engine_residual = _engine_reservation_bytes(
        colocated and knobs.uses_generation_engine, arch, knobs.group_size
    )
    ray_overhead = formulas.RAY_ACTOR_OVERHEAD_BYTES if orchestrated else 0
    # An engine-less trainer still pays its own CUDA library workspaces; a
    # colocated run's are folded into the measured engine-floor constants.
    trainer_libs = (
        0 if knobs.uses_generation_engine else formulas.TRAINER_LIB_OVERHEAD_BYTES
    )

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
            detail=(
                {
                    "chunk_tiles": max(int(logits) - fused_head, 0),
                    "fp32_head_upcast": fused_head,
                }
                if logits and fused_head
                else {}
            ),
            note=(
                "The fused logprob path tiles the lm_head matmul; the full "
                "batch x seq x vocab slab is never materialised."
                + (
                    " Includes the fp32 lm_head copy the chunk loop hoists."
                    if fused_head
                    else ""
                )
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
            "zero3_state",
            "ZeRO-3 gathers + backward surcharge",
            zero3_backward + zero3_persistent,
            detail={
                "packed_moe_backward": zero3_backward,
                "persistent_state": zero3_persistent,
            },
            note=(
                "Measured at device level on a packed-MoE hybrid: the "
                "checkpointed backward of gathered expert leaves plus "
                "partition caches, reduce buckets and standing "
                "effective-weight copies. Fragmentation is folded in, so "
                "no allocator markup applies."
                if zero3_active
                else "Not ZeRO-3."
            ),
        ),
        _component(
            "overhead",
            "Overhead (context + slack)",
            device.context_bytes + rollout + ray_overhead + trainer_libs,
            detail={
                "cuda_context": device.context_bytes,
                "rollout_tensors": rollout,
                "ray_actor_overhead": ray_overhead,
                "trainer_lib_overhead": trainer_libs,
            },
        ),
    )

    # The device is charged what the caching allocator *reserves*, and every
    # term above is an allocation size. Non-torch terms are already device
    # figures and are not marked up — the ZeRO-3 terms carry their measured
    # fragmentation already.
    torch_side = sum(
        c.bytes_
        for c in components
        if c.key not in ("vllm_residual", "overhead", "zero3_state")
    )
    components = (
        *components,
        _component(
            "allocator_reserve",
            "Caching-allocator slack",
            formulas.allocator_reserve_bytes(torch_side),
            note=(
                "PyTorch reserves more than it allocates: segment rounding plus "
                "blocks it cannot reuse for a differently-shaped request."
            ),
        ),
    )

    return PhaseBreakdown(
        phase="training",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        warnings=tuple(warnings),
    )


def estimate_generation(
    model: ModelSpec,
    device: DeviceSpec,
    knobs: GenerationKnobs,
    train_knobs: TrainingKnobs | None = None,
    colocated: bool = False,
    orchestrated: bool = False,
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
    mamba_state = terms["mamba_state"]

    extrapolated = _context_extrapolation_warning(knobs.max_model_len)
    if extrapolated:
        warnings.append(extrapolated)

    gap = _geometry_gap_warning(formulas.param_counts(model.arch, model.n_params))
    if gap:
        warnings.append(gap)

    if arch.is_moe:
        # The dispatch copies are excluded engine-side (vLLM routes in-kernel),
        # but the fused-MoE kernel's own chunked intermediate caches are not
        # modelled either — their size hangs on VLLM_FUSED_MOE_CHUNK_SIZE,
        # which no measurement has pinned on this stack. Measured OLMoE
        # serving peaks ran 0.5-6% above this bar, worst where
        # max_num_batched_tokens is largest (5.8% at 16 slots x 4096 on the
        # full-plan re-sweep).
        warnings.append(
            "MoE serving peaks can exceed this bar by a few percent: vLLM's "
            "fused-MoE kernel holds chunked intermediate caches the model "
            "does not charge (measured up to 6% on OLMoE at 16 slots x 4096 "
            "context)."
        )

    if arch.multimodal_tower_params or arch.per_layer_input_dim:
        # Gemma 4 E2B under the ray-worker image: engine residency did not
        # track gpu_memory_utilization (flat at weights + ~3.9 GiB across
        # 0.28-0.44). Construction peaked at 1.76x the serving bar at 16
        # slots x 4096. This bar cannot be trusted to follow the budget
        # setting on a multimodal checkpoint.
        warnings.append(
            "Multimodal checkpoint: measured engine residency does not track "
            "gpu_memory_utilization (Gemma 4 E2B sat flat at weights + "
            "~4 GiB across 0.28-0.44 budgets, exceeding the low-end budget "
            "by ~2.3 GiB), and engine construction transiently peaked at "
            "1.76x this bar at 16 slots x 4096 context. Treat the bar as "
            "indicative and leave headroom for construction."
        )

    if arch.is_hybrid_ssm:
        block_size = formulas.aligned_kv_block_size(
            arch, knobs.kv_cache_dtype, knobs.weight_dtype
        )
        warnings.append(
            f"Hybrid state-space model: {arch.n_mamba_layers} recurrent layers "
            f"and {arch.attention_layers} attention. Only the attention layers "
            "hold a KV cache; the recurrent state is constant in context "
            "length. No hybrid model has been measured against this estimate — "
            "the terms follow vLLM's own state shapes but carry no profiled "
            "ground truth, so treat the bar as indicative."
        )
        if knobs.mamba_cache_mode == "all":
            aligned = formulas.mamba_state_bytes(
                arch,
                knobs.max_num_seqs,
                "align",
                knobs.max_model_len,
                block_size,
                state_dtype=knobs.weight_dtype,
            )
            blocks = formulas.mamba_state_blocks_per_seq(
                "all", knobs.max_model_len, block_size
            )
            warnings.append(
                f"mamba_cache_mode='all' keeps one recurrent state per block, "
                f"{blocks} per sequence at this context, so the state scales "
                "with context again — the cost the architecture exists to "
                f"avoid ({mamba_state / GiB:.1f} GiB against "
                f"{aligned / GiB:.1f} GiB under 'align'). vLLM selects it when "
                "prefix caching is on; 'align' caps residency at two blocks."
            )
        warnings.append(
            f"vLLM raises the attention block size to {block_size} tokens (from "
            "16) so an attention page holds a Mamba page, then pads the Mamba "
            "page to match. Every sequence's KV therefore rounds up to that "
            "page, which dominates the cache for short sequences."
        )

    if knobs.kv_cache_dtype == "fp8" and not device.supports_fp8:
        warnings.append(
            "fp8 KV cache requires compute capability >= 8.9; this device "
            "does not support it."
        )

    non_kv = weights + profiling_peak + graphs + lora_slots + mamba_state
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
            arch,
            1,
            runtime_tokens,
            act_bytes,
            device.has_flash_attention,
            engine_forward=True,
        )
        + sampler
    )

    # Trainer state that stays on the GPU while the base is offloaded
    # for the rollout. With ``use_memory_efficient_params=False`` the
    # trainer base and adapters stay on-device (PPO today).
    trainer_residual = 0.0
    if colocated and train_knobs is not None:
        lora_params = formulas.lora_param_count(
            model.arch, train_knobs.lora_rank, train_knobs.lora_target_scope
        )
        trainer_residual = (
            lora_params *
            train_knobs.n_trained_adapters *
            formulas.optimizer_bytes_per_trainable_param(train_knobs.distributed) /
            train_knobs.dp_world_size
        )
        if not train_knobs.use_memory_efficient_params:
            counts = formulas.param_counts(arch, model.n_params)
            trainer_variant = model.variant(
                "base"
                if train_knobs.quantization == "none"
                else train_knobs.quantization
            )
            value_head = arch.hidden_size + 1 if train_knobs.uses_critic else 0
            trainer_residual += (
                formulas.weight_bytes(
                    counts,
                    train_knobs.weight_dtype,
                    trainer_variant,
                    kbit_prepared=train_knobs.quantization != "none",
                    tied_embeddings=arch.tied_embeddings,
                )
                + (lora_params * train_knobs.n_resident_adapters + value_head) *
                formulas.ADAPTER_BYTES_PER_PARAM
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
        _component(
            "mamba_state",
            "Mamba recurrent state",
            mamba_state,
            detail={
                "recurrent_layers": arch.n_mamba_layers,
                "state_blocks_per_seq": formulas.mamba_state_blocks_per_seq(
                    knobs.mamba_cache_mode, knobs.max_model_len
                ),
            },
            note=(
                f"{arch.n_mamba_layers} recurrent layers against "
                f"{arch.attention_layers} attention. Constant in context "
                f"length; mamba_cache_mode={knobs.mamba_cache_mode!r}."
                if arch.is_hybrid_ssm
                else "Not a hybrid state-space model."
            ),
        ),
        _component("lora_slots", "LoRA adapter slots", lora_slots),
        _component(
            "trainer_residual",
            "Offloaded trainer residual",
            trainer_residual,
            note=(
                (
                    "use_memory_efficient_params=False: the whole trainer — "
                    "base weights, adapters, optimizer state — stays on the "
                    "device through the rollout."
                    if train_knobs is not None
                    and not train_knobs.use_memory_efficient_params
                    else "Optimizer state stays on-device while the trainer "
                    "base sits in host RAM during rollout."
                )
                if colocated
                else "Not colocated."
            ),
        ),
        _component(
            "overhead",
            "Overhead (context + slack)",
            device.context_bytes
            + formulas.ENGINE_PROCESS_OVERHEAD_BYTES
            + (formulas.RAY_ACTOR_OVERHEAD_BYTES if orchestrated else 0),
            detail={
                "cuda_context": device.context_bytes,
                "engine_process_overhead": formulas.ENGINE_PROCESS_OVERHEAD_BYTES,
                "ray_actor_overhead": (
                    formulas.RAY_ACTOR_OVERHEAD_BYTES if orchestrated else 0
                ),
            },
        ),
    )

    return PhaseBreakdown(
        phase="generation",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        warnings=tuple(warnings),
    )


def generation_can_serve(breakdown: PhaseBreakdown) -> bool:
    """Whether this generation bar is actually usable at the advertised context.

    ``fits`` only says the resident peak is under the card. Serving also
    needs a KV pool that starts (non-empty) and covers worst-case demand
    for every concurrent sequence at ``max_model_len``. Demand above the
    pool is preemption, not an OOM — still not a context the engine can
    honour.
    """
    if breakdown.phase != "generation" or not breakdown.fits:
        return False
    kv = next((c for c in breakdown.components if c.key == "kv_cache"), None)
    if kv is None:
        return False
    pool = kv.detail.get("pool", kv.bytes_)
    demand = kv.detail.get("worst_case_demand", 0)
    return pool > 0 and demand <= pool


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


def estimate_run(config: RunConfig) -> RunEstimate:
    """Estimate both phases for a run configuration.

    An algorithm that never starts a generation engine (SFT, DPO) gets an
    empty generation bar rather than the bar of an engine it would not run —
    a phantom engine could block a run that fits.
    """
    model = config.model
    engine = config.training.uses_generation_engine
    # The trainer always loads the base checkpoint; QLoRA is modelled through
    # the quantization setting, not through a prequantized weight variant. Only
    # the profiling rig sizes non-base trainer variants, and it calls
    # :func:`estimate_training` directly.
    training = estimate_training(
        model,
        config.train_device,
        config.training,
        colocated=config.colocated and engine,
        orchestrated=config.orchestrated,
    )
    if not engine:
        device = config.generation_device
        generation = PhaseBreakdown(
            phase="generation",
            components=(),
            device_total_bytes=device.total_bytes,
            device_usable_bytes=device.usable_bytes,
            warnings=(
                (
                    f"{config.training.algorithm} trains from a fixed dataset "
                    "and starts no generation engine; there is nothing to size."
                ),
            ),
        )
    else:
        generation = estimate_generation(
            model,
            config.generation_device,
            config.generation,
            train_knobs=config.training,
            colocated=config.colocated,
            orchestrated=config.orchestrated,
        )
    return RunEstimate(training=training, generation=generation)
