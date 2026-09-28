# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form peak-memory estimation for LLM RL training and generation.

One stacked-bar breakdown per phase. Training and generation run on
separate devices, so each phase sizes its own device with no cross-phase
residuals.

Component keys are a published JSON contract; change them only with a
schema-version bump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GenerationSettings,
    GiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
)

PhaseName = Literal["training", "generation"]

# Storage of an FSDP-replicated parameter, keyed by ``FSDPConfig.param_dtype``.
FSDP_PARAM_BYTES = {
    "bfloat16": 2.0,
    "float16": 2.0,
    "float32": 4.0,
    "float64": 8.0,
}


class MemoryComponent(BaseModel):
    """One segment of a phase's stacked bar."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    key: str
    label: str
    n_bytes: int = Field(alias="bytes", serialization_alias="bytes")
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
        return sum(c.n_bytes for c in self.components)

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.device_usable_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.device_usable_bytes - self.total_bytes


class RunEstimate(BaseModel):
    """The two independent phase bars."""

    model_config = ConfigDict(frozen=True)

    training: PhaseBreakdown
    generation: PhaseBreakdown

    @property
    def fits(self) -> bool:
        return self.training.fits and self.generation.fits


def geometry_gap_warning(counts: formulas.ParamCounts) -> str | None:
    """Warn when parsed geometry leaves a large share of parameters unattributed."""
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


def engine_terms(
    model: ModelSpec, settings: GenerationSettings
) -> tuple[dict[str, int], int, int]:
    """Engine-budget terms, scheduler-step token cap, and sampler buffer bytes."""
    arch = model.arch
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    counts = formulas.param_counts(arch, model.n_params)
    variant = model.variant(settings.weight_variant)
    batched_tokens = formulas.resolve_max_num_batched_tokens(
        settings.max_num_seqs, settings.max_model_len, settings.max_num_batched_tokens
    )
    # Sampler buffers: per-sequence fp32 logits and probs over the vocab.
    sampler = 2 * settings.max_num_seqs * arch.vocab_size * 4
    terms = {
        "weights": formulas.weight_bytes(counts, settings.weight_dtype, variant),
        "startup_peak": formulas.block_recompute_bytes(
            arch,
            1,
            batched_tokens,
            act_bytes,
            engine_forward=True,
        )
        + sampler,
        "cuda_graphs": 0 if settings.enforce_eager else formulas.CUDA_GRAPH_POOL_BYTES,
        "lora_slots": int(
            settings.max_loras
            * formulas.lora_param_count(arch, settings.max_lora_rank)
            * act_bytes
        ),
        "kv_demand": formulas.kv_cache_demand_bytes(
            arch,
            settings.weight_dtype,
            settings.concurrency,
            settings.max_model_len,
        ),
        "mamba_state": formulas.mamba_state_bytes(
            arch,
            settings.max_num_seqs,
            state_dtype=settings.weight_dtype,
        ),
    }
    return terms, batched_tokens, sampler


def training_warnings(
    counts: formulas.ParamCounts,
    settings: TrainingSettings,
) -> list[str]:
    """User-facing training-phase warnings."""
    warnings: list[str] = []
    if settings.beta == 0.0 and settings.algorithm not in ("sft", "dpo"):
        warnings.append(
            "At beta=0 the estimate skips the reference forward, but the fused "
            "no-grad pass still builds that row, so the run pays for one extra "
            "row of activations."
        )
    if settings.n_training_gpus > 1:
        n_gpus = settings.n_training_gpus
        if settings.fsdp is None:
            warnings.append(
                f"Flat data parallel over {n_gpus} GPUs. Each GPU holds a full "
                "copy of the weights, gradients, and Adam state."
            )
        elif settings.fsdp.reshard_after_forward:
            warnings.append(
                f"FSDP2 over {n_gpus} GPUs. Parameters are resharded after "
                "each unit's forward."
            )
        else:
            warnings.append(
                f"FSDP2 over {n_gpus} GPUs. Parameters stay gathered from "
                "forward through backward."
            )
    if (
        settings.lora_packed_target_matrices
        and settings.packed_moe_dispatch == "contracted"
    ):
        warnings.append(
            "packed_moe_dispatch='contracted' charges the split expert-LoRA "
            "residual (top-k gather, per-expert cat, fp32 GEMMs) on the "
            "checkpointed backward."
        )
    gap = geometry_gap_warning(counts)
    if gap:
        warnings.append(gap)
    if not settings.gradient_checkpointing:
        warnings.append(
            "gradient_checkpointing=False saves every block's activations; "
            "expect roughly n_layers x the checkpointed footprint."
        )
    return warnings


@dataclass(frozen=True)
class AdapterState:
    """LoRA adapter, gradient, and Adam bytes on one training GPU."""

    lora_params: int
    packed_lora_params: int
    adapters: float
    grads: float
    step_grads: float
    adam: float
    value_head_params: int


def _value_head_placement(hidden: int, threshold: int | None) -> tuple[int, int]:
    """(replicated, sharded) elements of the critic's ``Linear(hidden, 1)``."""
    replicated = 0
    sharded = 0
    for numel in (hidden, 1):
        if threshold is None or numel < threshold:
            replicated += numel
        else:
            sharded += numel
    return replicated, sharded


def adapter_state(
    arch: ModelArch, settings: TrainingSettings, shards: int
) -> AdapterState:
    """Adapter copies, gradients, and fp32 Adam moments on one GPU."""
    lora_params = formulas.lora_param_count(
        arch, settings.lora_rank, settings.lora_target_scope
    )
    packed_lora_params = formulas.packed_lora_param_count(
        arch, settings.lora_rank, settings.lora_packed_target_matrices
    )
    value_head_params = arch.hidden_size + 1 if settings.uses_critic else 0
    fsdp = settings.fsdp
    threshold = None if fsdp is None else fsdp.param_persistence_threshold
    replicated, sharded = formulas.lora_tensor_placement(
        arch,
        settings.lora_rank,
        settings.lora_target_scope,
        settings.lora_packed_target_matrices,
        threshold,
    )
    if settings.uses_critic:
        head_repl, head_shard = _value_head_placement(arch.hidden_size, threshold)
    else:
        head_repl, head_shard = 0, 0
    if fsdp is None:
        replicated_bytes = formulas.ADAPTER_BYTES_PER_PARAM
        sharded_bytes = formulas.ADAPTER_BYTES_PER_PARAM
    else:
        replicated_bytes = FSDP_PARAM_BYTES[fsdp.param_dtype]
        sharded_bytes = DTYPE_BYTES[settings.weight_dtype]
    offload = fsdp is not None and fsdp.cpu_offload
    hold_grads = (
        fsdp is not None
        and fsdp.defer_grad_sync
        and shards > 1
        and settings.trajectories > 1
        and not offload
    )

    def placed(
        kept: int,
        split: int,
        kept_bytes: float,
        split_bytes: float,
        *,
        gather: bool = False,
    ) -> float:
        if offload:
            return kept * kept_bytes
        divisor = 1 if gather else shards
        return kept * kept_bytes + split * split_bytes / divisor

    n_resident = settings.n_resident_adapters
    n_trained = settings.n_trained_adapters
    adapters = placed(
        replicated * n_resident, sharded * n_resident, replicated_bytes, sharded_bytes
    )
    adapters += placed(head_repl, head_shard, replicated_bytes, sharded_bytes)
    trained_repl = replicated * n_trained + head_repl
    trained_shard = sharded * n_trained + head_shard
    grads = placed(
        trained_repl,
        trained_shard,
        replicated_bytes,
        sharded_bytes,
        gather=hold_grads,
    )
    step_grads = placed(trained_repl, trained_shard, replicated_bytes, sharded_bytes)
    if offload:
        adam = 0.0
    else:
        adam = placed(
            trained_repl,
            trained_shard,
            formulas.ADAM_BYTES_PER_PARAM,
            formulas.ADAM_BYTES_PER_PARAM,
        )
    return AdapterState(
        lora_params=lora_params,
        packed_lora_params=packed_lora_params,
        adapters=adapters,
        grads=grads,
        step_grads=step_grads,
        adam=adam,
        value_head_params=value_head_params,
    )


@dataclass(frozen=True)
class ActivationPeak:
    """Resident activations, logits, and grads at the binding training instant."""

    activations: float
    logits: float
    live_grads: float
    adam: float
    backward_peak: float
    loss_peak: float
    nograd_peak: float
    optimizer_peak: float
    saved: float
    recompute: float
    split_lora: float
    loss_hidden: float
    lora_casts: float
    fused_head: int


def activation_peak(
    arch: ModelArch,
    settings: TrainingSettings,
    *,
    act_bytes: float,
    grads: float,
    step_grads: float,
    adam: float,
    seq_len: int,
) -> ActivationPeak:
    """Peak of backward, fused loss, no-grad logprob, and optimizer instants."""
    graph_rows = settings.grad_graph_rows
    adam_resident = settings.fsdp is None or (
        not settings.fsdp.cpu_offload and not settings.fsdp.optim_cpu_offload
    )
    adam_always = adam if adam_resident else 0.0
    recompute = formulas.block_recompute_bytes(
        arch, 1, seq_len, act_bytes, backward=True
    )
    if settings.gradient_checkpointing:
        saved = int(graph_rows * seq_len * arch.hidden_size * arch.n_layers * act_bytes)
        saved += formulas.moe_resident_gather_bytes(
            arch, graph_rows, seq_len, act_bytes
        )
    else:
        saved = recompute * arch.n_layers * graph_rows
    if settings.activation_offload:
        saved = 0
    loss_hidden = graph_rows * seq_len * arch.hidden_size * act_bytes
    lora_casts = formulas.lora_input_cast_bytes(
        arch,
        1,
        seq_len,
        settings.lora_target_scope,
        settings.gradient_checkpointing,
    )
    loss_lora_casts = 0 if settings.lora_casts_recompute_only else lora_casts
    # Two fp32 tiles are live at the loss instant: recomputed logits plus
    # that tile's gradient.
    logit_rows = formulas.resolve_chunk_rows(arch.vocab_size, settings.chunk_rows)
    logit_tile = logit_rows * arch.vocab_size * 4
    # The chunk loop hoists an fp32 lm_head copy when the head runs below fp32.
    fused_head = int(arch.vocab_size * arch.hidden_size * 4) if act_bytes < 4 else 0
    logits = logit_rows * arch.vocab_size * 8 + fused_head
    # The no-grad pass chunks rows by the same per-GPU cap as the gradient pass.
    nograd_rows = settings.n_adapter_rows
    nograd_pass = (
        formulas.nograd_forward_bytes(arch, nograd_rows, seq_len, act_bytes)
        if settings.has_nograd_pass
        else 0
    )
    split_lora = formulas.split_moe_lora_recompute_bytes(
        arch,
        1,
        seq_len,
        settings.lora_packed_target_matrices,
        settings.packed_moe_dispatch,
        act_bytes,
    )
    backward_peak = grads + saved + recompute + loss_hidden + lora_casts + split_lora
    loss_peak = grads + saved + loss_hidden + loss_lora_casts + logits
    nograd_peak = (
        (nograd_pass + logit_tile + fused_head) if settings.has_nograd_pass else 0
    )
    optimizer_peak = step_grads
    backward_total = backward_peak + adam_always
    loss_total = loss_peak + adam_always
    nograd_total = nograd_peak + (adam_always if settings.has_nograd_pass else 0)
    optimizer_total = optimizer_peak + adam
    peak = max(backward_total, loss_total, nograd_total, optimizer_total)
    if peak == backward_total:
        activations, live_logits, live_grads, live_adam = (
            backward_peak - grads,
            0,
            grads,
            adam_always,
        )
    elif peak == loss_total:
        activations, live_logits, live_grads, live_adam = (
            loss_peak - logits - grads,
            logits,
            grads,
            adam_always,
        )
    elif peak == nograd_total:
        activations, live_logits, live_grads, live_adam = (
            nograd_pass,
            logit_tile + fused_head,
            0,
            adam_always if settings.has_nograd_pass else 0,
        )
    else:
        activations, live_logits, live_grads, live_adam = 0, 0, step_grads, adam
    return ActivationPeak(
        activations=activations,
        logits=live_logits,
        live_grads=live_grads,
        adam=live_adam,
        backward_peak=backward_peak,
        loss_peak=loss_peak,
        nograd_peak=nograd_peak,
        optimizer_peak=optimizer_peak,
        saved=saved,
        recompute=recompute,
        split_lora=split_lora,
        loss_hidden=loss_hidden,
        lora_casts=lora_casts,
        fused_head=fused_head,
    )


def trainer_base_bytes(
    model: ModelSpec,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
) -> float:
    """Frozen base weights on one training GPU, including the FSDP gather."""
    full = formulas.weight_bytes(counts, settings.weight_dtype, model.variant("base"))
    fsdp = settings.fsdp
    if fsdp is None or (shards <= 1 and not fsdp.cpu_offload):
        return float(full)
    token = min(formulas.token_embedding_params(model.arch), max(counts.total, 0))
    sharded_params = max(counts.total - token, 0)
    gathered = formulas.fsdp_gathered_params(
        counts,
        model.arch,
        n_gpus=shards,
        reshard_after_forward=fsdp.reshard_after_forward,
        prefetch_units=fsdp.prefetch_units,
        wrap_every_n_blocks=fsdp.wrap_every_n_blocks,
        cpu_offload=fsdp.cpu_offload,
    )
    byte = DTYPE_BYTES[settings.weight_dtype]
    resident = 0.0 if fsdp.cpu_offload else sharded_params * byte / shards
    return token * byte + resident + gathered * byte


def estimate_training(
    model: ModelSpec,
    device: DeviceSpec,
    settings: TrainingSettings,
    orchestrated: bool = False,
) -> PhaseBreakdown:
    """Peak training-phase memory on the training device."""
    arch = model.arch
    counts = formulas.param_counts(arch, model.n_params)
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    seq_len = settings.max_model_len
    warnings = training_warnings(counts, settings)
    shards = settings.n_training_gpus
    base = trainer_base_bytes(model, settings, counts, shards)
    adapters = adapter_state(arch, settings, shards)
    peak = activation_peak(
        arch,
        settings,
        act_bytes=act_bytes,
        grads=adapters.grads,
        step_grads=adapters.step_grads,
        adam=adapters.adam,
        seq_len=seq_len,
    )
    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages.
    rollout = 6 * settings.trajectories * seq_len * 4
    job_overhead = formulas.JOB_OVERHEAD_BYTES if orchestrated else 0
    if settings.fsdp is None:
        adapter_bytes_per_param = formulas.ADAPTER_BYTES_PER_PARAM
    else:
        adapter_bytes_per_param = FSDP_PARAM_BYTES[settings.fsdp.param_dtype]
    overhead = (
        device.context_bytes
        + rollout
        + job_overhead
        + formulas.TRAINER_LIB_OVERHEAD_BYTES
    )
    torch_side = (
        base
        + adapters.adapters
        + peak.live_grads
        + peak.activations
        + peak.logits
        + peak.adam
    )
    reserve = formulas.allocator_reserve_bytes(torch_side)
    components = (
        MemoryComponent(
            key="base_weights",
            label="Base weights (frozen)",
            n_bytes=max(int(base), 0),
        ),
        MemoryComponent(
            key="adapters",
            label="LoRA adapters",
            n_bytes=max(int(adapters.adapters), 0),
            detail={
                "actor": int(adapters.lora_params * adapter_bytes_per_param),
                "reference": int(
                    adapters.lora_params * adapter_bytes_per_param
                    if settings.n_resident_adapters > 1
                    else 0
                ),
                "value_head": int(adapters.value_head_params * adapter_bytes_per_param),
            },
            note="The reference is a frozen adapter copy. beta does not change this footprint.",
        ),
        MemoryComponent(
            key="grads",
            label="Gradients",
            n_bytes=max(int(peak.live_grads), 0),
            note="LoRA-only training: scales with adapter params.",
        ),
        MemoryComponent(
            key="activations",
            label="Activations",
            n_bytes=max(int(peak.activations), 0),
            detail={
                "backward_peak": int(peak.backward_peak),
                "loss_peak": int(peak.loss_peak),
                "nograd_peak": int(peak.nograd_peak),
                "optimizer_peak": int(peak.optimizer_peak),
                "checkpoint_boundaries": int(peak.saved),
                "block_recompute": int(peak.recompute),
                "split_moe_lora": int(peak.split_lora),
                "loss_hidden_state": int(peak.loss_hidden),
                "lora_fp32_input_casts": int(peak.lora_casts),
                "backward_grads": int(adapters.grads),
                "step_grads": int(adapters.step_grads),
            },
            note=(
                "Peak of the gradient micro-batch pass vs the fused no-grad "
                "logprob pass (actor+reference rows)."
            ),
        ),
        MemoryComponent(
            key="logits_workspace",
            label="Logit workspace (chunked)",
            n_bytes=max(int(peak.logits), 0),
            detail=(
                {
                    "chunk_tiles": max(int(peak.logits) - peak.fused_head, 0),
                    "fp32_head_upcast": peak.fused_head,
                }
                if peak.logits and peak.fused_head
                else {}
            ),
            note=(
                "The fused logprob path tiles the lm_head matmul."
                + (
                    " Includes the fp32 lm_head copy the chunk loop hoists."
                    if peak.fused_head
                    else ""
                )
            ),
        ),
        MemoryComponent(
            key="optimizer_state",
            label="AdamW state",
            n_bytes=max(int(peak.adam), 0),
            detail={"step_bytes": int(adapters.adam)},
            note=(
                "fp32 Adam moments. Resident for the whole step, or only "
                "during step() when FSDP offloads them."
            ),
        ),
        MemoryComponent(
            key="overhead",
            label="Overhead (context + slack)",
            n_bytes=max(int(overhead), 0),
            detail={
                "cuda_context": int(device.context_bytes),
                "rollout_tensors": int(rollout),
                "job_overhead": int(job_overhead),
                "trainer_lib_overhead": int(formulas.TRAINER_LIB_OVERHEAD_BYTES),
            },
        ),
        MemoryComponent(
            key="allocator_reserve",
            label="Caching-allocator slack",
            n_bytes=max(int(reserve), 0),
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


def generation_warnings(
    arch: ModelArch,
    model: ModelSpec,
    settings: GenerationSettings,
    *,
    kv_pool: int,
    kv_demand: int,
) -> list[str]:
    """User-facing generation-phase warnings."""
    warnings: list[str] = []
    gap = geometry_gap_warning(formulas.param_counts(arch, model.n_params))
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
        block_size = formulas.aligned_kv_block_size(arch, settings.weight_dtype)
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
    if kv_pool <= 0:
        warnings.append(
            "gpu_memory_utilization budget is consumed by weights and engine "
            "overhead before any KV cache: vLLM will fail at init. Raise "
            "gpu_memory_utilization or shrink the model/variant."
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
    arch = model.arch
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    budget = int(settings.gpu_memory_utilization * device.total_bytes)
    terms, batched_tokens, sampler = engine_terms(model, settings)
    weights = terms["weights"]
    startup_peak = terms["startup_peak"]
    graphs = terms["cuda_graphs"]
    lora_slots = terms["lora_slots"]
    kv_demand = terms["kv_demand"]
    mamba_state = terms["mamba_state"]
    non_kv = weights + startup_peak + graphs + lora_slots + mamba_state
    if settings.kv_cache_memory_bytes is not None:
        kv_pool = settings.kv_cache_memory_bytes
    else:
        kv_pool = max(budget - non_kv, 0)
    warnings = generation_warnings(
        arch,
        model,
        settings,
        kv_pool=kv_pool,
        kv_demand=kv_demand,
    )
    runtime_tokens = min(settings.concurrency * settings.prompt_len, batched_tokens)
    runtime_activation = (
        formulas.block_recompute_bytes(
            arch,
            1,
            runtime_tokens,
            act_bytes,
            engine_forward=True,
        )
        + sampler
    )
    components = (
        MemoryComponent(
            key="weights",
            label="Model weights (engine copy)",
            n_bytes=max(int(weights), 0),
        ),
        MemoryComponent(
            key="kv_cache",
            label="KV cache pool",
            n_bytes=max(int(kv_pool), 0),
            detail={
                "pool": int(kv_pool),
                "worst_case_demand": int(kv_demand),
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
            n_bytes=max(int(runtime_activation), 0),
            detail={
                "runtime_tokens": int(runtime_tokens),
                "sampler_buffers": int(sampler),
                "startup_peak": int(startup_peak),
            },
            note=(
                f"Serving peak over {runtime_tokens} tokens. The start-up "
                f"scheduler step of {batched_tokens} tokens is larger and "
                "sizes the KV pool."
            ),
        ),
        MemoryComponent(
            key="cuda_graphs",
            label="CUDA graphs",
            n_bytes=max(int(graphs), 0),
            note="Set enforce_eager=True to skip capture (throughput cost).",
        ),
        MemoryComponent(
            key="mamba_state",
            label="Mamba recurrent state",
            n_bytes=max(int(mamba_state), 0),
            detail={
                "recurrent_layers": arch.n_mamba_layers,
                "state_blocks_per_seq": formulas.MAMBA_STATE_BLOCKS_PER_SEQ,
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
            n_bytes=max(int(lora_slots), 0),
        ),
        MemoryComponent(
            key="overhead",
            label="Overhead (context + slack)",
            n_bytes=max(
                int(
                    device.context_bytes
                    + formulas.ENGINE_PROCESS_OVERHEAD_BYTES
                    + (formulas.JOB_OVERHEAD_BYTES if orchestrated else 0)
                ),
                0,
            ),
            detail={
                "cuda_context": int(device.context_bytes),
                "engine_process_overhead": int(formulas.ENGINE_PROCESS_OVERHEAD_BYTES),
                "job_overhead": int(formulas.JOB_OVERHEAD_BYTES if orchestrated else 0),
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
    """Whether this generation bar can serve the advertised context.

    ``fits`` is resident peak under the card. Serving also needs a non-empty
    KV pool that covers worst-case demand at ``max_model_len``. Demand above
    the pool means preemption under that context.
    """
    if breakdown.phase != "generation" or not breakdown.fits:
        return False
    kv = next((c for c in breakdown.components if c.key == "kv_cache"), None)
    if kv is None:
        return False
    pool = kv.detail.get("pool", kv.n_bytes)
    demand = kv.detail.get("worst_case_demand", 0)
    return pool > 0 and demand <= pool


def estimate_run(config: RunConfig) -> RunEstimate:
    """Estimate both phases for a run configuration.

    Algorithms with no generation engine (SFT, DPO) get an empty
    generation bar.
    """
    model = config.model
    engine = config.training.uses_generation_engine
    training = estimate_training(
        model,
        config.train_device,
        config.training,
        orchestrated=config.orchestrated,
    )
    if not engine:
        device = config.gen_device
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
            config.gen_device,
            config.generation,
            orchestrated=config.orchestrated,
        )
    return RunEstimate(training=training, generation=generation)
