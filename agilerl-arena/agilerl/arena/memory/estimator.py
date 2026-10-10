# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form peak-memory estimation for LLM RL training and generation.

One stacked-bar breakdown per phase. Training and generation run on
separate devices, so each phase is sized against its own device.

Component keys are a published JSON contract; change them only with a
schema-version bump.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agilerl.arena.memory import formulas
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    EAGER_MAMBA_SCAN_CAPABILITY,
    BlockKind,
    DeviceSpec,
    GenerationSettings,
    GiB,
    MiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
)
from agilerl.arena.models.fsdp import FSDPConfig

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
    # Host RAM one training process holds beside the device peak. Not part
    # of ``total_bytes`` or ``fits``.
    host: tuple[MemoryComponent, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(c.n_bytes for c in self.components)

    @property
    def host_total_bytes(self) -> int:
        return sum(c.n_bytes for c in self.host)

    @property
    def fits(self) -> bool:
        return self.total_bytes <= self.device_usable_bytes

    @property
    def fits_with_buffer(self) -> bool:
        """Training fit leaves the underprediction share free; generation does not."""
        if self.phase != "training":
            return self.fits
        return formulas.recommendation_fits(self.total_bytes, self.device_usable_bytes)

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

    @property
    def fits_with_buffer(self) -> bool:
        """Both phases fit after the underprediction buffer."""
        return self.training.fits_with_buffer and self.generation.fits_with_buffer


def geometry_gap_warning(counts: formulas.ParamCounts) -> str | None:
    """Warn when parsed geometry leaves a large share of parameters unattributed.

    :param counts: Analytic parameter split from :func:`formulas.param_counts`.
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
    if not settings.uses_reference and settings.algorithm != "sft":
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

    bytes_per_param: float
    adapters: float
    grads: float
    step_grads: float
    adam: float
    # Foreach AdamW temporaries during step() on the GPU.
    adam_workspace: float
    # Trained adapters and value head, unsharded.
    trained_params: int
    trained_bytes: float


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
    """Adapter copies, gradients, and fp32 Adam moments on one GPU.

    :param arch: Decoder geometry.
    :param settings: Training settings (rank, scope, FSDP).
    :param shards: GPUs one weight copy is sharded across; 1 means no FSDP shard.
    :return: Adapter-state bytes on one GPU.
    """
    fsdp = settings.fsdp
    threshold = None if fsdp is None else fsdp.param_persistence_threshold
    replicated, sharded = formulas.lora_tensor_placement(
        arch,
        settings.lora_rank,
        settings.lora_target_scope,
        settings.lora_packed_target_matrices,
        threshold,
    )
    head_repl, head_shard = (
        _value_head_placement(arch.hidden_size, threshold)
        if settings.uses_critic
        else (0, 0)
    )
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

    def resident_bytes(
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
    adapters = resident_bytes(
        replicated * n_resident, sharded * n_resident, replicated_bytes, sharded_bytes
    )
    adapters += resident_bytes(head_repl, head_shard, replicated_bytes, sharded_bytes)
    trained_repl = replicated * n_trained + head_repl
    trained_shard = sharded * n_trained + head_shard
    grads = resident_bytes(
        trained_repl,
        trained_shard,
        replicated_bytes,
        sharded_bytes,
        gather=hold_grads,
    )
    step_grads = resident_bytes(
        trained_repl, trained_shard, replicated_bytes, sharded_bytes
    )
    if offload:
        adam = 0.0
    else:
        adam = resident_bytes(
            trained_repl,
            trained_shard,
            formulas.ADAM_BYTES_PER_PARAM,
            formulas.ADAM_BYTES_PER_PARAM,
        )
    # Flat data parallel steps foreach AdamW on the GPU. FSDP steps fused
    # AdamW on the GPU, or steps on the host when offloaded.
    adam_workspace = (
        resident_bytes(
            trained_repl,
            trained_shard,
            formulas.FOREACH_ADAM_WORKSPACE_BYTES_PER_PARAM,
            formulas.FOREACH_ADAM_WORKSPACE_BYTES_PER_PARAM,
        )
        if fsdp is None
        else 0.0
    )
    return AdapterState(
        bytes_per_param=replicated_bytes,
        adapters=adapters,
        grads=grads,
        step_grads=step_grads,
        adam=adam,
        adam_workspace=adam_workspace,
        trained_params=trained_repl + trained_shard,
        trained_bytes=trained_repl * replicated_bytes + trained_shard * sharded_bytes,
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
    routed_chunks: float
    loss_hidden: float
    lora_casts: float
    lora_dropout: float
    fused_head: int


def block_backward_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    seq_len: int,
    act_bytes: float,
    adapter_bytes: float,
    eager_mamba_scan: bool = False,
) -> tuple[int, int, int, int, int]:
    """Recompute, fp32 casts, LoRA dropout, split expert-LoRA and routed chunks of one checkpointed block.

    A block-exclusive stack recomputes one attention, FFN or Mamba layer at a
    time, so each kind's terms are summed and the largest kind is kept.
    """
    rows = settings.grad_forward_rows
    # PEFT casts a wrapped linear's input only when the adapters are wider.
    casts_apply = adapter_bytes > act_bytes
    dropout_applies = settings.lora_dropout > 0
    split = formulas.split_moe_lora_recompute_bytes(
        arch,
        rows,
        seq_len,
        settings.lora_packed_target_matrices,
        settings.packed_moe_dispatch,
        act_bytes,
        settings.lora_rank,
    )
    fsdp = settings.fsdp
    chunks = formulas.routed_chunk_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        (
            formulas.DEFAULT_ROUTED_CHUNK_BYTES
            if fsdp is None
            else fsdp.routed_expert_chunk_mib * MiB
        ),
        expert_lora=settings.lora_packed_target_matrices > 0,
        contracted=settings.packed_moe_dispatch == "contracted",
        ep=1 if fsdp is None else fsdp.ep,
    )

    def terms(layer: BlockKind | None) -> tuple[int, int, int, int, int]:
        recompute = formulas.block_recompute_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            backward=True,
            layer=layer,
            eager_mamba_scan=eager_mamba_scan,
        )
        casts = 0
        if casts_apply:
            casts = formulas.lora_input_cast_bytes(
                arch,
                rows,
                seq_len,
                settings.lora_target_scope,
                settings.gradient_checkpointing,
                layer=layer,
            )
        dropout = 0
        if dropout_applies:
            dropout = formulas.lora_dropout_bytes(
                arch,
                rows,
                seq_len,
                adapter_bytes,
                settings.lora_target_scope,
                settings.gradient_checkpointing,
                layer=layer,
            )
        moe_here = layer in (None, "ffn")
        return (
            recompute,
            casts,
            dropout,
            split if moe_here else 0,
            chunks if moe_here else 0,
        )

    if not (arch.block_exclusive_layers and settings.gradient_checkpointing):
        return terms(None)
    return max((terms(kind) for kind in formulas.block_kinds(arch)), key=sum)


def activation_peak(
    arch: ModelArch,
    settings: TrainingSettings,
    *,
    act_bytes: float,
    adapter_bytes: float,
    grads: float,
    step_grads: float,
    adam: float,
    adam_workspace: float,
    seq_len: int,
    eager_mamba_scan: bool = False,
) -> ActivationPeak:
    """Peak of backward, fused loss, no-grad logprob, and optimizer instants.

    :param adapter_bytes: Bytes per LoRA element: fp32 under flat data
        parallel, FSDP's ``param_dtype`` when sharded.
    :param adam_workspace: Optimizer temporaries live only during step().
    :param eager_mamba_scan: Mamba layers run HF's eager scan.
    """
    graph_rows = settings.grad_graph_rows
    forward_rows = settings.grad_forward_rows
    live_rows = graph_rows * forward_rows
    adam_resident = settings.fsdp is None or (
        not settings.fsdp.cpu_offload and not settings.fsdp.optim_cpu_offload
    )
    adam_always = adam if adam_resident else 0.0
    recompute = formulas.block_recompute_bytes(
        arch,
        forward_rows,
        seq_len,
        act_bytes,
        backward=True,
        eager_mamba_scan=eager_mamba_scan,
    )
    if settings.gradient_checkpointing:
        saved = int(live_rows * seq_len * arch.hidden_size * arch.n_layers * act_bytes)
    else:
        saved = recompute * arch.n_layers * graph_rows
    if settings.activation_offload:
        saved = 0
    loss_hidden = live_rows * seq_len * arch.hidden_size * act_bytes
    recompute, lora_casts, lora_dropout, split_lora, routed_chunks = (
        block_backward_terms(
            arch, settings, seq_len, act_bytes, adapter_bytes, eager_mamba_scan
        )
    )
    loss_lora_casts = 0 if settings.lora_casts_recompute_only else lora_casts
    # Two fp32 tiles are live at the loss instant: recomputed logits plus
    # that tile's gradient. Tiles are fixed-size, so a fused critic row adds none.
    logit_rows = formulas.resolve_chunk_rows(arch.vocab_size, settings.chunk_rows)
    logit_tile = logit_rows * arch.vocab_size * 4
    # The chunk loop hoists an fp32 lm_head copy when the head runs below fp32.
    fused_head = int(arch.vocab_size * arch.hidden_size * 4) if act_bytes < 4 else 0
    logits = logit_rows * arch.vocab_size * 8 + fused_head
    nograd_rows = settings.n_adapter_rows
    nograd_pass = (
        formulas.nograd_forward_bytes(
            arch, nograd_rows, seq_len, act_bytes, eager_mamba_scan=eager_mamba_scan
        )
        if settings.has_nograd_pass
        else 0
    )
    backward_peak = (
        grads
        + saved
        + recompute
        + loss_hidden
        + lora_casts
        + lora_dropout
        + split_lora
        + routed_chunks
    )
    loss_peak = grads + saved + loss_hidden + loss_lora_casts + logits
    nograd_peak = (
        (nograd_pass + logit_tile + fused_head) if settings.has_nograd_pass else 0
    )
    optimizer_peak = step_grads
    backward_total = backward_peak + adam_always
    loss_total = loss_peak + adam_always
    nograd_total = nograd_peak + (adam_always if settings.has_nograd_pass else 0)
    optimizer_total = optimizer_peak + adam + adam_workspace
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
        activations, live_logits, live_grads, live_adam = (
            0,
            0,
            step_grads,
            adam + adam_workspace,
        )
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
        routed_chunks=routed_chunks,
        loss_hidden=loss_hidden,
        lora_casts=lora_casts,
        lora_dropout=lora_dropout,
        fused_head=fused_head,
    )


def trainer_base_bytes(
    model: ModelSpec,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
) -> float:
    """Frozen base weights on one training GPU, including the FSDP gather.

    :param model: Training-side model spec.
    :param settings: Training settings (dtype, FSDP).
    :param counts: Analytic parameter split.
    :param shards: GPUs one weight copy is sharded across.
    :return: Base-weight bytes on one GPU.
    """
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
    """Peak training-phase memory on the training device.

    :param model: Training-side model spec.
    :param device: Training GPU.
    :param settings: Training settings.
    :param orchestrated: Charge the per-process job overhead.
    :return: Stacked-bar breakdown for the training phase.
    """
    fsdp = settings.fsdp
    resolved_note = None
    if fsdp is not None and (
        fsdp.routed_expert_chunk_mib is None or fsdp.optim_cpu_offload is None
    ):
        resolved = resolve_fsdp_memory(model, device, settings, orchestrated)
        settings = settings.model_copy(update={"fsdp": resolved})
        resolved_note = fsdp_resolution_note(fsdp, resolved)
    arch = model.arch
    counts = formulas.param_counts(arch, model.n_params)
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    seq_len = settings.max_model_len
    warnings = training_warnings(counts, settings)
    if resolved_note:
        warnings.append(resolved_note)
    shards = settings.shard_gpus
    base = trainer_base_bytes(model, settings, counts, shards)
    adapters = adapter_state(arch, settings, shards)
    peak = activation_peak(
        arch,
        settings,
        act_bytes=act_bytes,
        adapter_bytes=adapters.bytes_per_param,
        grads=adapters.grads,
        step_grads=adapters.step_grads,
        adam=adapters.adam,
        adam_workspace=adapters.adam_workspace,
        seq_len=seq_len,
        eager_mamba_scan=arch.eager_mamba_scan_family
        and device.compute_capability == EAGER_MAMBA_SCAN_CAPABILITY,
    )
    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages.
    rollout = 6 * settings.trajectories * seq_len * 4
    job_overhead = formulas.JOB_OVERHEAD_BYTES if orchestrated else 0
    lib_overhead = formulas.TRAINER_LIB_OVERHEAD_BYTES if shards == 1 else 0
    distributed_overhead = device.nccl_bytes if shards > 1 else 0
    overhead = (
        device.context_bytes
        + rollout
        + job_overhead
        + lib_overhead
        + distributed_overhead
    )
    torch_side = (
        base
        + adapters.adapters
        + peak.live_grads
        + peak.activations
        + peak.logits
        + peak.adam
    )
    reserve = formulas.allocator_reserve_bytes(
        torch_side, n_ranks=shards, sharded=settings.fsdp is not None and shards > 1
    )
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
                "routed_expert_chunks": int(peak.routed_chunks),
                "loss_hidden_state": int(peak.loss_hidden),
                "lora_fp32_input_casts": int(peak.lora_casts),
                "lora_dropout": int(peak.lora_dropout),
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
            detail={
                "step_bytes": int(adapters.adam),
                "foreach_workspace": int(adapters.adam_workspace),
            },
            note=(
                "fp32 Adam moments. Resident for the whole step, or only "
                "during step() when FSDP offloads them. Foreach AdamW adds "
                "one fp32 temporary per parameter during step()."
            ),
        ),
        MemoryComponent(
            key="overhead",
            label="Overhead (context, rollout, runtime)",
            n_bytes=max(int(overhead), 0),
            detail={
                "cuda_context": int(device.context_bytes),
                "rollout_tensors": int(rollout),
                "job_overhead": int(job_overhead),
                "trainer_lib_overhead": int(lib_overhead),
                "distributed_trainer_overhead": int(distributed_overhead),
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
        host=training_host_components(arch, settings, adapters, rollout, act_bytes),
    )


def training_host_components(
    arch: ModelArch,
    settings: TrainingSettings,
    adapters: AdapterState,
    rollout: int,
    act_bytes: float,
) -> tuple[MemoryComponent, ...]:
    """Host RAM of one training process during learn.

    :param adapters: Adapter state on one GPU.
    :param rollout: Bytes of one update's held rollout tensors.
    :param act_bytes: Bytes per vision feature element.
    :return: Host components.
    """
    fsdp = settings.fsdp
    offloaded = fsdp is not None and bool(fsdp.optim_cpu_offload)
    # Offloaded AdamW keeps pinned copies of each shard's params and grads
    # beside its fp32 moments.
    optimizer = adapters.adam + 2 * adapters.step_grads if offloaded else 0.0
    image_pixels = 3 * arch.vision_image_size**2
    vision_cache = (
        settings.images_per_update
        * arch.vision_patches_per_image
        * arch.vision_hidden_size
        * act_bytes
    )
    snapshot_adam = (
        adapters.trained_params * formulas.ADAM_BYTES_PER_PARAM
        if settings.checkpoint_optimizer
        else 0.0
    )
    prefetched = (
        rollout + settings.images_per_update * image_pixels * 4
        if settings.async_rollout
        else 0
    )
    return (
        MemoryComponent(
            key="optimizer_offload",
            label="Offloaded AdamW (pinned)",
            n_bytes=int(optimizer),
            note="fp32 moments plus param and grad copies of this rank's shard.",
        ),
        MemoryComponent(
            key="vision_feature_cache",
            label="Vision feature cache (pinned)",
            n_bytes=int(vision_cache),
            detail={
                "images": settings.images_per_update,
                "patches_per_image": arch.vision_patches_per_image,
            },
            note="Frozen vision tower output per unique image, kept for one learn step.",
        ),
        MemoryComponent(
            key="checkpoint_snapshot",
            label="Async checkpoint snapshot (pinned)",
            n_bytes=int(adapters.trained_bytes + snapshot_adam),
            detail={
                "trainable_params": int(adapters.trained_bytes),
                "optimizer_state": int(snapshot_adam),
            },
            note="Main rank only: the gathered trained adapters, plus Adam moments when checkpoint_optimizer.",
        ),
        MemoryComponent(
            key="prefetched_rollout",
            label="Prefetched rollout batch",
            n_bytes=int(prefetched),
            note="The next update's tokens, masks and fp32 pixels while this one learns.",
        ),
    )


def fsdp_resolution_note(requested: FSDPConfig, resolved: FSDPConfig) -> str:
    """Warning naming the values picked for unset FSDP memory fields.

    :param requested: Config with ``None`` fields.
    :param resolved: Config :func:`resolve_fsdp_memory` returned.
    :return: Warning text.
    """
    picks = []
    if requested.routed_expert_chunk_mib is None:
        picks.append(f"routed_expert_chunk_mib={resolved.routed_expert_chunk_mib}")
    if requested.optim_cpu_offload is None:
        placement = "CPU offload" if resolved.optim_cpu_offload else "GPU, fused AdamW"
        picks.append(f"optim_cpu_offload={resolved.optim_cpu_offload} ({placement})")
    return "Picked from this estimate: " + ", ".join(picks) + "."


def resolve_fsdp_memory(
    model: ModelSpec,
    device: DeviceSpec,
    settings: TrainingSettings,
    orchestrated: bool = False,
) -> FSDPConfig:
    """FSDP config with unset chunk size and optimizer placement picked to fit the device.

    ``routed_expert_chunk_mib=None`` takes the largest of
    :data:`formulas.ROUTED_CHUNK_MIB_CHOICES` that fits with the optimizer
    offloaded (or at the set placement), else the smallest. At that chunk,
    ``optim_cpu_offload=None`` keeps the optimizer on the GPU when it fits
    with the :data:`formulas.MAX_UNDERPREDICTION` buffer, else offloads it.
    ``cpu_offload`` already steps on the host, so it resolves to ``False``.
    Set values are kept.

    :param model: Training-side model spec.
    :param device: Training GPU.
    :param settings: Training settings; ``fsdp`` must be set.
    :param orchestrated: Charge the per-process job overhead.
    :return: Config with every memory field set.
    """
    fsdp = settings.fsdp
    if fsdp is None:
        msg = "resolve_fsdp_memory needs settings.fsdp"
        raise ValueError(msg)

    def with_choice(chunk: int, offload: bool) -> FSDPConfig:
        return replace(fsdp, routed_expert_chunk_mib=chunk, optim_cpu_offload=offload)

    def estimate(chunk: int, offload: bool) -> PhaseBreakdown:
        trial = settings.model_copy(update={"fsdp": with_choice(chunk, offload)})
        return estimate_training(model, device, trial, orchestrated)

    if fsdp.cpu_offload:
        lightest = False
    elif fsdp.optim_cpu_offload is None:
        lightest = True
    else:
        lightest = fsdp.optim_cpu_offload
    chunk = fsdp.routed_expert_chunk_mib
    if chunk is None:
        chunk = next(
            (
                c
                for c in formulas.ROUTED_CHUNK_MIB_CHOICES
                if estimate(c, lightest).fits
            ),
            formulas.ROUTED_CHUNK_MIB_CHOICES[-1],
        )
    offload = fsdp.optim_cpu_offload
    if offload is None:
        offload = not fsdp.cpu_offload and not estimate(chunk, False).fits_with_buffer
    return with_choice(chunk, offload)


def generation_warnings(
    model: ModelSpec,
    settings: GenerationSettings,
    *,
    kv_pool: int,
    kv_demand: int,
) -> list[str]:
    """User-facing generation-phase warnings."""
    arch = model.arch
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
    kv_pool = (
        settings.kv_cache_memory_bytes
        if settings.kv_cache_memory_bytes is not None
        else max(budget - non_kv, 0)
    )
    warnings = generation_warnings(
        model,
        settings,
        kv_pool=kv_pool,
        kv_demand=kv_demand,
    )
    runtime_tokens = min(settings.concurrency * settings.prompt_len, batched_tokens)
    job_overhead = formulas.JOB_OVERHEAD_BYTES if orchestrated else 0
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
            label="Overhead (CUDA context, runtime)",
            n_bytes=max(int(device.context_bytes + job_overhead), 0),
            detail={
                "cuda_context": int(device.context_bytes),
                "job_overhead": int(job_overhead),
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
    pool = kv.detail["pool"]
    demand = kv.detail["worst_case_demand"]
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
