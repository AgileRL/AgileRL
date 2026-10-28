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

from pydantic import BaseModel, ConfigDict

import agilerl.arena.memory.formulas as formulas
import agilerl.arena.memory.vision as vision
from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown
from agilerl.arena.memory.formulas import geometry_gap_warning
from agilerl.arena.memory.generation import (
    estimate_generation,
)
from agilerl.arena.memory.generation import (
    generation_can_serve as generation_can_serve,
)
from agilerl.arena.memory.lora_placement import (
    lora_tensor_placement,
    split_moe_lora_recompute_bytes,
)
from agilerl.arena.memory.nccl import TP_RUNTIME_BYTES, trainer_nccl_bytes
from agilerl.arena.memory.resolve.chunk import resolve_chunk_rows
from agilerl.arena.memory.resolve.fsdp import resolve_fsdp_memory
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    UNFUSED_MAMBA_SCAN_CAPABILITY,
    BlockKind,
    DeviceSpec,
    MiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
)
from agilerl.arena.memory.trainer_residency import (
    trainer_allocator_slack,
    trainer_base_bytes,
)
from agilerl.arena.models.fsdp import FSDPConfig


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
        resolved = resolve_fsdp_memory(
            model, device, settings, training_breakdown, orchestrated
        )
        settings = settings.model_copy(update={"fsdp": resolved})
        resolved_note = fsdp_resolution_note(fsdp, resolved)
    breakdown = training_breakdown(model, device, settings, orchestrated)
    if resolved_note is None:
        return breakdown
    return breakdown.model_copy(
        update={"warnings": (*breakdown.warnings, resolved_note)}
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
            device_usable_bytes=device.generation_usable_bytes,
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


# Storage of an FSDP-replicated parameter, keyed by ``FSDPConfig.param_dtype``.
FSDP_PARAM_BYTES = {
    "bfloat16": 2.0,
    "float16": 2.0,
    "float32": 4.0,
    "float64": 8.0,
}


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
    if settings.sequence_packing:
        warnings.append(
            "use_sequence_packing packs the gradient forward into one row. "
            "This estimate sizes every row padded to max_model_len."
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


def _resident_bytes(
    kept: int,
    split: int,
    kept_bytes: float,
    split_bytes: float,
    shards: int,
    offload: bool,
    gather: bool = False,
) -> float:
    """Bytes of one parameter group resident on this GPU."""
    if offload:
        return kept * kept_bytes
    divisor = 1 if gather else shards
    return kept * kept_bytes + split * split_bytes / divisor


def _adam_resident_bytes(
    trained_repl: int,
    trained_shard: int,
    replicated_bytes: float,
    sharded_bytes: float,
    shards: int,
    offload: bool,
    flat_data_parallel: bool,
) -> tuple[float, float]:
    """Adam moments, and the foreach workspace when Adam steps on the GPU."""
    if offload:
        adam = 0.0
    else:
        adam = _resident_bytes(
            trained_repl,
            trained_shard,
            formulas.ADAM_MOMENTS_PER_PARAM * replicated_bytes,
            formulas.ADAM_MOMENTS_PER_PARAM * sharded_bytes,
            shards,
            offload,
        )
    # Flat data parallel steps foreach AdamW on the GPU. FSDP steps fused
    # AdamW on the GPU, or steps on the host when offloaded.
    workspace = (
        _resident_bytes(
            trained_repl,
            trained_shard,
            formulas.FOREACH_ADAM_WORKSPACE_BYTES_PER_PARAM,
            formulas.FOREACH_ADAM_WORKSPACE_BYTES_PER_PARAM,
            shards,
            offload,
        )
        if flat_data_parallel
        else 0.0
    )
    return adam, workspace


def _adapter_state_bytes(
    settings: TrainingSettings,
    replicated: int,
    sharded: int,
    head_repl: int,
    head_shard: int,
    replicated_bytes: float,
    sharded_bytes: float,
    shards: int,
    offload: bool,
    held_grad_bytes: float | None,
) -> AdapterState:
    """Adapter, gradient, and Adam bytes for one placement and dtype.

    :param held_grad_bytes: Bytes per element of the unsharded grads held
        between micro-batches, or ``None`` when every backward reduce-scatters.
    """
    n_resident = settings.n_resident_adapters
    n_trained = settings.n_trained_adapters
    adapters = _resident_bytes(
        replicated * n_resident,
        sharded * n_resident,
        replicated_bytes,
        sharded_bytes,
        shards,
        offload,
    )
    adapters += _resident_bytes(
        head_repl, head_shard, replicated_bytes, sharded_bytes, shards, offload
    )
    trained_repl = replicated * n_trained + head_repl
    trained_shard = sharded * n_trained + head_shard
    grads = _resident_bytes(
        trained_repl,
        trained_shard,
        replicated_bytes,
        sharded_bytes if held_grad_bytes is None else held_grad_bytes,
        shards,
        offload,
        gather=held_grad_bytes is not None,
    )
    step_grads = _resident_bytes(
        trained_repl,
        trained_shard,
        replicated_bytes,
        sharded_bytes,
        shards,
        offload,
    )
    adam, adam_workspace = _adam_resident_bytes(
        trained_repl,
        trained_shard,
        replicated_bytes,
        sharded_bytes,
        shards,
        offload,
        settings.fsdp is None,
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


def _with_expert_lora(
    state: AdapterState,
    settings: TrainingSettings,
    expert_lora: int,
    sharded_bytes: float,
    ep: int,
) -> AdapterState:
    """``state`` plus expert LoRA split across the EP group, outside FSDP."""
    n_trained = settings.n_trained_adapters
    expert_lora_bytes = expert_lora * sharded_bytes / ep
    trained_expert_lora = expert_lora * n_trained
    return replace(
        state,
        adapters=state.adapters + expert_lora_bytes * settings.n_resident_adapters,
        grads=state.grads + expert_lora_bytes * n_trained,
        step_grads=state.step_grads + expert_lora_bytes * n_trained,
        adam=state.adam
        + trained_expert_lora * formulas.ADAM_MOMENTS_PER_PARAM * sharded_bytes / ep,
        trained_params=state.trained_params + trained_expert_lora,
        trained_bytes=state.trained_bytes + trained_expert_lora * sharded_bytes,
    )


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
    ep = 1 if fsdp is None else fsdp.ep
    # Expert LoRA under EP splits across the EP group only, outside FSDP.
    expert_lora = (
        formulas.packed_lora_param_count(
            arch, settings.lora_rank, settings.lora_packed_target_matrices
        )
        if ep > 1
        else 0
    )
    replicated, sharded = lora_tensor_placement(
        arch,
        settings.lora_rank,
        settings.lora_target_scope,
        0 if ep > 1 else settings.lora_packed_target_matrices,
        threshold,
    )
    head_repl, head_shard = (
        _value_head_placement(arch.hidden_size, threshold)
        if settings.uses_critic
        else (0, 0)
    )
    offload = fsdp is not None and fsdp.cpu_offload
    held_grad_bytes = None
    if fsdp is None:
        replicated_bytes = formulas.ADAPTER_BYTES_PER_PARAM
        sharded_bytes = formulas.ADAPTER_BYTES_PER_PARAM
    else:
        replicated_bytes = FSDP_PARAM_BYTES[fsdp.param_dtype]
        sharded_bytes = DTYPE_BYTES[settings.weight_dtype]
        if (
            fsdp.defer_grad_sync
            and shards > 1
            and settings.trajectories > 1
            and not offload
        ):
            # FSDP2 accumulates held unsharded grads in the reduce dtype.
            held_grad_bytes = FSDP_PARAM_BYTES[fsdp.reduce_dtype]
    state = _adapter_state_bytes(
        settings,
        replicated,
        sharded,
        head_repl,
        head_shard,
        replicated_bytes,
        sharded_bytes,
        shards,
        offload,
        held_grad_bytes,
    )
    if not expert_lora:
        return state
    return _with_expert_lora(state, settings, expert_lora, sharded_bytes, ep)


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
    router_topk: float
    recompute: float
    split_lora: float
    routed_chunks: float
    ep_exchange: float
    loss_hidden: float
    lora_casts: float
    lora_dropout: float
    fused_head: int
    tower_lora: float
    nograd_pixels: float
    balance_peak: float
    head_grad: int
    tp_head_gather: int


def training_tensor_parallel(arch: ModelArch, fsdp: FSDPConfig | None) -> int:
    """Degree the trainer shards ``arch``'s decoder layers by; 1 for families without a plan."""
    if fsdp is None or not arch.tensor_parallel_plan:
        return 1
    return fsdp.tp


def block_backward_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    seq_len: int,
    act_bytes: float,
    adapter_bytes: float,
    unfused_mamba_scan: bool = False,
) -> tuple[int, int, int, int, int, int]:
    """Recompute, fp32 casts, LoRA dropout, split expert-LoRA, routed chunks and EP exchange of one checkpointed block.

    When :attr:`ModelArch.sublayer_transients` holds, the recompute peaks in
    one attention, FFN or recurrent sub-layer, so each kind's terms are
    summed and the largest kind is kept. A mixer kind on a stack whose
    layers carry an FFN beside it includes that FFN's terms.
    """
    rows = settings.grad_forward_rows
    # PEFT casts a wrapped linear's input only when the adapters are wider.
    casts_apply = adapter_bytes > act_bytes
    dropout_applies = settings.lora_dropout > 0
    split = split_moe_lora_recompute_bytes(
        arch,
        rows,
        seq_len,
        settings.lora_packed_target_matrices,
        settings.packed_moe_dispatch,
        act_bytes,
        settings.lora_rank,
    )
    fsdp = settings.fsdp
    chunk_bytes = (
        formulas.DEFAULT_ROUTED_CHUNK_BYTES
        if fsdp is None
        else fsdp.routed_expert_chunk_mib * MiB
    )
    expert_lora = settings.lora_packed_target_matrices > 0
    # Under expert parallel the TP ranks of a sequence route a share of its
    # tokens each, whether or not the family has a TP plan.
    routed_split = 1 if fsdp is None or fsdp.ep <= 1 else fsdp.tp
    chunks = formulas.routed_chunk_bytes(
        arch,
        rows,
        -(-seq_len // routed_split),
        act_bytes,
        chunk_bytes,
        expert_lora=expert_lora,
        contracted=settings.packed_moe_dispatch == "contracted",
        ep=1 if fsdp is None else fsdp.ep,
    )
    # EP splits tokens across the TP group whether or not the family has a plan.
    exchange = formulas.ep_exchange_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        1 if fsdp is None else fsdp.ep,
        1 if fsdp is None else fsdp.tp,
    )
    tp = training_tensor_parallel(arch, fsdp)

    def terms(layer: BlockKind | None) -> tuple[int, int, int, int, int, int]:
        recompute = formulas.block_recompute_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            backward=True,
            layer=layer,
            unfused_mamba_scan=unfused_mamba_scan,
            tensor_parallel=tp,
            routed_split=routed_split,
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
                tensor_parallel=tp,
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
                tensor_parallel=tp,
            )
        moe_here = layer in (None, "ffn") or not arch.block_exclusive_layers
        return (
            recompute,
            casts,
            dropout,
            split if moe_here else 0,
            chunks if moe_here else 0,
            exchange if moe_here else 0,
        )

    if not (arch.sublayer_transients and settings.gradient_checkpointing):
        return terms(None)
    return max((terms(kind) for kind in formulas.block_kinds(arch)), key=sum)


@dataclass(frozen=True)
class ActivationTerms:
    """Checkpoint, logit, and no-grad bytes inside one training step."""

    saved: float
    router_topk: float
    recompute: float
    split_lora: float
    routed_chunks: float
    ep_exchange: float
    loss_hidden: float
    lora_casts: float
    lora_dropout: float
    logits: float
    fused_head: int
    nograd_logits: float
    head_grad: int
    tp_head_gather: int
    nograd_pass: float
    nograd_hidden: float
    nograd_pixels: float
    balance_pixels: float
    loss_lora_casts: float
    tower_lora: float


@dataclass(frozen=True)
class BindingInstant:
    """Resident bytes of the instant that sets the training peak."""

    activations: float
    logits: float
    live_grads: float
    adam: float
    backward_peak: float
    loss_peak: float
    nograd_peak: float
    optimizer_peak: float


def _vision_activation_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    adapters: AdapterState,
    act_bytes: float,
    live_rows: int,
) -> tuple[float, float, float]:
    """Tower LoRA activations, no-grad vision row copies, and balanced segment rows."""
    if not settings.images_per_row:
        return 0, 0, 0
    rows, cols = vision.vision_patch_grid(
        arch, settings.image_width, settings.image_height
    )
    row_pixels = settings.images_per_row * vision.vision_pixel_bytes(arch, rows * cols)
    tower_lora = (
        vision.vision_tower_lora_bytes(
            arch,
            live_rows * settings.images_per_row * rows * cols,
            act_bytes,
            adapters.bytes_per_param,
            settings.lora_dropout > 0,
        )
        if settings.vision_tower_lora
        else 0
    )
    # The no-grad pass copies each chunk's vision rows one chunk ahead.
    nograd_pixels = (
        2 * settings.micro_batch_size * row_pixels if settings.has_nograd_pass else 0
    )
    # Balancing sends one row to and receives one row from each peer per round.
    balance_pixels = (
        2 * (settings.shard_gpus - 1) * row_pixels if settings.segmented_rows else 0
    )
    return tower_lora, nograd_pixels, balance_pixels


def _logit_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
) -> tuple[float, int, float, int, int]:
    """Loss-instant logits, the hoisted fp32 head, no-grad head bytes, the TP head gather and the weight-dtype head grad."""
    # Two fp32 tiles are live at the loss instant: recomputed logits plus
    # that tile's gradient. Token-level tiles are fixed-size, so a fused
    # critic row adds none.
    logit_rows = resolve_chunk_rows(arch.vocab_size, settings.chunk_rows)
    # A sequence-chunked Liger loss tiles one whole row per graph: DPO's
    # chosen and rejected rows of a pair are one chunk.
    loss_tile = (
        settings.grad_graph_rows
        * settings.max_model_len
        * arch.vocab_size
        * formulas.LIGER_SEQUENCE_LOGIT_BYTES
        if settings.liger_sequence_chunks
        else logit_rows * arch.vocab_size * 8
    )
    # The chunk loop hoists an fp32 lm_head copy when the head runs below fp32.
    fused_head = int(arch.vocab_size * arch.hidden_size * 4) if act_bytes < 4 else 0
    tp_head_gather = tensor_parallel_head_gather_bytes(arch, settings, act_bytes)
    head_grad = weight_dtype_head_grad(arch, settings, act_bytes)
    logits = loss_tile + fused_head + tp_head_gather + head_grad
    nograd_logits = logit_rows * arch.vocab_size * 4 + fused_head + tp_head_gather
    if settings.sampling_support_width:
        support_rows = formulas.support_rows_bytes(
            arch, logit_rows, settings.sampling_support_width, act_bytes
        )
        nograd_logits = support_rows
        if settings.support_scored_loss:
            logits = support_rows
            fused_head = 0
    return logits, fused_head, nograd_logits, tp_head_gather, head_grad


def _activation_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    adapters: AdapterState,
    act_bytes: float,
    seq_len: int,
    live_rows: int,
    graph_rows: int,
    forward_rows: int,
    unfused_mamba_scan: bool = False,
) -> ActivationTerms:
    """Checkpoint boundaries, block recompute, logits, and the no-grad pass."""
    tp = training_tensor_parallel(arch, settings.fsdp)
    recompute = formulas.block_recompute_bytes(
        arch,
        forward_rows,
        seq_len,
        act_bytes,
        backward=True,
        unfused_mamba_scan=unfused_mamba_scan,
        tensor_parallel=tp,
    )

    def checkpoint_saved() -> tuple[float, float]:
        if settings.gradient_checkpointing:
            saved = int(
                live_rows * seq_len * arch.hidden_size * arch.n_layers * act_bytes
            )
        else:
            saved = recompute * arch.n_layers * graph_rows
        if settings.activation_offload:
            saved = 0
        # FSDP selective checkpointing stores aten.topk in its cache.
        router_topk = 0
        if settings.gradient_checkpointing and settings.fsdp is not None:
            router_topk = formulas.saved_router_topk_bytes(arch, live_rows, seq_len)
        return saved, router_topk

    saved, router_topk = checkpoint_saved()
    loss_hidden = live_rows * seq_len * arch.hidden_size * act_bytes
    recompute, lora_casts, lora_dropout, split_lora, routed_chunks, ep_exchange = (
        block_backward_terms(
            arch,
            settings,
            seq_len,
            act_bytes,
            adapters.bytes_per_param,
            unfused_mamba_scan,
        )
    )
    loss_lora_casts = 0 if settings.lora_casts_recompute_only else lora_casts
    logits, fused_head, nograd_logits, tp_head_gather, head_grad = _logit_terms(
        arch, settings, act_bytes
    )

    def nograd_pass_bytes() -> float:
        if not settings.has_nograd_pass:
            return 0
        fsdp = settings.fsdp
        return formulas.nograd_forward_bytes(
            arch,
            settings.micro_batch_size,
            seq_len,
            act_bytes,
            unfused_mamba_scan=unfused_mamba_scan,
            tensor_parallel=tp,
            routed_split=1 if fsdp is None or fsdp.ep <= 1 else fsdp.tp,
        )

    nograd_pass = nograd_pass_bytes()
    nograd_hidden = settings.micro_batch_size * seq_len * arch.hidden_size * act_bytes
    tower_lora, nograd_pixels, balance_pixels = _vision_activation_terms(
        arch, settings, adapters, act_bytes, live_rows
    )
    return ActivationTerms(
        saved=saved,
        router_topk=router_topk,
        recompute=recompute,
        split_lora=split_lora,
        routed_chunks=routed_chunks,
        ep_exchange=ep_exchange,
        loss_hidden=loss_hidden,
        lora_casts=lora_casts,
        lora_dropout=lora_dropout,
        logits=logits,
        fused_head=fused_head,
        nograd_logits=nograd_logits,
        head_grad=head_grad,
        tp_head_gather=tp_head_gather,
        nograd_pass=nograd_pass,
        nograd_hidden=nograd_hidden,
        nograd_pixels=nograd_pixels,
        balance_pixels=balance_pixels,
        loss_lora_casts=loss_lora_casts,
        tower_lora=tower_lora,
    )


def _binding_instant(
    terms: ActivationTerms,
    settings: TrainingSettings,
    adapters: AdapterState,
    grads: float,
    step_grads: float,
    adam: float,
    adam_always: float,
) -> BindingInstant:
    """Resident bytes of whichever instant sets the training peak."""
    backward_peak = (
        grads
        + terms.saved
        + terms.router_topk
        + terms.recompute
        + terms.loss_hidden
        + terms.lora_casts
        + terms.lora_dropout
        + terms.split_lora
        + terms.routed_chunks
        + terms.tower_lora
        + terms.ep_exchange
    )
    loss_peak = (
        grads
        + terms.saved
        + terms.router_topk
        + terms.loss_hidden
        + terms.loss_lora_casts
        + terms.tower_lora
        + terms.logits
    )
    # Block transients are freed before the head scores the final hidden state.
    nograd_head = terms.nograd_hidden + terms.nograd_logits
    nograd_peak = (
        max(terms.nograd_pass, nograd_head) + terms.nograd_pixels
        if settings.has_nograd_pass
        else 0
    )
    optimizer_peak = step_grads
    if terms.nograd_pass >= nograd_head:
        nograd_live = (terms.nograd_pass + terms.nograd_pixels, 0, 0, adam_always)
    else:
        nograd_live = (
            terms.nograd_hidden + terms.nograd_pixels,
            terms.nograd_logits,
            0,
            adam_always,
        )
    # (total, (activations, logits, grads, adam)); a tie goes to the earlier instant.
    instants = (
        (backward_peak + adam_always, (backward_peak - grads, 0, grads, adam_always)),
        (
            loss_peak + adam_always,
            (loss_peak - terms.logits - grads, terms.logits, grads, adam_always),
        ),
        (nograd_peak + (adam_always if settings.has_nograd_pass else 0), nograd_live),
        (terms.balance_pixels + adam_always, (terms.balance_pixels, 0, 0, adam_always)),
        (
            optimizer_peak + adam + adapters.adam_workspace,
            (0, 0, step_grads, adam + adapters.adam_workspace),
        ),
    )
    _, (activations, live_logits, live_grads, live_adam) = max(
        instants, key=lambda instant: instant[0]
    )
    return BindingInstant(
        activations=activations,
        logits=live_logits,
        live_grads=live_grads,
        adam=live_adam,
        backward_peak=backward_peak,
        loss_peak=loss_peak,
        nograd_peak=nograd_peak,
        optimizer_peak=optimizer_peak,
    )


def tensor_parallel_head_gather_bytes(
    arch: ModelArch, settings: TrainingSettings, act_bytes: float
) -> int:
    """Dense copy of a tensor-parallel lm_head the logprob loop gathers, or 0."""
    if arch.tied_embeddings or settings.fsdp is None or settings.fsdp.tp <= 1:
        return 0
    return int(arch.vocab_size * arch.hidden_size * act_bytes)


def weight_dtype_head_grad(
    arch: ModelArch, settings: TrainingSettings, act_bytes: float
) -> int:
    """Bytes of the fused kernel's weight-dtype lm_head gradient, or 0.

    The sequence tile already includes this buffer. At fp32 the kernel's grad
    is the head copy charged separately.
    """
    if (
        settings.liger_sequence_chunks
        or not settings.liger_weight_grad
        or act_bytes >= 4
    ):
        return 0
    return int(arch.vocab_size * arch.hidden_size * act_bytes)


def activation_peak(
    arch: ModelArch,
    settings: TrainingSettings,
    adapters: AdapterState,
    act_bytes: float,
    seq_len: int,
    unfused_mamba_scan: bool = False,
) -> ActivationPeak:
    """Peak of backward, fused loss, no-grad logprob, and optimizer instants.

    :param adapters: Adapter, gradient, and Adam bytes on one training GPU.
    :param unfused_mamba_scan: Mamba layers run HF's unfused path.
    """
    grads = adapters.grads
    step_grads = adapters.step_grads
    adam = adapters.adam
    graph_rows = settings.grad_graph_rows
    forward_rows = settings.grad_forward_rows
    live_rows = graph_rows * forward_rows
    adam_resident = settings.fsdp is None or (
        not settings.fsdp.cpu_offload and not settings.fsdp.optim_cpu_offload
    )
    adam_always = adam if adam_resident else 0.0
    terms = _activation_terms(
        arch,
        settings,
        adapters,
        act_bytes,
        seq_len,
        live_rows,
        graph_rows,
        forward_rows,
        unfused_mamba_scan,
    )
    instant = _binding_instant(
        terms,
        settings,
        adapters,
        grads,
        step_grads,
        adam,
        adam_always,
    )
    return ActivationPeak(
        activations=instant.activations,
        logits=instant.logits,
        live_grads=instant.live_grads,
        adam=instant.adam,
        backward_peak=instant.backward_peak,
        loss_peak=instant.loss_peak,
        nograd_peak=instant.nograd_peak,
        optimizer_peak=instant.optimizer_peak,
        saved=terms.saved,
        router_topk=terms.router_topk,
        recompute=terms.recompute,
        split_lora=terms.split_lora,
        routed_chunks=terms.routed_chunks,
        ep_exchange=terms.ep_exchange,
        loss_hidden=terms.loss_hidden,
        lora_casts=terms.lora_casts,
        lora_dropout=terms.lora_dropout,
        fused_head=terms.fused_head,
        tower_lora=terms.tower_lora,
        nograd_pixels=terms.nograd_pixels,
        balance_peak=terms.balance_pixels,
        head_grad=terms.head_grad,
        tp_head_gather=terms.tp_head_gather,
    )


def _vision_row_staging_bytes(settings: TrainingSettings, image_pixels: float) -> float:
    """Two pinned buffers, each grown to the largest no-grad chunk's vision rows."""
    if not (settings.images_per_update and settings.has_nograd_pass):
        return 0
    chunk = settings.micro_batch_size * settings.images_per_row * image_pixels
    return 2 * vision.pinned_block_bytes(chunk)


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
    rows, cols = vision.vision_patch_grid(
        arch, settings.image_width, settings.image_height
    )
    images = settings.images_per_update
    image_pixels = vision.vision_pixel_bytes(arch, rows * cols)
    vision_cache = (
        images * vision.vision_feature_cache_bytes(arch, rows * cols, act_bytes)
        if arch.vision_feature_cache and not settings.vision_tower_lora
        else 0
    )
    staging = _vision_row_staging_bytes(settings, image_pixels)
    snapshot_adam = (
        adapters.trained_bytes * formulas.ADAM_MOMENTS_PER_PARAM
        if settings.checkpoint_optimizer
        else 0.0
    )
    prefetched = rollout + images * image_pixels if settings.async_rollout else 0
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
            detail={"images": images, "patches_per_image": rows * cols},
            note="Frozen vision tower output per unique image, kept for one learn step.",
        ),
        MemoryComponent(
            key="vision_rows",
            label="Vision rows",
            n_bytes=int(images * image_pixels),
            detail={"images": images, "patches_per_image": rows * cols},
            note="This update's fp32 pixels, one host tensor every adapter run reads.",
        ),
        MemoryComponent(
            key="vision_row_staging",
            label="Vision row staging (pinned)",
            n_bytes=int(staging),
            note="Two page-locked buffers no-grad chunks' vision rows pass through to the GPU.",
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
            detail={"images": images, "patches_per_image": rows * cols},
            note="The next update's tokens, masks and fp32 pixels while this one learns.",
        ),
    )


def _model_state_components(
    base: float,
    adapters: AdapterState,
    peak: ActivationPeak,
) -> tuple[MemoryComponent, ...]:
    """Frozen weights, resident adapters, and live gradients."""
    return (
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
    )


def _activation_workspace_components(
    peak: ActivationPeak,
    adapters: AdapterState,
) -> tuple[MemoryComponent, ...]:
    """Activation peak and the chunked logit workspace."""
    return (
        MemoryComponent(
            key="activations",
            label="Activations",
            n_bytes=max(int(peak.activations), 0),
            detail={
                "backward_peak": int(peak.backward_peak),
                "loss_peak": int(peak.loss_peak),
                "nograd_peak": int(peak.nograd_peak),
                "segment_balance_peak": int(peak.balance_peak),
                "optimizer_peak": int(peak.optimizer_peak),
                "checkpoint_boundaries": int(peak.saved),
                "saved_router_topk": int(peak.router_topk),
                "block_recompute": int(peak.recompute),
                "split_moe_lora": int(peak.split_lora),
                "routed_expert_chunks": int(peak.routed_chunks),
                "ep_token_exchange": int(peak.ep_exchange),
                "loss_hidden_state": int(peak.loss_hidden),
                "lora_fp32_input_casts": int(peak.lora_casts),
                "lora_dropout": int(peak.lora_dropout),
                "vision_tower_lora": int(peak.tower_lora),
                "nograd_vision_rows": int(peak.nograd_pixels),
                "backward_grads": int(adapters.grads),
                "step_grads": int(adapters.step_grads),
            },
            note=(
                "Peak of the gradient micro-batch pass vs one no-grad logprob "
                "micro-batch forward and segment-row balancing."
            ),
        ),
        MemoryComponent(
            key="logits_workspace",
            label="Logit workspace (chunked)",
            n_bytes=max(int(peak.logits), 0),
            detail=(
                {
                    "chunk_tiles": max(
                        int(peak.logits)
                        - peak.fused_head
                        - peak.head_grad
                        - peak.tp_head_gather,
                        0,
                    ),
                    "fp32_head_upcast": peak.fused_head,
                    "weight_dtype_head_grad": peak.head_grad,
                    "tp_head_gather": peak.tp_head_gather,
                }
                if peak.logits
                and (peak.fused_head or peak.head_grad or peak.tp_head_gather)
                else {}
            ),
            note=(
                "The fused logprob path tiles the lm_head matmul."
                + (
                    " Includes the fp32 lm_head copy the chunk loop hoists."
                    if peak.fused_head
                    else ""
                )
                + (
                    " Includes the dense copy of the tensor-parallel lm_head."
                    if peak.tp_head_gather
                    else ""
                )
            ),
        ),
    )


def _runtime_components(
    peak: ActivationPeak,
    adapters: AdapterState,
    context_bytes: int,
    rollout: int,
    overhead: int,
    reserve: float,
    gather_pool: float,
    job_overhead: int,
    lib_overhead: int,
    distributed_overhead: int,
) -> tuple[MemoryComponent, ...]:
    """Adam state, process overhead, and caching-allocator slack."""
    return (
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
                "cuda_context": int(context_bytes),
                "rollout_tensors": int(rollout),
                "job_overhead": int(job_overhead),
                "trainer_lib_overhead": int(lib_overhead),
                "distributed_trainer_overhead": int(distributed_overhead),
            },
        ),
        MemoryComponent(
            key="allocator_reserve",
            label="Caching-allocator slack",
            n_bytes=max(int(reserve + gather_pool), 0),
            detail={
                "compute_stream": int(reserve),
                "all_gather_pool": int(gather_pool),
            },
            note=(
                "PyTorch reserves more than it allocates: segment rounding plus "
                "blocks it cannot reuse for a differently-shaped request. FSDP's "
                "all-gather stream keeps a pool of its own."
            ),
        ),
    )


def trainer_distributed_bytes(
    device: DeviceSpec, settings: TrainingSettings, shards: int
) -> int:
    """NCCL buffers and the tensor-parallel runtime on one rank of a multi-rank trainer.

    :param shards: GPUs one weight copy is sharded across.
    """
    fsdp = settings.fsdp
    tp = 1 if fsdp is None else fsdp.tp
    nccl = trainer_nccl_bytes(
        device.nccl_group_bytes,
        settings.n_training_gpus,
        shards,
        1 if fsdp is None else fsdp.ep,
        tp,
        device.gpus_per_node,
    )
    return nccl + (TP_RUNTIME_BYTES if tp > 1 else 0)


def training_breakdown(
    model: ModelSpec,
    device: DeviceSpec,
    settings: TrainingSettings,
    orchestrated: bool = False,
) -> PhaseBreakdown:
    """Peak training-phase memory for one device.

    :param model: Training-side model spec.
    :param device: Training GPU.
    :param settings: Training settings. When ``fsdp`` is set, its chunk size
        and optimizer placement are set.
    :param orchestrated: Charge the per-process job overhead.
    :return: Stacked-bar breakdown for the training phase.
    """
    arch = model.arch
    counts = formulas.param_counts(arch, model.n_params)
    act_bytes = DTYPE_BYTES[settings.weight_dtype]
    seq_len = settings.max_model_len
    warnings = training_warnings(counts, settings)
    shards = settings.shard_gpus
    adapters = adapter_state(arch, settings, shards)
    peak = activation_peak(
        arch,
        settings,
        adapters,
        act_bytes=act_bytes,
        seq_len=seq_len,
        unfused_mamba_scan=arch.unfused_mamba_scan_family
        and device.compute_capability == UNFUSED_MAMBA_SCAN_CAPABILITY,
    )
    base = trainer_base_bytes(model, settings, counts, shards, peak.activations)
    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages.
    rollout = 6 * settings.trajectories * seq_len * 4
    job_overhead = formulas.JOB_OVERHEAD_BYTES if orchestrated else 0
    lib_overhead = formulas.TRAINER_LIB_OVERHEAD_BYTES if shards == 1 else 0
    distributed_overhead = (
        trainer_distributed_bytes(device, settings, shards) if shards > 1 else 0
    )
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
    reserve, gather_pool = trainer_allocator_slack(
        arch, settings, counts, shards, torch_side
    )
    components = (
        _model_state_components(base, adapters, peak)
        + _activation_workspace_components(peak, adapters)
        + _runtime_components(
            peak,
            adapters,
            device.context_bytes,
            rollout,
            overhead,
            reserve,
            gather_pool,
            job_overhead,
            lib_overhead,
            distributed_overhead,
        )
    )
    return PhaseBreakdown(
        phase="training",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        warnings=tuple(warnings),
        host=training_host_components(arch, settings, adapters, rollout, act_bytes),
    )
