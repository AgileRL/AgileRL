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

import agilerl.arena.memory.block_activations as block_activations
import agilerl.arena.memory.formulas as formulas
import agilerl.arena.memory.lora_formulas as lora_formulas
import agilerl.arena.memory.vision as vision
from agilerl.arena.memory.breakdown import MemoryComponent, PhaseBreakdown
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
from agilerl.arena.memory.nccl import trainer_nccl_bytes
from agilerl.arena.memory.resolve.mamba import mamba_scan_is_unfused
from agilerl.arena.memory.resolve.moe import (
    expert_gemm_backend,
    grouped_gemm_supported,
    padded_expert_lora_rank,
)
from agilerl.arena.memory.resolve.rollout import merged_sync_bucket_bytes
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    BlockKind,
    DeviceSpec,
    MiB,
    ModelArch,
    ModelSpec,
    RunConfig,
    TrainingSettings,
)
from agilerl.arena.memory.trainer_residency import (
    fsdp_cpu_offload_host_bytes,
    fsdp_forward_gather_bytes,
    trainer_allocator_slack,
    trainer_base_bytes,
)
from agilerl.arena.memory.training_warnings import training_warnings
from agilerl.arena.models.fsdp import (
    FSDP_MEMORY_FIELDS,
    PARAM_PERSISTENCE_THRESHOLD,
    FSDPConfig,
)


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
    :param settings: Training settings, with checkpointing and every FSDP
        memory field set.
    :param orchestrated: Charge the per-process job overhead.
    :return: Stacked-bar breakdown for the training phase.
    :raises ValueError: When checkpointing or an FSDP memory field is unset,
        or when an MoE model sets ``ep < tp``.
    """
    fsdp = settings.fsdp
    if fsdp is not None and fsdp.ep < fsdp.tp and model.arch.is_moe:
        msg = (
            f"FSDPConfig(ep={fsdp.ep}, tp={fsdp.tp}): MoE models need "
            "ep >= tp, since tensor parallel does not split routed experts."
        )
        raise ValueError(msg)
    unset = (
        []
        if settings.gradient_checkpointing is not None
        else ["gradient_checkpointing"]
    )
    if fsdp is not None:
        unset += [
            f"fsdp.{name}" for name in FSDP_MEMORY_FIELDS if getattr(fsdp, name) is None
        ]
    if unset:
        msg = (
            f"{', '.join(unset)} unset; resolve_training_memory picks them "
            "before the estimate."
        )
        raise ValueError(msg)
    return training_breakdown(model, device, settings, orchestrated)


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


@dataclass(frozen=True)
class AdapterState:
    """LoRA adapter, gradient, and Adam bytes on one training GPU."""

    # Bytes per adapter element in the forward; LoRA inputs are cast to it
    # when it is wider than the activations.
    compute_bytes: float
    adapters: float
    grads: float
    step_grads: float
    adam: float
    # Trained adapters and value head, unsharded.
    trained_params: int
    trained_bytes: float


def _resident_bytes(
    kept: int,
    split: int,
    kept_bytes: float,
    split_bytes: float,
    shards: int,
    offload: bool,
) -> float:
    """Bytes of one parameter group resident on this GPU."""
    if offload:
        return kept * kept_bytes
    return kept * kept_bytes + split * split_bytes / shards


def _fsdp_lora_placement(
    arch: ModelArch, settings: TrainingSettings, fsdp: FSDPConfig, packed: int
) -> tuple[int, int, int]:
    """One adapter's elements on one FSDP rank: all of them, then (replicated, sharded).

    :param packed: Packed-expert LoRA matrices FSDP shards with the dense weights.
    """
    tp = training_tensor_parallel(arch, fsdp)
    rank, scope = settings.lora_rank, settings.lora_target_scope
    tower = settings.tower_lora_targets
    per_rank, _ = lora_tensor_placement(arch, rank, scope, packed, None, tp, tower)
    repl, shard = lora_tensor_placement(
        arch, rank, scope, packed, PARAM_PERSISTENCE_THRESHOLD, tp, tower
    )
    return per_rank, repl, shard


def adapter_state(
    arch: ModelArch, settings: TrainingSettings, shards: int
) -> AdapterState:
    """Adapter copies, gradients, and fp32 Adam moments on one GPU.

    Trained adapters and the value head are fp32 on both paths. Without FSDP
    the fused LoRA forward runs them in the base weight dtype; FSDP shards
    every trainable parameter and runs it in ``param_dtype``. The frozen
    reference keeps FSDP's persistence split: small matrices replicated in
    ``param_dtype``, the rest sharded in fp32. Expert LoRA under EP splits
    across the EP group only, outside FSDP. Tensor parallel keeps ``1/tp``
    of every LoRA factor its plan splits on each rank. Tower LoRA adds the
    tower and connector adapters its targets name to every resident adapter.

    :param arch: Decoder geometry.
    :param settings: Training settings (rank, scope, FSDP).
    :param shards: GPUs one dense weight copy is sharded across; 1 means no FSDP shard.
    :return: Adapter-state bytes on one GPU.
    """
    fsdp = settings.fsdp
    ep = 1 if fsdp is None else fsdp.ep
    rank = settings.lora_rank
    packed = settings.lora_packed_target_matrices
    expert_lora = (
        lora_formulas.packed_lora_param_count(arch, rank, packed) if ep > 1 else 0
    )
    dense_packed = 0 if ep > 1 else packed
    per_adapter = lora_formulas.lora_param_count(
        arch, rank, settings.lora_target_scope
    ) + lora_formulas.packed_lora_param_count(arch, rank, dense_packed)
    per_adapter += lora_formulas.tower_lora_param_count(
        arch, rank, settings.tower_lora_targets
    )
    n_trained = settings.n_trained_adapters
    n_frozen = settings.n_resident_adapters - n_trained
    head = arch.hidden_size + 1 if settings.uses_critic else 0
    trained = per_adapter * n_trained + head
    stored = formulas.ADAPTER_BYTES_PER_PARAM
    offload = fsdp is not None and bool(fsdp.cpu_offload)
    held_grad_bytes = None
    if fsdp is None:
        compute_bytes = DTYPE_BYTES[settings.weight_dtype]
        frozen_repl_bytes = stored
        trained_repl, trained_shard = trained, 0
        frozen_repl, frozen_shard = per_adapter * n_frozen, 0
    else:
        compute_bytes = FSDP_PARAM_BYTES[fsdp.param_dtype]
        frozen_repl_bytes = compute_bytes
        per_rank_adapter, repl, shard = _fsdp_lora_placement(
            arch, settings, fsdp, dense_packed
        )
        trained_repl, trained_shard = 0, per_rank_adapter * n_trained + head
        frozen_repl, frozen_shard = repl * n_frozen, shard * n_frozen
        if shards > 1 and settings.trajectories > 1:
            # Without gradient sync (agilerl/distributed/runtime.py
            # set_requires_gradient_sync) FSDP2 accumulates unsharded grads in
            # the reduce dtype on the GPU, with or without cpu_offload.
            held_grad_bytes = FSDP_PARAM_BYTES[fsdp.reduce_dtype]
    adapters = _resident_bytes(
        trained_repl, trained_shard, stored, stored, shards, offload
    ) + _resident_bytes(
        frozen_repl, frozen_shard, frozen_repl_bytes, stored, shards, offload
    )
    grads = (
        trained_repl * stored + trained_shard * held_grad_bytes
        if held_grad_bytes is not None
        else _resident_bytes(
            trained_repl, trained_shard, stored, stored, shards, offload
        )
    )
    step_grads = _resident_bytes(
        trained_repl, trained_shard, stored, stored, shards, offload
    )
    # Fused AdamW steps without temporaries.
    adam = 0.0 if offload else formulas.ADAM_MOMENTS_PER_PARAM * step_grads
    expert_trained = expert_lora * n_trained / ep
    adapters += expert_trained * stored + expert_lora * n_frozen * compute_bytes / ep
    grads += expert_trained * stored
    step_grads += expert_trained * stored
    adam += expert_trained * formulas.ADAM_MOMENTS_PER_PARAM * stored
    trained_params = trained + expert_lora * n_trained
    return AdapterState(
        compute_bytes=compute_bytes,
        adapters=adapters,
        grads=grads,
        step_grads=step_grads,
        adam=adam,
        trained_params=trained_params,
        trained_bytes=trained_params * stored,
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
    forward_peak: float
    # FSDP all-gather bytes the decoder forward holds beyond the gathered units.
    forward_gather: float
    # Adam bytes resident outside ``step()``.
    resident_adam: float
    saved: float
    offloaded: float
    router_topk: float
    saved_collectives: float
    recompute: float
    split_lora: float
    routed_chunks: float
    ep_exchange: float
    loss_hidden: float
    lora_casts: float
    lora_dropout: float
    tower_lora: float
    nograd_pixels: float
    balance_peak: float
    tp_head_gather: int


def training_tensor_parallel(arch: ModelArch, fsdp: FSDPConfig | None) -> int:
    """Degree the trainer shards ``arch``'s decoder layers by; 1 for families without a plan."""
    if fsdp is None or not arch.tensor_parallel_plan:
        return 1
    return fsdp.tp


def stored_lora_cast_bytes(
    arch: ModelArch, settings: TrainingSettings, seq_len: int, tensor_parallel: int
) -> int:
    """Wide LoRA input casts the layers left out of checkpointing keep until their backward.

    Each wrapped linear casts its input to the adapter compute dtype when
    that is wider than the activations.
    """
    stored = arch.n_layers - formulas.training_checkpointed_blocks(arch, settings)
    every_layer = formulas.lora_input_cast_bytes(
        arch,
        settings.grad_forward_rows,
        seq_len,
        settings.lora_target_scope,
        gradient_checkpointing=False,
        tensor_parallel=tensor_parallel,
    )
    return every_layer * stored // arch.n_layers


def block_backward_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    seq_len: int,
    act_bytes: float,
    adapter_bytes: float,
    unfused_mamba_scan: bool = False,
    grouped_gemm_bytes: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Recompute, fp32 casts, LoRA dropout, split expert-LoRA, routed chunks and EP exchange of one checkpointed block.

    When :attr:`ModelArch.sublayer_transients` holds, the recompute peaks in
    one attention, FFN or recurrent sub-layer, so each kind's terms are
    summed and the largest kind is kept. A mixer kind on a stack whose
    layers carry an FFN beside it includes that FFN's terms. With
    ``grouped_gemm_bytes`` set, the split ``x @ A`` runs at the expert rank
    padded for that element size. Blocks left out of checkpointing add the
    casts and dropout copies their forward keeps.
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
        (
            settings.lora_rank
            if grouped_gemm_bytes is None
            else padded_expert_lora_rank(settings.lora_rank, grouped_gemm_bytes)
        ),
    )
    fsdp = settings.fsdp
    chunk_mib = (
        settings.routed_expert_chunk_mib
        if fsdp is None
        else fsdp.routed_expert_chunk_mib
    )
    chunk_bytes = (
        formulas.DEFAULT_ROUTED_CHUNK_BYTES if chunk_mib is None else chunk_mib * MiB
    )
    expert_lora = settings.lora_packed_target_matrices > 0
    # Under expert parallel the TP ranks of a sequence route a share of its
    # tokens each, whether or not the family has a TP plan.
    routed_split = 1 if fsdp is None or fsdp.ep <= 1 else fsdp.tp
    checkpointed = formulas.training_checkpointed_blocks(arch, settings)
    stored_layers = arch.n_layers - checkpointed
    chunks = formulas.routed_chunk_bytes(
        arch,
        rows,
        -(-seq_len // routed_split),
        act_bytes,
        chunk_bytes,
        expert_lora=expert_lora,
        contracted=settings.packed_moe_dispatch == "contracted",
        ep=1 if fsdp is None else fsdp.ep,
        recomputed=stored_layers > 0 and arch.routed_expert_recompute,
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

    def held(per_block: int, every_layer: int) -> int:
        """One recomputed block's tensors plus those of every stored layer."""
        recomputed = per_block if checkpointed else 0
        return recomputed + every_layer * stored_layers // arch.n_layers

    def terms(layer: BlockKind | None) -> tuple[int, int, int, int, int, int]:
        recompute = block_activations.block_recompute_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            backward=True,
            layer=layer,
            unfused_mamba_scan=unfused_mamba_scan,
            tensor_parallel=tp,
            routed_split=routed_split,
            document_len=settings.max_model_len,
            context_parallel=1 if fsdp is None else fsdp.cp,
        )
        casts = 0
        if casts_apply:
            running = (
                formulas.lora_input_cast_bytes(
                    arch,
                    rows,
                    seq_len,
                    settings.lora_target_scope,
                    layer=layer,
                    tensor_parallel=tp,
                )
                if checkpointed
                else formulas.lora_fp32_output_bytes(
                    arch, rows, seq_len, settings.lora_target_scope, tp
                )
            )
            casts = running + stored_lora_cast_bytes(arch, settings, seq_len, tp)
        dropout = 0
        if dropout_applies:
            dropout = held(
                formulas.lora_dropout_bytes(
                    arch,
                    rows,
                    seq_len,
                    adapter_bytes,
                    settings.lora_target_scope,
                    layer=layer,
                    tensor_parallel=tp,
                ),
                formulas.lora_dropout_bytes(
                    arch,
                    rows,
                    seq_len,
                    adapter_bytes,
                    settings.lora_target_scope,
                    gradient_checkpointing=False,
                    tensor_parallel=tp,
                ),
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
    # Saved tensors activation offload moves to pinned host RAM.
    offloaded: float
    router_topk: float
    # Collective outputs every checkpointed MoE block saves, and the share the
    # backward peak adds beside the block in backward, whose own saved outputs
    # are its exchange buffers.
    saved_collectives: float
    backward_collectives: float
    recompute: float
    split_lora: float
    routed_chunks: float
    ep_exchange: float
    loss_hidden: float
    # Residual stream the block backward holds, every hyper-connection stream
    # wide, over a CP rank's shard of the sequence.
    block_hidden: float
    lora_casts: float
    lora_dropout: float
    logits: float
    nograd_logits: float
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
            adapters.compute_bytes,
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
        2 * (settings.shard_group_gpus - 1) * row_pixels
        if settings.segmented_rows
        else 0
    )
    return tower_lora, nograd_pixels, balance_pixels


def _logit_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
) -> tuple[float, float, int]:
    """Loss-instant logits, no-grad head bytes, and the TP head gather."""
    logits, nograd_logits = formulas.logit_terms(arch, settings, act_bytes)
    tp_head_gather = 0
    if not settings.sampling_support_width:
        tp_head_gather = tensor_parallel_head_gather_bytes(arch, settings, act_bytes)
    return logits + tp_head_gather, nograd_logits + tp_head_gather, tp_head_gather


def _fsdp_saved_moe_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
    block_seq: int,
    live_rows: int,
    tp: int,
    routed_split: int,
) -> tuple[int, int, int]:
    """Router top-k FSDP's checkpoint cache stores, and saved MoE collective outputs.

    :return: Router top-k bytes, collective outputs every checkpointed MoE
        block saves, and the share the backward peak adds beside the block
        in backward.
    """
    fsdp = settings.fsdp
    if fsdp is None:
        return 0, 0, 0
    # FSDP selective checkpointing stores aten.topk in its cache.
    router_topk = formulas.saved_router_topk_bytes(arch, live_rows, block_seq)
    moe_blocks = formulas.checkpointed_moe_blocks(arch, settings)
    if not (moe_blocks and fsdp.save_moe_collectives):
        return router_topk, 0, 0
    saved = moe_blocks * formulas.saved_moe_collective_bytes(
        arch, live_rows, block_seq, act_bytes, fsdp.ep, routed_split, tp
    )
    in_backward = formulas.saved_moe_collective_bytes(
        arch,
        settings.grad_forward_rows,
        block_seq,
        act_bytes,
        fsdp.ep,
        routed_split,
        tp,
    )
    return router_topk, saved, saved - in_backward


def _saved_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
    seq_len: int,
    live_rows: int,
    unfused_mamba_scan: bool,
    tp: int,
    routed_split: int,
    casts_apply: bool,
) -> tuple[int, int, int, int]:
    """Forward tensors every block keeps for backward, then :func:`_fsdp_saved_moe_terms`.

    :return: Checkpoint inputs or saved block tensors, router top-k bytes,
        saved MoE collective outputs, and their backward-peak share.
    """
    if block_activations.prices_stored_tensors(arch):
        block_saved = block_activations.stored_block_bytes(
            arch,
            settings.grad_forward_rows,
            seq_len,
            act_bytes,
            expert_lora=settings.lora_packed_target_matrices > 0,
            mamba_lora=settings.lora_target_scope == "all-linear",
            expert_parallel=1 if settings.fsdp is None else settings.fsdp.ep,
            routed_split=routed_split,
        )
    else:
        block_saved = (
            block_activations.mean_block_recompute_bytes(
                arch,
                settings.grad_forward_rows,
                seq_len,
                act_bytes,
                unfused_mamba_scan,
                tp,
                routed_split,
                settings.max_model_len,
                1 if settings.fsdp is None else settings.fsdp.cp,
            )
            * formulas.STORED_BLOCK_SHARE
        )
        if casts_apply or settings.lora_target_scope == "attention-only":
            block_saved -= formulas.lora_only_stored_input_bytes(
                arch, settings.grad_forward_rows, seq_len, act_bytes, tp
            )
    checkpointed = formulas.training_checkpointed_blocks(arch, settings)
    # A checkpointed block keeps its input; any other block keeps its forward's
    # saved tensors.
    stored = int(
        checkpointed * live_rows * seq_len * arch.residual_width * act_bytes
        + (arch.n_layers - checkpointed) * block_saved * settings.grad_graph_rows
    )
    return stored, *_fsdp_saved_moe_terms(
        arch, settings, act_bytes, seq_len, live_rows, tp, routed_split
    )


def _nograd_pass_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
    seq_len: int,
    unfused_mamba_scan: bool,
    tp: int,
    routed_split: int,
) -> tuple[int, float]:
    """No-grad forward block transients (0 without the pass), and its final hidden state."""
    hidden = settings.micro_batch_size * seq_len * arch.hidden_size * act_bytes
    if not settings.has_nograd_pass:
        return 0, hidden
    # The fused pass splits the stacked adapter rows into micro-batch forwards.
    block = block_activations.nograd_forward_bytes(
        arch,
        settings.micro_batch_size,
        seq_len,
        act_bytes,
        unfused_mamba_scan=unfused_mamba_scan,
        tensor_parallel=tp,
        routed_split=routed_split,
        document_len=settings.max_model_len,
        context_parallel=1 if settings.fsdp is None else settings.fsdp.cp,
    )
    return block, hidden


def _activation_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    adapters: AdapterState,
    act_bytes: float,
    seq_len: int,
    live_rows: int,
    unfused_mamba_scan: bool = False,
    grouped_gemm_bytes: int | None = None,
) -> ActivationTerms:
    """Checkpoint boundaries, block recompute, logits, and the no-grad pass.

    :param seq_len: Full sequence length. Context-parallel ranks run their
        blocks and the head on a ``1 / cp`` shard of the padded action frame
        and gather only the fp32 per-token outputs.
    """
    fsdp = settings.fsdp
    tp = training_tensor_parallel(arch, fsdp)
    cp = 1 if fsdp is None else fsdp.cp
    block_seq = formulas.context_parallel_shard_len(arch, seq_len, cp)
    block_hidden = live_rows * block_seq * arch.residual_width * act_bytes
    routed_split = 1 if fsdp is None or fsdp.ep <= 1 else fsdp.tp
    casts_apply = adapters.compute_bytes > act_bytes
    stored, router_topk, saved_collectives, backward_collectives = _saved_terms(
        arch,
        settings,
        act_bytes,
        block_seq,
        live_rows,
        unfused_mamba_scan,
        tp,
        routed_split,
        casts_apply,
    )
    recompute, lora_casts, lora_dropout, split_lora, routed_chunks, ep_exchange = (
        block_backward_terms(
            arch,
            settings,
            block_seq,
            act_bytes,
            adapters.compute_bytes,
            unfused_mamba_scan,
            grouped_gemm_bytes,
        )
    )
    logits, nograd_logits, tp_head_gather = _logit_terms(arch, settings, act_bytes)
    nograd_pass, nograd_hidden = _nograd_pass_terms(
        arch, settings, act_bytes, block_seq, unfused_mamba_scan, tp, routed_split
    )
    tower_lora, nograd_pixels, balance_pixels = _vision_activation_terms(
        arch, settings, adapters, act_bytes, live_rows
    )
    return ActivationTerms(
        saved=0 if settings.activation_offload else stored,
        offloaded=stored if settings.activation_offload else 0,
        router_topk=router_topk,
        saved_collectives=saved_collectives,
        backward_collectives=backward_collectives,
        recompute=recompute,
        split_lora=split_lora,
        routed_chunks=routed_chunks,
        ep_exchange=ep_exchange,
        loss_hidden=live_rows * block_seq * arch.hidden_size * act_bytes
        + (live_rows * seq_len * 4 if cp > 1 else 0),
        block_hidden=block_hidden,
        lora_casts=lora_casts,
        lora_dropout=lora_dropout,
        logits=logits,
        nograd_logits=nograd_logits,
        tp_head_gather=tp_head_gather,
        nograd_pass=nograd_pass,
        nograd_hidden=nograd_hidden,
        nograd_pixels=nograd_pixels,
        balance_pixels=balance_pixels,
        loss_lora_casts=(
            stored_lora_cast_bytes(arch, settings, block_seq, tp) if casts_apply else 0
        ),
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
        + terms.backward_collectives
        + terms.recompute
        + terms.block_hidden
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
        + terms.saved_collectives
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
        (optimizer_peak + adam, (0, 0, step_grads, adam)),
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
    if settings.fsdp is None or settings.fsdp.tp <= 1:
        return 0
    if arch.tied_embeddings and not arch.tensor_parallel_tied_embedding:
        return 0
    return int(arch.vocab_size * arch.hidden_size * act_bytes)


def activation_peak(
    arch: ModelArch,
    settings: TrainingSettings,
    adapters: AdapterState,
    act_bytes: float,
    seq_len: int,
    unfused_mamba_scan: bool = False,
    grouped_gemm_bytes: int | None = None,
    forward_gather: float = 0.0,
) -> ActivationPeak:
    """Peak of backward, fused loss, no-grad logprob, and optimizer instants.

    The gradient forward instant is returned beside them in ``forward_peak``;
    it contends with the base weights, so ``training_breakdown`` picks it.

    :param adapters: Adapter, gradient, and Adam bytes on one training GPU.
    :param unfused_mamba_scan: Mamba layers run HF's unfused path.
    :param grouped_gemm_bytes: Element size of the routed-expert LoRA grouped
        GEMM, or ``None`` when it does not run.
    :param forward_gather: FSDP all-gather bytes the decoder forward holds
        beyond the gathered units.
    """
    grads = adapters.grads
    step_grads = adapters.step_grads
    adam = adapters.adam
    live_rows = settings.grad_graph_rows * settings.grad_forward_rows
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
        unfused_mamba_scan,
        grouped_gemm_bytes,
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
        # The last decoder unit's gradient forward, every checkpoint input saved.
        forward_peak=grads + terms.saved + terms.router_topk + forward_gather,
        forward_gather=forward_gather,
        resident_adam=adam_always,
        saved=terms.saved,
        offloaded=terms.offloaded,
        router_topk=terms.router_topk,
        saved_collectives=terms.saved_collectives,
        recompute=terms.recompute,
        split_lora=terms.split_lora,
        routed_chunks=terms.routed_chunks,
        ep_exchange=terms.ep_exchange,
        loss_hidden=terms.loss_hidden,
        lora_casts=terms.lora_casts,
        lora_dropout=terms.lora_dropout,
        tower_lora=terms.tower_lora,
        nograd_pixels=terms.nograd_pixels,
        balance_peak=terms.balance_pixels,
        tp_head_gather=terms.tp_head_gather,
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
    staging = vision.vision_row_staging_bytes(settings, image_pixels)
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


def _process_host_components(
    offloaded_activations: float,
    offloaded_params: float,
) -> tuple[MemoryComponent, ...]:
    """The trainer process's runtime and the tensors it offloads."""
    return (
        MemoryComponent(
            key="trainer_process",
            label="Trainer process",
            n_bytes=formulas.TRAINER_PROCESS_HOST_BYTES,
            note="Python, torch and the CUDA runtime.",
        ),
        MemoryComponent(
            key="activation_offload",
            label="Offloaded activations (pinned)",
            n_bytes=int(offloaded_activations),
            note="Tensors one micro-batch's forward saves for backward.",
        ),
        MemoryComponent(
            key="parameter_offload",
            label="Offloaded FSDP shard (pinned)",
            n_bytes=int(offloaded_params),
            note="Sharded parameters CPUOffloadPolicy keeps on host.",
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
            note="The reference is a frozen adapter copy.",
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
                "forward_peak": int(peak.forward_peak),
                "forward_all_gather": int(peak.forward_gather),
                "checkpoint_boundaries": int(peak.saved),
                "saved_router_topk": int(peak.router_topk),
                "saved_moe_collectives": int(peak.saved_collectives),
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
                    "chunk_tiles": max(int(peak.logits) - peak.tp_head_gather, 0),
                    "tp_head_gather": peak.tp_head_gather,
                }
                if peak.logits and peak.tp_head_gather
                else {}
            ),
            note=(
                "The fused logprob path tiles the lm_head matmul."
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
    device: DeviceSpec,
    rollout: int,
    settings: TrainingSettings,
    orchestrated: bool,
    reserve: float,
    gather_pool: float,
) -> tuple[MemoryComponent, ...]:
    """Adam state, process overhead, and caching-allocator slack."""
    context_bytes = device.context_bytes
    job_overhead = formulas.JOB_OVERHEAD_BYTES if orchestrated else 0
    multi_rank = settings.shard_group_gpus > 1
    lib_overhead = 0 if multi_rank else formulas.TRAINER_LIB_OVERHEAD_BYTES
    distributed_overhead = (
        trainer_distributed_bytes(device, settings) if multi_rank else 0
    )
    overhead = (
        context_bytes + rollout + job_overhead + lib_overhead + distributed_overhead
    )
    return (
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


def _lora_send_buffer_component(
    arch: ModelArch, settings: TrainingSettings
) -> MemoryComponent:
    """The main rank's registered NIXL buffer the rollout engines READ the adapter, or two rounds of merged weights, from."""
    dtype_bytes = DTYPE_BYTES[settings.weight_dtype]
    if settings.async_rollout and settings.merged_lora_sync:
        # Double-buffered: the trainer packs one round while the engines READ the other.
        return MemoryComponent(
            key="lora_send_buffer",
            label="Merged weight send buffer",
            n_bytes=2
            * merged_sync_bucket_bytes(
                *lora_formulas.merged_lora_sync_bytes(
                    arch,
                    settings.lora_target_scope,
                    settings.lora_packed_target_matrices,
                    dtype_bytes,
                )
            ),
            note="Main rank only: two rounds of merged weights at the engines' dtype, outside the caching allocator.",
        )
    n_bytes = (
        lora_formulas.lora_sync_buffer_bytes(
            arch,
            settings.lora_rank,
            settings.lora_target_scope,
            settings.lora_packed_target_matrices,
            dtype_bytes,
        )
        if settings.async_rollout
        else 0
    )
    return MemoryComponent(
        key="lora_send_buffer",
        label="LoRA send buffer",
        n_bytes=n_bytes,
        note="Main rank only: the policy adapter at the engines' dtype, outside the caching allocator.",
    )


def _training_kernels(
    arch: ModelArch, settings: TrainingSettings, device: DeviceSpec
) -> tuple[bool, int | None]:
    """Device-dependent kernels the training forward runs.

    :param arch: Model architecture.
    :param settings: Training settings.
    :param device: Training GPU.
    :return: Whether Mamba layers run HF's unfused path, and the element size
        of the routed-expert LoRA grouped GEMM, or ``None`` without one.
    """
    # Context-parallel Mamba-2 mixers run their head shard on the unfused path.
    context_parallel_mamba = (
        settings.fsdp is not None
        and settings.fsdp.cp > 1
        and arch.n_mamba_layers > 0
        and not arch.mamba_gated_delta
    )
    unfused_mamba_scan = context_parallel_mamba or (
        arch.unfused_mamba_scan_family
        and device.compute_capability is not None
        and mamba_scan_is_unfused(device.compute_capability)
    )
    # Expert LoRA factors are cast to the activation dtype: the FSDP param
    # dtype, else the weight dtype.
    torch_dtypes = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    gemm_dtype = (
        torch_dtypes[settings.weight_dtype]
        if settings.fsdp is None
        else settings.fsdp.param_dtype
    )
    # Training images ship Triton.
    grouped_gemm_bytes = (
        int(FSDP_PARAM_BYTES[gemm_dtype])
        if device.compute_capability is not None
        and grouped_gemm_supported(device.compute_capability, gemm_dtype)
        and expert_gemm_backend(device.compute_capability, gemm_dtype, True)
        == "grouped_mm"
        else None
    )
    return unfused_mamba_scan, grouped_gemm_bytes


def trainer_distributed_bytes(device: DeviceSpec, settings: TrainingSettings) -> int:
    """NCCL buffers on one rank of a multi-rank trainer."""
    fsdp = settings.fsdp
    return trainer_nccl_bytes(
        device.nccl_group_bytes,
        settings.n_training_gpus,
        settings.shard_group_gpus,
        1 if fsdp is None else fsdp.ep,
        1 if fsdp is None else fsdp.tp,
        device.gpus_per_node,
        1 if fsdp is None else fsdp.cp,
    )


def _torch_side_bytes(
    base: float, adapters: AdapterState, peak: ActivationPeak
) -> float:
    """Bytes the trainer allocates through torch at the binding instant."""
    return (
        base
        + adapters.adapters
        + peak.live_grads
        + peak.activations
        + peak.logits
        + peak.adam
    )


def _binding_pass(
    model: ModelSpec,
    settings: TrainingSettings,
    counts: formulas.ParamCounts,
    shards: int,
    adapters: AdapterState,
    peak: ActivationPeak,
    retained_gather: float,
) -> tuple[ActivationPeak, float, float, float]:
    """The gradient pass or its forward, whichever allocates more through torch.

    :param shards: GPUs the dense weights are sharded over.
    :param peak: Activation peak of the gradient pass.
    :param retained_gather: All-gather output the forward retains.
    :return: The binding peak, its base and torch-side bytes, and the bytes
        the all-gather pool already reserves.
    """
    base = trainer_base_bytes(model, settings, counts, shards, peak.activations)
    torch_side = _torch_side_bytes(base, adapters, peak)
    forward = replace(
        peak,
        activations=peak.forward_peak - adapters.grads,
        logits=0.0,
        live_grads=adapters.grads,
        adam=peak.resident_adam,
    )
    forward_base = trainer_base_bytes(
        model, settings, counts, shards, forward.activations
    )
    forward_side = _torch_side_bytes(forward_base, adapters, forward)
    if forward_side <= torch_side:
        return peak, base, torch_side, 0.0
    # The all-gather pool already reserves the output the forward retains.
    pooled = min(forward_side - torch_side, retained_gather)
    return forward, forward_base, forward_side, pooled


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
    shards = settings.dense_fsdp_gpus
    adapters = adapter_state(arch, settings, shards)
    unfused_mamba_scan, grouped_gemm_bytes = _training_kernels(arch, settings, device)
    # A token-budget micro-batch is one packed row of the budget.
    if settings.micro_batch_tokens is None:
        forward_settings, forward_len = settings, seq_len
    else:
        forward_settings = settings.model_copy(update={"micro_batch_size": 1})
        forward_len = settings.micro_batch_tokens
    retained_gather, offload_staging = fsdp_forward_gather_bytes(
        arch, settings, counts, shards
    )
    peak = activation_peak(
        arch,
        forward_settings,
        adapters,
        act_bytes=act_bytes,
        seq_len=forward_len,
        unfused_mamba_scan=unfused_mamba_scan,
        grouped_gemm_bytes=grouped_gemm_bytes,
        forward_gather=retained_gather + offload_staging,
    )
    peak, base, torch_side, pooled = _binding_pass(
        model, settings, counts, shards, adapters, peak, retained_gather
    )
    offloaded_params = fsdp_cpu_offload_host_bytes(model, settings, counts, shards)
    # Held per-update tensors: completions, masks, old/ref/sampling logprobs,
    # advantages.
    rollout = 6 * settings.trajectories * seq_len * 4
    reserve, gather_pool = trainer_allocator_slack(
        arch, settings, counts, shards, torch_side - pooled
    )
    gather_pool -= pooled
    components = (
        _model_state_components(base, adapters, peak)
        + _activation_workspace_components(peak, adapters)
        + _runtime_components(
            peak,
            adapters,
            device,
            rollout,
            settings,
            orchestrated,
            reserve,
            gather_pool,
        )
        + (_lora_send_buffer_component(arch, settings),)
    )
    return PhaseBreakdown(
        phase="training",
        components=components,
        device_total_bytes=device.total_bytes,
        device_usable_bytes=device.usable_bytes,
        warnings=tuple(warnings),
        host=_process_host_components(peak.offloaded, offloaded_params)
        + training_host_components(arch, settings, adapters, rollout, act_bytes),
        host_usable_bytes=device.host_bytes_per_gpu,
    )
