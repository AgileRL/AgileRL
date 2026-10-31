# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The training-memory search: DDP or FSDP and speed rungs by modeled cost, then memory-only last resorts."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Literal

from agilerl.arena.memory.breakdown import PhaseBreakdown
from agilerl.arena.memory.estimator import estimate_training
from agilerl.arena.memory.formulas import (
    DEFAULT_ROUTED_CHUNK_BYTES,
    ROUTED_CHUNK_MIB_CHOICES,
    checkpointed_blocks,
)
from agilerl.arena.memory.resolve.batch import (
    ROW_FILL,
    micro_batch_row_candidates,
    micro_batch_token_candidates,
    rows_within_budget,
)
from agilerl.arena.memory.resolve.learn_costs import LearnCosts, learn_costs
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GiB,
    MiB,
    ModelArch,
    ModelSpec,
    TrainingSettings,
)
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.fsdp import FSDP_MEMORY_FIELDS, FSDPConfig

logger = logging.getLogger(__name__)

Residency = Literal["gathered", "reshard"]

# Strides an unset fsdp.checkpoint_every_n_blocks may take, least recompute first.
CHECKPOINT_STRIDES = (4, 2, 1)
# Residencies, fewest all-gathers first.
RESIDENCIES: tuple[Residency, ...] = ("gathered", "reshard")


def cpu_offload_blocker(fsdp: FSDPConfig | None, async_rollout: bool) -> str | None:
    """Why FSDP ``cpu_offload`` cannot run, or ``None`` when it can.

    :param fsdp: The run's FSDP config, or ``None``.
    :param async_rollout: vLLM generates on rollout engines outside the trainer.
    :return: A short reason, or ``None`` when full CPU offload is off or served.
    """
    if fsdp is None or not fsdp.cpu_offload or not async_rollout:
        return None
    return (
        "fsdp.cpu_offload needs colocated vLLM generation: FSDP2's "
        "CPUOffloadPolicy keeps parameters on CPU. Use fsdp.optim_cpu_offload "
        "instead, which keeps parameters on the GPU."
    )


def can_train_data_parallel(fsdp: FSDPConfig) -> bool:
    """Whether a run configured for FSDP may train DDP instead, every rank holding the whole model.

    Expert and tensor parallel split the model across ranks, and a set
    memory field or stride is an FSDP setting the run keeps.

    :param fsdp: The run's FSDP config.
    :return: True when DDP is allowed.
    """
    return (
        fsdp.ep == 1
        and fsdp.tp == 1
        and fsdp.checkpoint_every_n_blocks is None
        and all(getattr(fsdp, name) is None for name in FSDP_MEMORY_FIELDS)
    )


def gather_cost(residency: Residency | None, costs: LearnCosts) -> float:
    """Frozen-base all-gather time of one micro-batch, in tokens of learn work.

    Gathered keeps forward's gather for backward, so it pays half of reshard.

    :param residency: Where the frozen base lives, or ``None`` under DDP.
    :param costs: Learn-time constants of the training GPU.
    :return: Token-equivalents per micro-batch.
    """
    return {
        None: 0.0,
        "gathered": costs.reshard_gather_tokens / 2,
        "reshard": costs.reshard_gather_tokens,
    }[residency]


def learn_cost(
    n_layers: int,
    gradient_checkpointing: bool,
    every_n_blocks: int,
    passes: int,
    tokens: float,
    residency: Residency | None,
    costs: LearnCosts,
    routed_chunks: int = 0,
) -> float:
    """Modeled time of one optimizer step, in tokens of uncheckpointed learn work.

    DDP's gradient all-reduce and FSDP's reduce-scatter both run once per
    step over the same trainable gradients, so gradient sync adds no term.

    :param n_layers: Decoder blocks.
    :param gradient_checkpointing: Activation checkpointing is on.
    :param every_n_blocks: Checkpoint stride.
    :param passes: Micro-batches per optimizer step.
    :param tokens: Tokens the optimizer step trains on.
    :param residency: Where the frozen base lives, or ``None`` under DDP.
    :param costs: Learn-time constants of the training GPU.
    :param routed_chunks: Routed-expert row chunks of one micro-batch, from
        :func:`routed_chunks_per_pass`.
    :return: Token-equivalents of work.
    """
    checkpointed = checkpointed_blocks(n_layers, gradient_checkpointing, every_n_blocks)
    recompute = costs.recompute_cost * checkpointed / n_layers
    per_pass = (
        costs.micro_batch_overhead_tokens
        + gather_cost(residency, costs)
        + routed_chunks * costs.routed_chunk_tokens
    )
    return tokens * (1 + recompute) + passes * per_pass


def routed_chunks_per_pass(
    arch: ModelArch,
    expert_lora: bool,
    tokens: float,
    act_bytes: float,
    chunk_bytes: int,
) -> int:
    """Routed-expert row chunks one micro-batch runs, summed over MoE layers.

    The self-routing expert-LoRA wrapper (``arch.routed_expert_recompute``)
    splits each layer's routed rows into chunks whose widest row fits
    ``chunk_bytes``.

    :param arch: Model architecture.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param tokens: Tokens in the micro-batch.
    :param act_bytes: Bytes per activation element.
    :param chunk_bytes: Widest ``[rows, features]`` activation of one chunk.
    :return: Chunks over every MoE layer; zero off the chunked wrapper.
    """
    if not (expert_lora and arch.routed_expert_recompute):
        return 0
    inter = arch.expert_intermediate_size or arch.intermediate_size
    row_bytes = max(arch.expert_width, inter) * act_bytes
    routed_bytes = tokens * (arch.n_experts_per_tok or 1) * row_bytes
    return arch.moe_layers * math.ceil(routed_bytes / chunk_bytes)


@dataclass(frozen=True)
class MemoryPins:
    """Speed fields set before the search; ``None`` fields are searched.

    The checkpoint stride and the FSDP memory fields are read from the
    settings' ``fsdp``, and the DDP routed-expert chunk from the settings'
    ``routed_expert_chunk_mib``, where ``None`` also means searched.
    """

    gradient_checkpointing: bool | None = None
    micro_batch_size_per_gpu: int | None = None
    micro_batch_tokens_per_gpu: int | None = None

    @classmethod
    def from_spec(cls, spec: LLMAlgorithmSpec) -> MemoryPins:
        """Pins an algorithm spec carries.

        :param spec: Validated LLM algorithm spec.
        :return: The spec's checkpointing and micro-batch values.
        """
        return cls(
            gradient_checkpointing=spec.gradient_checkpointing,
            micro_batch_size_per_gpu=spec.micro_batch_size_per_gpu,
            micro_batch_tokens_per_gpu=(
                spec.micro_batch_tokens_per_gpu
                if isinstance(spec, RolloutLLMSpec)
                else None
            ),
        )


@dataclass(frozen=True)
class TrainingMemoryPlan:
    """Speed rungs and memory fields one trainer rank runs with."""

    gradient_checkpointing: bool
    checkpoint_every_n_blocks: int
    micro_batch_size_per_gpu: int
    # Tokens per packed micro-batch; ``None`` when micro-batches count rows.
    micro_batch_tokens_per_gpu: int | None
    # Every memory field set; ``None`` trains DDP.
    fsdp: FSDPConfig | None
    # Routed-expert row chunk under DDP; ``None`` under FSDP and on dense models.
    routed_expert_chunk_mib: int | None
    # Modeled step, and the tokens it trains, from :func:`learn_cost`.
    cost: float
    tokens: float
    # The settings the plan was estimated at, and that estimate.
    settings: TrainingSettings
    breakdown: PhaseBreakdown

    @property
    def learn_cost_ratio(self) -> float:
        """Modeled step over its trained tokens: 1 plus recompute and per-pass costs."""
        return self.cost / self.tokens

    def describe(self) -> str:
        """Plain-English summary for logs and errors."""
        if self.micro_batch_tokens_per_gpu is None:
            unit = f"micro_batch_size_per_gpu={self.micro_batch_size_per_gpu}"
        else:
            unit = f"micro_batch_tokens_per_gpu={self.micro_batch_tokens_per_gpu}"
        fsdp = self.fsdp
        parts = [
            "DDP" if fsdp is None else "FSDP",
            unit,
            f"gradient_checkpointing={self.gradient_checkpointing}",
        ]
        if self.routed_expert_chunk_mib is not None:
            parts.append(f"routed_expert_chunk_mib={self.routed_expert_chunk_mib}")
        if fsdp is not None:
            parts += [
                f"fsdp.checkpoint_every_n_blocks={fsdp.checkpoint_every_n_blocks}",
                f"fsdp.reshard_after_forward={fsdp.reshard_after_forward}",
                f"fsdp.routed_expert_chunk_mib={fsdp.routed_expert_chunk_mib}",
                f"fsdp.optim_cpu_offload={fsdp.optim_cpu_offload}",
                f"fsdp.activation_offload={fsdp.activation_offload}",
                f"fsdp.cpu_offload={fsdp.cpu_offload}",
            ]
        return ", ".join(parts)


class TrainingMemoryUnfitError(ValueError):
    """Nothing fits one GPU, even at the lightest settings.

    :param plan: The lightest plan, priced, for reporting how far over it is.
    :param device: The training GPU.
    """

    def __init__(self, plan: TrainingMemoryPlan, device: DeviceSpec) -> None:
        breakdown = plan.breakdown
        super().__init__(
            f"Estimated training peak {breakdown.total_bytes / GiB:.2f} GiB is over "
            f"the {breakdown.device_usable_bytes / GiB:.2f} GiB usable on one "
            f"{device.name} at the lightest settings ({plan.describe()}). Add "
            "training_gpus_per_agent, use a larger GPU, or shorten max_model_len."
        )
        self.plan = plan


def _recompute_candidates(
    gradient_checkpointing: bool | None, stride: int | None, sharded: bool
) -> list[tuple[bool, int]]:
    """``(gradient_checkpointing, checkpoint_every_n_blocks)`` pairs, least recompute first.

    Flat data parallel checkpoints every block or none.
    """
    if gradient_checkpointing is False:
        return [(False, 1)]
    if stride is not None:
        strides: tuple[int, ...] = (stride,)
    elif sharded:
        strides = CHECKPOINT_STRIDES
    else:
        strides = (1,)
    checkpointed = [(True, every) for every in strides]
    return checkpointed if gradient_checkpointing else [(False, 1), *checkpointed]


def _residency_candidates(fsdp: FSDPConfig) -> list[tuple[Residency, FSDPConfig]]:
    """Frozen-base placements the set fields allow, fewest all-gathers first."""
    return [
        (residency, replace(fsdp, reshard_after_forward=residency == "reshard"))
        for residency in RESIDENCIES
        if fsdp.reshard_after_forward is None
        or fsdp.reshard_after_forward == (residency == "reshard")
    ]


def _memory_rungs(
    model: ModelSpec, settings: TrainingSettings, fsdp: FSDPConfig
) -> list[tuple[str | None, FSDPConfig]]:
    """FSDP memory-only fields, fastest first, then each last resort on top of the one before.

    Last resorts, in order: a smaller routed-expert chunk (MoE only), the
    optimizer on CPU, activation offload, then full CPU offload of parameters
    and gradients. Full CPU offload turns optimizer offload off (the two are
    exclusive) and needs colocated vLLM, as CPUOffloadPolicy keeps parameters
    on CPU. Set fields never move.
    """
    current = replace(
        fsdp,
        routed_expert_chunk_mib=(
            ROUTED_CHUNK_MIB_CHOICES[0]
            if fsdp.routed_expert_chunk_mib is None
            else fsdp.routed_expert_chunk_mib
        ),
        optim_cpu_offload=bool(fsdp.optim_cpu_offload),
        activation_offload=bool(fsdp.activation_offload),
        cpu_offload=bool(fsdp.cpu_offload),
    )
    rungs: list[tuple[str | None, FSDPConfig]] = [(None, current)]
    if fsdp.routed_expert_chunk_mib is None and model.arch.is_moe:
        for chunk in ROUTED_CHUNK_MIB_CHOICES[1:]:
            current = replace(current, routed_expert_chunk_mib=chunk)
            rungs.append((f"fsdp.routed_expert_chunk_mib={chunk}", current))
    if fsdp.optim_cpu_offload is None and not fsdp.cpu_offload:
        current = replace(current, optim_cpu_offload=True)
        rungs.append(("fsdp.optim_cpu_offload=True", current))
    if fsdp.activation_offload is None:
        current = replace(current, activation_offload=True)
        rungs.append(("fsdp.activation_offload=True", current))
    if (
        fsdp.cpu_offload is None
        and not fsdp.optim_cpu_offload
        and settings.uses_generation_engine
        and not settings.async_rollout
    ):
        current = replace(current, cpu_offload=True, optim_cpu_offload=False)
        rungs.append(("fsdp.cpu_offload=True", current))
    return rungs


def _micro_batch_units(
    settings: TrainingSettings, pins: MemoryPins, mini_batch_rows: int
) -> tuple[float, list[int]]:
    """Tokens one optimizer step trains on one rank, and the micro-batch sizes to try.

    :param settings: Training settings; ``micro_batch_tokens`` set means the forward packs.
    :param pins: Micro-batch values the search keeps.
    :param mini_batch_rows: Rows one optimizer step covers on one rank.
    :return: Mini-batch tokens and ascending micro-batch sizes, in tokens
        when packed, else rows.
    """
    row_budget = settings.max_model_len
    if settings.micro_batch_tokens is None:
        rows = (
            [pins.micro_batch_size_per_gpu]
            if pins.micro_batch_size_per_gpu is not None
            else micro_batch_row_candidates(mini_batch_rows)
        )
        return float(mini_batch_rows * row_budget), rows
    mini_tokens = mini_batch_rows * row_budget
    if pins.micro_batch_tokens_per_gpu is not None:
        # A micro-batch never packs more than the whole mini-batch.
        units = [min(pins.micro_batch_tokens_per_gpu, mini_tokens)]
    elif pins.micro_batch_size_per_gpu is not None:
        units = [pins.micro_batch_size_per_gpu * row_budget]
    else:
        units = micro_batch_token_candidates(row_budget, mini_batch_rows)
    return mini_tokens * ROW_FILL, units


def _ddp_chunk_candidates(
    model: ModelSpec, settings: TrainingSettings
) -> tuple[int | None, ...]:
    """Routed-expert chunk sizes DDP tries, largest first; ``None`` for dense models.

    :param model: Training-side model spec.
    :param settings: Training settings; a set ``routed_expert_chunk_mib`` is kept.
    :return: Chunk sizes in MiB.
    """
    if settings.routed_expert_chunk_mib is not None:
        return (settings.routed_expert_chunk_mib,)
    if model.arch.is_moe:
        return ROUTED_CHUNK_MIB_CHOICES
    return (None,)


def resolve_training_memory(
    model: ModelSpec,
    device: DeviceSpec,
    settings: TrainingSettings,
    pins: MemoryPins,
    mini_batch_rows: int,
) -> TrainingMemoryPlan:
    """The fastest plan that fits: DDP or FSDP, its speed rungs, then FSDP memory-only last resorts.

    Speed rungs are searched jointly and the fitting combination with the
    lowest :func:`learn_cost` wins (ties go to less recompute, then fewer
    gathers): the parallelism (DDP when the run's ``fsdp`` allows it, see
    :func:`can_train_data_parallel`, or FSDP), the micro-batch (a token
    budget when ``settings`` packs, else rows; the largest that fits),
    checkpointing (off, every 4, 2 or 1 blocks; every block or none under
    DDP), and under FSDP the frozen-base residency (gathered after forward,
    or resharded). DDP holds the whole model on every GPU and gathers
    nothing; it wins ties. Memory-only fields start at their fastest; when
    neither mode has a speed combination that fits, the last resorts turn on
    one at a time and the search reruns: a smaller routed-expert chunk under
    DDP (MoE only), and :func:`_memory_rungs` under FSDP.
    Fits means fits with the underprediction buffer; the lightest plan is
    kept when it fits only without the buffer.

    :param model: Training-side model spec.
    :param device: Training GPU.
    :param settings: Training settings. ``micro_batch_tokens`` set means the
        forward packs; ``max_model_len`` is the longest row. ``fsdp`` unset
        trains DDP; set ``fsdp`` memory fields and stride, and a set
        ``routed_expert_chunk_mib``, are kept.
    :param pins: Checkpointing and micro-batch values the search keeps. A
        token budget above the mini-batch's longest rows is cut to them.
    :param mini_batch_rows: Rows one optimizer step covers on one rank.
    :return: The chosen plan, with every field set.
    :raises TrainingMemoryUnfitError: When the lightest plan is over the GPU.
    """
    packed = settings.micro_batch_tokens is not None
    row_budget = settings.max_model_len
    fsdp = settings.fsdp
    n_layers = model.arch.n_layers
    costs = learn_costs(device)
    tokens, units = _micro_batch_units(settings, pins, mini_batch_rows)
    recompute = _recompute_candidates(
        pins.gradient_checkpointing,
        None if fsdp is None else fsdp.checkpoint_every_n_blocks,
        fsdp is not None,
    )
    ddp_chunks = _ddp_chunk_candidates(model, settings)

    def priced(
        on: bool,
        stride: int,
        residency: Residency | None,
        trial: FSDPConfig | None,
        ddp_chunk: int | None,
        unit: int,
    ) -> TrainingMemoryPlan:
        if packed:
            rows = pins.micro_batch_size_per_gpu or rows_within_budget(
                mini_batch_rows, unit, row_budget
            )
            budget: int | None = unit
            passes = math.ceil(tokens / unit)
        else:
            rows, budget, passes = unit, None, mini_batch_rows // unit
        if trial is not None:
            trial = replace(trial, checkpoint_every_n_blocks=stride)
        sized = settings.model_copy(
            update={
                "gradient_checkpointing": on,
                "fsdp": trial,
                "activation_offload": (
                    settings.activation_offload
                    if trial is None
                    else bool(trial.activation_offload)
                ),
                "micro_batch_size": rows,
                "micro_batch_tokens": budget,
                "routed_expert_chunk_mib": ddp_chunk,
            }
        )
        chunk_mib = ddp_chunk if trial is None else trial.routed_expert_chunk_mib
        chunks = routed_chunks_per_pass(
            model.arch,
            settings.lora_packed_target_matrices > 0,
            tokens / passes,
            DTYPE_BYTES[settings.weight_dtype],
            DEFAULT_ROUTED_CHUNK_BYTES if chunk_mib is None else chunk_mib * MiB,
        )
        return TrainingMemoryPlan(
            gradient_checkpointing=on,
            checkpoint_every_n_blocks=stride,
            micro_batch_size_per_gpu=rows,
            micro_batch_tokens_per_gpu=budget,
            fsdp=trial,
            routed_expert_chunk_mib=ddp_chunk,
            cost=learn_cost(
                n_layers, on, stride, passes, tokens, residency, costs, chunks
            ),
            tokens=tokens,
            settings=sized,
            breakdown=estimate_training(model, device, sized, orchestrated=True),
        )

    def largest_fit(
        on: bool,
        stride: int,
        residency: Residency | None,
        trial: FSDPConfig | None,
        ddp_chunk: int | None,
    ) -> TrainingMemoryPlan | None:
        # The estimate grows with the unit, so the largest fit is a bisection.
        low, high = 0, len(units) - 1
        best = None
        while low <= high:
            middle = (low + high) // 2
            plan = priced(on, stride, residency, trial, ddp_chunk, units[middle])
            if plan.breakdown.fits_with_buffer:
                best, low = plan, middle + 1
            else:
                high = middle - 1
        return best

    def placements(
        memory: FSDPConfig | None,
    ) -> list[tuple[Residency | None, FSDPConfig | None]]:
        if memory is None:
            return [(None, None)]
        return list(_residency_candidates(memory))

    def floor(memory: FSDPConfig | None, ddp_chunk: int | None) -> TrainingMemoryPlan:
        on, stride = recompute[-1]
        residency, trial = placements(memory)[-1]
        return priced(on, stride, residency, trial, ddp_chunk, units[0])

    def cheapest(
        strides: list[tuple[bool, int]],
        memory: FSDPConfig | None,
        ddp_chunk: int | None,
    ) -> TrainingMemoryPlan | None:
        fitting = [
            plan
            for on, stride in strides
            for residency, trial in placements(memory)
            if (plan := largest_fit(on, stride, residency, trial, ddp_chunk))
            is not None
        ]
        return min(fitting, key=lambda plan: plan.cost, default=None)

    def data_parallel(strides: list[tuple[bool, int]]) -> TrainingMemoryPlan | None:
        for index, chunk in enumerate(ddp_chunks):
            if index > 0:
                logger.info(
                    "Training memory: DDP turning on routed_expert_chunk_mib=%s; "
                    "nothing fits one %s at %s",
                    chunk,
                    device.name,
                    floor(None, ddp_chunks[index - 1]).describe(),
                )
            plan = cheapest(strides, None, chunk)
            if plan is not None:
                return plan
        return None

    if fsdp is None:
        plan = data_parallel(recompute)
        if plan is not None:
            return plan
        lightest = floor(None, ddp_chunks[-1])
    else:
        ddp = (
            data_parallel(
                _recompute_candidates(pins.gradient_checkpointing, None, False)
            )
            if can_train_data_parallel(fsdp)
            else None
        )
        rungs = _memory_rungs(model, settings, fsdp)
        for index, (resort, memory) in enumerate(rungs):
            if index > 0:
                logger.info(
                    "Training memory: turning on %s; nothing fits one %s at %s",
                    resort,
                    device.name,
                    floor(rungs[index - 1][1], None).describe(),
                )
            plan = cheapest(recompute, memory, None)
            # A fitting DDP plan beats every FSDP last resort.
            if ddp is not None and (plan is None or ddp.cost <= plan.cost):
                return ddp
            if plan is not None:
                return plan
        lightest = floor(rungs[-1][1], None)
    if lightest.breakdown.fits:
        logger.info(
            "Training memory: %s fits one %s only inside the underprediction buffer",
            lightest.describe(),
            device.name,
        )
        return lightest
    raise TrainingMemoryUnfitError(lightest, device)
