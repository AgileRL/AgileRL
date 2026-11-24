# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""The training-memory search: DDP or FSDP and speed rungs by modeled cost, then memory-only last resorts."""

from __future__ import annotations

import copy
import logging
import math
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import product
from typing import Literal, NamedTuple

from agilerl.arena.memory.breakdown import PhaseBreakdown
from agilerl.arena.memory.estimator import (
    FSDP_PARAM_BYTES,
    shared_training_estimate,
    training_tensor_parallel,
)
from agilerl.arena.memory.formulas import (
    ADAPTER_BYTES_PER_PARAM,
    DEFAULT_ROUTED_CHUNK_BYTES,
    MAX_UNDERPREDICTION,
    ROUTED_CHUNK_MIB_CHOICES,
    checkpointed_blocks,
    moe_collectives_saveable,
    param_counts,
    recommendation_fits,
    routed_expert_params_per_layer,
)
from agilerl.arena.memory.lora_formulas import trained_param_counts
from agilerl.arena.memory.resolve.batch import (
    ROW_FILL,
    micro_batch_row_candidates,
    micro_batch_token_candidates,
    rows_within_budget,
)
from agilerl.arena.memory.resolve.learn_costs import LearnCosts, learn_costs
from agilerl.arena.memory.settings_key import TrainingSettingsKey
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    GiB,
    MiB,
    ModelArch,
    ModelSpec,
    TrainingSettings,
    WeightDtype,
)
from agilerl.arena.memory.training_host import fixed_host_bytes
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.fsdp import FSDP_MEMORY_FIELDS, FSDPConfig

logger = logging.getLogger(__name__)

Residency = Literal["gathered", "reshard"]
# A ``logger.info`` format and its arguments.
LogNote = tuple[str, tuple[object, ...]]

# Strides an unset fsdp.checkpoint_every_n_blocks may take, least recompute first.
CHECKPOINT_STRIDES = (4, 2, 1)
# Residencies, fewest all-gathers first.
RESIDENCIES: tuple[Residency, ...] = ("gathered", "reshard")


def can_train_data_parallel(fsdp: FSDPConfig) -> bool:
    """Whether a run configured for FSDP may train DDP instead, every rank holding the whole model.

    Expert and tensor parallel split the model across ranks, context parallel
    runs on the FSDP mesh, and a set memory field or stride is an FSDP setting
    the run keeps.

    :param fsdp: The run's FSDP config.
    :return: True when DDP is allowed.
    """
    return (
        fsdp.ep == 1
        and fsdp.tp == 1
        and fsdp.cp == 1
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


def buffer_room(breakdown: PhaseBreakdown) -> float:
    """Bytes a training breakdown has left inside the underprediction buffer; negative when over.

    :param breakdown: Training breakdown.
    :return: The smaller of the device's and, when its share is known, the host's room.
    """
    kept = 1 - MAX_UNDERPREDICTION
    room = breakdown.device_usable_bytes * kept - breakdown.total_bytes
    if breakdown.host_usable_bytes is None:
        return room
    return min(room, breakdown.host_usable_bytes * kept - breakdown.host_total_bytes)


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


def offloaded_optimizer_bytes(
    fsdp: FSDPConfig | None, breakdown: PhaseBreakdown
) -> int:
    """AdamW state one rank copies to the GPU and back around each optimizer step.

    :param fsdp: The plan's FSDP config, or ``None`` under DDP.
    :param breakdown: The plan's training estimate.
    :return: Bytes; zero when the state stays on the GPU.
    """
    if fsdp is None or not fsdp.optim_cpu_offload:
        return 0
    state = next(part for part in breakdown.components if part.key == "optimizer_state")
    return state.detail["step_bytes"]


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


class CrossNodeSeconds(NamedTuple):
    """Seconds one optimizer step on one rank spends on traffic between nodes."""

    # FSDP weight all-gathers and the reduce-scatters backward hides, which
    # overlap with compute.
    gathers: float = 0.0
    # Routed-token dispatch and combine, which the layer waits on.
    exchange: float = 0.0
    # Gradient sync the step waits on: the last block's reduce-scatter and
    # the expert-LoRA and DDP all-reduces after backward.
    grad_sync: float = 0.0


@dataclass(frozen=True)
class StepTiming:
    """What the training-memory search needs to rank plans by learn seconds instead of learn work.

    Given only when trainer ranks span nodes; ``None`` ranks by
    :func:`learn_cost` alone.
    """

    # Compute seconds one optimizer step takes at a learn cost ratio of 1.
    unit_step_seconds: float
    # No-grad forwards over one optimizer step's tokens.
    no_grad_sweeps: float

    def learn_seconds(self, plan: TrainingMemoryPlan) -> float:
        """Seconds one optimizer step takes: compute or weight gathers, whichever is longer, then the exchange, gradient sync and offload.

        :param plan: A priced plan.
        :return: Seconds.
        """
        traffic = plan.cross_node
        return (
            max(self.compute_seconds(plan.cost, plan.tokens), traffic.gathers)
            + traffic.exchange
            + traffic.grad_sync
            + plan.optimizer_offload_seconds
        )

    def compute_seconds(self, cost: float, tokens: float) -> float:
        """Compute seconds of one optimizer step of ``cost`` learn work over ``tokens`` tokens."""
        return self.unit_step_seconds * cost / tokens


def routed_token_bytes(
    arch: ModelArch,
    tokens: float,
    weight_dtype: WeightDtype,
    tp: int,
    ep: int,
    cp: int,
    node: int,
) -> float:
    """Routed-token bytes one node sends the others in one dispatch or combine over an optimizer step's tokens.

    :param arch: Model architecture.
    :param tokens: Tokens one optimizer step trains on one data-parallel slice.
    :param weight_dtype: Activation dtype name, as ``TrainingSettings.weight_dtype``.
    :param tp: Tensor-parallel degree; each tensor-parallel rank routes its share of ``tokens``.
    :param ep: Expert-parallel degree.
    :param cp: Context-parallel degree.
    :param node: GPUs on one node.
    :return: Bytes; zero when the expert group stays on one node.
    """
    if ep <= node:
        return 0.0
    return (
        node
        * tokens
        / (cp * tp)
        * (arch.n_experts_per_tok or 1)
        * arch.expert_width
        * DTYPE_BYTES[weight_dtype]
        * arch.moe_layers
        * (ep - node)
        / ep
    )


def _rings_leaving_node(group: int, stride: int, node: int) -> int:
    """Rings that include the busiest node and a rank on another node.

    :param group: Ranks in one ring.
    :param stride: Rank distance between two members of a ring.
    :param node: GPUs on one node.
    :return: Ring count for the node that sits in the most rings.
    """
    span = group * stride
    rings_per_node = [0] * ((span + node - 1) // node)
    for offset in range(stride):
        hosts = {(offset + step * stride) // node for step in range(group)}
        if len(hosts) == 1:
            continue
        for host in hosts:
            rings_per_node[host] += 1
    return max(rings_per_node)


def _ring_gather_bytes(
    tensor_bytes: float, group: int, stride: int, node: int
) -> float:
    """Bytes the busiest node sends in a ring all-gather of a tensor sharded over ``group`` ranks ``stride`` apart.

    Each group's ring leaves a node once and carries every piece but one.

    :param tensor_bytes: Bytes of the whole gathered tensor.
    :param group: Ranks the tensor is sharded over.
    :param stride: Rank distance between two members of the group.
    :param node: GPUs on one node.
    :return: Bytes; zero when every group stays on one node.
    """
    if group <= 1 or group * stride <= node:
        return 0.0
    rings = _rings_leaving_node(group, stride, node)
    return rings * (group - 1) * tensor_bytes / group


def cross_node_seconds(
    model: ModelSpec,
    plan: TrainingMemoryPlan,
    trainers: int,
    node: int,
    no_grad_sweeps: float,
    costs: LearnCosts,
) -> CrossNodeSeconds:
    """Seconds one optimizer step spends on FSDP weight gathers and routed tokens crossing nodes.

    Every micro-batch gathers the dense and expert weights in forward, again
    in backward when weights reshard after forward, and once per no-grad
    sweep. An expert group wider than a node sends routed tokens out and
    back in forward, backward, each no-grad sweep and, unless MoE
    collectives are saved, the checkpoint recompute. Once per optimizer step
    the trained gradients sync: FSDP reduce-scatters them over the shard
    group in its reduce dtype and all-reduces them over HSDP replicas, expert
    LoRA all-reduces over the ranks holding the same experts, and DDP
    all-reduces every fp32 gradient over all ranks. FSDP's per-block
    reduce-scatters overlap with backward, so all but the last block's share
    joins the gathers. Every collective is a ring, priced at the GPU class's measured all-gather rate between nodes
    (the token exchange at its exchange rate).

    :param model: Training-side model spec.
    :param plan: The trainer plan; its ``fsdp`` holds the layout.
    :param trainers: Trainer ranks, filling whole nodes from the first.
    :param node: GPUs on one node.
    :param no_grad_sweeps: No-grad forwards over one optimizer step's tokens.
    :param costs: Learn-time constants of the training GPU.
    :return: Gather, exchange and gradient-sync seconds; zero on one node.
    """
    micro_batch = (
        plan.micro_batch_tokens_per_gpu
        or plan.micro_batch_size_per_gpu * plan.settings.max_model_len
    )
    return _cross_node_seconds(
        model,
        plan.fsdp,
        plan.settings,
        plan.tokens,
        math.ceil(plan.tokens / micro_batch),
        plan.gradient_checkpointing,
        plan.checkpoint_every_n_blocks,
        trainers,
        node,
        no_grad_sweeps,
        costs,
    )


def _cross_node_seconds(
    model: ModelSpec,
    fsdp: FSDPConfig | None,
    settings: TrainingSettings,
    tokens: float,
    micro_batches: int,
    gradient_checkpointing: bool,
    every_n_blocks: int,
    trainers: int,
    node: int,
    no_grad_sweeps: float,
    costs: LearnCosts,
) -> CrossNodeSeconds:
    """:func:`cross_node_seconds` from a plan's fields, so the search can bound a candidate before pricing it."""
    if trainers <= node:
        return CrossNodeSeconds()
    hidden_sync, grad_sync = _grad_sync_seconds(
        model.arch, fsdp, settings, trainers, node, costs
    )
    if fsdp is None:
        return CrossNodeSeconds(grad_sync=grad_sync)
    arch = model.arch
    byte = DTYPE_BYTES[settings.weight_dtype]
    shard = fsdp.shard_group_size or trainers
    total = param_counts(arch, model.n_params).total * byte
    experts = (
        arch.moe_layers * routed_expert_params_per_layer(arch) * byte
        if arch.is_moe and fsdp.ep > 1
        else 0.0
    )
    # The trainer mesh puts tensor parallel innermost for dense weights and
    # expert parallel innermost for routed experts.
    gather_bytes = _ring_gather_bytes(
        (total - experts) / fsdp.tp, shard // fsdp.tp, fsdp.tp, node
    ) + _ring_gather_bytes(experts / fsdp.ep, shard // fsdp.ep, fsdp.ep, node)
    gathers = 0.0
    if gather_bytes and costs.cross_node_gather_bytes_per_second:
        backward = 0 if fsdp.reshard_after_forward is False else 1
        gathers = (
            micro_batches
            * (1 + backward + no_grad_sweeps)
            * gather_bytes
            / costs.cross_node_gather_bytes_per_second
        )
    exchange = 0.0
    token_bytes = (
        routed_token_bytes(
            arch, tokens, settings.weight_dtype, fsdp.tp, fsdp.ep, fsdp.cp, node
        )
        if arch.is_moe
        else 0.0
    )
    if token_bytes and costs.cross_node_exchange_bytes_per_second:
        # Saved MoE collectives let the checkpoint recompute skip the exchange.
        recomputed = (
            0.0
            if fsdp.save_moe_collectives
            else checkpointed_blocks(
                arch.n_layers, gradient_checkpointing, every_n_blocks
            )
            / arch.n_layers
        )
        exchanges = 4 + 2 * recomputed + 2 * no_grad_sweeps
        exchange = exchanges * token_bytes / costs.cross_node_exchange_bytes_per_second
    return CrossNodeSeconds(gathers + hidden_sync, exchange, grad_sync)


def _grad_sync_seconds(
    arch: ModelArch,
    fsdp: FSDPConfig | None,
    settings: TrainingSettings,
    trainers: int,
    node: int,
    costs: LearnCosts,
) -> tuple[float, float]:
    """Seconds the once-per-step gradient sync spends between nodes: the part backward hides, and the part it waits on.

    FSDP2 reduce-scatters each block's gradients, and all-reduces them over
    HSDP replicas, in that block's post-backward hook while earlier blocks
    run backward, so only the last block's share is waited on. The expert-LoRA
    and DDP all-reduces run after backward (``runtime.py`` ``sync_grads``).
    """
    rate = costs.cross_node_gather_bytes_per_second
    if not rate:
        return 0.0, 0.0
    ep = 1 if fsdp is None else fsdp.ep
    dense, expert = trained_param_counts(arch, settings, ep)
    if fsdp is None:
        # A ring all-reduce moves twice a ring reduce-scatter's bytes.
        moved = 2 * _ring_gather_bytes(
            dense * ADAPTER_BYTES_PER_PARAM, trainers, 1, node
        )
        return 0.0, moved / rate
    reduce = FSDP_PARAM_BYTES[fsdp.reduce_dtype]
    shard = fsdp.shard_group_size or trainers
    dense_group = shard // fsdp.tp
    dense_bytes = dense * reduce / fsdp.tp
    per_block = _ring_gather_bytes(dense_bytes, dense_group, fsdp.tp, node)
    # One rank's slice all-reduces over the ranks holding the same slice.
    replicas = trainers // shard
    per_block += 2 * _ring_gather_bytes(
        dense_bytes / dense_group, replicas, shard, node
    )
    last_block = per_block / max(arch.n_layers, 1)
    after = 2 * _ring_gather_bytes(expert * reduce / ep, trainers // ep, ep, node)
    return (per_block - last_block) / rate, (last_block + after) / rate


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
    # ``fsdp.save_moe_collectives`` has collectives to save on this model.
    moe_collectives_saveable: bool
    # Modeled step, and the tokens it trains, from :func:`learn_cost`.
    cost: float
    tokens: float
    # The settings the plan was estimated at, and that estimate.
    settings: TrainingSettings
    breakdown: PhaseBreakdown
    # Seconds one optimizer step spends moving CPU-kept AdamW state and stepping
    # it. Seconds, not tokens: the cost is host-link bytes, not model FLOPs.
    optimizer_offload_seconds: float
    # Traffic between nodes per optimizer step; zero when searched without
    # :class:`StepTiming`.
    cross_node: CrossNodeSeconds = field(default_factory=CrossNodeSeconds)

    @property
    def learn_cost_ratio(self) -> float:
        """Modeled step over its trained tokens: 1 plus recompute and per-pass costs."""
        return self.cost / self.tokens

    def detached(self) -> TrainingMemoryPlan:
        """A copy holding its own ``fsdp`` configs and breakdown ``detail`` dicts."""
        fsdp = copy.copy(self.fsdp)
        settings_fsdp = (
            fsdp if self.settings.fsdp is self.fsdp else copy.copy(self.settings.fsdp)
        )
        return replace(
            self,
            fsdp=fsdp,
            settings=self.settings.model_copy(update={"fsdp": settings_fsdp}),
            breakdown=self.breakdown.detached(),
        )

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
                f"fsdp.tp={fsdp.tp}",
                f"fsdp.ep={fsdp.ep}",
                f"fsdp.cp={fsdp.cp}",
                f"fsdp.shard_group_size={fsdp.shard_group_size}",
                f"fsdp.checkpoint_every_n_blocks={fsdp.checkpoint_every_n_blocks}",
                f"fsdp.reshard_after_forward={fsdp.reshard_after_forward}",
                f"fsdp.routed_expert_chunk_mib={fsdp.routed_expert_chunk_mib}",
                f"fsdp.optim_cpu_offload={fsdp.optim_cpu_offload}",
                f"fsdp.activation_offload={fsdp.activation_offload}",
                f"fsdp.cpu_offload={fsdp.cpu_offload}",
            ]
            if fsdp.tp > 1:
                parts.append(f"fsdp.shard_frozen_tp={fsdp.shard_frozen_tp}")
            if self.gradient_checkpointing and self.moe_collectives_saveable:
                parts.append(f"fsdp.save_moe_collectives={fsdp.save_moe_collectives}")
        return ", ".join(parts)


class TrainingMemoryUnfitError(ValueError):
    """Nothing fits one GPU, even at the lightest settings.

    :param plan: The lightest plan, priced, for reporting how far over it is.
    :param device: The training GPU.
    """

    def __init__(self, plan: TrainingMemoryPlan, device: DeviceSpec) -> None:
        breakdown = plan.breakdown
        host_usable = breakdown.host_usable_bytes
        if host_usable is None or breakdown.host_fits_with_buffer:
            over = (
                f"Estimated training peak {breakdown.total_bytes / GiB:.2f} GiB is "
                f"over the {breakdown.device_usable_bytes / GiB:.2f} GiB usable on "
                f"one {device.name}"
            )
            hint = "Use more or larger GPUs per agent, or shorten the context length."
            failure = "gpu_memory"
        else:
            over = (
                f"Estimated host RAM {breakdown.host_total_bytes / GiB:.2f} GiB per "
                f"trainer is over its {host_usable / GiB:.2f} GiB "
                "share of the node"
            )
            hint = "Add host RAM, or shorten the context length."
            failure = "host_memory"
        super().__init__(f"{over} at the lightest settings ({plan.describe()}). {hint}")
        self.plan = plan
        self.failure = failure


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

    Last resorts, in order: recomputing the MoE collectives checkpointing
    would save (MoE with expert parallel or a MoE tensor-parallel plan only),
    FSDP-sharding frozen tensor-parallel weights (``tp > 1`` only), a smaller
    routed-expert chunk (MoE only), the optimizer on CPU, activation offload,
    then full CPU offload of parameters and gradients. Full CPU offload turns
    optimizer offload off (the two are exclusive) and needs colocated vLLM,
    as CPUOffloadPolicy keeps parameters on CPU. Set fields never move.
    """
    saveable = fsdp.save_moe_collectives is None and moe_collectives_saveable(
        model.arch, fsdp.ep, training_tensor_parallel(model.arch, fsdp)
    )
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
        shard_frozen_tp=bool(fsdp.shard_frozen_tp),
        save_moe_collectives=True if saveable else fsdp.save_moe_collectives,
    )
    rungs: list[tuple[str | None, FSDPConfig]] = [(None, current)]
    if saveable:
        current = replace(current, save_moe_collectives=False)
        rungs.append(("fsdp.save_moe_collectives=False", current))
    if fsdp.shard_frozen_tp is None and fsdp.tp > 1:
        current = replace(current, shard_frozen_tp=True)
        rungs.append(("fsdp.shard_frozen_tp=True", current))
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
        and current.ep == 1
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
    timing: StepTiming | None = None,
) -> TrainingMemoryPlan:
    """The fastest plan that fits: DDP or FSDP, its speed rungs, then FSDP memory-only last resorts.

    Speed rungs are searched jointly and the fitting combination with the
    lowest :func:`learn_cost` wins, or with ``timing`` the fewest learn
    seconds counting traffic between nodes (:func:`cross_node_seconds`;
    ties go to less recompute, then fewer gathers): the parallelism (DDP when the run's ``fsdp`` allows it, see
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
    :param timing: Ranks plans by learn seconds and prices their traffic
        between nodes; ``None`` ranks by learn work.
    :return: The chosen plan, with every field set; the caller's own copy.
    :raises TrainingMemoryUnfitError: When the lightest plan is over the GPU.
    """
    plan, fits, notes = _cached_training_memory(
        model, device, TrainingSettingsKey.of(settings), pins, mini_batch_rows, timing
    )
    for message, args in notes:
        logger.info(message, *args)
    plan = plan.detached()
    if not fits:
        raise TrainingMemoryUnfitError(plan, device)
    return plan


def lightest_training_memory(
    model: ModelSpec,
    device: DeviceSpec,
    settings: TrainingSettings,
    pins: MemoryPins,
    mini_batch_rows: int,
) -> TrainingMemoryPlan:
    """The plan :func:`resolve_training_memory` falls back to: every memory rung, the most recompute, the smallest micro-batch.

    :param model: Training-side model spec.
    :param device: Training GPU.
    :param settings: Training settings, as :func:`resolve_training_memory` reads them.
    :param pins: Checkpointing and micro-batch values the search keeps.
    :param mini_batch_rows: Rows one optimizer step covers on one rank.
    :return: The plan, which may not fit; the caller's own copy.
    """
    plan, _, _ = _cached_training_memory(
        model,
        device,
        TrainingSettingsKey.of(settings),
        pins,
        mini_batch_rows,
        lightest_only=True,
    )
    return plan.detached()


# ~23 KiB per entry; 512 holds the plans of ~8 requests.
@lru_cache(maxsize=512)
def _cached_training_memory(
    model: ModelSpec,
    device: DeviceSpec,
    key: TrainingSettingsKey,
    pins: MemoryPins,
    mini_batch_rows: int,
    timing: StepTiming | None = None,
    lightest_only: bool = False,
) -> tuple[TrainingMemoryPlan, bool, tuple[LogNote, ...]]:
    """:func:`resolve_training_memory`'s search, run once per input.

    :param timing: As in :func:`resolve_training_memory`.
    :param lightest_only: Skip the search and return the lightest plan.
    :return: The chosen plan and ``True``, or the lightest plan and ``False``
        when it is over the GPU, and the log records the search emits.
    """
    settings = key.settings
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
    expert_lora = settings.lora_packed_target_matrices > 0
    notes: list[LogNote] = []

    trainers = settings.n_training_gpus
    node = device.gpus_per_node or trainers

    def passes_at(unit: int) -> int:
        return math.ceil(tokens / unit) if packed else mini_batch_rows // unit

    def rank(plan: TrainingMemoryPlan) -> float:
        return plan.cost if timing is None else timing.learn_seconds(plan)

    def rank_bound(
        cost: float, on: bool, stride: int, trial: FSDPConfig | None, passes: int
    ) -> float:
        # A candidate's fewest passes bound both its learn work and its gathers.
        if timing is None:
            return cost
        traffic = _cross_node_seconds(
            model,
            trial,
            settings,
            tokens,
            passes,
            on,
            stride,
            trainers,
            node,
            timing.no_grad_sweeps,
            costs,
        )
        return (
            max(timing.compute_seconds(cost, tokens), traffic.gathers)
            + traffic.exchange
            + traffic.grad_sync
        )

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
        else:
            rows, budget = unit, None
        passes = passes_at(unit)
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
            expert_lora,
            tokens / passes,
            DTYPE_BYTES[settings.weight_dtype],
            DEFAULT_ROUTED_CHUNK_BYTES if chunk_mib is None else chunk_mib * MiB,
        )
        breakdown = shared_training_estimate(model, device, sized, orchestrated=True)
        plan = TrainingMemoryPlan(
            gradient_checkpointing=on,
            checkpoint_every_n_blocks=stride,
            micro_batch_size_per_gpu=rows,
            micro_batch_tokens_per_gpu=budget,
            fsdp=trial,
            routed_expert_chunk_mib=ddp_chunk,
            moe_collectives_saveable=trial is not None
            and moe_collectives_saveable(
                model.arch, trial.ep, training_tensor_parallel(model.arch, trial)
            ),
            cost=learn_cost(
                n_layers, on, stride, passes, tokens, residency, costs, chunks
            ),
            tokens=tokens,
            settings=sized,
            breakdown=breakdown,
            optimizer_offload_seconds=costs.optimizer_offload_seconds_per_gib
            * offloaded_optimizer_bytes(trial, breakdown)
            / GiB,
        )
        if timing is None:
            return plan
        return replace(
            plan,
            cross_node=cross_node_seconds(
                model, plan, trainers, node, timing.no_grad_sweeps, costs
            ),
        )

    def largest_fit(
        on: bool,
        stride: int,
        residency: Residency | None,
        trial: FSDPConfig | None,
        ddp_chunk: int | None,
    ) -> TrainingMemoryPlan | None:
        # Memory grows with the unit, so the units that fit are a prefix. A
        # packed mini-batch has thousands of budgets; memory is close to linear
        # in the unit, so the next probe goes where the spare bytes run out.
        low = priced(on, stride, residency, trial, ddp_chunk, units[0])
        if not low.breakdown.fits_with_buffer:
            return None
        if len(units) == 1:
            return low
        high = priced(on, stride, residency, trial, ddp_chunk, units[-1])
        if high.breakdown.fits_with_buffer:
            return high
        low_index, high_index = 0, len(units) - 1
        interpolate = True
        while high_index - low_index > 1:
            spare, over = buffer_room(low.breakdown), buffer_room(high.breakdown)
            if interpolate and spare >= 0 > over:
                crossing = units[low_index] + (
                    units[high_index] - units[low_index]
                ) * spare / (spare - over)
                middle = bisect_right(units, crossing) - 1
                middle = min(max(middle, low_index + 1), high_index - 1)
            else:
                middle = (low_index + high_index) // 2
            width = high_index - low_index
            plan = priced(on, stride, residency, trial, ddp_chunk, units[middle])
            if plan.breakdown.fits_with_buffer:
                low, low_index = plan, middle
            else:
                high, high_index = plan, middle
            # Bisect after a probe that did not halve the bracket.
            interpolate = 2 * (high_index - low_index) <= width
        return low

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
        # The most recompute and the lightest residency at the smallest unit
        # is this rung's lightest plan; nothing else fits when it does not.
        on, stride = strides[-1]
        residency, trial = placements(memory)[-1]
        lightest = priced(on, stride, residency, trial, ddp_chunk, units[0])
        if not lightest.breakdown.fits_with_buffer:
            return None
        # No plan costs less than its fewest passes, each with one routed chunk
        # per MoE layer; ties go to the earlier candidate.
        fewest_chunks = (
            model.arch.moe_layers
            if expert_lora and model.arch.routed_expert_recompute
            else 0
        )
        fewest_passes = passes_at(units[-1])
        candidates = [
            (
                rank_bound(
                    learn_cost(
                        n_layers,
                        on,
                        stride,
                        fewest_passes,
                        tokens,
                        residency,
                        costs,
                        fewest_chunks,
                    ),
                    on,
                    stride,
                    trial,
                    fewest_passes,
                ),
                index,
                (on, stride, residency, trial),
            )
            for index, ((on, stride), (residency, trial)) in enumerate(
                product(strides, placements(memory))
            )
        ]
        best: tuple[float, int, TrainingMemoryPlan] | None = None
        for bound, index, candidate in sorted(candidates, key=lambda c: c[:2]):
            if best is not None and (bound, index) >= best[:2]:
                break
            plan = largest_fit(*candidate, ddp_chunk)
            if plan is not None and (best is None or (rank(plan), index) < best[:2]):
                best = (rank(plan), index, plan)
        return None if best is None else best[2]

    def data_parallel(strides: list[tuple[bool, int]]) -> TrainingMemoryPlan | None:
        for index, chunk in enumerate(ddp_chunks):
            if index > 0:
                notes.append(
                    (
                        (
                            "Training memory: DDP turning on routed_expert_chunk_mib=%s; "
                            "nothing fits one %s at %s"
                        ),
                        (
                            chunk,
                            device.name,
                            floor(None, ddp_chunks[index - 1]).describe(),
                        ),
                    )
                )
            plan = cheapest(strides, None, chunk)
            if plan is not None:
                return plan
        return None

    rungs = [] if fsdp is None else _memory_rungs(model, settings, fsdp)

    def lightest_plan() -> TrainingMemoryPlan:
        if fsdp is None:
            return floor(None, ddp_chunks[-1])
        return floor(rungs[-1][1], None)

    if lightest_only:
        lightest = lightest_plan()
        return lightest, lightest.breakdown.fits, ()

    def fastest() -> TrainingMemoryPlan | None:
        if fsdp is None:
            return data_parallel(recompute)
        ddp = (
            data_parallel(
                _recompute_candidates(pins.gradient_checkpointing, None, False)
            )
            if can_train_data_parallel(fsdp)
            else None
        )
        for index, (resort, memory) in enumerate(rungs):
            if index > 0:
                notes.append(
                    (
                        "Training memory: turning on %s; nothing fits one %s at %s",
                        (
                            resort,
                            device.name,
                            floor(rungs[index - 1][1], None).describe(),
                        ),
                    )
                )
            plan = cheapest(recompute, memory, None)
            # A fitting DDP plan beats every FSDP last resort.
            if ddp is not None and (plan is None or rank(ddp) <= rank(plan)):
                return ddp
            if plan is not None:
                return plan
        return None

    host_bytes = device.host_bytes_per_gpu
    # Every plan holds these host bytes, so none fits with the buffer when they do not.
    hopeless = host_bytes is not None and not recommendation_fits(
        fixed_host_bytes(lightest_plan().breakdown), host_bytes
    )
    if not hopeless and (plan := fastest()) is not None:
        return plan, True, tuple(notes)
    lightest = lightest_plan()
    if lightest.breakdown.fits:
        notes.append(
            (
                "Training memory: %s fits one %s only inside the underprediction buffer",
                (lightest.describe(), device.name),
            )
        )
        return lightest, True, tuple(notes)
    return lightest, False, tuple(notes)
