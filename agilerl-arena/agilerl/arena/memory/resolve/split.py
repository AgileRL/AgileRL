# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose an async member's trainer / rollout-engine GPU split and its training-memory plan."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.formulas import (
    engine_weight_bytes,
    param_counts,
    weight_bytes,
)
from agilerl.arena.memory.kv_cache import (
    aligned_kv_block_size,
    kv_cache_bytes_per_token,
)
from agilerl.arena.memory.manifest import (
    engine_weight_quantization,
    rollout_logit_bias,
    speculative_drafting,
    training_memory_from_manifest,
    training_settings_from_manifest,
    unsupported_gpu_reason,
)
from agilerl.arena.memory.resolve.batch import ROW_FILL, rollout_group_size
from agilerl.arena.memory.resolve.decode_costs import (
    DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
    MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
    decode_costs,
)
from agilerl.arena.memory.resolve.engine import servable_engine_widths
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
)
from agilerl.arena.memory.resolve.rollout import (
    default_rollout_batch_size,
    drafts_by_default,
)
from agilerl.arena.memory.resolve.sequences import resolve_gpu_memory_utilization
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    ModelArch,
    ModelSpec,
    TrainingSettings,
    WeightVariant,
)
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest

# The efficiencies below are fitted to Qwen3-4B LoRA (FSDP2, vLLM TP=1) on
# 8x A100-80: GRPO countdown (1k-token
# single-turn rows) and CISPO Sudoku-v0-hard (9k-token rows, about 12 turns),
# each run at 4T4E/5T3E/6T2E and 6T2E/4T4E/3T5E.

# Share of peak bf16 the learn step's matmuls reach.
LEARN_MATMUL_EFFICIENCY = 0.62
# Share of peak bf16 the learn step's causal attention reaches.
LEARN_ATTENTION_EFFICIENCY = 0.11
# Share of peak HBM bandwidth a decode step reads weights and resident KV at.
DECODE_BANDWIDTH_EFFICIENCY = 0.53
# Seconds a multi-turn episode spends between turns: env step, prefill, requeue.
TURN_OVERHEAD_SECONDS = 3.5
# Multi-turn cycle time over the mean episode's, from slow episodes.
MULTI_TURN_STRAGGLER_FACTOR = 1.22
# Share of a multi-turn row the model decodes; the rest is prompt and env text.
MULTI_TURN_DECODE_SHARE = 0.5
# Multi-turn generation time without drafting over its time with n-gram drafts:
# 1.9x on Sudoku at 4T4E on both A100 and H100 (about 2.7 tokens accepted per step).
NGRAM_GENERATION_SPEEDUP = 1.9
# Tensor-parallel time per decode step over the ring all-reduce model: vLLM's
# per-step sync, kernel launches and custom all-reduce.
TENSOR_PARALLEL_OVERHEAD_FACTOR = 3.0
# Engine widths whose predicted step is within this share of the fastest tie,
# and the narrowest of them is chosen.
ENGINE_WIDTH_TIE_SHARE = 0.02


class SplitWorkload(BaseModel):
    """One member's learn step and rollout cycle, at expected lengths."""

    model_config = ConfigDict(frozen=True)

    # Groups one learn step trains.
    batch_size: int
    group_size: int
    # Mean tokens in one trained row, and the share of them an engine decodes.
    row_tokens: int
    decode_tokens: int
    turns: int
    # Forward-equivalents per trained token (no-grad, forward, backward).
    learn_passes: int
    # Pinned groups per rollout cycle; ``None`` rounds ``batch_size`` up to
    # ``lcm(trainers, engines)``.
    rollout_batch_size: int | None = None
    # Pinned per-engine sequence cap; ``None`` runs every engine episode at once.
    max_num_seqs: int | None = None
    gpu_memory_utilization: float = 0.9
    # Trainer counts must be a multiple of this (FSDP tp, ep, shard group).
    trainer_multiple: int = 1
    # Trainer ranks per data shard (FSDP tp).
    trainer_tensor_parallel: int = 1
    # Trajectories per data shard must be a multiple of this (pinned micro/mini).
    per_rank_multiple: int = 1
    # Rollout-engine weight format.
    weight_quantization: Literal["none", "fp8"] = "none"
    # The rollout algorithm's ``min_p``.
    min_p: float = 0.0
    # Rollout requests carry a ``logit_bias`` masking the image token.
    logit_bias: bool = False
    # ``vllm_engine_args.speculative_config`` set (``True``) or null
    # (``False``); ``None`` takes the engine's multi-turn default.
    speculative_drafting: bool | None = None


class GpuSplit(BaseModel):
    """Trainers and engines for one member, with the predicted step."""

    model_config = ConfigDict(frozen=True)

    trainers: int
    engines: int
    # GPUs per engine (vLLM tensor parallel).
    engine_width: int
    rollout_batch_size: int
    learn_seconds: float
    # Engine time to generate one learn step's groups.
    generation_seconds: float

    @property
    def step_seconds(self) -> float:
        """Learn and generation overlap, so the slower side sets the step."""
        return max(self.learn_seconds, self.generation_seconds)


def active_matmul_params(arch: ModelArch) -> int:
    """Weights each token multiplies through: projections, routed-in experts and the LM head."""
    counts = param_counts(arch)
    inactive_experts = 0
    if arch.is_moe:
        matrices = 3 if arch.gated_mlp else 2
        expert_inter = arch.expert_intermediate_size or arch.intermediate_size
        idle_per_layer = arch.n_experts - (arch.n_experts_per_tok or 1)
        inactive_experts = (
            arch.moe_layers
            * idle_per_layer
            * matrices
            * arch.expert_width
            * expert_inter
        )
    lm_head = arch.vocab_size * arch.hidden_size
    return (
        counts.attention
        + counts.mlp
        + counts.mamba
        + counts.moe_experts
        - inactive_experts
        + lm_head
    )


def learn_forward_passes(settings: TrainingSettings) -> int:
    """Forward-equivalents a learn step spends per trained token.

    The backward of frozen base weights costs one forward (input grads only),
    and one no-grad pass scores both old and reference log-probs. Recompute
    and per-micro-batch costs come from the training-memory plan.

    :param settings: Trainer settings.
    :return: Forward and backward, plus the no-grad pass when run.
    """
    return 2 + int(settings.has_nograd_pass)


def _learn_gpu_seconds(
    model: ModelSpec, peak_flops: float, workload: SplitWorkload
) -> float:
    """GPU-seconds one learn step costs, summed over trainer ranks."""
    arch = model.arch
    rows = workload.batch_size * workload.group_size
    tokens = workload.row_tokens
    matmul = 2 * active_matmul_params(arch) * tokens / LEARN_MATMUL_EFFICIENCY
    attention = (
        2
        * arch.attention_layers
        * arch.n_heads
        * arch.head_dim
        * tokens**2
        / LEARN_ATTENTION_EFFICIENCY
    )
    return rows * workload.learn_passes * (matmul + attention) / peak_flops


def _engine_weight_bytes(
    model: ModelSpec, device: DeviceSpec, workload: SplitWorkload
) -> tuple[float, float]:
    """Engine weight bytes, and the bf16 bytes a decode step takes as long to read.

    fp8 weights are sized by :func:`engine_weight_bytes` and read at
    :func:`decode_costs`'s measured speedup.
    """
    counts = param_counts(model.arch, model.n_params)
    weights = weight_bytes(counts, model.checkpoint_dtype, WeightVariant())
    if workload.weight_quantization == "none":
        return weights, weights
    costs = decode_costs(device)
    speedup = (
        costs.fp8_moe_weight_read_speedup
        if model.arch.is_moe
        else costs.fp8_weight_read_speedup
    )
    stored = engine_weight_bytes(
        counts, model.checkpoint_dtype, WeightVariant(), workload.weight_quantization
    )
    return stored, weights / speedup


def tensor_parallel_seconds(
    model: ModelSpec, device: DeviceSpec, engine_width: int, sequences: int
) -> float:
    """Seconds one decode step of an engine spends on tensor parallel.

    vLLM all-reduces each row-parallel output: the token mixer and the MLP or
    MoE of every layer, or one per layer where each layer is a single kind.
    Each all-reduce carries one ``hidden_size`` activation per running
    sequence; a ring moves ``2 (width - 1) / width`` of it over the bus. The
    ring time is scaled by :data:`TENSOR_PARALLEL_OVERHEAD_FACTOR`.

    :param model: Model the engine serves.
    :param device: One GPU of the engine.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param sequences: Sequences the engine decodes in the step.
    :return: 0 for a one-GPU engine.
    :raises ValueError: When a wider engine's GPU has no all-reduce figures.
    """
    if engine_width == 1:
        return 0.0
    latency = device.all_reduce_latency_seconds
    bus_bandwidth = device.all_reduce_bytes_per_second
    if latency is None or bus_bandwidth is None:
        msg = (
            f"{device.name or 'This GPU'} has no all-reduce figures in the GPU "
            "catalogue, so a tensor-parallel engine cannot be priced. Set "
            "vllm_config.tensor_parallel_size to 1."
        )
        raise ValueError(msg)
    arch = model.arch
    single_kind_layers = arch.n_mlp_layers is not None or arch.n_moe_layers is not None
    all_reduces = arch.n_layers if single_kind_layers else 2 * arch.n_layers
    message = sequences * arch.hidden_size * DTYPE_BYTES[model.checkpoint_dtype]
    ring_share = 2 * (engine_width - 1) / engine_width
    ring = all_reduces * (latency + message * ring_share / bus_bandwidth)
    return TENSOR_PARALLEL_OVERHEAD_FACTOR * ring


def _engine_prefill_rate(
    model: ModelSpec, peak_flops: float, engine_width: int
) -> float:
    """Prompt tokens per second one engine prefills while it decodes alongside.

    The engine's peak bf16 at the MoE or dense prefill efficiency, over two
    FLOPs per active matmul weight.
    """
    efficiency = (
        MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY
        if model.arch.is_moe
        else DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY
    )
    return (
        efficiency * engine_width * peak_flops / (2 * active_matmul_params(model.arch))
    )


def uncached_prefill_tokens(
    model: ModelSpec, workload: SplitWorkload, engine_width: int
) -> int:
    """Prompt and env tokens one episode prefills with every earlier turn prefix-cached.

    Each turn prefills the tokens added since the last turn, rounded up to
    whole KV blocks of :func:`aligned_kv_block_size`; the prompt and env
    tokens are spread evenly over the turns.

    :param model: Model the engine serves.
    :param workload: The member's rollout cycle.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :return: Tokens prefilled over the episode.
    """
    block = aligned_kv_block_size(model.arch, model.checkpoint_dtype, engine_width)
    added = workload.row_tokens - workload.decode_tokens
    return workload.turns * block * math.ceil(added / workload.turns / block)


def _generation_seconds(
    model: ModelSpec,
    device: DeviceSpec,
    peak_flops: float,
    bandwidth: float,
    workload: SplitWorkload,
    engines: int,
    engine_width: int,
    rollout_batch_size: int,
) -> float:
    """Seconds the engines take to produce one rollout cycle.

    The slower of two bounds: decode steps read at memory bandwidth, one per
    decoded token per wave of resident episodes, and every episode's
    :func:`uncached_prefill_tokens` at the engine's prefill rate, which
    decoded tokens ride along with. Only multi-turn cycles price n-gram
    drafting; single-turn completions share too little with their prompt to
    accept drafts.
    """
    arch = model.arch
    weights, weight_read = _engine_weight_bytes(model, device, workload)
    kv_per_token = kv_cache_bytes_per_token(arch, model.checkpoint_dtype)
    multi_turn = workload.turns > 1
    # Context grows across turns; a single turn decodes after its whole prompt.
    context = (
        workload.row_tokens / 2
        if multi_turn
        else workload.row_tokens - workload.decode_tokens / 2
    )
    kv_pool = (
        engine_width * workload.gpu_memory_utilization * device.total_bytes - weights
    )
    seqs = math.ceil(rollout_batch_size / engines) * workload.group_size
    slots = min(
        seqs,
        workload.max_num_seqs or seqs,
        int(kv_pool // (context * kv_per_token)),
    )
    if slots < 1:
        return math.inf
    decode_step = (weight_read + slots * context * kv_per_token) / (
        DECODE_BANDWIDTH_EFFICIENCY * engine_width * bandwidth
    ) + tensor_parallel_seconds(model, device, engine_width, slots)
    episode = (
        workload.decode_tokens * decode_step
        + (workload.turns - 1) * TURN_OVERHEAD_SECONDS
    )
    decode_bound = math.ceil(seqs / slots) * episode
    if multi_turn:
        drafts = (
            drafts_by_default(
                multi_turn, arch.is_hybrid_ssm, workload.min_p, workload.logit_bias
            )
            if workload.speculative_drafting is None
            else workload.speculative_drafting
        )
        speedup = NGRAM_GENERATION_SPEEDUP if drafts else 1.0
        decode_bound *= MULTI_TURN_STRAGGLER_FACTOR / speedup
    prefill_bound = (
        seqs
        * uncached_prefill_tokens(model, workload, engine_width)
        / _engine_prefill_rate(model, peak_flops, engine_width)
    )
    return max(decode_bound, prefill_bound)


def _cycle_rollout_batch_size(
    workload: SplitWorkload, trainers: int, engines: int
) -> int | None:
    """Groups per rollout cycle, or None when the trainer count cannot shard the batch."""
    if trainers % workload.trainer_multiple:
        return None
    shards = trainers // workload.trainer_tensor_parallel
    if workload.batch_size % shards:
        return None
    if (
        workload.batch_size // shards * workload.group_size
    ) % workload.per_rank_multiple:
        return None
    if workload.rollout_batch_size is not None:
        return (
            None
            if workload.rollout_batch_size % trainers
            else workload.rollout_batch_size
        )
    return default_rollout_batch_size(workload.batch_size, trainers, engines)


def choose_gpu_split(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    gpus: int,
    engine_width: int,
    learn_cost_ratio: Callable[[int], float | None],
) -> GpuSplit | None:
    """Split one member's GPUs so the slower of learn and generation is fastest.

    Ties go to fewer idle GPUs, then fewer engines, so an engine that does not
    shorten the cycle goes to training.

    :param model: Model the trainers and engines run.
    :param device: One GPU of the allocation.
    :param workload: The member's learn step and rollout cycle.
    :param gpus: GPUs the member owns.
    :param engine_width: GPUs per rollout engine (vLLM tensor parallel).
    :param learn_cost_ratio: Trainer count to its training-memory plan's
        :attr:`TrainingMemoryPlan.learn_cost_ratio`, or ``None`` when no
        plan fits that many trainers.
    :return: The fastest valid split, or None when no split fits.
    :raises ValueError: When the device has no peak bf16 or bandwidth figure.
    """
    peak_flops = device.peak_bf16_flops
    bandwidth = device.memory_bytes_per_second
    if peak_flops is None or bandwidth is None:
        msg = (
            f"{device.name or 'This GPU'} has no peak bf16 or memory bandwidth "
            "figure in the GPU catalogue, so the trainer / rollout-engine split "
            "cannot be searched. Set training.training_gpus_per_agent and "
            "training.rollout_engines_per_agent."
        )
        raise ValueError(msg)
    learn_gpu_seconds = _learn_gpu_seconds(model, peak_flops, workload)
    best: tuple[tuple[float, int, int], GpuSplit] | None = None
    for engines in range(1, (gpus - 1) // engine_width + 1):
        for trainers in range(1, gpus - engines * engine_width + 1):
            rollout_batch_size = _cycle_rollout_batch_size(workload, trainers, engines)
            if rollout_batch_size is None:
                continue
            ratio = learn_cost_ratio(trainers)
            if ratio is None:
                continue
            cycles_per_step = workload.batch_size / rollout_batch_size
            split = GpuSplit(
                trainers=trainers,
                engines=engines,
                engine_width=engine_width,
                rollout_batch_size=rollout_batch_size,
                learn_seconds=learn_gpu_seconds * ratio / trainers,
                generation_seconds=cycles_per_step
                * _generation_seconds(
                    model,
                    device,
                    peak_flops,
                    bandwidth,
                    workload,
                    engines,
                    engine_width,
                    rollout_batch_size,
                ),
            )
            idle = gpus - trainers - engines * engine_width
            rank = (split.step_seconds, idle, engines)
            if best is None or rank < best[0]:
                best = (rank, split)
    return None if best is None else best[1]


def split_workload_from_manifest(manifest: TrainingManifest) -> SplitWorkload:
    """Expected lengths and batch constraints from a rollout manifest's budgets.

    :param manifest: Validated async rollout manifest.
    :return: The workload at :data:`ROW_FILL` of the row budget.
    :raises TypeError: When the algorithm does not generate rollouts.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec):
        msg = f"{spec.name} does not generate rollouts"
        raise TypeError(msg)
    row_budget = spec.max_row_tokens or spec.max_model_len
    row_tokens = math.ceil(ROW_FILL * row_budget)
    turn_tokens = spec.max_output_tokens or row_budget
    env = manifest.environment
    max_turns = env.max_turns if isinstance(env, LLMEnvSpec) else 1
    if max_turns == 1:
        decode_tokens = min(turn_tokens, row_tokens)
        turns = 1
    else:
        decode_tokens = math.ceil(MULTI_TURN_DECODE_SHARE * row_tokens)
        turns = max(2, math.ceil(row_tokens / turn_tokens))
        if max_turns is not None:
            turns = min(turns, max_turns)
    vllm = spec.vllm_config
    fsdp = spec.fsdp
    fsdp_tp = 1 if fsdp is None else fsdp.tp
    trainer_multiple = (
        1 if fsdp is None else math.lcm(fsdp.tp, fsdp.ep, fsdp.shard_group_size or 1)
    )
    pinned_rows = [
        size
        for size in (spec.micro_batch_size_per_gpu, spec.mini_batch_size)
        if size is not None
    ]
    return SplitWorkload(
        batch_size=spec.batch_size,
        group_size=rollout_group_size(spec),
        row_tokens=row_tokens,
        decode_tokens=decode_tokens,
        turns=turns,
        learn_passes=learn_forward_passes(
            training_settings_from_manifest(manifest, micro_batch_size=1)
        ),
        rollout_batch_size=manifest.training.rollout_batch_size,
        max_num_seqs=None if vllm is None else vllm.max_num_seqs,
        gpu_memory_utilization=resolve_gpu_memory_utilization(
            None if vllm is None else vllm.gpu_memory_utilization, colocated=False
        ),
        trainer_multiple=trainer_multiple,
        trainer_tensor_parallel=fsdp_tp,
        per_rank_multiple=math.lcm(*pinned_rows) if pinned_rows else 1,
        weight_quantization=engine_weight_quantization(
            None if vllm is None else vllm.quantization
        ),
        min_p=spec.min_p,
        logit_bias=rollout_logit_bias(spec),
        speculative_drafting=speculative_drafting(spec),
    )


def choose_engine_width(splits: list[GpuSplit]) -> GpuSplit | None:
    """Pick the fastest split; widths within :data:`ENGINE_WIDTH_TIE_SHARE` of it tie to the narrowest.

    :param splits: The best split at each engine width, narrowest first.
    :return: The chosen split, or None when there are none.
    """
    if not splits:
        return None
    fastest = min(split.step_seconds for split in splits)
    return next(
        split
        for split in splits
        if split.step_seconds <= (1 + ENGINE_WIDTH_TIE_SHARE) * fastest
    )


@dataclass(frozen=True)
class RunMemoryPlan:
    """The GPU split and trainer memory plan a run starts with."""

    # ``None`` when the manifest's trainer and engine counts are kept.
    split: GpuSplit | None
    training: TrainingMemoryPlan


def resolve_run_memory(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int | None,
    gpus_per_node: int | None = None,
) -> RunMemoryPlan:
    """Choose the GPU split, then the trainer memory plan.

    Each candidate trainer count is priced with its own
    :func:`training_memory_from_manifest` plan, so the split weighs learn
    time at the speed rungs that fit that many trainers. A searched split
    with the engine width omitted is searched at every width one engine can
    serve a sequence at and picked by :func:`choose_engine_width`; a pinned
    width is searched alone.

    :param manifest: Validated run manifest. Not written.
    :param model: Training-side catalog model.
    :param device: One GPU of the allocation.
    :param member_gpus: GPUs one member owns, to search the split; ``None``
        keeps the manifest's trainer and engine counts.
    :param gpus_per_node: GPUs on one node; an engine stays on one node.
        ``None`` puts the member's GPUs on one node.
    :return: The plan.
    :raises ValueError: When the GPU cannot run LLM training, or the split is
        searched and none fits.
    :raises TrainingMemoryUnfitError: When nothing fits one trainer GPU.
    """
    unsupported = unsupported_gpu_reason(device)
    if unsupported is not None:
        raise ValueError(unsupported)
    chosen = manifest.model_copy(deep=True)
    split = None
    plans: dict[int, TrainingMemoryPlan | None] = {}
    if member_gpus is not None:
        spec = manifest.algorithm
        if not isinstance(spec, RolloutLLMSpec) or spec.vllm_config is None:
            msg = f"{spec.name} does not generate rollouts; it has no GPU split"
            raise TypeError(msg)
        pinned_width = spec.vllm_config.tensor_parallel_size
        if pinned_width is None:
            widest = min(member_gpus - 1, gpus_per_node or member_gpus)
            engine_widths = servable_engine_widths(manifest, device, widest)
        else:
            engine_widths = [pinned_width]

        def learn_cost_ratio(trainers: int) -> float | None:
            if trainers not in plans:
                trial = manifest.model_copy(deep=True)
                trial.training.training_gpus_per_agent = trainers
                try:
                    plans[trainers] = training_memory_from_manifest(
                        trial, model, device
                    )
                except TrainingMemoryUnfitError:
                    plans[trainers] = None
            plan = plans[trainers]
            return None if plan is None else plan.learn_cost_ratio

        workload = split_workload_from_manifest(manifest)
        splits = [
            found
            for width in engine_widths
            if (
                found := choose_gpu_split(
                    model, device, workload, member_gpus, width, learn_cost_ratio
                )
            )
            is not None
        ]
        split = choose_engine_width(splits)
        if split is None:
            msg = (
                f"No trainer / rollout-engine split of {member_gpus} "
                f"{device.name} GPUs fits this run. Set "
                "training.training_gpus_per_agent and "
                "training.rollout_engines_per_agent."
            )
            raise ValueError(msg)
        chosen.training.training_gpus_per_agent = split.trainers
        chosen.training.rollout_engines_per_agent = split.engines
        chosen.training.rollout_batch_size = split.rollout_batch_size
    training = plans.get(chosen.training.training_gpus_per_agent)
    if training is None:
        training = training_memory_from_manifest(chosen, model, device)
    return RunMemoryPlan(split=split, training=training)
