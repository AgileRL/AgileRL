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
    block_recompute_bytes,
    engine_weight_bytes,
    param_counts,
    weight_bytes,
)
from agilerl.arena.memory.generation import engine_step_buffer_bytes
from agilerl.arena.memory.kv_cache import (
    ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS,
    aligned_kv_block_size,
    hybrid_kv_demand_bytes,
    kv_cache_bytes_per_token,
    kv_heads_per_rank,
    mamba_state_bytes,
    shares_hybrid_pool,
)
from agilerl.arena.memory.manifest import (
    can_sync_merged_lora,
    engine_weight_quantization,
    manifest_max_num_batched_tokens,
    rollout_logit_bias,
    speculative_drafting,
    training_memory_from_manifest,
    unsupported_gpu_reason,
)
from agilerl.arena.memory.resolve.batch import (
    KV_POOL_FILL,
    rollout_group_size,
)
from agilerl.arena.memory.resolve.decode_costs import (
    DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
    MERGED_DECODE_SPEEDUP,
    MERGED_MOE_PREFILL_SPEEDUP,
    MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
    decode_costs,
)
from agilerl.arena.memory.resolve.engine import servable_engine_widths
from agilerl.arena.memory.resolve.fsdp import (
    TrainingMemoryPlan,
    TrainingMemoryUnfitError,
)
from agilerl.arena.memory.resolve.learn_costs import learn_costs
from agilerl.arena.memory.resolve.rollout import (
    default_rollout_batch_size,
    drafts_by_default,
)
from agilerl.arena.memory.resolve.sequences import resolve_gpu_memory_utilization
from agilerl.arena.memory.resolve.training import lora_target_scope
from agilerl.arena.memory.resolve.weight_sync import (
    SyncLink,
    adapter_sync_seconds,
    merged_sync_seconds,
)
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    LoraTargetScope,
    ModelArch,
    ModelSpec,
    WeightVariant,
)
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.arena.models.env import LLMEnvSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import (
    SUPPORTED_MODEL_INFO,
    resolve_reasoning_parser,
)
from agilerl.arena.models.networks import LoraConfigDict, VLLMConfig
from agilerl.arena.models.training import LLMRolloutBufferSpec

# The efficiencies below are fitted to Qwen3-4B LoRA (FSDP2, vLLM TP=1) on
# 8x A100-80: GRPO countdown (1k-token
# single-turn rows) and CISPO Sudoku-v0-hard (9k-token rows, about 12 turns),
# each run at 4T4E/5T3E/6T2E and 6T2E/4T4E/3T5E.

# Share of peak bf16 the learn step's matmuls reach.
LEARN_MATMUL_EFFICIENCY = 0.62
# The same share for a MoE model's matmuls, routed experts included: Qwen3.6-35B-A3B
# CISPO LoRA on AWM at 4 FSDP trainers on H100 (12.7k-token rows, learn tokens
# balanced across ranks) learns in 453 s.
MOE_LEARN_MATMUL_EFFICIENCY = 0.17
# Share of peak bf16 the learn step's causal attention reaches.
LEARN_ATTENTION_EFFICIENCY = 0.15
# Share of peak HBM bandwidth a decode step reads weights and resident KV at.
DECODE_BANDWIDTH_EFFICIENCY = 0.53
# The same share for a MoE engine: 0.26-0.30 from the step times of
# Qwen3.6-35B-A3B LoRA tensor-parallel-2 engines on AWM, 8x H100.
MOE_DECODE_BANDWIDTH_EFFICIENCY = 0.3
# Seconds a multi-turn episode spends between turns: env step, prefill, requeue.
TURN_OVERHEAD_SECONDS = 3.5
# A multi-turn wave's time over the mean episode's, from slow episodes.
MULTI_TURN_STRAGGLER_FACTOR = 1.22
# Share of max_output_tokens a single-turn completion decodes: 0.85-0.86 on
# countdown with Qwen3-4B thinking, 0.28-0.44 with Granite MoEs, which do not.
SINGLE_TURN_REASONING_DECODE_SHARE = 0.85
SINGLE_TURN_DECODE_SHARE = 0.35
# Share of the row budget past max_output_tokens a single-turn prompt fills:
# 0.20-0.22 on countdown.
SINGLE_TURN_PROMPT_FILL = 0.2
# Share of the row budget a multi-turn row fills: 0.50-0.54 on Sudoku-v0-hard
# (Qwen3-4B), 0.38-0.43 on AWM (Qwen3.6-35B-A3B).
MULTI_TURN_ROW_FILL = 0.46
# Share of a multi-turn row the model decodes; the rest is prompt and env text.
MULTI_TURN_DECODE_SHARE = 0.5
# Mean turns of an episode over the env's max_turns: 11-12 of 50 on
# Sudoku-v0-hard, 2-5 of 20 on AWM, rising as the policy trains.
MULTI_TURN_TURN_SHARE = 0.2
# A turn's mean decoded tokens: 380-430 on Sudoku-v0-hard, 220-450 on AWM.
MULTI_TURN_TURN_DECODE_TOKENS = 352
# Multi-turn generation time without drafting over its time with n-gram drafts:
# 1.9x on Sudoku at 4T4E on both A100 and H100 (about 2.7 tokens accepted per step).
NGRAM_GENERATION_SPEEDUP = 1.9
# Tensor-parallel time per decode step over the ring all-reduce model: vLLM's
# per-step sync, kernel launches and custom all-reduce.
TENSOR_PARALLEL_OVERHEAD_FACTOR = 3.0
# Engine widths whose predicted step is within this share of the fastest tie,
# and the narrowest of them is chosen: about the generation error left on AWM
# (Qwen3.6-35B-A3B), where every measured TP2 / TP4 comparison favoured TP2
# by more than predicted.
ENGINE_WIDTH_TIE_SHARE = 0.1
# Merged weight sync is kept while its predicted step is within this share of
# the adapter's. An adapter step the model prices learn-bound still waits on
# slow episodes. On 8x H100, merged was predicted 0.4% slower on AWM
# (Qwen3.6-35B-A3B) and ran 7% faster, and 6.7% slower on countdown
# (Qwen3-4B) and ran 1.6% faster.
WEIGHT_SYNC_TIE_SHARE = 0.1
# Decoded tokens generated per trained token when tied groups are dropped, on
# AWM (Qwen3.6-35B-A3B) with adaptive task sampling and group replay: 1.22 over
# steps 3-8, 1.10 over steps 5-8. Dropped groups are shorter than trained ones.
TIED_DROP_GENERATION_FACTOR = 1.2


class SplitWorkload(BaseModel):
    """One member's learn step and rollout cycle, at expected lengths."""

    model_config = ConfigDict(frozen=True)

    # Groups one learn step trains.
    batch_size: int
    group_size: int
    # Trained groups' worth of rollouts generated per trained group; above 1
    # when tied groups are dropped.
    generated_per_trained: float = 1.0
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
    # FSDP expert-parallel degree; 1 is no expert split.
    trainer_expert_parallel: int = 1
    # FSDP shard-group width; 1 does not constrain the trainer count.
    trainer_shard_group: int = 1
    # Trajectories per data shard must be a multiple of this (pinned micro/mini).
    per_rank_multiple: int = 1
    # Pinned micro/mini that set ``per_rank_multiple``.
    micro_batch_size_per_gpu: int | None = None
    mini_batch_size: int | None = None
    # Rollout-engine weight format.
    weight_quantization: Literal["none", "fp8"] = "none"
    # Engines serve merged weights with LoRA off (:func:`can_sync_merged_lora`).
    merged_lora_sync: bool = False
    # The adapter weight sync sends or merges.
    lora_rank: int = 16
    lora_target_scope: LoraTargetScope = "all-linear"
    lora_packed_target_matrices: int = 0
    # Link the engines read synced weights over; ``None`` leaves sync unpriced.
    sync_link: SyncLink | None = None
    # The rollout algorithm's ``min_p``.
    min_p: float = 0.0
    # Rollout requests carry a ``logit_bias`` masking the image token.
    logit_bias: bool = False
    # ``vllm_engine_args.speculative_config`` set (``True``) or null
    # (``False``); ``None`` takes the engine's multi-turn default.
    speculative_drafting: bool | None = None
    # Pinned engine scheduler-step token cap; ``None`` lets the split pick one.
    max_num_batched_tokens: int | None = None


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
    # Engine scheduler-step token cap the split raised; ``None`` runs the
    # pinned or default step.
    max_num_batched_tokens: int | None = None
    # One weight sync; 0 when the link is unknown.
    sync_seconds: float = 0.0
    merged_lora_sync: bool = False

    @property
    def step_seconds(self) -> float:
        """Learn and generation overlap, so the slower side sets the step."""
        return max(self.learn_seconds, self.generation_seconds)

    @property
    def synced_step_seconds(self) -> float:
        """:attr:`step_seconds` with one weight sync.

        Weight sync blocks the trainer. A merged sync also pauses the
        engines; an adapter sync lets them keep generating.
        """
        if self.merged_lora_sync:
            return self.step_seconds + self.sync_seconds
        return max(self.learn_seconds + self.sync_seconds, self.generation_seconds)


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


def learn_forward_passes(spec: RolloutLLMSpec) -> int:
    """Forward-equivalents an async learn step spends per trained token.

    The backward of frozen base weights costs one forward (input grads only).
    Async GRPO-family runs take old log-probs from the rollout, so their
    no-grad pass runs only for reference log-probs, when ``beta > 0``; other
    algorithms always score old log-probs. Recompute and per-micro-batch
    costs come from the training-memory plan.

    :param spec: The rollout algorithm.
    :return: Forward and backward, plus the no-grad pass when run.
    """
    nograd = not isinstance(spec, GRPOSpec) or spec.beta > 0
    return 2 + int(nograd)


def _learn_gpu_seconds(
    model: ModelSpec, device: DeviceSpec, peak_flops: float, workload: SplitWorkload
) -> float:
    """GPU-seconds one learn step costs, summed over trainer ranks."""
    arch = model.arch
    rows = workload.batch_size * workload.group_size
    tokens = workload.row_tokens
    matmul_efficiency = (
        MOE_LEARN_MATMUL_EFFICIENCY if arch.is_moe else LEARN_MATMUL_EFFICIENCY
    )
    matmul = 2 * active_matmul_params(arch) * tokens / matmul_efficiency
    attention = (
        2
        * arch.attention_layers
        * arch.n_heads
        * arch.head_dim
        * tokens**2
        / LEARN_ATTENTION_EFFICIENCY
    )
    speed = peak_flops * learn_costs(device).peak_speed
    return rows * workload.learn_passes * (matmul + attention) / speed


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


def engine_prefill_rate(
    model: ModelSpec,
    device: DeviceSpec,
    peak_flops: float,
    engine_width: int,
    max_num_batched_tokens: int,
    merged_lora_sync: bool,
) -> float:
    """Prompt tokens per second one engine prefills while it decodes alongside.

    The engine's peak bf16 at the dense prefill efficiency, or the MoE one of
    the nearest fitted width and the largest fitted step not above
    ``max_num_batched_tokens``, over two FLOPs per active matmul weight, plus
    each scheduler step's :func:`tensor_parallel_seconds` over its
    ``max_num_batched_tokens`` tokens. An engine serving merged weights
    computes faster by :data:`MERGED_MOE_PREFILL_SPEEDUP` (MoE) or
    :data:`MERGED_DECODE_SPEEDUP` (dense).

    :param max_num_batched_tokens: Tokens one scheduler step runs.
    :param merged_lora_sync: The engine runs with LoRA off.
    """
    efficiency = DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY
    if model.arch.is_moe:
        by_step = MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY[
            min(
                MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
                key=lambda fitted: (abs(fitted - engine_width), fitted),
            )
        ]
        step = max(
            (fitted for fitted in by_step if fitted <= max_num_batched_tokens),
            default=min(by_step),
        )
        efficiency = by_step[step]
    if merged_lora_sync:
        efficiency *= (
            MERGED_MOE_PREFILL_SPEEDUP if model.arch.is_moe else MERGED_DECODE_SPEEDUP
        )
    compute = 2 * active_matmul_params(model.arch) / (efficiency * peak_flops)
    all_reduce = tensor_parallel_seconds(
        model, device, engine_width, max_num_batched_tokens
    )
    return engine_width / (compute + engine_width * all_reduce / max_num_batched_tokens)


def engine_steps(model: ModelSpec, workload: SplitWorkload) -> tuple[int, ...]:
    """Scheduler-step token caps the split may run an engine at, the fallback first.

    A pinned step is the only one. A MoE engine may raise the
    :data:`ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS` default to any step
    :data:`MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY` is fitted at; a dense engine
    keeps it, as no dense prefill rate is measured by step.

    :param model: Model the engine serves.
    :param workload: The member's rollout cycle.
    :return: The candidate steps, ascending.
    """
    if workload.max_num_batched_tokens is not None:
        return (workload.max_num_batched_tokens,)
    default = ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS
    if not model.arch.is_moe:
        return (default,)
    fitted = {
        step
        for by_step in MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY.values()
        for step in by_step
    }
    return (default, *sorted(step for step in fitted if step > default))


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


def _mean_context(workload: SplitWorkload) -> float:
    """Mean tokens a decoding sequence holds; context grows across turns, a single turn decodes after its whole prompt."""
    if workload.turns > 1:
        return workload.row_tokens / 2
    return workload.row_tokens - workload.decode_tokens / 2


def resident_sequences(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    engine_width: int,
    max_num_batched_tokens: int,
) -> int:
    """Sequences one engine keeps resident.

    Each rank's share of the KV pool, after the weights and the decoder
    activations and :func:`engine_step_buffer_bytes` a step above
    :data:`ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS` adds, over one sequence's KV
    on a rank: whole hybrid blocks and recurrent state, or attention heads
    with replicated heads stored on every rank that holds them. Single-turn
    sequences are counted at their mean context. Multi-turn admission starts
    episodes only while :data:`KV_POOL_FILL` of the pool holds them at the
    final context, the row.

    :param model: Model the engine serves.
    :param device: One GPU of the engine.
    :param workload: The member's rollout cycle.
    :param engine_width: GPUs per engine (vLLM tensor parallel).
    :param max_num_batched_tokens: The engine's scheduler-step token cap.
    :return: The KV fit, capped at a pinned ``max_num_seqs``; 0 when none fits.
    """
    arch = model.arch
    dtype = model.checkpoint_dtype
    act_bytes = DTYPE_BYTES[dtype]
    multi_turn = workload.turns > 1
    context = workload.row_tokens if multi_turn else _mean_context(workload)
    weights, _ = _engine_weight_bytes(model, device, workload)
    raised_activations = block_recompute_bytes(
        arch, 1, max_num_batched_tokens, act_bytes, engine_forward=True
    ) - block_recompute_bytes(
        arch, 1, ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS, act_bytes, engine_forward=True
    )
    step = max(raised_activations, 0) + engine_step_buffer_bytes(
        arch, max_num_batched_tokens, act_bytes
    )
    rank_pool = (
        workload.gpu_memory_utilization * device.total_bytes
        - weights / engine_width
        - step
    )
    if shares_hybrid_pool(arch):
        kv = hybrid_kv_demand_bytes(arch, dtype, 1, math.ceil(context), engine_width)
    else:
        heads = kv_heads_per_rank(arch.n_kv_heads, engine_width)
        kv = context * kv_cache_bytes_per_token(arch, dtype) * heads / arch.n_kv_heads
    sequence = kv + mamba_state_bytes(arch, 1, dtype, engine_width)
    fill = KV_POOL_FILL if multi_turn else 1.0
    fit = max(int(fill * rank_pool // sequence), 0)
    return min(fit, workload.max_num_seqs or fit)


def _generation_seconds(
    model: ModelSpec,
    device: DeviceSpec,
    peak_flops: float,
    bandwidth: float,
    workload: SplitWorkload,
    engines: int,
    engine_width: int,
    rollout_batch_size: int,
    resident: int,
    max_num_batched_tokens: int,
) -> float:
    """Seconds the engines take to produce one rollout cycle.

    Each scheduler step decodes a token of every resident episode and
    prefills prompt tokens alongside, so the cycle is its decode steps, one
    per decoded token per wave of resident episodes, each reading the weights
    and resident KV at memory bandwidth, plus every episode's
    :func:`uncached_prefill_tokens` at the engine's prefill rate for the
    prompt tokens a step runs. Rollout admission starts a multi-turn episode
    only once the KV pool holds it, so episodes past the resident count wait
    and keep their cached prefix. A finished episode's slot refills at once,
    so waves are fractional; a multi-turn cycle's last wave waits on its
    slowest episode. Only multi-turn cycles price n-gram drafting;
    single-turn completions share too little with their prompt to accept
    drafts. An engine serving merged weights decodes at
    :data:`MERGED_DECODE_SPEEDUP` times the fitted bandwidth share.

    :param resident: :func:`resident_sequences` of one engine.
    :param max_num_batched_tokens: The engine's scheduler-step token cap.
    """
    arch = model.arch
    _, weight_read = _engine_weight_bytes(model, device, workload)
    kv_per_token = kv_cache_bytes_per_token(arch, model.checkpoint_dtype)
    multi_turn = workload.turns > 1
    context = _mean_context(workload)
    seqs = math.ceil(rollout_batch_size / engines) * workload.group_size
    slots = min(seqs, resident)
    if slots < 1:
        return math.inf
    read_efficiency = (
        MOE_DECODE_BANDWIDTH_EFFICIENCY if arch.is_moe else DECODE_BANDWIDTH_EFFICIENCY
    )
    if workload.merged_lora_sync:
        read_efficiency *= MERGED_DECODE_SPEEDUP
    decode_step = (weight_read + slots * context * kv_per_token) / (
        read_efficiency * engine_width * bandwidth
    ) + tensor_parallel_seconds(model, device, engine_width, slots)
    episode = (
        workload.decode_tokens * decode_step
        + (workload.turns - 1) * TURN_OVERHEAD_SECONDS
    )
    waves = seqs / slots
    speedup = 1.0
    decode = waves * episode
    if multi_turn:
        drafts = (
            drafts_by_default(
                multi_turn, arch.is_hybrid_ssm, workload.min_p, workload.logit_bias
            )
            if workload.speculative_drafting is None
            else workload.speculative_drafting
        )
        speedup = NGRAM_GENERATION_SPEEDUP if drafts else 1.0
        decode = (waves + MULTI_TURN_STRAGGLER_FACTOR - 1) * episode / speedup
    prompt = seqs * uncached_prefill_tokens(model, workload, engine_width)
    steps = waves * workload.decode_tokens / speedup
    step_tokens = min(max_num_batched_tokens, math.ceil(prompt / steps) + slots)
    return decode + prompt / engine_prefill_rate(
        model, device, peak_flops, engine_width, step_tokens, workload.merged_lora_sync
    )


def _cycle_step(
    model: ModelSpec,
    workload: SplitWorkload,
    engines: int,
    engine_width: int,
    rollout_batch_size: int,
    residents: dict[int, int],
) -> int:
    """The largest step that keeps every episode of an engine's cycle resident.

    A step is skipped when one turn's prompts across the engine's episodes
    cannot fill it. With none raised, the fallback step runs.

    :param residents: :func:`resident_sequences` of one engine at each
        :func:`engine_steps` step, the fallback first.
    :return: The step.
    """
    seqs = math.ceil(rollout_batch_size / engines) * workload.group_size
    turn_prompt = math.ceil(
        uncached_prefill_tokens(model, workload, engine_width) / workload.turns
    )
    fallback, *raised = residents
    return max(
        (
            step
            for step in raised
            if residents[step] >= seqs and step <= seqs * turn_prompt
        ),
        default=fallback,
    )


def _cycle_rollout_batch_size(
    workload: SplitWorkload, trainers: int, engines: int, resident: int
) -> int | None:
    """Groups per rollout cycle, or None when the trainer count cannot shard the batch.

    An unpinned multi-turn cycle keeps no more groups in flight than the
    engines' ``resident`` sequences hold, so episodes keep their cached prefix
    between turns, and at least ``lcm(trainers, engines)``.
    """
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
    groups = default_rollout_batch_size(workload.batch_size, trainers, engines)
    if workload.turns == 1:
        return groups
    multiple = math.lcm(trainers, engines)
    held = resident * engines // workload.group_size // multiple * multiple
    return max(multiple, min(groups, held))


def _sync_seconds(
    model: ModelSpec, workload: SplitWorkload, engines: int, engine_width: int
) -> float:
    """One weight sync over the workload's link, or 0 when the link is unknown."""
    link = workload.sync_link
    if link is None:
        return 0.0
    if workload.merged_lora_sync:
        return merged_sync_seconds(
            model,
            workload.lora_target_scope,
            workload.lora_packed_target_matrices,
            engines,
            engine_width,
            workload.turns > 1,
            link,
        )
    return adapter_sync_seconds(
        model,
        workload.lora_rank,
        workload.lora_target_scope,
        workload.lora_packed_target_matrices,
        engines,
        engine_width,
        link,
    )


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
    shorten the cycle goes to training. The cycle is sized at the fallback
    step of :func:`engine_steps`; its engines run the largest step whose KV
    pool still holds every episode of the cycle.

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
    learn_gpu_seconds = _learn_gpu_seconds(model, device, peak_flops, workload)
    residents = {
        step: resident_sequences(model, device, workload, engine_width, step)
        for step in engine_steps(model, workload)
    }
    fallback = next(iter(residents))
    best: tuple[tuple[float, int, int], GpuSplit] | None = None
    for engines in range(1, (gpus - 1) // engine_width + 1):
        for trainers in range(1, gpus - engines * engine_width + 1):
            rollout_batch_size = _cycle_rollout_batch_size(
                workload, trainers, engines, residents[fallback]
            )
            if rollout_batch_size is None:
                continue
            ratio = learn_cost_ratio(trainers)
            if ratio is None:
                continue
            step = _cycle_step(
                model, workload, engines, engine_width, rollout_batch_size, residents
            )
            cycles_per_step = (
                workload.batch_size
                * workload.generated_per_trained
                / rollout_batch_size
            )
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
                    residents[step],
                    step,
                ),
                max_num_batched_tokens=None if step == fallback else step,
                sync_seconds=_sync_seconds(model, workload, engines, engine_width),
                merged_lora_sync=workload.merged_lora_sync,
            )
            idle = gpus - trainers - engines * engine_width
            rank = (split.step_seconds, idle, engines)
            if best is None or rank < best[0]:
                best = (rank, split)
    return None if best is None else best[1]


def split_workload_from_manifest(
    manifest: TrainingManifest, adapter_sync: bool = False
) -> SplitWorkload:
    """Expected lengths and batch constraints from a rollout manifest's budgets.

    :param manifest: Validated async rollout manifest.
    :param adapter_sync: Weight sync pushes the adapter even where the engines
        could serve merged weights.
    :return: The workload at :func:`_expected_lengths`.
    :raises TypeError: When the algorithm does not generate rollouts.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec):
        msg = f"{spec.name} does not generate rollouts"
        raise TypeError(msg)
    row_tokens, decode_tokens, turns = _expected_lengths(manifest, spec)
    vllm = spec.vllm_config
    fsdp = spec.fsdp
    fsdp_tp = 1 if fsdp is None else fsdp.tp
    fsdp_ep = 1 if fsdp is None else fsdp.ep
    fsdp_shard = 1 if fsdp is None else (fsdp.shard_group_size or 1)
    pinned_rows = [
        size
        for size in (spec.micro_batch_size_per_gpu, spec.mini_batch_size)
        if size is not None
    ]
    group_size = rollout_group_size(spec)
    lora = spec.lora_config or LoraConfigDict()
    return SplitWorkload(
        batch_size=spec.batch_size,
        group_size=group_size,
        generated_per_trained=_generated_per_trained(manifest, group_size),
        row_tokens=row_tokens,
        decode_tokens=decode_tokens,
        turns=turns,
        learn_passes=learn_forward_passes(spec),
        rollout_batch_size=manifest.training.rollout_batch_size,
        max_num_seqs=None if vllm is None else vllm.max_num_seqs,
        gpu_memory_utilization=resolve_gpu_memory_utilization(
            None if vllm is None else vllm.gpu_memory_utilization, colocated=False
        ),
        trainer_multiple=math.lcm(fsdp_tp, fsdp_ep, fsdp_shard),
        trainer_tensor_parallel=fsdp_tp,
        trainer_expert_parallel=fsdp_ep,
        trainer_shard_group=fsdp_shard,
        per_rank_multiple=math.lcm(*pinned_rows) if pinned_rows else 1,
        micro_batch_size_per_gpu=spec.micro_batch_size_per_gpu,
        mini_batch_size=spec.mini_batch_size,
        weight_quantization=engine_weight_quantization(
            None if vllm is None else vllm.quantization
        ),
        merged_lora_sync=not adapter_sync and can_sync_merged_lora(spec),
        lora_rank=lora.lora_r,
        lora_target_scope=lora_target_scope(lora.target_modules),
        lora_packed_target_matrices=len(set(lora.target_parameters or ())),
        min_p=spec.min_p,
        logit_bias=rollout_logit_bias(spec),
        speculative_drafting=speculative_drafting(spec),
        max_num_batched_tokens=_pinned_max_num_batched_tokens(spec),
    )


def _expected_lengths(
    manifest: TrainingManifest, spec: RolloutLLMSpec
) -> tuple[int, int, int]:
    """Mean trained-row tokens, decoded tokens per row and turns from the budgets.

    A single turn decodes a share of ``max_output_tokens`` that depends on
    whether the model reasons, after a prompt filling a share of the rest of
    the row. A multi-turn row fills :data:`MULTI_TURN_ROW_FILL` of its budget
    over :data:`MULTI_TURN_TURN_SHARE` of ``max_turns``, or as many full turns
    as the row holds when turns are unbounded.
    """
    row_budget = spec.max_row_tokens or spec.max_model_len
    turn_tokens = min(spec.max_output_tokens or row_budget, row_budget)
    env = manifest.environment
    max_turns = env.max_turns if isinstance(env, LLMEnvSpec) else 1
    if max_turns == 1:
        share = (
            SINGLE_TURN_REASONING_DECODE_SHARE
            if _reasons(manifest, spec)
            else SINGLE_TURN_DECODE_SHARE
        )
        decode_tokens = math.ceil(share * turn_tokens)
        prompt_tokens = math.ceil(SINGLE_TURN_PROMPT_FILL * (row_budget - turn_tokens))
        return decode_tokens + prompt_tokens, decode_tokens, 1
    row_tokens = math.ceil(MULTI_TURN_ROW_FILL * row_budget)
    turns = max(
        2,
        math.ceil(row_tokens / turn_tokens)
        if max_turns is None
        else math.ceil(MULTI_TURN_TURN_SHARE * max_turns),
    )
    decode_tokens = min(
        math.ceil(MULTI_TURN_DECODE_SHARE * row_tokens),
        turns * MULTI_TURN_TURN_DECODE_TOKENS,
    )
    return row_tokens, decode_tokens, turns


def _reasons(manifest: TrainingManifest, spec: RolloutLLMSpec) -> bool:
    """Whether rollouts reason: the model has a reasoning parser and thinking is not turned off."""
    info = SUPPORTED_MODEL_INFO.get(spec.pretrained_model_name_or_path or "")
    model_type = None if info is None else info.config["model_type"]
    env = manifest.environment
    template = env.chat_template_kwargs if isinstance(env, LLMEnvSpec) else {}
    return (
        resolve_reasoning_parser(spec.vllm_engine_args, model_type) is not None
        and template.get("enable_thinking") is not False
    )


def _generated_per_trained(manifest: TrainingManifest, group_size: int) -> float:
    """Groups generated per trained group: more when the buffer drops tied groups."""
    buffer = manifest.replay_buffer
    if (
        isinstance(buffer, LLMRolloutBufferSpec)
        and buffer.drop_uninformative_groups
        and group_size > 1
    ):
        return TIED_DROP_GENERATION_FACTOR
    return 1.0


def _pinned_max_num_batched_tokens(spec: RolloutLLMSpec) -> int | None:
    """The step ``vllm_engine_args``, ``vllm_config`` or the model family fixes.

    Without chunked prefill the step is at least one full prompt.
    """
    engine_args = spec.vllm_engine_args
    if engine_args.get("max_num_batched_tokens") is not None:
        return int(engine_args["max_num_batched_tokens"])
    pinned = manifest_max_num_batched_tokens(spec, spec.vllm_config or VLLMConfig())
    if engine_args.get("enable_chunked_prefill") is False:
        # vLLM rejects a step smaller than one full prompt.
        return max(pinned or ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS, spec.max_model_len)
    return pinned


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


def _fsdp_width_need(workload: SplitWorkload) -> str:
    """``FSDP <dimensions> need(s)`` for the degrees that set ``trainer_multiple``."""
    names = [
        label
        for label, width in (
            ("tensor parallel", workload.trainer_tensor_parallel),
            ("expert parallel", workload.trainer_expert_parallel),
            ("shard group", workload.trainer_shard_group),
        )
        if width > 1
    ]
    if len(names) == 1:
        return f"FSDP {names[0]} needs"
    if len(names) == 2:
        return f"FSDP {names[0]} and {names[1]} need"
    return f"FSDP {names[0]}, {names[1]}, and {names[2]} need"


def _per_rank_pin_reason(workload: SplitWorkload) -> str:
    """Why a pinned micro-batch or mini-batch blocked every sharded trainer count."""
    pins = [
        (name, size)
        for name, size in (
            ("micro_batch_size_per_gpu", workload.micro_batch_size_per_gpu),
            ("mini_batch_size", workload.mini_batch_size),
        )
        if size is not None
    ]
    name, size = pins[0]
    if len(pins) == 1:
        return (
            f"{name} {size} does not divide the per-rank batch at any trainer "
            "count that shards the batch. Use a "
            f"{name} that divides those ranks, or leave it unset."
        )
    other_name, other_size = pins[1]
    return (
        f"{name} {size} and {other_name} {other_size} do not divide the "
        "per-rank batch at any trainer count that shards the batch. Use values "
        "that divide those ranks, or leave them unset."
    )


def _no_split_reason(
    workload: SplitWorkload,
    trainer_counts: range,
    plans: dict[int, TrainingMemoryPlan | None],
) -> str:
    """Why the split search found nothing, and what the run can change.

    :param workload: The member's learn step and rollout cycle.
    :param trainer_counts: Every trainer count a split could keep.
    :param plans: The trainer counts that shard the batch, each ``None``
        because no training-memory plan fits it.
    :return: The cause and its fix, in plain English.
    """
    more_gpus = (
        "Use more or larger GPUs per agent (a bigger resource tier or a "
        "smaller population size)"
    )
    if not trainer_counts:
        return (
            "a trainer and a rollout engine do not both fit. "
            f"{more_gpus}, or shorten the context length."
        )
    multiple = workload.trainer_multiple
    # More trainers lower per-GPU training memory, so the largest FSDP-legal
    # count not fitting means none of those counts fit.
    fsdp_counts = [trainers for trainers in trainer_counts if trainers % multiple == 0]
    if fsdp_counts and fsdp_counts[-1] in plans:
        return (
            "training does not fit in memory at any trainer count. "
            f"{more_gpus}, or shorten the context length."
        )
    if not fsdp_counts:
        return (
            f"{_fsdp_width_need(workload)} a multiple of {multiple} "
            f"trainer GPUs, and at most {trainer_counts[-1]} fit beside a "
            "rollout engine. Use more GPUs per agent (a bigger resource tier "
            "or a smaller population size)."
        )
    unpinned = workload.model_copy(update={"rollout_batch_size": None})
    if any(
        trainers not in plans
        and _cycle_rollout_batch_size(unpinned, trainers, 1, 1) is not None
        for trainers in trainer_counts
    ):
        return (
            f"rollout batch size {workload.rollout_batch_size} does not divide "
            "evenly across any trainer count that fits in memory. Use a rollout "
            "batch size divisible by more trainer counts, such as a power of "
            "two, or leave it unset."
        )
    row_free = workload.model_copy(update={"per_rank_multiple": 1})
    if any(
        _cycle_rollout_batch_size(workload, trainers, 1, 1) is None
        and _cycle_rollout_batch_size(row_free, trainers, 1, 1) is not None
        for trainers in trainer_counts
    ):
        return _per_rank_pin_reason(workload)
    return (
        f"batch size {workload.batch_size} does not divide evenly across any "
        "trainer count that fits in memory. Use a batch size divisible by more "
        "trainer counts, such as a power of two."
    )


@dataclass(frozen=True)
class RunMemoryPlan:
    """The GPU split and trainer memory plan a run starts with."""

    # ``None`` when the manifest's trainer and engine counts are kept.
    split: GpuSplit | None
    training: TrainingMemoryPlan
    # Async weight sync pushes merged weights to engines that run with LoRA
    # off. With no link it is merged whenever the engines can serve it.
    merged_lora_sync: bool


def merged_lora_sync_candidates(
    manifest: TrainingManifest, sync_link: SyncLink | None
) -> tuple[bool, ...]:
    """``merged_lora_sync`` values :func:`resolve_run_memory` prices, merged first.

    :param manifest: Validated run manifest.
    :param sync_link: Link the engines read synced weights over, or ``None``.
    :return: Both for engines that can serve merged weights over a known
        link, merged alone when the link is unknown, adapter alone otherwise.
    """
    if not can_sync_merged_lora(manifest.algorithm):
        return (False,)
    if sync_link is None:
        return (True,)
    return (True, False)


def _search_split(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int,
    engine_widths: list[int],
    sync_link: SyncLink | None,
    merged_lora_sync: bool,
    plans: dict[int, TrainingMemoryPlan | None],
) -> GpuSplit | None:
    """The fastest split of one member's GPUs, over every engine width.

    :param merged_lora_sync: Weight sync pushes merged weights.
    :param plans: Training-memory plans by trainer count, filled as the
        search prices each count.
    :return: The split :func:`choose_engine_width` picks, or ``None``.
    """

    def learn_cost_ratio(trainers: int) -> float | None:
        if trainers not in plans:
            trial = manifest.model_copy(deep=True)
            trial.training.training_gpus_per_agent = trainers
            try:
                plans[trainers] = training_memory_from_manifest(
                    trial, model, device, not merged_lora_sync
                )
            except TrainingMemoryUnfitError:
                plans[trainers] = None
        plan = plans[trainers]
        return None if plan is None else plan.learn_cost_ratio

    workload = split_workload_from_manifest(manifest, not merged_lora_sync).model_copy(
        update={"sync_link": sync_link}
    )
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
    return choose_engine_width(splits)


def resolve_run_memory(
    manifest: TrainingManifest,
    model: ModelSpec,
    device: DeviceSpec,
    member_gpus: int | None,
    gpus_per_node: int | None = None,
    sync_link: SyncLink | None = None,
) -> RunMemoryPlan:
    """Choose the GPU split and weight-sync mode, then the trainer memory plan.

    Each candidate trainer count is priced with its own
    :func:`training_memory_from_manifest` plan, so the split weighs learn
    time at the speed rungs that fit that many trainers. A searched split
    with the engine width omitted is searched at every width one engine can
    serve a sequence at and picked by :func:`choose_engine_width`; a pinned
    width is searched alone. Each of :func:`merged_lora_sync_candidates` is
    searched, and its split priced with one sync over ``sync_link``; merged is kept
    while its :attr:`GpuSplit.synced_step_seconds` is within
    :data:`WEIGHT_SYNC_TIE_SHARE` of the adapter's. Kept trainer and engine
    counts are not priced, so they sync merged weights unless the link is
    TCP.

    :param manifest: Validated run manifest. Not written.
    :param model: Training-side catalog model.
    :param device: One GPU of the allocation.
    :param member_gpus: GPUs one member owns, to search the split; ``None``
        keeps the manifest's trainer and engine counts.
    :param gpus_per_node: GPUs on one node; an engine stays on one node.
        ``None`` puts the member's GPUs on one node.
    :param sync_link: Link the engines read synced weights over; ``None``
        leaves sync unpriced and syncs merged weights where the engines can
        serve them.
    :return: The plan.
    :raises ValueError: When the GPU cannot run LLM training, or the split is
        searched and none fits.
    :raises TrainingMemoryUnfitError: When nothing fits one trainer GPU.
    """
    unsupported = unsupported_gpu_reason(device)
    if unsupported is not None:
        raise ValueError(unsupported)
    modes = merged_lora_sync_candidates(manifest, sync_link)
    plans: dict[bool, dict[int, TrainingMemoryPlan | None]] = {
        mode: {} for mode in modes
    }
    split = None
    merged = modes[0]
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
        searched = [
            (found, mode)
            for mode in modes
            if (
                found := _search_split(
                    manifest,
                    model,
                    device,
                    member_gpus,
                    engine_widths,
                    sync_link,
                    mode,
                    plans[mode],
                )
            )
            is not None
        ]
        if not searched:
            trainer_counts = range(
                1, member_gpus - min(engine_widths, default=member_gpus) + 1
            )
            workload = split_workload_from_manifest(manifest, not modes[0])
            reason = _no_split_reason(workload, trainer_counts, plans[modes[0]])
            msg = (
                f"No trainer / rollout-engine split of {member_gpus} "
                f"{device.name} GPUs fits this run: {reason}"
            )
            raise ValueError(msg)
        fastest = min(found.synced_step_seconds for found, _ in searched)
        split, merged = next(
            (found, mode)
            for found, mode in searched
            if found.synced_step_seconds <= (1 + WEIGHT_SYNC_TIE_SHARE) * fastest
        )
    elif len(modes) > 1:
        # Merged weights over TCP move at 0.4 GB/s: about a minute per sync
        # for a 4B model on 4 engines.
        merged = sync_link != "tcp"
    chosen = manifest.model_copy(deep=True)
    if split is not None:
        chosen.training.training_gpus_per_agent = split.trainers
        chosen.training.rollout_engines_per_agent = split.engines
        chosen.training.rollout_batch_size = split.rollout_batch_size
    training = plans[merged].get(chosen.training.training_gpus_per_agent)
    if training is None:
        training = training_memory_from_manifest(chosen, model, device, not merged)
    return RunMemoryPlan(split=split, training=training, merged_lora_sync=merged)
