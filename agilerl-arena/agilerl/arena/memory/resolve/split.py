# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Price an async member's trainer / rollout-engine GPU splits and choose the fastest."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from agilerl.arena.memory.block_activations import block_recompute_bytes
from agilerl.arena.memory.formulas import (
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
)
from agilerl.arena.memory.resolve.batch import (
    KV_POOL_FILL,
    rollout_group_size,
)
from agilerl.arena.memory.resolve.decode_costs import (
    DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
    H100_DENSE_DECODE_TIME_PER_GB,
    H100_DENSE_DECODE_TIME_SCALE,
    H100_DENSE_PREFILL_SLOWDOWN,
    H100_HBM3_BYTES_PER_SECOND,
    H100_HYBRID_TURN_OVERHEAD_SECONDS,
    H100_MAMBA_LAYER_STEP_SECONDS,
    H100_MOE_DECODE_BANDWIDTH_EFFICIENCY,
    MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY,
    decode_costs,
    merged_engine_speedups,
)
from agilerl.arena.memory.resolve.fsdp import TrainingMemoryPlan
from agilerl.arena.memory.resolve.learn_costs import learn_costs
from agilerl.arena.memory.resolve.rollout import (
    default_rollout_batch_size,
    drafts_by_default,
)
from agilerl.arena.memory.resolve.sequences import resolve_gpu_memory_utilization
from agilerl.arena.memory.resolve.training import lora_target_scope
from agilerl.arena.memory.resolve.weight_sync import (
    SyncLink,
    WeightSyncTransport,
    adapter_sync_seconds,
    merged_sync_seconds,
    nccl_reshard_costs,
    nccl_reshard_sync_seconds,
)
from agilerl.arena.memory.specs import (
    DTYPE_BYTES,
    DeviceSpec,
    LoraTargetScope,
    ModelArch,
    ModelSpec,
    WeightVariant,
)
from agilerl.arena.models.algorithms.llmppo import LLMPPOSpec
from agilerl.arena.models.algorithms.llmreinforce import LLMREINFORCESpec
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
# A split that leaves GPUs idle is kept only when its predicted step beats
# every split using more GPUs by more than this share. AWM (Qwen3.6-35B-A3B)
# ran on 8x H100 at 4 trainers and two TP2 engines; one engine, leaving 2
# GPUs idle, was predicted 0.5% faster.
IDLE_GPU_TIE_SHARE = 0.02
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
    # Forward-equivalents per trained token, from :func:`learn_forward_passes`.
    learn_passes: int
    # Pinned groups per rollout cycle; ``None`` rounds ``batch_size`` up to
    # ``lcm(trainers, engines)``.
    rollout_batch_size: int | None = None
    # Pinned per-engine sequence cap; ``None`` runs every engine episode at once.
    max_num_seqs: int | None = None
    gpu_memory_utilization: float = 0.9
    # Trainer counts must be a multiple of this (FSDP tp * cp, ep, shard group).
    trainer_multiple: int = 1
    # FSDP tensor-parallel degree.
    trainer_tensor_parallel: int = 1
    # FSDP context-parallel degree. Trainer ranks per data shard are tp * cp.
    trainer_context_parallel: int = 1
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
    # Transport whose sync time is priced.
    sync_transport: WeightSyncTransport = "nixl"
    # The rollout algorithm's ``min_p``.
    min_p: float = 0.0
    # Rollout requests carry a ``logit_bias`` masking the image token.
    logit_bias: bool = False
    # ``vllm_engine_args.speculative_config`` set (``True``) or null
    # (``False``); ``None`` takes the engine's multi-turn default.
    speculative_drafting: bool | None = None
    # Pinned engine scheduler-step token cap; ``None`` lets the split pick one.
    max_num_batched_tokens: int | None = None


class TrainerCost(NamedTuple):
    """What one trainer count's training-memory plan sets in a split's price."""

    # :attr:`TrainingMemoryPlan.learn_cost_ratio`.
    learn_cost_ratio: float
    # The plan trains FSDP, so weight sync gathers each tensor whole first.
    fsdp: bool

    @classmethod
    def of(cls, plan: TrainingMemoryPlan) -> TrainerCost:
        """The cost of one training-memory plan.

        :param plan: The plan for one trainer count.
        :return: Its learn cost ratio and whether it trains FSDP.
        """
        return cls(plan.learn_cost_ratio, plan.fsdp is not None)


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
    # Transport this split's sync time was priced with.
    sync_transport: WeightSyncTransport = "nixl"

    @property
    def gpus(self) -> int:
        """GPUs the trainers and engines occupy."""
        return self.trainers + self.engines * self.engine_width

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

    The backward of frozen base weights costs one forward (input grads only),
    and every update epoch repeats the forward and backward over each
    gradient row copy. Async rollouts carry old log-probs. No-grad passes:
    the reference when ``beta > 0``; LLMPPO always scores critic values, and
    LLMREINFORCE scores old log-probs alongside the reference. Recompute and
    per-micro-batch costs come from the training-memory plan.

    :param spec: The rollout algorithm.
    :return: Update passes plus no-grad passes.
    """
    update = 2 * spec.gradient_row_copies * spec.update_epochs
    reference = int(spec.beta > 0)
    if isinstance(spec, LLMPPOSpec):
        return update + 1 + reference
    if isinstance(spec, LLMREINFORCESpec):
        return update + 2 * reference
    return update + reference


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

    fp8 and pre-quantized checkpoint weights are sized by
    :func:`engine_weight_bytes` and read at :func:`decode_costs`'s measured
    fp8 speedup.
    """
    counts = param_counts(model.arch, model.n_params)
    weights = weight_bytes(counts, model.checkpoint_dtype, WeightVariant())
    if (
        workload.weight_quantization == "none"
        and model.arch.checkpoint_quantization is None
    ):
        return weights, weights
    costs = decode_costs(device)
    speedup = (
        costs.fp8_moe_weight_read_speedup
        if model.arch.is_moe
        else costs.fp8_weight_read_speedup
    )
    stored = engine_weight_bytes(
        model.arch,
        counts,
        model.checkpoint_dtype,
        WeightVariant(),
        workload.weight_quantization,
    )
    return stored, weights / speedup


def _h100_hbm3(device: DeviceSpec) -> bool:
    """Whether the GPU is the H100 80GB HBM3 the generation fit is measured on."""
    bandwidth = device.memory_bytes_per_second
    return bandwidth is not None and abs(bandwidth - H100_HBM3_BYTES_PER_SECOND) <= 1e9


def _turn_overhead_seconds(arch: ModelArch, device: DeviceSpec) -> float:
    """Seconds between turns.

    A hybrid episode on H100 HBM3 uses the fitted overhead.
    """
    if arch.is_hybrid_ssm and _h100_hbm3(device):
        return H100_HYBRID_TURN_OVERHEAD_SECONDS
    return TURN_OVERHEAD_SECONDS


def _decode_read_efficiency(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    weight_read: float,
) -> float:
    """Share of HBM bandwidth one decode step reads weights and KV at.

    H100 HBM3 dense reads take longer as the checkpoint grows. MoE on that GPU
    uses its own share. Merged weights then scale the share by
    :func:`merged_engine_speedups`.
    """
    efficiency = (
        MOE_DECODE_BANDWIDTH_EFFICIENCY
        if model.arch.is_moe
        else DECODE_BANDWIDTH_EFFICIENCY
    )
    if _h100_hbm3(device):
        if model.arch.is_moe:
            efficiency = H100_MOE_DECODE_BANDWIDTH_EFFICIENCY
        else:
            checkpoint_gb = weight_read / 1e9
            efficiency /= (
                H100_DENSE_DECODE_TIME_SCALE
                + H100_DENSE_DECODE_TIME_PER_GB * checkpoint_gb
            )
    if workload.merged_lora_sync:
        speedups = merged_engine_speedups(device)
        efficiency *= (
            speedups.moe_decode if model.arch.is_moe else speedups.dense_decode
        )
    return efficiency


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
    computes faster by :func:`merged_engine_speedups` for this GPU.

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
    elif _h100_hbm3(device):
        efficiency /= H100_DENSE_PREFILL_SLOWDOWN
    if merged_lora_sync:
        speedups = merged_engine_speedups(device)
        efficiency *= (
            speedups.moe_prefill if model.arch.is_moe else speedups.dense_prefill
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
    drafts. An engine serving merged weights reads weights and KV at
    :func:`merged_engine_speedups` times the fitted bandwidth share.

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
    read_efficiency = _decode_read_efficiency(model, device, workload, weight_read)
    decode_step = (weight_read + slots * context * kv_per_token) / (
        read_efficiency * engine_width * bandwidth
    ) + tensor_parallel_seconds(model, device, engine_width, slots)
    episode = workload.decode_tokens * decode_step + (workload.turns - 1) * (
        _turn_overhead_seconds(arch, device)
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
    prefill = prompt / engine_prefill_rate(
        model, device, peak_flops, engine_width, step_tokens, workload.merged_lora_sync
    )
    cycles_per_step = (
        workload.batch_size * workload.generated_per_trained / rollout_batch_size
    )
    mamba_step = (
        H100_MAMBA_LAYER_STEP_SECONDS * arch.n_mamba_layers / cycles_per_step
        if _h100_hbm3(device) and not arch.is_moe
        else 0.0
    )
    return decode + prefill + mamba_step


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


def _trainers_shard_batch(workload: SplitWorkload, trainers: int) -> bool:
    """Whether ``trainers`` ranks shard the batch, its per-rank rows and a pinned rollout batch."""
    if trainers % workload.trainer_multiple:
        return False
    shards = trainers // (
        workload.trainer_tensor_parallel * workload.trainer_context_parallel
    )
    if workload.batch_size % shards:
        return False
    if (
        workload.batch_size // shards * workload.group_size
    ) % workload.per_rank_multiple:
        return False
    return (
        workload.rollout_batch_size is None
        or workload.rollout_batch_size % trainers == 0
    )


def _cycle_rollout_batch_size(
    workload: SplitWorkload, trainers: int, engines: int, resident: int
) -> int | None:
    """Groups per rollout cycle, or None when the trainer count cannot shard the batch.

    An unpinned multi-turn cycle keeps no more groups in flight than the
    engines' ``resident`` sequences hold, so episodes keep their cached prefix
    between turns, and at least ``lcm(trainers, engines)``.
    """
    if not _trainers_shard_batch(workload, trainers):
        return None
    if workload.rollout_batch_size is not None:
        return workload.rollout_batch_size
    groups = default_rollout_batch_size(workload.batch_size, trainers, engines)
    if workload.turns == 1:
        return groups
    multiple = math.lcm(trainers, engines)
    held = resident * engines // workload.group_size // multiple * multiple
    return max(multiple, min(groups, held))


def _sync_seconds(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    trainers: int,
    engines: int,
    engine_width: int,
    trainer_fsdp: bool,
) -> float:
    """One weight sync over the workload's link, or 0 when the link is unknown.

    :param trainer_fsdp: The trainers shard weights with FSDP.
    :raises ValueError: When ``nccl_reshard`` is priced on a GPU
        :func:`nccl_reshard_costs` has no rates for.
    """
    link = workload.sync_link
    if link is None:
        return 0.0
    if workload.merged_lora_sync and workload.sync_transport == "nccl_reshard":
        costs = nccl_reshard_costs(device)
        if costs is None:
            msg = f"{device.name} has no nccl_reshard rates"
            raise ValueError(msg)
        shards = (
            workload.trainer_shard_group
            if workload.trainer_shard_group > 1
            else trainers
        )
        return nccl_reshard_sync_seconds(
            model,
            workload.lora_target_scope,
            engines,
            engine_width,
            shards,
            trainer_fsdp,
            costs,
        )
    if workload.merged_lora_sync:
        return merged_sync_seconds(
            model,
            workload.lora_target_scope,
            workload.lora_packed_target_matrices,
            engines,
            engine_width,
            trainer_fsdp,
            device,
            link,
        )
    return adapter_sync_seconds(
        model,
        workload.lora_rank,
        workload.lora_target_scope,
        workload.lora_packed_target_matrices,
        engines,
        engine_width,
        trainer_fsdp,
        link,
    )


def _split_pricer(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    engine_width: int,
    trainer_cost: Callable[[int], TrainerCost | None],
) -> Callable[[int, int], GpuSplit | None]:
    """:func:`price_gpu_split` at one engine width, as a function of trainers and engines.

    Engine residents and learn GPU-seconds do not depend on the counts, so
    they are computed once here. Each trainer count's cost, each count's
    sync and each cycle's generation seconds are computed once per pricer.

    :raises ValueError: When the device has no peak bf16 or bandwidth figure.
    """
    peak_flops = device.peak_bf16_flops
    bandwidth = device.memory_bytes_per_second
    if peak_flops is None or bandwidth is None:
        msg = (
            f"{device.name or 'This GPU'} has no peak bf16 or memory bandwidth "
            "figure in the GPU catalogue, so the trainer / rollout-engine split "
            "cannot be priced. Set training.training_gpus_per_agent and "
            "training.rollout_engines_per_agent."
        )
        raise ValueError(msg)
    residents = {
        step: resident_sequences(model, device, workload, engine_width, step)
        for step in engine_steps(model, workload)
    }
    fallback = next(iter(residents))
    learn_gpu_seconds = _learn_gpu_seconds(model, device, peak_flops, workload)
    trainer_costs: dict[int, TrainerCost | None] = {}
    syncs: dict[tuple[int, int], float] = {}
    # (engines, rollout_batch_size) -> (step, seconds of one cycle)
    cycles: dict[tuple[int, int], tuple[int, float]] = {}

    def price(trainers: int, engines: int) -> GpuSplit | None:
        rollout_batch_size = _cycle_rollout_batch_size(
            workload, trainers, engines, residents[fallback]
        )
        if rollout_batch_size is None:
            return None
        if trainers not in trainer_costs:
            trainer_costs[trainers] = trainer_cost(trainers)
        cost = trainer_costs[trainers]
        if cost is None:
            return None
        cycle = (engines, rollout_batch_size)
        if cycle not in cycles:
            step = _cycle_step(
                model, workload, engines, engine_width, rollout_batch_size, residents
            )
            cycles[cycle] = (
                step,
                _generation_seconds(
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
            )
        step, cycle_seconds = cycles[cycle]
        sync_key = (trainers, engines)
        if sync_key not in syncs:
            syncs[sync_key] = _sync_seconds(
                model, device, workload, trainers, engines, engine_width, cost.fsdp
            )
        cycles_per_step = (
            workload.batch_size * workload.generated_per_trained / rollout_batch_size
        )
        return GpuSplit(
            trainers=trainers,
            engines=engines,
            engine_width=engine_width,
            rollout_batch_size=rollout_batch_size,
            learn_seconds=learn_gpu_seconds * cost.learn_cost_ratio / trainers,
            generation_seconds=cycles_per_step * cycle_seconds,
            max_num_batched_tokens=None if step == fallback else step,
            sync_seconds=syncs[sync_key],
            merged_lora_sync=workload.merged_lora_sync,
            sync_transport=workload.sync_transport,
        )

    return price


def price_gpu_split(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    trainers: int,
    engines: int,
    engine_width: int,
    trainer_cost: Callable[[int], TrainerCost | None],
) -> GpuSplit | None:
    """Predict the learn, generation and sync seconds of one trainer / engine count.

    The cycle is sized at the fallback step of :func:`engine_steps`; its
    engines run the largest step whose KV pool still holds every episode of
    the cycle.

    :param model: Model the trainers and engines run.
    :param device: One GPU of the allocation.
    :param workload: The member's learn step and rollout cycle.
    :param trainers: Trainer GPUs.
    :param engines: Rollout engines.
    :param engine_width: GPUs per rollout engine (vLLM tensor parallel).
    :param trainer_cost: Trainer count to its training-memory plan's
        :class:`TrainerCost`, or ``None`` when no plan fits that many trainers.
    :return: The split, or None when the trainer count cannot shard the
        batch or no plan fits it.
    :raises ValueError: When the device has no peak bf16 or bandwidth figure,
        or ``nccl_reshard`` is priced on a GPU with no rates for it.
    """
    price = _split_pricer(model, device, workload, engine_width, trainer_cost)
    return price(trainers, engines)


def fullest_split_within(splits: list[GpuSplit], share: float) -> GpuSplit:
    """The split using the most GPUs among those whose synced step is within ``share`` of the fastest.

    :param splits: Candidate splits of the same GPUs, preferred first.
    :param share: Slack over the fastest :attr:`GpuSplit.synced_step_seconds`.
    :return: The earliest such split that uses the most GPUs.
    """
    fastest = min(split.synced_step_seconds for split in splits)
    tied = [
        split for split in splits if split.synced_step_seconds <= (1 + share) * fastest
    ]
    most = max(split.gpus for split in tied)
    return next(split for split in tied if split.gpus == most)


def choose_gpu_split(
    model: ModelSpec,
    device: DeviceSpec,
    workload: SplitWorkload,
    gpus: int,
    engine_width: int,
    trainer_cost: Callable[[int], TrainerCost | None],
) -> GpuSplit | None:
    """Split one member's GPUs so the step with one weight sync is fastest.

    Every count is priced by :func:`price_gpu_split` and picked by
    :func:`fullest_split_within` at :data:`IDLE_GPU_TIE_SHARE`. Among the
    splits using the most GPUs the fastest wins, and exact ties go to fewer
    engines, so an engine that does not shorten the cycle goes to training.

    :param model: Model the trainers and engines run.
    :param device: One GPU of the allocation.
    :param workload: The member's learn step and rollout cycle.
    :param gpus: GPUs the member owns.
    :param engine_width: GPUs per rollout engine (vLLM tensor parallel).
    :param trainer_cost: Trainer count to its training-memory plan's
        :class:`TrainerCost`, or ``None`` when no plan fits that many trainers.
    :return: The chosen split, or None when no split fits.
    :raises ValueError: When the device has no peak bf16 or bandwidth figure.
    """
    if gpus <= engine_width:
        return None
    price = _split_pricer(model, device, workload, engine_width, trainer_cost)
    sharding = [
        trainers
        for trainers in range(1, gpus - engine_width + 1)
        if _trainers_shard_batch(workload, trainers)
    ]
    splits = [
        split
        for engines in range(1, (gpus - 1) // engine_width + 1)
        for trainers in sharding
        if trainers <= gpus - engines * engine_width
        and (split := price(trainers, engines)) is not None
    ]
    if not splits:
        return None
    splits.sort(key=lambda split: (split.synced_step_seconds, split.engines))
    return fullest_split_within(splits, IDLE_GPU_TIE_SHARE)


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
    fsdp_cp = 1 if fsdp is None else fsdp.cp
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
            None if vllm is None else vllm.gpu_memory_utilization,
            colocated=False,
            hub_id=spec.pretrained_model_name_or_path,
        ),
        trainer_multiple=math.lcm(fsdp_tp * fsdp_cp, fsdp_ep, fsdp_shard),
        trainer_tensor_parallel=fsdp_tp,
        trainer_context_parallel=fsdp_cp,
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
    model_type = None if info is None else info.model_type
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
    """Pick one width's split by :func:`fullest_split_within` at :data:`ENGINE_WIDTH_TIE_SHARE`.

    Among tied widths using the most GPUs, the narrowest wins.

    :param splits: The best split at each engine width, narrowest first.
    :return: The chosen split, or None when there are none.
    """
    if not splits:
        return None
    return fullest_split_within(splits, ENGINE_WIDTH_TIE_SHARE)


def _fsdp_width_need(workload: SplitWorkload) -> str:
    """``FSDP <dimensions> need(s)`` for the degrees that set ``trainer_multiple``."""
    names = [
        label
        for label, width in (
            ("tensor parallel", workload.trainer_tensor_parallel),
            ("context parallel", workload.trainer_context_parallel),
            ("expert parallel", workload.trainer_expert_parallel),
            ("shard group", workload.trainer_shard_group),
        )
        if width > 1
    ]
    if len(names) == 1:
        return f"FSDP {names[0]} needs"
    if len(names) == 2:
        return f"FSDP {names[0]} and {names[1]} need"
    return f"FSDP {', '.join(names[:-1])}, and {names[-1]} need"


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


def no_split_reason(
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
