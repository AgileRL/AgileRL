# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Rollout-engine decode-speed constants per GPU class for the GPU split search."""

from __future__ import annotations

from dataclasses import dataclass

from agilerl.arena.memory.specs import DeviceSpec

# Share of peak bf16 a bf16 MoE engine prefills at while it decodes, by engine
# width then prompt tokens per scheduler step, excluding the TP all-reduce.
# Qwen3.6-35B-A3B LoRA on AWM, 8x H100, vLLM 0.32: a prompt token adds 19.8 us
# to a TP2 step (2k tokens) and 18.7 us to a TP4 step (7.6k). Larger steps take
# the 10k-token prefill gain over 8192; TP2 16384 is the log midpoint.
# Unmeasured widths take the nearest one. Scales by peak bf16, width and weights.
MOE_ENGINE_PREFILL_FLOPS_EFFICIENCY: dict[int, dict[int, float]] = {
    2: {8192: 0.18, 16384: 0.215, 32768: 0.257, 65536: 0.272},
    4: {8192: 0.10, 16384: 0.158, 32768: 0.163, 65536: 0.166},
}
# The same share for a dense model. Unmeasured, at any step: assumes vLLM
# prefills a dense LoRA model at about 40% of peak bf16.
DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY = 0.4

# Generation speed of an engine serving merged weights with LoRA off, over the
# LoRA engine the efficiencies above and the split's decode bandwidth are
# fitted to. Qwen3.6-35B-A3B prefills 2.1x faster with LoRA off (vLLM 0.32, H100).
MERGED_MOE_PREFILL_SPEEDUP = 2.1
# Qwen3.5-9B, all-linear r=16, vLLM 0.32, 4x A100-SXM4-80GB, GuessTheNumber,
# 2 trainers and 1 TP2 engine: merged weights 1054 completion tokens per step
# second, the LoRA engine 406, at 2.75 s and 4.12 s per step. It scales the
# decode bandwidth share (weights and KV) and the dense prefill FLOPs share; TP
# all-reduce and turn overhead are unscaled, so that layout prices a merged
# step 1.50x faster, as measured, not 2.6x.
SM80_MERGED_DENSE_SPEEDUP = 1054 / 406
# Qwen3.5-2B, 4B, 9B and Qwen3.8-27B, all-linear r=16, vLLM 0.32, 8x H100,
# GuessTheNumber, 4 trainers and 2 TP2 engines. Scales the weight and KV
# read and the dense prefill. The footprint term and the Mamba step set how
# much of a small model's step that read is.
SM90_MERGED_DENSE_SPEEDUP = 2.146
# Qwen3.6-35B-A3B on the same layout: merged weights generate 1.23x the LoRA
# engines' tokens/s.
SM90_MERGED_MOE_DECODE_SPEEDUP = 1.205
# Catalogue HBM bandwidth of H100 80GB HBM3. H200 is the same compute
# capability and a faster bus, and does not use the fit below.
H100_HBM3_BYTES_PER_SECOND = 3350 * 1e9
# Dense decode time on that GPU, over a read at DECODE_BANDWIDTH_EFFICIENCY:
# scale + per_gb * (checkpoint GB). A Qwen3-4B countdown step stays inside its
# measured upper bound; a 55 GB checkpoint reads about twice as long per byte.
H100_DENSE_DECODE_TIME_SCALE = 0.963
H100_DENSE_DECODE_TIME_PER_GB = 0.0229
# Seconds one learn step spends per Mamba layer on that GPU, on top of the
# bandwidth read. Qwen3-4B has no Mamba layers, so Sudoku and countdown do not
# pay it.
H100_MAMBA_LAYER_STEP_SECONDS = 0.082
# Seconds between turns of a hybrid episode on that GPU. Prefill is priced on
# its own and these engines do not draft.
H100_HYBRID_TURN_OVERHEAD_SECONDS = 0.0026
# Dense prefill on that GPU, as a divisor of DENSE_ENGINE_PREFILL_FLOPS_EFFICIENCY.
H100_DENSE_PREFILL_SLOWDOWN = 2.798
# MoE decode bandwidth share on that GPU.
H100_MOE_DECODE_BANDWIDTH_EFFICIENCY = 0.308
# Merged decode on other GPUs, and MoE merged decode on A100, are not measured
# on these engines. SqueezeBits reports a 23.9% throughput loss to all-linear
# LoRA; that is the fallback.
UNMEASURED_MERGED_DENSE_SPEEDUP = 1 / (1 - 0.239)


@dataclass(frozen=True)
class DecodeCosts:
    """Decode speed of engine weight formats on one GPU class.

    Each speedup scales the time a decode step spends reading weights, against
    bf16 weights. Measured with LoRA adapters on vLLM 0.32 with dual-stream LoRA.
    """

    # fp8 weights of a dense model.
    fp8_weight_read_speedup: float
    # fp8 weights of a mixture-of-experts model, with nearly every expert active.
    fp8_moe_weight_read_speedup: float


UNMEASURED_DECODE_COSTS = DecodeCosts(
    fp8_weight_read_speedup=1.0,
    fp8_moe_weight_read_speedup=1.0,
)
"""A GPU below Hopper with no measured row, or no capability.

fp8 weights are priced at the bf16 rate.
"""

SM80_DECODE_COSTS = DecodeCosts(
    # Marlin W8A16 dequantizes in the GEMM: Qwen3-4B at 128 sequences.
    fp8_weight_read_speedup=0.85,
    # granite-3.1-3b-a800m at 64 sequences.
    fp8_moe_weight_read_speedup=1.08,
)
"""A100-80."""

SM90_DECODE_COSTS = DecodeCosts(
    # Qwen3-4B at 128 sequences x 672 tokens and at 32 x 6256 tokens.
    fp8_weight_read_speedup=1.29,
    # granite-3.1-3b-a800m and granite-4.0-h-tiny at 64 sequences.
    fp8_moe_weight_read_speedup=1.12,
)
"""H100 80GB HBM3."""


@dataclass(frozen=True)
class MergedEngineSpeedups:
    """Speed of merged weights over the LoRA engine on one GPU class.

    The decode terms scale weight reads; the prefill terms scale prefill compute.
    """

    dense_decode: float
    moe_decode: float
    dense_prefill: float
    moe_prefill: float


SM80_MERGED_SPEEDUPS = MergedEngineSpeedups(
    dense_decode=SM80_MERGED_DENSE_SPEEDUP,
    moe_decode=UNMEASURED_MERGED_DENSE_SPEEDUP,
    dense_prefill=SM80_MERGED_DENSE_SPEEDUP,
    moe_prefill=MERGED_MOE_PREFILL_SPEEDUP,
)
"""A100-80. Dense terms are fitted to the Qwen3.5-9B tokens/s ratio."""

SM90_MERGED_SPEEDUPS = MergedEngineSpeedups(
    dense_decode=SM90_MERGED_DENSE_SPEEDUP,
    moe_decode=SM90_MERGED_MOE_DECODE_SPEEDUP,
    dense_prefill=SM90_MERGED_DENSE_SPEEDUP,
    moe_prefill=MERGED_MOE_PREFILL_SPEEDUP,
)
"""H100 80GB HBM3. Dense terms are fitted to the Qwen3.5 and Qwen3.8 tokens/s ratios."""

UNMEASURED_MERGED_SPEEDUPS = MergedEngineSpeedups(
    dense_decode=UNMEASURED_MERGED_DENSE_SPEEDUP,
    moe_decode=UNMEASURED_MERGED_DENSE_SPEEDUP,
    dense_prefill=UNMEASURED_MERGED_DENSE_SPEEDUP,
    moe_prefill=MERGED_MOE_PREFILL_SPEEDUP,
)
"""A GPU below Hopper with no measured row, or no capability."""

MERGED_SPEEDUPS_BY_CAPABILITY: dict[tuple[int, int], MergedEngineSpeedups] = {
    (8, 0): SM80_MERGED_SPEEDUPS,
    (9, 0): SM90_MERGED_SPEEDUPS,
}

DECODE_COSTS_BY_CAPABILITY: dict[tuple[int, int], DecodeCosts] = {
    (8, 0): SM80_DECODE_COSTS,
    (9, 0): SM90_DECODE_COSTS,
}


def decode_costs(device: DeviceSpec) -> DecodeCosts:
    """Decode-speed constants for a rollout GPU.

    An exact compute-capability row wins. Hopper and newer with no row use
    the SM90 entry. A GPU below Hopper, or one with no capability, uses
    :data:`UNMEASURED_DECODE_COSTS`.

    :param device: The rollout GPU.
    :return: The decode costs for that GPU.
    """
    capability = device.compute_capability
    if capability is None:
        return UNMEASURED_DECODE_COSTS
    measured = DECODE_COSTS_BY_CAPABILITY.get(capability)
    if measured is not None:
        return measured
    # No measured row: Hopper and newer use the SM90 costs.
    if capability >= (9, 0):
        return DECODE_COSTS_BY_CAPABILITY[(9, 0)]
    return UNMEASURED_DECODE_COSTS


def merged_engine_speedups(device: DeviceSpec) -> MergedEngineSpeedups:
    """Merged-weight generation speedups for a rollout GPU.

    An exact compute-capability row wins. Hopper and newer with no row use
    the SM90 entry. A GPU below Hopper, or one with no capability, uses
    :data:`UNMEASURED_MERGED_SPEEDUPS`.

    :param device: The rollout GPU.
    :return: The merged-engine speedups for that GPU.
    """
    capability = device.compute_capability
    if capability is None:
        return UNMEASURED_MERGED_SPEEDUPS
    measured = MERGED_SPEEDUPS_BY_CAPABILITY.get(capability)
    if measured is not None:
        return measured
    # No measured row: Hopper and newer use the SM90 speedups.
    if capability >= (9, 0):
        return MERGED_SPEEDUPS_BY_CAPABILITY[(9, 0)]
    return UNMEASURED_MERGED_SPEEDUPS
