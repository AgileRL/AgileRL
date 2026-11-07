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
# fitted to. Qwen3.6-35B-A3B prefills 2.1x faster with LoRA off (vLLM 0.32).
# vLLM loses 23.9% to 47% throughput to all-linear LoRA on dense models
# (SqueezeBits); decode, and dense prefill, take the low end. Unmeasured on
# AgileRL engines beyond the MoE prefill.
MERGED_MOE_PREFILL_SPEEDUP = 2.1
MERGED_DECODE_SPEEDUP = 1 / (1 - 0.239)


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
"""GPUs with no measured entry: fp8 weights are priced at the bf16 rate."""

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

DECODE_COSTS_BY_CAPABILITY: dict[tuple[int, int], DecodeCosts] = {
    (8, 0): SM80_DECODE_COSTS,
    (9, 0): SM90_DECODE_COSTS,
}


def decode_costs(device: DeviceSpec) -> DecodeCosts:
    """Decode-speed constants for a rollout GPU.

    :param device: The rollout GPU.
    :return: The entry for its compute capability, or :data:`UNMEASURED_DECODE_COSTS`.
    """
    if device.compute_capability is None:
        return UNMEASURED_DECODE_COSTS
    return DECODE_COSTS_BY_CAPABILITY.get(
        device.compute_capability, UNMEASURED_DECODE_COSTS
    )
