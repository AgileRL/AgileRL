# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Learn-time constants per GPU class for the training-memory search and the GPU split."""

from __future__ import annotations

from dataclasses import dataclass, replace

from agilerl.arena.memory.specs import DeviceSpec


@dataclass(frozen=True)
class LearnCosts:
    """Learn-time constants measured on one GPU class.

    Token costs are in tokens of uncheckpointed learn work on that GPU.
    """

    # Recomputing one checkpointed token.
    recompute_cost: float
    # Fixed cost of one micro-batch: launches, hooks and host work. The
    # training-memory search ranks plans by it, and GRPO learn seconds and
    # step timing price it. SFT and DPO step seconds use
    # ``pass_overhead_seconds`` instead.
    micro_batch_overhead_tokens: float
    # Frozen-base all-gather of one micro-batch resharded after forward.
    reshard_gather_tokens: float
    # One routed-expert row chunk in one MoE layer: its kernel launches and the
    # expert-LoRA recompute's autograd pass in backward.
    routed_chunk_tokens: float
    # Seconds one optimizer step with AdamW state kept on CPU adds, per GiB of
    # the rank's state: the copy to the GPU, the unfused step and the copy
    # back. Zero on a GPU class with no measurement.
    optimizer_offload_seconds_per_gib: float = 0.0
    # Learn work per second per peak bf16 FLOP, over the A100's, for the GPU
    # split's learn time.
    peak_speed: float = 1.0
    # Host seconds one forward or backward call of an SFT or DPO learn step
    # takes whatever its length: kernel launches, hooks and adapter switches.
    # It does not grow with the model or the row, so short rows on a fast GPU
    # wait on it. Zero leaves it unpriced.
    pass_overhead_seconds: float = 0.0
    # Bytes per second one node sends the others, per direction, in an FSDP
    # ring all-gather and in a routed-token exchange, over the pod network
    # with NCCL's default socket settings. Zero leaves that traffic unpriced.
    cross_node_gather_bytes_per_second: float = 0.0
    cross_node_exchange_bytes_per_second: float = 0.0


SM80_LEARN_COSTS = LearnCosts(
    recompute_cost=0.30,
    micro_batch_overhead_tokens=730,
    reshard_gather_tokens=1320,
    # granite-3.1-3b-a800m GRPO LoRA under DDP, GC off: 64 MiB chunks
    # (15 per layer) learn 12% slower than 512 MiB chunks (2 per layer).
    routed_chunk_tokens=12,
    # 2 x 4 L4, 671 MiB gathered per GPU: an 8-rank ring all-gather in 841 ms;
    # token dispatch and combine at ep=8 at 1.29 GB/s. The A100 is unmeasured.
    cross_node_gather_bytes_per_second=0.73e9,
    cross_node_exchange_bytes_per_second=1.29e9,
    # Fitted to measured LoRA SFT/DPO steps on L4, A100 and H100. Short rows
    # on one A100 left the GPU 80% idle, waiting on this.
    pass_overhead_seconds=0.20,
)
"""A100-80.

A GPU below Hopper with no measured row, or no capability, uses these costs.
"""

SM90_LEARN_COSTS = LearnCosts(
    recompute_cost=0.255,
    micro_batch_overhead_tokens=7000,
    reshard_gather_tokens=1170,
    # Nemotron-3.5-Super-VL-120B GRPO LoRA, 8 trainers ep8, 16k-token
    # micro-batches, every block checkpointed: 64 MiB chunks (2880 per step)
    # learn 21% / 20% slower than 512 MiB (480) at tp1 / tp2. Nemotron-30B ep4
    # measured 0-4% faster at 64 MiB, so this overprices it by about 4%.
    routed_chunk_tokens=7.3,
    # Qwen3.6-35B-A3B ep4 at LoRA r16 / r32 (1.76 / 3.52 GiB AdamW per rank):
    # +0.281 / +0.547 s per step; Nemotron-30B ep4 r16 (0.82 GiB): +0.128 s.
    optimizer_offload_seconds_per_gib=0.157,
    # Qwen3-4B GRPO countdown LoRA at 4 trainers learns in 9.5 s, 1.25x the
    # time the A100 efficiencies predict.
    peak_speed=0.8,
    # 2 x 8 H100: a 16-rank ring all-gather of 720 MiB per rank in 291 ms;
    # token dispatch and combine at ep=16, 8192 tokens per GPU, at 3.4 GB/s.
    cross_node_gather_bytes_per_second=2.43e9,
    cross_node_exchange_bytes_per_second=3.4e9,
    # Fitted to measured LoRA SFT/DPO steps on L4, A100 and H100.
    pass_overhead_seconds=0.14,
)
"""H100 80GB HBM3: each SFT or DPO model call adds 0.14 s of host time, so fewer, larger micro-batches win."""

L4_LEARN_COSTS = replace(
    SM80_LEARN_COSTS,
    # GRPO LoRA on four GPUs at 16k and 32k context (Qwen3.5 0.8B to 4B,
    # Nemotron-3-Nano-4B) learns 3.8-5.9x slower on L4 than on A100, 4.6x on
    # geometric mean, against a 2.58x peak bf16 ratio. Long SFT/DPO steps fit
    # the same speed once pass_overhead_seconds is priced.
    peak_speed=0.55,
    # Fitted to measured LoRA SFT/DPO steps on L4, A100 and H100.
    pass_overhead_seconds=0.13,
)
"""L4: the A100 costs at its measured learn speed."""

LEARN_COSTS_BY_CAPABILITY: dict[tuple[int, int], LearnCosts] = {
    (8, 0): SM80_LEARN_COSTS,
    (9, 0): SM90_LEARN_COSTS,
}

LEARN_COSTS_BY_GPU: dict[str, LearnCosts] = {"NVIDIA L4": L4_LEARN_COSTS}
"""GPUs measured apart from others of their compute capability.

The L40S shares only the L4's compute capability, not its row, so it uses
:data:`SM80_LEARN_COSTS`.
"""


def learn_costs(device: DeviceSpec) -> LearnCosts:
    """Learn-time constants for a training GPU.

    A row for the GPU's name wins, then an exact compute-capability row.
    Hopper and newer with no row use the SM90 entry. A GPU below Hopper, or
    one with no capability, uses :data:`SM80_LEARN_COSTS`.

    :param device: The training GPU.
    :return: The learn costs for that GPU.
    """
    named = LEARN_COSTS_BY_GPU.get(device.name or "")
    if named is not None:
        return named
    capability = device.compute_capability
    if capability is None:
        return SM80_LEARN_COSTS
    measured = LEARN_COSTS_BY_CAPABILITY.get(capability)
    if measured is not None:
        return measured
    # No measured row: Hopper and newer use the SM90 costs.
    if capability >= (9, 0):
        return LEARN_COSTS_BY_CAPABILITY[(9, 0)]
    return SM80_LEARN_COSTS
