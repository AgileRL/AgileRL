# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""One-pass Triton stochastic rounding of fp32 to bf16.

Bit-identical to the torch path in :mod:`agilerl.utils.stochastic_round`: it
reads fp32 once and writes bf16 once, the same traffic as ``tensor.to(bf16)``.
"""

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.heuristics({"BLOCK": lambda _args: BLOCK_SIZE})
@triton.jit
def stochastic_round_bf16_kernel(src, dst, numel, salt, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    values = tl.load(src + offsets, mask=mask)
    bits = values.to(tl.int32, bitcast=True)
    mixed = (bits >> 16) & 0xFFFF
    mixed ^= salt
    mixed *= 0x7FEB352D
    mixed ^= mixed >> 16
    mixed &= 0xFFFF
    mixed = (mixed + bits) & -0x10000
    rounded = mixed.to(tl.float32, bitcast=True)
    finite = (bits & 0x7F800000) != 0x7F800000
    out = tl.where(finite, rounded, values)
    tl.store(dst + offsets, out.to(tl.bfloat16), mask=mask)


def stochastic_round_bf16_into(
    values: torch.Tensor, out: torch.Tensor, salt: int
) -> None:
    """Write the stochastic bf16 rounding of ``values`` into ``out``.

    :param values: Contiguous fp32 CUDA tensor.
    :param out: Contiguous bf16 tensor with the same number of elements.
    :param salt: Signed 32-bit step salt.
    """
    numel = values.numel()
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    stochastic_round_bf16_kernel[grid](values, out, numel, salt)
