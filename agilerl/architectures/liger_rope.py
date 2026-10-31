# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Liger RoPE on its Triton kernel.

On Hopper and newer, Liger's RoPE dispatcher sends large grad-carrying shapes
to a CuTe DSL kernel that compiles once per sequence length. Packed
micro-batches change length every step, so each training forward and backward
would stall the GPU on a compile. The Triton kernel takes the length at run time.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from liger_kernel.ops import LigerRopeFunction
from liger_kernel.transformers.rope import liger_rotary_pos_emb

__all__ = ["pin_triton_rope", "triton_rotary_pos_emb"]


def triton_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` and ``k`` with Liger's Triton RoPE kernel.

    :param q: ``(B, H_q, T, D)`` queries.
    :param k: ``(B, H_kv, T, D)`` keys.
    :param cos: ``(1, T, D)`` or ``(B, T, D)`` cosines.
    :param sin: ``(1, T, D)`` or ``(B, T, D)`` sines.
    :param position_ids: Position ids, as HuggingFace passes them.
    :param unsqueeze_dim: Head dimension of ``q`` and ``k``.
    :return: Rotated ``(q, k)``.
    """
    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def pin_triton_rope(model: torch.nn.Module) -> None:
    """Point the Liger RoPE patch of ``model``'s modeling module at the Triton kernel.

    A modeling module whose ``apply_rotary_pos_emb`` is not Liger's is left as is.

    :param model: HuggingFace model Liger has patched.
    """
    modeling: Any = sys.modules[type(model).__module__]
    if getattr(modeling, "apply_rotary_pos_emb", None) is liger_rotary_pos_emb:
        modeling.apply_rotary_pos_emb = triton_rotary_pos_emb
