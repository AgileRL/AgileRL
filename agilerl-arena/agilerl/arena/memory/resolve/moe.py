# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert LoRA kernel choices."""

from __future__ import annotations

# SM majors where ``torch._grouped_mm`` runs one fused kernel. Elsewhere torch
# loops over every group after copying the offsets to the host.
GROUPED_GEMM_SM_MAJORS = frozenset({9, 10})
GROUPED_GEMM_DTYPE = "bfloat16"
# ``torch._grouped_mm`` needs every inner-dim row stride on a 16-byte boundary.
GROUPED_GEMM_ALIGNMENT_BYTES = 16


def grouped_gemm_supported(compute_capability: tuple[int, int], dtype: str) -> bool:
    """Whether the fused grouped GEMM serves this device and dtype.

    :param compute_capability: CUDA ``(major, minor)``.
    :param dtype: Activation dtype name, e.g. ``bfloat16``.
    :return: ``True`` on SM 9.x or 10.x in bf16.
    """
    return (
        dtype == GROUPED_GEMM_DTYPE and compute_capability[0] in GROUPED_GEMM_SM_MAJORS
    )


def padded_expert_lora_rank(rank: int, element_size: int) -> int:
    """Smallest expert LoRA rank at or above ``rank`` the grouped GEMM accepts.

    :param rank: Configured LoRA rank.
    :param element_size: Bytes per element of the LoRA factors (2 for bf16).
    :return: ``rank`` rounded up to a multiple of ``16 // element_size``.
    """
    multiple = GROUPED_GEMM_ALIGNMENT_BYTES // element_size
    return -(-rank // multiple) * multiple


def resolve_moe_lora_recompute(checkpointed: bool) -> bool:
    """Whether one routed-experts LoRA block recomputes its up-projection in backward.

    :param checkpointed: The block holding the experts is activation-checkpointed,
        so its backward already reruns the expert forward.
    :return: ``not checkpointed``.
    """
    return not checkpointed
