# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert LoRA kernel choices."""

from __future__ import annotations

from typing import Literal

ExpertGemmBackend = Literal["grouped_mm", "triton"]

# Triton's half-precision ``tl.dot`` compiles from sm80.
TRITON_MIN_SM_MAJOR = 8
TRITON_DTYPES = frozenset({"bfloat16", "float16"})

# SM majors where ``torch.nn.functional.grouped_mm`` runs one fused kernel. Elsewhere torch
# loops over every group after copying the offsets to the host.
GROUPED_GEMM_SM_MAJORS = frozenset({9, 10})
# SM majors where the Triton kernels beat the fused ``grouped_mm``: 6-25% faster
# forward + backward on H100 across Granite, Qwen3.6 and Nemotron expert shapes.
TRITON_OVER_GROUPED_GEMM_SM_MAJORS = frozenset({9})
GROUPED_GEMM_DTYPE = "bfloat16"
# ``torch.nn.functional.grouped_mm`` needs every inner-dim row stride on a 16-byte boundary.
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


def expert_gemm_backend(
    compute_capability: tuple[int, int], dtype: str, triton_installed: bool
) -> ExpertGemmBackend:
    """Kernel the per-expert GEMMs over expert-sorted rows run on a CUDA device.

    :param compute_capability: CUDA ``(major, minor)``.
    :param dtype: Activation dtype name, e.g. ``bfloat16``.
    :param triton_installed: Triton is importable.
    :return: ``triton`` on sm80+ in bf16 or fp16, except ``grouped_mm`` where
        :func:`grouped_gemm_supported` passes outside
        :data:`TRITON_OVER_GROUPED_GEMM_SM_MAJORS`. Without Triton,
        ``grouped_mm`` wherever it is supported.
    :raises RuntimeError: If neither kernel serves the device and dtype.
    """
    triton_runs = (
        triton_installed
        and compute_capability[0] >= TRITON_MIN_SM_MAJOR
        and dtype in TRITON_DTYPES
    )
    if grouped_gemm_supported(compute_capability, dtype) and not (
        triton_runs and compute_capability[0] in TRITON_OVER_GROUPED_GEMM_SM_MAJORS
    ):
        return "grouped_mm"
    if triton_runs:
        return "triton"
    major, minor = compute_capability
    msg = (
        f"MoE expert GEMMs have no kernel on SM {major}.{minor} in {dtype}"
        f"{'' if triton_installed else ' without Triton'}: they need SM 9.x or "
        "10.x in bfloat16, or sm80+ in bfloat16 or float16 with Triton installed"
    )
    raise RuntimeError(msg)


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
