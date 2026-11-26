# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Fused or unfused Mamba scan."""

from __future__ import annotations

from agilerl.arena.memory.specs import UNFUSED_MAMBA_SCAN_CAPABILITY


def mamba_scan_is_unfused(compute_capability: tuple[int, int]) -> bool:
    """Whether Mamba layers train with HF's unfused path on this device.

    :param compute_capability: CUDA ``(major, minor)``.
    :return: ``True`` on compute capability 8.9 (Ada, e.g. L4), where the
        fused SSM kernels return NaN.
    """
    return compute_capability == UNFUSED_MAMBA_SCAN_CAPABILITY
