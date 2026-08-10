# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H patches: Liger kernels and Mamba2 mixer workarounds."""

from __future__ import annotations

from agilerl.architectures.nemotron_h.liger import (
    apply_liger_kernel_to_nemotron_h,
    register_nemotron_h_liger,
)
from agilerl.architectures.nemotron_h.mamba import (
    patch_nemotron_mamba_fused_path,
    patch_nemotron_mamba_stream_ordering,
)

__all__ = [
    "apply_liger_kernel_to_nemotron_h",
    "patch_nemotron_mamba_fused_path",
    "patch_nemotron_mamba_stream_ordering",
    "register_nemotron_h_liger",
]
