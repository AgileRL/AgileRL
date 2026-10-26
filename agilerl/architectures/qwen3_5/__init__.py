# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 GatedDeltaNet packed-sequence and vision-row patches."""

from __future__ import annotations

from agilerl.architectures.qwen3_5.packed import (
    install_qwen_gdn_patches,
    patch_qwen_gdn_packed_sequences,
)
from agilerl.architectures.qwen3_5.vision import (
    install_qwen3_5_vl_patches,
    patch_qwen_vision_rows,
)

__all__ = [
    "install_qwen3_5_vl_patches",
    "install_qwen_gdn_patches",
    "patch_qwen_gdn_packed_sequences",
    "patch_qwen_vision_rows",
]
