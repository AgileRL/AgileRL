# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 GatedDeltaNet packed-sequence patches."""

from __future__ import annotations

from agilerl.architectures.qwen3_5.packed import (
    install_qwen_gdn_patches,
    patch_qwen_gdn_packed_sequences,
)

__all__ = [
    "install_qwen_gdn_patches",
    "patch_qwen_gdn_packed_sequences",
]
