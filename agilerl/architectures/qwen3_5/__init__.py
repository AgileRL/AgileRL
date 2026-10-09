# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 language-tower mapping for vLLM."""

from __future__ import annotations

from agilerl.architectures.qwen3_5.language_tower import (
    QWEN3_5_LANGUAGE_ARCHITECTURE,
    QWEN3_5_MODEL_CLASS_OVERRIDES,
    QWEN3_5_MOE_LANGUAGE_ARCHITECTURE,
    qwen3_5_language_tower_hf_override,
)
from agilerl.architectures.qwen3_5.packed import (
    install_qwen_gdn_patches,
    patch_qwen_gdn_packed_sequences,
)

__all__ = [
    "QWEN3_5_LANGUAGE_ARCHITECTURE",
    "QWEN3_5_MODEL_CLASS_OVERRIDES",
    "QWEN3_5_MOE_LANGUAGE_ARCHITECTURE",
    "install_qwen_gdn_patches",
    "patch_qwen_gdn_packed_sequences",
    "qwen3_5_language_tower_hf_override",
]
