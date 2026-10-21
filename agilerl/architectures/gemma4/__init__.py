# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 language-tower mapping for vLLM."""

from __future__ import annotations

from agilerl.architectures.gemma4.language_tower import (
    GEMMA4_LANGUAGE_ARCHITECTURE,
    gemma4_language_tower_hf_override,
)

__all__ = [
    "GEMMA4_LANGUAGE_ARCHITECTURE",
    "gemma4_language_tower_hf_override",
]
