# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3 text and vision-language."""

from __future__ import annotations

from agilerl.architectures.runtime import ModelRuntimeConfig, ModelTypes, VisionSupport

FAMILY = ModelRuntimeConfig(
    name="gemma3",
    model_types=ModelTypes(text=("gemma3_text",), multimodal=("gemma3",)),
    vision=VisionSupport(),
)
