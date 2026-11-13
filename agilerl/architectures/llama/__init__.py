# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Llama dense."""

from __future__ import annotations

from agilerl.architectures.runtime import ModelRuntimeConfig, ModelTypes

FAMILY = ModelRuntimeConfig(
    name="llama",
    model_types=ModelTypes(text=("llama",)),
)
