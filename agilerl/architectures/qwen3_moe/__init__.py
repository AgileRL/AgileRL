# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 MoE."""

from __future__ import annotations

from agilerl.architectures.runtime import (
    DENSE_ATTENTION_VLLM_CONFIG,
    ModelRuntimeConfig,
    ModelTypes,
)
from agilerl.arena.models.model_info import REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="qwen3_moe",
    model_types=ModelTypes(text=("qwen3_moe",)),
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["qwen3_moe"],
)
