# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 dense; its tensor-parallel plan also serves Qwen2."""

from __future__ import annotations

from agilerl.architectures.runtime import (
    DENSE_ATTENTION_VLLM_CONFIG,
    ModelRuntimeConfig,
    ModelTypes,
)
from agilerl.arena.models.model_info import REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="qwen3",
    model_types=ModelTypes(text=("qwen3",)),
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["qwen3"],
    tensor_parallel_plan="agilerl.architectures.qwen3.tensor_parallel:TENSOR_PARALLEL_PLAN",
)
