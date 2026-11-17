# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen2 dense."""

from __future__ import annotations

from agilerl.architectures.runtime import (
    DENSE_ATTENTION_VLLM_CONFIG,
    ModelRuntimeConfig,
    ModelTypes,
)

FAMILY = ModelRuntimeConfig(
    name="qwen2",
    model_types=ModelTypes(text=("qwen2",)),
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    chat_format="tool_call",
    tensor_parallel_plan="agilerl.architectures.qwen3.tensor_parallel:TENSOR_PARALLEL_PLAN",
)
