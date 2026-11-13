# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4."""

from __future__ import annotations

from agilerl.architectures.deepseek_v4.patches import install_deepseek_v4_patches
from agilerl.architectures.runtime import (
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
)
from agilerl.arena.models.model_info import LORA_EXCLUDED_MODULES, REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="deepseek_v4",
    model_types=ModelTypes(text=("deepseek_v4",)),
    chat_format="dsml",
    reasoning_parser=REASONING_PARSERS["deepseek_v4"],
    lora_exclude_modules=LORA_EXCLUDED_MODULES["deepseek_v4"],
    patch=PatchRuntimeConfig(install=install_deepseek_v4_patches),
    tensor_parallel_plan=(
        "agilerl.architectures.deepseek_v4.tensor_parallel:TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.deepseek_v4.context_parallel:CONTEXT_PARALLEL_PLAN"
    ),
)
