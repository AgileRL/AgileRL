# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Granite 4 hybrid Mamba + MoE."""

from __future__ import annotations

from agilerl.architectures.granitemoehybrid.patches import (
    install_granitemoehybrid_patches,
)
from agilerl.architectures.runtime import (
    DENSE_ATTENTION_VLLM_CONFIG,
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
)

FAMILY = ModelRuntimeConfig(
    name="granitemoehybrid",
    model_types=ModelTypes(text=("granitemoehybrid",)),
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    chat_format="tool_call",
    patch=PatchRuntimeConfig(install=install_granitemoehybrid_patches),
    liger="agilerl.architectures.granite_liger:register_granite_liger",
    tensor_parallel_plan=(
        "agilerl.architectures.granitemoehybrid.tensor_parallel:TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.granitemoehybrid.context_parallel:CONTEXT_PARALLEL_PLAN"
    ),
)
