# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Granite 3.1 MoE."""

from __future__ import annotations

from agilerl.architectures.runtime import (
    DENSE_ATTENTION_VLLM_CONFIG,
    ModelRuntimeConfig,
    ModelTypes,
)

FAMILY = ModelRuntimeConfig(
    name="granitemoe",
    model_types=ModelTypes(text=("granitemoe",)),
    vllm=DENSE_ATTENTION_VLLM_CONFIG,
    chat_format="granite3",
    liger="agilerl.architectures.granite_liger:register_granite_liger",
    tensor_parallel_plan=(
        "agilerl.architectures.granitemoe.tensor_parallel:TENSOR_PARALLEL_PLAN"
    ),
)
