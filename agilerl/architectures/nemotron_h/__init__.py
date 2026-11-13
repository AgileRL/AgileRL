# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nemotron-H hybrid Mamba2."""

from __future__ import annotations

from agilerl.architectures.nemotron_h.patches import install_nemotron_h_patches
from agilerl.architectures.runtime import (
    MambaPatchConfig,
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
    VllmRuntimeConfig,
)
from agilerl.arena.models.model_info import REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="nemotron_h",
    model_types=ModelTypes(text=("nemotron_h",)),
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        dense_mamba_prefix_cache=True,
    ),
    patch=PatchRuntimeConfig(
        install=install_nemotron_h_patches,
        mamba=MambaPatchConfig(
            mixer="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer",
            block="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHBlock",
        ),
    ),
    liger="agilerl.architectures.nemotron_h.liger:register_nemotron_h_liger",
    tensor_parallel_plan=(
        "agilerl.architectures.nemotron_h.tensor_parallel:TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.nemotron_h.context_parallel:CONTEXT_PARALLEL_PLAN"
    ),
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["nemotron_h"],
)
