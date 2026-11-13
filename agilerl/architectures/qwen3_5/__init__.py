# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 / Qwen3.6 GatedDeltaNet hybrids, text and vision-language."""

from __future__ import annotations

from agilerl.architectures.qwen3_5.patches import install_qwen3_5_patches
from agilerl.architectures.qwen3_5.vision import install_qwen3_5_vision
from agilerl.architectures.runtime import (
    GdnMixerConfig,
    GdnPatchConfig,
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
    VisionSupport,
    VllmRuntimeConfig,
)
from agilerl.arena.models.model_info import REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="qwen3_5",
    model_types=ModelTypes(
        text=("qwen3_5_text", "qwen3_5_moe_text"),
        # AutoModelForCausalLM maps these to a text-only class that drops model.visual.
        multimodal=("qwen3_5", "qwen3_5_moe"),
    ),
    # Linear-attention rejects mamba_cache_mode "all"; align needs prefix caching.
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        enable_prefix_caching=True,
        dense_mamba_prefix_cache=True,
    ),
    patch=PatchRuntimeConfig(
        install=install_qwen3_5_patches,
        gdn=GdnPatchConfig(
            mixers=(
                GdnMixerConfig(
                    mixer=(
                        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                        "Qwen3_5MoeGatedDeltaNet"
                    ),
                    text_model=(
                        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
                        "Qwen3_5MoeTextModel"
                    ),
                ),
                GdnMixerConfig(
                    mixer=(
                        "transformers.models.qwen3_5.modeling_qwen3_5."
                        "Qwen3_5GatedDeltaNet"
                    ),
                    text_model=(
                        "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel"
                    ),
                ),
            ),
        ),
    ),
    tensor_parallel_plan=(
        "agilerl.architectures.qwen3_5.tensor_parallel:TENSOR_PARALLEL_PLAN"
    ),
    context_parallel_plan=(
        "agilerl.architectures.qwen3_5.context_parallel:CONTEXT_PARALLEL_PLAN"
    ),
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["qwen3_5_text"],
    vision=VisionSupport(
        install=install_qwen3_5_vision,
        # The patch embedding ``proj`` is a Conv3d that vLLM serves without LoRA.
        tower_connector_lora_targets=frozenset({"qkv", "linear_fc1", "linear_fc2"}),
        tower_modalities={"visual": frozenset({"image", "video"})},
    ),
)
