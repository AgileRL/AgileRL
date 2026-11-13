# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 text and vision-language."""

from __future__ import annotations

from agilerl.architectures.gemma4.vision import install_gemma4_vision
from agilerl.architectures.runtime import ModelRuntimeConfig, ModelTypes, VisionSupport
from agilerl.arena.models.model_info import REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="gemma4",
    model_types=ModelTypes(text=("gemma4_text",), multimodal=("gemma4",)),
    chat_format="gemma4",
    reasoning_parser=REASONING_PARSERS["gemma4"],
    tensor_parallel_plan="agilerl.architectures.gemma4.tensor_parallel:TENSOR_PARALLEL_PLAN",
    vision=VisionSupport(
        install=install_gemma4_vision,
        # vLLM swaps every tower linear for a LoRA layer and keeps HF names;
        # attention and MLP linears sit under clippable wrappers as ``linear``.
        tower_connector_lora_targets=frozenset(
            {
                "linear",
                "input_proj",
                "embedding_projection",
                "input_proj_linear",
                "output_proj",
                "relative_k_proj",
            }
        ),
        tower_modalities={
            "vision_tower": frozenset({"image", "video"}),
            "embed_vision": frozenset({"image", "video"}),
            "audio_tower": frozenset({"audio"}),
            "embed_audio": frozenset({"audio"}),
        },
    ),
)
