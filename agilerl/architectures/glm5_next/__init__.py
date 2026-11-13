# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 text and vision-language."""

from __future__ import annotations

from agilerl.architectures.glm5_next.patches import install_glm5_next_patches
from agilerl.architectures.glm5_next.vision import install_glm5_next_vision
from agilerl.architectures.runtime import (
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
    VisionSupport,
)
from agilerl.arena.models.model_info import LORA_EXCLUDED_MODULES, REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="glm5_next",
    model_types=ModelTypes(text=("glm5_next_text",), multimodal=("glm5_next",)),
    chat_format="glm47",
    reasoning_parser=REASONING_PARSERS["glm5_next"],
    lora_exclude_modules=LORA_EXCLUDED_MODULES["glm5_next"],
    patch=PatchRuntimeConfig(install=install_glm5_next_patches),
    liger="agilerl.architectures.glm5_next.liger:register_glm5_next_liger",
    tensor_parallel_plan="agilerl.architectures.glm5_next.tensor_parallel:TENSOR_PARALLEL_PLAN",
    context_parallel_plan=(
        "agilerl.architectures.glm5_next.context_parallel:CONTEXT_PARALLEL_PLAN"
    ),
    vision=VisionSupport(
        install=install_glm5_next_vision,
        # The GLM-5.3 tokenizer names no image_token; its processor expands <|image|>.
        image_token="<|image|>",
        # The patch embedding ``proj`` is a Conv3d that vLLM serves without LoRA.
        tower_connector_lora_targets=frozenset(
            {"qkv", "gate_proj", "up_proj", "down_proj"}
        ),
        tower_modalities={"visual": frozenset({"image", "video"})},
    ),
)
