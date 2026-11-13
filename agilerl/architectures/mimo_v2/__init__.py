# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""MiMo-V2 vision-language, loaded natively."""

from __future__ import annotations

from agilerl.architectures.mimo_v2.configuration import MiMoV2Config
from agilerl.architectures.mimo_v2.patches import install_mimo_v2_patches
from agilerl.architectures.mimo_v2.processing import MiMoV2Processor
from agilerl.architectures.runtime import (
    ModelRuntimeConfig,
    ModelTypes,
    PatchRuntimeConfig,
    TrainerRuntimeConfig,
    VisionSupport,
    VllmRuntimeConfig,
)
from agilerl.arena.models.model_info import LORA_EXCLUDED_MODULES, REASONING_PARSERS

FAMILY = ModelRuntimeConfig(
    name="mimo_v2",
    model_types=ModelTypes(multimodal=("mimo_v2",)),
    # The Hub auto_map names repo code: vLLM parses it, the trainer loads the
    # config with the native class.
    vllm=VllmRuntimeConfig(
        trust_remote_code=True,
        # vLLM maps the checkpoint's MiMoV2ForCausalLM to a text-only model.
        hf_overrides={"architectures": ["MiMoV2OmniForCausalLM"]},
    ),
    trainer=TrainerRuntimeConfig(trust_remote_code=False, config_class=MiMoV2Config),
    patch=PatchRuntimeConfig(install=install_mimo_v2_patches),
    liger="agilerl.architectures.mimo_v2.liger:register_mimo_v2_liger",
    tensor_parallel_plan="agilerl.architectures.mimo_v2.tensor_parallel:TENSOR_PARALLEL_PLAN",
    chat_format="tool_call",
    reasoning_parser=REASONING_PARSERS["mimo_v2"],
    lora_exclude_modules=LORA_EXCLUDED_MODULES["mimo_v2"],
    vision=VisionSupport(
        image_token=MiMoV2Processor.image_token,
        processor_class=MiMoV2Processor,
        # The patch embedding ``proj`` is a Conv3d that vLLM serves without LoRA.
        tower_connector_lora_targets=frozenset(
            {"qkv", "gate_proj", "up_proj", "down_proj"}
        ),
        tower_modalities={"visual": frozenset({"image", "video"})},
    ),
)
