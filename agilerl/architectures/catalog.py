# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Family trainer, vLLM, and patch defaults keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Mapping

from transformers import AutoModelForImageTextToText
from transformers.configuration_utils import PretrainedConfig

from agilerl.architectures.gemma4 import install_gemma4_vision_patches
from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.qwen3_5.packed import install_qwen_gdn_patches
from agilerl.architectures.qwen3_5.vision import install_qwen3_5_vl_patches
from agilerl.architectures.runtime import (
    MambaPatchConfig,
    ModelRuntimeConfig,
    PatchRuntimeConfig,
    TrainerModelClass,
    TrainerRuntimeConfig,
    VllmRuntimeConfig,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.utils.patching import library_shim

# Trainer uses transformers' nemotron_h class; the checkpoint's auto_map code
# supports only eager attention.
NEMOTRON_H_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flash_attention_2"),
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        reasoning_parser="nemotron_v3",
        enable_prefix_caching=True,
        dense_mamba_prefix_cache=True,
    ),
    patch=PatchRuntimeConfig(
        install=install_mamba_patches,
        mamba=MambaPatchConfig(
            mixer="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer",
            block="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHBlock",
        ),
    ),
    tensor_parallel_plan=(
        "agilerl.architectures.nemotron_h.tensor_parallel:"
        "NEMOTRON_H_TENSOR_PARALLEL_PLAN"
    ),
    tool_call_parser="tool_call",
)


@library_shim("vllm", "==0.32.*")
def nemotron_h_omni_vllm_hf_override(config: object) -> object:
    """Point a native ``nemotron_h_omni`` config at NemotronHVLLoRA and restore ``vit_hidden_size``.

    :param config: Hugging Face config loaded from the checkpoint.
    :type config: object
    :return: The config vLLM's Nemotron VL model can build from.
    :rtype: object
    """
    fields = vars(config)
    # Serve through NemotronHVLLoRA, which adds tower and connector LoRA.
    fields["architectures"] = ["NemotronH_Super_Omni_Reasoning_V3"]
    # vLLM's Nemotron VL model reads the checkpoint's legacy name; the native
    # config stores it as vision_hidden_size. vLLM also probes with a dummy config.
    if "vision_hidden_size" in fields:
        fields["vit_hidden_size"] = fields["vision_hidden_size"]
    return config


NEMOTRON_H_OMNI_RUNTIME_CONFIG = NEMOTRON_H_RUNTIME_CONFIG.model_copy(
    update={
        # Native nemotron_h_omni; remote code stopped loading on 5.19.
        "trainer": TrainerRuntimeConfig(
            model_class=AutoModelForImageTextToText,
            attn_implementation="flash_attention_2",
            trust_remote_code=False,
        ),
        "vllm": NEMOTRON_H_RUNTIME_CONFIG.vllm.model_copy(
            update={"hf_overrides": nemotron_h_omni_vllm_hf_override},
        ),
        "tower_connector_lora_targets": frozenset(
            {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "fc1",
                "fc2",
                "patch_projection",
                "video_patch_projection",
                "linear_1",
                "linear_2",
            }
        ),
    },
)

GEMMA_SWA_TEXT_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flex_attention"),
)

GEMMA_SWA_RUNTIME_CONFIG = GEMMA_SWA_TEXT_RUNTIME_CONFIG.model_copy(
    update={
        "trainer": TrainerRuntimeConfig(
            model_class=AutoModelForImageTextToText,
            attn_implementation="flex_attention",
        ),
    },
)

GEMMA4_TEXT_RUNTIME_CONFIG = GEMMA_SWA_TEXT_RUNTIME_CONFIG.model_copy(
    update={"tool_call_parser": "gemma4"},
)

GEMMA4_RUNTIME_CONFIG = GEMMA_SWA_RUNTIME_CONFIG.model_copy(
    update={
        "tool_call_parser": "gemma4",
        "patch": PatchRuntimeConfig(install=install_gemma4_vision_patches),
        # vLLM swaps every tower linear for a LoRA layer and keeps HF names;
        # attention and MLP linears sit under clippable wrappers as ``linear``.
        "tower_connector_lora_targets": frozenset(
            {
                "linear",
                "input_proj",
                "embedding_projection",
                "input_proj_linear",
                "output_proj",
                "relative_k_proj",
            }
        ),
        "tower_modalities": {
            "vision_tower": frozenset({"image", "video"}),
            "embed_vision": frozenset({"image", "video"}),
            "audio_tower": frozenset({"audio"}),
            "embed_audio": frozenset({"audio"}),
        },
    },
)

GPT_OSS_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flex_attention"),
    tool_call_parser="harmony",
)

# Families whose only catalog setting is the tool-call format.
TOOL_CALL_TAG_RUNTIME_CONFIG = ModelRuntimeConfig(tool_call_parser="tool_call")
GRANITE3_RUNTIME_CONFIG = ModelRuntimeConfig(tool_call_parser="granite3")


# Linear-attention rejects mamba_cache_mode "all"; align needs prefix caching.
QWEN3_5_TEXT_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        enable_prefix_caching=True,
        dense_mamba_prefix_cache=True,
    ),
    patch=PatchRuntimeConfig(install=install_qwen_gdn_patches),
    tool_call_parser="tool_call",
)

# AutoModelForCausalLM maps qwen3_5 to a text-only class that drops model.visual.
QWEN3_5_RUNTIME_CONFIG = QWEN3_5_TEXT_RUNTIME_CONFIG.model_copy(
    update={
        "trainer": TrainerRuntimeConfig(model_class=AutoModelForImageTextToText),
        "patch": PatchRuntimeConfig(install=install_qwen3_5_vl_patches),
        # The patch embedding ``proj`` is a Conv3d that vLLM serves without LoRA.
        "tower_connector_lora_targets": frozenset({"qkv", "linear_fc1", "linear_fc2"}),
        "tower_modalities": {"visual": frozenset({"image", "video"})},
    },
)

FAMILY_RUNTIME_CONFIGS: Mapping[str, ModelRuntimeConfig] = {
    "qwen2": TOOL_CALL_TAG_RUNTIME_CONFIG,
    "qwen3": TOOL_CALL_TAG_RUNTIME_CONFIG,
    "qwen3_moe": TOOL_CALL_TAG_RUNTIME_CONFIG,
    "granitemoe": GRANITE3_RUNTIME_CONFIG,
    "granitemoehybrid": TOOL_CALL_TAG_RUNTIME_CONFIG,
    "nemotron_h": NEMOTRON_H_RUNTIME_CONFIG,
    "nemotron_h_omni": NEMOTRON_H_OMNI_RUNTIME_CONFIG,
    "gemma3": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma3_text": GEMMA_SWA_TEXT_RUNTIME_CONFIG,
    "gemma4": GEMMA4_RUNTIME_CONFIG,
    "gemma4_text": GEMMA4_TEXT_RUNTIME_CONFIG,
    "gpt_oss": GPT_OSS_RUNTIME_CONFIG,
    "qwen3_5": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_text": QWEN3_5_TEXT_RUNTIME_CONFIG,
    "qwen3_5_moe": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_moe_text": QWEN3_5_TEXT_RUNTIME_CONFIG,
}


def pretrained_model_type(model_name_or_path: str) -> str:
    """Return Hugging Face ``model_type`` from a checkpoint id or local path.

    Supported ids read the bundled config instead of the Hub.
    """
    entry = SUPPORTED_MODEL_INFO.get(model_name_or_path)
    if entry is not None:
        return entry.config["model_type"]
    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    return config_dict["model_type"]


def family_runtime(model_name_or_path: str) -> ModelRuntimeConfig:
    """Return catalog runtime for a checkpoint id or local path."""
    return FAMILY_RUNTIME_CONFIGS.get(
        pretrained_model_type(model_name_or_path),
        ModelRuntimeConfig(),
    )


def trainer_model_class(model_name_or_path: str) -> TrainerModelClass:
    """Return the auto class every trainer load of this checkpoint builds.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: ``AutoModelForCausalLM``, or ``AutoModelForImageTextToText`` for VL families.
    :rtype: TrainerModelClass
    """
    return family_runtime(model_name_or_path).trainer.model_class
