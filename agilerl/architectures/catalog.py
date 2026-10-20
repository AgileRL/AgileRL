# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Family trainer, vLLM, and patch defaults keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Mapping

from transformers.configuration_utils import PretrainedConfig

from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.qwen3_5 import (
    QWEN3_5_MODEL_CLASS_OVERRIDES,
    qwen3_5_language_tower_hf_override,
)
from agilerl.architectures.qwen3_5.packed import install_qwen_gdn_patches
from agilerl.architectures.runtime import (
    LanguageTowerRuntimeConfig,
    MambaPatchConfig,
    ModelRuntimeConfig,
    PatchRuntimeConfig,
    TrainerRuntimeConfig,
    VllmRuntimeConfig,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

# Trainer uses transformers' nemotron_h class; the checkpoint's auto_map code
# supports only eager attention.
NEMOTRON_H_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flash_attention_2"),
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        reasoning_parser="nemotron_v3",
        enable_prefix_caching=True,
        trust_remote_code=True,
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
)

NEMOTRON_H_OMNI_RUNTIME_CONFIG = NEMOTRON_H_RUNTIME_CONFIG.model_copy(
    update={
        # nemotron_h_omni is not a transformers model type; its config needs checkpoint code.
        "trainer": TrainerRuntimeConfig(
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ),
        # The checkpoint's NemotronH_Omni_Reasoning_V3 is not a vLLM architecture.
        "vllm": NEMOTRON_H_RUNTIME_CONFIG.vllm.model_copy(
            update={
                "hf_overrides": {
                    "architectures": ["NemotronH_Super_Omni_Reasoning_V3"],
                },
            },
        ),
        "tower_connector_lora_targets": frozenset(
            {
                "query",
                "key",
                "value",
                "dense",
                "fc1",
                "fc2",
                "patch_projection",
                "video_patch_projection",
                "linear1",
                "linear2",
            }
        ),
    },
)

GEMMA_SWA_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flex_attention"),
)

GPT_OSS_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flex_attention"),
)

# Linear-attention rejects mamba_cache_mode "all"; align needs prefix caching.
QWEN3_5_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        enable_prefix_caching=True,
    ),
    language_tower=LanguageTowerRuntimeConfig(
        hf_overrides=qwen3_5_language_tower_hf_override,
        model_class_overrides=QWEN3_5_MODEL_CLASS_OVERRIDES,
    ),
    patch=PatchRuntimeConfig(install=install_qwen_gdn_patches),
)

FAMILY_RUNTIME_CONFIGS: Mapping[str, ModelRuntimeConfig] = {
    "nemotron_h": NEMOTRON_H_RUNTIME_CONFIG,
    "nemotron_h_omni": NEMOTRON_H_OMNI_RUNTIME_CONFIG,
    "gemma3": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma3_text": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma4": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma4_text": GEMMA_SWA_RUNTIME_CONFIG,
    "gpt_oss": GPT_OSS_RUNTIME_CONFIG,
    "qwen3_5": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_text": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_moe": QWEN3_5_RUNTIME_CONFIG,
    "qwen3_5_moe_text": QWEN3_5_RUNTIME_CONFIG,
}


def pretrained_model_type(model_name_or_path: str) -> str:
    """Return Hugging Face ``model_type`` from a checkpoint id or local path.

    Supported ids read the bundled config instead of the Hub.
    """
    entry = SUPPORTED_MODEL_INFO.get(model_name_or_path)
    if entry is not None:
        return entry.config["model_type"]
    config_dict, _ = PretrainedConfig.get_config_dict(
        model_name_or_path,
        trust_remote_code=True,
    )
    return config_dict["model_type"]


def family_runtime(model_name_or_path: str) -> ModelRuntimeConfig:
    """Return catalog runtime for a checkpoint id or local path."""
    return FAMILY_RUNTIME_CONFIGS.get(
        pretrained_model_type(model_name_or_path),
        ModelRuntimeConfig(),
    )
