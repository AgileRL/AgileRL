# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Family trainer, vLLM, and patch defaults keyed by Hugging Face ``model_type``."""

from __future__ import annotations

from collections.abc import Mapping

from transformers import AutoConfig

from agilerl.architectures.nemotron_h.mamba import install_mamba_patches
from agilerl.architectures.runtime import (
    MambaPatchConfig,
    ModelRuntimeConfig,
    PatchRuntimeConfig,
    TrainerRuntimeConfig,
    VllmRuntimeConfig,
)

NEMOTRON_H_RUNTIME_CONFIG = ModelRuntimeConfig(
    vllm=VllmRuntimeConfig(
        mamba_cache_mode="align",
        max_num_batched_tokens=8192,
        reasoning_parser="nemotron_v3",
        enable_prefix_caching=True,
    ),
    patch=PatchRuntimeConfig(
        install=install_mamba_patches,
        mamba=MambaPatchConfig(
            mixer="transformers.models.nemotron_h.modeling_nemotron_h.NemotronHMamba2Mixer",
        ),
    ),
)

GEMMA_SWA_RUNTIME_CONFIG = ModelRuntimeConfig(
    trainer=TrainerRuntimeConfig(attn_implementation="flex_attention"),
)

FAMILY_RUNTIME_CONFIGS: Mapping[str, ModelRuntimeConfig] = {
    "nemotron_h": NEMOTRON_H_RUNTIME_CONFIG,
    "gemma3": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma3_text": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma4": GEMMA_SWA_RUNTIME_CONFIG,
    "gemma4_text": GEMMA_SWA_RUNTIME_CONFIG,
}


def pretrained_model_type(model_name_or_path: str) -> str:
    """Return Hugging Face ``model_type`` from a checkpoint id or local path."""
    return AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    ).model_type


def family_runtime(model_name_or_path: str) -> ModelRuntimeConfig:
    """Return catalog runtime for a checkpoint id or local path."""
    return FAMILY_RUNTIME_CONFIGS.get(
        pretrained_model_type(model_name_or_path),
        ModelRuntimeConfig(),
    )
