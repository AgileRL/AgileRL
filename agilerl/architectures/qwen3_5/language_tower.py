# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 language-tower mapping for vLLM."""

from __future__ import annotations

from agilerl.architectures.vllm_language import nested_language_config

QWEN3_5_LANGUAGE_ARCHITECTURE = "Qwen3_5LanguageForCausalLM"
QWEN3_5_MOE_LANGUAGE_ARCHITECTURE = "Qwen3_5MoeLanguageForCausalLM"
QWEN3_5_MOE_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})
QWEN3_5_MODEL_CLASS_OVERRIDES = {
    QWEN3_5_LANGUAGE_ARCHITECTURE: (
        "agilerl.architectures.qwen3_5.causal_lm:Qwen3_5LanguageForCausalLM"
    ),
    QWEN3_5_MOE_LANGUAGE_ARCHITECTURE: (
        "agilerl.architectures.qwen3_5.causal_lm:Qwen3_5MoeLanguageForCausalLM"
    ),
}


def qwen3_5_language_tower_hf_override(config: object) -> object:
    """Point vLLM at a Qwen3.5 checkpoint's language tower.

    Qwen3.5 ``text_config`` has ``model_type`` ``qwen3_5_text`` or
    ``qwen3_5_moe_text`` and no ``architectures`` list. vLLM only registers the
    multimodal ConditionalGeneration classes.

    :param config: Hugging Face config loaded from the checkpoint.
    :type config: object
    :return: Nested language config with the language-tower architecture.
    :rtype: object
    """
    language = nested_language_config(config)
    model_type = getattr(language, "model_type", None) or getattr(
        config, "model_type", None
    )
    architecture = (
        QWEN3_5_MOE_LANGUAGE_ARCHITECTURE
        if model_type in QWEN3_5_MOE_MODEL_TYPES
        else QWEN3_5_LANGUAGE_ARCHITECTURE
    )
    vars(language)["architectures"] = [architecture]
    return language
