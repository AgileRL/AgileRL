# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 language-tower mapping for vLLM."""

from __future__ import annotations

from agilerl.architectures.vllm_language import nested_language_config

GEMMA4_LANGUAGE_ARCHITECTURE = "Gemma4ForCausalLM"


def gemma4_language_tower_hf_override(config: object) -> object:
    """Point vLLM at a Gemma 4 checkpoint's language tower.

    Gemma 4 ``text_config`` has ``model_type`` ``gemma4_text`` and no
    ``architectures`` list. vLLM's language-only class is
    ``Gemma4ForCausalLM``.

    :param config: Hugging Face config loaded from the checkpoint.
    :type config: object
    :return: Nested language config with the language-tower architecture.
    :rtype: object
    """
    language = nested_language_config(config)
    vars(language)["architectures"] = [GEMMA4_LANGUAGE_ARCHITECTURE]
    return language
