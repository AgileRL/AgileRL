# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nemotron omni language-tower mapping for vLLM."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agilerl.architectures.vllm_language import nested_language_config


@runtime_checkable
class HfArchitectures(Protocol):
    """Hugging Face config field vLLM reads for the model class."""

    architectures: list[str]


def omni_language_tower_hf_override(config: object) -> HfArchitectures:
    """Point vLLM at a Nemotron omni checkpoint's language tower.

    :param config: Hugging Face config loaded from the checkpoint.
    :type config: object
    :return: Nested language config with the language-tower architecture.
    :rtype: HfArchitectures
    """
    language = nested_language_config(config)
    if not isinstance(language, HfArchitectures):
        msg = (
            f"{type(language).__name__} has no architectures list for "
            "Nemotron omni language-tower mapping"
        )
        raise TypeError(msg)
    language.architectures = ["NemotronHOmniLanguageForCausalLM"]
    return language
