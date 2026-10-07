# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Serve a multimodal checkpoint in vLLM with some or all of its towers skipped."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agilerl.architectures.runtime import ModelRuntimeConfig
from agilerl.arena.models.networks import VllmModality


@runtime_checkable
class NestedTextConfig(Protocol):
    """Hugging Face VL config that nests the language model under ``text_config``."""

    text_config: object


@runtime_checkable
class NestedLlmConfig(Protocol):
    """Hugging Face VL config that nests the language model under ``llm_config``."""

    llm_config: object


def nested_language_config(config: object) -> object:
    """Return the nested language config on a VL checkpoint, or *config* itself.

    Hugging Face VL configs nest the language model under ``text_config`` or
    ``llm_config``. A language-only checkpoint has neither.
    """
    if isinstance(config, NestedTextConfig) and config.text_config is not None:
        return config.text_config
    if isinstance(config, NestedLlmConfig) and config.llm_config is not None:
        return config.llm_config
    return config


def apply_multimodal_engine_kwargs(
    kwargs: dict[str, Any],
    strip_multimodal_towers: bool | list[VllmModality],
    runtime: ModelRuntimeConfig,
) -> None:
    """Skip stripped towers in vLLM and enable LoRA on the towers it still loads.

    vLLM leaves a tower unbuilt and unloaded when every modality it serves has a
    zero ``limit_mm_per_prompt``. The model class and its module names do not
    change, so LoRA keys map the same way whether towers are stripped or kept.

    :param kwargs: vLLM engine kwargs mutated in place.
    :type kwargs: dict[str, Any]
    :param strip_multimodal_towers: ``True`` serves the language model only; a
        list names the modalities (``image``, ``video``, ``audio``) to drop.
    :type strip_multimodal_towers: bool | list[VllmModality]
    :param runtime: Family runtime; only families whose vLLM towers support LoRA
        enable it.
    :type runtime: ModelRuntimeConfig
    """
    if strip_multimodal_towers is True:
        kwargs["language_model_only"] = True
        return
    if strip_multimodal_towers:
        kwargs["limit_mm_per_prompt"] = {
            **kwargs.get("limit_mm_per_prompt", {}),
            **dict.fromkeys(strip_multimodal_towers, 0),
        }
    if runtime.enable_tower_connector_lora:
        kwargs["enable_tower_connector_lora"] = True
