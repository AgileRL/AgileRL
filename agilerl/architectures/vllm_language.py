# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Serve a multimodal checkpoint in vLLM with or without its towers."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol, runtime_checkable

from agilerl.architectures.runtime import ModelRuntimeConfig


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
    strip_multimodal_towers: bool,
) -> None:
    """Serve the language model only when the towers are stripped.

    ``language_model_only`` keeps the served module names unchanged.

    :param kwargs: vLLM engine kwargs mutated in place.
    :type kwargs: dict[str, Any]
    :param strip_multimodal_towers: Whether the engine skips the towers.
    :type strip_multimodal_towers: bool
    """
    if strip_multimodal_towers:
        kwargs["language_model_only"] = True


def unbuilt_tower_modules(
    strip_multimodal_towers: bool,
    runtime: ModelRuntimeConfig,
) -> list[str]:
    """Trainer tower and connector modules vLLM leaves unbuilt.

    :param strip_multimodal_towers: Whether the engine skips the towers.
    :type strip_multimodal_towers: bool
    :param runtime: Family runtime mapping tower modules to their modalities.
    :type runtime: ModelRuntimeConfig
    :return: Every tower module when stripped, else none; sorted.
    :rtype: list[str]
    """
    return sorted(runtime.tower_modalities) if strip_multimodal_towers else []


def apply_tower_connector_lora_engine_kwargs(
    kwargs: dict[str, Any],
    strip_multimodal_towers: bool,
    runtime: ModelRuntimeConfig,
    tower_target_modules: Collection[str] | None,
) -> None:
    """Enable vLLM LoRA on multimodal towers and connectors when the adapter names them.

    vLLM wraps every tower and connector linear once this is on, so it stays
    off for adapters on the language model only.

    :param kwargs: vLLM engine kwargs mutated in place.
    :type kwargs: dict[str, Any]
    :param strip_multimodal_towers: Whether the engine skips the towers.
    :type strip_multimodal_towers: bool
    :param runtime: Family runtime naming the tower targets vLLM supports.
    :type runtime: ModelRuntimeConfig
    :param tower_target_modules: ``lora_config.tower_target_modules``.
    :type tower_target_modules: Collection[str] | None
    :raises ValueError: If towers are stripped, or a target is not a tower
        module vLLM serves LoRA on for this family.
    """
    if not tower_target_modules:
        return
    if strip_multimodal_towers:
        msg = (
            "lora_config.tower_target_modules needs the towers served; the "
            "engine strips them"
        )
        raise ValueError(msg)
    unsupported = sorted(
        set(tower_target_modules) - runtime.tower_connector_lora_targets
    )
    if unsupported:
        supported = sorted(runtime.tower_connector_lora_targets) or "none"
        msg = (
            f"vLLM cannot serve tower LoRA on {unsupported}; this family "
            f"supports {supported}"
        )
        raise ValueError(msg)
    kwargs["enable_tower_connector_lora"] = True
