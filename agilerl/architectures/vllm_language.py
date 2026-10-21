# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Map a multimodal checkpoint onto the language tower vLLM serves."""

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


def apply_language_tower_engine_kwargs(
    kwargs: dict[str, Any],
    strip_multimodal_towers: bool | list[str],
    runtime: ModelRuntimeConfig,
) -> None:
    """Set ``hf_overrides`` and family class mapping when serving a language tower.

    :param kwargs: vLLM engine kwargs mutated in place.
    :type kwargs: dict[str, Any]
    :param strip_multimodal_towers: ``True`` serves the language tower only. A
        list still loads the multimodal engine; named towers are freed later.
    :type strip_multimodal_towers: bool | list[str]
    :param runtime: Family runtime whose language-tower mapping, if any, is applied.
    :type runtime: ModelRuntimeConfig
    """
    if strip_multimodal_towers is True:
        tower = runtime.language_tower
        if tower.hf_overrides is not None:
            kwargs["hf_overrides"] = tower.hf_overrides
        else:
            kwargs["hf_overrides"] = nested_language_config
        if tower.model_class_overrides:
            kwargs["model_class_overrides"] = {
                **kwargs.get("model_class_overrides", {}),
                **tower.model_class_overrides,
            }
        return

    kept_override = runtime.multimodal_towers_kept_hf_override
    if kept_override is not None:
        kwargs["hf_overrides"] = kept_override


def apply_tower_connector_lora_engine_kwargs(
    kwargs: dict[str, Any],
    strip_multimodal_towers: bool | list[str],
    runtime: ModelRuntimeConfig,
    lora_target_modules: str | Collection[str] | None,
) -> None:
    """Enable vLLM LoRA on multimodal towers and connectors when the adapter can reach them.

    vLLM wraps every tower and connector linear once this is on, so an adapter
    that only targets the language model would pay for empty LoRA layers.

    :param kwargs: vLLM engine kwargs mutated in place.
    :type kwargs: dict[str, Any]
    :param strip_multimodal_towers: ``True`` serves the language tower only.
    :type strip_multimodal_towers: bool | list[str]
    :param runtime: Family runtime; only families whose vLLM towers support LoRA enable it.
    :type runtime: ModelRuntimeConfig
    :param lora_target_modules: PEFT ``target_modules``. A string spec
        (``"all-linear"`` or a regex) or ``None`` counts as reaching the towers.
    :type lora_target_modules: str | Collection[str] | None
    """
    tower_targets = runtime.tower_connector_lora_targets
    if strip_multimodal_towers is True or not tower_targets:
        return
    named_targets = not isinstance(lora_target_modules, str | None)
    if named_targets and tower_targets.isdisjoint(lora_target_modules):
        return
    kwargs["enable_tower_connector_lora"] = True
