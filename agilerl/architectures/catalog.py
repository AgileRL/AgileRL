# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Model families and per-``model_type`` runtime lookups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.runtime import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    ModelRuntimeConfig,
    TrainerModelClass,
    VisionSupport,
    discover_families,
    runtime_configs_by_model_type,
)
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from transformers.configuration_utils import PretrainedConfig

# Family declarations import transformers; a core install serves no LLM family.
FAMILIES = discover_families("agilerl.architectures") if HAS_LLM_DEPENDENCIES else ()
FAMILY_RUNTIME_CONFIGS: Mapping[str, ModelRuntimeConfig] = (
    runtime_configs_by_model_type(FAMILIES)
)


def pretrained_config(model_name_or_path: str) -> dict[str, Any]:
    """Return ``config.json`` contents from a checkpoint id or local path.

    Supported ids read the bundled config instead of the Hub.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: The checkpoint's top-level config.
    :rtype: dict[str, Any]
    """
    entry = SUPPORTED_MODEL_INFO.get(model_name_or_path)
    if entry is not None:
        return entry.config
    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    return config_dict


def pretrained_model_type(model_name_or_path: str) -> str:
    """Return Hugging Face ``model_type`` from a checkpoint id or local path.

    Supported ids read the bundled config instead of the Hub.
    """
    return pretrained_config(model_name_or_path)["model_type"]


def model_type_runtime(model_type: str) -> ModelRuntimeConfig:
    """Return the family serving a Hugging Face ``model_type``.

    :raises ValueError: If no family serves ``model_type``.
    """
    runtime = FAMILY_RUNTIME_CONFIGS.get(model_type)
    if runtime is None:
        msg = f"model_type {model_type!r} has no family in agilerl.architectures"
        raise ValueError(msg)
    return runtime


def model_type_vision(model_type: str) -> VisionSupport | None:
    """Return the family's vision settings, or None for a text ``model_type``."""
    runtime = model_type_runtime(model_type)
    if model_type not in runtime.model_types.multimodal:
        return None
    return runtime.vision


def model_type_trainer_class(model_type: str) -> TrainerModelClass:
    """Return the auto class every trainer load of a ``model_type`` builds.

    Multimodal checkpoints build their full VL class so towers load, frozen.
    """
    if model_type in model_type_runtime(model_type).model_types.multimodal:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def family_runtime(model_name_or_path: str) -> ModelRuntimeConfig:
    """Return the family serving a checkpoint id or local path."""
    return model_type_runtime(pretrained_model_type(model_name_or_path))


def family_vision(model_name_or_path: str) -> VisionSupport | None:
    """Return the vision settings of a checkpoint id or local path, or None for a text checkpoint."""
    return model_type_vision(pretrained_model_type(model_name_or_path))


def sm80_attention_backend(model_name_or_path: str) -> str | None:
    """Return the async engine attention backend for this checkpoint on sm80 GPUs.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: The family's ``sm80_attention_backend``, or ``None`` when unset or
        the checkpoint has Mamba layers.
    :rtype: str | None
    """
    config = pretrained_config(model_name_or_path)
    # granitemoehybrid checkpoints range from attention-only to mostly Mamba.
    if "mamba" in config.get("layer_types", ()):
        return None
    return model_type_runtime(config["model_type"]).vllm.sm80_attention_backend


def trainer_model_class(model_name_or_path: str) -> TrainerModelClass:
    """Return the auto class every trainer load of this checkpoint builds.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: ``AutoModelForCausalLM``, or ``AutoModelForImageTextToText`` for VL families.
    :rtype: TrainerModelClass
    """
    return model_type_trainer_class(pretrained_model_type(model_name_or_path))


def family_chat_template(model_name_or_path: str) -> str | None:
    """Return the family chat format's template for a tokenizer that ships none.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :return: The Jinja template, or ``None`` when the format has none.
    :rtype: str | None
    """
    # Import cycle: agilerl.llm_envs imports agilerl.utils.llm_utils, which
    # imports this package.
    from agilerl.llm_envs.chat_formats import CHAT_FORMATS

    chat_format = family_runtime(model_name_or_path).chat_format
    if chat_format is None:
        return None
    return CHAT_FORMATS[chat_format].CHAT_TEMPLATE
