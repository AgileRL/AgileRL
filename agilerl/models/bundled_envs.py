# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run settings Arena gives a manifest that trains on a bundled environment."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from agilerl.models.algorithms.grpo import GRPOSpec
from agilerl.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.models.model_info import (
    SUPPORTED_MODEL_INFO,
    enforces_thinking_budget,
    resolve_reasoning_parser,
)
from agilerl.models.networks import default_rollout_vllm_config
from agilerl.models.registry import MANIFEST_REGISTRY

T = TypeVar("T")


class ContextLength(BaseModel):
    """Context window a bundled environment needs."""

    model_config = ConfigDict(extra="forbid")

    default: int
    # None is segment_prompt_tokens plus one turn: max_output_tokens, twice that
    # with answer_continuation.
    minimum: int | None = None


class BundledRunSettings(BaseModel):
    """Manifest settings a run on one bundled environment gets when it leaves them unset."""

    model_config = ConfigDict(extra="forbid")

    defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Applied only to a GRPO-family algorithm with group_size >= 2.
    group_defaults: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # thinking_token_budget is max_output_tokens less this; unset when that is
    # not positive or the model's vLLM reasoning parser cannot enforce a budget.
    thinking_budget_reserve: int | None = None
    context_length: ContextLength | None = None
    # environment.segment_prompt_tokens when the manifest leaves it unset.
    segment_prompt_tokens: int | None = None
    # The environment sends images, so a catalog model must take vision input.
    needs_vision: bool = False


BUNDLED_RUN_SETTINGS: dict[str, BundledRunSettings] = {
    "AgentWorldModel": BundledRunSettings(
        defaults={"algorithm": {"max_output_tokens": 2048}},
        # list_tools output reaches ~21k tokens.
        context_length=ContextLength(default=32768, minimum=32768),
    ),
    "VisualWebArena": BundledRunSettings(
        defaults={
            "algorithm": {
                "max_output_tokens": 5120,
                # Matches segment_max_images, and keeps video out of vLLM's multimodal profiling.
                "vllm_config": {"limit_mm_per_prompt": {"image": 4, "video": 0}},
            },
            "replay_buffer": {"kind": "llm", "drop_uninformative_groups": True},
        },
        group_defaults={
            # A group whose returns all tie gets the task's stored completion with
            # the highest other return, up to 10 policy versions old.
            "replay_buffer": {"group_replay_max_age": 10},
        },
        # 4096 at the default max_output_tokens. In eval that budget binds on ~1%
        # of turns and caps runaway reasoning.
        thinking_budget_reserve=1024,
        # A segment's prompt plus one turn's generation has to fit the context.
        context_length=ContextLength(default=32768),
        # A page is about 4.7k prompt tokens (accessibility tree 3.6k, screenshot
        # 920 image tokens, instruction 150) and a turn generates 1.6-2.0k tokens.
        # The longest turn, generation plus the next page, is about 6.85k. The
        # context restarts once the next prompt would pass 24000 tokens, and
        # 24000 + 6.85k = 30.85k fits the 32768-token context.
        segment_prompt_tokens=24000,
        needs_vision=True,
    ),
}


def _mapping_section(value: object, name: str) -> dict[str, Any]:
    """Return *value* when it is a mapping, or an empty mapping when it is absent.

    :param value: A manifest section.
    :param name: Section name, for the error.
    :return: The section.
    :raises ValueError: When *value* is set and is not a mapping.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    msg = f"The {name} section must be a mapping, got {type(value).__name__}."
    raise ValueError(msg)


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge *overlay* onto a copy of *base*; nested dicts merge, other values replace.

    :param base: Defaults.
    :param overlay: Caller-supplied fields.
    :return: Merged mapping.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, overlay_value in overlay.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            merged[key] = deep_merge(base_value, overlay_value)
        else:
            merged[key] = overlay_value
    return merged


def lax(kind: type[T], value: object) -> T | None:
    """Return *value* read as *kind* the way pydantic's lax mode reads it, or ``None`` when it is not one.

    :param kind: ``int`` or ``bool``.
    :param value: A raw manifest value, which the fill sees before pydantic checks it.
    :return: The value as *kind*, or ``None``.
    """
    if type(value) is kind:
        return value
    try:
        return TypeAdapter(kind).validate_python(value)
    except ValidationError:
        return None


def is_already_expanded(env: Mapping[str, Any]) -> bool:
    """Return whether *env* already names a rollout source."""
    if env.get("env_image") or env.get("env_url") or env.get("entrypoint"):
        return True
    services = env.get("env_services")
    return isinstance(services, list) and len(services) > 0


def is_spec_family(algorithm: Mapping[str, Any], family: type) -> bool:
    """Return whether *algorithm* names a registered spec that subclasses *family*."""
    name = algorithm.get("name")
    return name in MANIFEST_REGISTRY.names() and issubclass(
        MANIFEST_REGISTRY.get(name), family
    )


def trains_on_groups(algorithm: Mapping[str, Any]) -> bool:
    """Return whether *algorithm* is GRPO-family with group_size >= 2, as group replay needs."""
    if not is_spec_family(algorithm, GRPOSpec):
        return False
    group_size = lax(
        int,
        algorithm.get(
            "group_size",
            MANIFEST_REGISTRY.get(algorithm["name"]).model_fields["group_size"].default,
        ),
    )
    return group_size is not None and group_size > 1


def manifest_model(manifest: Mapping[str, Any]) -> str | None:
    """Return the checkpoint the manifest trains, from the network or the algorithm."""
    network = manifest.get("network")
    algorithm = manifest.get("algorithm")
    network_model = (
        network.get("pretrained_model_name_or_path")
        if isinstance(network, dict)
        else None
    )
    if network_model:
        model = network_model
    elif isinstance(algorithm, dict):
        model = algorithm.get("pretrained_model_name_or_path")
    else:
        return None
    return model if isinstance(model, str) else None


def manifest_model_type(manifest: Mapping[str, Any]) -> str | None:
    """Return the ``model_type`` of the manifest's checkpoint when known offline.

    A supported Hub id reads its bundled config and a local checkpoint its
    ``config.json``. Any other checkpoint, or none, returns ``None``.
    """
    model = manifest_model(manifest)
    if model is None:
        return None
    entry = SUPPORTED_MODEL_INFO.get(model)
    if entry is not None:
        return entry.model_type
    local_config = Path(model) / "config.json"
    if not local_config.is_file():
        return None
    return json.loads(local_config.read_text())["model_type"]


def context_minimum(
    manifest: Mapping[str, Any], settings: BundledRunSettings
) -> tuple[int | None, str]:
    """Return the smallest context a run on the environment can use, and how it is derived.

    :param manifest: Manifest with its run settings filled.
    :param settings: The environment's run settings.
    :return: The minimum, or ``None`` when a turn has no fixed size, and the
        derivation for an error message.
    """
    context = settings.context_length
    if context is None:
        return None, ""
    if context.minimum is not None:
        return context.minimum, ""
    algorithm = manifest.get("algorithm") or {}
    segment = lax(
        int,
        manifest["environment"].get(
            "segment_prompt_tokens", settings.segment_prompt_tokens
        ),
    )
    turn = lax(int, algorithm.get("max_output_tokens"))
    # Without either, a turn has no fixed size to reserve.
    if segment is None or turn is None:
        return None, ""
    basis = f" (segment_prompt_tokens {segment} + max_output_tokens {turn}"
    # The answer continuation generates up to max_output_tokens after the turn.
    if lax(bool, algorithm.get("answer_continuation")):
        turn *= 2
        basis += " x 2 for answer_continuation"
    return segment + turn, basis + ")"


def arena_context_length(value: int) -> int:
    """Return the smallest power of 2 from 1024 that is at least *value*.

    :param value: Tokens the run needs.
    :return: An Arena context length. The caller caps it at the catalog window.
    """
    floor = 1024
    rounded = 1 << (value - 1).bit_length()
    return max(rounded, floor)


def apply_context_length(
    manifest: dict[str, Any], name: str, settings: BundledRunSettings
) -> None:
    """Fill an unset network.max_context_length, then reject a manifest context below the minimum.

    The fill is algorithm.max_model_len when the manifest sets it, else the
    larger of the environment default and the minimum, raised to a power of
    2 from 1024 and capped by the catalog model's context window.

    :param manifest: Manifest with its run settings filled, updated in place.
    :param name: Bundled environment name, for the error message.
    :param settings: The environment's run settings.
    :raises ValueError: If a manifest context field is below the minimum, or
        the catalog model's window is.
    """
    context = settings.context_length
    if context is None:
        return
    algorithm = _mapping_section(manifest.get("algorithm"), "algorithm")
    network = _mapping_section(manifest.get("network"), "network")
    if manifest.get("algorithm") is not algorithm:
        manifest["algorithm"] = algorithm
    if manifest.get("network") is not network:
        manifest["network"] = network
    minimum, basis = context_minimum(manifest, settings)

    # Both set the engine and trainer context; the Arena copies one onto the other.
    context_fields = (("network", "max_context_length"), ("algorithm", "max_model_len"))
    values = {
        f"{section}.{key}": manifest[section][key]
        for section, key in context_fields
        if manifest[section].get(key) is not None
    }
    if not values:
        fill = arena_context_length(max(context.default, minimum or 0))
        model = manifest_model(manifest)
        entry = SUPPORTED_MODEL_INFO.get(model) if model else None
        if entry is not None:
            if minimum is not None and entry.max_context_length < minimum:
                msg = (
                    f"{model} is not supported on {name}: it has "
                    f"{entry.max_context_length} tokens of context and {name} "
                    f"needs at least {minimum}{basis}."
                )
                raise ValueError(msg)
            fill = min(fill, entry.max_context_length)
        network["max_context_length"] = fill
        return
    network.setdefault("max_context_length", algorithm.get("max_model_len"))
    if minimum is None:
        return
    for field, value in values.items():
        # Pydantic reports a value that is not an int.
        tokens = lax(int, value)
        if tokens is not None and tokens < minimum:
            msg = f"{name} needs {field} >= {minimum}{basis}; the manifest sets {value}"
            raise ValueError(msg)


def apply_run_settings(
    manifest: dict[str, Any], name: str, settings: BundledRunSettings
) -> None:
    """Fill unset fields from *settings*, then apply its context length.

    :param manifest: Manifest, updated in place.
    :param name: Bundled environment name, for the error message.
    :param settings: The environment's run settings.
    :raises ValueError: If the catalog model takes no images the environment
        sends, the context is below the environment minimum, or a section the
        settings merge is not a mapping.
    """
    algorithm = _mapping_section(manifest.get("algorithm"), "algorithm")
    model = manifest_model(manifest)
    entry = SUPPORTED_MODEL_INFO.get(model) if model else None
    if settings.needs_vision and entry is not None and "vision" not in entry.modalities:
        msg = f"{model} cannot train on {name}: it takes text only and {name} sends images."
        raise ValueError(msg)
    if manifest.get("algorithm") is not algorithm:
        manifest["algorithm"] = algorithm
    defaults = settings.defaults
    if trains_on_groups(algorithm):
        defaults = deep_merge(defaults, settings.group_defaults)
    if (
        "vllm_config" in defaults.get("algorithm", {})
        and algorithm.get("vllm_config") is None
        and is_spec_family(algorithm, RolloutLLMSpec)
    ):
        # The Arena gives an unset vllm_config the default engine; the settings
        # merge onto it.
        algorithm["vllm_config"] = default_rollout_vllm_config().model_dump(
            exclude_unset=True
        )
    for section, values in defaults.items():
        overlay = (
            algorithm
            if section == "algorithm"
            else _mapping_section(manifest.get(section), section)
        )
        manifest[section] = deep_merge(values, overlay)
    algorithm = manifest.get("algorithm", {})
    max_output_tokens = lax(int, algorithm.get("max_output_tokens"))
    engine_args = algorithm.get("vllm_engine_args")
    if (
        settings.thinking_budget_reserve is not None
        and max_output_tokens is not None
        and "thinking_token_budget" not in algorithm
        and enforces_thinking_budget(
            resolve_reasoning_parser(
                engine_args if isinstance(engine_args, dict) else {},
                manifest_model_type(manifest),
            )
        )
    ):
        budget = max_output_tokens - settings.thinking_budget_reserve
        # A zero budget would turn thinking off.
        if budget > 0:
            algorithm["thinking_token_budget"] = budget
    apply_context_length(manifest, name, settings)


def apply_bundled_run_settings(manifest: dict[str, Any]) -> BundledRunSettings | None:
    """Fill a bundled environment's run settings the manifest leaves unset, whatever its source.

    Filling only what is unset makes a second pass, on the expanded
    environment, change nothing.

    :param manifest: Raw manifest document, updated in place.
    :return: The settings applied, or ``None`` when the environment is not bundled.
    :raises ValueError: If the model or context does not fit the environment.
    """
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        return None
    name = environment.get("name")
    settings = BUNDLED_RUN_SETTINGS.get(name) if isinstance(name, str) else None
    if settings is not None:
        apply_run_settings(manifest, name, settings)
    return settings


def fill_and_expand(
    manifest: dict[str, Any],
    expand_environment: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Fill a bundled environment's run settings, then expand ``environment`` with *expand_environment*.

    :param manifest: Raw manifest document, updated in place.
    :param expand_environment: Returns the expanded copy of an environment section.
    :raises ValueError: If the model or context does not fit the environment.
    """
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        return
    settings = apply_bundled_run_settings(manifest)
    expanded = expand_environment(environment)
    if settings is not None and settings.segment_prompt_tokens is not None:
        expanded.setdefault("segment_prompt_tokens", settings.segment_prompt_tokens)
    if expanded != environment:
        manifest["environment"] = expanded
