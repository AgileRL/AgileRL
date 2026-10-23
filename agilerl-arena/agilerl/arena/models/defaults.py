# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Fully-defaulted run specs: the single source of manifest defaults.

:func:`default_manifest` builds the minimal document for an algorithm and
validates it, so every default in the output comes from a pydantic field
default or validator. The platform passes org-resolved identity (environment,
dataset, model) in; arena never invents it.
"""

from __future__ import annotations

from typing import Any

from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.manifest import TrainingManifest
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.registry import MANIFEST_REGISTRY


def default_manifest(
    algorithm: str,
    model: str | None = None,
    environment: dict[str, Any] | None = None,
    network: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the fully-defaulted run spec for an algorithm.

    :param algorithm: Registry name, e.g. ``"GRPO"``.
    :type algorithm: str
    :param model: Hub id for LLM runs. Required for LLM algorithms.
    :type model: str | None
    :param environment: Org-resolved environment or dataset identity.
    :type environment: dict[str, Any] | None
    :param network: Network identity (context length, LoRA overrides, arch).
    :type network: dict[str, Any] | None
    :returns: The validated manifest payload.
    :rtype: dict[str, Any]
    :raises ValueError: On an unknown algorithm, a missing model for an LLM
        algorithm, a model for a non-LLM algorithm, or a missing environment.
    """
    try:
        spec_cls = MANIFEST_REGISTRY.get(algorithm)
    except KeyError as err:
        msg = f"Unknown algorithm {algorithm!r}. Registered: {', '.join(MANIFEST_REGISTRY.names())}."
        raise ValueError(msg) from err
    if environment is None:
        msg = (
            f"{algorithm} needs an environment: pass the org-resolved env or "
            "dataset identity, arena never invents it."
        )
        raise ValueError(msg)

    document: dict[str, Any] = {
        "algorithm": {"name": algorithm},
        "environment": environment,
    }
    if issubclass(spec_cls, LLMAlgorithmSpec):
        if model is None:
            msg = f"{algorithm} fine-tunes a model: pass one with --model."
            raise ValueError(msg)
        document["network"] = _llm_network(model, network)
    else:
        if model is not None:
            msg = f"{algorithm} builds its own networks: --model is LLM-only."
            raise ValueError(msg)
        if network is not None:
            document["network"] = network
    return TrainingManifest.get_validated(document)


def _llm_network(model: str, network: dict[str, Any] | None) -> dict[str, Any]:
    """Network section for an LLM default: model identity plus overrides."""
    overrides = dict(network or {})
    override_model = overrides.pop("pretrained_model_name_or_path", model)
    if override_model != model:
        msg = (
            f"--model {model!r} disagrees with network "
            f"pretrained_model_name_or_path {override_model!r}."
        )
        raise ValueError(msg)
    if "max_context_length" not in overrides and model not in SUPPORTED_MODEL_INFO:
        msg = (
            f"Unknown model {model!r}: pass max_context_length in --network, "
            "arena only knows the catalog in `models supported`."
        )
        raise ValueError(msg)
    return {"pretrained_model_name_or_path": model, **overrides}
