# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema for the training manifest, including form-facing extras."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from typing import Any, cast, get_args

from pydantic import AliasChoices, BaseModel

from agilerl.arena.models.algorithms import (
    AlgoSpec,
    GRPOSpec,
    LLMAlgorithmSpec,
    MultiAgentAlgorithmSpec,
    RolloutLLMSpec,
)
from agilerl.arena.models.env import LLMEnvType
from agilerl.arena.models.manifest import API_VERSION, TrainingManifest, _is_numeric
from agilerl.arena.models.registry import MANIFEST_REGISTRY

SCHEMA_ID = "https://schemas.agilerl.com/training-manifest/v1.json"


def _package_version() -> str:
    """Return the installed agilerl-arena version, or ``0+unknown`` from source."""
    try:
        return distribution_version("agilerl-arena")
    except PackageNotFoundError:
        return "0+unknown"


REF_TEMPLATE = "#/$defs/{model}"


def registered_algorithm_names(
    match: Callable[[type[AlgoSpec]], bool],
) -> tuple[str, ...]:
    """Return registry names whose spec class satisfies *match*."""
    return tuple(
        name for name, spec_cls in MANIFEST_REGISTRY.items() if match(spec_cls)
    )


def algorithm_name_if(algorithm_names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "properties": {
            "algorithm": {
                "properties": {"name": {"enum": list(algorithm_names)}},
                "required": ["name"],
            }
        },
        "required": ["algorithm"],
    }


def training_then(defaults: dict[str, Any]) -> dict[str, Any]:
    return {"properties": {"training": {"properties": defaults}}}


def environment_rollout_type_if() -> dict[str, Any]:
    return {
        "properties": {
            "environment": {
                "properties": {"env_type": {"const": "rollout"}},
                "required": ["env_type"],
            }
        },
        "required": ["environment"],
    }


def environment_dataset_identity_if() -> dict[str, Any]:
    return {
        "properties": {
            "environment": {
                "anyOf": [
                    {"required": ["dataset"]},
                    {"required": ["dataset_path"]},
                    {"required": ["hf_dataset_id"]},
                    {"required": ["columns"]},
                    {"required": ["prompt_template"]},
                ]
            }
        },
        "required": ["environment"],
    }


def environment_without_generative_source_if() -> dict[str, Any]:
    return {
        "not": {
            "properties": {
                "environment": {
                    "anyOf": [
                        {"required": ["env_url"]},
                        {"required": ["env_image"]},
                        {"required": ["factory"]},
                        {"required": ["entrypoint"]},
                    ]
                }
            },
            "required": ["environment"],
        }
    }


def dataset_backed_grpo_rollout_if() -> dict[str, Any]:
    def grpo_family(spec: type[AlgoSpec]) -> bool:
        return issubclass(spec, GRPOSpec)

    return {
        "allOf": [
            algorithm_name_if(registered_algorithm_names(grpo_family)),
            environment_rollout_type_if(),
            environment_dataset_identity_if(),
            environment_without_generative_source_if(),
        ]
    }


def training_schema_conditionals() -> list[dict[str, Any]]:
    def epsilon_greedy(spec: type[AlgoSpec]) -> bool:
        return spec.off_policy and "expl_noise" not in spec.model_fields

    def llm_algorithm(spec: type[AlgoSpec]) -> bool:
        return issubclass(spec, LLMAlgorithmSpec)

    def rollout_llm(spec: type[AlgoSpec]) -> bool:
        return issubclass(spec, RolloutLLMSpec)

    def dataset_llm(spec: type[AlgoSpec]) -> bool:
        return spec.env_type == LLMEnvType.DATASET

    def multi_agent(spec: type[AlgoSpec]) -> bool:
        return issubclass(spec, MultiAgentAlgorithmSpec)

    def off_policy(spec: type[AlgoSpec]) -> bool:
        return spec.off_policy

    return [
        {
            "if": algorithm_name_if(registered_algorithm_names(epsilon_greedy)),
            "then": training_then(
                {
                    "eps_start": {"default": 1.0},
                    "eps_end": {"default": 0.01},
                    "eps_decay": {"default": 0.99999},
                }
            ),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(llm_algorithm)),
            "then": training_then({"reporting_interval": {"default": 1}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(llm_algorithm)),
            "then": training_then({"evo_steps": {"default": 10}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(llm_algorithm)),
            "then": training_then({"evaluation_interval": {"default": 10}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(rollout_llm)),
            "then": training_then({"max_steps": {"default": 200}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(dataset_llm)),
            "then": training_then({"num_epochs": {"default": 1}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(dataset_llm)),
            "then": training_then({"max_steps": {"default": None}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(multi_agent)),
            "then": training_then({"sum_scores": {"default": True}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(off_policy)),
            "then": training_then({"experience_sharing": {"default": True}}),
        },
        {
            "if": dataset_backed_grpo_rollout_if(),
            "then": training_then({"num_epochs": {"default": 1}}),
        },
    ]


def attach_training_schema_conditionals(schema: dict[str, Any]) -> None:
    """Append algorithm-conditional training defaults to the root schema."""
    schema["allOf"] = [*(schema.get("allOf") or []), *training_schema_conditionals()]


def algorithm_schema() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``algorithm`` section as a ``oneOf`` discriminated on ``name``.

    The manifest model dispatches on the registry at validation time, which a
    schema cannot express, so the registry is expanded here instead.

    :returns: The section schema and the definitions it references.
    :rtype: tuple[dict[str, Any], dict[str, Any]]
    """
    names_by_class: dict[type[AlgoSpec], list[str]] = {}
    for name, spec_cls in MANIFEST_REGISTRY.items():
        names_by_class.setdefault(spec_cls, []).append(name)

    variants: list[dict[str, Any]] = []
    defs: dict[str, Any] = {}
    for spec_cls, names in names_by_class.items():
        # An algorithm registered under an alias is still one choice; the aliases
        # stay valid input, but a form must not offer the same algorithm twice.
        canonical = spec_cls.__name__.removesuffix("Spec")
        default_name = spec_cls.schema_name or canonical
        accepted = [canonical, *sorted(n for n in names if n != canonical)]
        schema = spec_cls.model_json_schema(ref_template=REF_TEMPLATE)
        defs.update(schema.pop("$defs", {}))
        schema["title"] = canonical
        schema.setdefault("properties", {})["name"] = {
            "type": "string",
            "enum": sorted(set(accepted)),
            "default": default_name,
            "title": "Algorithm",
            "description": f"Selects {default_name}.",
        }
        required = schema.setdefault("required", [])
        if "name" not in required:
            required.insert(0, "name")
        _add_alias_spellings(schema, spec_cls)
        _add_hpo_ranges(schema, spec_cls)
        variants.append(schema)

    variants.sort(key=lambda v: v["title"])
    return {
        "title": "Algorithm",
        "description": "What to train, and the hyperparameters it trains with.",
        "oneOf": variants,
    }, defs


def _models(root: type[BaseModel]) -> dict[str, type[BaseModel]]:
    """Collect every model reachable from *root*, keyed by the name ``$defs`` uses.

    :param root: The model to walk from.
    :type root: type[BaseModel]
    :returns: Model classes by class name.
    :rtype: dict[str, type[BaseModel]]
    """
    found: dict[str, type[BaseModel]] = {}
    queue: list[object] = [root, *(cls for _, cls in MANIFEST_REGISTRY.items())]
    while queue:
        cls = queue.pop()
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            continue
        if cls.__name__ in found:
            continue
        found[cls.__name__] = cls
        for field in cls.model_fields.values():
            queue.extend(_nested_models(field.annotation))
    return found


def _nested_models(
    annotation: object,
) -> list[object]:
    """Return the model classes mentioned anywhere in a field annotation."""
    out: list[object] = []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        out.append(annotation)
    for arg in get_args(annotation):
        out.extend(_nested_models(arg))
    return out


def _add_alias_spellings(schema: dict[str, Any], model: type[BaseModel]) -> None:
    """Declare every spelling a field accepts, not just the first.

    ``AliasChoices`` lets ``memory_size`` stand in for ``max_size`` and
    ``tournament_selection`` for ``selection_strategy``, but pydantic emits only
    the canonical name. Combined with ``additionalProperties: false`` that turns
    an accepted spelling into a schema violation.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for name, field in model.model_fields.items():
        alias = field.validation_alias
        if not isinstance(alias, AliasChoices) or name not in properties:
            continue
        for choice in alias.choices:
            if isinstance(choice, str) and choice != name and choice not in properties:
                # The alias is the same input under a second name. A validator
                # has to accept it; a form must not draw it twice.
                properties[choice] = {
                    **properties[name],
                    "x-ui-alias-of": name,
                }


def _add_hpo_ranges(schema: dict[str, Any], model: type[AlgoSpec]) -> None:
    """Declare the bounds HPO may mutate this algorithm's hyperparameters between.

    :param schema: The variant's schema, modified in place.
    :type schema: dict[str, Any]
    :param model: The spec class the variant was built from.
    :type model: type[AlgoSpec]
    """
    applicable = {
        name: bounds.model_dump()
        for name, bounds in model.hpo_ranges.items()
        if name in model.model_fields
        and _is_numeric(model.model_fields[name].annotation)
    }
    if applicable:
        schema["x-hpo-ranges"] = applicable


def _collapse_optional(node: dict[str, Any]) -> dict[str, Any]:
    """Fold ``anyOf: [T, null]`` into a nullable ``T``.

    An optional field is one input that may be left empty, but pydantic states
    it as a union. Form generators read that as a choice between two types and
    render a type picker in front of every optional field. Folding the null
    branch into the type keeps validation identical — ``minimum`` and friends
    only ever applied to the non-null branch — while leaving one input behind.

    Only a single non-null branch carrying a plain ``type`` can fold this way; a
    genuine union such as ``str | dict | None`` still needs the picker.

    :param node: A JSON Schema node.
    :type node: dict[str, Any]
    :returns: The node, folded when it was a plain optional.
    :rtype: dict[str, Any]
    """
    branches = node.get("anyOf")
    if not isinstance(branches, list):
        return node
    if not any(b == {"type": "null"} for b in branches):
        return node

    rest = [b for b in branches if b != {"type": "null"}]
    if len(rest) != 1 or "type" not in rest[0]:
        return node

    folded = {k: v for k, v in node.items() if k != "anyOf"}
    folded.update(rest[0])
    folded["type"] = [rest[0]["type"], "null"]
    return folded


def _annotate_free_form(node: dict[str, Any]) -> dict[str, Any]:
    """Mark an object with no declared shape for a raw JSON editor."""
    if (
        node.get("type") == "object"
        and not node.get("properties")
        and not isinstance(node.get("additionalProperties"), dict)
    ):
        node["x-ui-widget"] = "json"
    return node


def _relax_discriminated_union(node: dict[str, Any]) -> dict[str, Any]:
    """Turn a discriminated ``oneOf`` into ``anyOf``.

    :param node: A JSON Schema node.
    :type node: dict[str, Any]
    :returns: The node, with an overlapping union relaxed.
    :rtype: dict[str, Any]
    """
    if "discriminator" not in node or "oneOf" not in node:
        return node
    relaxed = {k: v for k, v in node.items() if k != "oneOf"}
    relaxed["anyOf"] = node["oneOf"]
    return relaxed


def _walk(node: object) -> object:
    """Apply the form-facing rewrites to every node in the document."""
    if isinstance(node, list):
        return [_walk(item) for item in node]
    if not isinstance(node, dict):
        return node

    node = {key: _walk(value) for key, value in node.items()}
    node = _relax_discriminated_union(node)
    return _annotate_free_form(_collapse_optional(node))


def manifest_schema() -> dict[str, Any]:
    """Return the JSON Schema for a whole training manifest.

    :returns: A self-contained JSON Schema document.
    :rtype: dict[str, Any]
    """
    schema = TrainingManifest.model_json_schema(ref_template=REF_TEMPLATE)
    algorithm, defs = algorithm_schema()
    schema.setdefault("$defs", {}).update(defs)
    schema["properties"]["algorithm"] = algorithm

    models = _models(TrainingManifest)
    _add_alias_spellings(schema, TrainingManifest)
    for name, definition in schema["$defs"].items():
        if name in models:
            _add_alias_spellings(definition, models[name])

    schema = cast("dict[str, Any]", _walk(schema))
    attach_training_schema_conditionals(schema)
    schema["$id"] = SCHEMA_ID
    schema["x-manifest-version"] = _package_version()
    schema["title"] = "AgileRL training manifest"
    schema["description"] = (
        f"apiVersion {API_VERSION}. Describes a normalized manifest — the "
        "document to_payload() emits. A hand-written "
        "manifest may omit what the models infer across sections "
        "(environment.env_type from the algorithm, network.arch, "
        "replay_buffer.kind), which no schema can express; validate those "
        "through the manifest contract itself."
    )
    return schema
