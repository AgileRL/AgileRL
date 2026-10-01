# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema for the training manifest, including form-facing extras."""

from __future__ import annotations

import copy
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
from agilerl.arena.models.algorithms.ppo import PPOSpec, RecurrentPPOSpec
from agilerl.arena.models.algorithms.rainbow_dqn import RainbowDQNSpec
from agilerl.arena.models.algorithms.sft import SFTSpec
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


def _llm_algorithm(spec: type[AlgoSpec]) -> bool:
    return issubclass(spec, LLMAlgorithmSpec)


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


def training_spec_ref(def_name: str) -> dict[str, Any]:
    return {"properties": {"training": {"$ref": f"#/$defs/{def_name}"}}}


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


def environment_dataset_type_if() -> dict[str, Any]:
    return {
        "properties": {
            "environment": {
                "properties": {"env_type": {"const": "dataset"}},
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


def environment_env_image_if() -> dict[str, Any]:
    return {
        "properties": {
            "environment": {
                "anyOf": [{"required": ["env_image"]}],
            }
        },
        "required": ["environment"],
    }


def environment_env_image_then(defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        "properties": {
            "environment": {"properties": defaults},
        }
    }


def environment_then(defaults: dict[str, Any]) -> dict[str, Any]:
    return {"properties": {"environment": {"properties": defaults}}}


def network_encoder_arch_if(archs: tuple[str, ...]) -> dict[str, Any]:
    """Match a classic network whose encoder arch is one of *archs*."""
    arch_enum = {"enum": list(archs)}
    return {
        "properties": {
            "network": {
                "anyOf": [
                    {"properties": {"arch": arch_enum}, "required": ["arch"]},
                    {
                        "properties": {
                            "encoder_config": {
                                "properties": {"arch": arch_enum},
                                "required": ["arch"],
                            }
                        },
                        "required": ["encoder_config"],
                    },
                ]
            }
        },
        "required": ["network"],
    }


def async_llm_rollout_if() -> dict[str, Any]:
    return {
        "allOf": [
            algorithm_name_if(registered_algorithm_names(_llm_algorithm)),
            {
                "properties": {
                    "training": {
                        "properties": {"rollout_mode": {"const": "async"}},
                        "required": ["rollout_mode"],
                    }
                },
                "required": ["training"],
            },
        ]
    }


def training_schema_conditionals() -> list[dict[str, Any]]:
    def epsilon_greedy(spec: type[AlgoSpec]) -> bool:
        return (
            spec.off_policy
            and "expl_noise" not in spec.model_fields
            and spec is not RainbowDQNSpec
        )

    def rollout_llm(spec: type[AlgoSpec]) -> bool:
        return issubclass(spec, RolloutLLMSpec)

    def dataset_llm(spec: type[AlgoSpec]) -> bool:
        return spec.env_type == LLMEnvType.DATASET

    def multi_agent(spec: type[AlgoSpec]) -> bool:
        return issubclass(spec, MultiAgentAlgorithmSpec)

    def off_policy(spec: type[AlgoSpec]) -> bool:
        return spec.off_policy

    def classic(spec: type[AlgoSpec]) -> bool:
        return not issubclass(spec, LLMAlgorithmSpec)

    classic_names = registered_algorithm_names(classic)
    cnn_or_multiinput = ("cnn", "multiinput")

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
            "if": algorithm_name_if(registered_algorithm_names(_llm_algorithm)),
            "then": training_then({"reporting_interval": {"default": 1}}),
        },
        {
            "if": algorithm_name_if(classic_names),
            "then": training_then({"evo_steps": {"default": 160000}}),
        },
        {
            "if": {
                "allOf": [
                    algorithm_name_if(classic_names),
                    network_encoder_arch_if(cnn_or_multiinput),
                ]
            },
            "then": training_then({"evo_steps": {"default": 320000}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(_llm_algorithm)),
            "then": training_then({"evo_steps": {"default": 20}}),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(_llm_algorithm)),
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
        {
            "if": environment_env_image_if(),
            "then": environment_env_image_then(
                {
                    "env_port": {"default": 8000},
                    "cpus_per_env_host": {"default": 1.0},
                }
            ),
        },
        {
            "if": algorithm_name_if(registered_algorithm_names(_llm_algorithm)),
            "then": training_then(
                {
                    "checkpoint_export": {
                        "default": {"format": "adapter", "trigger": "final"}
                    }
                }
            ),
        },
        {
            "if": async_llm_rollout_if(),
            "then": training_then(
                {"rollout_version_stamp": {"default": "oldest_turn"}}
            ),
        },
        {
            "if": environment_dataset_identity_if(),
            "then": environment_then({"train_test_split": {"default": 0.9}}),
        },
        {
            "if": {
                "allOf": [
                    environment_dataset_identity_if(),
                    environment_rollout_type_if(),
                ]
            },
            "then": environment_then({"rubric_name": {"default": "reward_fn"}}),
        },
        {
            "if": environment_dataset_type_if(),
            "then": environment_then({"response_column": {"default": "response"}}),
        },
        {
            "if": environment_rollout_type_if(),
            "then": environment_then(
                {
                    "num_envs": {"default": 1},
                    "strict_chat_template_boundary": {"default": True},
                }
            ),
        },
    ]


def strip_non_form_algorithm_fields(
    schema: dict[str, Any], spec_cls: type[AlgoSpec]
) -> None:
    """Drop algorithm fields the manifest form must not expose for *spec_cls*."""
    schema["properties"].pop("answer_pattern", None)
    if "answer_pattern" in schema.get("required", []):
        schema["required"].remove("answer_pattern")
    schema.get("x-hpo-ranges", {}).pop("answer_pattern", None)
    if spec_cls is SFTSpec:
        schema["properties"].pop("beta")
        schema["x-hpo-ranges"].pop("beta", None)
    if getattr(spec_cls, "env_type", None) == LLMEnvType.DATASET:
        for field_name in ("answer_continuation",):
            schema["properties"].pop(field_name, None)
            if field_name in schema.get("required", []):
                schema["required"].remove(field_name)
            schema.get("x-hpo-ranges", {}).pop(field_name, None)
    if issubclass(spec_cls, PPOSpec) and spec_cls is not RecurrentPPOSpec:
        for field_name in ("max_seq_len", "bptt_sequence_type"):
            schema["properties"].pop(field_name, None)
            if field_name in schema.get("required", []):
                schema["required"].remove(field_name)
            schema.get("x-hpo-ranges", {}).pop(field_name, None)


def _algorithm_variant_schema(
    spec_cls: type[AlgoSpec], names: list[str]
) -> dict[str, Any]:
    """Build one ``oneOf`` branch for *spec_cls* and its registry aliases."""
    canonical = spec_cls.__name__.removesuffix("Spec")
    default_name = spec_cls.schema_name or canonical
    accepted = [canonical, *sorted(n for n in names if n != canonical)]
    schema = spec_cls.model_json_schema(ref_template=REF_TEMPLATE)
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
    strip_non_form_algorithm_fields(schema, spec_cls)
    _attach_ou_schema_dependencies(schema, spec_cls)
    return schema


def _attach_ou_schema_dependencies(
    schema: dict[str, Any], spec_cls: type[AlgoSpec]
) -> None:
    """Gate OU theta / dt / mean_noise on O_U_noise in the served schema."""
    if "O_U_noise" not in spec_cls.model_fields:
        return
    schema.setdefault("allOf", []).append(
        {
            "if": {
                "properties": {"O_U_noise": {"const": True}},
                "required": ["O_U_noise"],
            },
            "then": {
                "properties": {
                    "theta": {"default": spec_cls.model_fields["theta"].default},
                    "dt": {"default": spec_cls.model_fields["dt"].default},
                    "mean_noise": {
                        "default": spec_cls.model_fields["mean_noise"].default
                    },
                }
            },
        }
    )


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
        schema = _algorithm_variant_schema(spec_cls, names)
        variant_defs = schema.pop("$defs", {})
        defs.update(variant_defs)
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


LLM_ENV_SCHEMA_DEFAULT_FIELDS = (
    "train_test_split",
    "response_column",
    "rubric_name",
    "strict_chat_template_boundary",
    "num_envs",
    "action_field",
)

LLM_ONLY_TRAINING_SCHEMA_FIELDS = ("training_gpus_per_agent",)


def attach_training_schema(schema: dict[str, Any]) -> None:
    """Split classic vs LLM training defs and attach algorithm training defaults."""
    training_def = schema["$defs"]["TrainingSpec"]
    schema["$defs"]["TrainingSpecLLM"] = copy.deepcopy(training_def)
    schema["$defs"]["TrainingSpecOnPolicy"] = copy.deepcopy(training_def)
    for name in LLM_ONLY_TRAINING_SCHEMA_FIELDS:
        training_def["properties"].pop(name, None)
        schema["$defs"]["TrainingSpecOnPolicy"]["properties"].pop(name, None)
    schema["$defs"]["TrainingSpecOnPolicy"]["properties"].pop("learning_delay", None)

    training_section = schema["properties"]["training"]
    training_description = training_section.get("description")
    schema["properties"]["training"] = {"$ref": "#/$defs/TrainingSpecLLM"}
    if training_description is not None:
        schema["properties"]["training"]["description"] = training_description

    def off_policy(spec: type[AlgoSpec]) -> bool:
        return spec.off_policy

    schema["allOf"] = [
        {
            "if": algorithm_name_if(registered_algorithm_names(_llm_algorithm)),
            "then": training_spec_ref("TrainingSpecLLM"),
            "else": {
                "if": algorithm_name_if(registered_algorithm_names(off_policy)),
                "then": training_spec_ref("TrainingSpec"),
                "else": training_spec_ref("TrainingSpecOnPolicy"),
            },
        },
        *(schema.get("allOf") or []),
        *training_schema_conditionals(),
    ]


def _strip_llm_env_schema_defaults(schema: dict[str, Any]) -> None:
    """Null is not a property default. Root conditionals set these fields."""
    properties = schema["$defs"]["LLMEnvSpec"]["properties"]
    for name in LLM_ENV_SCHEMA_DEFAULT_FIELDS:
        properties[name].pop("default", None)


LORA_CONFIG_REF = "#/$defs/LoraConfigDict"


def _is_optional_lora_ref(node: dict[str, Any]) -> bool:
    if node.get("$ref") == LORA_CONFIG_REF:
        return True
    branches = node.get("anyOf")
    if not isinstance(branches, list):
        return False
    refs = [
        branch
        for branch in branches
        if isinstance(branch, dict) and branch.get("$ref") == LORA_CONFIG_REF
    ]
    nulls = [branch for branch in branches if branch == {"type": "null"}]
    return len(refs) == 1 and len(nulls) >= 1


def _inlined_optional_lora(
    node: dict[str, Any], lora_def: dict[str, Any]
) -> dict[str, Any]:
    """Copy LoRA properties onto the field so the form does not have to follow $ref."""
    inlined = {
        key: value for key, value in node.items() if key not in {"anyOf", "$ref"}
    }
    inlined["type"] = ["object", "null"]
    inlined["properties"] = copy.deepcopy(lora_def["properties"])
    if "additionalProperties" in lora_def:
        inlined["additionalProperties"] = lora_def["additionalProperties"]
    required = lora_def.get("required")
    if required:
        inlined["required"] = list(required)
    return inlined


def _inline_lora_config_inplace(node: object, lora_def: dict[str, Any]) -> None:
    if isinstance(node, list):
        for item in node:
            _inline_lora_config_inplace(item, lora_def)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        field = properties.get("lora_config")
        if isinstance(field, dict) and _is_optional_lora_ref(field):
            properties["lora_config"] = _inlined_optional_lora(field, lora_def)
    for value in node.values():
        _inline_lora_config_inplace(value, lora_def)


def _publish_lora_config_schema(schema: dict[str, Any]) -> None:
    """Put LoRA rank / alpha / dropout on the served object the wizard reads."""
    lora_def = schema["$defs"]["LoraConfigDict"]
    lora_r = lora_def["properties"]["lora_r"]
    lora_r["default"] = 1
    lora_r["title"] = "Rank"
    _inline_lora_config_inplace(schema, lora_def)


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
    _strip_llm_env_schema_defaults(schema)
    attach_training_schema(schema)
    _publish_lora_config_schema(schema)
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
