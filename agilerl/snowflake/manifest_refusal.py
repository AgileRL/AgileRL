# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Plain-English refusals for a Snowflake app manifest that fails validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from agilerl.models.algorithms.base import LLMAlgorithmSpec
from agilerl.models.manifest import IN_SNOWFLAKE_APP, TrainingManifest
from agilerl.models.registry import MANIFEST_REGISTRY, did_you_mean

NOTHING_STARTED = "Nothing was started."
# What a value of each pydantic type error should be, in words.
EXPECTED_TYPES = {
    "int_type": "a whole number",
    "int_parsing": "a whole number",
    "int_from_float": "a whole number",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "true or false",
    "bool_parsing": "true or false",
    "string_type": "text",
    "list_type": "a list",
    "dict_type": "a mapping of fields",
    "model_type": "a mapping of fields",
    "model_attributes_type": "a mapping of fields",
}


def load_app_manifest(path: Path) -> TrainingManifest:
    """Validate a manifest for the Snowflake app, refusing a bad one in plain words.

    :param path: Manifest YAML.
    :return: The validated manifest.
    :raises ValueError: The algorithm is not one the app trains, or the
        manifest fails validation. The message names each field, what is
        wrong, and what to change.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    refusal = algorithm_refusal(document)
    if refusal is not None:
        msg = f"{refusal}\n{NOTHING_STARTED}"
        raise ValueError(msg)
    try:
        return TrainingManifest.model_validate(
            document, context={IN_SNOWFLAKE_APP: True}
        )
    except ValidationError as exc:
        # The pydantic text is what the consumer must not see, so it is not chained.
        raise ValueError(manifest_refusal(exc, document)) from None


def app_algorithms() -> list[str]:
    """Algorithm names the app trains, sorted.

    :return: The registered LLM algorithm names.
    """
    return [
        name
        for name, spec in MANIFEST_REGISTRY.items()
        if issubclass(spec, LLMAlgorithmSpec)
    ]


def algorithm_refusal(document: object) -> str | None:
    """Why ``algorithm.name`` is not one the app trains.

    :param document: The raw manifest.
    :return: The refusal, or ``None`` when the name is fine or there is no
        algorithm section for validation to report.
    """
    algorithm = document.get("algorithm") if isinstance(document, dict) else None
    if not isinstance(algorithm, dict):
        return None
    names = app_algorithms()
    name = algorithm.get("name")
    if name in names:
        return None
    if name is None:
        problem = "algorithm.name is missing."
    else:
        problem = (
            f"algorithm.name is {shown(name)}, which the app does not train."
            f"{did_you_mean(str(name), names)}"
        )
    return f"{problem} Set it to one of: {', '.join(names)}."


def manifest_refusal(error: ValidationError, document: object) -> str:
    """One refusal for every problem pydantic found, without its formatting.

    :param error: The validation failure.
    :param document: The manifest that failed.
    :return: A line per problem, then ``Nothing was started.``
    """
    lines: list[str] = []
    for details in chosen_errors(error, document):
        line = problem_line(details, field_path(details["loc"], document)[0])
        if line not in lines:
            lines.append(line)
    return "\n".join(["The manifest is not valid.", *lines, NOTHING_STARTED])


def field_path(loc: tuple[int | str, ...], document: object) -> tuple[str, str | None]:
    """Dotted manifest path of an error, without pydantic's union branch names.

    A location entry that is not a key of the document at that point is a
    branch pydantic tried, unless it is the last entry: a missing or extra field.

    :param loc: The error location.
    :param document: The manifest that failed.
    :return: The path, and the first branch name in the location, if any.
    """
    parts: list[str] = []
    branch: str | None = None
    node: Any = document
    for index, entry in enumerate(loc):
        if isinstance(entry, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{entry}]"
            node = node[entry] if isinstance(node, list) and entry < len(node) else None
        elif isinstance(node, dict) and entry in node:
            parts.append(entry)
            node = node[entry]
        elif index == len(loc) - 1:
            parts.append(entry)
        elif branch is None:
            branch = ".".join([*parts, entry])
    return ".".join(parts), branch


def chosen_errors(error: ValidationError, document: object) -> list[ErrorDetails]:
    """The errors of the closest union branch at each field, plus every unbranched error.

    pydantic reports every branch of a union that failed. The branch with the
    fewest errors is the one the manifest meant; the first such branch wins a tie.

    :param error: The validation failure.
    :param document: The manifest that failed.
    :return: The errors, in pydantic's order.
    """
    details = error.errors(include_url=False)
    branches = [field_path(item["loc"], document)[1] for item in details]
    counts: dict[str, int] = {}
    for branch in branches:
        if branch is not None:
            counts[branch] = counts.get(branch, 0) + 1
    closest: dict[str, str] = {}
    for branch, count in counts.items():
        field = branch.rpartition(".")[0]
        if field not in closest or count < counts[closest[field]]:
            closest[field] = branch
    return [
        item
        for item, branch in zip(details, branches, strict=True)
        if branch is None or closest[branch.rpartition(".")[0]] == branch
    ]


def problem_line(details: ErrorDetails, path: str) -> str:
    """One sentence on what is wrong with a field and what to change.

    :param details: One pydantic error.
    :param path: The field's dotted path.
    :return: The sentence.
    """
    kind = details["type"]
    context = details.get("ctx", {})
    field = path or "The manifest"
    if kind == "missing":
        return f"{field} is missing. Add it to the manifest."
    if kind == "extra_forbidden":
        return f"{field} is not a manifest field. Remove it or fix its spelling."
    if kind == "union_tag_invalid":
        tag_path = f"{path}.{context['discriminator'].strip(chr(39))}"
        return (
            f"{tag_path} is {shown(context['tag'])}, which is not one of the "
            f"options. Set it to one of: {context['expected_tags']}."
        )
    if kind in EXPECTED_TYPES:
        return (
            f"{field} is {shown(details['input'])}. Set it to {EXPECTED_TYPES[kind]}."
        )
    message = details["msg"]
    if message.startswith("Input should be "):
        expected = message.removeprefix("Input should be ")
        return f"{field} is {shown(details['input'])}. It should be {expected}."
    sentence = message.removeprefix("Value error, ").removeprefix("Assertion failed, ")
    return f"{path}: {sentence}" if path else sentence


def shown(value: object) -> str:
    """A manifest value as the consumer wrote it, or what kind of value it is.

    :param value: The value pydantic rejected.
    :return: Quoted text, a JSON scalar, or ``a list`` or ``a mapping``.
    """
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, list):
        return "a list"
    return json.dumps(value)
