# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Who sets each manifest field: the user, or Arena.

A field marked :data:`AUTO` is left out of the starting manifest and accepted
when set. A field marked :data:`ADVANCED` is left out and accepted, with a
warning when it holds a value Arena would not choose. A field marked
:data:`HIDDEN` is left out and refused at submit unless it holds the value
Arena would choose. Unmarked fields are the user's.

A marked field the platform writes before it validates (:data:`PLATFORM_WRITTEN_KEY`)
is neither refused nor warned about.
"""

from __future__ import annotations

import copy
import types
from functools import cache
from typing import Annotated, Any, NamedTuple, Union, get_args, get_origin

from pydantic import AliasChoices, BaseModel
from pydantic.fields import FieldInfo

VISIBILITY_KEY = "x-arena-visibility"
PLATFORM_WRITTEN_KEY = "x-arena-platform-written"
AUTO = {VISIBILITY_KEY: "auto"}
ADVANCED = {VISIBILITY_KEY: "advanced"}
HIDDEN = {VISIBILITY_KEY: "hidden"}
ADVANCED_PLATFORM_WRITTEN = {**ADVANCED, PLATFORM_WRITTEN_KEY: True}
HIDDEN_PLATFORM_WRITTEN = {**HIDDEN, PLATFORM_WRITTEN_KEY: True}

MISSING = object()


class MarkedField(NamedTuple):
    """A manifest field marked :data:`AUTO`, :data:`ADVANCED` or :data:`HIDDEN`.

    :param path: Dotted path of payload keys, such as ``training.pop_size``.
    :type path: str
    :param keys: Every key a document may spell each path segment with, the payload key first.
    :type keys: tuple[tuple[str, ...], ...]
    :param tier: ``auto``, ``advanced`` or ``hidden``.
    :type tier: str
    :param platform_written: Whether the platform writes the field before it validates.
    :type platform_written: bool
    """

    path: str
    keys: tuple[tuple[str, ...], ...]
    tier: str
    platform_written: bool


def field_marks(field: FieldInfo) -> tuple[str, bool] | None:
    """The tier a field is marked with and whether the platform writes it, or None for a user field."""
    extra = field.json_schema_extra
    if not isinstance(extra, dict) or VISIBILITY_KEY not in extra:
        return None
    return extra[VISIBILITY_KEY], bool(extra.get(PLATFORM_WRITTEN_KEY))


def field_keys(name: str, field: FieldInfo) -> tuple[str, ...]:
    """The payload key of a field, then every other key it validates from."""
    keys = [field.serialization_alias or field.alias or name, name]
    if isinstance(field.validation_alias, AliasChoices):
        keys += [key for key in field.validation_alias.choices if isinstance(key, str)]
    elif isinstance(field.validation_alias, str):
        keys.append(field.validation_alias)
    return tuple(dict.fromkeys(keys))


def model_types(annotation: object) -> list[type[BaseModel]]:
    """The models an annotation holds directly, through ``Annotated`` and unions, with their subclasses."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return model_types(get_args(annotation)[0])
    if origin in (Union, types.UnionType):
        return [model for arg in get_args(annotation) for model in model_types(arg)]
    if (
        origin is None
        and isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
    ):
        subclasses = annotation.__subclasses__()
        return [
            annotation,
            *(model for sub in subclasses for model in model_types(sub)),
        ]
    return []


@cache
def schema_skipped_fields(model_cls: type[BaseModel]) -> dict[str, FieldInfo]:
    """Fields of *model_cls* that validation accepts and the JSON schema leaves out."""
    published = model_cls.model_json_schema()["properties"]
    return {
        name: field
        for name, field in model_cls.model_fields.items()
        if not field.exclude
        and not any(key in published for key in field_keys(name, field))
    }


@cache
def marked_fields(model_cls: type[BaseModel]) -> tuple[MarkedField, ...]:
    """Every field under *model_cls* marked :data:`AUTO`, :data:`ADVANCED` or :data:`HIDDEN`, in any union member or subclass.

    :param model_cls: The model to walk, usually ``TrainingManifest``.
    :type model_cls: type[BaseModel]
    :returns: One entry per marked path.
    :rtype: tuple[MarkedField, ...]
    """
    found: dict[str, MarkedField] = {}
    for name, field in model_cls.model_fields.items():
        keys = field_keys(name, field)
        marks = field_marks(field)
        if marks is not None:
            found[keys[0]] = MarkedField(keys[0], (keys,), *marks)
            continue
        for nested_cls in model_types(field.annotation):
            for nested in marked_fields(nested_cls):
                path = f"{keys[0]}.{nested.path}"
                found.setdefault(
                    path, nested._replace(path=path, keys=(keys, *nested.keys))
                )
    return tuple(found.values())


def drop_key(document: dict[str, Any], keys: tuple[tuple[str, ...], ...]) -> bool:
    """Remove the key at *keys* from *document*, and any mapping that leaves empty.

    :param document: A manifest document, edited in place.
    :type document: dict[str, Any]
    :param keys: The spellings of each path segment.
    :type keys: tuple[tuple[str, ...], ...]
    :returns: Whether the document held the key.
    :rtype: bool
    """
    spellings, *rest = keys
    key = next((key for key in spellings if key in document), None)
    if key is None:
        return False
    if not rest:
        del document[key]
        return True
    child = document[key]
    if not isinstance(child, dict) or not drop_key(child, tuple(rest)):
        return False
    if not child:
        del document[key]
    return True


def user_view(
    payload: dict[str, Any], fields: tuple[MarkedField, ...]
) -> dict[str, Any]:
    """A copy of *payload* without the marked fields, as a user starts from it.

    :param payload: A manifest payload, or part of one under its section key.
    :type payload: dict[str, Any]
    :param fields: The marked fields, from :func:`marked_fields`.
    :type fields: tuple[MarkedField, ...]
    :returns: The payload without them.
    :rtype: dict[str, Any]
    """
    return without_keys(payload, [field.keys for field in fields])


def path_value(document: dict[str, Any], path: str) -> object:
    """The value at a dotted path of *document*, or :data:`MISSING`."""
    value: object = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def has_key(document: dict[str, Any], keys: tuple[tuple[str, ...], ...]) -> bool:
    """Whether *document* holds the key at *keys*, in any spelling."""
    value: object = document
    for spellings in keys:
        if not isinstance(value, dict):
            return False
        key = next((key for key in spellings if key in value), None)
        if key is None:
            return False
        value = value[key]
    return True


def resolved_dump(
    manifest_cls: type[BaseModel],
    document: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any] | object:
    """*document* validated and dumped as Arena resolves it, or :data:`MISSING` when it is invalid."""
    try:
        manifest = manifest_cls.model_validate(document, context=context)
    except ValueError:
        return MISSING
    return manifest.model_dump(mode="json", by_alias=True)


def without_keys(
    document: dict[str, Any], keys: list[tuple[tuple[str, ...], ...]]
) -> dict[str, Any]:
    """A copy of *document* without each key in *keys*."""
    copied = copy.deepcopy(document)
    for key in keys:
        drop_key(copied, key)
    return copied


def arena_chosen_paths(
    manifest_cls: type[BaseModel],
    document: dict[str, Any],
    manifest: BaseModel,
    context: dict[str, Any] | None = None,
    tier: str = "hidden",
) -> list[str]:
    """Fields of *tier* that *document* sets to a value Arena would not resolve without the key or its block.

    Fields the platform writes are left out.

    :param manifest_cls: The manifest model, usually ``TrainingManifest``.
    :type manifest_cls: type[BaseModel]
    :param document: The raw manifest document.
    :type document: dict[str, Any]
    :param manifest: *document*, validated as *manifest_cls*.
    :type manifest: BaseModel
    :param context: The pydantic validation context *manifest* was validated with.
    :type context: dict[str, Any] | None
    :param tier: ``hidden`` or ``advanced``.
    :type tier: str
    :returns: The dotted paths, in field order.
    :rtype: list[str]
    """
    # Validators fill marked fields, so only the raw document shows what the user set.
    present = [
        field
        for field in marked_fields(manifest_cls)
        if field.tier == tier
        and not field.platform_written
        and has_key(document, field.keys)
    ]
    if not present:
        return []
    resolved = manifest.model_dump(mode="json", by_alias=True)
    # One validation without every set field settles most of them at once.
    baseline = resolved_dump(
        manifest_cls, without_keys(document, [field.keys for field in present]), context
    )
    # Fields in one block share the dump without that block.
    block_dumps: dict[tuple[tuple[str, ...], ...], dict[str, Any] | object] = {}
    chosen = []
    for field in present:
        value = path_value(resolved, field.path)
        if baseline is not MISSING and path_value(baseline, field.path) == value:
            continue
        dump = resolved_dump(
            manifest_cls, without_keys(document, [field.keys]), context
        )
        if dump is not MISSING and path_value(dump, field.path) == value:
            continue
        if len(field.keys) > 1:
            block = field.keys[:-1]
            if block not in block_dumps:
                block_dumps[block] = resolved_dump(
                    manifest_cls, without_keys(document, [block]), context
                )
            dump = block_dumps[block]
            if dump is not MISSING and path_value(dump, field.path) == value:
                continue
        chosen.append(field.path)
    return chosen


__all__ = [
    "ADVANCED",
    "ADVANCED_PLATFORM_WRITTEN",
    "AUTO",
    "HIDDEN",
    "HIDDEN_PLATFORM_WRITTEN",
    "PLATFORM_WRITTEN_KEY",
    "arena_chosen_paths",
    "marked_fields",
    "user_view",
]
