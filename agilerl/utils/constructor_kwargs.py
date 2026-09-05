# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rebuild constructor kwargs from flat attributes or a flat hyperparameter dict."""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass, replace
from functools import wraps
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

T = TypeVar("T")


def annotation_allows_none(annotation: object) -> bool:
    """True when *annotation* is ``T | None`` / ``Optional[T]``."""
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(arg is type(None) for arg in get_args(annotation))
    return False


def dataclass_type(annotation: object) -> type | None:
    """Return the dataclass hidden in an annotation, including ``T | None``."""
    if annotation is inspect.Parameter.empty:
        return None
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(candidates) == 1:
            return dataclass_type(candidates[0])
        return None
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    return None


def _type_hints(obj: object) -> dict[str, Any]:
    """Resolve annotations; empty dict when forward refs cannot be evaluated."""
    try:
        return get_type_hints(obj)
    except (NameError, TypeError, AttributeError):
        return {}


def _callable_for_signature(obj: object) -> object:
    if inspect.isclass(obj):
        return obj.__init__
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return obj
    return type(obj).__init__


def _signature(obj: object) -> inspect.Signature:
    return inspect.signature(_callable_for_signature(obj))


def _hints_for(obj: object) -> dict[str, Any]:
    return _type_hints(_callable_for_signature(obj))


def unpack_dataclass(config: object) -> dict[str, Any]:
    """Map dataclass fields to a dict of ``field.name -> value``."""
    return {item.name: getattr(config, item.name) for item in fields(config)}


def _field_hints(config_cls: type) -> dict[str, Any]:
    return _type_hints(config_cls)


def _build_dataclass(config_cls: type, values: Mapping[str, Any]) -> object:
    """Construct *config_cls* from matching keys, recursing into nested dataclasses."""
    remaining = dict(values)
    kwargs: dict[str, Any] = {}
    hints = _field_hints(config_cls)
    for item in fields(config_cls):
        annotation = hints.get(item.name, item.type)
        inner_cls = dataclass_type(annotation)
        if item.name in remaining and is_dataclass(remaining[item.name]) and not isinstance(
            remaining[item.name],
            type,
        ):
            kwargs[item.name] = remaining.pop(item.name)
            continue
        if inner_cls is not None:
            inner_names = {inner.name for inner in fields(inner_cls)}
            subset = {
                key: remaining.pop(key)
                for key in list(remaining)
                if key in inner_names
            }
            if not subset and annotation_allows_none(annotation):
                if item.name in remaining:
                    kwargs[item.name] = remaining.pop(item.name)
                continue
            kwargs[item.name] = _build_dataclass(inner_cls, subset)
            continue
        if item.name in remaining:
            kwargs[item.name] = remaining.pop(item.name)
    return config_cls(**kwargs)


def constructor_kwargs_from_flat(
    obj: type | Callable[..., Any],
    flat: Mapping[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Map a flat dict onto *obj*'s parameters, grouping dataclass parameters.

    Keys that already hold a dataclass instance for a parameter keep that
    instance. Remaining keys fill parameters and dataclass fields (including
    nested dataclass fields). Unknown keys are dropped.
    """
    remaining = dict(flat)
    hints = _hints_for(obj)
    kwargs: dict[str, Any] = {}

    for name, param in _signature(obj).parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, param.annotation)
        config_cls = dataclass_type(annotation)
        if config_cls is None:
            if name in remaining:
                kwargs[name] = remaining.pop(name)
            continue
        existing = remaining.pop(name, None)
        if is_dataclass(existing) and not isinstance(existing, type):
            kwargs[name] = existing
            continue
        field_names = _flat_field_names(config_cls)
        subset = {
            key: remaining.pop(key) for key in list(remaining) if key in field_names
        }
        if existing is not None:
            cls_name = obj.__name__ if hasattr(obj, "__name__") else type(obj).__name__
            msg = (
                f"{cls_name}.{name} must be a {config_cls.__name__} instance "
                f"or omitted; got {type(existing).__name__}."
            )
            raise TypeError(msg)
        kwargs[name] = _build_dataclass(config_cls, subset)

    if strict and remaining:
        unexpected = next(iter(remaining))
        cls_name = getattr(obj, "__name__", type(obj).__name__)
        msg = f"{cls_name}() got an unexpected keyword argument {unexpected!r}"
        raise TypeError(msg)

    return kwargs


def _flat_field_names(config_cls: type) -> set[str]:
    """Field names of *config_cls* plus nested dataclass field names."""
    return set(_ordered_flat_field_names(config_cls))


def _ordered_flat_field_names(config_cls: type) -> list[str]:
    """Dataclass field names in declaration order, nested grouping dataclasses inlined.

    Optional nested dataclasses (``T | None``) stay a single field name so a
    whole config object can be passed under that name.
    """
    names: list[str] = []
    hints = _field_hints(config_cls)
    for item in fields(config_cls):
        annotation = hints.get(item.name, item.type)
        inner = dataclass_type(annotation)
        if inner is not None and not annotation_allows_none(annotation):
            names.extend(_ordered_flat_field_names(inner))
        else:
            names.append(item.name)
    return names


def constructor_kwargs_from_obj(obj: object) -> dict[str, Any]:
    """Rebuild ``__init__`` kwargs from attributes on *obj* (clone / checkpoint)."""
    cls = _init_class_without_var_params(type(obj))
    hints = _hints_for(cls)
    kwargs: dict[str, Any] = {}
    for name, param in _signature(cls).parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, param.annotation)
        config_cls = dataclass_type(annotation)
        if config_cls is None:
            if hasattr(obj, name):
                kwargs[name] = getattr(obj, name)
            continue
        values = _values_for_dataclass(config_cls, obj)
        kwargs[name] = _build_dataclass(config_cls, values)
    return kwargs


def _values_for_dataclass(config_cls: type, obj: object) -> dict[str, Any]:
    values: dict[str, Any] = {}
    hints = _field_hints(config_cls)
    for item in fields(config_cls):
        inner = dataclass_type(hints.get(item.name, item.type))
        if inner is not None and hasattr(obj, item.name):
            current = getattr(obj, item.name)
            if current is None or (
                is_dataclass(current) and not isinstance(current, type)
            ):
                values[item.name] = current
                continue
        if inner is not None:
            values.update(_values_for_dataclass(inner, obj))
        elif hasattr(obj, item.name):
            values[item.name] = getattr(obj, item.name)
    return values


def with_runtime_wrap(kwargs: dict[str, Any], wrap: bool) -> dict[str, Any]:
    """Set ``wrap`` on a ``runtime`` dataclass, or as a top-level argument."""
    out = dict(kwargs)
    runtime = out.get("runtime")
    if (
        is_dataclass(runtime)
        and not isinstance(runtime, type)
        and hasattr(runtime, "wrap")
    ):
        out["runtime"] = replace(runtime, wrap=wrap)
        return out
    out["wrap"] = wrap
    return out


def from_hparams(cls: type[T], *args: Any, **hparams: Any) -> T:
    """Construct *cls* from positional spaces plus a flat hyperparameter mapping."""
    hints = _hints_for(cls)
    bound: dict[str, Any] = {}
    arg_index = 0
    for name, param in _signature(cls).parameters.items():
        if name == "self":
            continue
        if dataclass_type(hints.get(name, param.annotation)) is not None:
            continue
        if arg_index >= len(args):
            break
        bound[name] = args[arg_index]
        arg_index += 1
    if arg_index != len(args):
        msg = (
            f"{cls.__name__}.from_hparams() takes {arg_index} positional "
            f"argument(s) but {len(args)} were given"
        )
        raise TypeError(msg)
    overlap = set(bound) & set(hparams)
    if overlap:
        msg = f"got multiple values for argument {next(iter(overlap))!r}"
        raise TypeError(msg)
    return cls(**constructor_kwargs_from_flat(cls, {**bound, **hparams}))


def assemble_init_kwargs(
    cls: type,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind positional args, then group leftover keywords into dataclass params.

    Non-dataclass parameters consume positionals first. Any leftover positionals
    fill flattened dataclass fields in declaration order (nested fields inlined).
    """
    hints = _hints_for(cls)
    bound: dict[str, Any] = dict(kwargs)
    remaining = list(args)
    cls_name = getattr(cls, "__name__", type(cls).__name__)
    for name, param in _signature(cls).parameters.items():
        if name == "self":
            continue
        if dataclass_type(hints.get(name, param.annotation)) is not None:
            continue
        if not remaining:
            break
        if name in bound:
            msg = f"{cls_name}() got multiple values for argument {name!r}"
            raise TypeError(msg)
        bound[name] = remaining.pop(0)
    if remaining:
        _bind_leftover_positionals(cls, hints, remaining, bound)
    if remaining:
        taken = len(args) - len(remaining)
        msg = (
            f"{cls_name}() takes {taken} positional argument(s) "
            f"but {len(args)} were given"
        )
        raise TypeError(msg)
    return constructor_kwargs_from_flat(cls, bound, strict=True)


def _bind_leftover_positionals(
    cls: type,
    hints: Mapping[str, Any],
    remaining: list[Any],
    bound: dict[str, Any],
) -> None:
    """Assign leftover positionals to grouped dataclass params or their fields."""
    for name, param in _signature(cls).parameters.items():
        if name == "self" or not remaining:
            continue
        config_cls = dataclass_type(hints.get(name, param.annotation))
        if config_cls is None:
            continue
        first = remaining[0]
        if (
            name not in bound
            and is_dataclass(first)
            and not isinstance(first, type)
            and isinstance(first, config_cls)
        ):
            bound[name] = remaining.pop(0)
            continue
        for field_name in _ordered_flat_field_names(config_cls):
            if not remaining:
                return
            if field_name in bound:
                continue
            bound[field_name] = remaining.pop(0)


def _init_has_var_params(init: object) -> bool:
    """True when *init* accepts ``*args`` or ``**kwargs``."""
    func = inspect.unwrap(init)
    code = getattr(func, "__code__", None)
    if code is None:
        return False
    return bool(code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS))


def _defined_init(cls: type) -> object | None:
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None:
            return init
    return None


def own_init_has_var_params(cls: type) -> bool:
    """True when the ``__init__`` that will run has ``*args`` or ``**kwargs``."""
    init = _defined_init(cls)
    if init is None:
        return False
    return _init_has_var_params(init)


def _init_class_without_var_params(cls: type) -> type:
    """First class in *cls* MRO whose ``__init__`` is not ``*args``/``**kwargs``."""
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        if not _init_has_var_params(init):
            return klass
    return cls


F = TypeVar("F", bound=Callable[..., object])


def accept_flat_kwargs(fn: F) -> F:
    """Map flat kwargs onto *fn*'s dataclass parameters, then call *fn*."""

    @wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        params = list(inspect.signature(fn).parameters)
        if params and params[0] == "self":
            return fn(args[0], **assemble_init_kwargs(fn, args[1:], kwargs))
        return fn(**assemble_init_kwargs(fn, args, kwargs))

    wrapper.__signature__ = inspect.signature(fn)
    return cast("F", wrapper)
