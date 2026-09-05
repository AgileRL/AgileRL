# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Patch a name on every algorithms.core module that may look it up."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps
from unittest.mock import DEFAULT, MagicMock
from unittest.mock import patch as unittest_patch

CORE_MODULES = (
    "agilerl.algorithms.core.base",
    "agilerl.algorithms.core.evolvable_algorithm",
    "agilerl.algorithms.core.evolvable_checkpoint",
    "agilerl.algorithms.core.evolvable_helpers",
    "agilerl.algorithms.core.llm_algorithm",
    "agilerl.algorithms.core.llm_checkpoint",
    "agilerl.algorithms.core.llm_actors",
    "agilerl.algorithms.core.llm_forward",
    "agilerl.algorithms.core.llm_vllm",
    "agilerl.algorithms.core.llm_init",
    "agilerl.algorithms.core.multi_agent",
)

BASE_PREFIX = "agilerl.algorithms.core.base."


def _has_dotted_parent(module_name: str, dotted: str) -> bool:
    """True when every parent of ``dotted`` exists on the imported module."""
    module = importlib.import_module(module_name)
    current: object = module
    for part in dotted.split(".")[:-1]:
        current = getattr(current, part, None)
        if current is None:
            return False
    return True


def _shared_new(new: object, kwargs: dict[str, object]) -> object:
    """One mock object applied to every core module."""
    if new is not DEFAULT:
        return new
    new_callable = kwargs.pop("new_callable", None)
    mock_kwargs: dict[str, object] = {}
    for key in ("spec", "spec_set", "wraps"):
        if key in kwargs:
            mock_kwargs[key] = kwargs.pop(key)
    kwargs.pop("autospec", None)
    kwargs.pop("unsafe", None)
    if callable(new_callable):
        mock = new_callable()
    else:
        mock = MagicMock(**mock_kwargs)
    if "return_value" in kwargs:
        mock.return_value = kwargs.pop("return_value")
    if "side_effect" in kwargs:
        mock.side_effect = kwargs.pop("side_effect")
    return mock


class PatchCore:
    """Context manager / decorator: patch ``name`` on every algorithms.core module."""

    def __init__(self, name: str, *args: object, **kwargs: object) -> None:
        self.name = name
        self._kwargs: dict[str, object] = dict(kwargs)
        self._kwargs.setdefault("create", True)
        self._new = args[0] if args else self._kwargs.pop("new", DEFAULT)
        self._stack: ExitStack | None = None
        self._mock: object | None = None

    def _enter(self) -> object:
        kwargs = dict(self._kwargs)
        create = bool(kwargs.pop("create", True))
        shared = _shared_new(self._new, kwargs)
        stack = ExitStack()
        stack.__enter__()
        for module in CORE_MODULES:
            if not _has_dotted_parent(module, self.name):
                continue
            stack.enter_context(
                unittest_patch(f"{module}.{self.name}", new=shared, create=create)
            )
        self._stack = stack
        self._mock = shared
        return shared

    def __enter__(self) -> object:
        return self._enter()

    def __exit__(self, *exc: object) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def start(self) -> object:
        return self._enter()

    def stop(self) -> None:
        self.__exit__(None, None, None)

    def __call__(self, fn: Callable[..., object]) -> Callable[..., object]:
        @wraps(fn)
        def wrapped(*args: object, **kwargs: object) -> object:
            with self as mock:
                return fn(*args, mock, **kwargs)

        return wrapped


def patch_core(name: str, *args: object, **kwargs: object) -> PatchCore:
    """Patch ``name`` on every split algorithms.core module."""
    return PatchCore(name, *args, **kwargs)


def make_core_patch(unittest_patch_fn: Callable[..., object]) -> Callable[..., object]:
    """Wrap ``unittest.mock.patch`` so ``algorithms.core.base.X`` is applied on every core module."""

    def patch(target: object, *args: object, **kwargs: object) -> object:
        if isinstance(target, str) and target.startswith(BASE_PREFIX):
            return PatchCore(target[len(BASE_PREFIX) :], *args, **kwargs)
        return unittest_patch_fn(target, *args, **kwargs)

    patch.object = unittest_patch_fn.object
    patch.dict = unittest_patch_fn.dict
    patch.multiple = unittest_patch_fn.multiple
    patch.stopall = unittest_patch_fn.stopall
    return patch


def setattr_core(monkeypatch: object, name: str, value: object) -> None:
    """Set ``name`` on every algorithms.core module that may look it up."""
    for module in CORE_MODULES:
        monkeypatch.setattr(f"{module}.{name}", value, raising=False)
