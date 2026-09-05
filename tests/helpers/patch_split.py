# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Patch a name on every split of a historical import-path barrel."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from functools import wraps
from unittest.mock import DEFAULT, MagicMock
from unittest.mock import patch as unittest_patch

ALGO_UTILS_MODULES = (
    "agilerl.utils.algo_utils",
    "agilerl.utils.algo_spaces",
    "agilerl.utils.algo_obs",
    "agilerl.utils.algo_batch",
)
ALGO_UTILS_PREFIX = "agilerl.utils.algo_utils."

LLM_UTILS_MODULES = (
    "agilerl.utils.llm_utils",
    "agilerl.utils.llm_model",
    "agilerl.utils.llm_rollout",
    "agilerl.utils.llm_prompts",
)
LLM_UTILS_PREFIX = "agilerl.utils.llm_utils."

UTILS_MODULES = (
    "agilerl.utils.utils",
    "agilerl.utils.population",
)
UTILS_PREFIX = "agilerl.utils.utils."


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
    """One mock object applied to every module in the split."""
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


class PatchSplit:
    """Context manager / decorator: patch ``name`` on every module in ``modules``."""

    def __init__(
        self,
        modules: Sequence[str],
        name: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.modules = modules
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
        for module in self.modules:
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


def make_split_patch(
    prefix: str,
    modules: Sequence[str],
    unittest_patch_fn: Callable[..., object],
) -> Callable[..., object]:
    """Wrap ``unittest.mock.patch`` so ``prefix.X`` is applied on every split module."""

    def patch(target: object, *args: object, **kwargs: object) -> object:
        if isinstance(target, str) and target.startswith(prefix):
            return PatchSplit(modules, target[len(prefix) :], *args, **kwargs)
        return unittest_patch_fn(target, *args, **kwargs)

    patch.object = unittest_patch_fn.object
    patch.dict = unittest_patch_fn.dict
    patch.multiple = unittest_patch_fn.multiple
    patch.stopall = unittest_patch_fn.stopall
    return patch


def setattr_split(
    monkeypatch: object,
    modules: Sequence[str],
    name: str,
    value: object,
) -> None:
    """Set ``name`` on every module in the split that may look it up."""
    for module in modules:
        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, name, value, raising=False)
