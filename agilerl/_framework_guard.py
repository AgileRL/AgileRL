# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Explain imports agilerl 3 does not ship: ``agilerl`` 2.x framework modules and agilerl-arena's ``agilerl.arena``."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec, PathFinder
from types import ModuleType

# Top-level agilerl 2.x framework packages that are not agilerl 3 modules.
FRAMEWORK_PACKAGES = frozenset(
    {
        "algorithms",
        "architectures",
        "builders",
        "components",
        "data",
        "distributed",
        "hpo",
        "llm_envs",
        "logger",
        "lora",
        "metrics",
        "modules",
        "net_configs",
        "networks",
        "population",
        "protocols",
        "rollouts",
        "strategies",
        "train",
        "training",
        "vector",
        "wrappers",
    }
)


def pin_message(missing: str) -> str:
    """Append the ``agilerl<3`` pointer to an import failure message.

    :param missing: What failed to import, e.g. ``No module named 'agilerl.algorithms'``.
    :return: ``missing`` followed by the pin instructions.
    """
    return (
        f"{missing}. agilerl 3 is the Arena CLI and SDK; the AgileRL RL framework "
        "is agilerl 2.x. To keep using it, install agilerl<3 in a separate "
        "virtualenv: python -m venv .venv-agilerl2 && "
        ".venv-agilerl2/bin/pip install 'agilerl<3'. To downgrade this environment "
        "instead: pip install 'agilerl<3' && "
        "pip install --force-reinstall --no-deps 'agilerl-arena<2'"
    )


def rename_message(missing: str, fullname: str) -> str:
    """Append the agilerl-arena rename to an import failure under ``agilerl.arena``.

    :param missing: What failed to import, e.g. ``No module named 'agilerl.arena.client'``.
    :param fullname: The ``agilerl.arena`` module path that failed.
    :return: ``missing`` followed by the agilerl 3 module to import instead.
    """
    renamed = "agilerl" + fullname.removeprefix("agilerl.arena")
    return (
        f"{missing}. The SDK is now the agilerl package (formerly agilerl-arena): "
        f"imports drop '.arena', so import {renamed!r} instead of {fullname!r}, "
        "e.g. from agilerl import ArenaClient"
    )


def missing_attribute_error(module: ModuleType, name: str) -> AttributeError:
    """The error for an agilerl 2.x name that agilerl 3 ``module`` does not define.

    :param module: The agilerl 3 module.
    :param name: Attribute looked up on it.
    :return: An AttributeError naming the ``agilerl<3`` pin, so ``hasattr`` is False.
    """
    return AttributeError(
        pin_message(f"module {module.__name__!r} has no attribute {name!r}"),
        name=name,
        obj=module,
    )


def is_framework_module(fullname: str) -> bool:
    """Whether ``fullname`` is an agilerl 2.x framework module path absent from agilerl 3.

    :param fullname: Dotted module name.
    :return: True for ``agilerl.<framework package>[.*]`` and ``agilerl.utils.*``.
    """
    parts = fullname.split(".")
    if parts[0] != "agilerl" or len(parts) < 2:
        return False
    return parts[1] in FRAMEWORK_PACKAGES or (parts[1] == "utils" and len(parts) > 2)


def is_renamed_module(fullname: str) -> bool:
    """Whether ``fullname`` is an agilerl-arena ``agilerl.arena`` path, which agilerl 3 renamed.

    :param fullname: Dotted module name.
    :return: True for ``agilerl.arena`` and ``agilerl.arena.*``.
    """
    return fullname == "agilerl.arena" or fullname.startswith("agilerl.arena.")


class FrameworkImportGuard:
    """Refuse a 2.x framework import or an ``agilerl.arena`` import."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        """Raise for a 2.x framework module or an ``agilerl.arena`` path.

        :param fullname: Dotted module name being imported.
        :param path: Parent package ``__path__``.
        :param target: Module being reloaded, if any.
        :return: None so a later finder can load the module.
        :raises ModuleNotFoundError: ``fullname`` is a 2.x framework module that
            is not on ``path`` (named ``agilerl``), or an ``agilerl.arena`` path.
        """
        if is_renamed_module(fullname):
            message = rename_message(f"No module named {fullname!r}", fullname)
            raise ModuleNotFoundError(message, name=fullname)
        if is_framework_module(fullname):
            if PathFinder.find_spec(fullname, path, target) is not None:
                return None
            message = pin_message(f"No module named {fullname!r}")
            # name="agilerl" (the missing 2.x distribution): importlib drops a
            # ModuleNotFoundError named after the submodule in ``from agilerl import x``.
            raise ModuleNotFoundError(message, name="agilerl")
        return None


def install() -> None:
    """Install the guard once, ahead of PathFinder."""
    if any(isinstance(finder, FrameworkImportGuard) for finder in sys.meta_path):
        return
    guard = FrameworkImportGuard()
    # A leftover agilerl-arena 1.x tree sits on agilerl.__path__. PathFinder would load it.
    for index, finder in enumerate(sys.meta_path):
        if finder is PathFinder:
            sys.meta_path.insert(index, guard)
            return
    sys.meta_path.append(guard)
