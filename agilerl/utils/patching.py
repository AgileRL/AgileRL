# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Primitives shared by every class-level patch module.

This sits below :mod:`agilerl.architectures` so the generic and per-family
patches can share resolution and idempotence without importing each other.
"""

from __future__ import annotations

import functools
import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import ParamSpec, TypeVar

from packaging.specifiers import SpecifierSet

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

__all__ = [
    "LIBRARY_SHIMS",
    "LibraryShim",
    "UnsupportedLibraryVersionError",
    "class_is_patched",
    "library_shim",
    "try_import",
]


def try_import(module_path: str) -> ModuleType | None:
    """Import *module_path*, or None when the package cannot be loaded.

    :param module_path: Importable module path.
    :type module_path: str
    :return: The imported module, or None.
    :rtype: ModuleType | None
    """
    try:
        return importlib.import_module(module_path)
    except Exception:
        logger.debug("[patches] %s could not be imported", module_path)
        return None


def class_is_patched(cls: type, flag: str) -> bool:
    """Whether *cls* itself carries *flag*, ignoring any inherited value.

    :param cls: Class to inspect.
    :type cls: type
    :param flag: Marker attribute name.
    :type flag: str
    :return: True when this exact class has already been patched.
    :rtype: bool
    """
    return bool(vars(cls).get(flag, False))


class UnsupportedLibraryVersionError(RuntimeError):
    """The installed release of a library is outside the range a shim was checked against."""


@dataclass(frozen=True)
class LibraryShim:
    """A workaround for one library, checked against the releases in ``versions``."""

    name: str
    package: str
    versions: SpecifierSet

    def installed_version(self) -> str | None:
        """Return the installed release of ``package``, or None when it is not installed."""
        try:
            return version(self.package)
        except PackageNotFoundError:
            return None

    def covers_installed(self) -> bool:
        """Return whether the installed release is in ``versions``; True when ``package`` is absent."""
        installed = self.installed_version()
        return installed is None or self.versions.contains(installed, prereleases=True)

    def check(self) -> None:
        """Raise when ``package`` is installed at a release outside ``versions``.

        :raises UnsupportedLibraryVersionError: If the installed release is out of range.
        """
        if self.covers_installed():
            return
        installed = self.installed_version()
        msg = (
            f"{self.name} was checked against {self.package}{self.versions}, but "
            f"{self.package} {installed} is installed. Re-check it on {installed}, "
            "then widen its range or delete it."
        )
        raise UnsupportedLibraryVersionError(msg)


LIBRARY_SHIMS: list[LibraryShim] = []


def library_shim(
    package: str, versions: str
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register a workaround for ``package`` and refuse to run it outside ``versions``.

    :param package: Distribution name, e.g. ``"vllm"``.
    :type package: str
    :param versions: PEP 440 specifier the shim was checked against, e.g. ``"==0.32.*"``.
    :type versions: str
    :return: Decorator for the shim function.
    :rtype: Callable[[Callable[P, R]], Callable[P, R]]
    """

    def decorate(shim_fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(shim_fn)
        def checked(*args: P.args, **kwargs: P.kwargs) -> R:
            shim.check()
            return shim_fn(*args, **kwargs)

        shim = LibraryShim(
            name=f"{checked.__module__}.{checked.__qualname__}",
            package=package,
            versions=SpecifierSet(versions),
        )
        LIBRARY_SHIMS.append(shim)
        return checked

    return decorate
