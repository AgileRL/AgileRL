# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install the dependencies an env declares, into the environment training runs in.

``env_packages`` lists what an env needs to import. If the entrypoint is not
importable, those packages are installed here and the env is imported like any
other entrypoint.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from agilerl.utils.env_utils import resolve_entrypoint_target


def package_list(env_packages: Mapping[str, Any]) -> list[str]:
    """Flatten ``{"uv" | "pip": [...] | {"packages": [...]}}`` into requirement strings."""
    packages: list[str] = []
    for field in ("uv", "pip"):
        value = env_packages.get(field)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("packages") or []
        packages.extend(_package_names(value, field))
    for package in packages:
        if package.startswith("-"):
            msg = (
                f"env_packages package {package!r} starts with '-'; "
                "installer flags are not allowed."
            )
            raise ValueError(msg)
    if not packages:
        msg = (
            f"env_packages named no packages to install: {dict(env_packages)!r}. "
            "Expected {'uv': [...]} or {'pip': [...]}."
        )
        raise ValueError(msg)
    return packages


def _package_names(value: object, field: str) -> list[str]:
    """Return named packages from a string, or a sequence of strings."""
    if isinstance(value, str):
        return [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        msg = (
            f"env_packages.{field} must be a package name, a list of package "
            f"names, or {{'packages': [...]}}, got {type(value).__name__}."
        )
        raise TypeError(msg)
    packages: list[str] = []
    for item in value:
        if not isinstance(item, str):
            msg = (
                f"env_packages.{field} must be a sequence of strings, "
                f"got {type(item).__name__}."
            )
            raise TypeError(msg)
        packages.append(item)
    return packages


def ensure_importable(entrypoint: str, env_packages: Mapping[str, Any]) -> None:
    """Install ``env_packages`` into this environment if ``entrypoint`` cannot be imported."""
    try:
        resolve_entrypoint_target(entrypoint)
    except ImportError:
        _install(package_list(env_packages))
        # Site-packages gained a directory that the import system has already
        # listed, so the fresh install is invisible until the caches are dropped.
        importlib.invalidate_caches()


def _install(packages: list[str]) -> None:
    """Install ``packages`` into the running interpreter's environment with uv."""
    uv = shutil.which("uv")
    if uv is None:
        msg = (
            f"This env needs {packages}, and env_packages installs them with uv, "
            "which is not on PATH. Install uv (https://docs.astral.sh/uv/), or "
            "install those packages yourself and drop env_packages."
        )
        raise RuntimeError(msg)
    result = subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--", *packages],
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"uv could not install {packages} alongside this environment's own "
            "dependencies (its output is above). An env whose requirements "
            "conflict with the trainer's needs its own environment; train from "
            "one the packages do resolve in, or host the env and set env_url."
        )
        raise RuntimeError(msg)
