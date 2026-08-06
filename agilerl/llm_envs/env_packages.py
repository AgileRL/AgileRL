# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install the dependencies an env declares, into the environment training runs in.

``env_packages`` says what an env needs to import. An orchestrator satisfies it
by building a dedicated environment for the env and serving it over ``/ws``;
running in-process there is only one environment to satisfy it in, so the
packages are installed here and the env is imported like any other entrypoint.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def package_list(env_packages: Mapping[str, Any]) -> list[str]:
    """Flatten ``{"uv" | "pip": [...] | {"packages": [...]}}`` into requirement strings."""
    packages: list[str] = []
    for field in ("uv", "pip"):
        value = env_packages.get(field)
        if value is None:
            continue
        if isinstance(value, dict):
            packages.extend(value.get("packages") or [])
        else:
            packages.extend(value)
    if not packages:
        msg = (
            f"env_packages named no packages to install: {dict(env_packages)!r}. "
            "Expected {'uv': [...]} or {'pip': [...]}."
        )
        raise ValueError(msg)
    return [str(package) for package in packages]


def ensure_importable(entrypoint: str, env_packages: Mapping[str, Any]) -> None:
    """Install ``env_packages`` into this environment if ``entrypoint`` cannot be imported."""
    from agilerl.utils.env_utils import resolve_entrypoint_target

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
        [uv, "pip", "install", "--python", sys.executable, *packages],
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"uv could not install {packages} alongside this environment's own "
            "dependencies (its output is above). An env whose requirements "
            "conflict with the trainer's needs a virtualenv, and so a process, of "
            "its own -- which is what env_packages means to an orchestrator: "
            "AgileRL Arena runs this same manifest by installing these onto a "
            "dedicated env host and driving the env over /ws. To run it here "
            "instead, train from an environment the packages do resolve in."
        )
        raise RuntimeError(msg)
