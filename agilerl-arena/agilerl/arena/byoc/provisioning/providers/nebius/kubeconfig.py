# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Nebius CLI and build kubeconfig retrieval commands."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

NEBIUS_CLI_ENV = "NEBIUS_CLI"
NEBIUS_DEFAULT_CLI = Path("~/.nebius/bin/nebius")


def resolve_kubeconfig_executable(command: list[str]) -> Path | None:
    """Return the kubeconfig command executable, or ``None`` if it cannot be run."""
    if not command:
        return None
    exe = command[0]
    candidate = Path(exe).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_absolute():
        return None

    env_cli = os.environ.get(NEBIUS_CLI_ENV, "").strip()
    if env_cli:
        env_path = Path(env_cli).expanduser()
        if env_path.is_file():
            return env_path

    resolved = shutil.which(exe)
    if resolved:
        return Path(resolved)

    if exe == "nebius":
        default_cli = NEBIUS_DEFAULT_CLI.expanduser()
        if default_cli.is_file():
            return default_cli

    return None


def build_kubeconfig_argv(command: list[str]) -> list[str]:
    """Return *command* with its executable resolved to an absolute path."""
    executable = resolve_kubeconfig_executable(command)
    if executable is None:
        return command
    return [str(executable), *command[1:]]


def build_nebius_kubeconfig_argv(
    cluster_id: str,
    context_name: str,
    kubeconfig_path: Path,
) -> list[str]:
    """Build ``nebius mk8s cluster get-credentials`` for a provisioned cluster."""
    return build_kubeconfig_argv(
        [
            "nebius",
            "mk8s",
            "cluster",
            "get-credentials",
            "--id",
            cluster_id,
            "--external",
            "--context-name",
            context_name,
            "--kubeconfig",
            str(kubeconfig_path),
            # Reruns must refresh the context instead of failing on the existing one.
            "--force",
        ]
    )
