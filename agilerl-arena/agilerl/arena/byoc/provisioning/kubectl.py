# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run kubectl against a provisioned cluster."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import yaml

KubectlRun = Callable[..., subprocess.CompletedProcess[str]]


def require_kubectl() -> None:
    """Fail early when kubectl is missing."""
    if not shutil.which("kubectl"):
        msg = "kubectl not found on PATH; install kubectl before provisioning Gateway API."
        raise click.ClickException(msg)


def kubeconfig_env(kubeconfig_path: Path) -> dict[str, str]:
    """Return the process environment pointed at *kubeconfig_path*."""
    env = os.environ.copy()
    env["KUBECONFIG"] = str(kubeconfig_path.expanduser().resolve())
    return env


def apply_manifests(
    manifests: list[dict[str, Any]],
    env: dict[str, str],
    run: KubectlRun,
) -> None:
    """Apply manifests in order from stdin."""
    apply = run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml.safe_dump_all(manifests, sort_keys=False),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if apply.returncode != 0:
        msg = f"Could not apply Kubernetes manifests: {apply.stderr.strip()}"
        raise click.ClickException(msg)


def ensure_namespace(namespace: str, env: dict[str, str], run: KubectlRun) -> None:
    """Create a namespace if it does not exist."""
    manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace},
    }
    apply_manifests([manifest], env=env, run=run)
