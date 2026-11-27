# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Resolve the AWS CLI and build EKS kubeconfig commands."""

from __future__ import annotations

import shutil
from pathlib import Path


def resolve_aws_executable() -> str | None:
    """Return the AWS CLI path, or ``None`` when it is not on ``PATH``."""
    return shutil.which("aws")


def build_eks_kubeconfig_argv(
    cluster_name: str,
    region: str,
    kubeconfig_path: Path,
) -> list[str]:
    """Return ``aws eks update-kubeconfig`` arguments."""
    executable = resolve_aws_executable()
    if executable is None:
        msg = "AWS CLI not found."
        raise FileNotFoundError(msg)
    return [
        executable,
        "eks",
        "update-kubeconfig",
        "--name",
        cluster_name,
        "--region",
        region,
        "--kubeconfig",
        str(kubeconfig_path),
    ]
