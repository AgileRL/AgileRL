# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Download and extract Arena enterprise cluster Helm install packages."""

from __future__ import annotations

import io
import stat
import tarfile
from pathlib import Path

import click

INSTALL_PACKAGE_DIR = "arena-byoc-install"


def resolve_install_package_root(extract_dir: Path) -> Path:
    """Return the directory containing Helm values and chart trees.

    :param extract_dir: Directory passed to :func:`tarfile.TarFile.extractall`.
    :type extract_dir: Path
    :returns: Install package root (``arena-byoc-install`` or *extract_dir*).
    :rtype: Path
    :raises click.ClickException: If required files are missing.
    """
    nested = extract_dir / INSTALL_PACKAGE_DIR
    for candidate in (nested, extract_dir):
        if (candidate / "agent-helm-values.yaml").is_file():
            return candidate
    msg = "Install package archive is missing agent-helm-values.yaml."
    raise click.ClickException(msg)


def extract_cluster_install_package(data: bytes, dest_dir: Path) -> Path:
    """Extract an Arena install package tarball and return its root directory.

    :param data: ``application/gzip`` bytes from Arena.
    :type data: bytes
    :param dest_dir: Parent directory for extracted files.
    :type dest_dir: Path
    :returns: Resolved install package root.
    :rtype: Path
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest_dir)
    root = resolve_install_package_root(dest_dir)
    _make_scripts_executable(root)
    return root


def _make_scripts_executable(root: Path) -> None:
    """Add the executable bit to package scripts; Arena packs them mode 0644."""
    executable = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    for script in root.rglob("*.sh"):
        script.chmod(script.stat().st_mode | executable)
