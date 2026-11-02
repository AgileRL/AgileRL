# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arena install package extraction."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import click
import pytest

from agilerl.arena.byoc.cluster_install_package import (
    extract_cluster_install_package,
    resolve_install_package_root,
)


def _sample_package_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"clusterToken: tok\n"
        info = tarfile.TarInfo(name="arena-byoc-install/agent-helm-values.yaml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        script = b"#!/usr/bin/env bash\nexit 0\n"
        script_info = tarfile.TarInfo(
            name="arena-byoc-install/arena-byoc-agent/validate.sh"
        )
        script_info.size = len(script)
        script_info.mode = 0o644
        tar.addfile(script_info, io.BytesIO(script))
    return buf.getvalue()


def test_resolve_install_package_root_nested(tmp_path: Path) -> None:
    root = tmp_path / "arena-byoc-install"
    root.mkdir()
    (root / "agent-helm-values.yaml").write_text("clusterToken: tok\n")
    assert resolve_install_package_root(tmp_path) == root


def test_extract_cluster_install_package(tmp_path: Path) -> None:
    root = extract_cluster_install_package(_sample_package_tar(), tmp_path)
    assert (
        (root / "agent-helm-values.yaml")
        .read_text(encoding="utf-8")
        .startswith("clusterToken:")
    )


def test_extract_cluster_install_package_makes_scripts_executable(
    tmp_path: Path,
) -> None:
    root = extract_cluster_install_package(_sample_package_tar(), tmp_path)

    validate = root / "arena-byoc-agent" / "validate.sh"
    assert os.access(validate, os.X_OK)
    if sys.platform != "win32":
        assert subprocess.run([str(validate)], check=False).returncode == 0


def test_resolve_install_package_root_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="agent-helm-values"):
        resolve_install_package_root(tmp_path)
