# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""EFS StorageClass so pods can mount the shared file system."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    apply_manifests,
    kubeconfig_env,
    require_kubectl,
)

ARENA_SHARED_STORAGE_CLASS = "arena-shared"


def build_efs_storage_class_manifest(file_system_id: str) -> dict[str, Any]:
    """Return a StorageClass that provisions EFS access points on *file_system_id*."""
    return {
        "apiVersion": "storage.k8s.io/v1",
        "kind": "StorageClass",
        "metadata": {"name": ARENA_SHARED_STORAGE_CLASS},
        "provisioner": "efs.csi.aws.com",
        "parameters": {
            "provisioningMode": "efs-ap",
            "fileSystemId": file_system_id,
            "directoryPerms": "700",
        },
    }


def ensure_efs_storage_class(
    kubeconfig_path: Path,
    file_system_id: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Create the ``arena-shared`` StorageClass for the EFS CSI controller."""
    require_kubectl()
    apply_manifests(
        [build_efs_storage_class_manifest(file_system_id)],
        env=kubeconfig_env(kubeconfig_path),
        run=run,
    )
