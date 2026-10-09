# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Create the GPU RuntimeClass that Arena training workers request."""

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

GPU_RUNTIME_CLASS_NAME = "nvidia"


def build_gpu_runtime_class_manifest() -> dict[str, Any]:
    """Return the RuntimeClass routing GPU pods to the containerd nvidia handler."""
    return {
        "apiVersion": "node.k8s.io/v1",
        "kind": "RuntimeClass",
        "metadata": {"name": GPU_RUNTIME_CLASS_NAME},
        "handler": GPU_RUNTIME_CLASS_NAME,
        "scheduling": {
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "NoSchedule",
                }
            ]
        },
    }


def ensure_gpu_runtime_class(
    kubeconfig_path: Path,
    run: KubectlRun = subprocess.run,
) -> None:
    """Create the GPU RuntimeClass.

    Arena training workers set `runtimeClassName: nvidia`. The RuntimeClass also
    carries the GPU NoSchedule toleration so those pods can land on worker nodes.
    """
    require_kubectl()
    apply_manifests(
        [build_gpu_runtime_class_manifest()],
        env=kubeconfig_env(kubeconfig_path),
        run=run,
    )
