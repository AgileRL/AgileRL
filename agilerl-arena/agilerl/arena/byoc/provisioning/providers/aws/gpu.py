# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install the NVIDIA device plugin so EKS advertises nvidia.com/gpu."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import click

from agilerl.arena.byoc.cluster_helm import helm_available
from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    apply_manifests,
    kubeconfig_env,
    require_kubectl,
)

NVIDIA_DEVICE_PLUGIN_CHART_VERSION = "0.17.1"
NVIDIA_DEVICE_PLUGIN_RELEASE = "nvidia-device-plugin"
NVIDIA_DEVICE_PLUGIN_NAMESPACE = "kube-system"
NVIDIA_DEVICE_PLUGIN_CHART = "nvidia-device-plugin"
NVIDIA_DEVICE_PLUGIN_REPO = "https://nvidia.github.io/k8s-device-plugin"
HELM_TIMEOUT = "10m"
GPU_STARTUP_TAINT_KEY = "startup-taint.cluster-autoscaler.kubernetes.io/nvidia-gpu"
GPU_STARTUP_CONTROLLER = "gpu-startup"
GPU_STARTUP_IMAGE = "public.ecr.aws/docker/library/python:3.12-alpine"
GPU_STARTUP_SCRIPT = """\
import json
import ssl
import time
import urllib.request

KEY = "startup-taint.cluster-autoscaler.kubernetes.io/nvidia-gpu"
TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
CA = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
API = "https://kubernetes.default.svc/api/v1/nodes"


def patch_body(node):
    taints = node.get("spec", {}).get("taints") or []
    if not any(taint.get("key") == KEY for taint in taints):
        return None
    gpu = node.get("status", {}).get("allocatable", {}).get("nvidia.com/gpu", "0")
    if int(gpu) < 1:
        return None
    kept = [taint for taint in taints if taint.get("key") != KEY]
    return {"spec": {"taints": kept}}


def call(method, path, body=None):
    token = open(TOKEN_PATH).read().strip()
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(API + path, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    if body is not None:
        request.add_header("Content-Type", "application/merge-patch+json")
    context = ssl.create_default_context(cafile=CA)
    with urllib.request.urlopen(request, context=context) as response:
        return json.load(response)


def main():
    while True:
        for node in call("GET", "")["items"]:
            body = patch_body(node)
            if body is None:
                continue
            name = node["metadata"]["name"]
            call("PATCH", "/" + name, body)
        time.sleep(2)


main()
"""


def ensure_nvidia_device_plugin(
    kubeconfig_path: Path,
    run: KubectlRun = subprocess.run,
) -> None:
    """Helm-install the NVIDIA device plugin DaemonSet."""
    if not helm_available():
        msg = (
            "helm not found on PATH; install Helm 3 before provisioning AWS GPU "
            "workers."
        )
        raise click.ClickException(msg)
    env = kubeconfig_env(kubeconfig_path)
    argv = [
        "helm",
        "upgrade",
        "--install",
        NVIDIA_DEVICE_PLUGIN_RELEASE,
        NVIDIA_DEVICE_PLUGIN_CHART,
        "--repo",
        NVIDIA_DEVICE_PLUGIN_REPO,
        "--namespace",
        NVIDIA_DEVICE_PLUGIN_NAMESPACE,
        "--version",
        NVIDIA_DEVICE_PLUGIN_CHART_VERSION,
        "--wait",
        "--timeout",
        HELM_TIMEOUT,
    ]
    result = run(argv, env=env, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        rendered = " ".join(argv)
        msg = f"{rendered} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)


def build_gpu_startup_manifests() -> list[dict[str, Any]]:
    """Return the controller that clears the GPU startup taint."""
    metadata = {
        "name": GPU_STARTUP_CONTROLLER,
        "namespace": NVIDIA_DEVICE_PLUGIN_NAMESPACE,
    }
    return [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": metadata,
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": GPU_STARTUP_CONTROLLER},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["nodes"],
                    "verbs": ["get", "list", "patch"],
                }
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": GPU_STARTUP_CONTROLLER},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": GPU_STARTUP_CONTROLLER,
            },
            "subjects": [
                {
                    "kind": "ServiceAccount",
                    "name": GPU_STARTUP_CONTROLLER,
                    "namespace": NVIDIA_DEVICE_PLUGIN_NAMESPACE,
                }
            ],
        },
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata,
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": GPU_STARTUP_CONTROLLER}},
                "template": {
                    "metadata": {"labels": {"app": GPU_STARTUP_CONTROLLER}},
                    "spec": {
                        "serviceAccountName": GPU_STARTUP_CONTROLLER,
                        "containers": [
                            {
                                "name": GPU_STARTUP_CONTROLLER,
                                "image": GPU_STARTUP_IMAGE,
                                "command": ["python", "-c", GPU_STARTUP_SCRIPT],
                            }
                        ],
                    },
                },
            },
        },
    ]


def ensure_gpu_startup_taint(
    kubeconfig_path: Path,
    run: KubectlRun = subprocess.run,
) -> None:
    """Install the controller that removes the GPU startup taint."""
    require_kubectl()
    apply_manifests(
        build_gpu_startup_manifests(),
        env=kubeconfig_env(kubeconfig_path),
        run=run,
    )
