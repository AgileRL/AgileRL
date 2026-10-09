# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install the cluster autoscaler for EKS worker node groups."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from agilerl.arena.byoc.cluster_helm import helm_available
from agilerl.arena.byoc.provisioning.kubectl import KubectlRun, kubeconfig_env

CLUSTER_AUTOSCALER_CHART_VERSION = "9.59.0"
CLUSTER_AUTOSCALER_RELEASE = "cluster-autoscaler"
CLUSTER_AUTOSCALER_NAMESPACE = "kube-system"
CLUSTER_AUTOSCALER_CHART = "cluster-autoscaler"
CLUSTER_AUTOSCALER_REPO = "https://kubernetes.github.io/autoscaler"
CLUSTER_AUTOSCALER_SERVICE_ACCOUNT = "cluster-autoscaler"
HELM_TIMEOUT = "10m"


def ensure_cluster_autoscaler(
    kubeconfig_path: Path,
    cluster_name: str,
    region: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Helm-install the cluster autoscaler for tagged worker groups."""
    if not helm_available():
        msg = "helm not found on PATH; install Helm 3 before provisioning the cluster autoscaler."
        raise click.ClickException(msg)
    env = kubeconfig_env(kubeconfig_path)
    argv = [
        "helm",
        "upgrade",
        "--install",
        CLUSTER_AUTOSCALER_RELEASE,
        CLUSTER_AUTOSCALER_CHART,
        "--repo",
        CLUSTER_AUTOSCALER_REPO,
        "--namespace",
        CLUSTER_AUTOSCALER_NAMESPACE,
        "--version",
        CLUSTER_AUTOSCALER_CHART_VERSION,
        "--set",
        f"autoDiscovery.clusterName={cluster_name}",
        "--set",
        f"awsRegion={region}",
        "--set",
        f"rbac.serviceAccount.name={CLUSTER_AUTOSCALER_SERVICE_ACCOUNT}",
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
