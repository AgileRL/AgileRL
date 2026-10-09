# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install Cilium as the EKS CNI with kube-proxy replacement."""

from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse

import click

from agilerl.arena.byoc.cluster_helm import helm_available
from agilerl.arena.byoc.provisioning.kubectl import KubectlRun, kubeconfig_env

CILIUM_CHART_VERSION = "1.20.2"
CILIUM_RELEASE = "cilium"
CILIUM_NAMESPACE = "kube-system"
CILIUM_CHART = "cilium"
CILIUM_REPO = "https://helm.cilium.io/"
COREDNS_ADDON = "coredns"
HELM_TIMEOUT = "10m"


def api_server_host(endpoint: str) -> str:
    """Return the hostname Cilium uses to reach the API server."""
    host = urlparse(endpoint).hostname
    if not host:
        msg = f"EKS API endpoint {endpoint!r} has no hostname."
        raise click.ClickException(msg)
    return host


def ensure_cilium(
    kubeconfig_path: Path,
    api_server_host: str,
    *,
    wait: bool = True,
    run: KubectlRun = subprocess.run,
) -> None:
    """Install Cilium with ENI IPAM and kube-proxy replacement.

    :param wait: Wait until the operator is Ready. The pods cannot schedule
        before node groups exist.
    """
    if not helm_available():
        msg = "helm not found on PATH; install Helm 3 before provisioning Cilium."
        raise click.ClickException(msg)
    env = kubeconfig_env(kubeconfig_path)
    argv = [
        "helm",
        "upgrade",
        "--install",
        CILIUM_RELEASE,
        CILIUM_CHART,
        "--repo",
        CILIUM_REPO,
        "--namespace",
        CILIUM_NAMESPACE,
        "--version",
        CILIUM_CHART_VERSION,
        "--set",
        "kubeProxyReplacement=true",
        "--set",
        f"k8sServiceHost={api_server_host}",
        "--set",
        "k8sServicePort=443",
        "--set",
        "eni.enabled=true",
        "--set",
        "ipam.mode=eni",
        "--set",
        "routingMode=native",
        "--set",
        "enableIPv4Masquerade=true",
        "--set",
        "bpf.masquerade=true",
    ]
    if wait:
        argv.extend(["--wait", "--timeout", HELM_TIMEOUT])
    _run(argv, env=env, run=run)


def ensure_coredns_addon(
    cluster_name: str,
    region: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Install the EKS CoreDNS addon. The cluster does not bootstrap it."""
    describe = _run(
        [
            "aws",
            "eks",
            "describe-addon",
            "--cluster-name",
            cluster_name,
            "--addon-name",
            COREDNS_ADDON,
            "--region",
            region,
        ],
        env=None,
        run=run,
        check=False,
    )
    if describe.returncode != 0:
        detail = _detail(describe)
        if "ResourceNotFoundException" not in detail:
            msg = f"aws eks describe-addon failed.{f' {detail}' if detail else ''}"
            raise click.ClickException(msg)
        _run(
            [
                "aws",
                "eks",
                "create-addon",
                "--cluster-name",
                cluster_name,
                "--addon-name",
                COREDNS_ADDON,
                "--region",
                region,
            ],
            env=None,
            run=run,
        )
    _run(
        [
            "aws",
            "eks",
            "wait",
            "addon-active",
            "--cluster-name",
            cluster_name,
            "--addon-name",
            COREDNS_ADDON,
            "--region",
            region,
        ],
        env=None,
        run=run,
    )


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or result.stdout.strip()


def _run(
    argv: list[str],
    *,
    env: dict[str, str] | None,
    run: KubectlRun,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = run(argv, env=env, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        detail = _detail(result)
        rendered = " ".join(argv)
        msg = f"{rendered} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
    return result
