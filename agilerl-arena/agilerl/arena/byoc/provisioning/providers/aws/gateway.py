# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Install the AWS Load Balancer Controller and a Cilium Gateway on an NLB."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from agilerl.arena.byoc.cluster_helm import AGENT_NAMESPACE, helm_available
from agilerl.arena.byoc.provisioning.gateway_api import (
    configure_gateway_api,
)
from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    kubeconfig_env,
    require_kubectl,
)

LOAD_BALANCER_CONTROLLER_CHART_VERSION = "3.6.0"
LOAD_BALANCER_CONTROLLER_RELEASE = "aws-load-balancer-controller"
LOAD_BALANCER_CONTROLLER_NAMESPACE = "kube-system"
LOAD_BALANCER_CONTROLLER_SERVICE_ACCOUNT = "aws-load-balancer-controller"
LOAD_BALANCER_CHART_REPO = "https://aws.github.io/eks-charts"
NLB_SERVICE_ANNOTATIONS = {
    "service.beta.kubernetes.io/aws-load-balancer-backend-protocol": "TCP",
    "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type": "instance",
    "service.beta.kubernetes.io/aws-load-balancer-scheme": "internet-facing",
    "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
}
HELM_TIMEOUT = "10m"
GATEWAY_DELETE_TIMEOUT = "10m"


def configure_eks_gateway(
    kubeconfig_path: Path,
    gateway_name: str,
    domain: str,
    cluster_name: str,
    region: str,
    vpc_id: str,
    tls_secret_name: str | None,
    run: KubectlRun = subprocess.run,
) -> list[dict[str, str]]:
    """Install the load balancer controller and a Cilium Gateway on an NLB."""
    env = kubeconfig_env(kubeconfig_path)
    install_aws_load_balancer_controller(
        cluster_name=cluster_name,
        region=region,
        vpc_id=vpc_id,
        env=env,
        run=run,
    )
    return configure_gateway_api(
        kubeconfig_path=kubeconfig_path,
        gateway_name=gateway_name,
        domain=domain,
        tls_secret_name=tls_secret_name,
        infrastructure_annotations=NLB_SERVICE_ANNOTATIONS,
        run=run,
    )


def delete_eks_gateway(
    kubeconfig_path: Path,
    gateway_name: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Delete the Gateway and wait until its NLB is gone."""
    require_kubectl()
    env = kubeconfig_env(kubeconfig_path)
    click.echo(f"Deleting Gateway {gateway_name!r} in namespace {AGENT_NAMESPACE!r}.")
    _run(
        [
            "kubectl",
            "delete",
            "gateway",
            gateway_name,
            "--namespace",
            AGENT_NAMESPACE,
            "--ignore-not-found=true",
            "--wait=true",
            "--cascade=foreground",
            "--timeout",
            GATEWAY_DELETE_TIMEOUT,
        ],
        env=env,
        run=run,
    )
    # The Service finalizer blocks until the NLB and its ENIs are gone.
    _run(
        [
            "kubectl",
            "delete",
            "service",
            f"cilium-gateway-{gateway_name}",
            "--namespace",
            AGENT_NAMESPACE,
            "--ignore-not-found=true",
            "--wait=true",
            "--timeout",
            GATEWAY_DELETE_TIMEOUT,
        ],
        env=env,
        run=run,
    )


def install_aws_load_balancer_controller(
    cluster_name: str,
    region: str,
    vpc_id: str,
    env: dict[str, str],
    run: KubectlRun,
) -> None:
    """Install or upgrade the AWS Load Balancer Controller for NLB Services."""
    if not helm_available():
        msg = "helm not found on PATH; install Helm 3 before provisioning the AWS load balancer."
        raise click.ClickException(msg)
    _run(
        [
            "helm",
            "upgrade",
            "--install",
            LOAD_BALANCER_CONTROLLER_RELEASE,
            LOAD_BALANCER_CONTROLLER_RELEASE,
            "--repo",
            LOAD_BALANCER_CHART_REPO,
            "--namespace",
            LOAD_BALANCER_CONTROLLER_NAMESPACE,
            "--version",
            LOAD_BALANCER_CONTROLLER_CHART_VERSION,
            "--set",
            f"clusterName={cluster_name}",
            "--set",
            f"region={region}",
            "--set",
            f"vpcId={vpc_id}",
            "--set",
            "serviceAccount.create=true",
            "--set",
            f"serviceAccount.name={LOAD_BALANCER_CONTROLLER_SERVICE_ACCOUNT}",
            "--wait",
            "--timeout",
            HELM_TIMEOUT,
        ],
        env=env,
        run=run,
    )


def _run(argv: list[str], env: dict[str, str], run: KubectlRun) -> None:
    result = run(
        argv,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        rendered = " ".join(argv)
        msg = f"{rendered} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
