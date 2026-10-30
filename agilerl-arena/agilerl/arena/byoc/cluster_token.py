# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rotate an enterprise cluster token and write it into the agent Secret."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

import click

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_helm import (
    agent_deployment_selector,
    discover_agent_install,
)
from agilerl.arena.byoc.provisioning.kubectl import (
    KubectlRun,
    apply_manifests,
    kubeconfig_env,
    require_kubectl,
)
from agilerl.arena.client import ArenaClient

DEFAULT_CLUSTER_TOKEN_SECRET_NAME = "cluster-token"
CLUSTER_TOKEN_SECRET_KEY = "cluster-token"


def _normalize_token(token: object) -> str | None:
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _secret_from_agent_helm_values_yaml(text: object) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    match = re.search(
        r"^existingClusterTokenSecret:\s*[\"']?([^\"'\s]+)[\"']?\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        return None
    return _normalize_token(match.group(1))


def cluster_token_secret_name(bundle: dict[str, Any]) -> str:
    """Return the Helm cluster-token Secret name from a rotation bundle."""
    helm = bundle.get("agentHelmValues")
    if isinstance(helm, dict):
        from_dict = _normalize_token(helm.get("existingClusterTokenSecret"))
        if from_dict is not None:
            return from_dict
    install_bundle = bundle.get("install_bundle")
    if not isinstance(install_bundle, dict):
        install_bundle = bundle.get("installBundle")
    if isinstance(install_bundle, dict):
        yaml_text = install_bundle.get("agent_helm_values_yaml")
        if yaml_text is None:
            yaml_text = install_bundle.get("agentHelmValuesYaml")
        from_yaml = _secret_from_agent_helm_values_yaml(yaml_text)
        if from_yaml is not None:
            return from_yaml
    return DEFAULT_CLUSTER_TOKEN_SECRET_NAME


def build_cluster_token_secret_manifest(
    secret_name: str,
    namespace: str,
    token: str,
) -> dict[str, Any]:
    """Return the Opaque Secret that holds the agent cluster token."""
    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": secret_name, "namespace": namespace},
        "type": "Opaque",
        "stringData": {CLUSTER_TOKEN_SECRET_KEY: token},
    }


def resolve_cluster_kubeconfig(name: str, kubeconfig: Path | None) -> Path | None:
    """Return an explicit kubeconfig, the provisioned default, or ``KUBECONFIG``."""
    if kubeconfig is not None:
        path = kubeconfig.expanduser().resolve()
        if not path.is_file():
            msg = f"Kubeconfig is not a file: {path}"
            raise click.ClickException(msg)
        return path
    default = Path(f"arena-cluster-{name}") / "kubeconfig"
    if default.is_file():
        return default.resolve()
    env = os.environ.get("KUBECONFIG", "").strip()
    if env:
        env_path = Path(env.split(os.pathsep, 1)[0]).expanduser()
        if env_path.is_file():
            return env_path.resolve()
    home_kube = Path.home() / ".kube" / "config"
    if home_kube.is_file():
        return home_kube.resolve()
    return None


def apply_cluster_token_secret(
    token: str,
    secret_name: str,
    namespace: str,
    kubeconfig_path: Path | None,
    agent_release: str,
    run: KubectlRun = subprocess.run,
) -> None:
    """Create or replace the cluster-token Secret and restart the agent."""
    require_kubectl()
    env = (
        kubeconfig_env(kubeconfig_path)
        if kubeconfig_path is not None
        else os.environ.copy()
    )
    apply_manifests(
        [
            build_cluster_token_secret_manifest(
                secret_name=secret_name,
                namespace=namespace,
                token=token,
            )
        ],
        env=env,
        run=run,
    )
    restart = run(
        [
            "kubectl",
            "rollout",
            "restart",
            "deployment",
            "-n",
            namespace,
            "-l",
            agent_deployment_selector(agent_release),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if restart.returncode != 0:
        detail = (restart.stderr or restart.stdout).strip()
        msg = (
            f"Updated secret {secret_name!r} but could not restart the agent "
            f"Deployment.{f' {detail}' if detail else ''}"
        )
        raise click.ClickException(msg)


def run_cluster_rotate_token(
    client: ArenaClient,
    name: str,
    kubeconfig: Path | None = None,
    secret_name: str | None = None,
    namespace: str | None = None,
    run: KubectlRun = subprocess.run,
) -> str:
    """Rotate the Arena cluster token and write it into the agent Secret.

    :param client: Authenticated Arena client.
    :param name: Registered cluster name.
    :param kubeconfig: Optional kubeconfig for the target Kubernetes cluster.
    :param secret_name: Secret to update; Helm default when omitted.
    :param namespace: Agent namespace; discovered from Helm when omitted.
    :param run: kubectl runner.
    :return: The rotated token.
    """
    api = ByocApi(client)
    if api.find_cluster(name) is None:
        msg = f"No registered BYOC cluster named {name!r}."
        raise click.ClickException(msg)
    bundle = api.rotate_cluster_token(name)
    token = _normalize_token(bundle.get("token"))
    if token is None:
        msg = (
            f"Arena rotated cluster {name!r} but returned no token. "
            "Retry rotate-token, or set clusterToken on the agent Helm release."
        )
        raise click.ClickException(msg)
    resolved_secret = (secret_name or "").strip() or cluster_token_secret_name(bundle)
    kubeconfig_path = resolve_cluster_kubeconfig(name, kubeconfig)
    agent_release, resolved_namespace = discover_agent_install(
        kubeconfig_path,
        cluster_name=name,
        agent_namespace=namespace,
    )
    apply_cluster_token_secret(
        token=token,
        secret_name=resolved_secret,
        namespace=resolved_namespace,
        kubeconfig_path=kubeconfig_path,
        agent_release=agent_release,
        run=run,
    )
    click.echo(
        f"Rotated cluster token for {name!r} and updated secret "
        f"{resolved_secret!r} in namespace {resolved_namespace!r}."
    )
    return token
