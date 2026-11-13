# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helm-upgrade the BYOC agent chart without rotating cluster credentials."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import click

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_helm import (
    AGENT_CHART,
    STORAGE_RELEASE,
    _load_agent_values_yaml,
    _restart_agent_deployment,
    agent_deployment_selector,
    discover_agent_install,
    helm_upgrade_install,
    list_installed_byoc_releases,
    sync_storage_secret_endpoint,
)
from agilerl.arena.byoc.cluster_install_package import extract_cluster_install_package
from agilerl.arena.byoc.cluster_token import resolve_cluster_kubeconfig
from agilerl.arena.client import ArenaClient


@contextmanager
def _kubeconfig_environment(kubeconfig_path: Path) -> Iterator[None]:
    """Set ``KUBECONFIG`` only while Helm runs."""
    old_value = os.environ.get("KUBECONFIG")
    os.environ["KUBECONFIG"] = str(kubeconfig_path)
    try:
        yield
    finally:
        if old_value is None:
            del os.environ["KUBECONFIG"]
        else:
            os.environ["KUBECONFIG"] = old_value


def _installed_agent_release(
    kubeconfig: Path | None,
    cluster_name: str,
    agent_namespace: str | None,
) -> tuple[str, str]:
    """Return the installed agent Helm ``(release, namespace)``.

    :param kubeconfig: Kubeconfig used as ``KUBECONFIG`` for Helm.
    :type kubeconfig: Path | None
    :param cluster_name: Registered cluster name.
    :type cluster_name: str
    :param agent_namespace: Explicit namespace, or ``None`` to search all of them.
    :type agent_namespace: str | None
    :returns: Installed agent release and namespace.
    :rtype: tuple[str, str]
    :raises click.ClickException: If no agent release is installed.
    """
    release, namespace = discover_agent_install(
        kubeconfig,
        cluster_name=cluster_name,
        agent_namespace=agent_namespace,
    )
    # discover_agent_install returns a release name when Helm has not installed it.
    installed = {
        pair
        for pair in list_installed_byoc_releases(kubeconfig, cluster_name=cluster_name)
        if pair[0] != STORAGE_RELEASE
    }
    if (release, namespace) in installed:
        return release, namespace
    if kubeconfig is None:
        msg = (
            "No kubeconfig found, so the installed agent Helm release could not "
            "be discovered. Pass --kubeconfig."
        )
        raise click.ClickException(msg)
    msg = (
        f"Helm release {release!r} is not installed in namespace {namespace!r}. "
        "This command only upgrades an agent that is already installed."
    )
    raise click.ClickException(msg)


def run_cluster_upgrade(
    client: ArenaClient,
    *,
    name: str,
    kubeconfig: Path | None = None,
    agent_namespace: str | None = None,
    helm_wait: bool = True,
) -> None:
    """Upgrade the installed BYOC agent chart without rotating credentials.

    Credential Secrets from the install package are not applied. The storage
    chart is not installed or upgraded. When the chart will not recreate the
    object-storage Secret, ``storage.endpoint`` is copied into that Secret and
    the agent Deployment is restarted.

    :param client: Authenticated Arena client.
    :type client: ArenaClient
    :param name: Registered cluster name.
    :type name: str
    :param kubeconfig: Kubeconfig for the agent cluster.
    :type kubeconfig: Path | None
    :param agent_namespace: Agent namespace. Helm's namespace when omitted.
    :type agent_namespace: str | None
    :param helm_wait: Wait for the agent Deployment when ``True``.
    :type helm_wait: bool
    :returns: None
    :rtype: None
    :raises click.ClickException: If no agent Helm release is installed.
    """
    cluster_name = name.strip()
    if not cluster_name:
        msg = "--name is required."
        raise click.ClickException(msg)
    explicit_namespace = (agent_namespace or "").strip() or None
    api = ByocApi(client)
    artifacts = api.fetch_cluster_install_artifacts(cluster_name)
    agent_yaml = artifacts.get("agentHelmValuesYaml")
    if not isinstance(agent_yaml, str) or not agent_yaml.strip():
        msg = "Arena did not return agent Helm values."
        raise click.ClickException(msg)
    package_bytes = api.download_cluster_install_package(
        cluster=cluster_name,
        agent_helm_values_yaml=agent_yaml,
    )
    kubeconfig_path = resolve_cluster_kubeconfig(cluster_name, kubeconfig)
    kubeconfig_cm = (
        _kubeconfig_environment(kubeconfig_path)
        if kubeconfig_path is not None
        else nullcontext()
    )
    with kubeconfig_cm:
        release, namespace = _installed_agent_release(
            kubeconfig_path,
            cluster_name,
            explicit_namespace,
        )
        with tempfile.TemporaryDirectory(prefix="arena-cluster-upgrade-") as tmp:
            package_root = extract_cluster_install_package(package_bytes, Path(tmp))
            values_file = package_root / "agent-helm-values.yaml"
            if not values_file.is_file():
                msg = f"Missing {values_file}; install package is incomplete."
                raise click.ClickException(msg)
            chart = package_root / AGENT_CHART / "chart"
            if not chart.is_dir():
                msg = f"Helm chart not found at {chart}"
                raise click.ClickException(msg)
            agent_values = _load_agent_values_yaml(values_file)
            endpoint_updated = sync_storage_secret_endpoint(
                agent_values, agent_namespace=namespace
            )
            click.echo(
                f"Upgrading Helm release {release!r} in namespace {namespace!r}."
            )
            helm_upgrade_install(
                release=release,
                chart=chart,
                namespace=namespace,
                values_file=values_file,
                wait=helm_wait,
                wait_for_selector=agent_deployment_selector(release),
            )
            if endpoint_updated:
                _restart_agent_deployment(namespace, release, wait=helm_wait)
    click.echo(f"Upgraded BYOC agent on cluster {cluster_name!r}.")
