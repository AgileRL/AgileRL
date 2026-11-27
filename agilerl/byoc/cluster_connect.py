# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Connect a Kubernetes cluster to Arena: preflight, register, install, wait for the agent."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import click

from agilerl.byoc.api import ByocApi
from agilerl.byoc.cluster_helm import helm_available, resolve_install_release
from agilerl.byoc.cluster_register import (
    offer_suggested_classes,
    run_cluster_register,
)
from agilerl.byoc.kube_target import (
    KubeTarget,
    kube_target,
    kubeconfig_environment,
    require_same_cluster,
)
from agilerl.byoc.preflight import (
    CheckStatus,
    PreflightCheck,
    any_failed,
    echo_checks,
    require_passed,
    run_preflight,
)
from agilerl.config import CommandConfig, arena_client
from agilerl.exceptions import ArenaAPIError


@dataclass(frozen=True)
class ConnectOptions:
    """What ``arena cluster connect`` was asked to do."""

    name: str
    kubeconfig: Path | None
    kube_context: str | None
    agent_namespace: str
    storage_endpoint: str | None
    storage_bucket: str | None
    storage_prefix: str | None
    storage_secret_name: str | None
    ray_data_storage_class_name: str | None
    rotate_token: bool
    wait_timeout: float
    yes: bool


def registered_cluster(
    api: ByocApi, name: str, target: KubeTarget
) -> dict[str, Any] | None:
    """Return Arena's cluster *name*, or ``None`` when Arena has no such cluster.

    :raises click.ClickException: If Arena stores another cluster UID for *name*.
    """
    try:
        registered = api.find_cluster(name)
    except ArenaAPIError as exc:
        # Arena answers 404 until the org's BYOC provider is enabled, which register does.
        if exc.status_code != HTTPStatus.NOT_FOUND:
            raise
        return None
    if registered is not None:
        require_same_cluster(registered, name, target)
    return registered


def _reported_at(cluster: dict[str, Any] | None) -> object:
    health = None if cluster is None else cluster.get("health")
    return None if health is None else health["reported_at"]


def _poll_cluster(
    api: ByocApi,
    name: str,
    deadline: float,
    ready: Callable[[dict[str, Any]], bool],
    timed_out: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    """Re-read cluster *name* every few seconds until *ready* holds."""
    while True:
        cluster = api.byoc_cluster(name)
        if cluster is None:
            msg = f"Arena no longer lists cluster {name!r}."
            raise click.ClickException(msg)
        if ready(cluster):
            return cluster
        if time.monotonic() >= deadline:
            raise click.ClickException(timed_out(cluster))
        time.sleep(5.0)


def wait_for_agent(
    api: ByocApi, name: str, before: dict[str, Any] | None, timeout: float
) -> dict[str, Any]:
    """Wait for a heartbeat and then a health report newer than those in *before*.

    :param before: Arena's row for *name* after Helm returns, or ``None``.
    :param timeout: Seconds to wait for both.
    :return: Arena's row for *name* with the new health report.
    :raises click.ClickException: If either does not arrive in time.
    """
    deadline = time.monotonic() + timeout
    last_poll = None if before is None else before.get("last_poll_at")
    last_report = _reported_at(before)

    def no_heartbeat(cluster: dict[str, Any]) -> str:
        return (
            f"The agent has not reached Arena after {timeout:.0f}s. Check that its pod "
            "runs (kubectl get pods --namespace <agent namespace>), and that the "
            "cluster can reach the Arena gateway over UDP and the Arena API over HTTPS."
            f" Last agent error: {cluster.get('last_error') or 'none'}."
        )

    def no_report(_cluster: dict[str, Any]) -> str:
        return (
            f"The agent is connected but sent no health report within {timeout:.0f}s. "
            "Run arena cluster upgrade if it is older than the chart Arena serves."
        )

    click.echo("Waiting for the agent to reach Arena…")
    cluster = _poll_cluster(
        api,
        name,
        deadline,
        lambda row: row.get("last_poll_at") not in (None, last_poll),
        no_heartbeat,
    )
    click.echo(f"Agent status: {cluster.get('connectivity_status')}.")
    click.echo("Waiting for the agent's health report…")
    return _poll_cluster(
        api,
        name,
        deadline,
        lambda row: _reported_at(row) not in (None, last_report),
        no_report,
    )


def health_checks(cluster: dict[str, Any]) -> list[PreflightCheck]:
    """Return the agent's health report as checks, green as pass and red as fail."""
    statuses = {
        "green": CheckStatus.PASS,
        "amber": CheckStatus.WARN,
        "red": CheckStatus.FAIL,
    }
    return [
        PreflightCheck(
            check["name"], statuses[check["status"]], check["detail"], check.get("fix")
        )
        for check in cluster["health"]["checks"]
    ]


def context_flag(context: str) -> str:
    """Return ``--context '<context>'``, quoted for a POSIX shell."""
    quoted = context.replace("'", "'\"'\"'")
    return f"--context '{quoted}'"


def run_cluster_connect(config: CommandConfig, options: ConnectOptions) -> None:
    """Preflight, register, install and wait until the agent reports its health.

    :raises click.ClickException: If helm is missing, a preflight check fails,
        the agent does not report in time, or its health report has a red check.
    """
    if not helm_available():
        msg = "helm is not on PATH. Install Helm 3, then run this again."
        raise click.ClickException(msg)
    name = options.name
    with (
        kube_target(name, options.kubeconfig, options.kube_context) as target,
        kubeconfig_environment(target.kubeconfig),
        arena_client(config) as client,
    ):
        api = ByocApi(client)
        registered = registered_cluster(api, name, target)
        # The agent of the cluster Arena already has here; a reinstall upgrades it.
        own_release = (
            None
            if registered is None
            else resolve_install_release(name, options.agent_namespace)
        )
        checks = run_preflight(
            target.kubeconfig,
            options.agent_namespace,
            own_release,
            options.ray_data_storage_class_name,
        )
        flag = context_flag(target.context)
        click.echo(f"Preflight on {flag}:")
        echo_checks(checks)
        require_passed(checks)
        run_cluster_register(
            client,
            name=name,
            output_dir=Path(f"arena-cluster-{name}"),
            skip_enable=False,
            force=False,
            install=True,
            no_write=True,
            agent_namespace=options.agent_namespace,
            storage_endpoint=options.storage_endpoint,
            storage_bucket=options.storage_bucket,
            storage_prefix=options.storage_prefix,
            storage_secret_name=options.storage_secret_name,
            ray_data_storage_class_name=options.ray_data_storage_class_name,
            placement="in-cluster",
            cluster_uid=target.cluster_uid,
            rotate_token=options.rotate_token,
            suggest_classes=False,
        )
        # The new ReplicaSet is ready. A report before this is the previous pod.
        before = None if registered is None else api.byoc_cluster(name)
        cluster = wait_for_agent(api, name, before, options.wait_timeout)
        health = health_checks(cluster)
        click.echo("Agent health:")
        echo_checks(health)
        offer_suggested_classes(api, name, options.yes)
    if any_failed(health):
        msg = (
            f"Cluster {name!r} ({flag}) is connected, but its health report has "
            "failing checks. Fix them; the agent reports again within 10 minutes."
        )
        raise click.ClickException(msg)
    click.echo(f"Cluster {name!r} is connected ({flag}).")
