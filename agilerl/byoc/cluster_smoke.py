# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Prove a BYOC cluster can run jobs: its agent deploys and removes a small train release."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click

from agilerl.byoc.api import ByocApi
from agilerl.byoc.kube_target import kube_target, require_registered_cluster
from agilerl.client import ArenaClient


def _wait_for_result(
    api: ByocApi,
    name: str,
    smoke: dict[str, Any],
    timeout: float,
    poll_seconds: float,
) -> dict[str, Any]:
    """Poll Arena until the smoke has passed or failed.

    :raises click.ClickException: If it has neither after *timeout* seconds.
    """
    progress = {
        "Requested": "Waiting for the agent to pick up the smoke (it polls once a minute)…",
        "Deploying": "The agent installed the release. Waiting for its Ray head to be Ready…",
    }
    deadline = time.monotonic() + timeout
    shown: str | None = None
    while smoke["status"] not in {"Passed", "Failed"}:
        if smoke["status"] != shown:
            shown = smoke["status"]
            click.echo(progress.get(str(shown), f"Smoke is {shown}…"))
        if time.monotonic() >= deadline:
            msg = (
                f"Smoke {smoke['id']} on cluster {name!r} is still {smoke['status']} "
                f"after {timeout:.0f}s. Re-run this command to keep waiting for it."
            )
            raise click.ClickException(msg)
        time.sleep(poll_seconds)
        smoke = api.cluster_smoke(name, int(smoke["id"]))
    return smoke


def run_cluster_smoke(
    client: ArenaClient,
    name: str,
    kubeconfig: Path | None,
    context: str | None,
    platform: bool,
    timeout: float,
    poll_seconds: float = 5.0,
) -> None:
    """Have the agent of cluster *name* deploy a small train release and remove it.

    The agent installs the release as its own ServiceAccount, the way it
    installs a job, waits until the Ray head is Ready and removes the release.
    The release mounts no storage and has no GPU workers. A smoke that is
    already running is waited on instead of starting another.

    :param client: Authenticated Arena client.
    :type client: ArenaClient
    :param name: Registered cluster name.
    :type name: str
    :param kubeconfig: Kubeconfig for the agent cluster, or ``None``.
    :type kubeconfig: Path | None
    :param context: Kubernetes context to use, or ``None``.
    :type context: str | None
    :param platform: Smoke one of AgileRL's platform clusters.
    :type platform: bool
    :param timeout: Seconds to wait for the result.
    :type timeout: float
    :param poll_seconds: Seconds between status reads.
    :type poll_seconds: float
    :raises click.ClickException: If the smoke fails or has no result in time.
    """
    api = ByocApi(client, platform=platform)
    with kube_target(name, kubeconfig, context) as target:
        require_registered_cluster(api, name, target)
        smoke = api.request_cluster_smoke(name, target.cluster_uid)
    click.echo(
        f"Smoke {smoke['id']} on cluster {name!r}: the agent deploys Helm release "
        f"{smoke['namespace']}, then removes it."
    )
    smoke = _wait_for_result(api, name, smoke, timeout, poll_seconds)
    if smoke["status"] == "Failed":
        msg = f"Smoke failed on cluster {name!r}: {smoke.get('error') or 'no reason given'}"
        raise click.ClickException(msg)
    click.echo(
        f"Smoke passed on cluster {name!r}: the agent deployed a train release as its "
        "ServiceAccount, its Ray head became Ready, and the agent removed it."
    )
