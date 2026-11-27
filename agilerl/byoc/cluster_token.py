# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rotate or revoke the credential an enterprise cluster's agent signs in with."""

from __future__ import annotations

from typing import Any

import click

from agilerl.byoc.api import ByocApi
from agilerl.client import ArenaClient


def run_cluster_rotate_token(
    client: ArenaClient, name: str, platform: bool = False
) -> dict[str, Any]:
    """Ask the agent of cluster *name* to renew its credential.

    The agent picks the new credential up on its next poll, and the current
    one keeps working until it does, so nothing in the cluster changes here.

    :param client: Authenticated Arena client.
    :param name: Registered cluster name.
    :param platform: Rotate for one of AgileRL's platform clusters.
    :return: The cluster's credential state.
    """
    credential = ByocApi(client, platform=platform).renew_cluster_credential(name)
    click.echo(
        f"Asked the agent on {name!r} to renew its credential. It switches on its "
        "next poll; the current credential keeps working until then."
    )
    return credential


def run_cluster_revoke_token(
    client: ArenaClient, name: str, platform: bool = False
) -> dict[str, Any]:
    """Revoke every credential of cluster *name* now.

    :param client: Authenticated Arena client.
    :param name: Registered cluster name.
    :param platform: Revoke for one of AgileRL's platform clusters.
    :return: The cluster's credential state.
    """
    credential = ByocApi(client, platform=platform).revoke_cluster_credentials(name)
    click.echo(
        f"Revoked every credential of {name!r}. The agent is refused from its next call."
    )
    click.echo(f"To reconnect it, {_reconnect_hint(name, platform)}")
    return credential


def _reconnect_hint(name: str, platform: bool) -> str:
    """How to give a revoked cluster's agent a new enrollment token.

    With no live token left, registering the cluster again issues one.
    """
    if platform:
        return (
            "issue it a new agent token on the Platform clusters staff page and "
            "install that token (for a GitOps cluster, put it in its secret store)."
        )
    return f"run: arena cluster connect --name {name}"
