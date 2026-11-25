# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Show the agent ClusterRole changes of an upgrade and ask before applying them."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

import click


def format_grant(grant: Mapping[str, Any]) -> str:
    """Return one grant as ``resource.group/name: verbs``.

    :param grant: ``{api_group, resource, resource_name?, verbs}`` from Arena.
    :type grant: Mapping[str, Any]
    :returns: The grant in kubectl resource notation.
    :rtype: str
    """
    target = grant["resource"]
    if grant["api_group"]:
        target = f"{target}.{grant['api_group']}"
    if grant.get("resource_name"):
        target = f"{target}/{grant['resource_name']}"
    return f"{target}: {', '.join(grant['verbs'])}"


def format_rbac_diff(rbac_diff: list[Mapping[str, Any]]) -> str:
    """Return the added (``+``) and removed (``-``) grants of each ClusterRole.

    :param rbac_diff: ``rbacDiff`` from the install-artifacts response.
    :type rbac_diff: list[Mapping[str, Any]]
    :returns: One block per ClusterRole, named by its chart role.
    :rtype: str
    """
    lines = []
    for role in rbac_diff:
        lines.append(f"  {role['cluster_role']}")
        lines.extend(f"    + {format_grant(grant)}" for grant in role["added"])
        lines.extend(f"    - {format_grant(grant)}" for grant in role["removed"])
    return "\n".join(lines)


def confirm_rbac_changes(
    artifacts: Mapping[str, Any], cluster_name: str, yes: bool
) -> None:
    """Print the ClusterRole changes of an upgrade and ask before it runs.

    An API that does not report ``agentUpgradeRequiresApproval`` upgrades
    without asking.

    :param artifacts: Install-artifacts response for the cluster.
    :type artifacts: Mapping[str, Any]
    :param cluster_name: Registered cluster name.
    :type cluster_name: str
    :param yes: Apply without asking.
    :type yes: bool
    :raises click.ClickException: If a prompt is needed and stdin is not a terminal.
    :raises click.Abort: If the user declines.
    """
    if not artifacts.get("agentUpgradeRequiresApproval"):
        return
    click.echo(
        f"This upgrade changes the agent's ClusterRoles on cluster {cluster_name!r}:"
    )
    diff = format_rbac_diff(artifacts.get("rbacDiff") or [])
    if diff:
        click.echo(diff)
    if yes:
        return
    if not sys.stdin.isatty():
        msg = (
            "This upgrade changes the agent's ClusterRoles and needs your approval. "
            "Run it in a terminal to confirm, or pass --yes."
        )
        raise click.ClickException(msg)
    click.confirm("Apply these ClusterRole changes?", default=False, abort=True)
