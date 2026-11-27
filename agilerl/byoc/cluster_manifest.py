# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Write the GitOps manifest that installs the BYOC agent at Arena's current release."""

from __future__ import annotations

from pathlib import Path

import click

from agilerl.byoc.api import ByocApi
from agilerl.byoc.upgrade_rbac import format_rbac_diff
from agilerl.client import ArenaClient

MANIFEST_KEYS = {
    "argocd": "argocdApplicationYaml",
    "flux": "fluxYaml",
}


def run_cluster_manifest(
    client: ArenaClient,
    *,
    name: str,
    manifest_format: str,
    output: Path | None = None,
    platform: bool = False,
) -> None:
    """Write the Argo CD or Flux manifest for the cluster named *name*.

    The manifest pins the agent chart this Arena release ships and carries the
    cluster's current values, which hold no secrets. When the release changes
    the agent's ClusterRoles, the changes are printed to stderr: the GitOps tool
    needs the rights to apply them.

    :param client: Authenticated Arena client.
    :type client: ArenaClient
    :param name: Registered cluster name.
    :type name: str
    :param manifest_format: ``argocd`` or ``flux``.
    :type manifest_format: str
    :param output: File to write. Standard output when ``None``.
    :type output: Path | None
    :param platform: Target one of AgileRL's platform clusters.
    :type platform: bool
    :returns: None
    :rtype: None
    :raises click.ClickException: If Arena serves no GitOps manifests.
    """
    cluster_name = name.strip()
    if not cluster_name:
        msg = "--name is required."
        raise click.ClickException(msg)
    artifacts = ByocApi(client, platform=platform).fetch_cluster_install_artifacts(
        cluster_name
    )
    manifest = artifacts.get(MANIFEST_KEYS[manifest_format])
    if not isinstance(manifest, str) or not manifest.strip():
        msg = (
            "This Arena serves no GitOps manifests: it has no private chart "
            "registry. Install the agent with `arena cluster connect` or the "
            "install package instead."
        )
        raise click.ClickException(msg)
    if artifacts.get("agentUpgradeRequiresApproval"):
        click.echo(
            "This release changes the agent's ClusterRoles on cluster "
            f"{cluster_name!r}. Your GitOps tool needs the rights to apply them:",
            err=True,
        )
        diff = format_rbac_diff(artifacts.get("rbacDiff") or [])
        if diff:
            click.echo(diff, err=True)
    if output is None:
        click.echo(manifest, nl=False)
        return
    output.write_text(manifest, encoding="utf-8")
    click.echo(f"Wrote {output}.", err=True)
