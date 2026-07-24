"""Register an enterprise on-prem cluster and write Helm install bundles."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import click
import yaml
from agilerl.arena.client import ArenaClient
from agilerl.arena.cli_manifest import write_text_atomic
from agilerl.arena.on_prem.api import OnPremApi
from agilerl.arena.on_prem.cluster_helm import (
    install_enterprise_agent_chart,
    install_lab_cluster_charts,
)

logger = logging.getLogger("agilerl.arena.on_prem")

_ENTERPRISE_REQUIRED = (
    "storage_endpoint",
    "storage_bucket",
    "storage_secret_name",
)


def _validate_enterprise_fields(*, profile: str, fields: dict[str, Any]) -> None:
    """Reject enterprise registration when required storage fields are missing.

    :param profile: Install profile (``lab`` or ``enterprise``).
    :type profile: str
    :param fields: Registration option values keyed by snake_case names.
    :type fields: dict[str, Any]
    :returns: None
    :rtype: None
    :raises click.ClickException: If enterprise profile lacks required storage fields.
    """
    if profile != "enterprise":
        return
    missing = [
        name
        for name in _ENTERPRISE_REQUIRED
        if not (fields.get(name) or "").strip()
    ]
    if missing:
        labels = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        msg = (
            "Enterprise profile requires storage options: "
            f"{labels}."
        )
        raise click.ClickException(msg)


def _write_yaml(path: Path, data: object, *, force: bool) -> None:
    """Serialize *data* as YAML to *path*.

    :param path: Destination file path.
    :type path: Path
    :param data: JSON-serializable object to dump.
    :type data: object
    :param force: Overwrite existing files when ``True``.
    :type force: bool
    :returns: None
    :rtype: None
    """
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    write_text_atomic(path, text, force=force)


def _write_token(path: Path, token: str, *, force: bool) -> None:
    """Write the cluster token with mode ``0600``.

    :param path: Destination file path.
    :type path: Path
    :param token: Cluster API bearer token.
    :type token: str
    :param force: Overwrite existing files when ``True``.
    :type force: bool
    :returns: None
    :rtype: None
    """
    write_text_atomic(path, token + "\n", force=force)
    os.chmod(path, 0o600)


def _print_next_steps(*, profile: str, output_dir: Path, installed: bool = False) -> None:
    """Print Helm install instructions for the registered cluster.

    :param profile: Install profile (``lab`` or ``enterprise``).
    :type profile: str
    :param output_dir: Directory containing generated values files.
    :type output_dir: Path
    :param installed: When ``True``, skip manual ``helm upgrade`` instructions.
    :type installed: bool
    :returns: None
    :rtype: None
    """
    if installed:
        click.echo("")
        click.echo("Charts are installed. Create an enterprise resource class linked to")
        click.echo("this cluster, then schedule training jobs from the Arena UI or CLI.")
        return

    agent_values = output_dir / "agent-helm-values.yaml"
    click.echo("")
    click.echo("Next steps:")
    if profile == "lab":
        storage_values = output_dir / "storage-helm-values.yaml"
        click.echo(
            "  1. Install storage (lab MinIO): "
            "helm upgrade --install arena-on-prem-storage <chart> "
            f"--namespace storage --create-namespace -f {storage_values}"
        )
        click.echo(
            "  2. Install agent: "
            "helm upgrade --install arena-on-prem-agent <chart> "
            f"--namespace arena-on-prem --create-namespace -f {agent_values}"
        )
    else:
        click.echo(
            "  1. Install agent: "
            "helm upgrade --install arena-on-prem-agent <chart> "
            f"--namespace arena-on-prem --create-namespace -f {agent_values}"
        )
    click.echo("")
    click.echo(
        "Charts ship with agilerl-platform under resources/helm-setup/. "
        "See each chart README for validate.sh and RBAC notes."
    )


def run_cluster_register(
    client: ArenaClient,
    *,
    name: str,
    profile: str,
    output_dir: Path,
    skip_enable: bool,
    force: bool,
    install: bool = False,
    charts_dir: Path | None = None,
    helm_wait: bool = True,
    storage_endpoint: str | None = None,
    storage_bucket: str | None = None,
    storage_prefix: str | None = None,
    storage_secret_name: str | None = None,
    ingress_class_name: str | None = None,
    hostname_template: str | None = None,
    gateway_api_parent_refs: Any | None = None,
    tls_secret_name: str | None = None,
    preprocessing_resource_class_id: int | None = None,
    ray_data_storage_class_name: str | None = None,
    ray_data_pvc_size: str | None = None,
) -> None:
    """Enable on-prem, register the cluster, and write Helm values files.

    :param client: Authenticated Arena client.
    :type client: ArenaClient
    :param name: Cluster name.
    :type name: str
    :param profile: Install profile (``lab`` or ``enterprise``).
    :type profile: str
    :param output_dir: Directory for generated YAML and token files.
    :type output_dir: Path
    :param skip_enable: Skip enabling the on-prem provider when ``True``.
    :type skip_enable: bool
    :param force: Overwrite existing output files when ``True``.
    :type force: bool
    :param install: Run ``helm upgrade --install`` for storage (lab) and agent.
    :type install: bool
    :param charts_dir: Platform ``resources/helm-setup`` root for ``--install``.
    :type charts_dir: Path | None
    :param helm_wait: Pass ``--wait`` to Helm when installing.
    :type helm_wait: bool
    :returns: None
    :rtype: None
    """
    api = OnPremApi(client)
    field_values = {
        "storage_endpoint": storage_endpoint,
        "storage_bucket": storage_bucket,
        "storage_secret_name": storage_secret_name,
    }
    _validate_enterprise_fields(profile=profile, fields=field_values)

    if not skip_enable:
        api.enable()

    bundle = api.register_cluster(
        name=name,
        install_profile=profile,
        storage_endpoint=storage_endpoint,
        storage_bucket=storage_bucket,
        storage_prefix=storage_prefix,
        storage_secret_name=storage_secret_name,
        ingress_class_name=ingress_class_name,
        hostname_template=hostname_template,
        gateway_api_parent_refs=gateway_api_parent_refs,
        tls_secret_name=tls_secret_name,
        preprocessing_resource_class_id=preprocessing_resource_class_id,
        ray_data_storage_class_name=ray_data_storage_class_name,
        ray_data_pvc_size=ray_data_pvc_size,
    )

    out = output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    agent_values = bundle.get("agentHelmValues")
    if not isinstance(agent_values, dict):
        msg = "Registration response missing agentHelmValues."
        raise click.ClickException(msg)
    _write_yaml(out / "agent-helm-values.yaml", agent_values, force=force)

    storage_values = bundle.get("storageHelmValues")
    if profile == "lab":
        if not isinstance(storage_values, dict):
            msg = "Lab registration response missing storageHelmValues."
            raise click.ClickException(msg)
        _write_yaml(out / "storage-helm-values.yaml", storage_values, force=force)

    token = bundle.get("token")
    if not isinstance(token, str) or not token:
        msg = "Registration response missing cluster token."
        raise click.ClickException(msg)
    _write_token(out / "cluster-token.txt", token, force=force)

    cluster_id = bundle.get("clusterId")
    cluster_api_url = bundle.get("clusterApiUrl")
    cluster_summary = bundle.get("cluster")
    cluster_name = name
    if isinstance(cluster_summary, dict):
        cluster_name = str(cluster_summary.get("name", name))

    click.echo(f"Registered cluster {cluster_name!r} (id={cluster_id}).")
    click.echo(f"Cluster API URL: {cluster_api_url}")
    click.echo(f"Wrote Helm values to {out}/")
    if profile == "lab":
        click.echo("  - storage-helm-values.yaml")
    click.echo("  - agent-helm-values.yaml")
    click.echo("  - cluster-token.txt (mode 0600)")

    if install:
        if profile == "lab":
            install_lab_cluster_charts(out, charts_dir=charts_dir, wait=helm_wait)
        else:
            install_enterprise_agent_chart(out, charts_dir=charts_dir, wait=helm_wait)
        click.echo("")
        click.echo("Helm install finished.")

    _print_next_steps(profile=profile, output_dir=out, installed=install)
    logger.info("Cluster registration finished for %r (%s).", name, profile)
