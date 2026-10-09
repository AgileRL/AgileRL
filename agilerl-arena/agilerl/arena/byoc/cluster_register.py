# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Register an enterprise BYOC cluster and write Helm install bundles."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import click
import yaml
from typing_extensions import NotRequired, Unpack

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_helm import (
    AGENT_NAMESPACE,
    AGENT_RELEASE,
    install_enterprise_agent_chart,
    install_from_install_package_root,
    install_lab_cluster_charts,
    normalize_agent_namespace,
    normalize_agent_release,
    resolve_install_release,
)
from agilerl.arena.byoc.cluster_install_package import (
    extract_cluster_install_package,
)
from agilerl.arena.byoc.provisioning.inference import (
    normalize_inference_domain,
    resolve_inference_hostname_template,
)
from agilerl.arena.cli_manifest import write_text_atomic
from agilerl.arena.client import ArenaClient

logger = logging.getLogger("agilerl.arena.byoc")


@dataclass(frozen=True)
class ClusterResourceClass:
    """BYOC resource class created after cluster registration."""

    name: str
    num_nodes: int
    node_selector: dict[str, str]
    metadata: dict[str, object]


def _write_yaml(path: Path, data: object, force: bool) -> None:
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


def _write_token(path: Path, token: str, force: bool) -> None:
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


def _print_next_steps(
    output_dir: Path,
    installed: bool = False,
    has_storage_values: bool = False,
    storage_installed: bool = False,
    resource_class_names: Sequence[str] = (),
    agent_namespace: str = AGENT_NAMESPACE,
    agent_release: str = AGENT_RELEASE,
) -> None:
    """Print Helm install instructions for the registered cluster.

    :param output_dir: Directory containing generated values files.
    :type output_dir: Path
    :param installed: When ``True``, the agent chart is already installed.
    :type installed: bool
    :param has_storage_values: When ``True``, include MinIO storage install steps.
    :type has_storage_values: bool
    :param storage_installed: When ``True``, skip the storage Helm instruction.
    :type storage_installed: bool
    :param resource_class_names: GPU classes created for this cluster.
    :type resource_class_names: Sequence[str]
    :param agent_namespace: Kubernetes namespace for the agent chart.
    :type agent_namespace: str
    :returns: None
    :rtype: None
    """
    if installed:
        click.echo("")
        if resource_class_names:
            listed = ", ".join(repr(name) for name in resource_class_names)
            noun = (
                "Resource class"
                if len(resource_class_names) == 1
                else "Resource classes"
            )
            verb = "is" if len(resource_class_names) == 1 else "are"
            click.echo(
                f"Charts are installed. {noun} {listed} {verb} linked to this cluster."
            )
            return
        click.echo(
            "Charts are installed. Create an enterprise resource class linked to"
        )
        click.echo(
            "this cluster, then schedule training jobs from the Arena UI or CLI."
        )
        return

    if resource_class_names:
        listed = ", ".join(repr(name) for name in resource_class_names)
        noun = (
            "Resource class" if len(resource_class_names) == 1 else "Resource classes"
        )
        verb = "is" if len(resource_class_names) == 1 else "are"
        click.echo(f"{noun} {listed} {verb} linked to this cluster.")
    agent_values = output_dir / "agent-helm-values.yaml"
    click.echo("")
    click.echo("Next steps:")
    step = 1
    if has_storage_values and not storage_installed:
        storage_values = output_dir / "storage-helm-values.yaml"
        click.echo(
            f"  {step}. Install storage (optional MinIO): "
            "helm upgrade --install arena-byoc-storage <chart> "
            f"--namespace storage --create-namespace -f {storage_values}"
        )
        step += 1
    click.echo(
        f"  {step}. Install agent: "
        f"helm upgrade --install {agent_release} <chart> "
        f"--namespace {agent_namespace} --create-namespace -f {agent_values}"
    )
    click.echo("")
    click.echo(
        "Run ``arena cluster register --install`` to deploy the agent, "
        "and ``--install-storage`` to also install bundled MinIO."
    )


def _normalize_cluster_token(token: object) -> str | None:
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _require_token_for_install(token: object) -> str:
    normalized = _normalize_cluster_token(token)
    if normalized:
        return normalized
    msg = (
        "Cannot --install without an agent token. "
        "Rotate token in Arena (Account → Enterprise K8s clusters), "
        "then register again or set clusterToken on the agent Helm release."
    )
    raise click.ClickException(msg)


def _agent_values_for_helm(
    agent_values: dict[str, Any],
    token: str | None,
    inference_domain: str | None = None,
    inference_hostname_template: str | None = None,
) -> dict[str, Any]:
    merged = dict(agent_values)
    if token:
        merged["clusterToken"] = token
        merged.pop("existingClusterTokenSecret", None)
    if inference_domain is not None or inference_hostname_template is not None:
        inference = merged.get("inference")
        if not isinstance(inference, dict):
            inference = {}
        if inference_domain is not None:
            inference["domain"] = inference_domain
        if inference_hostname_template is not None:
            inference["hostnameTemplate"] = inference_hostname_template
        merged["inference"] = inference
    return merged


def _bundle_helm_values_yaml(
    agent_values: dict[str, Any],
    storage_values: object,
    include_storage: bool,
) -> tuple[str, str | None]:
    agent_yaml = yaml.safe_dump(agent_values, sort_keys=False, default_flow_style=False)
    storage_yaml = None
    if include_storage and isinstance(storage_values, dict):
        storage_yaml = yaml.safe_dump(
            storage_values, sort_keys=False, default_flow_style=False
        )
    return agent_yaml, storage_yaml


def _registered_cluster_name(bundle: dict[str, Any], requested_name: str) -> str:
    """Return the cluster name from a register response, else *requested_name*."""
    cluster = bundle.get("cluster")
    if isinstance(cluster, dict):
        raw = cluster.get("name")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return requested_name


def _install_charts_from_arena_package(
    api: ByocApi,
    cluster_name: str,
    agent_values: dict[str, Any],
    storage_values: object,
    include_storage: bool,
    helm_wait: bool,
    agent_namespace: str,
    agent_release: str,
    install_agent: bool,
) -> None:
    agent_yaml, storage_yaml = _bundle_helm_values_yaml(
        agent_values,
        storage_values,
        include_storage=include_storage,
    )
    package_bytes = api.download_cluster_install_package(
        cluster=cluster_name,
        agent_helm_values_yaml=agent_yaml,
        storage_helm_values_yaml=storage_yaml,
    )
    with tempfile.TemporaryDirectory(prefix="arena-install-package-") as tmp:
        package_root = extract_cluster_install_package(package_bytes, Path(tmp))
        install_from_install_package_root(
            package_root,
            wait=helm_wait,
            agent_release=agent_release,
            agent_namespace=agent_namespace,
            install_agent=install_agent,
        )


def _install_charts_from_local_values(
    work: Path,
    include_storage: bool,
    charts_dir: Path,
    helm_wait: bool,
    agent_namespace: str,
    agent_release: str,
    install_agent: bool,
) -> None:
    if include_storage:
        install_lab_cluster_charts(
            work,
            charts_dir=charts_dir,
            wait=helm_wait,
            agent_namespace=agent_namespace,
            agent_release=agent_release,
            install_agent=install_agent,
        )
        return
    install_enterprise_agent_chart(
        work,
        charts_dir=charts_dir,
        wait=helm_wait,
        agent_namespace=agent_namespace,
        agent_release=agent_release,
    )


def _ensure_resource_class(
    api: ByocApi,
    cluster_name: str,
    resource_class: ClusterResourceClass,
) -> str:
    """Create the GPU resource class unless a class with that name already exists."""
    existing = api.find_class(resource_class.name)
    if existing is not None:
        click.echo(
            f"Resource class {resource_class.name!r} already exists; leaving it unchanged."
        )
        return resource_class.name
    created = api.create_class(
        name=resource_class.name,
        num_nodes=resource_class.num_nodes,
        cluster_name=cluster_name,
        node_selector=resource_class.node_selector,
        metadata=resource_class.metadata,
    )
    click.echo(f"Created resource class {created.get('name', resource_class.name)!r}.")
    return resource_class.name


class ClusterRegisterOptions(TypedDict):
    """Keyword fields for ``run_cluster_register``."""

    name: str
    output_dir: Path
    skip_enable: bool
    force: bool
    install: NotRequired[bool]
    install_storage: NotRequired[bool]
    narrow_allowed_ips: NotRequired[bool]
    no_write: NotRequired[bool]
    charts_dir: NotRequired[Path | None]
    helm_wait: NotRequired[bool]
    agent_namespace: NotRequired[str]
    storage_endpoint: NotRequired[str | None]
    storage_bucket: NotRequired[str | None]
    storage_prefix: NotRequired[str | None]
    storage_secret_name: NotRequired[str | None]
    ingress_class_name: NotRequired[str | None]
    hostname_template: NotRequired[str | None]
    gateway_api_parent_refs: NotRequired[object | None]
    tls_secret_name: NotRequired[str | None]
    preprocessing_resource_class: NotRequired[str | None]
    ray_data_storage_class_name: NotRequired[str | None]
    ray_data_pvc_size: NotRequired[str | None]
    inference_domain: NotRequired[str | None]
    byoc_provider: NotRequired[dict[str, Any] | None]
    resource_classes: NotRequired[Sequence[ClusterResourceClass]]


def run_cluster_register(
    client: ArenaClient, **options: Unpack[ClusterRegisterOptions]
) -> None:
    """Enable BYOC, register the cluster, and write Helm values files.

    :param client: Authenticated Arena client.
    :type client: ArenaClient
    :param name: Cluster name.
    :type name: str
    :param output_dir: Directory for generated YAML and token files.
    :type output_dir: Path
    :param skip_enable: Skip enabling the BYOC provider when ``True``.
    :type skip_enable: bool
    :param force: Overwrite existing output files when ``True``.
    :type force: bool
    :param install: Run ``helm upgrade --install`` for the agent chart.
    :type install: bool
    :param install_storage: Request bundled MinIO values and install the storage chart.
    :type install_storage: bool
    :param narrow_allowed_ips: Omit ``10.0.0.0/8`` from WireGuard AllowedIPs (k3d/k3s).
    :type narrow_allowed_ips: bool
    :param no_write: Skip persisting values and token files under ``output_dir``.
    :type no_write: bool
    :param charts_dir: Optional platform ``resources/helm-setup`` root for local chart overrides.
    :type charts_dir: Path | None
    :param helm_wait: Wait for storage Helm resources and the agent Deployment.
    :type helm_wait: bool
    :param agent_namespace: Kubernetes namespace for the agent Helm release.
    :type agent_namespace: str
    :param resource_classes: GPU classes to create for the registered cluster.
    :type resource_classes: Sequence[ClusterResourceClass]
    :returns: None
    :rtype: None
    """
    name = options["name"]
    output_dir = options["output_dir"]
    skip_enable = options["skip_enable"]
    force = options["force"]
    install = options.get("install", False)
    install_storage = options.get("install_storage", False)
    narrow_allowed_ips = options.get("narrow_allowed_ips", False)
    no_write = options.get("no_write", False)
    charts_dir = options.get("charts_dir")
    helm_wait = options.get("helm_wait", True)
    agent_namespace = options.get("agent_namespace", AGENT_NAMESPACE)
    storage_endpoint = options.get("storage_endpoint")
    storage_bucket = options.get("storage_bucket")
    storage_prefix = options.get("storage_prefix")
    storage_secret_name = options.get("storage_secret_name")
    ingress_class_name = options.get("ingress_class_name")
    hostname_template = options.get("hostname_template")
    gateway_api_parent_refs = options.get("gateway_api_parent_refs")
    tls_secret_name = options.get("tls_secret_name")
    preprocessing_resource_class = options.get("preprocessing_resource_class")
    ray_data_storage_class_name = options.get("ray_data_storage_class_name")
    ray_data_pvc_size = options.get("ray_data_pvc_size")
    inference_domain = options.get("inference_domain")
    byoc_provider = options.get("byoc_provider")
    resource_classes = options.get("resource_classes", ())
    api = ByocApi(client)
    resolved_agent_namespace = normalize_agent_namespace(agent_namespace)
    resolved_inference_domain = (
        normalize_inference_domain(inference_domain) if inference_domain else None
    )
    resolved_hostname_template = (
        resolve_inference_hostname_template(hostname_template)
        if resolved_inference_domain is not None or hostname_template is not None
        else None
    )

    if not skip_enable:
        api.enable()

    bundle = api.register_cluster(
        name=name,
        install_storage=install_storage,
        narrow_allowed_ips=narrow_allowed_ips,
        storage_endpoint=storage_endpoint,
        storage_bucket=storage_bucket,
        storage_prefix=storage_prefix,
        storage_secret_name=storage_secret_name,
        ingress_class_name=ingress_class_name,
        hostname_template=resolved_hostname_template,
        inference_domain=resolved_inference_domain,
        gateway_api_parent_refs=gateway_api_parent_refs,
        tls_secret_name=tls_secret_name,
        preprocessing_resource_class=preprocessing_resource_class,
        ray_data_storage_class_name=ray_data_storage_class_name,
        ray_data_pvc_size=ray_data_pvc_size,
        byoc_provider=byoc_provider,
    )

    agent_values = bundle.get("agentHelmValues")
    if not isinstance(agent_values, dict):
        msg = "Registration response missing agentHelmValues."
        raise click.ClickException(msg)

    storage_values = bundle.get("storageHelmValues")
    has_storage_values = isinstance(storage_values, dict)
    upserted = bool(bundle.get("upserted"))
    if install_storage and not has_storage_values:
        if upserted:
            click.echo(
                "Existing MinIO credentials reused — storage chart not reinstalled. "
                "Pass storage flags only if you need to change endpoint, bucket, or secret name."
            )
        else:
            msg = (
                "--install-storage requested MinIO values, but Arena did not return "
                "storageHelmValues."
            )
            raise click.ClickException(msg)

    token = bundle.get("token")
    if upserted and isinstance(token, str) and token:
        click.echo(
            "Cluster token rotated for Helm install (previous token valid during grace)."
        )

    cluster_api_url = bundle.get("clusterApiUrl")
    cluster_name = _registered_cluster_name(bundle, name)

    if upserted:
        click.echo(f"Updated cluster {cluster_name!r}.")
    else:
        click.echo(f"Registered cluster {cluster_name!r}.")
    click.echo(f"Cluster API URL: {cluster_api_url}")

    include_storage = install_storage and has_storage_values
    install_token: str | None = None
    if install:
        install_token = _require_token_for_install(token)
    helm_charts = install or include_storage

    def merged_agent_values(token_value: str | None) -> dict[str, Any]:
        return _agent_values_for_helm(
            agent_values,
            token_value,
            inference_domain=resolved_inference_domain,
            inference_hostname_template=resolved_hostname_template,
        )

    def write_bundle_files(target: Path) -> None:
        helm_values = merged_agent_values(
            _normalize_cluster_token(token) or install_token
        )
        _write_yaml(target / "agent-helm-values.yaml", helm_values, force=force)
        if has_storage_values:
            _write_yaml(
                target / "storage-helm-values.yaml", storage_values, force=force
            )
        if isinstance(token, str) and token:
            _write_token(target / "cluster-token.txt", token, force=force)

    agent_release = (
        resolve_install_release(cluster_name, resolved_agent_namespace)
        if install
        else normalize_agent_release(cluster_name)
    )

    def run_helm_install(work: Path | None) -> None:
        configured = os.environ.get("KUBECONFIG", "").strip()
        kubeconfig = configured or str(Path.home() / ".kube" / "config")
        click.echo(f"Using kubeconfig {kubeconfig}")
        if charts_dir is not None:
            assert work is not None
            _install_charts_from_local_values(
                work,
                include_storage=include_storage,
                charts_dir=charts_dir,
                helm_wait=helm_wait,
                agent_namespace=resolved_agent_namespace,
                agent_release=agent_release,
                install_agent=install,
            )
            return
        _install_charts_from_arena_package(
            api,
            cluster_name=cluster_name,
            agent_values=merged_agent_values(install_token),
            storage_values=storage_values,
            include_storage=include_storage,
            helm_wait=helm_wait,
            agent_namespace=resolved_agent_namespace,
            agent_release=agent_release,
            install_agent=install,
        )

    if no_write:
        if helm_charts:
            work: Path | None
            if charts_dir is not None:
                with tempfile.TemporaryDirectory(prefix="arena-cluster-") as tmp:
                    work = Path(tmp)
                    write_bundle_files(work)
                    run_helm_install(work)
            else:
                run_helm_install(None)
            click.echo("")
            click.echo("Helm install finished (no local config files written).")
    else:
        out = output_dir.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        write_bundle_files(out)
        click.echo(f"Wrote Helm values to {out}/")
        if has_storage_values:
            click.echo("  - storage-helm-values.yaml")
        click.echo("  - agent-helm-values.yaml")
        if isinstance(token, str) and token:
            click.echo("  - cluster-token.txt (mode 0600)")
        if helm_charts:
            run_helm_install(out if charts_dir is not None else None)
            click.echo("")
            click.echo("Helm install finished.")

    created_resource_classes = [
        _ensure_resource_class(
            api, cluster_name=cluster_name, resource_class=resource_class
        )
        for resource_class in resource_classes
    ]

    out_for_steps = output_dir.expanduser().resolve()
    _print_next_steps(
        output_dir=out_for_steps,
        installed=install,
        has_storage_values=has_storage_values,
        storage_installed=include_storage,
        resource_class_names=created_resource_classes,
        agent_namespace=resolved_agent_namespace,
        agent_release=agent_release,
    )
    logger.info("Cluster registration finished for %r.", name)
