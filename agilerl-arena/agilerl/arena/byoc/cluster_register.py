# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Register an enterprise BYOC cluster and write Helm install bundles."""

from __future__ import annotations

import difflib
import logging
import os
import re
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
class ClassAvailability:
    """One region where a resource class can be scheduled."""

    region: str
    zones: tuple[str, ...]
    enabled: bool
    available: int


@dataclass(frozen=True)
class ClusterResourceClass:
    """BYOC resource class created after cluster registration."""

    name: str
    num_nodes: int
    node_selector: dict[str, str]
    metadata: dict[str, object]
    description: str = ""
    availabilities: tuple[ClassAvailability, ...] | None = None
    enabled: bool | None = None


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


def _resource_class_fields(
    num_nodes: object,
    node_selector: object,
    metadata: object,
    description: str,
) -> dict[str, object]:
    """Return the class fields registration compares and updates."""
    selector = node_selector if isinstance(node_selector, dict) else {}
    recorded = metadata if isinstance(metadata, dict) else {}
    fields: dict[str, object] = {
        "num_nodes": num_nodes,
        "node_selector": selector,
        "metadata": recorded,
    }
    if description:
        fields["description"] = description
    return fields


def _parse_byte_size(text: str) -> int | None:
    """Return the byte count of a size string such as ``64 GB`` or ``59.6 GiB``."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([A-Za-z]+)", text.strip())
    if match is None:
        return None
    units = {
        "b": 1,
        "kb": 1_000,
        "mb": 1_000**2,
        "gb": 1_000**3,
        "tb": 1_000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }
    unit = units.get(match.group(2).lower())
    if unit is None:
        return None
    return round(float(match.group(1)) * unit)


def _iec_byte_label(num_bytes: int) -> str:
    """Return the 1-decimal binary size label Arena stores for a byte count."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    size = float(num_bytes)
    exponent = 0
    while size >= 1024 and exponent < 6:
        size /= 1024
        exponent += 1
    return f"{size:.1f} {'KMGTPE'[exponent - 1]}iB"


def _same_byte_size(left: str, right: str) -> bool:
    """Return whether two size strings are the same count after display rounding."""
    left_bytes = _parse_byte_size(left)
    right_bytes = _parse_byte_size(right)
    if left_bytes is None or right_bytes is None:
        return left == right
    return _iec_byte_label(left_bytes) == _iec_byte_label(right_bytes)


def _compute_resource(fields: dict[str, object]) -> dict[str, object] | None:
    metadata = fields.get("metadata")
    if not isinstance(metadata, dict):
        return None
    compute = metadata.get("computeResource")
    if not isinstance(compute, dict):
        return None
    return compute


def _align_same_memory(
    current: dict[str, object], desired: dict[str, object]
) -> dict[str, object]:
    """Keep stored size labels when they are the same byte count as the new ones."""
    current_compute = _compute_resource(current)
    desired_compute = _compute_resource(desired)
    if current_compute is None or desired_compute is None:
        return desired
    aligned = dict(desired_compute)
    for field in ("memoryBytes", "gpuMemoryBytes"):
        current_value = current_compute.get(field)
        desired_value = desired_compute.get(field)
        if not isinstance(current_value, str):
            continue
        if not isinstance(desired_value, str):
            aligned[field] = current_value
            continue
        if desired_value != current_value and _same_byte_size(
            current_value, desired_value
        ):
            aligned[field] = current_value
    if aligned == desired_compute:
        return desired
    metadata = dict(desired["metadata"])
    metadata["computeResource"] = aligned
    return {**desired, "metadata": metadata}


def _resource_class_diff(current: dict[str, object], desired: dict[str, object]) -> str:
    """Return a unified diff of the stored class and the class to register."""
    current_text = yaml.safe_dump(current, sort_keys=True, default_flow_style=False)
    desired_text = yaml.safe_dump(desired, sort_keys=True, default_flow_style=False)
    return "\n".join(
        difflib.unified_diff(
            current_text.splitlines(),
            desired_text.splitlines(),
            fromfile="current",
            tofile="updated",
            lineterm="",
        )
    )


def _enabled_arg(enabled: bool | None) -> dict[str, bool]:
    if enabled is None:
        return {}
    return {"enabled": enabled}


def _availability_body(
    rows: Sequence[ClassAvailability],
) -> list[dict[str, object]]:
    return [
        {"region": row.region, "zones": list(row.zones), "enabled": row.enabled}
        for row in rows
    ]


def _echo_availability(
    name: str, rows: Sequence[ClassAvailability], added: bool
) -> None:
    enabled = [row for row in rows if row.enabled]
    if not enabled:
        click.echo(
            f"Disabled {name!r}. The preset is not offered in any Nebius region."
        )
        return
    listed = ", ".join(f"{row.region} ({row.available})" for row in enabled)
    verb = "Added" if added else "Updated"
    click.echo(f"{verb} availability for {name!r}: {listed}.")


def _stored_class_fields(
    existing: dict[str, object], resource_class: ClusterResourceClass
) -> dict[str, object]:
    """Return comparable fields for a class already stored in Arena."""
    stored_description = existing.get("description")
    current_description = ""
    if resource_class.description and isinstance(stored_description, str):
        current_description = stored_description
    return _resource_class_fields(
        existing.get("num_nodes", existing.get("numNodes")),
        existing.get("node_selector", existing.get("nodeSelector")),
        existing.get("metadata"),
        current_description,
    )


def _update_existing_resource_class(
    api: ByocApi,
    resource_class: ClusterResourceClass,
    existing: dict[str, object],
    desired: dict[str, object],
    availability_body: list[dict[str, object]] | None,
    yes: bool,
) -> str:
    """Update a stored class, or leave it when it already matches."""
    current = _stored_class_fields(existing, resource_class)
    desired_view = _align_same_memory(current, desired)
    availabilities = resource_class.availabilities
    if current == desired_view:
        if availabilities is None:
            click.echo(
                f"Resource class {resource_class.name!r} already matches; "
                "leaving it unchanged."
            )
            return resource_class.name
        api.replace_class_availabilities(
            resource_class.name,
            _availability_body(availabilities),
            **_enabled_arg(resource_class.enabled),
        )
        click.echo(f"Resource class {resource_class.name!r} already matches.")
        _echo_availability(resource_class.name, availabilities, False)
        return resource_class.name
    click.echo(f"Resource class {resource_class.name!r} already exists:")
    click.echo(_resource_class_diff(current, desired_view))
    if not yes and not click.confirm(
        f"Update resource class {resource_class.name!r}?",
        default=False,
    ):
        click.echo(f"Left resource class {resource_class.name!r} unchanged.")
        return resource_class.name
    updated = api.update_class(
        name=resource_class.name,
        num_nodes=resource_class.num_nodes,
        node_selector=resource_class.node_selector,
        metadata=desired_view["metadata"],
        description=resource_class.description,
        availabilities=availability_body,
        **_enabled_arg(resource_class.enabled),
    )
    click.echo(f"Updated resource class {updated.get('name', resource_class.name)!r}.")
    if availabilities is not None:
        _echo_availability(resource_class.name, availabilities, False)
    return resource_class.name


def ensure_resource_class(
    api: ByocApi,
    cluster_name: str,
    resource_class: ClusterResourceClass,
    yes: bool = False,
) -> str:
    """Create the resource class, or update it after showing the diff."""
    desired = _resource_class_fields(
        resource_class.num_nodes,
        resource_class.node_selector,
        resource_class.metadata,
        resource_class.description,
    )
    availabilities = resource_class.availabilities
    availability_body = (
        None if availabilities is None else _availability_body(availabilities)
    )
    existing = api.find_class(resource_class.name)
    if existing is not None:
        return _update_existing_resource_class(
            api,
            resource_class,
            existing,
            desired,
            availability_body,
            yes,
        )
    created = api.create_class(
        name=resource_class.name,
        num_nodes=resource_class.num_nodes,
        cluster_name=cluster_name,
        node_selector=resource_class.node_selector,
        metadata=resource_class.metadata,
        description=resource_class.description,
        availabilities=availability_body,
        **_enabled_arg(resource_class.enabled),
    )
    click.echo(f"Created resource class {created.get('name', resource_class.name)!r}.")
    if availabilities is not None:
        _echo_availability(resource_class.name, availabilities, True)
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
    placement: NotRequired[str | None]
    byoc_provider: NotRequired[dict[str, Any] | None]
    resource_classes: NotRequired[Sequence[ClusterResourceClass]]
    yes: NotRequired[bool]


@dataclass(frozen=True)
class RegisterBundle:
    """Helm values and names from a successful cluster register call."""

    agent_values: dict[str, Any]
    storage_values: object
    cluster_name: str
    token: object
    include_storage: bool
    agent_release: str
    agent_namespace: str
    inference_domain: str | None
    inference_hostname_template: str | None

    @property
    def has_storage_values(self) -> bool:
        """Return whether Arena sent MinIO Helm values."""
        return isinstance(self.storage_values, dict)


def _echo_register_bundle(
    bundle: dict[str, Any],
    name: str,
    install_storage: bool,
) -> tuple[dict[str, Any], object, str]:
    """Validate the register response and print the cluster name."""
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
    cluster_name = _registered_cluster_name(bundle, name)
    if upserted:
        click.echo(f"Updated cluster {cluster_name!r}.")
    else:
        click.echo(f"Registered cluster {cluster_name!r}.")
    click.echo(f"Cluster API URL: {bundle.get('clusterApiUrl')}")
    return agent_values, storage_values, cluster_name


def _write_register_bundle_files(
    target: Path,
    bundle: RegisterBundle,
    install_token: str | None,
    force: bool,
) -> None:
    """Write Helm values and the cluster token under *target*."""
    helm_values = _agent_values_for_helm(
        bundle.agent_values,
        _normalize_cluster_token(bundle.token) or install_token,
        inference_domain=bundle.inference_domain,
        inference_hostname_template=bundle.inference_hostname_template,
    )
    _write_yaml(target / "agent-helm-values.yaml", helm_values, force=force)
    if bundle.has_storage_values:
        _write_yaml(
            target / "storage-helm-values.yaml", bundle.storage_values, force=force
        )
    if isinstance(bundle.token, str) and bundle.token:
        _write_token(target / "cluster-token.txt", bundle.token, force=force)


def _run_register_helm_install(
    api: ByocApi,
    bundle: RegisterBundle,
    options: ClusterRegisterOptions,
    work: Path | None,
    install_token: str | None,
) -> None:
    """Install charts from local values or the Arena install package."""
    configured = os.environ.get("KUBECONFIG", "").strip()
    kubeconfig = configured or str(Path.home() / ".kube" / "config")
    click.echo(f"Using kubeconfig {kubeconfig}")
    charts_dir = options.get("charts_dir")
    helm_wait = options.get("helm_wait", True)
    install = options.get("install", False)
    if charts_dir is not None:
        assert work is not None
        _install_charts_from_local_values(
            work,
            include_storage=bundle.include_storage,
            charts_dir=charts_dir,
            helm_wait=helm_wait,
            agent_namespace=bundle.agent_namespace,
            agent_release=bundle.agent_release,
            install_agent=install,
        )
        return
    _install_charts_from_arena_package(
        api,
        cluster_name=bundle.cluster_name,
        agent_values=_agent_values_for_helm(
            bundle.agent_values,
            install_token,
            inference_domain=bundle.inference_domain,
            inference_hostname_template=bundle.inference_hostname_template,
        ),
        storage_values=bundle.storage_values,
        include_storage=bundle.include_storage,
        helm_wait=helm_wait,
        agent_namespace=bundle.agent_namespace,
        agent_release=bundle.agent_release,
        install_agent=install,
    )


def _persist_and_install_registration(
    api: ByocApi,
    bundle: RegisterBundle,
    options: ClusterRegisterOptions,
) -> None:
    """Write register files and optionally install Helm charts."""
    install = options.get("install", False)
    install_token = _require_token_for_install(bundle.token) if install else None
    helm_charts = install or bundle.include_storage
    force = options["force"]
    charts_dir = options.get("charts_dir")
    no_write = options.get("no_write", False)
    if no_write:
        if helm_charts:
            if charts_dir is not None:
                with tempfile.TemporaryDirectory(prefix="arena-cluster-") as tmp:
                    work = Path(tmp)
                    _write_register_bundle_files(work, bundle, install_token, force)
                    _run_register_helm_install(
                        api, bundle, options, work, install_token
                    )
            else:
                _run_register_helm_install(api, bundle, options, None, install_token)
            click.echo("")
            click.echo("Helm install finished (no local config files written).")
        return
    out = options["output_dir"].expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_register_bundle_files(out, bundle, install_token, force)
    click.echo(f"Wrote Helm values to {out}/")
    if bundle.has_storage_values:
        click.echo("  - storage-helm-values.yaml")
    click.echo("  - agent-helm-values.yaml")
    if isinstance(bundle.token, str) and bundle.token:
        click.echo("  - cluster-token.txt (mode 0600)")
    if helm_charts:
        work = out if charts_dir is not None else None
        _run_register_helm_install(api, bundle, options, work, install_token)
        click.echo("")
        click.echo("Helm install finished.")


def _resolved_register_inference(
    options: ClusterRegisterOptions,
) -> tuple[str | None, str | None]:
    """Return the normalized inference domain and hostname template."""
    inference_domain = options.get("inference_domain")
    hostname_template = options.get("hostname_template")
    resolved_inference_domain = (
        normalize_inference_domain(inference_domain) if inference_domain else None
    )
    resolved_hostname_template = (
        resolve_inference_hostname_template(hostname_template)
        if resolved_inference_domain is not None or hostname_template is not None
        else None
    )
    return resolved_inference_domain, resolved_hostname_template


def _submit_cluster_register(
    api: ByocApi,
    options: ClusterRegisterOptions,
    hostname_template: str | None,
    inference_domain: str | None,
) -> dict[str, Any]:
    """Register or update the cluster and return the Arena response."""
    return api.register_cluster(
        name=options["name"],
        install_storage=options.get("install_storage", False),
        narrow_allowed_ips=options.get("narrow_allowed_ips", False),
        storage_endpoint=options.get("storage_endpoint"),
        storage_bucket=options.get("storage_bucket"),
        storage_prefix=options.get("storage_prefix"),
        storage_secret_name=options.get("storage_secret_name"),
        ingress_class_name=options.get("ingress_class_name"),
        hostname_template=hostname_template,
        inference_domain=inference_domain,
        gateway_api_parent_refs=options.get("gateway_api_parent_refs"),
        tls_secret_name=options.get("tls_secret_name"),
        preprocessing_resource_class=options.get("preprocessing_resource_class"),
        ray_data_storage_class_name=options.get("ray_data_storage_class_name"),
        ray_data_pvc_size=options.get("ray_data_pvc_size"),
        placement=options.get("placement"),
        byoc_provider=options.get("byoc_provider"),
    )


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
    install = options.get("install", False)
    install_storage = options.get("install_storage", False)
    api = ByocApi(client)
    resolved_agent_namespace = normalize_agent_namespace(
        options.get("agent_namespace", AGENT_NAMESPACE)
    )
    resolved_inference_domain, resolved_hostname_template = (
        _resolved_register_inference(options)
    )
    if not skip_enable:
        api.enable()
    bundle = _submit_cluster_register(
        api, options, resolved_hostname_template, resolved_inference_domain
    )
    agent_values, storage_values, cluster_name = _echo_register_bundle(
        bundle, name, install_storage
    )
    include_storage = install_storage and isinstance(storage_values, dict)
    agent_release = (
        resolve_install_release(cluster_name, resolved_agent_namespace)
        if install
        else normalize_agent_release(cluster_name)
    )
    register_bundle = RegisterBundle(
        agent_values=agent_values,
        storage_values=storage_values,
        cluster_name=cluster_name,
        token=bundle.get("token"),
        include_storage=include_storage,
        agent_release=agent_release,
        agent_namespace=resolved_agent_namespace,
        inference_domain=resolved_inference_domain,
        inference_hostname_template=resolved_hostname_template,
    )
    _persist_and_install_registration(api, register_bundle, options)
    created_resource_classes = [
        ensure_resource_class(
            api,
            cluster_name=cluster_name,
            resource_class=resource_class,
            yes=options.get("yes", False),
        )
        for resource_class in options.get("resource_classes", ())
    ]
    _print_next_steps(
        output_dir=output_dir.expanduser().resolve(),
        installed=install,
        has_storage_values=register_bundle.has_storage_values,
        storage_installed=include_storage,
        resource_class_names=created_resource_classes,
        agent_namespace=resolved_agent_namespace,
        agent_release=agent_release,
    )
    logger.info("Cluster registration finished for %r.", name)
