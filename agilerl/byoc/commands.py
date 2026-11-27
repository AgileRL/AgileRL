# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The hardcoded ``arena cluster`` Click commands."""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

import click

from agilerl.byoc.api import ByocApi
from agilerl.byoc.cluster_connect import (
    ConnectOptions,
    registered_cluster,
    run_cluster_connect,
)
from agilerl.byoc.cluster_helm import (
    AGENT_NAMESPACE,
    AGENT_RELEASE,
    STORAGE_RELEASE,
    helm_available,
    list_installed_byoc_releases,
    normalize_agent_namespace,
    uninstall_byoc_releases,
)
from agilerl.byoc.cluster_manifest import MANIFEST_KEYS, run_cluster_manifest
from agilerl.byoc.cluster_register import (
    ClassAvailability,
    ClusterResourceClass,
    ensure_resource_class,
    run_cluster_register,
)
from agilerl.byoc.cluster_smoke import run_cluster_smoke
from agilerl.byoc.cluster_token import (
    run_cluster_revoke_token,
    run_cluster_rotate_token,
)
from agilerl.byoc.cluster_upgrade import run_cluster_upgrade
from agilerl.byoc.kube_target import (
    ClickDecorated,
    KubeTarget,
    NoKubeContextError,
    kube_target,
    kube_target_options,
    kubeconfig_environment,
    require_registered_cluster,
    require_same_cluster,
)
from agilerl.byoc.preflight import (
    any_failed,
    echo_checks,
    own_agent_release,
    preflight_report,
    require_passed,
    run_preflight,
)
from agilerl.byoc.provisioning.inference import normalize_inference_domain
from agilerl.byoc.provisioning.module_catalog import (
    open_module_catalog,
    read_module_pin,
    selected_module_version,
)
from agilerl.byoc.provisioning.providers import get_provider, provider_names
from agilerl.byoc.provisioning.providers.base import (
    CreatedProjectsQuery,
    EnsureGateway,
)
from agilerl.byoc.provisioning.providers.nebius.ephemeral import (
    ensure_ephemeral_regions,
)
from agilerl.byoc.provisioning.providers.nebius.platforms import (
    GpuPreset,
    append_worker_blocks,
    cuda_drivers_preset,
    list_capacity_advice,
    list_registerable_gpu_presets,
    project_id_from_outputs,
    region_capacity_for_preset,
    resolve_gpu_presets,
    worker_for_preset,
    worker_spec_for_preset,
)
from agilerl.byoc.provisioning.providers.nebius.spec import (
    WorkerNodeGroupSpec,
    attach_catalog_credentials,
    attach_service_account_credentials,
    resource_classes,
)
from agilerl.byoc.provisioning.provisioner import ClusterProvisioner
from agilerl.byoc.provisioning.spec import (
    ClusterSpec,
    apply_cluster_spec_overrides,
    default_cluster_name,
    load_cluster_spec,
    write_default_cluster_spec,
)
from agilerl.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
)
from agilerl.cli_manifest import _parse_json_cli_value
from agilerl.config import CommandConfig, arena_client
from agilerl.exceptions import ArenaAPIError
from agilerl.output import json_output_enabled

logger = logging.getLogger("agilerl.byoc")
SENSITIVE_STATUS_OUTPUTS = frozenset(
    {
        "service_account_private_key",
        "storage_secret_access_key",
        "catalog_secret_access_key",
    }
)


def _optional_text(value: object) -> str | None:
    """Return ``value`` when it is a string."""
    return value if isinstance(value, str) else None


def _cluster_provisioner(
    config: CommandConfig,
    spec: ClusterSpec,
    state_dir: Path,
    module_version: str | None = None,
) -> ClusterProvisioner:
    """Build a provisioner that installs the selected module release."""
    version = selected_module_version(state_dir, spec.module_version, module_version)
    catalog = open_module_catalog(
        config.module_catalog_url, config.module_catalog_public_key
    )
    return ClusterProvisioner(
        runner=TerraformRunner(module_version=version, catalog=catalog)
    )


def _state_bucket_should_be_created(spec: ClusterSpec, flag: bool) -> bool:
    """Return whether a missing Terraform state bucket should be created."""
    return flag or spec.terraform_state.create_bucket


def _recover_spec(config: CommandConfig, spec: ClusterSpec) -> ClusterSpec:
    """Fill omitted cloud ids from the cluster stored in Arena."""
    provider = get_provider(spec.provider)
    if not provider.needs_stored_config(spec):
        return spec
    with arena_client(config) as client:
        try:
            stored = ByocApi(client).stored_provider_config(spec.name, provider.name)
        except ArenaAPIError as exc:
            # Arena answers 404 until the org's BYOC provider is enabled.
            if exc.status_code != HTTPStatus.NOT_FOUND:
                raise
            return spec
    if stored is None:
        return spec
    recovered = provider.recover(spec, stored)
    if recovered != spec:
        click.echo(
            f"Using {provider.name} settings stored in Arena for cluster {spec.name!r}."
        )
    return recovered


def _register_provisioned_cluster(config: CommandConfig, **fields: object) -> None:
    """Register a cluster using values Terraform just wrote."""
    registration = cast("dict[str, object]", fields["registration"])
    outputs = cast("ClusterOutputs", fields["outputs"])
    cluster_spec = cast("ClusterSpec", fields["cluster_spec"])
    output_dir = cast("Path", fields["output_dir"])
    install = cast("bool", fields["install"])
    agent_namespace = cast("str", fields["agent_namespace"])
    skip_enable = cast("bool", fields.get("skip_enable", False))
    force = cast("bool", fields.get("force", False))
    install_storage = cast("bool | None", fields.get("install_storage"))
    narrow_allowed_ips = cast("bool", fields.get("narrow_allowed_ips", False))
    no_write = cast("bool", fields.get("no_write", False))
    charts_dir = cast("Path | None", fields.get("charts_dir"))
    helm_wait = cast("bool", fields.get("helm_wait", True))
    storage_install = (
        bool(registration["install_storage"])
        if install_storage is None
        else install_storage
    )
    with (
        _register_target(
            str(registration["name"]),
            outputs.kubeconfig_path,
            cast("str | None", fields.get("kube_context")),
            touches_cluster=install or storage_install,
        ) as target,
        arena_client(config) as client,
    ):
        run_cluster_register(
            client,
            name=str(registration["name"]),
            output_dir=output_dir,
            skip_enable=skip_enable,
            force=force,
            install=install,
            install_storage=storage_install,
            narrow_allowed_ips=narrow_allowed_ips,
            no_write=no_write,
            charts_dir=charts_dir,
            helm_wait=helm_wait,
            agent_namespace=agent_namespace,
            storage_endpoint=registration["storage_endpoint"]
            or outputs.storage_endpoint,
            storage_bucket=registration["storage_bucket"] or outputs.storage_bucket,
            storage_prefix=registration["storage_prefix"],
            storage_secret_name=registration["storage_secret_name"],
            ingress_class_name=registration["ingress_class_name"],
            gateway_api_parent_refs=outputs.gateway_api_parent_refs
            or registration["gateway_api_parent_refs"],
            preprocessing_resource_class=registration["preprocessing_resource_class"],
            ray_data_storage_class_name=registration["ray_data_storage_class_name"]
            or outputs.storage_class_name,
            ray_data_pvc_size=registration["ray_data_pvc_size"],
            hostname_template=registration["hostname_template"],
            tls_secret_name=outputs.inference_tls_secret_name,
            inference_domain=registration["inference_domain"],
            placement=cast("str", registration["placement"]),
            byoc_scope=cast("str | None", registration.get("byoc_scope")),
            byoc_provider=_byoc_provider_with_module(
                cluster_spec,
                outputs,
                cast("Path | None", fields.get("state_dir")),
            ),
            resource_classes=get_provider(cluster_spec.provider).resource_classes(
                cluster_spec,
                worker_node_group_ids=outputs.worker_node_group_ids,
            ),
            yes=cast("bool", fields.get("yes", False)),
            cluster_uid=None if target is None else target.cluster_uid,
            # The cluster was just provisioned, so it holds no agent token yet.
            rotate_token=True,
        )


@contextmanager
def _register_target(
    name: str, kubeconfig: Path | None, context: str | None, touches_cluster: bool
) -> Iterator[KubeTarget | None]:
    """Pin kubectl and Helm to the cluster register installs on.

    Yields ``None`` without resolving a context when nothing is installed.
    """
    if not touches_cluster:
        yield None
        return
    with (
        kube_target(name, kubeconfig, context) as target,
        kubeconfig_environment(target.kubeconfig),
    ):
        yield target


def _byoc_provider_with_module(
    cluster_spec: ClusterSpec,
    outputs: ClusterOutputs,
    state_dir: Path | None,
) -> dict[str, object]:
    """Return the registration body, including the installed module pin."""
    payload = get_provider(cluster_spec.provider).registration_payload(
        cluster_spec,
        project_id=outputs.project_id,
        storage_project_id=outputs.storage_project_id,
    )
    config = payload.get("config")
    if (
        isinstance(config, dict)
        and outputs.service_account_key
        and outputs.service_account_profile
    ):
        config["service_account_key"] = outputs.service_account_key
        config["profile"] = outputs.service_account_profile
    if (
        isinstance(config, dict)
        and outputs.catalog_access_key_id
        and outputs.catalog_secret_access_key
        and outputs.catalog_group_id
    ):
        config["catalog_access_key_id"] = outputs.catalog_access_key_id
        config["catalog_secret_access_key"] = outputs.catalog_secret_access_key
        config["catalog_group_id"] = outputs.catalog_group_id
    if state_dir is None:
        return payload
    pin = read_module_pin(state_dir)
    if pin is not None and isinstance(config, dict):
        config["module"] = pin.public_record()
    return payload


def _registration_values_from_spec(
    spec: ClusterSpec, **overrides: object
) -> tuple[ClusterSpec, dict[str, object]]:
    """Resolve spec registration values and explicit command-line overrides."""
    name = cast("str | None", overrides.get("name"))
    storage_endpoint = cast("str | None", overrides.get("storage_endpoint"))
    storage_bucket = cast("str | None", overrides.get("storage_bucket"))
    storage_prefix = cast("str | None", overrides.get("storage_prefix"))
    storage_secret_name = cast("str | None", overrides.get("storage_secret_name"))
    ingress_class_name = cast("str | None", overrides.get("ingress_class_name"))
    gateway_api_parent_refs = overrides.get("gateway_api_parent_refs")
    preprocessing_resource_class = cast(
        "str | None", overrides.get("preprocessing_resource_class")
    )
    ray_data_storage_class_name = cast(
        "str | None", overrides.get("ray_data_storage_class_name")
    )
    ray_data_pvc_size = cast("str | None", overrides.get("ray_data_pvc_size"))
    domain = cast("str | None", overrides.get("domain"))
    hostname_template = cast("str | None", overrides.get("hostname_template"))
    tls_secret_name = cast("str | None", overrides.get("tls_secret_name"))
    placement = cast("str | None", overrides.get("placement"))
    byoc_scope = cast("str | None", overrides.get("byoc_scope"))
    refs = gateway_api_parent_refs
    if isinstance(refs, str):
        refs = _parse_json_cli_value(refs)
    if refs is not None and not isinstance(refs, list):
        msg = "--gateway-api-parent-refs must contain a JSON list."
        raise click.ClickException(msg)
    resolved = apply_cluster_spec_overrides(
        spec,
        name=name,
        storage_endpoint=storage_endpoint,
        storage_bucket=storage_bucket,
        storage_prefix=storage_prefix,
        storage_secret_name=storage_secret_name,
        ingress_class_name=ingress_class_name,
        gateway_api_parent_refs=refs,
        preprocessing_resource_class=preprocessing_resource_class,
        ray_data_storage_class_name=ray_data_storage_class_name,
        ray_data_pvc_size=ray_data_pvc_size,
        domain=domain,
        hostname_template=hostname_template,
        tls_secret_name=tls_secret_name,
        placement=placement,
        byoc_scope=byoc_scope,
    )
    arena = resolved.arena
    return resolved, {
        "name": resolved.name,
        "storage_endpoint": arena.storage.endpoint,
        "storage_bucket": arena.storage.bucket,
        "storage_prefix": arena.storage.prefix,
        "storage_secret_name": arena.storage.resolved_secret_name(),
        "install_storage": arena.storage.install,
        "ingress_class_name": arena.gateway.ingress_class_name,
        "hostname_template": arena.inference.hostname_template,
        "inference_domain": arena.inference.domain,
        "gateway_api_parent_refs": arena.gateway.parent_refs,
        "tls_secret_name": arena.inference.tls_secret_name,
        "preprocessing_resource_class": arena.workloads.preprocessing_resource_class,
        "ray_data_storage_class_name": arena.workloads.ray_data_storage_class_name,
        "ray_data_pvc_size": arena.workloads.ray_data_pvc_size,
        "placement": resolved.values.placement,
        "byoc_scope": resolved.values.byoc_scope,
        "byoc_provider": get_provider(resolved.provider).registration_payload(resolved),
    }


def _apply_verbosity(verbose: bool) -> None:
    """``--verbose`` raises the Arena logger to DEBUG so command traces and live
    per-stage script output are shown instead of hidden.

    :param verbose: If ``True``, set the ``agilerl`` logger to DEBUG.
    :type verbose: bool
    :returns: None
    :rtype: None
    """
    if verbose:
        logging.getLogger("agilerl").setLevel(logging.DEBUG)


REGISTER_OVERRIDE_KEYS = (
    "name",
    "storage_endpoint",
    "storage_bucket",
    "storage_prefix",
    "storage_secret_name",
    "ingress_class_name",
    "gateway_api_parent_refs",
    "preprocessing_resource_class",
    "ray_data_storage_class_name",
    "ray_data_pvc_size",
    "domain",
    "hostname_template",
    "tls_secret_name",
    "placement",
    "byoc_scope",
)

REGISTER_RUNTIME_KEYS = (
    "storage_endpoint",
    "storage_bucket",
    "storage_prefix",
    "storage_secret_name",
    "ingress_class_name",
    "hostname_template",
    "inference_domain",
    "gateway_api_parent_refs",
    "tls_secret_name",
    "preprocessing_resource_class",
    "ray_data_storage_class_name",
    "ray_data_pvc_size",
    "placement",
    "byoc_scope",
    "byoc_provider",
)

REGISTER_COMMAND_FLAGS = (
    (
        ("--output-dir",),
        {
            "default": None,
            "help": "Directory for Helm values and token (default: ./arena-cluster-NAME).",
        },
    ),
    (
        ("--skip-enable",),
        {
            "is_flag": True,
            "default": False,
            "help": "Skip enabling the BYOC provider (use when it is already enabled).",
        },
    ),
    (
        ("--force",),
        {
            "is_flag": True,
            "default": False,
            "help": "Overwrite existing output files.",
        },
    ),
    (
        ("--install",),
        {
            "is_flag": True,
            "default": False,
            "help": "Run helm upgrade --install for the agent chart.",
        },
    ),
    (
        ("--install-storage",),
        {
            "is_flag": True,
            "default": False,
            "help": "Mint bundled MinIO values and install the storage chart.",
        },
    ),
    (
        ("--narrow-allowed-ips",),
        {
            "is_flag": True,
            "default": False,
            "help": "Omit 10.0.0.0/8 from WireGuard AllowedIPs (k3d/k3s ClusterIP).",
        },
    ),
    (
        ("--lab",),
        {
            "is_flag": True,
            "default": False,
            "help": "Lab k3d profile: same as --install-storage --narrow-allowed-ips.",
        },
    ),
    (
        ("--no-write",),
        {
            "is_flag": True,
            "default": False,
            "help": "Do not persist Helm values or token files under --output-dir.",
        },
    ),
    (
        ("--rotate-token",),
        {
            "is_flag": True,
            "default": False,
            "help": (
                "Issue a new agent token for a cluster that is already registered. "
                "The previous token keeps working for 24 hours."
            ),
        },
    ),
    (
        ("--charts-dir",),
        {
            "default": None,
            "help": (
                "Platform development only: path to "
                "agilerl-platform/resources/helm-setup instead of downloading "
                "charts from Arena."
            ),
        },
    ),
    (
        ("--no-helm-wait",),
        {
            "is_flag": True,
            "default": False,
            "help": "Do not wait for Helm storage resources or the agent Deployment.",
        },
    ),
    (
        ("--agent-namespace",),
        {
            "default": AGENT_NAMESPACE,
            "show_default": True,
            "help": "Kubernetes namespace for the agent Helm release.",
        },
    ),
    (
        ("--state-dir",),
        {
            "type": click.Path(file_okay=False, path_type=Path),
            "default": None,
            "help": "Terraform state directory (default: ./arena-cluster-NAME/terraform).",
        },
    ),
    (
        ("--module-version",),
        {
            "default": None,
            "help": (
                "Terraform module release to install. Upgrades a cluster that "
                "already has a pin."
            ),
        },
    ),
    (
        ("--create-state-bucket",),
        {
            "is_flag": True,
            "default": False,
            "help": (
                "Create the Terraform state bucket when provisioning a missing "
                "cluster. terraform_state.create_bucket in the spec does the same."
            ),
        },
    ),
    (
        ("--yes",),
        {
            "is_flag": True,
            "default": False,
            "help": "Provision and create suggested classes without prompting.",
        },
    ),
    (
        ("-v", "--verbose"),
        {
            "is_flag": True,
            "default": False,
            "help": "Show detailed command traces.",
        },
    ),
)


def _cluster_register_options(command: click.Command) -> click.Command:
    """Add register flags that plan and provision do not share."""
    command = click.option(
        "--spec",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Cloud cluster YAML file that supplies registration settings.",
    )(command)
    command = _cluster_registration_options(command)
    command = kube_target_options(command)
    for declaration, kwargs in REGISTER_COMMAND_FLAGS:
        command = click.option(*declaration, **kwargs)(command)
    return click.argument(
        "provider",
        required=False,
        default=None,
        type=click.Choice(provider_names(), case_sensitive=False),
    )(command)


def _apply_lab_flags(options: dict[str, object]) -> dict[str, object]:
    """Turn ``--lab`` into MinIO install plus the k3d AllowedIPs profile."""
    resolved = dict(options)
    if resolved["lab"]:
        resolved["install_storage"] = True
        resolved["narrow_allowed_ips"] = True
    return resolved


def _assert_provider_matches_spec(
    provider: str | None, cluster_spec: ClusterSpec | None
) -> None:
    """Reject a provider argument that does not match the loaded spec."""
    if provider is None:
        return
    if cluster_spec is None:
        msg = "--spec is required when a provider is given."
        raise click.ClickException(msg)
    if provider.lower() != cluster_spec.provider:
        msg = (
            f"Provider {provider!r} does not match spec provider "
            f"{cluster_spec.provider!r}."
        )
        raise click.ClickException(msg)


def _load_register_spec(
    config: CommandConfig, spec_path: Path, options: dict[str, object]
) -> tuple[ClusterSpec, dict[str, object]]:
    """Load, override, and recover a cluster spec for register."""
    try:
        cluster_spec = load_cluster_spec(spec_path)
        _, registration = _registration_values_from_spec(
            cluster_spec,
            **{key: options[key] for key in REGISTER_OVERRIDE_KEYS},
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    return _recover_spec(config, cluster_spec), registration


def _resolve_register_target(
    config: CommandConfig, options: dict[str, object]
) -> tuple[str, ClusterSpec | None, dict[str, object] | None]:
    """Return the cluster name, spec, and registration values for this invocation."""
    spec_path = cast("Path | None", options["spec"])
    name = cast("str | None", options["name"])
    provider = cast("str | None", options["provider"])
    cluster_spec: ClusterSpec | None = None
    registration: dict[str, object] | None = None
    if spec_path is not None:
        cluster_spec, registration = _load_register_spec(config, spec_path, options)
        cluster_name = str(registration["name"])
    elif name:
        cluster_name = name.strip()
    else:
        msg = "--name is required when --spec is not provided."
        raise click.ClickException(msg)
    _assert_provider_matches_spec(provider, cluster_spec)
    if cluster_spec is not None and registration is not None:
        registration["byoc_provider"] = get_provider(
            cluster_spec.provider
        ).registration_payload(cluster_spec)
    return cluster_name, cluster_spec, registration


def _register_state_dir(cluster_spec: ClusterSpec, options: dict[str, object]) -> Path:
    """Resolve the Terraform state directory for a spec-backed register."""
    state_dir = cast("Path | None", options["state_dir"])
    return absolute_path(
        state_dir or Path(f"./arena-cluster-{cluster_spec.name}/terraform")
    )


def _provision_missing_cluster(
    config: CommandConfig,
    cluster_spec: ClusterSpec | None,
    registration: dict[str, object] | None,
    options: dict[str, object],
    output_dir: Path,
    charts: Path | None,
) -> bool:
    """Provision and register a cluster that is not in Terraform state."""
    if cluster_spec is None or registration is None:
        return False
    resolved_state = _register_state_dir(cluster_spec, options)
    provisioner = _cluster_provisioner(
        config,
        cluster_spec,
        resolved_state,
        _optional_text(options.get("module_version")),
    )
    if provisioner.cluster_exists(cluster_spec, state_dir=resolved_state):
        return False
    yes = cast("bool", options["yes"])
    if not yes and not click.confirm(
        f"Provision {cluster_spec.provider} cluster {cluster_spec.name!r}?",
        default=False,
    ):
        click.echo("Aborted.")
        return True
    outputs = provisioner.provision(
        cluster_spec,
        state_dir=resolved_state,
        output_dir=output_dir,
        create_state_bucket=_state_bucket_should_be_created(
            cluster_spec, cast("bool", options["create_state_bucket"])
        ),
    )
    click.echo(f"Kubeconfig written to {outputs.kubeconfig_path}")
    _register_provisioned_cluster(
        config,
        registration=registration,
        outputs=outputs,
        cluster_spec=cluster_spec,
        output_dir=output_dir,
        install=cast("bool", options["install"]),
        agent_namespace=cast("str", options["agent_namespace"]),
        skip_enable=cast("bool", options["skip_enable"]),
        force=cast("bool", options["force"]),
        install_storage=cast("bool", options["install_storage"])
        or bool(registration["install_storage"]),
        narrow_allowed_ips=cast("bool", options["narrow_allowed_ips"]),
        no_write=cast("bool", options["no_write"]),
        charts_dir=charts,
        helm_wait=not cast("bool", options["no_helm_wait"]),
        state_dir=resolved_state,
        kube_context=options["kube_context"],
    )
    return True


def _kubeconfig_for_register_install(
    config: CommandConfig,
    cluster_spec: ClusterSpec | None,
    registration: dict[str, object] | None,
    options: dict[str, object],
    output_dir: Path,
) -> Path | None:
    """Return the kubeconfig Helm should use when installing onto this cluster."""
    if cluster_spec is None or registration is None:
        return None
    install = cast("bool", options["install"])
    install_storage = cast("bool", options["install_storage"])
    if not (install or install_storage or bool(registration["install_storage"])):
        return None
    kubeconfig = output_dir / "kubeconfig"
    if kubeconfig.is_file():
        return kubeconfig
    state_dir = _register_state_dir(cluster_spec, options)
    return _cluster_provisioner(
        config,
        cluster_spec,
        state_dir,
        _optional_text(options.get("module_version")),
    ).write_kubeconfig(
        cluster_spec,
        state_dir=state_dir,
        output_dir=output_dir,
    )


def _register_runtime_fields(
    registration: dict[str, object] | None, options: dict[str, object]
) -> dict[str, object]:
    """Resolve storage and inference fields for an existing-cluster register."""
    if registration is not None:
        return {key: registration[key] for key in REGISTER_RUNTIME_KEYS}
    gateway_refs: object | None = None
    gateway_api_parent_refs = cast("str | None", options["gateway_api_parent_refs"])
    if gateway_api_parent_refs is not None:
        gateway_refs = _parse_json_cli_value(gateway_api_parent_refs)
    inference_domain = None
    domain = cast("str | None", options["domain"])
    if domain is not None:
        try:
            inference_domain = normalize_inference_domain(domain)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    fields = {key: options.get(key) for key in REGISTER_RUNTIME_KEYS}
    fields["gateway_api_parent_refs"] = gateway_refs
    fields["inference_domain"] = inference_domain
    fields["byoc_provider"] = None
    if fields.get("placement") is None:
        fields["placement"] = (
            "ephemeral" if fields.get("byoc_scope") == "platform" else "in-cluster"
        )
    return fields


def _terraform_outputs_for_register(
    config: CommandConfig,
    cluster_spec: ClusterSpec,
    options: dict[str, object],
) -> dict[str, object]:
    """Read Terraform outputs for a cluster that is already provisioned."""
    state_dir = _register_state_dir(cluster_spec, options)
    return _cluster_provisioner(
        config,
        cluster_spec,
        state_dir,
        _optional_text(options.get("module_version")),
    ).status(cluster_spec, state_dir=state_dir)


def _resource_classes_from_outputs(
    cluster_spec: ClusterSpec, values: dict[str, object]
) -> list[ClusterResourceClass]:
    """Build GPU resource classes from the provisioned worker node groups."""
    raw = values.get("worker_node_group_ids")
    if raw is None:
        msg = (
            "Terraform state has no worker_node_group_ids output. "
            "Finish provisioning the cluster before registering it."
        )
        raise click.ClickException(msg)
    if not isinstance(raw, dict):
        msg = "Terraform output worker_node_group_ids must be a string map."
        raise click.ClickException(msg)
    worker_node_group_ids: dict[str, str] = {}
    for name, group_id in raw.items():
        if not isinstance(name, str) or not isinstance(group_id, str):
            msg = "Terraform output worker_node_group_ids must be a string map."
            raise click.ClickException(msg)
        worker_node_group_ids[name] = group_id
    return get_provider(cluster_spec.provider).resource_classes(
        cluster_spec, worker_node_group_ids=worker_node_group_ids
    )


def _ensure_post_apply(
    config: CommandConfig,
    cluster_spec: ClusterSpec | None,
    options: dict[str, object],
    output_dir: Path,
) -> None:
    """Run cluster setup when Terraform apply finished and setup did not."""
    if cluster_spec is None:
        return
    resolved_state = _register_state_dir(cluster_spec, options)
    _cluster_provisioner(
        config,
        cluster_spec,
        resolved_state,
        _optional_text(options.get("module_version")),
    ).ensure_post_apply(
        cluster_spec,
        state_dir=resolved_state,
        output_dir=output_dir,
    )


def _ensure_registered_ephemeral_regions(
    cluster_spec: ClusterSpec | None,
    registration: dict[str, object] | None,
) -> None:
    """Ensure hunt-region IAM when registering an ephemeral Nebius cluster."""
    if cluster_spec is None or cluster_spec.provider != "nebius":
        return
    placement = cluster_spec.values.placement
    registered = None if registration is None else registration.get("placement")
    if isinstance(registered, str):
        placement = registered
    if placement != "ephemeral":
        return
    if cluster_spec.values.placement != "ephemeral":
        cluster_spec = cluster_spec.model_copy(
            update={
                "values": cluster_spec.values.model_copy(
                    update={"placement": "ephemeral"}
                )
            }
        )
    ensure_ephemeral_regions(cluster_spec)


def _require_same_registered_cluster(
    config: CommandConfig,
    cluster_name: str,
    fields: dict[str, object],
    target: KubeTarget | None,
) -> None:
    """Stop before gateway setup or Helm when Arena stores another cluster UID.

    A name Arena does not have yet is a first registration and goes ahead.
    Without a target the command does not touch the cluster.
    """
    if target is None:
        return
    with arena_client(config) as client:
        try:
            registered = ByocApi(
                client, platform=fields.get("byoc_scope") == "platform"
            ).find_cluster(cluster_name)
        except ArenaAPIError as exc:
            # Arena answers 404 until the org's BYOC provider is enabled, which register does.
            if exc.status_code != HTTPStatus.NOT_FOUND:
                raise
            return
    if registered is not None:
        require_same_cluster(registered, cluster_name, target)


def _ensure_spec_gateway(
    cluster_spec: ClusterSpec,
    target: KubeTarget,
    values: dict[str, object] | None,
    spec_kubeconfig: Path | None,
    fields: dict[str, object],
) -> None:
    """Set up the spec's gateway on *target* and record its parent refs in *fields*."""
    if not cluster_spec.arena.gateway.enable:
        return
    parent_refs = get_provider(cluster_spec.provider).ensure_gateway(
        EnsureGateway(
            cluster_spec,
            target.kubeconfig,
            values or {},
            durable_kubeconfig=spec_kubeconfig,
        )
    )
    if parent_refs is not None:
        fields["gateway_api_parent_refs"] = parent_refs


def _register_existing_cluster(
    config: CommandConfig,
    cluster_name: str,
    cluster_spec: ClusterSpec | None,
    registration: dict[str, object] | None,
    options: dict[str, object],
    output_dir: Path,
    charts: Path | None,
) -> None:
    """Upsert an already provisioned cluster and optionally install Helm charts."""
    _ensure_registered_ephemeral_regions(cluster_spec, registration)
    fields = _register_runtime_fields(registration, options)
    values: dict[str, object] | None = None
    resource_classes: list[ClusterResourceClass] | tuple[()] = ()
    if cluster_spec is not None:
        values = _terraform_outputs_for_register(config, cluster_spec, options)
        resource_classes = _resource_classes_from_outputs(cluster_spec, values)

        def output_text(key: str) -> str | None:
            raw = values.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
            return None

        if isinstance(fields.get("byoc_provider"), dict):
            fields["byoc_provider"] = get_provider(
                cluster_spec.provider
            ).registration_payload(
                cluster_spec,
                project_id=output_text("project_id"),
                storage_project_id=output_text("storage_project_id"),
            )
            attach_service_account_credentials(fields["byoc_provider"], values)
            attach_catalog_credentials(fields["byoc_provider"], values)
    spec_kubeconfig = _kubeconfig_for_register_install(
        config, cluster_spec, registration, options, output_dir
    )
    install = cast("bool", options["install"])
    install_storage = cast("bool", options["install_storage"]) or bool(
        registration is not None and registration["install_storage"]
    )
    with _register_target(
        cluster_name,
        cast("Path | None", options["kubeconfig"]) or spec_kubeconfig,
        cast("str | None", options["kube_context"]),
        touches_cluster=install or install_storage,
    ) as target:
        _require_same_registered_cluster(config, cluster_name, fields, target)
        if target is not None and cluster_spec is not None:
            _ensure_spec_gateway(cluster_spec, target, values, spec_kubeconfig, fields)
        with arena_client(config) as client:
            run_cluster_register(
                client,
                name=cluster_name,
                output_dir=output_dir,
                skip_enable=cast("bool", options["skip_enable"]),
                force=cast("bool", options["force"]),
                install=install,
                install_storage=install_storage,
                narrow_allowed_ips=cast("bool", options["narrow_allowed_ips"]),
                no_write=cast("bool", options["no_write"]),
                charts_dir=charts,
                helm_wait=not cast("bool", options["no_helm_wait"]),
                agent_namespace=cast("str", options["agent_namespace"]),
                resource_classes=resource_classes,
                yes=cast("bool", options.get("yes", False)),
                rotate_token=cast("bool", options["rotate_token"]),
                cluster_uid=None if target is None else target.cluster_uid,
                **fields,
            )


def _run_register_command(config: CommandConfig, options: dict[str, object]) -> None:
    """Execute ``arena cluster register`` after Click has parsed the flags."""
    _apply_verbosity(verbose=cast("bool", options["verbose"]))
    options = _apply_lab_flags(options)
    cluster_name, cluster_spec, registration = _resolve_register_target(config, options)
    output_dir = absolute_path(
        Path(
            cast("str | None", options["output_dir"])
            or f"./arena-cluster-{cluster_name}"
        )
    )
    charts_dir = cast("str | None", options["charts_dir"])
    charts = Path(charts_dir).expanduser() if charts_dir else None
    if _provision_missing_cluster(
        config, cluster_spec, registration, options, output_dir, charts
    ):
        return
    _ensure_post_apply(config, cluster_spec, options, output_dir)
    _register_existing_cluster(
        config, cluster_name, cluster_spec, registration, options, output_dir, charts
    )


def build_cluster_register_command() -> click.Command:
    """``arena cluster register`` — register a cluster and write Helm values.

    :returns: The configured ``register`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "register",
        context_settings={"max_content_width": 100},
    )
    @_cluster_register_options
    @click.pass_obj
    def register_cmd(config: CommandConfig, **options: object) -> None:
        """Register a Kubernetes cluster and write Helm install bundles.

        Re-running with the same ``--name`` updates the existing cluster (upsert)
        and keeps its agent token; ``--rotate-token`` issues a new one.
        Object storage is optional. ``--install`` deploys the agent chart.
        ``--install-storage`` requests bundled MinIO and installs that chart.
        ``--lab`` is ``--install-storage --narrow-allowed-ips``.
        ``--byoc-scope platform`` shares the Arena catalog and bucket.
        """
        _run_register_command(config, options)

    return register_cmd


def _agent_namespace_option(command: ClickDecorated) -> ClickDecorated:
    """Add ``--agent-namespace`` for a command that installs or checks the agent."""
    return click.option(
        "--agent-namespace",
        default=AGENT_NAMESPACE,
        show_default=True,
        help="Kubernetes namespace for the agent Helm release.",
    )(command)


def _ray_data_storage_class_option(command: ClickDecorated) -> ClickDecorated:
    """Add ``--ray-data-storage-class-name``."""
    return click.option(
        "--ray-data-storage-class-name",
        default=None,
        help="StorageClass for Ray data. Defaults to the cluster's default StorageClass.",
    )(command)


def build_cluster_preflight_command() -> click.Command:
    """``arena cluster preflight`` — check a cluster before connecting it."""

    @click.command("preflight", context_settings={"max_content_width": 100})
    @click.option(
        "--name",
        default=None,
        help=(
            "Arena cluster name. When Arena already has this cluster, its own agent "
            "does not count as a second one."
        ),
    )
    @kube_target_options
    @_agent_namespace_option
    @_ray_data_storage_class_option
    @click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Print the report as one JSON document, for the Arena web wizard.",
    )
    @click.pass_obj
    def preflight_cmd(
        config: CommandConfig,
        name: str | None,
        kubeconfig: Path | None,
        kube_context: str | None,
        agent_namespace: str,
        ray_data_storage_class_name: str | None,
        as_json: bool,
    ) -> None:
        """Check that a Kubernetes cluster can run the Arena agent.

        Read-only: changes nothing in Arena or the cluster. Exits non-zero when
        a check fails.
        """
        namespace = normalize_agent_namespace(agent_namespace)
        with (
            kube_target(name, kubeconfig, kube_context) as target,
            kubeconfig_environment(target.kubeconfig),
        ):
            own_release = None
            if name is not None:
                with arena_client(config) as client:
                    registered = registered_cluster(ByocApi(client), name, target)
                if registered is not None:
                    own_release = own_agent_release(target.kubeconfig, namespace, name)
            checks = run_preflight(
                target.kubeconfig, namespace, own_release, ray_data_storage_class_name
            )
        if as_json or json_output_enabled():
            report = preflight_report(
                target.context, target.server, target.cluster_uid, checks
            )
            click.echo(json.dumps(report))
            if any_failed(checks):
                sys.exit(1)
            return
        echo_checks(checks)
        require_passed(checks)

    return preflight_cmd


def build_cluster_connect_command() -> click.Command:
    """``arena cluster connect`` — preflight, register, install and wait for the agent."""

    @click.command("connect", context_settings={"max_content_width": 100})
    @click.option("--name", required=True, help="Arena cluster name.")
    @kube_target_options
    @_agent_namespace_option
    @click.option("--storage-endpoint", default=None, help="Object storage endpoint.")
    @click.option("--storage-bucket", default=None, help="Object storage bucket.")
    @click.option("--storage-prefix", default=None, help="Object storage prefix.")
    @click.option(
        "--storage-secret-name",
        default=None,
        help="Secret in the agent namespace that holds the storage credentials.",
    )
    @_ray_data_storage_class_option
    @click.option(
        "--rotate-token",
        is_flag=True,
        default=False,
        help=(
            "Issue a new agent token for a cluster that is already registered. "
            "The previous token keeps working for 24 hours."
        ),
    )
    @click.option(
        "--wait-timeout",
        type=click.FloatRange(min=0),
        default=900.0,
        show_default=True,
        help="Seconds to wait for the agent to connect and send its health report.",
    )
    @click.option(
        "--yes",
        is_flag=True,
        default=False,
        help="Create the resource classes Arena suggests without prompting.",
    )
    @click.pass_obj
    def connect_cmd(
        config: CommandConfig,
        name: str,
        kubeconfig: Path | None,
        kube_context: str | None,
        agent_namespace: str,
        storage_endpoint: str | None,
        storage_bucket: str | None,
        storage_prefix: str | None,
        storage_secret_name: str | None,
        ray_data_storage_class_name: str | None,
        rotate_token: bool,
        wait_timeout: float,
        yes: bool,
    ) -> None:
        """Connect a Kubernetes cluster to Arena in one step.

        Runs preflight and stops on a failed check, then registers the cluster
        (an existing name keeps its token), installs the agent, and waits until
        it connects and reports its health after the install finishes. Lists
        the resource classes Arena suggests from the cluster's nodes and
        creates them on --yes or after a prompt.
        """
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        run_cluster_connect(
            config,
            ConnectOptions(
                name=cluster_name,
                kubeconfig=kubeconfig,
                kube_context=kube_context,
                agent_namespace=normalize_agent_namespace(agent_namespace),
                storage_endpoint=storage_endpoint,
                storage_bucket=storage_bucket,
                storage_prefix=storage_prefix,
                storage_secret_name=storage_secret_name,
                ray_data_storage_class_name=ray_data_storage_class_name,
                rotate_token=rotate_token,
                wait_timeout=wait_timeout,
                yes=yes,
            ),
        )

    return connect_cmd


def _helm_releases_to_uninstall(
    installed: list[tuple[str, str]],
    uninstall_helm: bool,
    uninstall_storage: bool,
    skip_prompts: bool,
) -> list[tuple[str, str]]:
    """Ask which releases to uninstall once Arena has removed the cluster.

    The agent is uninstalled unless declined. MinIO is kept unless asked for.
    """
    agent = [
        (release, namespace)
        for release, namespace in installed
        if release != STORAGE_RELEASE
    ]
    storage = [
        (release, namespace)
        for release, namespace in installed
        if release == STORAGE_RELEASE
    ]
    selected: list[tuple[str, str]] = []
    if agent:
        click.echo("Installed agent Helm release:")
        for release, namespace in agent:
            click.echo(f"  {release} (namespace {namespace})")
        remove_agent = uninstall_helm
        if not remove_agent and not skip_prompts:
            remove_agent = click.confirm(
                "Uninstall the Arena agent Helm release once Arena has removed the cluster?",
                default=True,
            )
        if remove_agent:
            selected.extend(agent)
    if storage:
        click.echo("Installed MinIO storage Helm release:")
        for release, namespace in storage:
            click.echo(f"  {release} (namespace {namespace})")
        click.echo(
            "Uninstalling MinIO deletes checkpoints, metrics, and datasets on this cluster."
        )
        remove_storage = uninstall_storage
        if not remove_storage and not skip_prompts:
            remove_storage = click.confirm(
                "Uninstall the MinIO storage Helm release?",
                default=False,
            )
        if remove_storage:
            selected.extend(storage)
        else:
            click.echo(
                "Keeping MinIO storage Helm release (checkpoints, metrics, and datasets)."
            )
    return selected


def _installed_byoc_releases(
    config: CommandConfig,
    cluster_name: str,
    target: KubeTarget | None,
    platform: bool,
) -> list[tuple[str, str]]:
    """Installed BYOC releases on *target* once Arena confirms it is this cluster.

    No target means the releases are not looked at, so there are none.
    """
    if target is None:
        return []
    with arena_client(config) as client:
        require_registered_cluster(
            ByocApi(client, platform=platform), cluster_name, target
        )
    return list_installed_byoc_releases(target.kubeconfig, cluster_name=cluster_name)


def _wait_until_archived(api: ByocApi, cluster_name: str, timeout: float) -> None:
    """Poll Arena every few seconds until the cluster leaves its active clusters.

    :raises click.ClickException: If it is still there after *timeout* seconds.
    """
    poll_seconds = 5.0
    click.echo(
        "Waiting for the agent to stop the cluster's runs and delete its job namespaces…"
    )
    deadline = time.monotonic() + timeout
    while api.find_cluster(cluster_name) is not None:
        if time.monotonic() >= deadline:
            msg = (
                f"Cluster {cluster_name!r} is still Deregistering after {timeout:.0f}s: "
                "its runs may still be stopping, or its agent may be offline. Re-run "
                "this command to keep waiting, or add --force if the agent is offline."
            )
            raise click.ClickException(msg)
        time.sleep(poll_seconds)


def _agent_cleanup_command(
    installed: list[tuple[str, str]], response: dict[str, Any]
) -> str:
    """Uninstall the agent (its hook deletes its job namespaces), else delete them by label."""
    owner_label = "arena.agilerl.com/agent-namespace"
    agent = [pair for pair in installed if pair[0] != STORAGE_RELEASE]
    if not agent:
        stored = response.get("cleanupCommand")
        if isinstance(stored, str) and stored:
            return stored
        agent = [(AGENT_RELEASE, AGENT_NAMESPACE)]
    release, namespace = agent[0]
    return (
        f"helm uninstall {release} --namespace {namespace} --timeout 10m || "
        f"kubectl delete namespace --selector={owner_label}={namespace}"
    )


def _platform_option(command: click.Command) -> click.Command:
    """Add ``--platform`` to a command that names a registered cluster."""
    return click.option(
        "--platform",
        is_flag=True,
        default=False,
        help="Target AgileRL's own platform clusters, which needs AgileRL staff access.",
    )(command)


def _is_platform_spec(spec: ClusterSpec) -> bool:
    """Whether *spec* describes one of AgileRL's platform clusters."""
    return spec.values.byoc_scope == "platform"


@contextmanager
def _unregister_target(
    cluster_name: str,
    kubeconfig: Path | None,
    context: str | None,
    uninstall_flagged: bool,
    yes: bool,
) -> Iterator[KubeTarget | None]:
    """Yield the cluster whose Helm releases unregister may remove, or ``None``.

    ``--yes`` without an uninstall flag never touches the cluster. Without any
    kubeconfig the releases are left alone unless an uninstall flag asked for
    them.
    """
    if yes and not uninstall_flagged:
        yield None
        return
    if not helm_available():
        click.echo("Could not check Helm releases: helm is not on PATH.")
        yield None
        return
    with ExitStack() as stack:
        target: KubeTarget | None = None
        try:
            target = stack.enter_context(kube_target(cluster_name, kubeconfig, context))
        except NoKubeContextError as exc:
            if uninstall_flagged:
                raise
            click.echo(f"Could not check Helm releases. {exc.message}")
        yield target


def build_cluster_unregister_command() -> click.Command:
    """``arena cluster unregister`` — remove a registered cluster by name.

    :returns: The configured ``unregister`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "unregister",
        context_settings={"max_content_width": 100},
    )
    @click.option("--name", required=True, help="Registered cluster name.")
    @kube_target_options
    @click.option(
        "--stop-runs",
        is_flag=True,
        default=False,
        help="Stop the cluster's experiments and deployments first.",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help=(
            "Archive the cluster now without waiting for its agent, for an agent that "
            "is offline. Arena ends its runs. Uninstalling the agent release removes "
            "its job namespaces; if the release stays, run the printed command."
        ),
    )
    @click.option(
        "--uninstall-helm",
        is_flag=True,
        default=False,
        help="Uninstall the agent Helm release without prompting. Does not uninstall MinIO.",
    )
    @click.option(
        "--keep-agent",
        is_flag=True,
        default=False,
        help="Leave the agent Helm release installed.",
    )
    @click.option(
        "--uninstall-storage",
        is_flag=True,
        default=False,
        help=(
            "Uninstall the MinIO storage Helm release without prompting. "
            "Deletes checkpoints, metrics, and datasets."
        ),
    )
    @click.option(
        "--wait-timeout",
        type=click.FloatRange(min=0),
        default=600.0,
        show_default=True,
        help="Seconds to wait for the agent to clean up.",
    )
    @click.option(
        "--yes",
        is_flag=True,
        default=False,
        help="Unregister without prompting. Uninstalls nothing without --uninstall-helm.",
    )
    @_platform_option
    @click.pass_obj
    def unregister_cmd(
        config: CommandConfig,
        name: str,
        kubeconfig: Path | None,
        kube_context: str | None,
        stop_runs: bool,
        force: bool,
        uninstall_helm: bool,
        keep_agent: bool,
        uninstall_storage: bool,
        wait_timeout: float,
        yes: bool,
        platform: bool,
    ) -> None:
        """Remove a BYOC cluster from Arena by name.

        Arena refuses a cluster with runs unless --stop-runs stops them. Its
        agent then deletes the job namespaces it created and Arena archives the
        cluster and its classes and revokes its tokens. The agent Helm release
        is uninstalled after that, unless declined.
        """
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        with _unregister_target(
            cluster_name,
            kubeconfig,
            kube_context,
            uninstall_flagged=uninstall_helm or uninstall_storage,
            yes=yes,
        ) as target:
            installed = _installed_byoc_releases(config, cluster_name, target, platform)
            offered = [
                pair
                for pair in installed
                if not keep_agent or pair[0] == STORAGE_RELEASE
            ]
            to_uninstall = _helm_releases_to_uninstall(
                offered,
                uninstall_helm=uninstall_helm,
                uninstall_storage=uninstall_storage,
                skip_prompts=yes,
            )
            if not yes and not click.confirm(
                f"Unregister BYOC cluster {cluster_name!r} from Arena?",
                default=False,
            ):
                click.echo("Aborted.")
                return
            with arena_client(config) as client:
                api = ByocApi(client, platform=platform)
                response = api.unregister_cluster(
                    cluster_name, stop_runs=stop_runs, force=force
                )
                deregistering = response.get("status") == "Deregistering"
                if deregistering:
                    _wait_until_archived(api, cluster_name, wait_timeout)
            # Arena sends a cleanup command when the agent does not clean up.
            agent_cleaned_up = deregistering and "cleanupCommand" not in response
            click.echo(f"Unregistered BYOC cluster {cluster_name!r}.")
            if target is not None and to_uninstall:
                uninstall_byoc_releases(to_uninstall, target.kubeconfig)
        if any(release != STORAGE_RELEASE for release, _ in to_uninstall):
            return
        if agent_cleaned_up:
            click.echo("The agent removed its job namespaces. To uninstall it, run:")
        else:
            click.echo(
                "The agent did not remove its job namespaces. To remove them and "
                "the agent, run:"
            )
        click.echo(f"  {_agent_cleanup_command(installed, response)}")

    return unregister_cmd


def _credential_command(
    command_name: str,
    help_text: str,
    question: str,
    run: Callable[..., object],
) -> click.Command:
    """Build a cluster credential command that names a cluster and confirms."""

    @click.command(command_name, help=help_text)
    @click.option("--name", required=True, help="Registered cluster name.")
    @click.option("--yes", is_flag=True, default=False, help="Do not ask first.")
    @_platform_option
    @click.pass_obj
    def credential_cmd(
        config: CommandConfig, name: str, yes: bool, platform: bool
    ) -> None:
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        if not yes and not click.confirm(question.format(name=cluster_name)):
            click.echo("Aborted.")
            return
        with arena_client(config) as client:
            run(client, name=cluster_name, platform=platform)

    return credential_cmd


def build_cluster_rotate_token_command() -> click.Command:
    """Build ``arena cluster rotate-token``."""
    return _credential_command(
        "rotate-token",
        "Ask the cluster's agent to renew its credential. The current one keeps "
        "working until the agent uses the new one.",
        "Ask the agent on {name!r} to renew its credential?",
        run_cluster_rotate_token,
    )


def build_cluster_revoke_token_command() -> click.Command:
    """Build ``arena cluster revoke-token``."""
    return _credential_command(
        "revoke-token",
        "Revoke every credential of the cluster now, for a leaked one. The agent "
        "is refused until you run `arena cluster connect` again.",
        "Revoke every credential of {name!r} now? Its agent is cut off until you "
        "connect the cluster again.",
        run_cluster_revoke_token,
    )


def build_cluster_upgrade_command() -> click.Command:
    """Build ``arena cluster upgrade``."""

    @click.command("upgrade")
    @click.option("--name", required=True, help="Registered cluster name.")
    @kube_target_options
    @click.option(
        "--agent-namespace",
        default=None,
        help=(
            "Kubernetes namespace for the agent Helm release. "
            "Defaults to the namespace Helm reports for this cluster."
        ),
    )
    @click.option(
        "--no-helm-wait",
        is_flag=True,
        default=False,
        help="Do not wait for the agent Deployment.",
    )
    @click.option(
        "--yes",
        is_flag=True,
        default=False,
        help="Apply agent ClusterRole changes without prompting.",
    )
    @_platform_option
    @click.pass_obj
    def upgrade_cmd(
        config: CommandConfig,
        name: str,
        kubeconfig: Path | None,
        kube_context: str | None,
        agent_namespace: str | None,
        no_helm_wait: bool,
        yes: bool,
        platform: bool,
    ) -> None:
        """Upgrade the BYOC agent chart without changing cluster credentials."""
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        with arena_client(config) as client:
            run_cluster_upgrade(
                client,
                name=cluster_name,
                kubeconfig=kubeconfig,
                context=kube_context,
                agent_namespace=agent_namespace,
                helm_wait=not no_helm_wait,
                platform=platform,
                yes=yes,
            )

    return upgrade_cmd


def build_cluster_manifest_command() -> click.Command:
    """Build ``arena cluster manifest``."""

    @click.command("manifest")
    @click.option("--name", required=True, help="Registered cluster name.")
    @click.option(
        "--format",
        "manifest_format",
        type=click.Choice(sorted(MANIFEST_KEYS)),
        required=True,
        help="argocd: an Argo CD Application. flux: a Flux HelmRepository and HelmRelease.",
    )
    @click.option(
        "--output",
        "-o",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="File to write. Standard output when omitted.",
    )
    @_platform_option
    @click.pass_obj
    def manifest_cmd(
        config: CommandConfig,
        name: str,
        manifest_format: str,
        output: Path | None,
        platform: bool,
    ) -> None:
        """Write the GitOps manifest for the agent at this Arena release.

        The manifest pins the agent chart and holds no secrets. On a new
        cluster, run the install package's bootstrap command first. To upgrade,
        commit the new manifest.
        """
        with arena_client(config) as client:
            run_cluster_manifest(
                client,
                name=name,
                manifest_format=manifest_format,
                output=output,
                platform=platform,
            )

    return manifest_cmd


def build_cluster_smoke_command() -> click.Command:
    """Build ``arena cluster smoke``."""

    @click.command("smoke")
    @click.option("--name", required=True, help="Registered cluster name.")
    @kube_target_options
    @click.option(
        "--timeout",
        type=click.FloatRange(min=0),
        default=3600.0,
        show_default=True,
        help="Seconds to wait for the result.",
    )
    @_platform_option
    @click.pass_obj
    def smoke_cmd(
        config: CommandConfig,
        name: str,
        kubeconfig: Path | None,
        kube_context: str | None,
        timeout: float,
        platform: bool,
    ) -> None:
        """Check that a BYOC cluster can run jobs.

        The cluster's agent installs a small train release as its own
        ServiceAccount, the way it installs a job, waits until its Ray head is
        Ready, and removes it. The release mounts no storage and asks for no
        GPU. The command fails with the agent's reason when the release does
        not come up.
        """
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        with arena_client(config) as client:
            run_cluster_smoke(
                client,
                name=cluster_name,
                kubeconfig=kubeconfig,
                context=kube_context,
                platform=platform,
                timeout=timeout,
            )

    return smoke_cmd


def _cluster_provision_options(command: click.Command) -> click.Command:
    """Add shared cloud-provisioning options to a Click command."""
    command = click.option(
        "--module-version",
        default=None,
        help="Terraform module release to install. Upgrades a cluster that already has a pin.",
    )(command)
    command = click.option(
        "--state-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Terraform state directory (default: OUTPUT_DIR/terraform).",
    )(command)
    return click.option(
        "--spec",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        required=True,
        help="Cloud cluster provisioning YAML file.",
    )(command)


def _cluster_registration_options(command: click.Command) -> click.Command:
    """Add registration overrides shared by plan and provision."""
    for declaration, kwargs in (
        ("--name", {"default": None, "help": "Override the cluster name from --spec."}),
        (
            "--storage-endpoint",
            {"default": None, "help": "Override the object storage endpoint."},
        ),
        (
            "--storage-bucket",
            {"default": None, "help": "Override the object storage bucket."},
        ),
        (
            "--storage-prefix",
            {"default": None, "help": "Override the object storage prefix."},
        ),
        (
            "--storage-secret-name",
            {"default": None, "help": "Override the object storage secret name."},
        ),
        (
            "--ingress-class-name",
            {"default": None, "help": "Override the inference ingress class."},
        ),
        ("--domain", {"default": None, "help": "Override the inference DNS domain."}),
        (
            "--hostname-template",
            {"default": None, "help": "Override the inference hostname template."},
        ),
        (
            "--gateway-api-parent-refs",
            {"default": None, "help": "Override Gateway API parent refs JSON."},
        ),
        (
            "--tls-secret-name",
            {"default": None, "help": "Override the inference TLS secret."},
        ),
        (
            "--preprocessing-resource-class",
            {
                "default": None,
                "help": "Override the preprocessing resource class name.",
            },
        ),
        (
            "--ray-data-storage-class-name",
            {"default": None, "help": "Override the Ray data storage class."},
        ),
        (
            "--ray-data-pvc-size",
            {"default": None, "help": "Override the Ray data PVC size."},
        ),
        (
            "--placement",
            {
                "type": click.Choice(["in-cluster", "ephemeral"]),
                "default": None,
                "help": (
                    "Override values.placement. "
                    "in-cluster runs jobs in this cluster; "
                    "ephemeral hunts capacity outside it."
                ),
            },
        ),
    ):
        command = click.option(declaration, **kwargs)(command)
    return _byoc_scope_option(command)


def _byoc_scope_option(command: click.Command) -> click.Command:
    """Add the customer or platform scope override."""
    return click.option(
        "--byoc-scope",
        type=click.Choice(["customer", "platform"]),
        default=None,
        help=(
            "customer keeps this org's bucket. "
            "platform shares the Arena catalog and bucket "
            "and requires a platform admin. "
            "Choosing platform uses ephemeral placement unless "
            "--placement is set."
        ),
    )(command)


def _load_provisioning(
    spec_path: Path, state_dir: Path | None
) -> tuple[ClusterSpec, Path]:
    """Load a cluster spec and resolve its state directory."""
    try:
        spec = load_cluster_spec(spec_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    resolved_state = absolute_path(
        state_dir or Path(f"./arena-cluster-{spec.name}/terraform")
    )
    return spec, resolved_state


def _register_gpu_resource_classes(
    config: CommandConfig,
    api: ByocApi,
    spec: ClusterSpec,
    spec_path: Path,
    state_dir: Path,
    output_dir: Path,
    register_all: bool,
    requested_presets: tuple[str, ...],
    yes: bool,
    module_version: str | None,
    register_node_group: bool,
) -> None:
    """Create Arena resource classes for Nebius GPU presets on a registered cluster."""
    if spec.provider != "nebius":
        msg = "register-classes supports Nebius clusters."
        raise click.ClickException(msg)
    if api.find_cluster(spec.name) is None:
        msg = (
            f"Cluster {spec.name!r} is not registered. "
            f"Run arena cluster register --spec {spec_path} first."
        )
        raise click.ClickException(msg)
    runner = _cluster_provisioner(config, spec, state_dir, module_version)
    terraform_outputs: dict[str, object] | None = None
    project_id = spec.nebius_cloud().project_id
    if project_id is None:
        terraform_outputs = runner.status(spec, state_dir=state_dir)
        project_id = project_id_from_outputs(terraform_outputs)
    presets = list_registerable_gpu_presets(spec, project_id)
    if not presets:
        if spec.values.placement == "ephemeral":
            msg = f"No GPU presets in region projects for cluster {spec.name!r}."
        else:
            msg = f"No GPU presets in Nebius project {project_id!r}."
        raise click.ClickException(msg)
    selected = resolve_gpu_presets(
        presets, register_all=register_all, requested=requested_presets
    )
    updated, new_workers = _workers_for_presets(
        spec, selected, register_node_group=register_node_group
    )
    if register_node_group:
        updated = _enable_node_group_registration(updated, selected)
    if not _confirm_gpu_classes(
        spec.name,
        selected,
        spec,
        register_node_group=register_node_group,
        yes=yes,
    ):
        click.echo("Aborted.")
        return
    group_ids = _node_groups_for_classes(
        updated,
        original=spec,
        spec_path=spec_path,
        state_dir=state_dir,
        output_dir=output_dir,
        new_workers=new_workers,
        selected=selected,
        terraform_outputs=terraform_outputs,
        runner=runner,
    )
    names = set(_selected_worker_names(updated, selected))
    classes = _classes_for_workers(updated, group_ids, names, selected)
    advice = list_capacity_advice(spec.nebius_cloud().tenant_id)
    api.load_classes()
    for resource_class in classes:
        _ensure_gpu_resource_class(api, spec.name, resource_class, advice, yes)


def _ensure_gpu_resource_class(
    api: ByocApi,
    cluster_name: str,
    resource_class: ClusterResourceClass,
    advice: list[dict[str, Any]],
    yes: bool,
) -> None:
    """Register one class. Skip a platform Arena does not recognize."""
    try:
        ensure_resource_class(
            api,
            cluster_name=cluster_name,
            resource_class=_with_region_capacity(resource_class, advice),
            yes=yes,
        )
    except ArenaAPIError as exc:
        if exc.status_code != 400 or not exc.detail.startswith(
            "unknown Nebius platform"
        ):
            raise
        click.echo(f"Skipped resource class {resource_class.name!r}: {exc.detail}")


def _with_region_capacity(
    resource_class: ClusterResourceClass, advice: list[dict[str, Any]]
) -> ClusterResourceClass:
    """Attach Nebius region capacity for this class's platform and preset."""
    compute = cast("dict[str, Any]", resource_class.metadata["computeResource"])
    gpu = cast("dict[str, Any]", compute["gpu"])
    rows = region_capacity_for_preset(advice, gpu["type"], compute["machineType"])
    return replace(
        resource_class,
        enabled=bool(rows),
        availabilities=tuple(
            ClassAvailability(
                region=row.region,
                zones=row.zones,
                enabled=row.enabled,
                available=row.available,
            )
            for row in rows
        ),
    )


def _classes_for_workers(
    spec: ClusterSpec,
    group_ids: dict[str, str],
    names: set[str],
    selected: list[GpuPreset],
) -> list[ClusterResourceClass]:
    """Build resource classes for the selected workers at the preset machine size."""
    cloud = spec.nebius_cloud()
    workers = [worker for worker in cloud.workers if worker.name in names]
    narrowed = spec.model_copy(
        update={"nebius": cloud.model_copy(update={"workers": workers})}
    )
    vram_gib_by_platform = {
        preset.platform: preset.vram_gib for preset in selected if preset.vram_gib
    }
    return resource_classes(
        narrowed,
        group_ids,
        vram_gib_by_platform=vram_gib_by_platform,
        allow_missing_node_group=True,
    )


def _selected_worker_names(spec: ClusterSpec, selected: list[GpuPreset]) -> list[str]:
    """Return worker names for the selected presets."""
    names: list[str] = []
    for preset in selected:
        worker = worker_for_preset(spec, preset)
        if worker is None:
            msg = (
                f"Worker for {preset.platform}:{preset.preset} "
                "was not added to the spec."
            )
            raise click.ClickException(msg)
        names.append(worker.name)
    return names


def _workers_for_presets(
    spec: ClusterSpec, selected: list[GpuPreset], register_node_group: bool
) -> tuple[ClusterSpec, list[WorkerNodeGroupSpec]]:
    """Return the spec plus worker groups that still need to be created."""
    updated = spec
    new_workers: list[WorkerNodeGroupSpec] = []
    drivers: dict[str, str] = {}
    for preset in selected:
        if worker_for_preset(updated, preset) is not None:
            continue
        if register_node_group:
            if preset.platform not in drivers:
                drivers[preset.platform] = cuda_drivers_preset(
                    updated.nebius_cloud().kubernetes_version, preset.platform
                )
            drivers_preset = drivers[preset.platform]
        else:
            drivers_preset = updated.nebius_cloud().workers[0].gpu_drivers_preset
        worker = worker_spec_for_preset(
            updated,
            preset,
            drivers_preset=drivers_preset,
            register_node_group=register_node_group,
        )
        if register_node_group:
            new_workers.append(worker)
        cloud = updated.nebius_cloud()
        updated = updated.model_copy(
            update={
                "nebius": cloud.model_copy(update={"workers": [*cloud.workers, worker]})
            }
        )
    return updated, new_workers


def _enable_node_group_registration(
    spec: ClusterSpec, selected: list[GpuPreset]
) -> ClusterSpec:
    """Mark the selected workers so this run creates their Nebius node groups."""
    selected_keys = {(preset.platform, preset.preset) for preset in selected}
    cloud = spec.nebius_cloud()
    workers = [
        (
            worker.model_copy(update={"register_node_group": True})
            if (worker.platform, worker.preset) in selected_keys
            else worker
        )
        for worker in cloud.workers
    ]
    return spec.model_copy(
        update={"nebius": cloud.model_copy(update={"workers": workers})}
    )


def _confirm_gpu_classes(
    cluster_name: str,
    selected: list[GpuPreset],
    spec: ClusterSpec,
    register_node_group: bool,
    yes: bool,
) -> bool:
    """Ask before registering classes, and before creating node groups."""
    lines = ["Resource classes:"]
    will_create = False
    for preset in selected:
        worker = worker_for_preset(spec, preset)
        create = register_node_group or (
            worker is not None and worker.register_node_group
        )
        if create and worker is None:
            note = "create node group"
            will_create = True
        elif create:
            note = "register node group"
            will_create = True
        else:
            note = "class only"
        lines.append(f"  {preset.platform}:{preset.preset} ({note})")
    click.echo("\n".join(lines))
    if yes:
        return True
    question = (
        f"Register these classes and Nebius node groups on cluster {cluster_name!r}?"
        if will_create
        else f"Register these classes on cluster {cluster_name!r}?"
    )
    return click.confirm(question, default=False)


def _recorded_node_group_ids(value: object) -> dict[str, str]:
    """Return node group ids Terraform already has."""
    if not isinstance(value, dict):
        return {}
    return {
        name: group_id
        for name, group_id in value.items()
        if isinstance(name, str)
        and name.strip()
        and isinstance(group_id, str)
        and group_id.strip()
    }


def _node_groups_for_classes(
    spec: ClusterSpec,
    original: ClusterSpec,
    spec_path: Path,
    state_dir: Path,
    output_dir: Path,
    new_workers: list[WorkerNodeGroupSpec],
    selected: list[GpuPreset],
    terraform_outputs: dict[str, object] | None,
    runner: ClusterProvisioner,
) -> dict[str, str]:
    """Return node group ids, applying Terraform only for opted-in workers."""
    if new_workers:
        append_worker_blocks(spec_path, new_workers)
        click.echo(f"Added {len(new_workers)} worker node groups to {spec_path}.")
        return _apply_worker_node_groups(
            spec, state_dir=state_dir, output_dir=output_dir, runner=runner
        )
    names = set(_selected_worker_names(spec, selected))
    create_names = [
        worker.name
        for worker in spec.nebius_cloud().workers
        if worker.register_node_group and worker.name in names
    ]
    if not create_names:
        return _existing_group_ids(
            original,
            selected,
            terraform_outputs=terraform_outputs,
            state_dir=state_dir,
            runner=runner,
        )
    if terraform_outputs is None:
        terraform_outputs = runner.status(spec, state_dir=state_dir)
    group_ids = _recorded_node_group_ids(terraform_outputs.get("worker_node_group_ids"))
    if any(name not in group_ids for name in create_names):
        return _apply_worker_node_groups(
            spec, state_dir=state_dir, output_dir=output_dir, runner=runner
        )
    return group_ids


def _existing_group_ids(
    spec: ClusterSpec,
    selected: list[GpuPreset],
    terraform_outputs: dict[str, object] | None,
    state_dir: Path,
    runner: ClusterProvisioner,
) -> dict[str, str]:
    """Return ids for selected workers already in the spec."""
    if not any(worker_for_preset(spec, preset) is not None for preset in selected):
        return {}
    if terraform_outputs is None:
        terraform_outputs = runner.status(spec, state_dir=state_dir)
    return _recorded_node_group_ids(terraform_outputs.get("worker_node_group_ids"))


def _apply_worker_node_groups(
    spec: ClusterSpec,
    state_dir: Path,
    output_dir: Path,
    runner: ClusterProvisioner,
) -> dict[str, str]:
    """Apply Terraform and return the worker node group ids."""
    click.echo(f"Applying Terraform for cluster {spec.name!r}.")
    outputs = runner.provision(spec, state_dir=state_dir, output_dir=output_dir)
    click.echo(f"Kubeconfig written to {outputs.kubeconfig_path}")
    return outputs.worker_node_group_ids


def build_cluster_register_classes_command() -> click.Command:
    """Build ``arena cluster register-classes``."""

    @click.command("register-classes")
    @_cluster_provision_options
    @click.option(
        "--output-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Directory for the kubeconfig written when new node groups are applied.",
    )
    @click.option(
        "--all",
        "register_all",
        is_flag=True,
        default=False,
        help="Register every GPU preset in the Nebius project.",
    )
    @click.option(
        "--preset",
        "requested_presets",
        multiple=True,
        help="Register one preset, as platform:preset. Repeat to register several.",
    )
    @click.option("--yes", is_flag=True, default=False, help="Apply without prompting.")
    @click.option(
        "--register-node-group",
        is_flag=True,
        default=False,
        help=(
            "Create the Nebius node group for each selected preset, "
            "then register the class."
        ),
    )
    @click.pass_obj
    def register_classes_cmd(
        config: CommandConfig,
        spec: Path,
        state_dir: Path | None,
        output_dir: Path | None,
        register_all: bool,
        requested_presets: tuple[str, ...],
        yes: bool,
        register_node_group: bool,
        module_version: str | None,
    ) -> None:
        """Register Nebius GPU presets as Arena resource classes on an existing cluster."""
        if register_all and requested_presets:
            msg = "Pass --all or --preset, not both."
            raise click.ClickException(msg)
        if not register_all and not requested_presets and not sys.stdin.isatty():
            msg = (
                "Pass --all, or repeat --preset platform:preset. "
                "Example: --preset gpu-h100-sxm:1gpu-16vcpu-200gb"
            )
            raise click.ClickException(msg)
        cluster_spec, resolved_state = _load_provisioning(spec, state_dir)
        cluster_spec = _recover_spec(config, cluster_spec)
        out = output_dir or Path(f"./arena-cluster-{cluster_spec.name}")
        with arena_client(config) as client:
            _register_gpu_resource_classes(
                config,
                ByocApi(client, platform=_is_platform_spec(cluster_spec)),
                cluster_spec,
                spec_path=spec,
                state_dir=resolved_state,
                output_dir=out,
                register_all=register_all,
                requested_presets=requested_presets,
                yes=yes,
                module_version=module_version,
                register_node_group=register_node_group,
            )

    return register_classes_cmd


def build_cluster_provision_command() -> click.Command:
    """Build ``arena cluster provision``."""

    @click.command("provision")
    @_cluster_provision_options
    @_cluster_registration_options
    @click.argument(
        "provider",
        required=False,
        default=None,
        type=click.Choice(provider_names(), case_sensitive=False),
    )
    @click.option(
        "--output-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Directory for kubeconfig and generated Helm values.",
    )
    @click.option(
        "--register-and-install",
        is_flag=True,
        default=False,
        help="Register the cluster with Arena and install the Helm agent.",
    )
    @click.option(
        "--agent-namespace",
        default=AGENT_NAMESPACE,
        show_default=True,
        help="Kubernetes namespace for the agent Helm release when installing.",
    )
    @click.option("--yes", is_flag=True, default=False, help="Apply without prompting.")
    @click.option(
        "--create-state-bucket",
        is_flag=True,
        default=False,
        help="Create the Terraform state bucket if it is missing.",
    )
    @click.pass_obj
    def provision_cmd(
        config: CommandConfig,
        provider: str | None,
        spec: Path,
        state_dir: Path | None,
        output_dir: Path | None,
        register_and_install: bool,
        agent_namespace: str,
        yes: bool,
        create_state_bucket: bool,
        module_version: str | None,
        **registration_overrides: object,
    ) -> None:
        """Provision a cloud Kubernetes cluster from a YAML spec."""
        cluster_spec, resolved_state = _load_provisioning(spec, state_dir)
        try:
            cluster_spec, registration = _registration_values_from_spec(
                cluster_spec,
                **registration_overrides,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        cluster_spec = _recover_spec(config, cluster_spec)
        _assert_provider_matches_spec(provider, cluster_spec)
        registration["byoc_provider"] = get_provider(
            cluster_spec.provider
        ).registration_payload(cluster_spec)
        out = output_dir or Path(f"./arena-cluster-{cluster_spec.name}")
        if not yes and not click.confirm(
            f"Provision {cluster_spec.provider} cluster {cluster_spec.name!r}?",
            default=False,
        ):
            click.echo("Aborted.")
            return
        outputs = _cluster_provisioner(
            config, cluster_spec, resolved_state, module_version
        ).provision(
            cluster_spec,
            state_dir=resolved_state,
            output_dir=out,
            create_state_bucket=_state_bucket_should_be_created(
                cluster_spec, create_state_bucket
            ),
        )
        click.echo(f"Kubeconfig written to {outputs.kubeconfig_path}")
        installed = read_module_pin(resolved_state)
        if installed is not None:
            click.echo(f"Module {installed.name} {installed.version}")
        if not register_and_install:
            click.echo(
                f"Next: arena cluster register --spec {spec} --install "
                f"--kubeconfig {outputs.kubeconfig_path}"
            )
            return
        _register_provisioned_cluster(
            config,
            registration=registration,
            outputs=outputs,
            cluster_spec=cluster_spec,
            output_dir=out,
            install=True,
            agent_namespace=agent_namespace,
            state_dir=resolved_state,
            yes=yes,
        )

    return provision_cmd


def build_cluster_plan_command() -> click.Command:
    """Build ``arena cluster plan``."""

    @click.command("plan")
    @_cluster_provision_options
    @_cluster_registration_options
    @click.option(
        "--create-state-bucket",
        is_flag=True,
        default=False,
        help="Create the Terraform state bucket if it is missing.",
    )
    @click.pass_obj
    def plan_cmd(
        config: CommandConfig,
        spec: Path,
        state_dir: Path | None,
        create_state_bucket: bool,
        module_version: str | None,
        **registration_overrides: object,
    ) -> None:
        """Create a cloud cluster plan without applying it."""
        cluster_spec, resolved_state = _load_provisioning(spec, state_dir)
        try:
            cluster_spec, _ = _registration_values_from_spec(
                cluster_spec,
                **registration_overrides,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        cluster_spec = _recover_spec(config, cluster_spec)
        _cluster_provisioner(config, cluster_spec, resolved_state, module_version).plan(
            cluster_spec,
            state_dir=resolved_state,
            create_state_bucket=_state_bucket_should_be_created(
                cluster_spec, create_state_bucket
            ),
        )

    return plan_cmd


def build_cluster_status_command() -> click.Command:
    """Build ``arena cluster status``."""

    @click.command("status")
    @_cluster_provision_options
    @click.pass_obj
    def status_cmd(
        config: CommandConfig,
        spec: Path,
        state_dir: Path | None,
        module_version: str | None,
    ) -> None:
        """Show Terraform outputs for a provisioned cluster."""
        cluster_spec, resolved_state = _load_provisioning(spec, state_dir)
        cluster_spec = _recover_spec(config, cluster_spec)
        outputs = _cluster_provisioner(
            config, cluster_spec, resolved_state, module_version
        ).status(cluster_spec, state_dir=resolved_state)
        for key, value in outputs.items():
            shown = "<sensitive>" if key in SENSITIVE_STATUS_OUTPUTS else value
            click.echo(f"{key}: {shown}")

    return status_cmd


def build_cluster_generate_spec_command() -> click.Command:
    """Build ``arena cluster generate-spec``."""

    @click.command("generate-spec")
    @click.option(
        "--provider",
        type=click.Choice(provider_names(), case_sensitive=False),
        default="nebius",
        show_default=True,
        help="Cloud provider for the generated spec.",
    )
    @click.option(
        "--name",
        default=None,
        help="Cluster name written into the spec (default: arena-PROVIDER).",
    )
    @click.option(
        "--output",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Destination YAML path (default: ./NAME.yaml).",
    )
    @click.option(
        "--force", is_flag=True, default=False, help="Overwrite an existing file."
    )
    def generate_spec_cmd(
        provider: str,
        name: str,
        output: Path | None,
        force: bool,
    ) -> None:
        """Write a default cluster spec YAML with placeholder cloud IDs."""
        provider_name = provider.lower()
        cluster_name = name.strip() if name else default_cluster_name(provider_name)
        destination = output or Path(f"./{cluster_name}.yaml")
        try:
            written = write_default_cluster_spec(
                destination,
                provider=provider_name,
                name=cluster_name,
                force=force,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Wrote cluster spec to {written}")

    return generate_spec_cmd


def _load_destroy_target(
    config: CommandConfig,
    spec: Path | None,
    name: str | None,
    state_dir: Path | None,
) -> tuple[ClusterSpec, Path, tuple[bool, bool] | None]:
    """Load the cluster to destroy from a spec file or its Arena name."""
    cluster_name = name.strip() if name else ""
    if spec is not None and cluster_name:
        msg = "Pass --spec or --name, not both."
        raise click.ClickException(msg)
    if spec is None and not cluster_name:
        msg = "Pass --spec or --name."
        raise click.ClickException(msg)
    if spec is not None:
        cluster_spec, resolved_state = _load_provisioning(spec, state_dir)
        created = get_provider(cluster_spec.provider).created_projects(
            CreatedProjectsQuery(cluster_spec)
        )
        return _recover_spec(config, cluster_spec), resolved_state, created
    with arena_client(config) as client:
        row = ByocApi(client).byoc_cluster(cluster_name)
    if row is None:
        msg = f"No cluster named {cluster_name!r} in Arena."
        raise click.ClickException(msg)
    stored = row.get("byoc_provider")
    if not isinstance(stored, dict):
        stored = row.get("byocProvider")
    provider_name = stored.get("provider") if isinstance(stored, dict) else None
    if not isinstance(provider_name, str) or not provider_name.strip():
        msg = f"Cluster {cluster_name!r} has no stored cloud settings."
        raise click.ClickException(msg)
    try:
        cluster_spec = get_provider(provider_name.strip()).spec_from_stored_row(row)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Using cluster {cluster_spec.name!r} stored in Arena.")
    resolved_state = absolute_path(
        state_dir or Path(f"./arena-cluster-{cluster_spec.name}/terraform")
    )
    return cluster_spec, resolved_state, None


def _destroy_prompt(
    cluster_spec: ClusterSpec,
    delete_storage: bool,
    delete_storage_project: bool,
    delete_all: bool,
) -> str:
    """Return the confirmation prompt for a destroy run."""
    cloud = get_provider(cluster_spec.provider).display_name
    bucket = cluster_spec.arena.storage.bucket
    state_bucket = cluster_spec.terraform_state.bucket
    if delete_all:
        region_projects = ""
        if cluster_spec.provider == "nebius":
            region_projects = ", including its region projects"
        return (
            f"Destroy {cloud} cluster {cluster_spec.name!r}, its object storage, "
            f"and its IAM{region_projects}? Terraform state bucket "
            f"{state_bucket!r} is deleted last."
        )
    if delete_storage_project:
        return (
            f"Destroy {cloud} cluster {cluster_spec.name!r}, force-delete object "
            f"storage bucket {bucket!r}, and delete its {cloud} storage project?"
        )
    if delete_storage:
        return (
            f"Destroy {cloud} cluster {cluster_spec.name!r} and force-delete "
            f"object storage bucket {bucket!r}?"
        )
    return (
        f"Destroy {cloud} cluster {cluster_spec.name!r}? "
        f"Object storage bucket {bucket!r} will be kept."
    )


def _echo_destroy_result(
    cluster_spec: ClusterSpec,
    resolved_state: Path,
    delete_storage: bool,
    delete_storage_project: bool,
    delete_all: bool,
    yes: bool,
) -> None:
    """Print what destroy kept or deleted, and drop Terraform state when asked."""
    bucket = cluster_spec.arena.storage.bucket
    state_bucket = cluster_spec.terraform_state.bucket
    state_key = cluster_spec.terraform_state.object_key(cluster_spec.name)
    if delete_all:
        click.echo(f"Deleted Terraform state bucket {state_bucket!r} last.")
        return
    if not delete_storage:
        click.echo(f"Kept object storage bucket {bucket!r}.")
    if delete_storage_project:
        click.echo(
            f"Deleted Terraform state bucket {state_bucket!r} with its Nebius "
            "storage project."
        )
        return
    if yes or not click.confirm(
        f"Delete Terraform state for cluster {cluster_spec.name!r} "
        f"({state_key} in bucket {state_bucket!r})?",
        default=False,
    ):
        click.echo(f"Kept Terraform state {state_key!r} in bucket {state_bucket!r}.")
        return
    get_provider(cluster_spec.provider).delete_terraform_state(
        cluster_spec, state_dir=resolved_state
    )
    click.echo(f"Deleted Terraform state {state_key!r} from bucket {state_bucket!r}.")


def _run_cluster_destroy(
    config: CommandConfig,
    spec: Path | None,
    name: str | None,
    state_dir: Path | None,
    delete_storage: bool,
    delete_storage_project: bool,
    delete_all: bool,
    module_version: str | None,
    yes: bool,
) -> None:
    """Destroy the cluster named by ``--spec`` or ``--name``."""
    cluster_spec, resolved_state, created_projects = _load_destroy_target(
        config, spec, name, state_dir
    )
    delete_storage = delete_storage or delete_storage_project or delete_all
    unregister = False
    platform = _is_platform_spec(cluster_spec)
    with arena_client(config) as client:
        if (
            ByocApi(client, platform=platform).find_cluster(cluster_spec.name)
            is not None
        ):
            unregister_prompt = (
                f"Unregister BYOC cluster {cluster_spec.name!r} from Arena "
                f"after destroying the cloud cluster?"
            )
            if not yes and not click.confirm(unregister_prompt, default=False):
                click.echo("Aborted.")
                return
            unregister = True
    prompt = _destroy_prompt(
        cluster_spec, delete_storage, delete_storage_project, delete_all
    )
    if not yes and not click.confirm(prompt, default=False):
        click.echo("Aborted.")
        return
    _cluster_provisioner(config, cluster_spec, resolved_state, module_version).destroy(
        cluster_spec,
        state_dir=resolved_state,
        delete_storage=delete_storage,
        delete_storage_project=delete_storage_project,
        delete_all=delete_all,
        created_projects=created_projects,
    )
    if unregister:
        # The cloud cluster and its agent are gone, so nothing is left for the
        # agent to clean up and Arena ends the cluster's runs itself.
        with arena_client(config) as client:
            ByocApi(client, platform=platform).unregister_cluster(
                cluster_spec.name, force=True
            )
        click.echo(f"Unregistered BYOC cluster {cluster_spec.name!r}.")
    _echo_destroy_result(
        cluster_spec,
        resolved_state,
        delete_storage,
        delete_storage_project,
        delete_all,
        yes,
    )


def build_cluster_destroy_command() -> click.Command:
    """Build ``arena cluster destroy``."""

    @click.command("destroy")
    @click.option(
        "--spec",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Cloud cluster provisioning YAML file.",
    )
    @click.option(
        "--name",
        default=None,
        help="Cluster name in Arena. Used when --spec is omitted.",
    )
    @click.option(
        "--state-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        help="Terraform state directory (default: ./arena-cluster-NAME/terraform).",
    )
    @click.option(
        "--delete-storage",
        is_flag=True,
        default=False,
        help=(
            "Force-delete experiment object storage (data, metrics, and checkpoints). "
            "Kept otherwise."
        ),
    )
    @click.option(
        "--delete-storage-project",
        is_flag=True,
        default=False,
        help=(
            "Delete experiment object storage, the Terraform state bucket, and the "
            "Nebius storage project that provisioning created."
        ),
    )
    @click.option(
        "--all",
        "delete_all",
        is_flag=True,
        default=False,
        help=(
            "Delete this cluster's buckets and IAM, including its Nebius region "
            "projects. The Terraform state bucket is deleted last."
        ),
    )
    @click.option(
        "--module-version",
        default=None,
        help="Terraform module release to install. Upgrades a cluster that already has a pin.",
    )
    @click.option(
        "--yes", is_flag=True, default=False, help="Destroy without prompting."
    )
    @click.pass_obj
    def destroy_cmd(
        config: CommandConfig,
        spec: Path | None,
        name: str | None,
        state_dir: Path | None,
        delete_storage: bool,
        delete_storage_project: bool,
        delete_all: bool,
        module_version: str | None,
        yes: bool,
    ) -> None:
        """Destroy the cloud cluster recorded in the selected Terraform state."""
        _run_cluster_destroy(
            config,
            spec,
            name,
            state_dir,
            delete_storage,
            delete_storage_project,
            delete_all,
            module_version,
            yes,
        )

    return destroy_cmd
