# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The hardcoded ``arena cluster`` Click commands."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import click

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_helm import (
    AGENT_NAMESPACE,
    STORAGE_RELEASE,
    helm_available,
    list_installed_byoc_releases,
    uninstall_byoc_releases,
)
from agilerl.arena.byoc.cluster_register import (
    ClassAvailability,
    ClusterResourceClass,
    ensure_resource_class,
    run_cluster_register,
)
from agilerl.arena.byoc.cluster_token import (
    resolve_cluster_kubeconfig,
    run_cluster_rotate_token,
)
from agilerl.arena.byoc.provisioning.inference import normalize_inference_domain
from agilerl.arena.byoc.provisioning.module_catalog import (
    open_module_catalog,
    read_module_pin,
    selected_module_version,
)
from agilerl.arena.byoc.provisioning.providers import get_provider, provider_names
from agilerl.arena.byoc.provisioning.providers.nebius.ephemeral import (
    ensure_ephemeral_regions,
)
from agilerl.arena.byoc.provisioning.providers.nebius.platforms import (
    GpuPreset,
    append_worker_blocks,
    cuda_drivers_preset,
    list_capacity_advice,
    list_gpu_presets,
    project_id_from_outputs,
    region_capacity_for_preset,
    resolve_gpu_presets,
    worker_for_preset,
    worker_spec_for_preset,
)
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    WorkerNodeGroupSpec,
    attach_service_account_credentials,
    resource_classes,
)
from agilerl.arena.byoc.provisioning.provisioner import ClusterProvisioner
from agilerl.arena.byoc.provisioning.spec import (
    ClusterSpec,
    apply_cluster_spec_overrides,
    default_cluster_name,
    load_cluster_spec,
    write_default_cluster_spec,
)
from agilerl.arena.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
)
from agilerl.arena.cli_manifest import _parse_json_cli_value
from agilerl.arena.config import CommandConfig, arena_client

logger = logging.getLogger("agilerl.arena.byoc")
SENSITIVE_STATUS_OUTPUTS = frozenset(
    {"service_account_private_key", "storage_secret_access_key"}
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
        stored = ByocApi(client).stored_provider_config(spec.name, provider.name)
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
        _kubeconfig_environment(outputs.kubeconfig_path),
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
        )


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
        "byoc_provider": get_provider(resolved.provider).registration_payload(resolved),
    }


@contextmanager
def _kubeconfig_environment(kubeconfig_path: Path) -> Iterator[None]:
    """Set KUBECONFIG only while a Helm installation runs."""
    old_value = os.environ.get("KUBECONFIG")
    os.environ["KUBECONFIG"] = str(kubeconfig_path)
    try:
        yield
    finally:
        if old_value is None:
            del os.environ["KUBECONFIG"]
        else:
            os.environ["KUBECONFIG"] = old_value


def _apply_verbosity(verbose: bool) -> None:
    """``--verbose`` raises the Arena logger to DEBUG so command traces and live
    per-stage script output are shown instead of hidden.

    :param verbose: If ``True``, set the ``agilerl.arena`` logger to DEBUG.
    :type verbose: bool
    :returns: None
    :rtype: None
    """
    if verbose:
        logging.getLogger("agilerl.arena").setLevel(logging.DEBUG)


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
        {"is_flag": True, "default": False, "help": "Provision without prompting."},
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
        fields["placement"] = "in-cluster"
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
    kubeconfig = _kubeconfig_for_register_install(
        config, cluster_spec, registration, options, output_dir
    )
    if (
        kubeconfig is not None
        and cluster_spec is not None
        and cluster_spec.arena.gateway.enable
    ):
        parent_refs = get_provider(cluster_spec.provider).ensure_gateway(
            cluster_spec,
            kubeconfig,
            values or {},
        )
        if parent_refs is not None:
            fields["gateway_api_parent_refs"] = parent_refs
    kubeconfig_cm = (
        _kubeconfig_environment(kubeconfig) if kubeconfig is not None else nullcontext()
    )
    with kubeconfig_cm, arena_client(config) as client:
        run_cluster_register(
            client,
            name=cluster_name,
            output_dir=output_dir,
            skip_enable=cast("bool", options["skip_enable"]),
            force=cast("bool", options["force"]),
            install=cast("bool", options["install"]),
            install_storage=cast("bool", options["install_storage"])
            or bool(registration is not None and registration["install_storage"]),
            narrow_allowed_ips=cast("bool", options["narrow_allowed_ips"]),
            no_write=cast("bool", options["no_write"]),
            charts_dir=charts,
            helm_wait=not cast("bool", options["no_helm_wait"]),
            agent_namespace=cast("str", options["agent_namespace"]),
            resource_classes=resource_classes,
            yes=cast("bool", options.get("yes", False)),
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
        """Register a customer Kubernetes cluster and write Helm install bundles.

        Re-running with the same ``--name`` updates the existing cluster (upsert).
        Object storage is optional. ``--install`` deploys the agent chart.
        ``--install-storage`` requests bundled MinIO and installs that chart.
        ``--lab`` is ``--install-storage --narrow-allowed-ips``.
        """
        _run_register_command(config, options)

    return register_cmd


def _helm_releases_to_uninstall(
    installed: list[tuple[str, str]],
    uninstall_helm: bool,
    uninstall_storage: bool,
    skip_prompts: bool,
) -> list[tuple[str, str]]:
    """Prompt for agent vs MinIO uninstall. MinIO is kept unless forced or confirmed."""
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
                "Uninstall the Arena agent Helm release?",
                default=False,
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


def build_cluster_unregister_command() -> click.Command:
    """``arena cluster unregister`` — archive a registered cluster by name.

    :returns: The configured ``unregister`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "unregister",
        context_settings={"max_content_width": 100},
    )
    @click.option("--name", required=True, help="Registered cluster name.")
    @click.option(
        "--kubeconfig",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Kubeconfig for the agent cluster. Defaults to "
            "arena-cluster-NAME/kubeconfig when that file exists."
        ),
    )
    @click.option(
        "--uninstall-helm",
        is_flag=True,
        default=False,
        help="Uninstall the agent Helm release without prompting. Does not uninstall MinIO.",
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
        "--yes",
        is_flag=True,
        default=False,
        help="Unregister without prompting.",
    )
    @click.pass_obj
    def unregister_cmd(
        config: CommandConfig,
        name: str,
        kubeconfig: Path | None,
        uninstall_helm: bool,
        uninstall_storage: bool,
        yes: bool,
    ) -> None:
        """Unregister a BYOC cluster from Arena by name."""
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        kubeconfig_path = resolve_cluster_kubeconfig(cluster_name, kubeconfig)
        to_uninstall: list[tuple[str, str]] = []
        if not helm_available():
            click.echo("Could not check Helm releases: helm is not on PATH.")
        elif kubeconfig_path is None:
            click.echo(
                "Could not check Helm releases: no kubeconfig. "
                "Pass --kubeconfig, set KUBECONFIG, or run from the directory "
                f"that contains arena-cluster-{cluster_name}/kubeconfig."
            )
        else:
            to_uninstall.extend(
                _helm_releases_to_uninstall(
                    list_installed_byoc_releases(
                        kubeconfig_path, cluster_name=cluster_name
                    ),
                    uninstall_helm=uninstall_helm,
                    uninstall_storage=uninstall_storage,
                    skip_prompts=yes,
                )
            )
        if not yes and not click.confirm(
            f"Unregister BYOC cluster {cluster_name!r} from Arena?",
            default=False,
        ):
            click.echo("Aborted.")
            return
        if to_uninstall and kubeconfig_path is not None:
            uninstall_byoc_releases(to_uninstall, kubeconfig_path)
        with arena_client(config) as client:
            ByocApi(client).unregister_cluster(cluster_name)
        click.echo(f"Unregistered BYOC cluster {cluster_name!r}.")

    return unregister_cmd


def build_cluster_rotate_token_command() -> click.Command:
    """Build ``arena cluster rotate-token``."""

    @click.command("rotate-token")
    @click.option("--name", required=True, help="Registered cluster name.")
    @click.option(
        "--kubeconfig",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Kubeconfig for the agent cluster. Defaults to "
            "arena-cluster-NAME/kubeconfig when that file exists."
        ),
    )
    @click.option(
        "--agent-namespace",
        default=None,
        help=(
            "Kubernetes namespace for the agent Helm release. "
            "Defaults to the namespace Helm reports for arena-byoc-agent."
        ),
    )
    @click.option(
        "--secret-name",
        default=None,
        help="Cluster-token Secret name. Defaults to the Helm agent Secret.",
    )
    @click.option(
        "--yes",
        is_flag=True,
        default=False,
        help="Rotate without prompting.",
    )
    @click.pass_obj
    def rotate_token_cmd(
        config: CommandConfig,
        name: str,
        kubeconfig: Path | None,
        agent_namespace: str | None,
        secret_name: str | None,
        yes: bool,
    ) -> None:
        """Rotate the Arena cluster token and update the agent Secret."""
        cluster_name = name.strip()
        if not cluster_name:
            msg = "--name is required."
            raise click.ClickException(msg)
        if not yes and not click.confirm(
            f"Rotate the cluster token for {cluster_name!r} and update the "
            "agent Secret?",
            default=False,
        ):
            click.echo("Aborted.")
            return
        with arena_client(config) as client:
            run_cluster_rotate_token(
                client,
                name=cluster_name,
                kubeconfig=kubeconfig,
                secret_name=secret_name,
                namespace=agent_namespace,
            )

    return rotate_token_cmd


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
    return command


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
    presets = list_gpu_presets(project_id)
    if not presets:
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
        ensure_resource_class(
            api,
            cluster_name=spec.name,
            resource_class=_with_region_capacity(resource_class, advice),
            yes=yes,
        )


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
                ByocApi(client),
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
                f"Next: KUBECONFIG={outputs.kubeconfig_path} arena cluster "
                f"register --spec {spec} --install"
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
        created = get_provider(cluster_spec.provider).created_projects(cluster_spec)
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
            region_projects = ", including its arena region projects"
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
    with arena_client(config) as client:
        if ByocApi(client).find_cluster(cluster_spec.name) is not None:
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
        with arena_client(config) as client:
            ByocApi(client).unregister_cluster(cluster_spec.name)
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
            "Delete this cluster's buckets and IAM, including Nebius arena region "
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
