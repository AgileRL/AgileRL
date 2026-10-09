# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The hardcoded ``arena cluster`` Click commands."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast

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
    ClusterResourceClass,
    run_cluster_register,
)
from agilerl.arena.byoc.cluster_token import (
    resolve_cluster_kubeconfig,
    run_cluster_rotate_token,
)
from agilerl.arena.byoc.provisioning.inference import normalize_inference_domain
from agilerl.arena.byoc.provisioning.providers import get_provider, provider_names
from agilerl.arena.byoc.provisioning.provisioner import ClusterProvisioner
from agilerl.arena.byoc.provisioning.spec import (
    ClusterSpec,
    apply_cluster_spec_overrides,
    default_cluster_name,
    load_cluster_spec,
    write_default_cluster_spec,
)
from agilerl.arena.byoc.provisioning.terraform import ClusterOutputs, absolute_path
from agilerl.arena.cli_manifest import _parse_json_cli_value
from agilerl.arena.config import CommandConfig, arena_client

logger = logging.getLogger("agilerl.arena.byoc")


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
            byoc_provider=get_provider(cluster_spec.provider).registration_payload(
                cluster_spec,
                project_id=outputs.project_id,
                storage_project_id=outputs.storage_project_id,
            ),
            resource_classes=get_provider(cluster_spec.provider).resource_classes(
                cluster_spec,
                worker_node_group_ids=outputs.worker_node_group_ids,
            ),
        )


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


def _assert_register_provider(
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
    _assert_register_provider(provider, cluster_spec)
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
    provisioner = ClusterProvisioner()
    resolved_state = _register_state_dir(cluster_spec, options)
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
    )
    return True


def _kubeconfig_for_register_install(
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
    return ClusterProvisioner().write_kubeconfig(
        cluster_spec,
        state_dir=_register_state_dir(cluster_spec, options),
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
    return fields


def _resource_classes_from_state(
    cluster_spec: ClusterSpec, options: dict[str, object]
) -> list[ClusterResourceClass]:
    """Build GPU resource classes from the provisioned worker node groups."""
    values = ClusterProvisioner().status(
        cluster_spec, state_dir=_register_state_dir(cluster_spec, options)
    )
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
    fields = _register_runtime_fields(registration, options)
    resource_classes = (
        _resource_classes_from_state(cluster_spec, options)
        if cluster_spec is not None
        else ()
    )
    kubeconfig = _kubeconfig_for_register_install(
        cluster_spec, registration, options, output_dir
    )
    if (
        kubeconfig is not None
        and cluster_spec is not None
        and cluster_spec.arena.gateway.enable
    ):
        parent_refs = get_provider(cluster_spec.provider).ensure_gateway(
            cluster_spec,
            kubeconfig,
            ClusterProvisioner().status(
                cluster_spec, state_dir=_register_state_dir(cluster_spec, options)
            ),
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
    ):
        command = click.option(declaration, **kwargs)(command)
    return command


def _load_provisioning(spec_path: Path, state_dir: Path | None) -> tuple[object, Path]:
    """Load a cluster spec and resolve its state directory."""
    try:
        spec = load_cluster_spec(spec_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    resolved_state = absolute_path(
        state_dir or Path(f"./arena-cluster-{spec.name}/terraform")
    )
    return spec, resolved_state


def build_cluster_provision_command() -> click.Command:
    """Build ``arena cluster provision``."""

    @click.command("provision")
    @_cluster_provision_options
    @_cluster_registration_options
    @click.argument(
        "provider", type=click.Choice(provider_names(), case_sensitive=False)
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
        provider: str,
        spec: Path,
        state_dir: Path | None,
        output_dir: Path | None,
        register_and_install: bool,
        agent_namespace: str,
        yes: bool,
        create_state_bucket: bool,
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
        if provider != cluster_spec.provider:
            msg = f"Provider {provider!r} does not match spec provider {cluster_spec.provider!r}."
            raise click.ClickException(msg)
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
        outputs = ClusterProvisioner().provision(
            cluster_spec,
            state_dir=resolved_state,
            output_dir=out,
            create_state_bucket=_state_bucket_should_be_created(
                cluster_spec, create_state_bucket
            ),
        )
        click.echo(f"Kubeconfig written to {outputs.kubeconfig_path}")
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
        ClusterProvisioner().plan(
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
    def status_cmd(config: CommandConfig, spec: Path, state_dir: Path | None) -> None:
        """Show Terraform outputs for a provisioned cluster."""
        cluster_spec, resolved_state = _load_provisioning(spec, state_dir)
        cluster_spec = _recover_spec(config, cluster_spec)
        outputs = ClusterProvisioner().status(cluster_spec, state_dir=resolved_state)
        for key, value in outputs.items():
            click.echo(f"{key}: {value}")

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
) -> tuple[ClusterSpec, Path]:
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
        return _recover_spec(config, cluster_spec), resolved_state
    with arena_client(config) as client:
        row = ByocApi(client).on_prem_cluster(cluster_name)
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
    return cluster_spec, resolved_state


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
        yes: bool,
    ) -> None:
        """Destroy the cloud cluster recorded in the selected Terraform state."""
        cluster_spec, resolved_state = _load_destroy_target(
            config, spec, name, state_dir
        )
        delete_storage = delete_storage or delete_storage_project
        bucket = cluster_spec.arena.storage.bucket
        state_bucket = cluster_spec.terraform_state.bucket
        state_key = cluster_spec.terraform_state.object_key(cluster_spec.name)
        with arena_client(config) as client:
            api = ByocApi(client)
            if api.find_cluster(cluster_spec.name) is not None:
                unregister_prompt = (
                    f"Unregister BYOC cluster {cluster_spec.name!r} from Arena "
                    f"before destroying the cloud cluster?"
                )
                if not yes and not click.confirm(unregister_prompt, default=False):
                    click.echo("Aborted.")
                    return
                api.unregister_cluster(cluster_spec.name)
                click.echo(f"Unregistered BYOC cluster {cluster_spec.name!r}.")
        cloud = get_provider(cluster_spec.provider).display_name
        if delete_storage_project:
            prompt = (
                f"Destroy {cloud} cluster {cluster_spec.name!r}, force-delete object "
                f"storage bucket {bucket!r}, and delete its {cloud} storage project?"
            )
        elif delete_storage:
            prompt = (
                f"Destroy {cloud} cluster {cluster_spec.name!r} and force-delete "
                f"object storage bucket {bucket!r}?"
            )
        else:
            prompt = (
                f"Destroy {cloud} cluster {cluster_spec.name!r}? "
                f"Object storage bucket {bucket!r} will be kept."
            )
        if not yes and not click.confirm(prompt, default=False):
            click.echo("Aborted.")
            return
        ClusterProvisioner().destroy(
            cluster_spec,
            state_dir=resolved_state,
            delete_storage=delete_storage,
            delete_storage_project=delete_storage_project,
        )
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
            click.echo(
                f"Kept Terraform state {state_key!r} in bucket {state_bucket!r}."
            )
            return
        get_provider(cluster_spec.provider).delete_terraform_state(
            cluster_spec, state_dir=resolved_state
        )
        click.echo(
            f"Deleted Terraform state {state_key!r} from bucket {state_bucket!r}."
        )

    return destroy_cmd
