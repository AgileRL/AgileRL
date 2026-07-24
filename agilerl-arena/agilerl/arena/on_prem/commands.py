# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The hardcoded ``arena on-prem install`` / ``teardown`` Click commands."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from agilerl.arena.cli_manifest import _parse_json_cli_value
from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.on_prem.cluster_register import run_cluster_register
from agilerl.arena.on_prem.installer import (
    run_on_prem_down,
    run_on_prem_install,
    run_on_prem_teardown,
)

logger = logging.getLogger("agilerl.arena.on_prem")


def _apply_verbosity(*, verbose: bool) -> None:
    """``--verbose`` raises the Arena logger to DEBUG so command traces and live
    per-stage script output are shown instead of hidden.

    :param verbose: If ``True``, set the ``agilerl.arena`` logger to DEBUG.
    :type verbose: bool
    :returns: None
    :rtype: None
    """
    if verbose:
        logging.getLogger("agilerl.arena").setLevel(logging.DEBUG)


def build_install_command() -> click.Command:
    """Primary ``arena on-prem install`` command.

    :returns: The configured ``install`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "install",
        context_settings={"max_content_width": 100},
    )
    @click.argument("name")
    @click.option(
        "--setup-type",
        "setup_type",
        default="dockerSwarm",
        show_default=True,
        type=click.Choice(["dockerSwarm", "helm"], case_sensitive=False),
        help=(
            "dockerSwarm: SSH install on --manager and --workers. "
            "helm: local helm upgrade --install (kubectl context only)."
        ),
    )
    @click.option(
        "--manager",
        default=None,
        help="[dockerSwarm] Swarm manager SSH host (required for dockerSwarm).",
    )
    @click.option(
        "--workers",
        default="",
        help="[dockerSwarm] Comma-separated worker SSH hosts.",
    )
    @click.option(
        "--ssh-user",
        default=None,
        help=(
            "[dockerSwarm] SSH login for remote hosts. "
            "Omit to use User/Host from ~/.ssh/config (same as ``ssh HOST``)."
        ),
    )
    @click.option(
        "--ssh-extra-opts",
        default=None,
        help="[dockerSwarm] Extra ssh(1) arguments.",
    )
    @click.option(
        "--advertise-addr",
        default=None,
        help="[dockerSwarm] Swarm --advertise-addr (default: --manager).",
    )
    @click.option(
        "--skip-enable",
        is_flag=True,
        default=False,
        help="Skip enabling the on-prem provider (use when it is already enabled).",
    )
    @click.option(
        "--skip-verify",
        is_flag=True,
        default=False,
        help="Skip post-install stack or Helm validation.",
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        default=False,
        help="Stream the full output of each install stage instead of hiding it on success.",
    )
    @click.pass_obj
    def install_cmd(
        config: CommandConfig,
        name: str,
        setup_type: str,
        manager: str | None,
        workers: str,
        ssh_user: str | None,
        ssh_extra_opts: str | None,
        advertise_addr: str | None,
        skip_enable: bool,
        skip_verify: bool,
        verbose: bool,
    ) -> None:
        """Install an on-prem worker cluster for CLASS_NAME.

        CLASS_NAME must already exist (create via the Arena UI or
        ``arena on-prem classes create``).

        **dockerSwarm** — downloads the deployment bundle and runs install-docker,
        NVIDIA setup, swarm init/join, GPU node labels, and stack deploy on
        ``--manager`` and ``--workers`` via SSH.

        **helm** — downloads the Helm chart bundle and runs its setup on this
        machine; requires Helm 3.x and a configured ``kubectl`` context.
        """
        _apply_verbosity(verbose=verbose)
        worker_hosts = tuple(h.strip() for h in workers.split(",") if h.strip())

        with arena_client(config) as client:
            run_on_prem_install(
                client,
                name=name.strip(),
                setup_type=setup_type,
                skip_enable=skip_enable,
                manager=manager.strip() if manager else None,
                workers=worker_hosts,
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                advertise_addr=advertise_addr,
                skip_verify=skip_verify,
            )

    return install_cmd


def build_down_command() -> click.Command:
    """``arena on-prem down`` — stop workloads; stack or Helm release remains.

    :returns: The configured ``down`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "down",
        context_settings={"max_content_width": 100},
    )
    @click.argument("name")
    @click.option(
        "--setup-type",
        "setup_type",
        default="dockerSwarm",
        show_default=True,
        type=click.Choice(["dockerSwarm", "helm"], case_sensitive=False),
        help="Must match how the cluster was installed.",
    )
    @click.option(
        "--manager",
        default=None,
        help="[dockerSwarm] Swarm manager SSH host (required for dockerSwarm).",
    )
    @click.option(
        "--workers",
        default="",
        help="Ignored for down (kept for symmetry with install).",
    )
    @click.option(
        "--stack-name",
        default="arena",
        show_default=True,
        help="[dockerSwarm] Docker stack name to stop.",
    )
    @click.option(
        "--ssh-user",
        default=None,
        help="[dockerSwarm] SSH login for the manager.",
    )
    @click.option(
        "--ssh-extra-opts",
        default=None,
        help="[dockerSwarm] Extra ssh(1) arguments.",
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        default=False,
        help="Show the underlying commands and their full output.",
    )
    @click.pass_obj
    def down_cmd(
        config: CommandConfig,
        name: str,
        setup_type: str,
        manager: str | None,
        workers: str,
        stack_name: str,
        ssh_user: str | None,
        ssh_extra_opts: str | None,
        verbose: bool,
    ) -> None:
        """Stop on-prem workloads for CLASS_NAME without removing the deployment.

        **dockerSwarm** — scales every service in the stack to zero replicas; the
        stack definition remains on the manager.

        **helm** — scales deployments to zero replicas; the Helm release remains.

        Re-run ``arena on-prem install`` to bring workloads back.
        """
        _apply_verbosity(verbose=verbose)

        with arena_client(config) as client:
            run_on_prem_down(
                client,
                name=name.strip(),
                setup_type=setup_type,
                manager=manager.strip() if manager else None,
                workers=tuple(h.strip() for h in workers.split(",") if h.strip()),
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                stack_name=stack_name,
            )

    return down_cmd


def build_teardown_command() -> click.Command:
    """``arena on-prem teardown`` — reverse install (cluster + optional API cleanup).

    :returns: The configured ``teardown`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "teardown",
        context_settings={"max_content_width": 100},
    )
    @click.argument("name")
    @click.option(
        "--setup-type",
        "setup_type",
        default="dockerSwarm",
        show_default=True,
        type=click.Choice(["dockerSwarm", "helm"], case_sensitive=False),
        help="Must match how the cluster was installed.",
    )
    @click.option(
        "--manager",
        default=None,
        help="[dockerSwarm] Swarm manager SSH host (required unless --skip-cluster).",
    )
    @click.option(
        "--workers",
        default="",
        help="[dockerSwarm] Worker SSH hosts for --leave-swarm (required on multi-node clusters).",
    )
    @click.option(
        "--stack-name",
        default="arena",
        show_default=True,
        help="[dockerSwarm] ``docker stack rm`` name on the manager.",
    )
    @click.option(
        "--ssh-user",
        default=None,
        help="[dockerSwarm] SSH login for the manager.",
    )
    @click.option(
        "--ssh-extra-opts",
        default=None,
        help="[dockerSwarm] Extra ssh(1) arguments.",
    )
    @click.option(
        "--skip-cluster",
        is_flag=True,
        default=False,
        help="Skip Helm/Swarm teardown (use with --disable-provider if needed).",
    )
    @click.option(
        "--disable-provider",
        is_flag=True,
        default=False,
        help="Also disable the on-prem provider after teardown.",
    )
    @click.option(
        "--leave-swarm",
        is_flag=True,
        default=False,
        help="[dockerSwarm] After stack removal, run docker swarm leave --force on all hosts.",
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        default=False,
        help="Show the underlying commands and their full output.",
    )
    @click.pass_obj
    def teardown_cmd(
        config: CommandConfig,
        name: str,
        setup_type: str,
        manager: str | None,
        workers: str,
        stack_name: str,
        ssh_user: str | None,
        ssh_extra_opts: str | None,
        skip_cluster: bool,
        disable_provider: bool,
        leave_swarm: bool,
        verbose: bool,
    ) -> None:
        """Remove the on-prem deployment for CLASS_NAME.

        **helm** — ``helm uninstall`` using ``clusterName`` from the class deployment bundle
        (same release/namespace as ``setup.sh``).

        **dockerSwarm** — ``docker stack rm`` on ``--manager`` (default stack name ``arena``).

        Removes the deployment only; does not uninstall Docker/NVIDIA or delete the Arena
        resource class. Use ``arena on-prem classes delete`` when you want to remove the
        class from Arena.
        """
        _apply_verbosity(verbose=verbose)

        with arena_client(config) as client:
            run_on_prem_teardown(
                client,
                name=name.strip(),
                setup_type=setup_type,
                skip_cluster=skip_cluster,
                disable_provider=disable_provider,
                manager=manager.strip() if manager else None,
                workers=tuple(h.strip() for h in workers.split(",") if h.strip()),
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                stack_name=stack_name,
                leave_swarm=leave_swarm,
            )

    return teardown_cmd


def build_cluster_register_command() -> click.Command:
    """``arena on-prem cluster register`` — register a cluster and write Helm values.

    :returns: The configured ``register`` Click command.
    :rtype: click.Command
    """

    @click.command(
        "register",
        context_settings={"max_content_width": 100},
    )
    @click.option("--name", required=True, help="Cluster name.")
    @click.option(
        "--profile",
        "--install-profile",
        "profile",
        default="enterprise",
        show_default=True,
        type=click.Choice(["lab", "enterprise"], case_sensitive=False),
        help="Install profile: lab (bundled MinIO) or enterprise (customer S3).",
    )
    @click.option(
        "--storage-endpoint",
        default=None,
        help="S3-compatible endpoint (required for enterprise).",
    )
    @click.option(
        "--storage-bucket",
        default=None,
        help="Object bucket name (required for enterprise).",
    )
    @click.option(
        "--storage-prefix",
        default=None,
        help="Optional object key prefix.",
    )
    @click.option(
        "--storage-secret-name",
        default=None,
        help="Kubernetes secret with S3 credentials (required for enterprise).",
    )
    @click.option(
        "--ingress-class-name",
        default=None,
        help="Ingress class for inference routes.",
    )
    @click.option(
        "--hostname-template",
        default=None,
        help="Hostname template for inference routes.",
    )
    @click.option(
        "--gateway-api-parent-refs",
        default=None,
        help="Gateway API parent refs JSON (or @path.json).",
    )
    @click.option(
        "--tls-secret-name",
        default=None,
        help="TLS secret for inference routes.",
    )
    @click.option(
        "--preprocessing-resource-class-id",
        type=int,
        default=None,
        help="Preprocessing resource class id.",
    )
    @click.option(
        "--ray-data-storage-class-name",
        default=None,
        help="RWX storage class for shared Ray data PVC.",
    )
    @click.option(
        "--ray-data-pvc-size",
        default=None,
        help="Ray shared data PVC size.",
    )
    @click.option(
        "--output-dir",
        default=None,
        help="Directory for Helm values and token (default: ./arena-cluster-NAME).",
    )
    @click.option(
        "--skip-enable",
        is_flag=True,
        default=False,
        help="Skip enabling the on-prem provider (use when it is already enabled).",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Overwrite existing output files.",
    )
    @click.option(
        "--install",
        is_flag=True,
        default=False,
        help="Run helm upgrade --install for storage (lab) and agent charts.",
    )
    @click.option(
        "--charts-dir",
        default=None,
        help=(
            "Path to agilerl-platform/resources/helm-setup "
            "(or set ARENA_HELM_CHARTS_DIR)."
        ),
    )
    @click.option(
        "--no-helm-wait",
        is_flag=True,
        default=False,
        help="Do not pass --wait to helm upgrade --install.",
    )
    @click.option(
        "-v",
        "--verbose",
        is_flag=True,
        default=False,
        help="Show detailed command traces.",
    )
    @click.pass_obj
    def register_cmd(
        config: CommandConfig,
        name: str,
        profile: str,
        storage_endpoint: str | None,
        storage_bucket: str | None,
        storage_prefix: str | None,
        storage_secret_name: str | None,
        ingress_class_name: str | None,
        hostname_template: str | None,
        gateway_api_parent_refs: str | None,
        tls_secret_name: str | None,
        preprocessing_resource_class_id: int | None,
        ray_data_storage_class_name: str | None,
        ray_data_pvc_size: str | None,
        output_dir: str | None,
        skip_enable: bool,
        force: bool,
        install: bool,
        charts_dir: str | None,
        no_helm_wait: bool,
        verbose: bool,
    ) -> None:
        """Register a customer Kubernetes cluster and write Helm install bundles.

        **lab** — Arena defaults storage; writes ``storage-helm-values.yaml`` and
        ``agent-helm-values.yaml``. Use ``--install`` to install storage (MinIO +
        bucket bootstrap Job) then the agent chart with operator storage env.

        **enterprise** — requires ``--storage-endpoint``, ``--storage-bucket``, and
        ``--storage-secret-name``; writes ``agent-helm-values.yaml`` only.
        ``--install`` installs the agent chart.
        """
        _apply_verbosity(verbose=verbose)
        cluster_name = name.strip()
        out = Path(output_dir or f"./arena-cluster-{cluster_name}")
        charts = Path(charts_dir).expanduser() if charts_dir else None
        gateway_refs: object | None = None
        if gateway_api_parent_refs is not None:
            gateway_refs = _parse_json_cli_value(gateway_api_parent_refs)

        with arena_client(config) as client:
            run_cluster_register(
                client,
                name=cluster_name,
                profile=profile.lower(),
                output_dir=out,
                skip_enable=skip_enable,
                force=force,
                install=install,
                charts_dir=charts,
                helm_wait=not no_helm_wait,
                storage_endpoint=storage_endpoint,
                storage_bucket=storage_bucket,
                storage_prefix=storage_prefix,
                storage_secret_name=storage_secret_name,
                ingress_class_name=ingress_class_name,
                hostname_template=hostname_template,
                gateway_api_parent_refs=gateway_refs,
                tls_secret_name=tls_secret_name,
                preprocessing_resource_class_id=preprocessing_resource_class_id,
                ray_data_storage_class_name=ray_data_storage_class_name,
                ray_data_pvc_size=ray_data_pvc_size,
            )

    return register_cmd


def register_on_prem_cluster(on_prem_group: click.Group) -> None:
    """Replace manifest ``clusters`` subgroup; register ``cluster register``.

    :param on_prem_group: The ``on-prem`` group to attach the command to.
    :type on_prem_group: click.Group
    :returns: None
    :rtype: None
    """
    on_prem_group.commands.pop("clusters", None)

    @click.group(
        "cluster",
        help="Enterprise on-prem cluster registration.",
    )
    def cluster_group() -> None:
        """Register customer Kubernetes clusters."""

    cluster_group.add_command(build_cluster_register_command())
    on_prem_group.add_command(cluster_group)


def register_on_prem_install(on_prem_group: click.Group) -> None:
    """Replace manifest ``install`` subgroup; register install + teardown commands.

    :param on_prem_group: The ``on-prem`` group to attach the commands to.
    :type on_prem_group: click.Group
    :returns: None
    :rtype: None
    """
    on_prem_group.commands.pop("install", None)
    on_prem_group.add_command(build_install_command())
    on_prem_group.add_command(build_down_command())
    on_prem_group.add_command(build_teardown_command())
