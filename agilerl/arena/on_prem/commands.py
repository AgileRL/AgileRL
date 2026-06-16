"""The hardcoded ``arena on-prem install`` / ``teardown`` Click commands."""

from __future__ import annotations

import logging

import click

from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.on_prem.installer import (
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
        "--num-nodes",
        type=click.IntRange(1),
        default=None,
        help=(
            "Node count when creating a new class. "
            "Default: manager+workers for dockerSwarm, 1 for helm."
        ),
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
        num_nodes: int | None,
        ssh_user: str | None,
        ssh_extra_opts: str | None,
        advertise_addr: str | None,
        skip_enable: bool,
        skip_verify: bool,
        verbose: bool,
    ) -> None:
        """Install an on-prem worker cluster for CLASS_NAME.

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
                num_nodes=num_nodes,
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                advertise_addr=advertise_addr,
                skip_verify=skip_verify,
            )

    return install_cmd


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
        help="Ignored for teardown (kept for symmetry with install).",
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
        help="Only update Arena (delete class / disable provider); do not touch Helm or Swarm.",
    )
    @click.option(
        "--keep-class",
        is_flag=True,
        default=False,
        help="Remove cluster workloads but keep the Arena resource class.",
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
        keep_class: bool,
        disable_provider: bool,
        leave_swarm: bool,
        verbose: bool,
    ) -> None:
        """Tear down an on-prem install for CLASS_NAME.

        **helm** — ``helm uninstall`` using ``clusterName`` from the class deployment bundle
        (same release/namespace as ``setup.sh``).

        **dockerSwarm** — ``docker stack rm`` on ``--manager`` (default stack name ``arena``).

        By default also deletes the Arena on-prem resource class; use ``--keep-class`` to
        leave the class registered. Use ``--skip-cluster`` for API-only cleanup.
        """
        _apply_verbosity(verbose=verbose)

        with arena_client(config) as client:
            run_on_prem_teardown(
                client,
                name=name.strip(),
                setup_type=setup_type,
                skip_cluster=skip_cluster,
                delete_class=not keep_class,
                disable_provider=disable_provider,
                manager=manager.strip() if manager else None,
                workers=tuple(h.strip() for h in workers.split(",") if h.strip()),
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                stack_name=stack_name,
                leave_swarm=leave_swarm,
            )

    return teardown_cmd


def register_on_prem_install(on_prem_group: click.Group) -> None:
    """Replace manifest ``install`` subgroup; register install + teardown commands.

    :param on_prem_group: The ``on-prem`` group to attach the commands to.
    :type on_prem_group: click.Group
    :returns: None
    :rtype: None
    """
    on_prem_group.commands.pop("install", None)
    on_prem_group.add_command(build_install_command())
    on_prem_group.add_command(build_teardown_command())
