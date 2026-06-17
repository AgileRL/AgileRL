"""Provider-specific install/teardown orchestration.

:class:`OnPremInstaller` holds the shared install/teardown flow (enable, ensure
class, download bundle, …) as template methods; :class:`SwarmInstaller` and
:class:`HelmInstaller` fill in the cluster-specific steps. The module-level
:func:`run_on_prem_install` / :func:`run_on_prem_teardown` are a thin functional
facade over the classes.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import click

from agilerl.arena.client import ArenaClient
from agilerl.arena.on_prem.api import OnPremApi, resolve_num_nodes
from agilerl.arena.on_prem.bundle import (
    extract_bundle,
    parse_helm_release_ids,
    validate_wireguard_bundle,
)
from agilerl.arena.on_prem.endpoints import SetupKind
from agilerl.arena.on_prem.scripts import (
    BundleScriptRunner,
    StageFailed,
    stage_failure,
    swarm_script_env,
)
from agilerl.arena.on_prem.ssh import SshExecutor, SshTarget

logger = logging.getLogger("agilerl.arena.on_prem")

STACK_VERIFY_INTERVAL_SEC = 10
STACK_VERIFY_MAX_WAIT_SEC = 300


def stack_readiness_state(
    output: str | None,
    *,
    service_ps_output: str | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Parse ``docker stack services`` / ``docker service ps`` output.

    :param output: The ``docker stack services`` tab-separated output.
    :type output: str | None
    :param service_ps_output: Optional ``docker service ps`` output.
    :type service_ps_output: str | None
    :returns: ``(ready, not_ready, scheduling_errors)`` where *ready* is true
        when every service replica count matches and no scheduling errors appear.
    :rtype: tuple[bool, list[str], list[str]]
    """
    if not output or not output.strip():
        return False, [], []
    not_ready: list[str] = []
    for line in output.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        name, replicas = parts[0], parts[1]
        running, _, desired = replicas.partition("/")
        if desired and running != desired:
            not_ready.append(f"{name} {replicas}")
    scheduling_errors: list[str] = []
    if service_ps_output:
        for line in service_ps_output.splitlines():
            lower = line.lower()
            if "no suitable node" in lower or "insufficient resources" in lower:
                scheduling_errors.append(line.strip())
    ready = not not_ready and not scheduling_errors
    return ready, not_ready, scheduling_errors


def normalize_setup_type(setup_type: str) -> SetupKind:
    """Map CLI ``--setup-type`` to a supported bundle flavor.

    :param setup_type: The raw ``--setup-type`` value (case/dash-insensitive).
    :type setup_type: str
    :returns: The normalized flavor, ``"dockerSwarm"`` or ``"helm"``.
    :rtype: SetupKind
    :raises click.ClickException: If *setup_type* is not recognized.
    """
    key = setup_type.strip().lower().replace("-", "")
    if key in {"dockerswarm", "kubernetes"}:
        return "dockerSwarm"
    if key == "helm":
        return "helm"
    msg = f"Unsupported setup type {setup_type!r} (use dockerSwarm or helm)."
    raise click.ClickException(msg)


def all_hosts(manager: str, workers: tuple[str, ...]) -> list[str]:
    """Deduplicate and order ``manager`` + ``workers``, dropping blanks.

    :param manager: The Swarm manager host.
    :type manager: str
    :param workers: The worker hosts.
    :type workers: tuple[str, ...]
    :returns: The unique, stripped hosts in manager-first order.
    :rtype: list[str]
    """
    seen: set[str] = set()
    out: list[str] = []
    for h in (manager, *workers):
        h = h.strip()
        if not h or h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def warn_ignored_swarm_flags(
    *,
    manager: str | None,
    workers: tuple[str, ...],
    ssh_user: str | None,
    ssh_extra_opts: str | None,
    advertise_addr: str | None,
) -> None:
    """Warn that Swarm-only flags have no effect on the local Helm path.

    :param manager: The ``--manager`` value, if supplied.
    :type manager: str | None
    :param workers: The ``--workers`` hosts, if any.
    :type workers: tuple[str, ...]
    :param ssh_user: The ``--ssh-user`` value, if supplied.
    :type ssh_user: str | None
    :param ssh_extra_opts: The ``--ssh-extra-opts`` value, if supplied.
    :type ssh_extra_opts: str | None
    :param advertise_addr: The ``--advertise-addr`` value, if supplied.
    :type advertise_addr: str | None
    :returns: None
    :rtype: None
    """
    ignored: list[str] = []
    if manager:
        ignored.append("--manager")
    if workers:
        ignored.append("--workers")
    if ssh_user:
        ignored.append("--ssh-user")
    if ssh_extra_opts:
        ignored.append("--ssh-extra-opts")
    if advertise_addr:
        ignored.append("--advertise-addr")
    if ignored:
        logger.warning(
            "helm install ignores %s "
            "(helm install runs locally; your kubectl context must reach the cluster).",
            ", ".join(ignored),
        )


def report_stack_readiness(
    stack_name: str,
    output: str | None,
    *,
    service_ps_output: str | None = None,
) -> None:
    r"""Log whether every service in *output* (``name\treplicas`` lines) is up.

    :param stack_name: The Docker stack name being reported on.
    :type stack_name: str
    :param output: The ``docker stack services`` output, or ``None`` if unavailable.
    :type output: str | None
    :param service_ps_output: Optional ``docker service ps`` output for pending tasks.
    :type service_ps_output: str | None
    :returns: None
    :rtype: None
    """
    if not output or not output.strip():
        logger.warning(
            "Could not read service status for stack %r; check it manually with "
            "'docker stack services %s'.",
            stack_name,
            stack_name,
        )
        return
    ready, not_ready, scheduling_errors = stack_readiness_state(
        output, service_ps_output=service_ps_output
    )
    for err in scheduling_errors[:3]:
        logger.warning("Stack %r scheduling issue: %s", stack_name, err)
    if scheduling_errors:
        return
    if not_ready:
        logger.warning(
            "Stack %r not fully up yet: %s (services may still be starting).",
            stack_name,
            ", ".join(not_ready),
        )
    elif ready:
        logger.info("Stack %r is up; all services running.", stack_name)


class OnPremInstaller(ABC):
    """Shared install/teardown flow; subclasses provide the cluster-specific steps."""

    kind: ClassVar[SetupKind]

    def __init__(self, api: OnPremApi, *, name: str) -> None:
        """Bind the installer to an on-prem API client and resource class name.

        :param api: The on-prem API wrapper to issue Arena requests through.
        :type api: OnPremApi
        :param name: The resource class name to install or tear down.
        :type name: str
        """
        self.api = api
        self.name = name

    def install(
        self,
        *,
        num_nodes: int | None = None,
        skip_enable: bool = False,
        skip_verify: bool = False,
    ) -> None:
        """Enable on-prem, ensure the class exists, download the bundle, install.

        :param num_nodes: Node count when creating a new class; ``None`` uses the
            per-flavor default.
        :type num_nodes: int | None
        :param skip_enable: If ``True``, skip enabling the on-prem provider.
        :type skip_enable: bool
        :param skip_verify: If ``True``, skip post-install verification.
        :type skip_verify: bool
        :returns: None
        :rtype: None
        """
        self.preflight_install()

        if not skip_enable:
            self.api.enable()

        existing = self.api.find_class(self.name)
        nodes = resolve_num_nodes(
            existing, explicit=num_nodes, default=self.default_num_nodes()
        )
        self.api.ensure_class(self.name, num_nodes=nodes)

        with tempfile.TemporaryDirectory(prefix="arena-on-prem-") as tmp:
            data = self.api.fetch_bundle(self.name, self.kind)
            bundle_root = extract_bundle(data, Path(tmp), class_name=self.name)
            validate_wireguard_bundle(bundle_root, self.kind)
            self.install_cluster(bundle_root)
            if not skip_verify:
                self.verify(bundle_root)

        logger.info("On-prem install finished for class %r (%s).", self.name, self.kind)

    def teardown(
        self,
        *,
        skip_cluster: bool = False,
        delete_class: bool = True,
        disable_provider: bool = False,
    ) -> None:
        """Remove cluster workloads and optionally delete the class / disable on-prem.

        :param skip_cluster: If ``True``, do not touch the cluster (API-only cleanup).
        :type skip_cluster: bool
        :param delete_class: If ``True``, delete the Arena resource class.
        :type delete_class: bool
        :param disable_provider: If ``True``, disable the on-prem provider afterward.
        :type disable_provider: bool
        :returns: None
        :rtype: None
        """
        if not skip_cluster:
            self.teardown_cluster()
        if delete_class:
            self.api.delete_class(self.name)
        if disable_provider:
            self.api.disable()
        logger.info(
            "On-prem teardown finished for class %r (%s).", self.name, self.kind
        )

    # --- hooks ---------------------------------------------------------------
    @abstractmethod
    def preflight_install(self) -> None:
        """Validate prerequisites / warn about ignored flags before any API call.

        :returns: None
        :rtype: None
        """

    @abstractmethod
    def default_num_nodes(self) -> int:
        """Node count used when creating a new class with no explicit value.

        :returns: The per-flavor default node count.
        :rtype: int
        """

    @abstractmethod
    def install_cluster(self, bundle_root: Path) -> None:
        """Run the cluster install steps for an extracted bundle.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        """

    @abstractmethod
    def verify(self, bundle_root: Path) -> None:
        """Post-install verification (best-effort; warns on problems).

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        """

    @abstractmethod
    def teardown_cluster(self) -> None:
        """Remove cluster workloads (no-op pieces are the subclass's choice).

        :returns: None
        :rtype: None
        """


class SwarmInstaller(OnPremInstaller):
    """Docker Swarm install over SSH on a manager + optional workers."""

    kind: ClassVar[SetupKind] = "dockerSwarm"

    def __init__(
        self,
        api: OnPremApi,
        *,
        name: str,
        manager: str | None = None,
        workers: tuple[str, ...] = (),
        ssh_user: str | None = None,
        ssh_extra_opts: str | None = None,
        advertise_addr: str | None = None,
        stack_name: str = "arena",
        leave_swarm: bool = False,
    ) -> None:
        """Configure a Docker Swarm installer.

        :param api: The on-prem API wrapper.
        :type api: OnPremApi
        :param name: The resource class name.
        :type name: str
        :param manager: The Swarm manager SSH host (required at point of use).
        :type manager: str | None
        :param workers: The worker SSH hosts.
        :type workers: tuple[str, ...]
        :param ssh_user: SSH login for remote hosts, or ``None`` for ssh_config.
        :type ssh_user: str | None
        :param ssh_extra_opts: Extra ssh(1) arguments, or ``None``.
        :type ssh_extra_opts: str | None
        :param advertise_addr: Swarm ``--advertise-addr``; defaults to *manager*.
        :type advertise_addr: str | None
        :param stack_name: The Docker stack name for deploy/verify/teardown.
        :type stack_name: str
        :param leave_swarm: If ``True``, run ``docker swarm leave`` during teardown.
        :type leave_swarm: bool
        """
        super().__init__(api, name=name)
        self._manager = manager.strip() if manager else None
        self._workers = workers
        self._advertise_addr = advertise_addr
        self._stack_name = stack_name
        self._leave_swarm = leave_swarm
        self._ssh_user = ssh_user
        self._ssh_extra_opts = ssh_extra_opts
        self._executor = SshExecutor(ssh_user=ssh_user, ssh_extra_opts=ssh_extra_opts)

    def _require_manager(self, message: str) -> str:
        """Return the configured manager host, or raise with *message* if unset.

        :param message: The error message to raise when no manager is configured.
        :type message: str
        :returns: The (stripped) manager host.
        :rtype: str
        :raises click.ClickException: If no manager host is configured.
        """
        if not self._manager:
            raise click.ClickException(message)
        return self._manager

    def preflight_install(self) -> None:
        """Require a manager host and ``ssh`` on PATH before installing.

        :returns: None
        :rtype: None
        :raises click.ClickException: If no manager is set or ssh is missing.
        """
        self._require_manager(
            "--manager is required for dockerSwarm "
            "(SSH hostname or IP of the swarm manager)."
        )
        if not shutil.which("ssh"):
            msg = "ssh not found on PATH; required for dockerSwarm install."
            raise click.ClickException(msg)

    def default_num_nodes(self) -> int:
        """Default to the number of distinct manager + worker hosts.

        :returns: ``max(1, number of hosts)``.
        :rtype: int
        """
        manager = self._require_manager(
            "--manager is required for dockerSwarm "
            "(SSH hostname or IP of the swarm manager)."
        )
        return max(1, len(all_hosts(manager, self._workers)))

    def install_cluster(self, bundle_root: Path) -> None:
        """Run the ordered Swarm install stages over SSH.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        :raises click.ClickException: If a stage script exits non-zero.
        """
        manager = self._require_manager(
            "--manager is required for dockerSwarm "
            "(SSH hostname or IP of the swarm manager)."
        )
        env = swarm_script_env()
        if self._ssh_user:
            env["SSH_USER"] = self._ssh_user
        else:
            # Let bundle scripts use ssh(1) Host aliases (User/IdentityFile in ssh_config).
            env.pop("SSH_USER", None)
        if self._ssh_extra_opts:
            env["SSH_EXTRA_OPTS"] = self._ssh_extra_opts
        adv_raw = (self._advertise_addr or manager).strip()
        adv = SshTarget.parse(adv_raw).hostname
        if (
            self._advertise_addr
            and SshTarget.parse(self._advertise_addr).port is not None
        ):
            logger.warning(
                "Using Swarm advertise-addr %r (port stripped from %r). "
                "Swarm uses port 2377; do not pass SSH ports here.",
                adv,
                self._advertise_addr,
            )
        elif (
            self._advertise_addr is None
            and re.search(r":\d+$", manager.strip())
            and adv != manager.strip()
        ):
            logger.warning(
                "Using Swarm advertise-addr %r (SSH port stripped from manager %r). "
                "Pass --advertise-addr explicitly if Swarm should use a different IP.",
                adv,
                manager,
            )
        env["SWARM_MANAGER_HOST"] = manager
        env["SWARM_ADVERTISE_ADDR"] = adv
        tokens_file = bundle_root / "swarm-tokens.txt"
        env["TOKENS_FILE"] = str(tokens_file)

        hosts = all_hosts(manager, self._workers)
        worker_only = [h for h in self._workers if h.strip() and h.strip() != manager]
        label_hosts = [manager, *worker_only]

        # Ordered (label, script, args). The join stage only applies with workers.
        stages: list[tuple[str, str, list[str]]] = [
            ("Installing Docker Engine", "install-docker.sh", hosts),
            ("Installing NVIDIA driver", "install-nvidia-driver.sh", hosts),
            (
                "Installing NVIDIA Container Toolkit",
                "install-nvidia-container-toolkit.sh",
                hosts,
            ),
            ("Initializing Docker Swarm", "init-docker-swarm.sh", [manager, adv]),
        ]
        if worker_only:
            stages.append(
                (
                    "Joining workers to the Swarm",
                    "join-docker-swarm.sh",
                    ["--tokens-file", str(tokens_file), *worker_only],
                )
            )
        stages.append(
            ("Labelling GPU nodes", "label-docker-swarm-gpus.sh", label_hosts)
        )
        stages.append(("Deploying Arena stack", "deploy-arena-stack.sh", [manager]))

        runner = BundleScriptRunner(bundle_root, env=env)
        total = len(stages)
        logger.info("Installing cluster on %s (%d stages)…", manager, total)
        for index, (label, script_name, script_args) in enumerate(stages, start=1):
            logger.info("[%d/%d] %s", index, total, label)
            try:
                runner.run(script_name, script_args)
            except StageFailed as exc:
                raise stage_failure(
                    label, manager, exc, index=index, total=total
                ) from exc

    def verify(self, bundle_root: Path) -> None:
        """Query the deployed stack and fail if services do not become ready.

        :param bundle_root: The extracted bundle root directory (unused; kept for
            interface symmetry with :class:`HelmInstaller`).
        :type bundle_root: Path
        :returns: None
        :rtype: None
        :raises click.ClickException: If services never reach the desired replica
            count or Swarm reports scheduling errors.
        """
        manager = self._require_manager(
            "--manager is required for dockerSwarm "
            "(SSH hostname or IP of the swarm manager)."
        )
        stack_name = self._stack_name
        logger.info("Verifying Docker stack %r on %s…", stack_name, manager)
        remote_cmd = (
            f"sudo docker stack services {shlex.quote(stack_name)} "
            "--format '{{.Name}}\\t{{.Replicas}}'"
        )
        ps_cmd = (
            f"sudo docker service ps {shlex.quote(f'{stack_name}_ray-worker')} "
            f"{shlex.quote(f'{stack_name}_ray-head')} --no-trunc 2>/dev/null || true"
        )
        deadline = time.monotonic() + STACK_VERIFY_MAX_WAIT_SEC
        last_not_ready: list[str] = []
        while True:
            output = self._executor.run(manager, remote_cmd, capture=True)
            ps_output = self._executor.run(manager, ps_cmd, capture=True)
            ready, not_ready, scheduling_errors = stack_readiness_state(
                output, service_ps_output=ps_output
            )
            if scheduling_errors:
                report_stack_readiness(stack_name, output, service_ps_output=ps_output)
                msg = (
                    f"Stack {stack_name!r} has scheduling errors; "
                    f"check 'docker service ps {stack_name}_ray-worker' on the manager."
                )
                raise click.ClickException(msg)
            if ready:
                report_stack_readiness(stack_name, output, service_ps_output=ps_output)
                return
            last_not_ready = not_ready
            if time.monotonic() >= deadline:
                break
            logger.info(
                "Waiting for stack %r (%s)…",
                stack_name,
                ", ".join(not_ready) if not_ready else "services starting",
            )
            time.sleep(STACK_VERIFY_INTERVAL_SEC)
        report_stack_readiness(stack_name, output, service_ps_output=ps_output)
        detail = ", ".join(last_not_ready) if last_not_ready else "unknown"
        msg = (
            f"Stack {stack_name!r} not ready after {STACK_VERIFY_MAX_WAIT_SEC}s "
            f"({detail}). Check 'docker stack services {stack_name}' on the manager."
        )
        raise click.ClickException(msg)

    def teardown_cluster(self) -> None:
        """Remove the Docker stack, optionally leaving the Swarm on every host.

        :returns: None
        :rtype: None
        :raises click.ClickException: If no manager host is configured.
        """
        manager = self._require_manager(
            "--manager is required for dockerSwarm teardown (unless --skip-cluster)."
        )
        logger.info("Removing Docker stack %r on %s…", self._stack_name, manager)
        self._executor.run(
            manager, f"sudo docker stack rm {shlex.quote(self._stack_name)}"
        )
        if not self._leave_swarm:
            return
        logger.info("Leaving Docker Swarm on cluster hosts…")
        for host in all_hosts(manager, self._workers):
            self._executor.run(
                host, "sudo docker swarm leave --force 2>/dev/null || true"
            )


class HelmInstaller(OnPremInstaller):
    """Local ``helm upgrade --install`` driven by the bundle's setup.sh."""

    kind: ClassVar[SetupKind] = "helm"

    def __init__(
        self,
        api: OnPremApi,
        *,
        name: str,
        manager: str | None = None,
        workers: tuple[str, ...] = (),
        ssh_user: str | None = None,
        ssh_extra_opts: str | None = None,
        advertise_addr: str | None = None,
    ) -> None:
        """Configure a local Helm installer.

        The Swarm-only options are accepted only so they can be reported as
        ignored; the Helm path runs locally against the active kubectl context.

        :param api: The on-prem API wrapper.
        :type api: OnPremApi
        :param name: The resource class name.
        :type name: str
        :param manager: Ignored Swarm manager host (warned about), if supplied.
        :type manager: str | None
        :param workers: Ignored Swarm worker hosts (warned about).
        :type workers: tuple[str, ...]
        :param ssh_user: Ignored SSH login (warned about), if supplied.
        :type ssh_user: str | None
        :param ssh_extra_opts: Ignored ssh(1) arguments (warned about), if supplied.
        :type ssh_extra_opts: str | None
        :param advertise_addr: Ignored Swarm advertise address (warned about).
        :type advertise_addr: str | None
        """
        super().__init__(api, name=name)
        self._manager = manager
        self._workers = workers
        self._ssh_user = ssh_user
        self._ssh_extra_opts = ssh_extra_opts
        self._advertise_addr = advertise_addr

    def preflight_install(self) -> None:
        """Warn about any Swarm-only flags that the Helm path will ignore.

        :returns: None
        :rtype: None
        """
        warn_ignored_swarm_flags(
            manager=self._manager,
            workers=self._workers,
            ssh_user=self._ssh_user,
            ssh_extra_opts=self._ssh_extra_opts,
            advertise_addr=self._advertise_addr,
        )

    def default_num_nodes(self) -> int:
        """Helm clusters default to a single node.

        :returns: ``1``.
        :rtype: int
        """
        return 1

    def install_cluster(self, bundle_root: Path) -> None:
        """Run the bundle's ``setup.sh`` against the local kubectl context.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``setup.sh`` or ``helm`` is missing, or
            the script exits non-zero.
        """
        setup = bundle_root / "setup.sh"
        if not setup.is_file():
            msg = (
                "Helm bundle has no setup.sh. Re-run ``arena on-prem install`` or "
                "use ``--setup-type dockerSwarm`` for SSH-based install."
            )
            raise click.ClickException(msg)
        if not shutil.which("helm"):
            msg = (
                "helm not found on PATH; install Helm 3.x or use "
                "--setup-type dockerSwarm."
            )
            raise click.ClickException(msg)
        logger.info("Running Helm setup (local kubectl context)…")
        runner = BundleScriptRunner(bundle_root, env=os.environ.copy())
        try:
            runner.run("setup.sh", [])
        except StageFailed as exc:
            err = stage_failure("Helm setup", "local", exc)
            raise err from exc

    def verify(self, bundle_root: Path) -> None:
        """Run the bundle's ``validate.sh`` if present, else warn to check kubectl.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``validate.sh`` exits non-zero.
        """
        validate = bundle_root / "validate.sh"
        if not validate.is_file():
            logger.warning("Bundle has no validate.sh; check pods with kubectl.")
            return
        logger.info("Running Helm post-install validation…")
        runner = BundleScriptRunner(bundle_root, env=os.environ.copy())
        try:
            runner.run("validate.sh", [])
        except StageFailed as exc:
            err = stage_failure("Helm post-install validation", "local", exc)
            raise err from exc

    def teardown_cluster(self) -> None:
        """Download the bundle to resolve release ids, then ``helm uninstall``.

        :returns: None
        :rtype: None
        """
        warn_ignored_swarm_flags(
            manager=self._manager,
            workers=self._workers,
            ssh_user=self._ssh_user,
            ssh_extra_opts=self._ssh_extra_opts,
            advertise_addr=None,
        )
        with tempfile.TemporaryDirectory(prefix="arena-on-prem-teardown-") as tmp:
            data = self.api.fetch_bundle(self.name, self.kind)
            bundle_root = extract_bundle(data, Path(tmp), class_name=self.name)
            release, namespace = parse_helm_release_ids(bundle_root)
            self._helm_uninstall(release, namespace)

    @staticmethod
    def _helm_uninstall(release: str, namespace: str) -> None:
        """Run ``helm uninstall`` for *release* in *namespace* (best-effort).

        :param release: The Helm release name.
        :type release: str
        :param namespace: The Kubernetes namespace.
        :type namespace: str
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``helm`` is not on PATH.
        """
        if not shutil.which("helm"):
            msg = "helm not found on PATH; install Helm 3.x or use --skip-cluster."
            raise click.ClickException(msg)
        logger.info("Removing Helm release %r (namespace %s)…", release, namespace)
        result = subprocess.run(
            ["helm", "uninstall", release, "--namespace", namespace],
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "helm uninstall exited %d (release may already be removed).",
                result.returncode,
            )


def build_installer(
    kind: SetupKind,
    api: OnPremApi,
    *,
    name: str,
    manager: str | None = None,
    workers: tuple[str, ...] = (),
    ssh_user: str | None = None,
    ssh_extra_opts: str | None = None,
    advertise_addr: str | None = None,
    stack_name: str = "arena",
    leave_swarm: bool = False,
) -> OnPremInstaller:
    """Construct the installer for *kind* (``dockerSwarm`` or ``helm``).

    :param kind: The bundle flavor to build an installer for.
    :type kind: SetupKind
    :param api: The on-prem API wrapper.
    :type api: OnPremApi
    :param name: The resource class name.
    :type name: str
    :param manager: The Swarm manager host (Swarm only; ignored for Helm).
    :type manager: str | None
    :param workers: The Swarm worker hosts (Swarm only; ignored for Helm).
    :type workers: tuple[str, ...]
    :param ssh_user: SSH login for remote hosts (Swarm only).
    :type ssh_user: str | None
    :param ssh_extra_opts: Extra ssh(1) arguments (Swarm only).
    :type ssh_extra_opts: str | None
    :param advertise_addr: Swarm ``--advertise-addr`` (Swarm only).
    :type advertise_addr: str | None
    :param stack_name: The Docker stack name (Swarm only).
    :type stack_name: str
    :param leave_swarm: Whether teardown leaves the Swarm (Swarm only).
    :type leave_swarm: bool
    :returns: A :class:`SwarmInstaller` or :class:`HelmInstaller`.
    :rtype: OnPremInstaller
    """
    if kind == "dockerSwarm":
        return SwarmInstaller(
            api,
            name=name,
            manager=manager,
            workers=workers,
            ssh_user=ssh_user,
            ssh_extra_opts=ssh_extra_opts,
            advertise_addr=advertise_addr,
            stack_name=stack_name,
            leave_swarm=leave_swarm,
        )
    return HelmInstaller(
        api,
        name=name,
        manager=manager,
        workers=workers,
        ssh_user=ssh_user,
        ssh_extra_opts=ssh_extra_opts,
        advertise_addr=advertise_addr,
    )


def run_on_prem_install(
    client: ArenaClient,
    *,
    name: str,
    setup_type: str,
    skip_enable: bool,
    manager: str | None = None,
    workers: tuple[str, ...] = (),
    ssh_user: str | None = None,
    ssh_extra_opts: str | None = None,
    advertise_addr: str | None = None,
    num_nodes: int | None = None,
    skip_verify: bool = False,
) -> None:
    """Enable on-prem, ensure class exists, download bundle, run install scripts.

    :param client: The authenticated Arena client.
    :type client: ArenaClient
    :param name: The resource class name to install.
    :type name: str
    :param setup_type: The bundle flavor (``dockerSwarm`` or ``helm``).
    :type setup_type: str
    :param skip_enable: If ``True``, skip enabling the on-prem provider.
    :type skip_enable: bool
    :param manager: The Swarm manager SSH host (dockerSwarm only).
    :type manager: str | None
    :param workers: The Swarm worker SSH hosts (dockerSwarm only).
    :type workers: tuple[str, ...]
    :param ssh_user: SSH login for remote hosts (dockerSwarm only).
    :type ssh_user: str | None
    :param ssh_extra_opts: Extra ssh(1) arguments (dockerSwarm only).
    :type ssh_extra_opts: str | None
    :param advertise_addr: Swarm ``--advertise-addr`` (dockerSwarm only).
    :type advertise_addr: str | None
    :param num_nodes: Node count when creating a new class; ``None`` uses the default.
    :type num_nodes: int | None
    :param skip_verify: If ``True``, skip post-install verification.
    :type skip_verify: bool
    :returns: None
    :rtype: None
    """
    kind = normalize_setup_type(setup_type)
    api = OnPremApi(client)
    installer = build_installer(
        kind,
        api,
        name=name,
        manager=manager,
        workers=workers,
        ssh_user=ssh_user,
        ssh_extra_opts=ssh_extra_opts,
        advertise_addr=advertise_addr,
        stack_name=os.environ.get("ARENA_STACK_NAME", "arena"),
    )
    installer.install(
        num_nodes=num_nodes, skip_enable=skip_enable, skip_verify=skip_verify
    )


def run_on_prem_teardown(
    client: ArenaClient,
    *,
    name: str,
    setup_type: str,
    skip_cluster: bool,
    delete_class: bool,
    disable_provider: bool,
    manager: str | None = None,
    workers: tuple[str, ...] = (),
    ssh_user: str | None = None,
    ssh_extra_opts: str | None = None,
    stack_name: str = "arena",
    leave_swarm: bool = False,
) -> None:
    """Remove cluster workloads and optionally delete the Arena on-prem class.

    :param client: The authenticated Arena client.
    :type client: ArenaClient
    :param name: The resource class name to tear down.
    :type name: str
    :param setup_type: The bundle flavor (``dockerSwarm`` or ``helm``).
    :type setup_type: str
    :param skip_cluster: If ``True``, do not touch the cluster (API-only cleanup).
    :type skip_cluster: bool
    :param delete_class: If ``True``, delete the Arena resource class.
    :type delete_class: bool
    :param disable_provider: If ``True``, disable the on-prem provider afterward.
    :type disable_provider: bool
    :param manager: The Swarm manager SSH host (dockerSwarm only).
    :type manager: str | None
    :param workers: The Swarm worker SSH hosts (dockerSwarm only).
    :type workers: tuple[str, ...]
    :param ssh_user: SSH login for the manager (dockerSwarm only).
    :type ssh_user: str | None
    :param ssh_extra_opts: Extra ssh(1) arguments (dockerSwarm only).
    :type ssh_extra_opts: str | None
    :param stack_name: The Docker stack name to remove (dockerSwarm only).
    :type stack_name: str
    :param leave_swarm: If ``True``, leave the Swarm on all hosts (dockerSwarm only).
    :type leave_swarm: bool
    :returns: None
    :rtype: None
    """
    kind = normalize_setup_type(setup_type)
    api = OnPremApi(client)
    installer = build_installer(
        kind,
        api,
        name=name,
        manager=manager,
        workers=workers,
        ssh_user=ssh_user,
        ssh_extra_opts=ssh_extra_opts,
        stack_name=stack_name,
        leave_swarm=leave_swarm,
    )
    installer.teardown(
        skip_cluster=skip_cluster,
        delete_class=delete_class,
        disable_provider=disable_provider,
    )
