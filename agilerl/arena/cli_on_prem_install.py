"""``arena on-prem install`` / ``teardown`` — cluster scripts + Arena on-prem API."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

import click

from agilerl.arena.client import ArenaClient, ManifestInvoke
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAPIError

logger = logging.getLogger(__name__)

SetupKind = Literal["dockerSwarm", "helm"]

_ENABLE_INVOKE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/enable",
    "responseKind": "json",
    "params": [],
}

_LIST_CLASSES_INVOKE: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/list",
    "responseKind": "json",
    "params": [],
}

_CREATE_CLASS_INVOKE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/classes/create",
    "responseKind": "json",
    "params": [],
}

_BUNDLE_INVOKE: ManifestInvoke = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/deployment-setup",
    "responseKind": "binary",
    "params": [],
}

_DELETE_CLASS_INVOKE: ManifestInvoke = {
    "method": "DELETE",
    "path": "/api/cli/v1/on-prem/classes/delete",
    "responseKind": "json",
    # The endpoint reads ``name`` from the query string, so declare it explicitly
    # rather than relying on the method-based fallback (which routes DELETE
    # payloads to the JSON body).
    "params": [{"name": "name", "in": "query", "type": "string", "required": True}],
}

_DISABLE_INVOKE: ManifestInvoke = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/disable",
    "responseKind": "json",
    "params": [],
}

_DEFAULT_METADATA: dict[str, Any] = {
    "computeResource": {
        "numCpus": 8,
        "numGpus": 0,
        "memoryBytes": "64 GiB",
    },
}

# Helm deploy is fast; gateway daemon polls Arena API on an interval (default 15s).
_DEFAULT_GATEWAY_WAIT_HELM_SECS = 75
# Swarm fresh-node install usually exceeds one gateway sync cycle.
_DEFAULT_GATEWAY_WAIT_SWARM_SECS = 0


class _StageFailed(Exception):
    """A bundle script exited non-zero; carries its captured output for reporting."""

    def __init__(self, script: str, returncode: int, output: str) -> None:
        self.script = script
        self.returncode = returncode
        self.output = output
        super().__init__(f"{script} exited {returncode}")


def _stage_failure(
    label: str,
    host: str,
    exc: _StageFailed,
    *,
    index: int | None = None,
    total: int | None = None,
) -> click.ClickException:
    """Turn a :class:`_StageFailed` into a clean, stage-named ClickException."""
    where = f"Stage {index}/{total} {label!r}" if index is not None else repr(label)
    tail = exc.output.strip()
    body = f"\n--- captured output ---\n{tail}" if tail else ""
    return click.ClickException(
        f"{where} failed on {host} (exit {exc.returncode}).{body}"
    )


def _apply_verbosity(*, verbose: bool) -> None:
    """``--verbose`` raises the Arena logger to DEBUG so command traces and live
    per-stage script output are shown instead of hidden.
    """
    if verbose:
        logging.getLogger("agilerl.arena").setLevel(logging.DEBUG)


def normalize_setup_type(setup_type: str) -> SetupKind:
    """Map CLI ``--setup-type`` to a supported bundle flavor."""
    key = setup_type.strip().lower().replace("-", "")
    if key in {"dockerswarm", "kubernetes"}:
        return "dockerSwarm"
    if key == "helm":
        return "helm"
    msg = f"Unsupported setup type {setup_type!r} (use dockerSwarm or helm)."
    raise click.ClickException(msg)


def _class_by_name(classes: object, name: str) -> dict[str, Any] | None:
    """Return the single resource class named *name*, or ``None`` if absent.

    *classes* is the raw ``classes/list`` response, so it is typed ``object``
    and validated to be a list here. Raises if the name is ambiguous.
    """
    if not isinstance(classes, list):
        return None
    matches = [c for c in classes if isinstance(c, dict) and c.get("name") == name]
    if not matches:
        return None
    if len(matches) > 1:
        msg = f"Multiple on-prem classes named {name!r}; resolve duplicates in Arena first."
        raise ArenaAPIError(msg)
    return matches[0]


def _num_nodes_for_create(
    existing: dict[str, Any] | None,
    *,
    kind: SetupKind,
    explicit: int | None,
    manager: str | None,
    workers: tuple[str, ...],
) -> int:
    """Decide the node count when creating a class.

    Prefers an existing class's ``num_nodes``, then an explicit ``--num-nodes``,
    then a per-flavor default (1 for helm; manager + workers for dockerSwarm).
    """
    if existing is not None:
        raw = existing.get("num_nodes")
        if isinstance(raw, int) and raw > 0:
            return raw
    if explicit is not None:
        return explicit
    if kind == "helm":
        return 1
    assert manager is not None
    return max(1, len(_all_hosts(manager, workers)))


def _ensure_class(
    client: ArenaClient,
    *,
    name: str,
    num_nodes: int,
) -> dict[str, Any]:
    """Return the existing resource class named *name*, creating it if needed."""
    listed = client._invoke_manifest_command(_LIST_CLASSES_INVOKE, {})
    existing = _class_by_name(listed, name)
    if existing is not None:
        logger.info("Using existing resource class %r.", name)
        return existing

    logger.info("Creating resource class %r (%d nodes)…", name, num_nodes)
    body: dict[str, Any] = {
        "name": name,
        "num_nodes": num_nodes,
        "enabled": True,
        "metadata": _DEFAULT_METADATA,
    }
    created = client._invoke_manifest_command(_CREATE_CLASS_INVOKE, body)
    if isinstance(created, dict):
        return created
    msg = "Create class response was not an object."
    raise ArenaAPIError(msg)


def resolve_bundle_root(extract_dir: Path) -> Path:
    """Find directory containing ``setup.sh`` after unzipping a deployment bundle.

    Bundles built by the Arena backend use the root prefix ``arena-train/``.
    Extracting into a pre-created ``…/arena-train`` folder would nest
    ``arena-train/arena-train/``.
    """
    for candidate in (extract_dir / "arena-train", extract_dir):
        if (candidate / "setup.sh").is_file():
            return candidate
    for setup in extract_dir.rglob("setup.sh"):
        parent = setup.parent
        if (parent / "chart").is_dir() or (parent / "install-docker.sh").is_file():
            return parent
    return extract_dir / "arena-train"


def _download_bundle(
    client: ArenaClient,
    *,
    class_name: str,
    setup_type: SetupKind,
    dest_dir: Path,
) -> Path:
    """Download and unzip the deployment bundle, returning its extracted root."""
    raw_b, _ctype, _disp = client._invoke_manifest_command(
        _BUNDLE_INVOKE,
        {
            "name": class_name,
            "setupType": setup_type,
            "archivedType": "zip",
        },
    )
    archive = dest_dir / f"{class_name}-setup.zip"
    archive.write_bytes(raw_b)
    extract_dir = dest_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)
    bundle_root = resolve_bundle_root(extract_dir)
    _prepare_bundle_scripts(bundle_root)
    return bundle_root


def _prepare_bundle_scripts(bundle_root: Path) -> None:
    """Zip extracts often drop the executable bit; run via bash and chmod for safety."""
    for path in bundle_root.rglob("*.sh"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o755)


def _shell_runner() -> str:
    bash = shutil.which("bash")
    if bash:
        return bash
    sh = shutil.which("sh")
    if sh:
        return sh
    msg = "bash or sh not found on PATH; required to run install scripts."
    raise click.ClickException(msg)


def _run_script(
    script: Path,
    args: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
) -> None:
    """Run a bundle script, raising :class:`_StageFailed` on a non-zero exit.

    Quiet by default: the script's stdout (its chatty progress) is captured and
    surfaced only on failure. stderr stays attached so ``sudo`` prompts and real
    errors remain visible/live. Run with ``--verbose`` (logger at DEBUG) to
    stream everything as it happens.
    """
    if not script.is_file():
        msg = f"Install bundle missing script {script.name} (re-download with arena on-prem install)."
        raise click.ClickException(msg)
    runner = _shell_runner()
    cmd = [runner, str(script.resolve()), *args]
    # markup disabled: paths/args in the transcript may contain Rich-special chars.
    logger.debug("$ %s", " ".join(cmd), extra={"markup": False})
    if logger.isEnabledFor(logging.DEBUG):
        result = subprocess.run(cmd, check=False, cwd=cwd, env=env)
        captured = ""
    else:
        result = subprocess.run(
            cmd, check=False, cwd=cwd, env=env, stdout=subprocess.PIPE, text=True
        )
        captured = result.stdout or ""
    if result.returncode != 0:
        raise _StageFailed(script.name, result.returncode, captured)


def _all_hosts(manager: str, workers: tuple[str, ...]) -> list[str]:
    hosts = [manager, *workers]
    seen: set[str] = set()
    out: list[str] = []
    for h in hosts:
        h = h.strip()
        if not h or h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def _swarm_script_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Non-interactive env for remote install scripts (fresh VMs may need reboot)."""
    env = dict(base if base is not None else os.environ)
    env.setdefault("DOCKER_REBOOT_ASSUME_YES", "1")
    return env


def _wait_for_gateway_peer_registration(seconds: int) -> None:
    if seconds <= 0:
        return
    logger.info(
        "Waiting %ds for Arena on-prem gateway to register the WireGuard peer "
        "(automatic API sync; no manual gateway steps)…",
        seconds,
    )
    time.sleep(seconds)


def _validate_tun0_conf(path: Path) -> None:
    if not path.is_file():
        msg = f"Bundle missing WireGuard config {path.name}."
        raise click.ClickException(msg)
    text = path.read_text(encoding="utf-8")
    required = (
        "[Interface]",
        "[Peer]",
        "PrivateKey = ",
        "PublicKey = ",
        "PresharedKey = ",
        "Endpoint = ",
        "AllowedIPs = ",
    )
    missing = [token for token in required if token not in text]
    if missing:
        msg = (
            f"Invalid {path.name} in install bundle (missing: {', '.join(missing)}). "
            "Re-create the on-prem class or re-download the bundle."
        )
        raise click.ClickException(msg)


def _validate_wireguard_bundle(bundle_root: Path, kind: SetupKind) -> None:
    if kind == "helm":
        values = bundle_root / "chart" / "values.yaml"
        if not values.is_file():
            msg = "Helm bundle missing chart/values.yaml."
            raise click.ClickException(msg)
        text = values.read_text(encoding="utf-8")
        for key in (
            "wireguard:",
            "gatewayHost:",
            "gatewayPublicKey:",
            "peerPrivateKey:",
            "peerIp:",
            "preSharedKey:",
        ):
            if key not in text:
                msg = f"Helm values.yaml missing {key!r} (WireGuard not rendered)."
                raise click.ClickException(msg)
    else:
        _validate_tun0_conf(bundle_root / "config.d" / "tun0.conf")
        stack = bundle_root / "arena-stack.yaml"
        if not stack.is_file():
            msg = "Docker Swarm bundle missing arena-stack.yaml."
            raise click.ClickException(msg)
        stack_text = stack.read_text(encoding="utf-8")
        if "/etc/wireguard/" not in stack_text:
            msg = "arena-stack.yaml does not mount WireGuard config (/etc/wireguard/)."
            raise click.ClickException(msg)


def _ssh_target_host(host: str) -> str:
    """Hostname part of ``HOST``, ``user@HOST``, or ``HOST:PORT``."""
    target = host.strip().split("@", 1)[-1]
    if target.startswith("[") and "]" in target:
        return target.split("]", 1)[0][1:]
    if ":" in target:
        return target.rsplit(":", 1)[0]
    return target


def _ssh_target_port(host: str) -> int | None:
    target = host.strip().split("@", 1)[-1]
    if target.startswith("[") and "]:" in target:
        port_s = target.split("]:", 1)[1]
    elif ":" in target and not target.startswith("["):
        port_s = target.rsplit(":", 1)[1]
    else:
        return None
    return int(port_s) if port_s.isdigit() else None


def _ssh_connection_target(host: str, ssh_user: str | None) -> str:
    """Build the ssh(1) destination so ``~/.ssh/config`` applies like ``ssh HOST``.

    When no user is forced, use the host alias as given (e.g. ``manager-head``) instead of
    ``$USER@manager-head``, which ignores ``User`` in ssh_config.
    """
    host = host.strip()
    explicit = (ssh_user or os.environ.get("SSH_USER") or "").strip()
    bare = _ssh_target_host(host)
    if explicit:
        return f"{explicit}@{bare}"
    if "@" in host:
        user = host.split("@", 1)[0]
        return f"{user}@{bare}"
    if _ssh_target_port(host) is not None:
        return bare
    return host


def _is_local_swarm_host(host: str) -> bool:
    """True when install scripts run commands on this machine (not real SSH)."""
    h = _ssh_target_host(host).lower()
    if h in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        short = socket.gethostname().split(".", 1)[0]
        fqdn = socket.getfqdn()
    except OSError:
        return False
    return h == short or h == fqdn or h == fqdn.split(".", 1)[0]


def _report_stack_readiness(stack_name: str, output: str | None) -> None:
    r"""Log whether every service in *output* (``name\treplicas`` lines) is up."""
    if not output or not output.strip():
        logger.warning(
            "Could not read service status for stack %r; check it manually with "
            "'docker stack services %s'.",
            stack_name,
            stack_name,
        )
        return
    not_ready: list[str] = []
    for line in output.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        name, replicas = parts[0], parts[1]
        running, _, desired = replicas.partition("/")
        if desired and running != desired:
            not_ready.append(f"{name} {replicas}")
    if not_ready:
        logger.warning(
            "Stack %r not fully up yet: %s (services may still be starting).",
            stack_name,
            ", ".join(not_ready),
        )
    else:
        logger.info("Stack %r is up; all services running.", stack_name)


def _verify_swarm_stack(
    manager: str,
    stack_name: str,
    *,
    ssh_user: str | None,
    ssh_extra_opts: str | None,
) -> None:
    logger.info("Verifying Docker stack %r on %s…", stack_name, manager)
    remote_cmd = (
        f"sudo docker stack services {shlex.quote(stack_name)} "
        "--format '{{.Name}}\\t{{.Replicas}}'"
    )
    output = _ssh_remote_command(
        manager,
        remote_cmd,
        ssh_user=ssh_user,
        ssh_extra_opts=ssh_extra_opts,
        capture=True,
    )
    _report_stack_readiness(stack_name, output)


def _run_helm_post_install(bundle_root: Path) -> None:
    validate = bundle_root / "validate.sh"
    if validate.is_file():
        logger.info("Running Helm post-install validation…")
        try:
            _run_script(validate, [], env=os.environ.copy(), cwd=bundle_root)
        except _StageFailed as exc:
            err = _stage_failure("Helm post-install validation", "local", exc)
            raise err from exc
    else:
        logger.warning("Bundle has no validate.sh; check pods with kubectl.")


def _warn_ignored_swarm_flags(
    *,
    manager: str | None,
    workers: tuple[str, ...],
    ssh_user: str | None,
    ssh_extra_opts: str | None,
    advertise_addr: str | None,
) -> None:
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


def _run_docker_swarm_install(
    bundle_root: Path,
    *,
    manager: str,
    workers: tuple[str, ...],
    ssh_user: str | None,
    ssh_extra_opts: str | None,
    advertise_addr: str | None,
) -> None:
    env = _swarm_script_env()
    if ssh_user:
        env["SSH_USER"] = ssh_user
    else:
        # Let bundle scripts use ssh(1) Host aliases (User/IdentityFile in ~/.ssh/config).
        env.pop("SSH_USER", None)
    if ssh_extra_opts:
        env["SSH_EXTRA_OPTS"] = ssh_extra_opts
    adv = (advertise_addr or manager).strip()
    env["SWARM_MANAGER_HOST"] = manager
    env["SWARM_ADVERTISE_ADDR"] = adv
    tokens_file = bundle_root / "swarm-tokens.txt"
    env["TOKENS_FILE"] = str(tokens_file)

    hosts = _all_hosts(manager, workers)
    worker_only = [h for h in workers if h.strip() and h.strip() != manager.strip()]
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
    stages.append(("Labelling GPU nodes", "label-docker-swarm-gpus.sh", label_hosts))
    stages.append(("Deploying Arena stack", "deploy-arena-stack.sh", [manager]))

    total = len(stages)
    logger.info("Installing cluster on %s (%d stages)…", manager, total)
    for index, (label, script_name, script_args) in enumerate(stages, start=1):
        logger.info("[%d/%d] %s", index, total, label)
        try:
            _run_script(
                bundle_root / script_name, script_args, env=env, cwd=bundle_root
            )
        except _StageFailed as exc:
            raise _stage_failure(label, manager, exc, index=index, total=total) from exc


def _parse_helm_release_ids(bundle_root: Path) -> tuple[str, str]:
    """Read ``clusterName`` from rendered chart values (matching the bundle's setup.sh)."""
    values_file = bundle_root / "chart" / "values.yaml"
    if not values_file.is_file():
        msg = f"Helm bundle missing {values_file.name}; cannot determine release name."
        raise click.ClickException(msg)
    text = values_file.read_text(encoding="utf-8")
    match = re.search(
        r"^clusterName:\s*[\"']?([^\"'\s]+)[\"']?\s*$",
        text,
        re.MULTILINE,
    )
    cluster_name = match.group(1) if match else "arena-train"
    release = os.environ.get("RELEASE_NAME", cluster_name)
    namespace = os.environ.get("NAMESPACE", cluster_name)
    return release, namespace


def _helm_uninstall(release: str, namespace: str) -> None:
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


def _ssh_remote_command(
    host: str,
    remote_cmd: str,
    *,
    ssh_user: str | None,
    ssh_extra_opts: str | None,
    capture: bool = False,
) -> str | None:
    """Run *remote_cmd* on *host* (locally or over ssh).

    ``remote_cmd`` is evaluated by a remote (or local) shell, so any
    caller-supplied values interpolated into it MUST be escaped with
    ``shlex.quote`` to avoid shell injection.

    With ``capture=True`` the command's stdout is captured and returned (stderr
    stays attached); otherwise output streams to the terminal and ``None`` is
    returned.
    """
    host = host.strip()
    run_kwargs: dict[str, Any] = {"check": False}
    if capture:
        run_kwargs["stdout"] = subprocess.PIPE
        run_kwargs["text"] = True

    if _is_local_swarm_host(host):
        # markup disabled: the command transcript can contain Rich-special chars.
        logger.debug("$ %s", remote_cmd, extra={"markup": False})
        result = subprocess.run(["bash", "-lc", remote_cmd], **run_kwargs)
        if result.returncode != 0:
            logger.warning("command exited %d", result.returncode)
        return result.stdout if capture else None

    if not shutil.which("ssh"):
        msg = "ssh not found on PATH; required for dockerSwarm teardown."
        raise click.ClickException(msg)

    ssh_port = _ssh_target_port(host)
    ssh_cmd = [
        "ssh",
        "-tt",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if ssh_port is not None:
        ssh_cmd.extend(["-p", str(ssh_port)])
    if ssh_extra_opts:
        ssh_cmd.extend(shlex.split(ssh_extra_opts))
    ssh_cmd.append(_ssh_connection_target(host, ssh_user))
    ssh_cmd.append(remote_cmd)
    # markup disabled: ssh targets like [::1] would be parsed as Rich markup.
    logger.debug("$ %s", " ".join(ssh_cmd), extra={"markup": False})
    result = subprocess.run(ssh_cmd, **run_kwargs)
    if result.returncode != 0:
        logger.warning("ssh exited %d", result.returncode)
    return result.stdout if capture else None


def _run_docker_swarm_teardown(
    *,
    manager: str,
    workers: tuple[str, ...],
    stack_name: str,
    ssh_user: str | None,
    ssh_extra_opts: str | None,
    leave_swarm: bool,
) -> None:
    logger.info("Removing Docker stack %r on %s…", stack_name, manager)
    _ssh_remote_command(
        manager,
        f"sudo docker stack rm {shlex.quote(stack_name)}",
        ssh_user=ssh_user,
        ssh_extra_opts=ssh_extra_opts,
    )
    if not leave_swarm:
        return
    hosts = _all_hosts(manager, workers)
    logger.info("Leaving Docker Swarm on cluster hosts…")
    for host in hosts:
        _ssh_remote_command(
            host,
            "sudo docker swarm leave --force 2>/dev/null || true",
            ssh_user=ssh_user,
            ssh_extra_opts=ssh_extra_opts,
        )


def _delete_class_if_present(client: ArenaClient, name: str) -> None:
    listed = client._invoke_manifest_command(_LIST_CLASSES_INVOKE, {})
    if _class_by_name(listed, name) is None:
        logger.info("No Arena resource class %r; skipping API delete.", name)
        return
    logger.info("Deleting on-prem resource class %r from Arena…", name)
    client._invoke_manifest_command(_DELETE_CLASS_INVOKE, {"name": name})


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
    """Remove cluster workloads and optionally delete the Arena on-prem class."""
    kind = normalize_setup_type(setup_type)

    if not skip_cluster:
        if kind == "dockerSwarm":
            if not manager or not manager.strip():
                msg = "--manager is required for dockerSwarm teardown (unless --skip-cluster)."
                raise click.ClickException(msg)
            _run_docker_swarm_teardown(
                manager=manager.strip(),
                workers=workers,
                stack_name=stack_name,
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                leave_swarm=leave_swarm,
            )
        else:
            _warn_ignored_swarm_flags(
                manager=manager,
                workers=workers,
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                advertise_addr=None,
            )
            with tempfile.TemporaryDirectory(prefix="arena-on-prem-teardown-") as tmp:
                bundle_root = _download_bundle(
                    client,
                    class_name=name,
                    setup_type=kind,
                    dest_dir=Path(tmp),
                )
                release, namespace = _parse_helm_release_ids(bundle_root)
                _helm_uninstall(release, namespace)

    if delete_class:
        _delete_class_if_present(client, name)

    if disable_provider:
        logger.info("Disabling on-prem provider…")
        client._invoke_manifest_command(_DISABLE_INVOKE, {})

    logger.info("On-prem teardown finished for class %r (%s).", name, kind)


def _run_helm_install(bundle_root: Path) -> None:
    setup = bundle_root / "setup.sh"
    if not setup.is_file():
        msg = (
            "Helm bundle has no setup.sh. Re-run ``arena on-prem install`` or "
            "use ``--setup-type dockerSwarm`` for SSH-based install."
        )
        raise click.ClickException(msg)
    if not shutil.which("helm"):
        msg = (
            "helm not found on PATH; install Helm 3.x or use --setup-type dockerSwarm."
        )
        raise click.ClickException(msg)
    logger.info("Running Helm setup (local kubectl context)…")
    try:
        _run_script(setup, [], env=os.environ.copy(), cwd=bundle_root)
    except _StageFailed as exc:
        err = _stage_failure("Helm setup", "local", exc)
        raise err from exc


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
    wait_gateway_secs: int | None = None,
    skip_verify: bool = False,
) -> None:
    """Enable on-prem, ensure class exists, download bundle, run install scripts."""
    kind = normalize_setup_type(setup_type)

    if kind == "dockerSwarm":
        if not manager or not manager.strip():
            msg = (
                "--manager is required for dockerSwarm "
                "(SSH hostname or IP of the swarm manager)."
            )
            raise click.ClickException(msg)
        if not shutil.which("ssh"):
            msg = "ssh not found on PATH; required for dockerSwarm install."
            raise click.ClickException(msg)
    else:
        _warn_ignored_swarm_flags(
            manager=manager,
            workers=workers,
            ssh_user=ssh_user,
            ssh_extra_opts=ssh_extra_opts,
            advertise_addr=advertise_addr,
        )

    if not skip_enable:
        logger.info("Enabling on-prem provider…")
        client._invoke_manifest_command(_ENABLE_INVOKE, {})

    listed = client._invoke_manifest_command(_LIST_CLASSES_INVOKE, {})
    existing = _class_by_name(listed, name)
    create_nodes = _num_nodes_for_create(
        existing,
        kind=kind,
        explicit=num_nodes,
        manager=manager.strip() if manager else None,
        workers=workers,
    )
    _ensure_class(client, name=name, num_nodes=create_nodes)

    if wait_gateway_secs is None:
        wait_gateway_secs = (
            _DEFAULT_GATEWAY_WAIT_HELM_SECS
            if kind == "helm"
            else _DEFAULT_GATEWAY_WAIT_SWARM_SECS
        )
    _wait_for_gateway_peer_registration(wait_gateway_secs)

    with tempfile.TemporaryDirectory(prefix="arena-on-prem-") as tmp:
        bundle_root = _download_bundle(
            client,
            class_name=name,
            setup_type=kind,
            dest_dir=Path(tmp),
        )
        _validate_wireguard_bundle(bundle_root, kind)
        if kind == "dockerSwarm":
            assert manager is not None
            _run_docker_swarm_install(
                bundle_root,
                manager=manager.strip(),
                workers=workers,
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
                advertise_addr=advertise_addr,
            )
            if not skip_verify:
                _verify_swarm_stack(
                    manager.strip(),
                    os.environ.get("ARENA_STACK_NAME", "arena"),
                    ssh_user=ssh_user,
                    ssh_extra_opts=ssh_extra_opts,
                )
        else:
            _run_helm_install(bundle_root)
            if not skip_verify:
                _run_helm_post_install(bundle_root)

    logger.info("On-prem install finished for class %r (%s).", name, kind)


def build_install_command() -> click.Command:
    """Primary ``arena on-prem install`` command."""

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
        "--wait-gateway-secs",
        type=click.IntRange(0),
        default=None,
        help=(
            "Seconds to wait after creating the class so the Arena gateway can register "
            "the WireGuard peer. Default: 75 for helm, 0 for dockerSwarm."
        ),
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
        wait_gateway_secs: int | None,
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

        from agilerl.arena.cli import arena_client

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
                wait_gateway_secs=wait_gateway_secs,
                skip_verify=skip_verify,
            )

    return install_cmd


def build_teardown_command() -> click.Command:
    """``arena on-prem teardown`` — reverse install (cluster + optional API cleanup)."""

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
        from agilerl.arena.cli import arena_client

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
    """Replace manifest ``install`` subgroup; register install + teardown commands."""
    on_prem_group.commands.pop("install", None)
    on_prem_group.add_command(build_install_command())
    on_prem_group.add_command(build_teardown_command())
