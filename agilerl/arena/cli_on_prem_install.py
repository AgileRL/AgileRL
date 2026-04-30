"""``arena on-prem install`` / ``teardown`` — cluster scripts + Arena on-prem API."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Literal

import click

from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAPIError

SetupKind = Literal["dockerSwarm", "helm"]

_ENABLE_INVOKE: dict[str, Any] = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/enable",
    "responseKind": "json",
    "params": [],
}

_LIST_CLASSES_INVOKE: dict[str, Any] = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/list",
    "responseKind": "json",
    "params": [],
}

_CREATE_CLASS_INVOKE: dict[str, Any] = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/classes/create",
    "responseKind": "json",
    "params": [],
}

_BUNDLE_INVOKE: dict[str, Any] = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/deployment-setup",
    "responseKind": "binary",
    "params": [],
}

_DELETE_CLASS_INVOKE: dict[str, Any] = {
    "method": "DELETE",
    "path": "/api/cli/v1/on-prem/classes/delete",
    "responseKind": "json",
    "params": [],
}

_DISABLE_INVOKE: dict[str, Any] = {
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


def normalize_setup_type(setup_type: str) -> SetupKind:
    """Map CLI ``--setup-type`` to a supported bundle flavor."""
    key = setup_type.strip().lower().replace("-", "")
    if key in {"dockerswarm", "kubernetes"}:
        return "dockerSwarm"
    if key == "helm":
        return "helm"
    msg = f"Unsupported setup type {setup_type!r} (use dockerSwarm or helm)."
    raise click.ClickException(msg)


def _class_by_name(classes: Any, name: str) -> dict[str, Any] | None:
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
    client: Any,
    *,
    name: str,
    num_nodes: int,
) -> dict[str, Any]:
    listed = client.invoke_manifest_command(_LIST_CLASSES_INVOKE, {})
    existing = _class_by_name(listed, name)
    if existing is not None:
        click.echo(f"Using existing resource class {name!r}.")
        return existing

    click.echo(f"Creating resource class {name!r} ({num_nodes} nodes)…")
    body: dict[str, Any] = {
        "name": name,
        "num_nodes": num_nodes,
        "enabled": True,
        "metadata": _DEFAULT_METADATA,
    }
    created = client.invoke_manifest_command(_CREATE_CLASS_INVOKE, body)
    if isinstance(created, dict):
        return created
    msg = "Create class response was not an object."
    raise ArenaAPIError(msg)


def resolve_bundle_root(extract_dir: Path) -> Path:
    """Find directory containing ``setup.sh`` after unzipping a deployment bundle.

    Platform archives use root prefix ``arena-train/`` (see ``build_bundle``). Extracting
    into a pre-created ``…/arena-train`` folder would nest ``arena-train/arena-train/``.
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
    client: Any,
    *,
    class_name: str,
    setup_type: SetupKind,
    dest_dir: Path,
) -> Path:
    raw_b, _ctype, _disp = client.invoke_manifest_command(
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
    if not script.is_file():
        msg = f"Install bundle missing script {script.name} (re-download with arena on-prem install)."
        raise click.ClickException(msg)
    runner = _shell_runner()
    cmd = [runner, str(script.resolve()), *args]
    click.echo(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


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
        click.echo(
            f"Note: helm install ignores {', '.join(ignored)} "
            "(runs ./setup.sh locally; kubectl must reach your cluster).",
            err=True,
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
    env = os.environ.copy()
    if ssh_user:
        env["SSH_USER"] = ssh_user
    if ssh_extra_opts:
        env["SSH_EXTRA_OPTS"] = ssh_extra_opts
    adv = (advertise_addr or manager).strip()
    env["SWARM_MANAGER_HOST"] = manager
    env["SWARM_ADVERTISE_ADDR"] = adv
    tokens_file = bundle_root / "swarm-tokens.txt"
    env["TOKENS_FILE"] = str(tokens_file)

    hosts = _all_hosts(manager, workers)
    worker_only = [h for h in workers if h.strip() and h.strip() != manager.strip()]

    click.echo("Running Docker Swarm install stages on cluster hosts…")
    _run_script(
        bundle_root / "install-docker.sh",
        hosts,
        env=env,
        cwd=bundle_root,
    )
    _run_script(
        bundle_root / "install-nvidia-driver.sh",
        hosts,
        env=env,
        cwd=bundle_root,
    )
    _run_script(
        bundle_root / "install-nvidia-container-toolkit.sh",
        hosts,
        env=env,
        cwd=bundle_root,
    )
    _run_script(
        bundle_root / "init-docker-swarm.sh",
        [manager, adv],
        env=env,
        cwd=bundle_root,
    )
    if worker_only:
        _run_script(
            bundle_root / "join-docker-swarm.sh",
            ["--tokens-file", str(tokens_file), *worker_only],
            env=env,
            cwd=bundle_root,
        )
    _run_script(
        bundle_root / "deploy-arena-stack.sh",
        [manager],
        env=env,
        cwd=bundle_root,
    )


def _parse_helm_release_ids(bundle_root: Path) -> tuple[str, str]:
    """Read ``clusterName`` from rendered chart values (same rules as ``setup.sh``)."""
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
    click.echo(f"Removing Helm release {release!r} (namespace {namespace})…")
    result = subprocess.run(
        ["helm", "uninstall", release, "--namespace", namespace],
        check=False,
    )
    if result.returncode != 0:
        click.echo(
            f"  helm uninstall exited {result.returncode} "
            "(release may already be removed).",
            err=True,
        )


def _ssh_remote_command(
    host: str,
    remote_cmd: str,
    *,
    ssh_user: str | None,
    ssh_extra_opts: str | None,
) -> None:
    user = ssh_user or os.environ.get("SSH_USER") or os.environ.get("USER")
    if not user:
        msg = "Cannot determine SSH user; set --ssh-user or SSH_USER."
        raise click.ClickException(msg)
    if not shutil.which("ssh"):
        msg = "ssh not found on PATH; required for dockerSwarm teardown."
        raise click.ClickException(msg)

    ssh_cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if ssh_extra_opts:
        ssh_cmd.extend(shlex.split(ssh_extra_opts))
    ssh_cmd.append(f"{user}@{host.strip()}")
    ssh_cmd.append(remote_cmd)
    click.echo(f"  $ {' '.join(ssh_cmd)}")
    result = subprocess.run(ssh_cmd, check=False)
    if result.returncode != 0:
        click.echo(f"  ssh exited {result.returncode}", err=True)


def _run_docker_swarm_teardown(
    *,
    manager: str,
    stack_name: str,
    ssh_user: str | None,
    ssh_extra_opts: str | None,
) -> None:
    click.echo(f"Removing Docker stack {stack_name!r} on {manager}…")
    _ssh_remote_command(
        manager,
        f"sudo docker stack rm {stack_name}",
        ssh_user=ssh_user,
        ssh_extra_opts=ssh_extra_opts,
    )


def _delete_class_if_present(client: Any, name: str) -> None:
    listed = client.invoke_manifest_command(_LIST_CLASSES_INVOKE, {})
    if _class_by_name(listed, name) is None:
        click.echo(f"No Arena resource class {name!r}; skipping API delete.")
        return
    click.echo(f"Deleting on-prem resource class {name!r} from Arena…")
    client.invoke_manifest_command(_DELETE_CLASS_INVOKE, {"name": name})


def run_on_prem_teardown(
    client: Any,
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
                stack_name=stack_name,
                ssh_user=ssh_user,
                ssh_extra_opts=ssh_extra_opts,
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
        click.echo("Disabling on-prem provider…")
        client.invoke_manifest_command(_DISABLE_INVOKE, {})

    click.echo(f"On-prem teardown finished for class {name!r} ({kind}).")


def _run_helm_install(bundle_root: Path) -> None:
    setup = bundle_root / "setup.sh"
    if not setup.is_file():
        msg = (
            "Helm bundle has no setup.sh. Re-run with --setup-type dockerSwarm "
            "for SSH-based install, or check the platform bundle."
        )
        raise click.ClickException(msg)
    if not shutil.which("helm"):
        msg = (
            "helm not found on PATH; install Helm 3.x or use --setup-type dockerSwarm."
        )
        raise click.ClickException(msg)
    click.echo("Running Helm setup (local kubectl context)…")
    _run_script(setup, [], env=os.environ.copy(), cwd=bundle_root)


def run_on_prem_install(
    client: Any,
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
) -> None:
    """Enable on-prem, ensure class exists, download bundle, run install scripts."""
    kind = normalize_setup_type(setup_type)

    if kind == "dockerSwarm":
        if not manager or not manager.strip():
            msg = """--manager is required for dockerSwarm\n
            (SSH hostname or IP of the swarm manager)."""
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
        click.echo("Enabling on-prem provider…")
        client.invoke_manifest_command(_ENABLE_INVOKE, {})

    listed = client.invoke_manifest_command(_LIST_CLASSES_INVOKE, {})
    existing = _class_by_name(listed, name)
    create_nodes = _num_nodes_for_create(
        existing,
        kind=kind,
        explicit=num_nodes,
        manager=manager.strip() if manager else None,
        workers=workers,
    )
    _ensure_class(client, name=name, num_nodes=create_nodes)

    with tempfile.TemporaryDirectory(prefix="arena-on-prem-") as tmp:
        bundle_root = _download_bundle(
            client,
            class_name=name,
            setup_type=kind,
            dest_dir=Path(tmp),
        )
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
        else:
            _run_helm_install(bundle_root)

    click.echo(f"On-prem install finished for class {name!r} ({kind}).")


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
        help="[dockerSwarm] SSH login for remote hosts.",
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
        help="Skip POST /on-prem/enable when the provider is already enabled.",
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
    ) -> None:
        """Install an on-prem worker cluster for CLASS_NAME.

        **dockerSwarm** — downloads the platform bundle and runs install-docker,
        NVIDIA, swarm, and stack scripts on ``--manager`` and ``--workers`` via SSH
        (see agilerl-platform ``resources/docker-swarm-setup/arena-train/SETUP.md``).

        **helm** — downloads the Helm chart bundle and runs ``./setup.sh`` on this
        machine; requires ``helm`` and a configured ``kubectl`` context only
        (see ``resources/helm-setup/arena-train/SETUP.md``).
        """
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
        help="Also POST /on-prem/disable after teardown.",
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
    ) -> None:
        """Tear down an on-prem install for CLASS_NAME.

        **helm** — ``helm uninstall`` using ``clusterName`` from the class deployment bundle
        (same release/namespace as ``setup.sh``).

        **dockerSwarm** — ``docker stack rm`` on ``--manager`` (default stack name ``arena``).

        By default also deletes the Arena on-prem resource class; use ``--keep-class`` to
        leave the class registered. Use ``--skip-cluster`` for API-only cleanup.
        """
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
            )

    return teardown_cmd


def register_on_prem_install(on_prem_group: click.Group) -> None:
    """Replace manifest ``install`` subgroup; register install + teardown commands."""
    on_prem_group.commands.pop("install", None)
    on_prem_group.add_command(build_install_command())
    on_prem_group.add_command(build_teardown_command())
