# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Pick the Kubernetes cluster a BYOC command acts on and check it is the registered one."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import click
import yaml

from agilerl.byoc.api import ByocApi
from agilerl.byoc.provisioning.kubectl import kubeconfig_env, require_kubectl

ClickDecorated = TypeVar("ClickDecorated", bound=Callable[..., Any] | click.Command)


class NoKubeContextError(click.ClickException):
    """No kubeconfig names a context for the command to use."""


@dataclass(frozen=True)
class KubeTarget:
    """One Kubernetes context, pinned in a kubeconfig of its own."""

    kubeconfig: Path
    context: str
    server: str
    cluster_uid: str

    def describe(self) -> str:
        """Return the context, API server and cluster UID as one phrase."""
        return (
            f"context {self.context!r} (API server {self.server}, "
            f"cluster UID {self.cluster_uid})"
        )


def provisioned_kubeconfig(name: str) -> Path:
    """Return where ``arena cluster provision`` writes the kubeconfig for *name*."""
    return Path(f"arena-cluster-{name}") / "kubeconfig"


def _kubectl(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kubectl", *args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _source_kubeconfig(name: str | None, kubeconfig: Path | None) -> Path | None:
    """Return the kubeconfig the user named, or the one provisioned for *name*."""
    if kubeconfig is not None:
        return kubeconfig.expanduser().resolve()
    if name is None:
        return None
    provisioned = provisioned_kubeconfig(name)
    if provisioned.is_file():
        return provisioned.resolve()
    return None


def _pinned_kubeconfig(source: Path | None, context: str | None) -> dict[str, Any]:
    """Return a self-contained kubeconfig that holds only the chosen context."""
    env = kubeconfig_env(source) if source is not None else os.environ.copy()
    args = ["config", "view", "--minify", "--flatten", "--output", "yaml"]
    if context is not None:
        args += ["--context", context]
    result = _kubectl(args, env)
    if result.returncode != 0:
        where = str(source) if source is not None else "KUBECONFIG or ~/.kube/config"
        msg = (
            f"No Kubernetes context to use from {where}: {result.stderr.strip()} "
            "Pass --kubeconfig and --context."
        )
        raise NoKubeContextError(msg)
    return yaml.safe_load(result.stdout)


def _confirm_implicit_context(context: str, server: str) -> None:
    """Ask before using the current context when no flag chose the cluster."""
    if not sys.stdin.isatty():
        msg = (
            f"No --context or --kubeconfig given. Pass --context {context} to use the "
            f"current context (API server {server}), or name another one."
        )
        raise click.ClickException(msg)
    click.confirm(
        f"Use the current Kubernetes context {context!r} (API server {server})?",
        default=False,
        abort=True,
    )


def _read_cluster_uid(kubeconfig: Path) -> str:
    result = _kubectl(
        ["get", "namespace", "kube-system", "--output", "jsonpath={.metadata.uid}"],
        kubeconfig_env(kubeconfig),
    )
    uid = result.stdout.strip()
    if result.returncode != 0 or not uid:
        msg = f"Could not read the kube-system namespace UID: {result.stderr.strip()}"
        raise click.ClickException(msg)
    return uid


@contextmanager
def kube_target(
    name: str | None, kubeconfig: Path | None, context: str | None
) -> Iterator[KubeTarget]:
    """Resolve the Kubernetes context for cluster *name* and read its identity.

    The context comes from ``--context`` or, failing that, the current context
    of ``--kubeconfig``, of ``arena-cluster-NAME/kubeconfig``, or of kubectl's
    own default. That last one is used only after the user confirms it.

    :param name: Arena cluster name, or ``None`` for a cluster not named yet.
    :param kubeconfig: ``--kubeconfig``, or ``None``.
    :param context: ``--context``, or ``None``.
    :return: The target, whose kubeconfig exists until the block exits.
    """
    require_kubectl()
    source = _source_kubeconfig(name, kubeconfig)
    config = _pinned_kubeconfig(source, context)
    resolved_context = str(config["current-context"])
    server = str(config["clusters"][0]["cluster"]["server"])
    if source is None and context is None:
        _confirm_implicit_context(resolved_context, server)
    with tempfile.TemporaryDirectory(prefix="arena-kube-") as tmp:
        pinned = Path(tmp) / "kubeconfig"
        descriptor = os.open(pinned, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        target = KubeTarget(
            kubeconfig=pinned,
            context=resolved_context,
            server=server,
            cluster_uid=_read_cluster_uid(pinned),
        )
        click.echo(f"Using Kubernetes {target.describe()}.", err=True)
        yield target


@contextmanager
def kubeconfig_environment(kubeconfig_path: Path) -> Iterator[None]:
    """Set ``KUBECONFIG`` only while Helm and kubectl run."""
    old_value = os.environ.get("KUBECONFIG")
    os.environ["KUBECONFIG"] = str(kubeconfig_path)
    try:
        yield
    finally:
        if old_value is None:
            del os.environ["KUBECONFIG"]
        else:
            os.environ["KUBECONFIG"] = old_value


def require_registered_cluster(api: ByocApi, name: str, target: KubeTarget) -> None:
    """Refuse *target* when Arena stores another cluster UID for *name*."""
    cluster = api.find_cluster(name)
    if cluster is None:
        msg = f"No registered BYOC cluster named {name!r}."
        raise click.ClickException(msg)
    if cluster.get("clusterUid") is None:
        click.echo(
            f"Arena has no cluster UID for {name!r} yet. Its agent records one on "
            "its next heartbeat."
        )
    require_same_cluster(cluster, name, target)


def require_same_cluster(
    cluster: dict[str, Any], name: str, target: KubeTarget
) -> None:
    """Refuse *target* when *cluster*, Arena's row for *name*, stores another UID."""
    stored = cluster.get("clusterUid")
    if stored is not None and stored != target.cluster_uid:
        msg = (
            f"Kubernetes {target.describe()} is not Arena cluster {name!r}, which runs "
            f"in cluster UID {stored}. Pass --context for that cluster."
        )
        raise click.ClickException(msg)


def kube_target_options(command: ClickDecorated) -> ClickDecorated:
    """Add ``--kubeconfig`` and ``--context`` to a command that talks to Kubernetes."""
    command = click.option(
        "--context",
        "kube_context",
        default=None,
        help=(
            "Kubernetes context to use. Without it or --kubeconfig, the command "
            "shows the current context and asks first."
        ),
    )(command)
    return click.option(
        "--kubeconfig",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Kubeconfig for the agent cluster. Defaults to "
            "arena-cluster-NAME/kubeconfig when that file exists."
        ),
    )(command)
