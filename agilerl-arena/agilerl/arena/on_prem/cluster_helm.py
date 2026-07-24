"""Helm install helpers for enterprise on-prem cluster registration."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import click

logger = logging.getLogger("agilerl.arena.on_prem")

STORAGE_RELEASE = "arena-on-prem-storage"
STORAGE_NAMESPACE = "storage"
AGENT_RELEASE = "arena-on-prem-agent"
AGENT_NAMESPACE = "arena-on-prem"
STORAGE_CHART = "arena-on-prem-storage"
AGENT_CHART = "arena-on-prem-agent"
DEFAULT_HELM_WAIT_TIMEOUT = "10m"


def resolve_helm_charts_root(charts_dir: Path | None) -> Path:
    """Return the platform ``resources/helm-setup`` directory.

    :param charts_dir: Explicit charts root, or ``None`` to use env/default.
    :type charts_dir: Path | None
    :returns: Absolute path to ``resources/helm-setup``.
    :rtype: Path
    :raises click.ClickException: If the directory cannot be resolved.
    """
    if charts_dir is not None:
        root = charts_dir.expanduser().resolve()
        if not root.is_dir():
            msg = f"Charts directory does not exist: {root}"
            raise click.ClickException(msg)
        return root

    env = os.environ.get("ARENA_HELM_CHARTS_DIR", "").strip()
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            msg = f"ARENA_HELM_CHARTS_DIR is not a directory: {root}"
            raise click.ClickException(msg)
        return root

    msg = (
        "Helm charts directory is required for --install. Pass --charts-dir "
        "pointing at agilerl-platform/resources/helm-setup or set "
        "ARENA_HELM_CHARTS_DIR."
    )
    raise click.ClickException(msg)


def _chart_path(charts_root: Path, chart_name: str) -> Path:
    chart = charts_root / chart_name / "chart"
    if not chart.is_dir():
        msg = f"Helm chart not found at {chart}"
        raise click.ClickException(msg)
    return chart


def _require_helm() -> None:
    if not shutil.which("helm"):
        msg = "helm not found on PATH; install Helm 3.x or omit --install."
        raise click.ClickException(msg)


def helm_upgrade_install(
    *,
    release: str,
    chart: Path,
    namespace: str,
    values_file: Path,
    wait: bool = True,
    timeout: str = DEFAULT_HELM_WAIT_TIMEOUT,
) -> None:
    """Run ``helm upgrade --install`` for *release*.

    :param release: Helm release name.
    :type release: str
    :param chart: Path to the chart directory.
    :type chart: Path
    :param namespace: Target namespace.
    :type namespace: str
    :param values_file: Values file passed with ``-f``.
    :type values_file: Path
    :param wait: Pass ``--wait`` when ``True``.
    :type wait: bool
    :param timeout: Helm ``--timeout`` value.
    :type timeout: str
    :returns: None
    :rtype: None
    :raises click.ClickException: If helm exits non-zero.
    """
    _require_helm()
    if not values_file.is_file():
        msg = f"Helm values file not found: {values_file}"
        raise click.ClickException(msg)

    cmd = [
        "helm",
        "upgrade",
        "--install",
        release,
        str(chart),
        "--namespace",
        namespace,
        "--create-namespace",
        "-f",
        str(values_file),
    ]
    if wait:
        cmd.extend(["--wait", "--timeout", timeout])

    logger.info("Running %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        msg = f"helm upgrade --install {release} failed (exit {result.returncode})"
        raise click.ClickException(msg)


def install_lab_cluster_charts(
    output_dir: Path,
    *,
    charts_dir: Path | None = None,
    wait: bool = True,
) -> None:
    """Install lab MinIO storage, then the on-prem agent, from registration output.

    :param output_dir: Directory containing ``storage-helm-values.yaml`` and
        ``agent-helm-values.yaml``.
    :type output_dir: Path
    :param charts_dir: Optional explicit ``resources/helm-setup`` root.
    :type charts_dir: Path | None
    :param wait: Wait for Helm resources (including bucket bootstrap hook).
    :type wait: bool
    :returns: None
    :rtype: None
    """
    charts_root = resolve_helm_charts_root(charts_dir)
    storage_values = output_dir / "storage-helm-values.yaml"
    agent_values = output_dir / "agent-helm-values.yaml"
    if not storage_values.is_file():
        msg = f"Missing {storage_values}; lab profile requires storage Helm values."
        raise click.ClickException(msg)
    if not agent_values.is_file():
        msg = f"Missing {agent_values}; registration did not write agent values."
        raise click.ClickException(msg)

    helm_upgrade_install(
        release=STORAGE_RELEASE,
        chart=_chart_path(charts_root, STORAGE_CHART),
        namespace=STORAGE_NAMESPACE,
        values_file=storage_values,
        wait=wait,
    )
    helm_upgrade_install(
        release=AGENT_RELEASE,
        chart=_chart_path(charts_root, AGENT_CHART),
        namespace=AGENT_NAMESPACE,
        values_file=agent_values,
        wait=wait,
    )


def install_enterprise_agent_chart(
    output_dir: Path,
    *,
    charts_dir: Path | None = None,
    wait: bool = True,
) -> None:
    """Install the on-prem agent chart for an enterprise-registered cluster.

    :param output_dir: Directory containing ``agent-helm-values.yaml``.
    :type output_dir: Path
    :param charts_dir: Optional explicit ``resources/helm-setup`` root.
    :type charts_dir: Path | None
    :param wait: Wait for Helm resources to become ready.
    :type wait: bool
    :returns: None
    :rtype: None
    """
    charts_root = resolve_helm_charts_root(charts_dir)
    agent_values = output_dir / "agent-helm-values.yaml"
    if not agent_values.is_file():
        msg = f"Missing {agent_values}; registration did not write agent values."
        raise click.ClickException(msg)

    helm_upgrade_install(
        release=AGENT_RELEASE,
        chart=_chart_path(charts_root, AGENT_CHART),
        namespace=AGENT_NAMESPACE,
        values_file=agent_values,
        wait=wait,
    )
