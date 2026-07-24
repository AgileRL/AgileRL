"""Tests for on-prem cluster Helm install helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from agilerl.arena.on_prem.cluster_helm import (
    AGENT_CHART,
    STORAGE_CHART,
    helm_upgrade_install,
    install_lab_cluster_charts,
    resolve_helm_charts_root,
)


def test_resolve_helm_charts_root_explicit(tmp_path: Path) -> None:
    charts = tmp_path / "helm-setup"
    charts.mkdir()
    assert resolve_helm_charts_root(charts) == charts.resolve()


def test_resolve_helm_charts_root_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(click.ClickException, match="does not exist"):
        resolve_helm_charts_root(tmp_path / "missing")


def test_resolve_helm_charts_root_requires_env_or_flag() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(click.ClickException, match="ARENA_HELM_CHARTS_DIR"):
            resolve_helm_charts_root(None)


def test_helm_upgrade_install_invokes_helm(tmp_path: Path) -> None:
    chart = tmp_path / "chart"
    chart.mkdir()
    values = tmp_path / "values.yaml"
    values.write_text("bucket:\n  name: arena-data\n", encoding="utf-8")

    with patch("agilerl.arena.on_prem.cluster_helm.shutil.which", return_value="/usr/bin/helm"), patch(
        "agilerl.arena.on_prem.cluster_helm.subprocess.run"
    ) as run_mock:
        run_mock.return_value = MagicMock(returncode=0)
        helm_upgrade_install(
            release="arena-on-prem-storage",
            chart=chart,
            namespace="storage",
            values_file=values,
        )

    cmd = run_mock.call_args.args[0]
    assert cmd[:4] == ["helm", "upgrade", "--install", "arena-on-prem-storage"]
    assert "--wait" in cmd


def test_install_lab_cluster_charts_orders_storage_then_agent(tmp_path: Path) -> None:
    charts_root = tmp_path / "helm-setup"
    for name in (STORAGE_CHART, AGENT_CHART):
        (charts_root / name / "chart").mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    (out / "storage-helm-values.yaml").write_text("bucket:\n  name: arena-data\n")
    (out / "agent-helm-values.yaml").write_text("storage:\n  endpoint: http://minio\n")

    calls: list[str] = []

    def _record(**kwargs: object) -> None:
        calls.append(str(kwargs["release"]))

    with patch(
        "agilerl.arena.on_prem.cluster_helm.helm_upgrade_install",
        side_effect=_record,
    ):
        install_lab_cluster_charts(out, charts_dir=charts_root)

    assert calls == ["arena-on-prem-storage", "arena-on-prem-agent"]
