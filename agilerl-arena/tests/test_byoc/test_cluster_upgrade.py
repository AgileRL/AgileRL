# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster upgrade``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.byoc import ClusterDynamicGroup, build_cluster_upgrade_command
from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_helm import STORAGE_RELEASE, agent_deployment_selector
from agilerl.arena.byoc.cluster_upgrade import run_cluster_upgrade
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAPIError

AGENT_YAML = "image:\n  tag: deadbeef\n"
STORAGE_VALUES = (
    "image:\n  tag: deadbeef\n"
    "storage:\n"
    "  secretName: corp-s3\n"
    "  endpoint: https://s3.example.com\n"
    "  createSecret: false\n"
)


def _install_package(tmp_path: Path, values: str = AGENT_YAML) -> tuple[Path, Path]:
    package_root = tmp_path / "arena-byoc-install"
    chart = package_root / "arena-byoc-agent" / "chart"
    chart.mkdir(parents=True)
    (package_root / "arena-byoc-storage" / "chart").mkdir(parents=True)
    values_file = package_root / "agent-helm-values.yaml"
    values_file.write_text(values, encoding="utf-8")
    (package_root / "storage-helm-values.yaml").write_text(
        "bucket:\n  name: arena-data\n",
        encoding="utf-8",
    )
    secrets = package_root / "secrets"
    secrets.mkdir()
    (secrets / "cluster-token").write_text("do-not-apply", encoding="utf-8")
    return package_root, values_file


def _helm_list(releases: list[dict[str, str]]) -> MagicMock:
    return MagicMock(returncode=0, stdout=json.dumps(releases), stderr="")


def test_upgrade_helm_installs_the_discovered_agent_and_skips_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, values_file = _install_package(tmp_path)
    chart = package_root / "arena-byoc-agent" / "chart"
    kubeconfig = tmp_path / "kubeconfig"
    listed = _helm_list(
        [
            {"name": "arena-byoc-agent", "namespace": "arena"},
            {"name": "corp-k8s", "namespace": "arena-agent"},
            {"name": STORAGE_RELEASE, "namespace": "storage"},
        ]
    )
    calls: list[str] = []
    monkeypatch.delenv("KUBECONFIG", raising=False)

    def sync_endpoint(values: dict[str, object], agent_namespace: str) -> bool:
        calls.append("sync")
        assert values == {"image": {"tag": "deadbeef"}}
        assert agent_namespace == "arena-agent"
        return False

    def helm(**kwargs: object) -> None:
        calls.append("helm")
        assert kwargs["release"] == "corp-k8s"
        assert kwargs["namespace"] == "arena-agent"

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ) as fetch,
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ) as download,
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.sync_storage_secret_endpoint",
            side_effect=sync_endpoint,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.helm_upgrade_install",
            side_effect=helm,
        ) as helm_mock,
        patch(
            "agilerl.arena.byoc.cluster_upgrade._restart_agent_deployment"
        ) as restart,
        patch(
            "agilerl.arena.byoc.cluster_helm._apply_install_package_secrets"
        ) as apply_secrets,
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s", helm_wait=True)

    fetch.assert_called_once_with("corp-k8s")
    download.assert_called_once_with(
        cluster="corp-k8s",
        agent_helm_values_yaml=AGENT_YAML,
    )
    assert calls == ["sync", "helm"]
    helm_mock.assert_called_once_with(
        release="corp-k8s",
        chart=chart,
        namespace="arena-agent",
        values_file=values_file,
        wait=True,
        wait_for_selector=agent_deployment_selector("corp-k8s"),
    )
    assert "arena-byoc-storage" not in str(helm_mock.call_args.kwargs["chart"])
    restart.assert_not_called()
    apply_secrets.assert_not_called()
    assert "KUBECONFIG" not in os.environ


def test_upgrade_copies_storage_endpoint_before_helm_and_restarts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, _values_file = _install_package(tmp_path, STORAGE_VALUES)
    kubeconfig = tmp_path / "kubeconfig"
    listed = _helm_list([{"name": "corp-k8s", "namespace": "arena-agent"}])
    calls: list[str] = []
    monkeypatch.setenv("KUBECONFIG", "/tmp/original-kubeconfig")

    def sync_endpoint(values: dict[str, object], agent_namespace: str) -> bool:
        calls.append("sync")
        storage = values["storage"]
        assert isinstance(storage, dict)
        assert storage["endpoint"] == "https://s3.example.com"
        assert storage["createSecret"] is False
        assert agent_namespace == "arena-agent"
        return True

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": STORAGE_VALUES},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.sync_storage_secret_endpoint",
            side_effect=sync_endpoint,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.helm_upgrade_install",
            side_effect=lambda **_kwargs: calls.append("helm"),
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade._restart_agent_deployment",
            side_effect=lambda *_args, **_kwargs: calls.append("restart"),
        ) as restart,
    ):
        run_cluster_upgrade(
            MagicMock(),
            name="corp-k8s",
            agent_namespace=" arena-agent ",
            helm_wait=False,
        )

    assert calls == ["sync", "helm", "restart"]
    restart.assert_called_once_with("arena-agent", "corp-k8s", wait=False)
    assert os.environ["KUBECONFIG"] == "/tmp/original-kubeconfig"


def test_upgrade_stops_when_no_agent_release_is_installed(tmp_path: Path) -> None:
    package_root, _values_file = _install_package(tmp_path)
    kubeconfig = tmp_path / "kubeconfig"

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.cluster_helm.subprocess.run",
            return_value=_helm_list([]),
        ),
        patch("agilerl.arena.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="was not found"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    helm.assert_not_called()


def test_upgrade_stops_when_the_named_namespace_has_no_release(
    tmp_path: Path,
) -> None:
    package_root, _values_file = _install_package(tmp_path)
    kubeconfig = tmp_path / "kubeconfig"
    listed = _helm_list([{"name": "corp-k8s", "namespace": "arena-agent"}])

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch("agilerl.arena.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="not installed in namespace 'arena'"),
    ):
        run_cluster_upgrade(
            MagicMock(),
            name="corp-k8s",
            agent_namespace="arena",
        )

    helm.assert_not_called()


def test_upgrade_stops_when_kubeconfig_is_missing(tmp_path: Path) -> None:
    package_root, _values_file = _install_package(tmp_path)

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=None,
        ),
        patch("agilerl.arena.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="No kubeconfig found"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    helm.assert_not_called()


def test_upgrade_stops_when_the_agent_is_in_multiple_namespaces(
    tmp_path: Path,
) -> None:
    package_root, _values_file = _install_package(tmp_path)
    kubeconfig = tmp_path / "kubeconfig"
    listed = _helm_list(
        [
            {"name": "corp-k8s", "namespace": "arena"},
            {"name": "corp-k8s", "namespace": "arena-agent"},
        ]
    )

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch("agilerl.arena.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="multiple namespaces"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    helm.assert_not_called()


def test_upgrade_requires_a_cluster_name() -> None:
    with pytest.raises(click.ClickException, match="--name is required"):
        run_cluster_upgrade(MagicMock(), name="  ")


@pytest.mark.parametrize(
    "artifacts",
    [{}, {"agentHelmValuesYaml": "  "}, {"agentHelmValuesYaml": None}],
)
def test_upgrade_requires_agent_helm_values(artifacts: dict[str, object]) -> None:
    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value=artifacts,
        ),
        pytest.raises(click.ClickException, match="did not return agent Helm values"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")


def test_upgrade_requires_the_values_file(tmp_path: Path) -> None:
    package_root = tmp_path / "pkg"
    (package_root / "arena-byoc-agent" / "chart").mkdir(parents=True)
    kubeconfig = tmp_path / "kubeconfig"
    listed = _helm_list([{"name": "corp-k8s", "namespace": "arena-agent"}])

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
        pytest.raises(click.ClickException, match="install package is incomplete"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")


def test_upgrade_requires_the_chart_directory(tmp_path: Path) -> None:
    package_root, _values_file = _install_package(tmp_path)
    chart = package_root / "arena-byoc-agent" / "chart"
    chart.rmdir()
    kubeconfig = tmp_path / "kubeconfig"
    listed = _helm_list([{"name": "corp-k8s", "namespace": "arena-agent"}])

    with (
        patch.object(
            ByocApi,
            "fetch_cluster_install_artifacts",
            return_value={"agentHelmValuesYaml": AGENT_YAML},
        ),
        patch.object(
            ByocApi,
            "download_cluster_install_package",
            return_value=b"pkg",
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch(
            "agilerl.arena.byoc.cluster_upgrade.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.arena.byoc.cluster_helm.subprocess.run", return_value=listed),
        pytest.raises(click.ClickException, match="Helm chart not found"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")


def test_upgrade_command_passes_install_flags(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")

    with patch("agilerl.arena.byoc.commands.run_cluster_upgrade") as upgrade:
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            [
                "--name",
                "corp-k8s",
                "--kubeconfig",
                str(kubeconfig),
                "--agent-namespace",
                "arena-agent",
                "--no-helm-wait",
            ],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    upgrade.assert_called_once()
    assert upgrade.call_args.kwargs["name"] == "corp-k8s"
    assert upgrade.call_args.kwargs["kubeconfig"] == kubeconfig
    assert upgrade.call_args.kwargs["agent_namespace"] == "arena-agent"
    assert upgrade.call_args.kwargs["helm_wait"] is False


def test_upgrade_command_discovers_namespace_when_omitted(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.arena.byoc.commands.run_cluster_upgrade") as upgrade:
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            ["--name", "corp-k8s"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert upgrade.call_args.kwargs["agent_namespace"] is None


def test_upgrade_command_rejects_a_blank_name(
    command_config: CommandConfig,
) -> None:
    result = CliRunner().invoke(
        build_cluster_upgrade_command(),
        ["--name", "   "],
        obj=command_config,
    )

    assert result.exit_code != 0
    assert "--name is required" in result.output


def test_cluster_group_includes_upgrade() -> None:
    assert "upgrade" in ClusterDynamicGroup().commands


def test_fetch_cluster_install_artifacts_posts_the_cluster_name(
    byoc_api: ByocApi, mock_client: MagicMock
) -> None:
    mock_client._invoke_manifest_command.return_value = {
        "agentHelmValuesYaml": AGENT_YAML,
    }

    result = byoc_api.fetch_cluster_install_artifacts("corp-k8s")

    invoke, body = mock_client._invoke_manifest_command.call_args.args
    assert result == {"agentHelmValuesYaml": AGENT_YAML}
    assert invoke["method"] == "POST"
    assert invoke["path"] == "/api/cli/v1/byoc/clusters/install-artifacts"
    assert body == {"name": "corp-k8s"}


def test_fetch_cluster_install_artifacts_rejects_non_dict(
    byoc_api: ByocApi, mock_client: MagicMock
) -> None:
    mock_client._invoke_manifest_command.return_value = "yaml"

    with pytest.raises(ArenaAPIError, match="Unexpected cluster install artifacts"):
        byoc_api.fetch_cluster_install_artifacts("corp-k8s")
