# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster upgrade``."""

from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import httpx
import pytest
from click.testing import CliRunner

from agilerl.byoc import ClusterDynamicGroup, build_cluster_upgrade_command
from agilerl.byoc.api import ByocApi
from agilerl.byoc.cluster_helm import STORAGE_RELEASE, agent_deployment_selector
from agilerl.byoc.cluster_upgrade import run_cluster_upgrade
from agilerl.byoc.kube_target import require_registered_cluster
from agilerl.client import ArenaClient
from agilerl.config import CommandConfig
from agilerl.exceptions import ArenaAPIError

UPGRADE_MODULE = "agilerl.byoc.cluster_upgrade"
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


@pytest.fixture(autouse=True)
def pinned_target(
    fake_kube_target: Callable[..., MagicMock], monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Point every upgrade at a fixed context that Arena accepts."""
    monkeypatch.setattr(f"{UPGRADE_MODULE}.require_registered_cluster", MagicMock())
    return fake_kube_target(UPGRADE_MODULE)


def _helm_list(releases: list[dict[str, str]]) -> MagicMock:
    return MagicMock(returncode=0, stdout=json.dumps(releases), stderr="")


def test_upgrade_helm_installs_the_discovered_agent_and_skips_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, values_file = _install_package(tmp_path)
    chart = package_root / "arena-byoc-agent" / "chart"
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
        assert os.environ["KUBECONFIG"] == str(tmp_path / "pinned-kubeconfig")
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch(
            "agilerl.byoc.cluster_upgrade.sync_storage_secret_endpoint",
            side_effect=sync_endpoint,
        ),
        patch(
            "agilerl.byoc.cluster_upgrade.helm_upgrade_install",
            side_effect=helm,
        ) as helm_mock,
        patch("agilerl.byoc.cluster_upgrade._restart_agent_deployment") as restart,
        patch(
            "agilerl.byoc.cluster_helm._apply_install_package_secrets"
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch(
            "agilerl.byoc.cluster_upgrade.sync_storage_secret_endpoint",
            side_effect=sync_endpoint,
        ),
        patch(
            "agilerl.byoc.cluster_upgrade.helm_upgrade_install",
            side_effect=lambda **_kwargs: calls.append("helm"),
        ),
        patch(
            "agilerl.byoc.cluster_upgrade._restart_agent_deployment",
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


def test_upgrade_platform_cluster_fetches_its_artifacts_and_runs_helm(
    tmp_path: Path,
    http_client: ArenaClient,
    sent_scope: Callable[[ArenaClient, str], dict[str, object]],
) -> None:
    # Arrange
    package_root, values_file = _install_package(tmp_path)
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")
    listed = _helm_list([{"name": "arena", "namespace": "arena-agent"}])

    # Act
    with (
        patch(
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ) as extract,
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch(
            "agilerl.byoc.cluster_upgrade.sync_storage_secret_endpoint",
            return_value=False,
        ),
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
    ):
        run_cluster_upgrade(
            http_client, name="arena", kubeconfig=kubeconfig, platform=True
        )

    # Assert
    assert sent_scope(http_client, "/clusters/install-artifacts")["body"] == "platform"
    assert sent_scope(http_client, "/clusters/install-package")["body"] == "platform"
    assert extract.call_args.args[0] == b"pkg"
    assert helm.call_args.kwargs["release"] == "arena"
    assert helm.call_args.kwargs["namespace"] == "arena-agent"
    assert helm.call_args.kwargs["values_file"] == values_file


def test_upgrade_stops_when_no_agent_release_is_installed(tmp_path: Path) -> None:
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch(
            "agilerl.byoc.cluster_helm.subprocess.run",
            return_value=_helm_list([]),
        ),
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="was not found"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    helm.assert_not_called()


def test_upgrade_stops_when_the_named_namespace_has_no_release(
    tmp_path: Path,
) -> None:
    package_root, _values_file = _install_package(tmp_path)
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="not installed in namespace 'arena'"),
    ):
        run_cluster_upgrade(
            MagicMock(),
            name="corp-k8s",
            agent_namespace="arena",
        )

    helm.assert_not_called()


def test_upgrade_stops_before_helm_when_the_context_is_another_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        f"{UPGRADE_MODULE}.require_registered_cluster", require_registered_cluster
    )

    with (
        patch.object(
            ByocApi,
            "find_cluster",
            return_value={"name": "corp-k8s", "clusterUid": "another-uid"},
        ),
        patch.object(ByocApi, "fetch_cluster_install_artifacts") as fetch,
        patch(f"{UPGRADE_MODULE}.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="is not Arena cluster 'corp-k8s'"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    fetch.assert_not_called()
    helm.assert_not_called()


def test_upgrade_stops_when_the_agent_is_in_multiple_namespaces(
    tmp_path: Path,
) -> None:
    package_root, _values_file = _install_package(tmp_path)
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        pytest.raises(click.ClickException, match="install package is incomplete"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")


def test_upgrade_requires_the_chart_directory(tmp_path: Path) -> None:
    package_root, _values_file = _install_package(tmp_path)
    chart = package_root / "arena-byoc-agent" / "chart"
    chart.rmdir()
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
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        pytest.raises(click.ClickException, match="Helm chart not found"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")


def test_upgrade_command_passes_install_flags(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nkind: Config\n", encoding="utf-8")

    with patch("agilerl.byoc.commands.run_cluster_upgrade") as upgrade:
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            [
                "--name",
                "corp-k8s",
                "--kubeconfig",
                str(kubeconfig),
                "--context",
                "corp",
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
    assert upgrade.call_args.kwargs["context"] == "corp"
    assert upgrade.call_args.kwargs["agent_namespace"] == "arena-agent"
    assert upgrade.call_args.kwargs["helm_wait"] is False
    assert upgrade.call_args.kwargs["yes"] is False


def test_upgrade_command_targets_platform_clusters_with_the_flag(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.byoc.commands.run_cluster_upgrade") as upgrade:
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            ["--platform", "--name", "arena"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert upgrade.call_args.kwargs["platform"] is True


def test_upgrade_command_targets_org_clusters_by_default(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.byoc.commands.run_cluster_upgrade") as upgrade:
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            ["--name", "corp-k8s"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert upgrade.call_args.kwargs["platform"] is False


def test_upgrade_command_explains_a_platform_refusal(
    command_config: CommandConfig,
    http_client: ArenaClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    http_client._http.request.side_effect = None
    http_client._http.request.return_value = httpx.Response(
        403, json={"error": "forbidden"}
    )

    with patch("agilerl.config.build_client", return_value=http_client):
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            ["--platform", "--name", "arena"],
            obj=command_config,
        )

    assert result.exit_code == 1
    printed = result.output + capsys.readouterr().err
    assert "Only AgileRL staff can manage platform clusters." in printed


def test_upgrade_command_discovers_namespace_when_omitted(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.byoc.commands.run_cluster_upgrade") as upgrade:
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


RBAC_DIFF = [
    {
        "cluster_role": "operator",
        "added": [
            {"api_group": "", "resource": "nodes", "verbs": ["get", "list"]},
            {
                "api_group": "storage.k8s.io",
                "resource": "storageclasses",
                "verbs": ["get"],
            },
        ],
        "removed": [
            {
                "api_group": "rbac.authorization.k8s.io",
                "resource": "clusterroles",
                "resource_name": "namespace-secrets",
                "verbs": ["bind"],
            }
        ],
    }
]
APPROVAL_ARTIFACTS = {
    "agentHelmValuesYaml": AGENT_YAML,
    "agentUpgradeRequiresApproval": True,
    "rbacDiff": RBAC_DIFF,
}


class TtyInput(io.StringIO):
    """Stdin that reports a terminal."""

    def isatty(self) -> bool:
        return True


def _upgrade_with(
    tmp_path: Path,
    artifacts: dict[str, object],
    yes: bool = False,
    context: str | None = None,
) -> MagicMock:
    """Run the upgrade against *artifacts* and return the Helm mock."""
    package_root, _values_file = _install_package(tmp_path)
    listed = _helm_list([{"name": "corp-k8s", "namespace": "arena"}])
    with (
        patch.object(
            ByocApi, "fetch_cluster_install_artifacts", return_value=artifacts
        ),
        patch.object(ByocApi, "download_cluster_install_package", return_value=b"pkg"),
        patch(
            "agilerl.byoc.cluster_upgrade.extract_cluster_install_package",
            return_value=package_root,
        ),
        patch("agilerl.byoc.cluster_helm.helm_available", return_value=True),
        patch("agilerl.byoc.cluster_helm.subprocess.run", return_value=listed),
        patch(
            "agilerl.byoc.cluster_upgrade.sync_storage_secret_endpoint",
            return_value=False,
        ),
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s", yes=yes, context=context)
    return helm


def test_upgrade_prints_the_cluster_role_diff_and_asks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "stdin", TtyInput("y\n"))

    helm = _upgrade_with(tmp_path, APPROVAL_ARTIFACTS)

    out = capsys.readouterr().out
    assert "changes the agent's ClusterRoles on cluster 'corp-k8s'" in out
    assert (
        "  operator\n"
        "    + nodes: get, list\n"
        "    + storageclasses.storage.k8s.io: get\n"
        "    - clusterroles.rbac.authorization.k8s.io/namespace-secrets: bind\n"
    ) in out
    assert "Apply these ClusterRole changes? [y/N]" in out
    helm.assert_called_once()


def test_upgrade_stops_when_the_cluster_role_changes_are_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", TtyInput("n\n"))

    with (
        patch.object(
            ByocApi, "fetch_cluster_install_artifacts", return_value=APPROVAL_ARTIFACTS
        ),
        patch.object(ByocApi, "download_cluster_install_package") as download,
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.Abort),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    download.assert_not_called()
    helm.assert_not_called()


def test_upgrade_with_yes_applies_cluster_role_changes_without_asking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pinned_target: MagicMock,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    helm = _upgrade_with(tmp_path, APPROVAL_ARTIFACTS, yes=True, context="corp")

    out = capsys.readouterr().out
    assert "+ nodes: get, list" in out
    assert "[y/N]" not in out
    pinned_target.assert_called_once_with("corp-k8s", None, "corp")
    helm.assert_called_once()


def test_upgrade_shows_no_cluster_role_changes_for_another_cluster(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", TtyInput("y\n"))
    monkeypatch.setattr(
        f"{UPGRADE_MODULE}.require_registered_cluster", require_registered_cluster
    )

    with (
        patch.object(
            ByocApi,
            "find_cluster",
            return_value={"name": "corp-k8s", "clusterUid": "another-uid"},
        ),
        patch.object(
            ByocApi, "fetch_cluster_install_artifacts", return_value=APPROVAL_ARTIFACTS
        ),
        patch(f"{UPGRADE_MODULE}.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="is not Arena cluster 'corp-k8s'"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s", context="corp", yes=True)

    assert "ClusterRoles" not in capsys.readouterr().out
    helm.assert_not_called()


def test_upgrade_without_a_terminal_stops_before_cluster_role_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    with (
        patch.object(
            ByocApi, "fetch_cluster_install_artifacts", return_value=APPROVAL_ARTIFACTS
        ),
        patch.object(ByocApi, "download_cluster_install_package") as download,
        patch("agilerl.byoc.cluster_upgrade.helm_upgrade_install") as helm,
        pytest.raises(click.ClickException, match="pass --yes"),
    ):
        run_cluster_upgrade(MagicMock(), name="corp-k8s")

    download.assert_not_called()
    helm.assert_not_called()


@pytest.mark.parametrize(
    "artifacts",
    [
        {"agentHelmValuesYaml": AGENT_YAML},
        {
            "agentHelmValuesYaml": AGENT_YAML,
            "agentUpgradeRequiresApproval": False,
            "rbacDiff": [],
        },
    ],
)
def test_upgrade_without_cluster_role_changes_does_not_ask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    artifacts: dict[str, object],
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    helm = _upgrade_with(tmp_path, artifacts)

    assert "ClusterRoles" not in capsys.readouterr().out
    helm.assert_called_once()


def test_upgrade_command_passes_yes_with_the_context(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.byoc.commands.run_cluster_upgrade") as upgrade:
        result = CliRunner().invoke(
            build_cluster_upgrade_command(),
            ["--name", "corp-k8s", "--context", "corp", "--yes"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert upgrade.call_args.kwargs["yes"] is True
    assert upgrade.call_args.kwargs["context"] == "corp"
