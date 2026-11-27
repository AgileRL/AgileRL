# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster manifest``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agilerl.byoc import ClusterDynamicGroup, build_cluster_manifest_command
from agilerl.config import CommandConfig

ARGOCD_YAML = "apiVersion: argoproj.io/v1alpha1\nkind: Application\n"
FLUX_YAML = "kind: HelmRepository\n---\nkind: HelmRelease\n"
ARTIFACTS = {
    "agentHelmValuesYaml": "image: {}\n",
    "agentUpgradeRequiresApproval": False,
    "rbacDiff": [],
    "argocdApplicationYaml": ARGOCD_YAML,
    "fluxYaml": FLUX_YAML,
}


def invoke(
    command_config: CommandConfig, args: list[str], artifacts: dict[str, Any]
) -> tuple[Any, MagicMock]:
    fetch = MagicMock(return_value=artifacts)
    with (
        patch("agilerl.byoc.commands.arena_client") as arena_client,
        patch(
            "agilerl.byoc.cluster_manifest.ByocApi.fetch_cluster_install_artifacts",
            fetch,
        ),
    ):
        arena_client.return_value.__enter__.return_value = MagicMock()
        result = CliRunner().invoke(
            build_cluster_manifest_command(), args, obj=command_config
        )
    return result, fetch


class TestClusterManifest:
    @pytest.mark.parametrize(
        ("manifest_format", "expected"), [("argocd", ARGOCD_YAML), ("flux", FLUX_YAML)]
    )
    def test_prints_the_requested_manifest(
        self, command_config: CommandConfig, manifest_format: str, expected: str
    ) -> None:
        result, fetch = invoke(
            command_config,
            ["--name", " corp-k8s ", "--format", manifest_format],
            ARTIFACTS,
        )

        assert result.exit_code == 0, result.output
        assert result.stdout == expected
        fetch.assert_called_once_with("corp-k8s")

    def test_writes_the_manifest_to_a_file(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        output = tmp_path / "arena-byoc-agent.yaml"

        result, _ = invoke(
            command_config,
            ["--name", "corp-k8s", "--format", "flux", "-o", str(output)],
            ARTIFACTS,
        )

        assert result.exit_code == 0, result.output
        assert output.read_text(encoding="utf-8") == FLUX_YAML
        assert result.stdout == ""

    def test_fails_when_arena_serves_no_manifests(
        self, command_config: CommandConfig
    ) -> None:
        result, _ = invoke(
            command_config,
            ["--name", "corp-k8s", "--format", "argocd"],
            {"agentHelmValuesYaml": "image: {}\n"},
        )

        assert result.exit_code != 0
        assert "serves no GitOps manifests" in result.output

    def test_prints_clusterrole_changes_to_stderr(
        self, command_config: CommandConfig
    ) -> None:
        artifacts = {
            **ARTIFACTS,
            "agentUpgradeRequiresApproval": True,
            "rbacDiff": [
                {
                    "cluster_role": "operator",
                    "added": [
                        {"api_group": "", "resource": "nodes", "verbs": ["list"]}
                    ],
                    "removed": [],
                }
            ],
        }

        result, _ = invoke(
            command_config, ["--name", "corp-k8s", "--format", "argocd"], artifacts
        )

        assert result.exit_code == 0, result.output
        assert result.stdout == ARGOCD_YAML
        assert "+ nodes: list" in result.stderr

    def test_rejects_a_blank_name(self, command_config: CommandConfig) -> None:
        result, fetch = invoke(
            command_config, ["--name", "  ", "--format", "argocd"], ARTIFACTS
        )

        assert result.exit_code != 0
        assert "--name is required." in result.output
        fetch.assert_not_called()

    def test_rejects_an_unknown_format(self, command_config: CommandConfig) -> None:
        result, fetch = invoke(
            command_config, ["--name", "corp-k8s", "--format", "helm"], ARTIFACTS
        )

        assert result.exit_code != 0
        fetch.assert_not_called()

    def test_is_a_cluster_command(self) -> None:
        assert "manifest" in ClusterDynamicGroup().commands
