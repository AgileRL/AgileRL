# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for cluster token rotation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_helm import (
    AGENT_NAMESPACE,
    AGENT_RELEASE,
    agent_deployment_selector,
)
from agilerl.arena.byoc.cluster_token import (
    DEFAULT_CLUSTER_TOKEN_SECRET_NAME,
    apply_cluster_token_secret,
    cluster_token_secret_name,
    resolve_cluster_kubeconfig,
    run_cluster_rotate_token,
)
from agilerl.arena.client import ArenaClient


class TestClusterTokenSecretName:
    def test_uses_install_bundle_agent_helm_values_yaml(self) -> None:
        name = cluster_token_secret_name(
            {
                "token": "rotated.tok",
                "token_id": "11111111-1111-1111-1111-111111111111",
                "install_bundle": {
                    "agent_helm_values_yaml": (
                        'existingClusterTokenSecret: "custom-token-secret"\n'
                    ),
                },
            }
        )

        assert name == "custom-token-secret"

    def test_uses_camel_case_install_bundle_yaml(self) -> None:
        name = cluster_token_secret_name(
            {
                "installBundle": {
                    "agentHelmValuesYaml": (
                        "existingClusterTokenSecret: camel-token-secret\n"
                    ),
                }
            }
        )

        assert name == "camel-token-secret"

    def test_uses_helm_existing_secret(self) -> None:
        name = cluster_token_secret_name(
            {
                "agentHelmValues": {
                    "existingClusterTokenSecret": " custom-token-secret ",
                }
            }
        )

        assert name == "custom-token-secret"

    def test_defaults_to_helm_release_secret(self) -> None:
        assert cluster_token_secret_name({}) == DEFAULT_CLUSTER_TOKEN_SECRET_NAME


class TestResolveClusterKubeconfig:
    def test_requires_explicit_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"

        with pytest.raises(click.ClickException, match="Kubeconfig is not a file"):
            resolve_cluster_kubeconfig("prod", missing)

    def test_uses_provisioned_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        kubeconfig = tmp_path / "arena-cluster-prod" / "kubeconfig"
        kubeconfig.parent.mkdir()
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

        resolved = resolve_cluster_kubeconfig("prod", None)

        assert resolved == kubeconfig.resolve()

    def test_uses_kubeconfig_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kubeconfig = tmp_path / "env-kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))

        resolved = resolve_cluster_kubeconfig("prod", None)

        assert resolved == kubeconfig.resolve()

    def test_uses_home_kube_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        kubeconfig = home / ".kube" / "config"
        kubeconfig.parent.mkdir(parents=True)
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("KUBECONFIG", raising=False)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))

        resolved = resolve_cluster_kubeconfig("prod", None)

        assert resolved == kubeconfig.resolve()


class TestApplyClusterTokenSecret:
    def test_applies_secret_and_restarts_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.byoc.cluster_token.require_kubectl", lambda: None
        )
        calls: list[list[str]] = []

        def run_side_effect(argv: list[str], **kwargs: object) -> MagicMock:
            calls.append(list(argv))
            if argv[:3] == ["kubectl", "apply", "-f"]:
                secret = next(yaml.safe_load_all(str(kwargs["input"])))
                assert secret["metadata"] == {
                    "name": "agent-token",
                    "namespace": AGENT_NAMESPACE,
                }
                assert secret["stringData"] == {"cluster-token": "rotated.tok"}
            return MagicMock(returncode=0, stderr="", stdout="")

        apply_cluster_token_secret(
            token="rotated.tok",
            secret_name="agent-token",
            namespace=AGENT_NAMESPACE,
            kubeconfig_path=kubeconfig,
            agent_release=AGENT_RELEASE,
            run=MagicMock(side_effect=run_side_effect),
        )

        assert calls[0] == ["kubectl", "apply", "-f", "-"]
        assert calls[1] == [
            "kubectl",
            "rollout",
            "restart",
            "deployment",
            "-n",
            AGENT_NAMESPACE,
            "-l",
            agent_deployment_selector(AGENT_RELEASE),
        ]

    def test_restarts_the_discovered_helm_release(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.byoc.cluster_token.require_kubectl", lambda: None
        )
        calls: list[list[str]] = []

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            calls.append(list(argv))
            return MagicMock(returncode=0, stderr="", stdout="")

        apply_cluster_token_secret(
            token="rotated.tok",
            secret_name="cluster-token",
            namespace="arena-fdh-2",
            kubeconfig_path=kubeconfig,
            agent_release="op-arena-fdh-2",
            run=MagicMock(side_effect=run_side_effect),
        )

        assert calls[1][calls[1].index("-l") + 1] == agent_deployment_selector(
            "op-arena-fdh-2"
        )

    def test_raises_when_agent_restart_fails(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

        def run(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[:2] == ["kubectl", "apply"]:
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="rollout failed")

        with patch("agilerl.arena.byoc.cluster_token.require_kubectl", lambda: None):
            with pytest.raises(click.ClickException, match="could not restart"):
                apply_cluster_token_secret(
                    token="tok",
                    secret_name="cluster-token",
                    namespace=AGENT_NAMESPACE,
                    kubeconfig_path=kubeconfig,
                    agent_release=AGENT_RELEASE,
                    run=run,
                )


class TestRunClusterRotateToken:
    def test_rotates_token_and_updates_secret(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        client = MagicMock(spec=ArenaClient)
        apply = MagicMock()

        with (
            patch.object(ByocApi, "find_cluster", return_value={"name": "prod"}),
            patch.object(
                ByocApi,
                "rotate_cluster_token",
                return_value={
                    "token": " rotated.tok ",
                    "token_id": "11111111-1111-1111-1111-111111111111",
                    "install_bundle": {
                        "agent_helm_values_yaml": (
                            'existingClusterTokenSecret: "agent-token"\n'
                        ),
                    },
                },
            ) as rotate_mock,
            patch(
                "agilerl.arena.byoc.cluster_token.apply_cluster_token_secret",
                apply,
            ),
            patch(
                "agilerl.arena.byoc.cluster_token.discover_agent_install",
                return_value=(AGENT_RELEASE, AGENT_NAMESPACE),
            ),
        ):
            token = run_cluster_rotate_token(client, name="prod", kubeconfig=kubeconfig)

        assert token == "rotated.tok"
        rotate_mock.assert_called_once_with("prod")
        assert apply.call_args.kwargs["token"] == "rotated.tok"
        assert apply.call_args.kwargs["secret_name"] == "agent-token"
        assert apply.call_args.kwargs["namespace"] == AGENT_NAMESPACE
        assert apply.call_args.kwargs["agent_release"] == AGENT_RELEASE
        assert apply.call_args.kwargs["kubeconfig_path"] == kubeconfig.resolve()

    def test_discovers_agent_namespace_from_helm(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        client = MagicMock(spec=ArenaClient)
        apply = MagicMock()

        with (
            patch.object(ByocApi, "find_cluster", return_value={"name": "prod"}),
            patch.object(
                ByocApi,
                "rotate_cluster_token",
                return_value={"token": "rotated.tok"},
            ),
            patch(
                "agilerl.arena.byoc.cluster_token.apply_cluster_token_secret",
                apply,
            ),
            patch(
                "agilerl.arena.byoc.cluster_token.discover_agent_install",
                return_value=("prod", "arena-agent"),
            ) as discover_mock,
        ):
            run_cluster_rotate_token(client, name="prod", kubeconfig=kubeconfig)

        assert discover_mock.call_args.args[0] == kubeconfig.resolve()
        assert discover_mock.call_args.kwargs["cluster_name"] == "prod"
        assert discover_mock.call_args.kwargs["agent_namespace"] is None
        assert apply.call_args.kwargs["namespace"] == "arena-agent"
        assert apply.call_args.kwargs["agent_release"] == "prod"

    def test_rejects_missing_cluster(self) -> None:
        client = MagicMock(spec=ArenaClient)

        with patch.object(ByocApi, "find_cluster", return_value=None):
            with pytest.raises(
                click.ClickException, match="No registered BYOC cluster"
            ):
                run_cluster_rotate_token(client, name="missing")

    def test_rejects_missing_token(self) -> None:
        client = MagicMock(spec=ArenaClient)

        with (
            patch.object(ByocApi, "find_cluster", return_value={"name": "prod"}),
            patch.object(ByocApi, "rotate_cluster_token", return_value={}),
        ):
            with pytest.raises(click.ClickException, match="returned no token"):
                run_cluster_rotate_token(client, name="prod")


class TestSecretFromAgentHelmValuesYaml:
    def test_ignores_non_string_values(self) -> None:
        assert (
            cluster_token_secret_name(
                {"install_bundle": {"agent_helm_values_yaml": 12}}
            )
            == DEFAULT_CLUSTER_TOKEN_SECRET_NAME
        )

    def test_ignores_yaml_without_secret_key(self) -> None:
        assert (
            cluster_token_secret_name(
                {"installBundle": {"agentHelmValuesYaml": "clusterName: prod\n"}}
            )
            == DEFAULT_CLUSTER_TOKEN_SECRET_NAME
        )
