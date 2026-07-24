"""Tests for ``arena on-prem cluster register`` and registration helpers."""

from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from agilerl.arena.config import CommandConfig
from agilerl.arena.on_prem import OnPremApi, build_cluster_register_command
from agilerl.arena.on_prem.cluster_register import run_cluster_register
from click.testing import CliRunner

LAB_BUNDLE = {
    "clusterId": 42,
    "token": "abc.def",
    "clusterApiUrl": "http://172.24.0.1:8443",
    "cluster": {
        "id": 42,
        "name": "lab-cluster",
        "installProfile": "lab",
        "storageEndpoint": "http://minio.storage.svc:9000",
        "storageBucket": "arena-data",
        "storageSecretName": "arena-storage",
    },
    "agentHelmValues": {
        "cluster": {
            "apiUrl": "http://172.24.0.1:8443",
            "id": 42,
        },
        "wireguard": {
            "gatewayHost": "gw.example.com",
            "gatewayPublicKey": "gw-pub",
            "peerPrivateKey": "peer-priv",
            "peerIp": "172.24.0.2/32",
            "preSharedKey": "psk",
        },
        "clusterToken": "abc.def",
        "storage": {
            "endpoint": "http://minio.storage.svc:9000",
            "bucket": "arena-data",
            "secretName": "arena-storage",
            "createSecret": True,
            "accessKeyId": "minioadmin",
            "secretAccessKey": "minioadmin",
        },
    },
    "storageHelmValues": {
        "bucket": {"name": "arena-data"},
        "secret": {
            "name": "arena-storage",
            "endpoint": "http://minio.storage.svc:9000",
        },
    },
}

ENTERPRISE_BUNDLE = {
    "clusterId": 7,
    "token": "ent.tok",
    "clusterApiUrl": "http://172.24.0.5:8443",
    "cluster": {
        "id": 7,
        "name": "prod-cluster",
        "installProfile": "enterprise",
        "storageEndpoint": "http://s3.corp.example.com:9000",
        "storageBucket": "arena-prod",
        "storageSecretName": "corp-s3",
        "storagePrefix": "org-1/",
    },
    "agentHelmValues": {
        "cluster": {
            "apiUrl": "http://172.24.0.5:8443",
            "id": 7,
        },
        "clusterToken": "ent.tok",
    },
}


class TestOnPremApiRegisterCluster:
    def test_register_cluster_passes_camel_case_body(
        self, on_prem_api: OnPremApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = LAB_BUNDLE
        on_prem_api.register_cluster(
            name="lab-cluster",
            install_profile="lab",
            storage_prefix="data/",
        )
        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/on-prem/clusters/register"
        assert body == {
            "name": "lab-cluster",
            "installProfile": "lab",
            "storagePrefix": "data/",
        }


class TestRunClusterRegister:
    def test_writes_lab_yaml_files(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with patch.object(OnPremApi, "enable") as enable_mock, patch.object(
            OnPremApi, "register_cluster", return_value=LAB_BUNDLE
        ) as register_mock:
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                profile="lab",
                output_dir=tmp_path,
                skip_enable=False,
                force=False,
            )
        enable_mock.assert_called_once()
        register_mock.assert_called_once()
        agent_path = tmp_path / "agent-helm-values.yaml"
        storage_path = tmp_path / "storage-helm-values.yaml"
        token_path = tmp_path / "cluster-token.txt"
        assert agent_path.is_file()
        assert storage_path.is_file()
        assert token_path.is_file()
        assert yaml.safe_load(agent_path.read_text(encoding="utf-8")) == LAB_BUNDLE[
            "agentHelmValues"
        ]
        assert yaml.safe_load(storage_path.read_text(encoding="utf-8")) == LAB_BUNDLE[
            "storageHelmValues"
        ]
        assert token_path.read_text(encoding="utf-8").strip() == "abc.def"
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    def test_enterprise_writes_agent_only(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with patch.object(OnPremApi, "enable"), patch.object(
            OnPremApi, "register_cluster", return_value=ENTERPRISE_BUNDLE
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                profile="enterprise",
                output_dir=tmp_path,
                skip_enable=False,
                force=False,
                storage_endpoint="http://s3.corp.example.com:9000",
                storage_bucket="arena-prod",
                storage_secret_name="corp-s3",
            )
        assert (tmp_path / "agent-helm-values.yaml").is_file()
        assert not (tmp_path / "storage-helm-values.yaml").exists()

    def test_enterprise_validation_requires_storage_fields(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        with pytest.raises(click.ClickException, match="storage-endpoint"):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                profile="enterprise",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
            )

    def test_skip_enable_skips_provider_enable(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with patch.object(OnPremApi, "enable") as enable_mock, patch.object(
            OnPremApi, "register_cluster", return_value=LAB_BUNDLE
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                profile="lab",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
            )
        enable_mock.assert_not_called()


class TestClusterRegisterCommand:
    def test_cli_invokes_run_cluster_register(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.on_prem.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.on_prem.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--name",
                    "lab-cluster",
                    "--profile",
                    "lab",
                    "--output-dir",
                    "/tmp/out",
                    "--skip-enable",
                ],
                obj=command_config,
            )
        assert result.exit_code == 0, result.output
        kwargs = run_mock.call_args.kwargs
        assert kwargs["name"] == "lab-cluster"
        assert kwargs["profile"] == "lab"
        assert kwargs["output_dir"] == Path("/tmp/out")
        assert kwargs["skip_enable"] is True

    def test_enterprise_cli_validation_fails_without_storage(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with patch(
            "agilerl.arena.on_prem.commands.arena_client",
            return_value=client_context(client),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "prod-cluster", "--profile", "enterprise", "--skip-enable"],
                obj=command_config,
            )
        assert result.exit_code != 0
        assert "storage-endpoint" in result.output
