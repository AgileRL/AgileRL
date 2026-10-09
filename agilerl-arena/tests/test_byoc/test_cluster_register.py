# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster register`` and registration helpers."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml
from click.testing import CliRunner

from agilerl.arena.byoc import ByocApi, build_cluster_register_command
from agilerl.arena.byoc.cluster_helm import AGENT_RELEASE
from agilerl.arena.byoc.cluster_register import (
    ClusterResourceClass,
    _agent_values_for_helm,
    _bundle_helm_values_yaml,
    _install_charts_from_arena_package,
    _registered_cluster_name,
    run_cluster_register,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.terraform import ClusterOutputs
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAPIError

NEBIUS_SPEC = """
provider: nebius
name: spec-cluster
terraform_state:
  bucket: spec-cluster-tfstate
arena:
  storage:
    bucket: provisioned-bucket
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant
  region: eu-north1
  object_storage:
    size_gib: 1
"""

LAB_BUNDLE = {
    "token": "abc.def",
    "upserted": False,
    "clusterApiUrl": "http://172.24.0.1:8443",
    "cluster": {
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
    "token": "ent.tok",
    "upserted": False,
    "clusterApiUrl": "http://172.24.0.5:8443",
    "cluster": {
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


class TestByocApiRegisterCluster:
    def test_register_cluster_passes_camel_case_body(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = LAB_BUNDLE
        byoc_api.register_cluster(
            name="lab-cluster",
            install_storage=True,
            storage_prefix="data/",
        )
        invoke, body = mock_client._invoke_manifest_command.call_args.args
        assert invoke["path"] == "/api/cli/v1/byoc/clusters/register"
        assert body == {
            "name": "lab-cluster",
            "installStorage": True,
            "storagePrefix": "data/",
        }


class TestAgentValuesForHelm:
    def test_merges_inference_domain_and_hostname_template(self) -> None:
        # Arrange
        agent_values = {"cluster": {"id": 1}}

        # Act
        merged = _agent_values_for_helm(
            agent_values,
            "token",
            inference_domain="inference.example.com",
            inference_hostname_template="inference-{deploymentId}",
        )

        # Assert
        assert merged["inference"] == {
            "domain": "inference.example.com",
            "hostnameTemplate": "inference-{deploymentId}",
        }


class TestByocApiRegisterClusterInference:
    def test_register_cluster_passes_domain_and_host_template_separately(
        self, byoc_api: ByocApi, mock_client: MagicMock
    ) -> None:
        mock_client._invoke_manifest_command.return_value = ENTERPRISE_BUNDLE
        byoc_api.register_cluster(
            name="prod-cluster",
            hostname_template="inference-{deploymentId}",
            inference_domain="inference.example.com",
        )
        _, body = mock_client._invoke_manifest_command.call_args.args
        assert body == {
            "name": "prod-cluster",
            "hostnameTemplate": "inference-{deploymentId}",
            "domain": "inference.example.com",
        }


class TestRunClusterRegister:
    @pytest.fixture(autouse=True)
    def _no_helm(self):
        # Release resolution runs `helm list` against the live cluster otherwise.
        with patch(
            "agilerl.arena.byoc.cluster_helm.helm_available", return_value=False
        ):
            yield

    def test_writes_lab_yaml_files(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable") as enable_mock,
            patch.object(
                ByocApi, "register_cluster", return_value=LAB_BUNDLE
            ) as register_mock,
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=False,
                force=False,
                install_storage=True,
            )
        enable_mock.assert_called_once()
        register_mock.assert_called_once()
        agent_path = tmp_path / "agent-helm-values.yaml"
        storage_path = tmp_path / "storage-helm-values.yaml"
        token_path = tmp_path / "cluster-token.txt"
        assert agent_path.is_file()
        assert storage_path.is_file()
        assert token_path.is_file()
        assert (
            yaml.safe_load(agent_path.read_text(encoding="utf-8"))
            == LAB_BUNDLE["agentHelmValues"]
        )
        assert (
            yaml.safe_load(storage_path.read_text(encoding="utf-8"))
            == LAB_BUNDLE["storageHelmValues"]
        )
        assert token_path.read_text(encoding="utf-8").strip() == "abc.def"
        if sys.platform != "win32":
            assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    def test_passes_byoc_provider_to_register_api(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        provider = {
            "provider": "nebius",
            "config": {"tenant_id": "tenant-1", "region": "eu-north1"},
        }
        with (
            patch.object(ByocApi, "enable"),
            patch.object(
                ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE
            ) as register_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                byoc_provider=provider,
            )

        assert register_mock.call_args.kwargs["byoc_provider"] == provider

    def test_writes_inference_settings_into_agent_values(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                hostname_template="inference-{deploymentId}",
                inference_domain="inference.example.com",
            )

        agent = yaml.safe_load(
            (tmp_path / "agent-helm-values.yaml").read_text(encoding="utf-8")
        )

        assert agent["inference"] == {
            "domain": "inference.example.com",
            "hostnameTemplate": "inference-{deploymentId}",
        }

    def test_enterprise_writes_agent_only(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=False,
                force=False,
                storage_endpoint="http://s3.corp.example.com:9000",
                storage_bucket="arena-prod",
                storage_secret_name="corp-s3",
            )
        assert (tmp_path / "agent-helm-values.yaml").is_file()
        assert not (tmp_path / "storage-helm-values.yaml").exists()

    def test_storage_fields_are_optional(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(
                ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE
            ) as register_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
            )
        register_mock.assert_called_once()
        assert (tmp_path / "agent-helm-values.yaml").is_file()

    def test_skip_enable_skips_provider_enable(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable") as enable_mock,
            patch.object(ByocApi, "register_cluster", return_value=LAB_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install_storage=True,
            )
        enable_mock.assert_not_called()

    def test_upsert_without_token_skips_token_file(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        upsert_bundle = {
            **ENTERPRISE_BUNDLE,
            "token": None,
            "upserted": True,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.5:8443", "id": 7},
                "existingClusterTokenSecret": (
                    "arena-byoc-agent-arena-byoc-agent-cluster-token"
                ),
                "existingWireguardSecret": (
                    "arena-byoc-agent-arena-byoc-agent-wireguard"
                ),
            },
        }
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                storage_endpoint="http://s3.corp.example.com:9000",
                storage_bucket="arena-prod",
                storage_secret_name="corp-s3",
            )
        assert not (tmp_path / "cluster-token.txt").exists()
        agent = yaml.safe_load(
            (tmp_path / "agent-helm-values.yaml").read_text(encoding="utf-8")
        )
        assert agent["existingClusterTokenSecret"] == (
            "arena-byoc-agent-arena-byoc-agent-cluster-token"
        )
        assert "clusterToken" not in agent
        assert "Updated cluster" in capsys.readouterr().out

    def test_install_without_token_fails_on_upsert(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        upsert_bundle = {
            **ENTERPRISE_BUNDLE,
            "token": None,
            "upserted": True,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.5:8443", "id": 7},
                "existingClusterTokenSecret": (
                    "arena-byoc-agent-arena-byoc-agent-cluster-token"
                ),
            },
        }
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
            pytest.raises(
                click.ClickException, match="Cannot --install without an agent token"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                no_write=True,
            )
        install_mock.assert_not_called()

    def test_install_with_rotated_token_on_upsert(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        upsert_bundle = {
            **ENTERPRISE_BUNDLE,
            "token": "rotated.token.secret",
            "upserted": True,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.5:8443", "id": 7},
                "clusterToken": "rotated.token.secret",
            },
        }
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                no_write=True,
            )
        install_mock.assert_called_once()
        agent_values = install_mock.call_args.kwargs["agent_values"]
        assert agent_values["clusterToken"] == "rotated.token.secret"
        assert "existingClusterTokenSecret" not in agent_values

    def test_install_passes_agent_namespace(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                agent_namespace="arena-agent",
            )

        assert install_mock.call_args.kwargs["agent_namespace"] == "arena-agent"

    def test_install_prints_the_kubeconfig_helm_will_use(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
            )

        assert f"Using kubeconfig {kubeconfig}" in capsys.readouterr().out

    def test_install_prints_the_default_kubeconfig_when_unset(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv("KUBECONFIG", raising=False)
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
            )

        default = Path.home() / ".kube" / "config"
        assert f"Using kubeconfig {default}" in capsys.readouterr().out

    def test_install_names_the_release_after_the_cluster(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                agent_namespace="arena-agent",
            )

        assert install_mock.call_args.kwargs["agent_release"] == "prod-cluster"

    def test_install_keeps_an_existing_default_release(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register.resolve_install_release",
                return_value=AGENT_RELEASE,
            ),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
            )

        assert install_mock.call_args.kwargs["agent_release"] == AGENT_RELEASE

    def test_install_storage_upsert_without_values_skips_storage(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        upsert_bundle = {
            **LAB_BUNDLE,
            "token": "rotated.tok",
            "upserted": True,
            "storageHelmValues": None,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.1:8443", "id": 42},
            },
        }
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                install_storage=True,
                no_write=True,
            )
        install_mock.assert_called_once()
        assert install_mock.call_args.kwargs["include_storage"] is False
        assert "Existing MinIO credentials reused" in capsys.readouterr().out

    def test_install_storage_without_install_skips_agent_chart(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bundle = {**LAB_BUNDLE, "token": None}
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=bundle),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install_storage=True,
            )

        install_mock.assert_called_once()
        assert install_mock.call_args.kwargs["include_storage"] is True
        assert install_mock.call_args.kwargs["install_agent"] is False
        out = capsys.readouterr().out
        assert "Install agent:" in out
        assert "Install storage" not in out

    def test_install_storage_from_local_charts_skips_agent(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        charts_dir = tmp_path / "helm-setup"
        charts_dir.mkdir()
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=LAB_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register.install_lab_cluster_charts"
            ) as lab_mock,
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path / "out",
                skip_enable=True,
                force=False,
                install_storage=True,
                charts_dir=charts_dir,
            )

        lab_mock.assert_called_once()
        assert lab_mock.call_args.kwargs["install_agent"] is False

    def test_install_from_local_charts_installs_agent_without_storage(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        charts_dir = tmp_path / "helm-setup"
        charts_dir.mkdir()
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register.install_enterprise_agent_chart"
            ) as agent_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path / "out",
                skip_enable=True,
                force=False,
                install=True,
                charts_dir=charts_dir,
            )

        agent_mock.assert_called_once()

    def test_no_write_install_storage_from_local_charts(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        charts_dir = tmp_path / "helm-setup"
        charts_dir.mkdir()
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=LAB_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register.install_lab_cluster_charts"
            ) as lab_mock,
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path / "out",
                skip_enable=True,
                force=False,
                install_storage=True,
                no_write=True,
                charts_dir=charts_dir,
            )

        lab_mock.assert_called_once()
        assert lab_mock.call_args.kwargs["install_agent"] is False
        assert not (tmp_path / "out" / "storage-helm-values.yaml").exists()

    def test_next_steps_include_storage_when_values_were_not_installed(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=LAB_BUNDLE),
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
            )

        out = capsys.readouterr().out
        assert "Install storage" in out
        assert "Install agent:" in out

    def test_creates_resource_class_without_agent_install(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        resource_class = ClusterResourceClass(
            name="arena-lab-gpu",
            num_nodes=1,
            node_selector={"class": "gpu"},
            metadata={},
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=LAB_BUNDLE),
            patch.object(ByocApi, "find_class", return_value=None),
            patch.object(
                ByocApi, "create_class", return_value={"name": "arena-lab-gpu"}
            ) as create_mock,
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install_storage=True,
                resource_classes=[resource_class],
            )

        create_mock.assert_called_once_with(
            name="arena-lab-gpu",
            num_nodes=1,
            on_prem_cluster="lab-cluster",
            node_selector={"class": "gpu"},
            metadata={},
        )
        assert "Resource class 'arena-lab-gpu' is linked" in capsys.readouterr().out

    def test_no_write_skips_persisted_files(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                no_write=True,
            )
        assert not (tmp_path / "agent-helm-values.yaml").exists()
        assert "Wrote Helm values" not in capsys.readouterr().out

    def test_no_write_install_uses_arena_package(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                no_write=True,
            )
        assert not (tmp_path / "agent-helm-values.yaml").exists()
        install_mock.assert_called_once()
        out = capsys.readouterr().out
        assert "no local config files written" in out

    def test_creates_gpu_resource_class_after_install(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        resource_class = ClusterResourceClass(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata={
                "computeResource": {
                    "numCpus": 16,
                    "numGpus": 1,
                    "memoryBytes": "200 GiB",
                }
            },
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch.object(ByocApi, "find_class", return_value=None),
            patch.object(
                ByocApi, "create_class", return_value={"name": "arena-nebius-gpu"}
            ) as create_mock,
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                resource_classes=[resource_class],
            )

        create_mock.assert_called_once_with(
            name="arena-nebius-gpu",
            num_nodes=1,
            on_prem_cluster="prod-cluster",
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata=resource_class.metadata,
        )
        assert "Created resource class 'arena-nebius-gpu'" in capsys.readouterr().out

    def test_skips_resource_class_when_name_exists(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        resource_class = ClusterResourceClass(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata={
                "computeResource": {
                    "numCpus": 16,
                    "numGpus": 1,
                    "memoryBytes": "200 GiB",
                }
            },
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch.object(
                ByocApi, "find_class", return_value={"name": "arena-nebius-gpu"}
            ),
            patch.object(ByocApi, "create_class") as create_mock,
            patch(
                "agilerl.arena.byoc.cluster_register._install_charts_from_arena_package"
            ),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                resource_classes=[resource_class],
            )

        create_mock.assert_not_called()
        assert "already exists" in capsys.readouterr().out


class TestClusterRegisterCommand:
    @pytest.fixture(autouse=True)
    def _cluster_already_in_terraform_state(self):
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.cluster_exists",
                return_value=True,
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.status",
                return_value={"worker_node_group_ids": {"workers": "ng-1"}},
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                return_value=None,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.aws.provider.AwsProvider.ensure_gateway",
                return_value=None,
            ),
        ):
            yield

    def test_cli_uses_registration_settings_from_spec(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        """Cluster registration accepts a cloud spec without duplicated flags."""
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(
            """
provider: nebius
name: spec-cluster
terraform_state:
  bucket: spec-cluster-tfstate
arena:
  storage:
    bucket: provisioned-bucket
    endpoint: https://storage.example.com
    prefix: models/
    secret_name: inference-storage
  inference:
    domain: inference.example.com
  workloads:
    ray_data_pvc_size: 20Gi
nebius:
  tenant_id: tenant
  project_id: project
  storage_project_id: storage-project
  subnet_id: subnet
  object_storage:
    size_gib: 1
""",
            encoding="utf-8",
        )
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        kwargs = run_mock.call_args.kwargs
        assert kwargs["name"] == "spec-cluster"
        assert kwargs["storage_bucket"] == "provisioned-bucket"
        assert kwargs["storage_prefix"] == "models/"
        assert kwargs["ray_data_pvc_size"] == "20Gi"
        assert kwargs["storage_secret_name"] == "inference-storage"
        assert kwargs["install_storage"] is False
        assert kwargs["resource_classes"][0].name == "spec-cluster-workers"
        assert kwargs["resource_classes"][0].node_selector == {
            "nebius.com/node-group-id": "ng-1"
        }
        assert kwargs["byoc_provider"] == {
            "provider": "nebius",
            "config": {
                "tenant_id": "tenant",
                "project_id": "project",
                "storage_project_id": "storage-project",
                "subnet_id": "subnet",
                "terraform_state": {"bucket": "spec-cluster-tfstate"},
                "object_storage_size_gib": 1,
                "workers": [
                    {
                        "name": "workers",
                        "min_node_count": 0,
                        "max_node_count": 10,
                        "platform": "gpu-h100-sxm",
                        "preset": "1gpu-16vcpu-200gb",
                        "gpu_drivers_preset": "cuda13.0",
                    }
                ],
            },
        }

    def test_register_rejects_worker_node_group_ids_that_are_not_strings(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.status",
                return_value={"worker_node_group_ids": {"workers": 1}},
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "string map" in result.output

    def test_cli_omits_empty_nebius_ids_from_byoc_provider(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(
            """
provider: nebius
name: spec-cluster
terraform_state:
  bucket: spec-cluster-tfstate
arena:
  storage:
    bucket: provisioned-bucket
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant
  region: eu-north1
  object_storage:
    size_gib: 1
""",
            encoding="utf-8",
        )
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert run_mock.call_args.kwargs["byoc_provider"] == {
            "provider": "nebius",
            "config": {
                "tenant_id": "tenant",
                "region": "eu-north1",
                "terraform_state": {"bucket": "spec-cluster-tfstate"},
                "object_storage_size_gib": 1,
                "workers": [
                    {
                        "name": "workers",
                        "min_node_count": 0,
                        "max_node_count": 10,
                        "platform": "gpu-h100-sxm",
                        "preset": "1gpu-16vcpu-200gb",
                        "gpu_drivers_preset": "cuda13.0",
                    }
                ],
            },
        }

    def test_install_uses_cluster_kubeconfig_when_the_cluster_exists(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        kubeconfig.parent.mkdir()
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        seen: dict[str, str | None] = {}

        def capture(*_args: object, **_kwargs: object) -> None:
            seen["kubeconfig"] = os.environ.get("KUBECONFIG")

        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.arena.byoc.commands.run_cluster_register",
                side_effect=capture,
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert seen["kubeconfig"] == str(kubeconfig.resolve())

    def test_install_creates_the_gateway(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        kubeconfig.parent.mkdir()
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        refs = [
            {
                "group": "gateway.networking.k8s.io",
                "kind": "Gateway",
                "name": "arena",
                "namespace": "arena",
                "sectionName": "https",
            }
        ]
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                return_value=refs,
            ) as gateway,
            patch("agilerl.arena.byoc.commands.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        gateway.assert_called_once()
        assert gateway.call_args.args[1] == kubeconfig.resolve()
        assert register.call_args.kwargs["gateway_api_parent_refs"] == refs

    def test_install_fetches_cluster_kubeconfig_when_the_file_is_missing(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        fetched = tmp_path / "fetched-kubeconfig"
        seen: dict[str, str | None] = {}

        def capture(*_args: object, **_kwargs: object) -> None:
            seen["kubeconfig"] = os.environ.get("KUBECONFIG")

        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.arena.byoc.commands.run_cluster_register",
                side_effect=capture,
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.write_kubeconfig",
                return_value=fetched,
            ) as write_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        write_mock.assert_called_once()
        assert seen["kubeconfig"] == str(fetched)

    def test_provisions_when_nebius_cluster_is_missing(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(
            """
provider: nebius
name: spec-cluster
terraform_state:
  bucket: spec-cluster-tfstate
  create_bucket: true
arena:
  storage:
    bucket: provisioned-bucket
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant
  region: eu-north1
  object_storage:
    size_gib: 1
""",
            encoding="utf-8",
        )
        outputs = ClusterOutputs(
            cluster_name="spec-cluster",
            context="spec-cluster",
            kubeconfig_path=tmp_path / "kubeconfig",
            storage_access_key_id="key",
            storage_secret_access_key="secret",
            worker_node_group_ids={"workers": "group-1"},
            project_id="project-created",
            storage_project_id="storage-created",
        )
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.cluster_exists",
                return_value=False,
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.provision",
                return_value=outputs,
            ) as provision_mock,
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec_path), "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert provision_mock.call_args.kwargs["create_state_bucket"] is True
        assert run_mock.call_args.kwargs["byoc_provider"]["config"]["project_id"] == (
            "project-created"
        )
        assert run_mock.call_args.kwargs["resource_classes"][0].name == (
            "spec-cluster-workers"
        )

    def test_recovers_nebius_settings_before_checking_terraform_state(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        seen: dict[str, str | None] = {}

        def exists(_self: object, spec: ClusterSpec, state_dir: Path) -> bool:
            assert state_dir.name == "terraform"
            seen["project_id"] = spec.nebius_cloud().project_id
            seen["storage_project_id"] = spec.nebius_cloud().storage_project_id
            seen["state_key"] = spec.terraform_state.key
            return True

        client = MagicMock()
        client._request.return_value = [
            {
                "name": "spec-cluster",
                "byoc_provider": {
                    "provider": "nebius",
                    "config": {
                        "project_id": "project-1",
                        "storage_project_id": "storage-project-1",
                        "subnet_id": "subnet-1",
                        "terraform_state": {
                            "bucket": "spec-cluster-tfstate",
                            "key": "clusters/spec-cluster/terraform.tfstate",
                        },
                    },
                },
            }
        ]
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.cluster_exists",
                exists,
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.provision",
            ) as provision_mock,
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Using nebius settings stored in Arena" in result.output
        assert seen == {
            "project_id": "project-1",
            "storage_project_id": "storage-project-1",
            "state_key": "clusters/spec-cluster/terraform.tfstate",
        }
        provision_mock.assert_not_called()
        config = run_mock.call_args.kwargs["byoc_provider"]["config"]
        assert config["project_id"] == "project-1"
        assert config["storage_project_id"] == "storage-project-1"
        assert config["subnet_id"] == "subnet-1"

    def test_does_not_provision_when_stored_settings_cannot_be_loaded(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        client = MagicMock()
        client._request.side_effect = ArenaAPIError("unauthorized", status_code=401)
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.cluster_exists",
                return_value=False,
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.provision",
            ) as provision_mock,
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec_path), "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        provision_mock.assert_not_called()

    def test_cli_installs_minio_from_spec(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(
            """
provider: nebius
name: spec-cluster
terraform_state:
  bucket: spec-cluster-tfstate
arena:
  storage:
    bucket: provisioned-bucket
    endpoint: https://storage.example.com
    install: true
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant
  project_id: project
  storage_project_id: storage-project
  subnet_id: subnet
  object_storage:
    size_gib: 1
""",
            encoding="utf-8",
        )
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.write_kubeconfig",
                return_value=tmp_path / "kubeconfig",
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        kwargs = run_mock.call_args.kwargs
        assert kwargs["install_storage"] is True
        assert kwargs["storage_secret_name"] == "arena-storage"

    def test_cli_lab_flag_sets_install_storage_and_narrow_ips(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "lab-cluster", "--lab", "--skip-enable"],
                obj=command_config,
            )
        assert result.exit_code == 0, result.output
        kwargs = run_mock.call_args.kwargs
        assert kwargs["install"] is False
        assert kwargs["install_storage"] is True
        assert kwargs["narrow_allowed_ips"] is True

    def test_cli_invokes_run_cluster_register(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--name",
                    "lab-cluster",
                    "--install-storage",
                    "--narrow-allowed-ips",
                    "--output-dir",
                    "/tmp/out",
                    "--skip-enable",
                ],
                obj=command_config,
            )
        assert result.exit_code == 0, result.output
        kwargs = run_mock.call_args.kwargs
        assert kwargs["name"] == "lab-cluster"
        assert kwargs["install_storage"] is True
        assert kwargs["narrow_allowed_ips"] is True
        assert kwargs["output_dir"] == Path("/tmp/out").resolve()
        assert kwargs["skip_enable"] is True
        assert kwargs["byoc_provider"] is None

    def test_cli_passes_agent_namespace(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--name",
                    "prod-cluster",
                    "--install",
                    "--agent-namespace",
                    "arena-agent",
                    "--skip-enable",
                ],
                obj=command_config,
            )
        assert result.exit_code == 0, result.output
        assert run_mock.call_args.kwargs["agent_namespace"] == "arena-agent"

    def test_cli_register_without_storage_succeeds(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "prod-cluster", "--skip-enable"],
                obj=command_config,
            )
        assert result.exit_code == 0, result.output
        run_mock.assert_called_once()


class TestBundleHelmValuesYaml:
    def test_omits_storage_yaml_when_not_requested(self) -> None:
        agent_yaml, storage_yaml = _bundle_helm_values_yaml(
            {"clusterName": "prod"},
            {"bucket": "arena-data"},
            include_storage=False,
        )

        assert "clusterName: prod" in agent_yaml
        assert storage_yaml is None

    def test_dumps_storage_yaml_when_requested(self) -> None:
        agent_yaml, storage_yaml = _bundle_helm_values_yaml(
            {"clusterName": "prod"},
            {"bucket": "arena-data"},
            include_storage=True,
        )

        assert "clusterName: prod" in agent_yaml
        assert storage_yaml is not None
        assert "bucket: arena-data" in storage_yaml


class TestRegisteredClusterName:
    def test_falls_back_to_requested_name(self) -> None:
        assert _registered_cluster_name({}, "requested") == "requested"

    def test_uses_cluster_name_from_bundle(self) -> None:
        assert (
            _registered_cluster_name({"cluster": {"name": " stored "}}, "requested")
            == "stored"
        )


class TestInstallChartsFromArenaPackage:
    def test_downloads_and_installs_package(self, tmp_path: Path) -> None:
        api = MagicMock(spec=ByocApi)
        api.download_cluster_install_package.return_value = b"pkg"

        with (
            patch(
                "agilerl.arena.byoc.cluster_register.extract_cluster_install_package",
                return_value=tmp_path,
            ),
            patch(
                "agilerl.arena.byoc.cluster_register.install_from_install_package_root"
            ) as install_mock,
        ):
            _install_charts_from_arena_package(
                api,
                cluster_name="prod-cluster",
                agent_values={"clusterName": "prod"},
                storage_values={"bucket": "arena-data"},
                include_storage=True,
                helm_wait=True,
                agent_namespace="arena",
                agent_release="arena-byoc-agent",
                install_agent=True,
            )

        assert api.download_cluster_install_package.call_args.kwargs[
            "storage_helm_values_yaml"
        ]
        install_mock.assert_called_once()


class TestRunClusterRegisterFailures:
    def test_missing_agent_helm_values(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(
                ByocApi,
                "register_cluster",
                return_value={"token": "t", "upserted": False},
            ),
        ):
            with pytest.raises(click.ClickException, match="missing agentHelmValues"):
                run_cluster_register(
                    mock_client,
                    name="prod-cluster",
                    output_dir=tmp_path,
                    skip_enable=True,
                    force=False,
                )

    def test_install_storage_without_storage_values(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(
                ByocApi,
                "register_cluster",
                return_value={
                    "token": "t",
                    "upserted": False,
                    "agentHelmValues": {"clusterName": "prod"},
                },
            ),
        ):
            with pytest.raises(click.ClickException, match="storageHelmValues"):
                run_cluster_register(
                    mock_client,
                    name="prod-cluster",
                    output_dir=tmp_path,
                    skip_enable=True,
                    force=False,
                    install_storage=True,
                )
