# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``arena cluster register`` and registration helpers."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import httpx
import pytest
import yaml
from click.testing import CliRunner

from agilerl.byoc import ByocApi, build_cluster_register_command
from agilerl.byoc.cluster_helm import AGENT_RELEASE
from agilerl.byoc.cluster_register import (
    ClassAvailability,
    ClusterResourceClass,
    _agent_values_for_helm,
    _align_same_memory,
    _bundle_helm_values_yaml,
    _compute_resource,
    _iec_byte_label,
    _install_charts_from_arena_package,
    _parse_byte_size,
    _registered_cluster_name,
    _registered_scope,
    _same_byte_size,
    ensure_resource_class,
    offer_suggested_classes,
    run_cluster_register,
)
from agilerl.byoc.provisioning.providers.aws.gateway import (
    WEBHOOK_CERT_DIRECTORY,
    WEBHOOK_HELM_VALUES,
)
from agilerl.byoc.provisioning.providers.base import EnsureGateway
from agilerl.byoc.provisioning.spec import ClusterSpec
from agilerl.byoc.provisioning.terraform import ClusterOutputs
from agilerl.client import ArenaClient
from agilerl.config import CommandConfig
from agilerl.exceptions import ArenaAPIError

from .conftest import FAKE_CLUSTER_UID

COMMANDS = "agilerl.byoc.commands"


@pytest.fixture(autouse=True)
def pinned_target(fake_kube_target: Callable[..., MagicMock]) -> MagicMock:
    """Point every register install at a fixed context."""
    return fake_kube_target(COMMANDS)


NO_SUGGESTIONS = {
    "reported_at": "2026-11-25T12:00:00Z",
    "suggestions": [],
    "classes_without_nodes": [],
    "needs_agent_upgrade": False,
}


@pytest.fixture(autouse=True)
def nothing_to_suggest() -> Iterator[MagicMock]:
    """Answer class suggestion calls as a cluster whose nodes all have classes."""
    with patch.object(
        ByocApi, "suggested_classes", return_value=NO_SUGGESTIONS
    ) as suggested:
        yield suggested


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


PULL_SECRET = '{"auths":{"registry.example.com":{"auth":"cHVsbC1zZWNyZXQ="}}}'

IMAGE_PULL_VALUES = {
    "mode": "api",
    "credential": {
        "dockerConfigJson": PULL_SECRET,
        "expiresAt": "2026-11-27T12:00:00Z",
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

    @pytest.mark.parametrize(
        ("rotate_token", "expected"), [(True, {"rotateToken": True}), (False, {})]
    )
    def test_register_cluster_sends_rotate_token_only_when_set(
        self,
        byoc_api: ByocApi,
        mock_client: MagicMock,
        rotate_token: bool,
        expected: dict[str, bool],
    ) -> None:
        mock_client._invoke_manifest_command.return_value = ENTERPRISE_BUNDLE

        byoc_api.register_cluster(name="prod-cluster", rotate_token=rotate_token)

        _, body = mock_client._invoke_manifest_command.call_args.args
        assert body == {"name": "prod-cluster", **expected}


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
        with patch("agilerl.byoc.cluster_helm.helm_available", return_value=False):
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
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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
            assert stat.S_IMODE(agent_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(storage_path.stat().st_mode) == 0o600

    def test_writes_image_pull_credential_unchanged_without_printing_it(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        bundle = {
            **ENTERPRISE_BUNDLE,
            "agentHelmValues": {
                **ENTERPRISE_BUNDLE["agentHelmValues"],
                "imagePull": IMAGE_PULL_VALUES,
            },
        }

        # Act
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=bundle),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
            )

        # Assert
        agent_path = tmp_path / "agent-helm-values.yaml"
        agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
        assert agent["imagePull"] == IMAGE_PULL_VALUES
        if sys.platform != "win32":
            assert stat.S_IMODE(agent_path.stat().st_mode) == 0o600
        captured = capsys.readouterr()
        assert PULL_SECRET not in captured.out + captured.err

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

    def test_passes_placement_to_register_api(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
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
                placement="ephemeral",
            )

        assert register_mock.call_args.kwargs["placement"] == "ephemeral"

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
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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

    def test_install_without_token_fails_when_the_cluster_has_no_token_secret(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Arrange
        upsert_bundle = {
            **ENTERPRISE_BUNDLE,
            "token": None,
            "upserted": True,
            "tokenRotated": False,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.5:8443", "id": 7},
                "clusterTokenSecret": {"name": "cluster-token", "key": "cluster-token"},
            },
        }

        # Act / Assert
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
            patch(
                "agilerl.byoc.cluster_register.secret_exists",
                return_value=False,
            ) as exists_mock,
            patch(
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
            pytest.raises(
                click.ClickException,
                match=r"has no token Secret 'cluster-token'.*--rotate-token",
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
        exists_mock.assert_called_once_with("cluster-token", "arena")
        install_mock.assert_not_called()

    def test_install_without_token_keeps_the_token_secret(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        upsert_bundle = {
            **ENTERPRISE_BUNDLE,
            "token": None,
            "upserted": True,
            "tokenRotated": False,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.5:8443", "id": 7},
                "existingClusterTokenSecret": "arena-cluster-token",
            },
        }

        # Act
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
            patch(
                "agilerl.byoc.cluster_register.secret_exists",
                return_value=True,
            ) as exists_mock,
            patch(
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
            ) as install_mock,
        ):
            run_cluster_register(
                mock_client,
                name="arena",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                no_write=True,
                byoc_scope="platform",
            )

        # Assert
        exists_mock.assert_called_once_with("arena-cluster-token", "arena")
        agent_values = install_mock.call_args.kwargs["agent_values"]
        assert agent_values["existingClusterTokenSecret"] == "arena-cluster-token"
        assert "clusterToken" not in agent_values
        out = capsys.readouterr().out
        assert "Kept the existing agent token. Pass --rotate-token" in out
        assert "Keeping the agent's cluster token Secret 'arena-cluster-token'" in out

    def test_install_with_rotated_token_on_upsert(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        upsert_bundle = {
            **ENTERPRISE_BUNDLE,
            "token": "rotated.token.secret",
            "upserted": True,
            "tokenRotated": True,
            "agentHelmValues": {
                "cluster": {"apiUrl": "http://172.24.0.5:8443", "id": 7},
                "clusterToken": "rotated.token.secret",
            },
        }
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=upsert_bundle),
            patch(
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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
        assert "Rotated the agent token." in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("token", "token_rotated", "message"),
        [
            (None, False, "Kept the existing agent token."),
            (
                "new.token",
                False,
                "Issued an agent token: the cluster had no active token.",
            ),
            ("new.token", True, "Rotated the agent token."),
            # An API without tokenRotated rotates on every re-register.
            ("new.token", None, "Rotated the agent token."),
        ],
    )
    def test_upsert_says_what_happened_to_the_token(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        token: str | None,
        token_rotated: bool | None,
        message: str,
    ) -> None:
        # Arrange
        upsert_bundle = {**ENTERPRISE_BUNDLE, "token": token, "upserted": True}
        if token_rotated is not None:
            upsert_bundle["tokenRotated"] = token_rotated

        # Act
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
            )

        # Assert
        assert message in capsys.readouterr().out

    def test_new_cluster_says_nothing_about_the_token(
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
            )

        out = capsys.readouterr().out
        assert "Registered cluster 'prod-cluster'." in out
        assert "agent token" not in out

    @pytest.mark.parametrize("rotate_token", [True, False])
    def test_passes_rotate_token_to_register_api(
        self, mock_client: MagicMock, tmp_path: Path, rotate_token: bool
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
                rotate_token=rotate_token,
            )

        assert register_mock.call_args.kwargs["rotate_token"] is rotate_token

    def test_install_passes_agent_namespace(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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

    def test_sends_the_cluster_uid_with_the_registration(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(
                ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE
            ) as register_mock,
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                cluster_uid=FAKE_CLUSTER_UID,
            )

        assert register_mock.call_args.kwargs["cluster_uid"] == FAKE_CLUSTER_UID

    def test_install_names_the_release_after_the_cluster(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch(
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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
                "agilerl.byoc.cluster_register.resolve_install_release",
                return_value=AGENT_RELEASE,
            ),
            patch(
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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
                "agilerl.byoc.cluster_register.install_lab_cluster_charts"
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
                "agilerl.byoc.cluster_register.install_enterprise_agent_chart"
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
                "agilerl.byoc.cluster_register.install_lab_cluster_charts"
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
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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
            cluster_name="lab-cluster",
            node_selector={"class": "gpu"},
            metadata={},
            description="",
            availabilities=None,
        )
        assert "Resource class 'arena-lab-gpu' is linked" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("byoc_scope", "sent"),
        [("platform", "platform"), ("customer", None), (None, None)],
    )
    def test_class_calls_follow_the_registered_scope(
        self,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
        tmp_path: Path,
        byoc_scope: str | None,
        sent: str | None,
    ) -> None:
        resource_class = ClusterResourceClass(
            name="arena-gpu",
            num_nodes=1,
            node_selector={"class": "gpu"},
            metadata={},
        )

        with (
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
        ):
            run_cluster_register(
                http_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                byoc_scope=byoc_scope,
                resource_classes=[resource_class],
            )

        assert sent_scope(http_client, "/classes/list")["query"] == sent
        assert sent_scope(http_client, "/classes/create")["body"] == sent

    def test_platform_register_403_names_the_staff_requirement(
        self, http_client: ArenaClient, tmp_path: Path
    ) -> None:
        http_client._http.request.side_effect = None
        http_client._http.request.return_value = httpx.Response(
            403, json={"error": "forbidden"}
        )

        with pytest.raises(
            ArenaAPIError, match="Only AgileRL staff can manage platform clusters"
        ):
            run_cluster_register(
                http_client,
                name="arena",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                byoc_scope="platform",
            )

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
                "agilerl.byoc.cluster_register._install_charts_from_arena_package"
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
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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
            cluster_name="prod-cluster",
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata=resource_class.metadata,
            description="",
            availabilities=None,
        )
        assert "Created resource class 'arena-nebius-gpu'" in capsys.readouterr().out

    def test_confirms_a_matching_class_on_the_cluster(
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
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "mk8snodegroup-test"},
                    "metadata": {
                        "computeResource": {
                            "numCpus": 16,
                            "numGpus": 1,
                            "memoryBytes": "200 GiB",
                        }
                    },
                },
            ),
            patch.object(ByocApi, "create_class") as create_mock,
            patch.object(
                ByocApi, "update_class", return_value={"name": "arena-nebius-gpu"}
            ) as update_mock,
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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
        update_mock.assert_called_once()
        assert update_mock.call_args.kwargs["cluster"] == "prod-cluster"
        assert "already matches" in capsys.readouterr().out

    def test_leaves_the_class_when_gib_display_is_the_same_size(
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
                    "memoryBytes": "64 GB",
                    "gramPerGpu": 48,
                }
            },
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch.object(
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "mk8snodegroup-test"},
                    "metadata": {
                        "computeResource": {
                            "numCpus": 16,
                            "numGpus": 1,
                            "memoryBytes": "59.6 GiB",
                            "gramPerGpu": 48,
                            "gpuMemoryBytes": "48 GB",
                        }
                    },
                },
            ),
            patch.object(ByocApi, "update_class") as update_mock,
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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

        update_mock.assert_called_once()
        sent = update_mock.call_args.kwargs
        assert sent["cluster"] == "prod-cluster"
        assert sent["metadata"]["computeResource"]["memoryBytes"] == "59.6 GiB"
        assert sent["metadata"]["computeResource"]["gpuMemoryBytes"] == "48 GB"
        assert "already matches" in capsys.readouterr().out

    def test_shows_a_real_memory_change(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        resource_class = ClusterResourceClass(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata={"computeResource": {"memoryBytes": "64 GB", "gramPerGpu": 48}},
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch.object(
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "mk8snodegroup-test"},
                    "metadata": {
                        "computeResource": {
                            "memoryBytes": "29.8 GiB",
                            "gramPerGpu": 48,
                        }
                    },
                },
            ),
            patch.object(ByocApi, "update_class") as update_mock,
            patch(
                "agilerl.byoc.cluster_register.click.confirm",
                return_value=False,
            ),
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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

        update_mock.assert_not_called()
        out = capsys.readouterr().out
        assert "29.8 GiB" in out
        assert "64 GB" in out

    def test_asks_before_updating_a_different_resource_class(
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
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "old"},
                    "metadata": {"computeResource": {"numCpus": 8}},
                },
            ),
            patch.object(ByocApi, "create_class") as create_mock,
            patch.object(ByocApi, "update_class") as update_mock,
            patch(
                "agilerl.byoc.cluster_register.click.confirm",
                return_value=False,
            ),
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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
        update_mock.assert_not_called()
        out = capsys.readouterr().out
        assert "already exists" in out
        assert "numCpus: 8" in out
        assert "numCpus: 16" in out
        assert "unchanged" in out

    def test_updates_a_different_resource_class_when_confirmed(
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
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "old"},
                    "metadata": resource_class.metadata,
                },
            ),
            patch.object(
                ByocApi, "update_class", return_value={"name": "arena-nebius-gpu"}
            ) as update_mock,
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                yes=True,
                resource_classes=[resource_class],
            )

        update_mock.assert_called_once_with(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata=resource_class.metadata,
            description="",
            availabilities=None,
            cluster="prod-cluster",
        )
        assert "Updated resource class 'arena-nebius-gpu'" in capsys.readouterr().out

    def test_update_keeps_the_stored_memory_label(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        resource_class = ClusterResourceClass(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata={
                "computeResource": {
                    "numCpus": 16,
                    "memoryBytes": "64 GB",
                    "gpuMemoryBytes": "48 GB",
                }
            },
            description="NVIDIA L40S",
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch.object(
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "old"},
                    "metadata": {
                        "computeResource": {
                            "numCpus": 8,
                            "memoryBytes": "59.6 GiB",
                            "gpuMemoryBytes": "44.7 GiB",
                        }
                    },
                },
            ),
            patch.object(
                ByocApi, "update_class", return_value={"name": "arena-nebius-gpu"}
            ) as update_mock,
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
        ):
            run_cluster_register(
                mock_client,
                name="prod-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                install=True,
                yes=True,
                resource_classes=[resource_class],
            )

        update_mock.assert_called_once_with(
            name="arena-nebius-gpu",
            num_nodes=1,
            node_selector={"nebius.com/node-group-id": "mk8snodegroup-test"},
            metadata={
                "computeResource": {
                    "numCpus": 16,
                    "memoryBytes": "59.6 GiB",
                    "gpuMemoryBytes": "44.7 GiB",
                }
            },
            description="NVIDIA L40S",
            availabilities=None,
            cluster="prod-cluster",
        )

    def test_shows_a_gpu_memory_change(
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
                    "memoryBytes": "64 GB",
                    "gpuMemoryBytes": "80 GB",
                }
            },
        )
        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=ENTERPRISE_BUNDLE),
            patch.object(
                ByocApi,
                "find_class",
                return_value={
                    "name": "arena-nebius-gpu",
                    "num_nodes": 1,
                    "node_selector": {"nebius.com/node-group-id": "mk8snodegroup-test"},
                    "metadata": {
                        "computeResource": {
                            "memoryBytes": "64 GB",
                            "gpuMemoryBytes": "48 GB",
                        }
                    },
                },
            ),
            patch.object(ByocApi, "update_class") as update_mock,
            patch(
                "agilerl.byoc.cluster_register.click.confirm",
                return_value=False,
            ),
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
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

        update_mock.assert_not_called()
        out = capsys.readouterr().out
        assert "48 GB" in out
        assert "80 GB" in out


class TestClusterRegisterCommand:
    @pytest.fixture(autouse=True)
    def _cluster_already_in_terraform_state(self):
        with (
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.cluster_exists",
                return_value=True,
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.ensure_post_apply",
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.status",
                return_value={"worker_node_group_ids": {"workers": "ng-1"}},
            ),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                return_value=None,
            ),
            patch(
                "agilerl.byoc.provisioning.providers.aws.provider.AwsProvider.ensure_gateway",
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
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
                        "platform": "gpu-l40s-a",
                        "preset": "1gpu-16vcpu-64gb",
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.status",
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
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
                        "platform": "gpu-l40s-a",
                        "preset": "1gpu-16vcpu-64gb",
                        "gpu_drivers_preset": "cuda13.0",
                    }
                ],
            },
        }

    def test_install_uses_cluster_kubeconfig_when_the_cluster_exists(
        self,
        pinned_target: MagicMock,
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

        def capture(*_args: object, **kwargs: object) -> None:
            seen["kubeconfig"] = os.environ.get("KUBECONFIG")
            seen["cluster_uid"] = kwargs.get("cluster_uid")

        client = MagicMock()
        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.byoc.commands.run_cluster_register",
                side_effect=capture,
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert pinned_target.call_args.args[1:] == (kubeconfig.resolve(), None)
        assert seen["kubeconfig"] == str(tmp_path / "pinned-kubeconfig")
        assert seen["cluster_uid"] == FAKE_CLUSTER_UID

    def test_register_finishes_cluster_setup_when_the_cluster_exists(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        client = MagicMock()

        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register"),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.ensure_post_apply"
            ) as finish,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        finish.assert_called_once()
        assert finish.call_args.kwargs["state_dir"].name == "terraform"
        assert (
            finish.call_args.kwargs["output_dir"]
            == (tmp_path / "arena-cluster-spec-cluster").resolve()
        )

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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                return_value=refs,
            ) as gateway,
            patch("agilerl.byoc.commands.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        gateway.assert_called_once()
        assert gateway.call_args.args[0].kubeconfig_path == (
            tmp_path / "pinned-kubeconfig"
        )
        assert register.call_args.kwargs["gateway_api_parent_refs"] == refs

    def test_install_configures_the_gateway_on_the_pinned_context(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Gateway setup uses the pinned kubeconfig while that context is active."""
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        spec_kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        spec_kubeconfig.parent.mkdir()
        spec_kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        other_kubeconfig = tmp_path / "other-kubeconfig"
        other_kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        seen: dict[str, object] = {}

        def capture_gateway(query: EnsureGateway) -> list[dict[str, str]]:
            seen["kubeconfig"] = query.kubeconfig_path
            seen["env"] = os.environ.get("KUBECONFIG")
            return [{"name": "arena"}]

        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                side_effect=capture_gateway,
            ),
            patch("agilerl.byoc.commands.run_cluster_register"),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--spec",
                    str(spec_path),
                    "--install",
                    "--skip-enable",
                    "--kubeconfig",
                    str(other_kubeconfig),
                    "--context",
                    "other",
                ],
                obj=command_config,
            )

        pinned = tmp_path / "pinned-kubeconfig"
        assert result.exit_code == 0, result.output
        pinned_target.assert_called_once_with(
            "spec-cluster", other_kubeconfig.resolve(), "other"
        )
        assert seen["kubeconfig"] == pinned
        assert seen["env"] == str(pinned)

    @pytest.mark.parametrize(
        ("registered", "proceeds"),
        [
            (None, True),
            ({"name": "spec-cluster", "clusterUid": None}, True),
            ({"name": "spec-cluster", "clusterUid": FAKE_CLUSTER_UID}, True),
            ({"name": "spec-cluster", "clusterUid": "another-uid"}, False),
        ],
    )
    def test_install_checks_the_registered_uid_before_the_gateway(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        registered: dict[str, object] | None,
        proceeds: bool,
    ) -> None:
        """A context Arena knows as another cluster stops before gateway setup."""
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        spec_kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        spec_kubeconfig.parent.mkdir()
        spec_kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

        with (
            patch(
                f"{COMMANDS}.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(f"{COMMANDS}.ByocApi.find_cluster", return_value=registered),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                return_value=[{"name": "arena"}],
            ) as gateway,
            patch(f"{COMMANDS}.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert (result.exit_code == 0) is proceeds, result.output
        assert gateway.called is proceeds
        assert register.called is proceeds
        if not proceeds:
            assert "another-uid" in result.output

    def test_install_registers_an_org_that_has_not_enabled_byoc(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        http_client: ArenaClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Arena lists no clusters (404) until register enables the provider."""
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        spec_kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        spec_kubeconfig.parent.mkdir()
        spec_kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        calls: list[str] = []

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            calls.append(path)
            assert path in {"/api/byoc-clusters", "/api/cli/v1/byoc/clusters/list"}
            return httpx.Response(404, json={"error": "Not Found"})

        http_client._http.request.side_effect = respond

        with (
            patch(
                f"{COMMANDS}.arena_client",
                return_value=client_context(http_client),
            ),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
                return_value=[{"name": "arena"}],
            ) as gateway,
            patch(
                f"{COMMANDS}.run_cluster_register",
                side_effect=lambda *_a, **_kw: calls.append("register"),
            ) as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert calls == [
            "/api/byoc-clusters",
            "/api/cli/v1/byoc/clusters/list",
            "register",
        ]
        gateway.assert_called_once()
        assert register.call_args.kwargs["skip_enable"] is False
        assert register.call_args.kwargs["cluster_uid"] == FAKE_CLUSTER_UID

    def test_install_stops_when_the_cluster_lookup_fails(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        spec_kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        spec_kubeconfig.parent.mkdir()
        spec_kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

        with (
            patch(
                f"{COMMANDS}.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(
                f"{COMMANDS}.ByocApi.find_cluster",
                side_effect=ArenaAPIError("boom", status_code=500),
            ),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
            ) as gateway,
            patch(f"{COMMANDS}.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert isinstance(result.exception, ArenaAPIError)
        gateway.assert_not_called()
        register.assert_not_called()

    def test_install_skips_the_gateway_when_the_spec_disables_it(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(
            NEBIUS_SPEC.replace(
                "  inference:\n    domain: inference.example.com\n",
                "  inference:\n    domain: inference.example.com\n"
                "  gateway:\n    enable: false\n",
            ),
            encoding="utf-8",
        )
        kubeconfig = tmp_path / "arena-cluster-spec-cluster" / "kubeconfig"
        kubeconfig.parent.mkdir()
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")

        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(
                "agilerl.byoc.provisioning.providers.nebius.provider.NebiusProvider.ensure_gateway",
            ) as gateway,
            patch("agilerl.byoc.commands.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec_path), "--install", "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        gateway.assert_not_called()
        assert register.call_args.kwargs["gateway_api_parent_refs"] is None

    def test_install_fetches_cluster_kubeconfig_when_the_file_is_missing(
        self,
        pinned_target: MagicMock,
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

        def capture(*_args: object, **kwargs: object) -> None:
            seen["kubeconfig"] = os.environ.get("KUBECONFIG")
            seen["cluster_uid"] = kwargs.get("cluster_uid")

        client = MagicMock()
        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch(
                "agilerl.byoc.commands.run_cluster_register",
                side_effect=capture,
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.write_kubeconfig",
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
        assert pinned_target.call_args.args[1:] == (fetched, None)

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
            catalog_access_key_id="catalog-key",
            catalog_secret_access_key="catalog-secret",
            catalog_group_id="group-catalog",
        )
        client = MagicMock()
        with (
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.cluster_exists",
                return_value=False,
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.provision",
                return_value=outputs,
            ) as provision_mock,
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec_path), "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert provision_mock.call_args.kwargs["create_state_bucket"] is True
        config = run_mock.call_args.kwargs["byoc_provider"]["config"]
        assert config["project_id"] == "project-created"
        assert config["catalog_access_key_id"] == "catalog-key"
        assert config["catalog_secret_access_key"] == "catalog-secret"
        assert config["catalog_group_id"] == "group-catalog"
        assert run_mock.call_args.kwargs["resource_classes"][0].name == (
            "spec-cluster-workers"
        )
        assert run_mock.call_args.kwargs["rotate_token"] is True

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
                "agilerl.byoc.commands.ClusterProvisioner.cluster_exists",
                exists,
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.provision",
            ) as provision_mock,
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
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

    def test_sends_terraform_project_ids_when_arena_stored_null(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(NEBIUS_SPEC, encoding="utf-8")
        client = MagicMock()
        client._request.return_value = [
            {
                "name": "spec-cluster",
                "byoc_provider": {
                    "provider": "nebius",
                    "config": {
                        "tenant_id": "tenant",
                        "region": "eu-north1",
                        "project_id": None,
                        "storage_project_id": None,
                    },
                },
            }
        ]
        with (
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.status",
                return_value={
                    "project_id": "project-created",
                    "storage_project_id": "storage-created",
                    "worker_node_group_ids": {"workers": "ng-1"},
                    "catalog_access_key_id": "catalog-key",
                    "catalog_secret_access_key": "catalog-secret",
                    "catalog_group_id": "group-catalog",
                },
            ),
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec_path), "--skip-enable"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        config = run_mock.call_args.kwargs["byoc_provider"]["config"]
        assert config["project_id"] == "project-created"
        assert config["storage_project_id"] == "storage-created"
        assert config["catalog_access_key_id"] == "catalog-key"
        assert config["catalog_secret_access_key"] == "catalog-secret"
        assert config["catalog_group_id"] == "group-catalog"

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
                "agilerl.byoc.commands.ClusterProvisioner.cluster_exists",
                return_value=False,
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.provision",
            ) as provision_mock,
            patch(
                "agilerl.byoc.commands.arena_client",
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.write_kubeconfig",
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
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
        assert kwargs["rotate_token"] is False

    def test_cli_passes_rotate_token(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "prod-cluster", "--install", "--rotate-token"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert run_mock.call_args.kwargs["rotate_token"] is True

    def test_cli_passes_agent_namespace(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        client = MagicMock()
        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
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
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch("agilerl.byoc.commands.run_cluster_register") as run_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "prod-cluster", "--skip-enable"],
                obj=command_config,
            )
        assert result.exit_code == 0, result.output
        run_mock.assert_called_once()


class TestAwsRegisterInstallWebhookCerts:
    def test_second_install_reuses_the_webhook_ca_beside_the_durable_kubeconfig(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second AWS register --install reuses the CA next to the output kubeconfig."""
        monkeypatch.chdir(tmp_path)
        spec_path = tmp_path / "cluster.yaml"
        spec_path.write_text(
            """
provider: aws
name: eks-byoc
terraform_state:
  bucket: eks-tfstate
arena:
  storage:
    bucket: eks-data
    endpoint: https://s3.eu-west-1.amazonaws.com
  inference:
    domain: inference.example.com
aws:
  region: eu-west-1
  object_storage:
    size_gib: 1
""",
            encoding="utf-8",
        )
        output_dir = tmp_path / "arena-cluster-eks-byoc"
        kubeconfig = output_dir / "kubeconfig"
        kubeconfig.parent.mkdir()
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        pinned = tmp_path / "pinned-kubeconfig"
        cert_dir = output_dir / WEBHOOK_CERT_DIRECTORY
        helm_kubeconfigs: list[str] = []
        helm_values: list[str] = []
        gateway_kubeconfigs: list[Path] = []

        def record_helm(
            argv: list[str], env: dict[str, str], **_kwargs: object
        ) -> None:
            helm_kubeconfigs.append(env["KUBECONFIG"])
            helm_values.append(argv[argv.index("-f") + 1])

        def record_gateway(**kwargs: object) -> list[dict[str, str]]:
            kubeconfig_path = kwargs["kubeconfig_path"]
            assert isinstance(kubeconfig_path, Path)
            gateway_kubeconfigs.append(kubeconfig_path)
            return [{"name": "arena"}]

        def invoke() -> None:
            with (
                patch(
                    "agilerl.byoc.commands.arena_client",
                    return_value=client_context(MagicMock()),
                ),
                patch(
                    "agilerl.byoc.commands.ClusterProvisioner.cluster_exists",
                    return_value=True,
                ),
                patch(
                    "agilerl.byoc.commands.ClusterProvisioner.ensure_post_apply",
                ),
                patch(
                    "agilerl.byoc.commands.ClusterProvisioner.status",
                    return_value={
                        "worker_node_group_ids": {},
                        "vpc_id": "vpc-123",
                    },
                ),
                patch(
                    "agilerl.byoc.provisioning.providers.aws.gateway.helm_available",
                    return_value=True,
                ),
                patch(
                    "agilerl.byoc.provisioning.providers.aws.gateway._run",
                    side_effect=record_helm,
                ),
                patch(
                    "agilerl.byoc.provisioning.providers.aws.gateway.configure_gateway_api",
                    side_effect=record_gateway,
                ),
                patch("agilerl.byoc.commands.run_cluster_register"),
            ):
                result = CliRunner().invoke(
                    build_cluster_register_command(),
                    ["--spec", str(spec_path), "--install", "--skip-enable"],
                    obj=command_config,
                )

            assert result.exit_code == 0, result.output

        invoke()
        ca = (cert_dir / "ca.crt").read_bytes()

        invoke()

        values_path = str(cert_dir / WEBHOOK_HELM_VALUES)
        assert (cert_dir / "ca.crt").read_bytes() == ca
        assert (cert_dir / "tls.crt").is_file()
        assert (cert_dir / "tls.key").is_file()
        assert not (tmp_path / WEBHOOK_CERT_DIRECTORY).exists()
        assert helm_kubeconfigs == [str(pinned.resolve()), str(pinned.resolve())]
        assert helm_values == [values_path, values_path]
        assert gateway_kubeconfigs == [pinned, pinned]


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


class TestRegisteredScope:
    def test_missing_cluster_has_no_scope(self) -> None:
        assert _registered_scope({}) is None
        assert _registered_scope({"cluster": None}) is None

    def test_blank_scope_is_unset(self) -> None:
        assert _registered_scope({"cluster": {"byocScope": "  "}}) is None
        assert _registered_scope({"cluster": {"byoc_scope": "\t"}}) is None

    def test_prints_the_registered_scope(
        self,
        mock_client: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bundle = {
            **LAB_BUNDLE,
            "cluster": {**LAB_BUNDLE["cluster"], "byocScope": "platform"},
        }

        with (
            patch.object(ByocApi, "enable"),
            patch.object(ByocApi, "register_cluster", return_value=bundle),
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
            )

        assert "Scope: platform" in capsys.readouterr().out


class TestInstallChartsFromArenaPackage:
    def test_downloads_and_installs_package(self, tmp_path: Path) -> None:
        api = MagicMock(spec=ByocApi)
        api.download_cluster_install_package.return_value = b"pkg"

        with (
            patch(
                "agilerl.byoc.cluster_register.extract_cluster_install_package",
                return_value=tmp_path,
            ),
            patch(
                "agilerl.byoc.cluster_register.install_from_install_package_root"
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

    def test_sends_image_pull_credential_unchanged(self, tmp_path: Path) -> None:
        api = MagicMock(spec=ByocApi)
        api.download_cluster_install_package.return_value = b"pkg"

        with (
            patch(
                "agilerl.byoc.cluster_register.extract_cluster_install_package",
                return_value=tmp_path,
            ),
            patch("agilerl.byoc.cluster_register.install_from_install_package_root"),
        ):
            _install_charts_from_arena_package(
                api,
                cluster_name="prod-cluster",
                agent_values={"clusterName": "prod", "imagePull": IMAGE_PULL_VALUES},
                storage_values=None,
                include_storage=False,
                helm_wait=True,
                agent_namespace="arena",
                agent_release="arena-byoc-agent",
                install_agent=True,
            )

        agent_yaml = api.download_cluster_install_package.call_args.kwargs[
            "agent_helm_values_yaml"
        ]
        assert yaml.safe_load(agent_yaml)["imagePull"] == IMAGE_PULL_VALUES


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


class TestByteSizeHelpers:
    def test_parse_rejects_unknown_text(self) -> None:
        assert _parse_byte_size("not-a-size") is None
        assert _parse_byte_size("12 widgets") is None

    def test_iec_label_keeps_bytes_under_one_kib(self) -> None:
        assert _iec_byte_label(512) == "512 B"

    def test_same_byte_size_falls_back_to_string_equality(self) -> None:
        assert _same_byte_size("unknown", "unknown") is True
        assert _same_byte_size("unknown", "other") is False

    def test_compute_resource_requires_metadata_mapping(self) -> None:
        assert _compute_resource({"metadata": "nope"}) is None

    def test_align_same_memory_keeps_desired_without_compute(self) -> None:
        desired = {"num_nodes": 1, "metadata": {}}
        assert _align_same_memory({"metadata": {}}, desired) == desired


class TestEnsureResourceClassDescription:
    def test_keeps_the_stored_description_when_updating(self) -> None:
        api = MagicMock()
        api.find_class.return_value = {
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
            "description": "stored",
        }
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
            description="stored",
        )

        with patch("agilerl.byoc.cluster_register.click.echo"):
            name = ensure_resource_class(api, "cluster", resource, yes=True)

        assert name == "gpu"
        api.update_class.assert_called_once()
        assert api.update_class.call_args.kwargs["cluster"] == "cluster"
        assert api.update_class.call_args.kwargs["description"] == "stored"
        api.create_class.assert_not_called()

    def test_replaces_availabilities_when_the_class_matches(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api = MagicMock()
        api.find_class.return_value = {
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
            "description": "",
        }
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
            availabilities=(
                ClassAvailability(
                    region="eu-north1",
                    zones=("fabric-2",),
                    enabled=True,
                    available=4,
                ),
                ClassAvailability(
                    region="us-central1",
                    zones=(),
                    enabled=False,
                    available=0,
                ),
            ),
        )

        name = ensure_resource_class(api, "cluster", resource, yes=True)

        assert name == "gpu"
        api.replace_class_availabilities.assert_called_once()
        assert (
            "Updated availability for 'gpu': eu-north1 (4)." in capsys.readouterr().out
        )

    def test_reports_when_no_region_has_capacity(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api = MagicMock()
        api.find_class.return_value = None
        api.create_class.return_value = {"name": "gpu"}
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
            availabilities=(),
            enabled=False,
        )

        ensure_resource_class(api, "cluster", resource, yes=True)

        assert (
            "Disabled 'gpu'. The preset is not offered in any Nebius region."
            in capsys.readouterr().out
        )
        assert api.create_class.call_args.kwargs["enabled"] is False

    def test_disables_a_matching_class_when_no_region_offers_the_preset(
        self,
    ) -> None:
        api = MagicMock()
        api.find_class.return_value = {
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
        }
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
            availabilities=(),
            enabled=False,
        )

        ensure_resource_class(api, "cluster", resource, yes=True)

        api.replace_class_availabilities.assert_called_once_with(
            "gpu", [], enabled=False, cluster="cluster"
        )

    def test_ignores_a_non_string_stored_description(self) -> None:
        api = MagicMock()
        api.find_class.return_value = {
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
            "description": 12,
        }
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=2,
            node_selector={"pool": "a"},
            metadata={},
            description="desired",
        )
        api.update_class.return_value = {"name": "gpu"}

        with patch("agilerl.byoc.cluster_register.click.echo"):
            name = ensure_resource_class(api, "cluster", resource, yes=True)

        assert name == "gpu"
        api.update_class.assert_called_once()
        assert api.update_class.call_args.kwargs["description"] == "desired"

    def test_reads_camel_case_stored_class_fields(self) -> None:
        api = MagicMock()
        api.find_class.return_value = {
            "numNodes": 1,
            "nodeSelector": {"pool": "a"},
            "metadata": {},
        }
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
        )

        with patch("agilerl.byoc.cluster_register.click.echo"):
            name = ensure_resource_class(api, "cluster", resource, yes=True)

        assert name == "gpu"
        api.update_class.assert_called_once()
        assert api.update_class.call_args.kwargs["cluster"] == "cluster"
        api.create_class.assert_not_called()

    def test_echoes_availability_after_updating(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        api = MagicMock()
        api.find_class.return_value = {
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
        }
        api.update_class.return_value = {"name": "gpu"}
        resource = ClusterResourceClass(
            name="gpu",
            num_nodes=2,
            node_selector={"pool": "a"},
            metadata={},
            availabilities=(
                ClassAvailability(
                    region="eu-north1",
                    zones=("fabric-2",),
                    enabled=True,
                    available=4,
                ),
            ),
        )

        name = ensure_resource_class(api, "cluster", resource, yes=True)

        assert name == "gpu"
        api.update_class.assert_called_once()
        assert api.update_class.call_args.kwargs["availabilities"] == [
            {"region": "eu-north1", "zones": ["fabric-2"], "enabled": True}
        ]
        assert (
            "Updated availability for 'gpu': eu-north1 (4)." in capsys.readouterr().out
        )


class TestRegisterKubeTarget:
    @pytest.mark.parametrize(
        ("flags", "expected"),
        [
            ([], (None, None)),
            (["--context", "corp"], (None, "corp")),
        ],
    )
    def test_install_resolves_the_named_context(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        flags: list[str],
        expected: tuple[object, object],
    ) -> None:
        with (
            patch(
                f"{COMMANDS}.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(f"{COMMANDS}.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "corp-k8s", "--install", "--no-write", *flags],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        pinned_target.assert_called_once_with("corp-k8s", *expected)
        assert register.call_args.kwargs["cluster_uid"] == FAKE_CLUSTER_UID

    def test_register_without_install_never_touches_the_cluster(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        with (
            patch(
                f"{COMMANDS}.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(f"{COMMANDS}.run_cluster_register") as register,
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "corp-k8s"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        pinned_target.assert_not_called()
        assert register.call_args.kwargs["cluster_uid"] is None

    def test_rotate_token_install_sends_the_uid_of_the_named_context(
        self,
        pinned_target: MagicMock,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        tmp_path: Path,
    ) -> None:
        seen: dict[str, object] = {}

        def register(_client: object, **kwargs: object) -> None:
            # The kept-token Secret check runs kubectl inside run_cluster_register.
            seen["KUBECONFIG"] = os.environ.get("KUBECONFIG")
            seen.update(kwargs)

        with (
            patch(
                f"{COMMANDS}.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch(f"{COMMANDS}.run_cluster_register", side_effect=register),
        ):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--name",
                    "corp-k8s",
                    "--install",
                    "--no-write",
                    "--rotate-token",
                    "--context",
                    "corp",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        pinned_target.assert_called_once_with("corp-k8s", None, "corp")
        assert seen["rotate_token"] is True
        assert seen["cluster_uid"] == FAKE_CLUSTER_UID
        assert seen["KUBECONFIG"] == str(tmp_path / "pinned-kubeconfig")


class TestEnsureResourceClassCluster:
    def test_update_names_the_class_cluster(self, http_client: ArenaClient) -> None:
        # Arrange
        resource = ClusterResourceClass(
            name="pool", num_nodes=2, node_selector={"pool": "a"}, metadata={}
        )

        # Act
        ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        # Assert
        update = http_client._http.request.call_args
        assert update.args[1].endswith("/classes/update")
        assert update.kwargs["params"] == {"name": "pool", "cluster": "arena"}

    def test_availability_update_names_the_class_cluster(
        self, http_client: ArenaClient
    ) -> None:
        # Arrange
        stored = {
            "name": "pool",
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
        }
        http_client._http.request.side_effect = lambda method, path, **_kw: (
            httpx.Response(200, json=[stored] if path.endswith("/list") else stored)
        )
        resource = ClusterResourceClass(
            name="pool",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
            availabilities=(),
        )

        # Act
        ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        # Assert
        update = http_client._http.request.call_args
        assert update.args[1].endswith("/classes/update")
        assert update.kwargs["params"] == {"name": "pool", "cluster": "arena"}
        assert update.kwargs["json"] == {"availabilities": []}

    def test_creates_the_class_when_the_update_is_not_on_this_cluster(
        self, http_client: ArenaClient
    ) -> None:
        # Arrange
        stored = {
            "name": "pool",
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
        }

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=[stored])
            if path.endswith("/classes/update"):
                return httpx.Response(404, json={"error": "Resource class not found"})
            if path.endswith("/classes/create"):
                return httpx.Response(200, json={"name": "pool"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool",
            num_nodes=2,
            node_selector={"pool": "b"},
            metadata={},
        )

        # Act
        name = ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        # Assert
        assert name == "pool"
        calls = http_client._http.request.call_args_list
        update = next(call for call in calls if call.args[1].endswith("/update"))
        create = next(call for call in calls if call.args[1].endswith("/create"))
        assert update.kwargs["params"] == {"name": "pool", "cluster": "arena"}
        assert create.kwargs["json"]["clusterName"] == "arena"

    def test_creates_when_a_matching_class_is_not_on_this_cluster(
        self, http_client: ArenaClient
    ) -> None:
        stored = {
            "name": "pool",
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
        }

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=[stored])
            if path.endswith("/classes/update"):
                return httpx.Response(404, json={"error": "Resource class not found"})
            if path.endswith("/classes/create"):
                return httpx.Response(200, json={"name": "pool"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
        )

        name = ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        assert name == "pool"
        calls = http_client._http.request.call_args_list
        assert any(call.args[1].endswith("/update") for call in calls)
        create = next(call for call in calls if call.args[1].endswith("/create"))
        assert create.kwargs["json"]["clusterName"] == "arena"

    def test_updates_when_the_class_name_is_shared(
        self, http_client: ArenaClient
    ) -> None:
        rows = [{"name": "pool", "id": 1}, {"name": "pool", "id": 2}]

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=rows)
            if path.endswith("/classes/update"):
                return httpx.Response(200, json={"name": "pool"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool", num_nodes=1, node_selector={"pool": "a"}, metadata={}
        )

        name = ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        assert name == "pool"
        update = http_client._http.request.call_args
        assert update.args[1].endswith("/classes/update")
        assert update.kwargs["params"] == {"name": "pool", "cluster": "arena"}

    def test_shared_name_asks_before_updating_this_cluster(
        self, http_client: ArenaClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange
        rows = [{"name": "pool", "id": 1}, {"name": "pool", "id": 2}]
        http_client._http.request.side_effect = lambda _method, _path, **_kw: (
            httpx.Response(200, json=rows)
        )
        resource = ClusterResourceClass(
            name="pool", num_nodes=1, node_selector={"pool": "a"}, metadata={}
        )

        # Act
        with patch(
            "agilerl.byoc.cluster_register.click.confirm", return_value=False
        ) as confirm:
            name = ensure_resource_class(ByocApi(http_client), "arena", resource)

        # Assert
        assert name == "pool"
        assert confirm.call_args.args[0] == (
            "Update resource class 'pool' on cluster 'arena'?"
        )
        sent = [call.args[1] for call in http_client._http.request.call_args_list]
        assert all(path.endswith("/classes/list") for path in sent)
        assert "Left resource class 'pool' unchanged." in capsys.readouterr().out

    def test_creates_when_a_shared_name_is_missing_on_this_cluster(
        self, http_client: ArenaClient
    ) -> None:
        rows = [{"name": "pool", "id": 1}, {"name": "pool", "id": 2}]

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=rows)
            if path.endswith("/classes/update"):
                return httpx.Response(404, json={"error": "Resource class not found"})
            if path.endswith("/classes/create"):
                return httpx.Response(200, json={"name": "pool"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool", num_nodes=1, node_selector={"pool": "a"}, metadata={}
        )

        name = ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        assert name == "pool"
        calls = http_client._http.request.call_args_list
        update = next(call for call in calls if call.args[1].endswith("/update"))
        create = next(call for call in calls if call.args[1].endswith("/create"))
        assert update.kwargs["params"] == {"name": "pool", "cluster": "arena"}
        assert create.kwargs["json"]["clusterName"] == "arena"

    def test_find_class_error_is_not_a_shared_name(self) -> None:
        api = MagicMock()
        api.find_class.side_effect = ArenaAPIError("boom", status_code=500)
        resource = ClusterResourceClass(
            name="pool", num_nodes=1, node_selector={}, metadata={}
        )

        with pytest.raises(ArenaAPIError, match="boom"):
            ensure_resource_class(api, "arena", resource, yes=True)

        api.update_class.assert_not_called()
        api.create_class.assert_not_called()

    def test_shared_name_update_error_is_not_a_missing_class(
        self, http_client: ArenaClient
    ) -> None:
        rows = [{"name": "pool", "id": 1}, {"name": "pool", "id": 2}]

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=rows)
            if path.endswith("/classes/update"):
                return httpx.Response(400, json={"error": "unknown Nebius platform"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool", num_nodes=1, node_selector={"pool": "a"}, metadata={}
        )

        with pytest.raises(ArenaAPIError, match="unknown Nebius platform"):
            ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        assert not any(
            call.args[1].endswith("/create")
            for call in http_client._http.request.call_args_list
        )

    def test_echoes_availability_when_a_shared_name_updates(
        self, http_client: ArenaClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rows = [{"name": "pool", "id": 1}, {"name": "pool", "id": 2}]

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=rows)
            if path.endswith("/classes/update"):
                return httpx.Response(200, json={"name": "pool"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool",
            num_nodes=1,
            node_selector={"pool": "a"},
            metadata={},
            availabilities=(
                ClassAvailability(
                    region="eu-north1",
                    zones=("fabric-2",),
                    enabled=True,
                    available=4,
                ),
            ),
        )

        name = ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        assert name == "pool"
        assert (
            "Updated availability for 'pool': eu-north1 (4)." in capsys.readouterr().out
        )

    def test_update_error_is_not_a_missing_class(
        self, http_client: ArenaClient
    ) -> None:
        stored = {
            "name": "pool",
            "num_nodes": 1,
            "node_selector": {"pool": "a"},
            "metadata": {},
        }

        def respond(_method: str, path: str, **_kwargs: object) -> httpx.Response:
            if path.endswith("/classes/list"):
                return httpx.Response(200, json=[stored])
            if path.endswith("/classes/update"):
                return httpx.Response(400, json={"error": "unknown Nebius platform"})
            raise AssertionError(path)

        http_client._http.request.side_effect = respond
        resource = ClusterResourceClass(
            name="pool",
            num_nodes=2,
            node_selector={"pool": "b"},
            metadata={},
        )

        with pytest.raises(ArenaAPIError, match="unknown Nebius platform"):
            ensure_resource_class(ByocApi(http_client), "arena", resource, yes=True)

        assert not any(
            call.args[1].endswith("/create")
            for call in http_client._http.request.call_args_list
        )


H100_SUGGESTION = {
    "name": "lab-cluster-h100-80-x8",
    "enabled": True,
    "description": "2 node(s) with 8x NVIDIA-H100-80GB-HBM3",
    "num_nodes": 2,
    "on_prem_cluster_id": 42,
    "on_prem_cluster": None,
    "node_selector": {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},
    "placement": None,
    "availabilities": None,
    "metadata": {
        "computeResource": {
            "numCpus": 128,
            "numGpus": 8,
            "memoryBytes": "2.0 TiB",
            "gpu": {"type": "nvidia-h100-80gb", "count": 8, "driverVersion": "default"},
        }
    },
}


def created_class_names(mock_client: MagicMock) -> list[str]:
    """Names of the classes posted to ``classes/create`` through *mock_client*."""
    return [
        call.args[1]["name"]
        for call in mock_client._invoke_manifest_command.call_args_list
        if call.args[0]["path"].endswith("/classes/create")
    ]


class TestOfferSuggestedClasses:
    def test_yes_creates_every_suggestion(
        self,
        byoc_api: ByocApi,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        nothing_to_suggest.return_value = {
            **NO_SUGGESTIONS,
            "suggestions": [H100_SUGGESTION],
        }

        # Act
        created = offer_suggested_classes(byoc_api, "lab-cluster", yes=True)

        # Assert
        assert created == ["lab-cluster-h100-80-x8"]
        assert created_class_names(mock_client) == ["lab-cluster-h100-80-x8"]
        body = mock_client._invoke_manifest_command.call_args.args[1]
        assert "placement" not in body
        out = capsys.readouterr().out
        assert (
            "lab-cluster-h100-80-x8: 2 node(s), 8x nvidia-h100-80gb, 128 CPUs, "
            "2.0 TiB memory, node selector nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3"
        ) in out
        assert "Created resource class 'lab-cluster-h100-80-x8'." in out
        nothing_to_suggest.assert_called_once_with("lab-cluster")

    @pytest.mark.parametrize(
        ("answer", "expected"), [(True, ["lab-cluster-h100-80-x8"]), (False, [])]
    )
    def test_a_terminal_asks_before_creating(
        self,
        byoc_api: ByocApi,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        answer: bool,
        expected: list[str],
    ) -> None:
        nothing_to_suggest.return_value = {
            **NO_SUGGESTIONS,
            "suggestions": [H100_SUGGESTION],
        }
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        with patch(
            "agilerl.byoc.cluster_register.click.confirm", return_value=answer
        ) as confirm:
            created = offer_suggested_classes(byoc_api, "lab-cluster", yes=False)

        confirm.assert_called_once_with("Create these classes?", default=False)
        assert created == expected
        assert created_class_names(mock_client) == expected

    def test_without_a_terminal_lists_them_and_creates_nothing(
        self,
        byoc_api: ByocApi,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nothing_to_suggest.return_value = {
            **NO_SUGGESTIONS,
            "suggestions": [H100_SUGGESTION],
        }
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        created = offer_suggested_classes(byoc_api, "lab-cluster", yes=False)

        assert created == []
        assert created_class_names(mock_client) == []
        out = capsys.readouterr().out
        assert "lab-cluster-h100-80-x8: 2 node(s)" in out
        assert "Re-run with --yes to create them" in out

    def test_a_refused_create_does_not_stop_the_others(
        self,
        byoc_api: ByocApi,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        b200 = {**H100_SUGGESTION, "name": "lab-cluster-b200-x8"}
        nothing_to_suggest.return_value = {
            **NO_SUGGESTIONS,
            "suggestions": [H100_SUGGESTION, b200],
        }
        mock_client._invoke_manifest_command.side_effect = [
            ArenaAPIError("name conflicts with a cloud tier", status_code=400),
            {"id": 2},
        ]

        # Act
        created = offer_suggested_classes(byoc_api, "lab-cluster", yes=True)

        # Assert
        assert created == ["lab-cluster-b200-x8"]
        assert created_class_names(mock_client) == [
            "lab-cluster-h100-80-x8",
            "lab-cluster-b200-x8",
        ]
        err = capsys.readouterr().err
        assert (
            "Could not create resource class 'lab-cluster-h100-80-x8': "
            "API error (400): name conflicts with a cloud tier"
        ) in err

    def test_flags_classes_without_nodes_and_an_old_agent(
        self,
        byoc_api: ByocApi,
        nothing_to_suggest: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nothing_to_suggest.return_value = {
            **NO_SUGGESTIONS,
            "classes_without_nodes": [
                {"resource_class_id": 3, "name": "a100-pool", "node_selector": {}}
            ],
            "needs_agent_upgrade": True,
        }

        created = offer_suggested_classes(byoc_api, "lab-cluster", yes=True)

        assert created == []
        out = capsys.readouterr().out
        assert "No node matches the node selector of class 'a100-pool'" in out
        assert "arena cluster upgrade --name lab-cluster" in out
        assert "Suggested classes" not in out

    def test_says_when_the_agent_has_not_reported(
        self,
        byoc_api: ByocApi,
        nothing_to_suggest: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nothing_to_suggest.return_value = {**NO_SUGGESTIONS, "reported_at": None}

        assert offer_suggested_classes(byoc_api, "lab-cluster", yes=True) == []

        assert "The agent has not reported its nodes yet" in capsys.readouterr().out

    def test_an_arena_without_suggestions_is_skipped(
        self,
        byoc_api: ByocApi,
        nothing_to_suggest: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nothing_to_suggest.side_effect = ArenaAPIError("Not Found", status_code=404)

        assert offer_suggested_classes(byoc_api, "lab-cluster", yes=True) == []

        assert capsys.readouterr().out == ""

    def test_other_errors_propagate(
        self, byoc_api: ByocApi, nothing_to_suggest: MagicMock
    ) -> None:
        nothing_to_suggest.side_effect = ArenaAPIError("boom", status_code=500)

        with pytest.raises(ArenaAPIError, match="boom"):
            offer_suggested_classes(byoc_api, "lab-cluster", yes=True)


class TestRunClusterRegisterSuggestedClasses:
    @pytest.fixture(autouse=True)
    def _no_helm(self) -> Iterator[None]:
        with (
            patch("agilerl.byoc.cluster_helm.helm_available", return_value=False),
            patch("agilerl.byoc.cluster_register._install_charts_from_arena_package"),
            patch.object(ByocApi, "register_cluster", return_value=LAB_BUNDLE),
        ):
            yield

    def test_yes_creates_the_suggestions_after_register(
        self,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        nothing_to_suggest.return_value = {
            **NO_SUGGESTIONS,
            "suggestions": [H100_SUGGESTION],
        }

        run_cluster_register(
            mock_client,
            name="lab-cluster",
            output_dir=tmp_path,
            skip_enable=True,
            force=False,
            yes=True,
        )

        assert created_class_names(mock_client) == ["lab-cluster-h100-80-x8"]
        assert (
            "Resource class 'lab-cluster-h100-80-x8' is linked"
            in capsys.readouterr().out
        )

    def test_connect_turns_the_offer_off(
        self,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        tmp_path: Path,
    ) -> None:
        run_cluster_register(
            mock_client,
            name="lab-cluster",
            output_dir=tmp_path,
            skip_enable=True,
            force=False,
            yes=True,
            suggest_classes=False,
        )

        nothing_to_suggest.assert_not_called()

    @pytest.mark.parametrize(
        ("byoc_scope", "resource_classes"),
        [
            ("platform", ()),
            (
                None,
                (
                    ClusterResourceClass(
                        name="spec-gpu",
                        num_nodes=1,
                        node_selector={"class": "gpu"},
                        metadata={},
                    ),
                ),
            ),
        ],
    )
    def test_platform_and_spec_classes_get_no_suggestions(
        self,
        mock_client: MagicMock,
        nothing_to_suggest: MagicMock,
        tmp_path: Path,
        byoc_scope: str | None,
        resource_classes: tuple[ClusterResourceClass, ...],
    ) -> None:
        with (
            patch.object(ByocApi, "find_class", return_value=None),
            patch.object(ByocApi, "create_class", return_value={"name": "spec-gpu"}),
        ):
            run_cluster_register(
                mock_client,
                name="lab-cluster",
                output_dir=tmp_path,
                skip_enable=True,
                force=False,
                byoc_scope=byoc_scope,
                resource_classes=resource_classes,
                yes=True,
            )

        nothing_to_suggest.assert_not_called()
