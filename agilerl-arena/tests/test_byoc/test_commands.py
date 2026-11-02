# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``arena cluster`` Click commands and verbosity helper."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.byoc import (
    ClusterDynamicGroup,
    build_cluster_plan_command,
    build_cluster_register_command,
    build_cluster_rotate_token_command,
    build_cluster_unregister_command,
)
from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.commands import (
    _apply_verbosity,
    _byoc_provider_with_module,
    _kubeconfig_environment,
    _load_destroy_target,
    _load_provisioning,
    _node_groups_for_classes,
    _recover_spec,
    _register_gpu_resource_classes,
    _register_runtime_fields,
    _registration_values_from_spec,
    _resource_classes_from_outputs,
    _selected_worker_names,
    build_cluster_generate_spec_command,
    build_cluster_provision_command,
    build_cluster_status_command,
)
from agilerl.arena.byoc.provisioning.module_catalog import ModulePin, write_module_pin
from agilerl.arena.byoc.provisioning.providers.nebius.platforms import GpuPreset
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    NEBIUS_CREDENTIALS_PATH,
    NEBIUS_PROFILE_NAME,
    attach_service_account_credentials,
    recover_stored_config,
    registration_payload,
    resource_classes,
    service_account_credentials,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.terraform import ClusterOutputs
from agilerl.arena.config import CommandConfig

INSTALLED_BYOC_RELEASES = [
    ("arena-byoc-agent", "arena"),
    ("arena-byoc-storage", "storage"),
]
AGENT_HELM_RELEASE = [("arena-byoc-agent", "arena")]


def test_cluster_unregister_requires_confirmation(
    command_config: CommandConfig,
) -> None:
    with (
        patch("agilerl.arena.byoc.commands.helm_available", return_value=False),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=[],
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="\n",
        )

    assert result.exit_code == 0, result.output
    assert "Aborted." in result.output
    unregister_mock.assert_not_called()


def test_cluster_unregister_confirms_then_calls_api(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch("agilerl.arena.byoc.commands.helm_available", return_value=False),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=[],
        ),
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="y\n",
        )

    assert result.exit_code == 0, result.output
    unregister_mock.assert_called_once_with("prod-cluster")
    assert "Unregistered BYOC cluster 'prod-cluster'." in result.output


def test_cluster_unregister_yes_skips_prompt(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch("agilerl.arena.byoc.commands.helm_available", return_value=False),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=[],
        ),
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster", "--yes"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert "Unregister BYOC cluster" not in result.output
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_skips_helm_check_without_kubeconfig_or_helm(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=None,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=False),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="y\n",
        )

    assert result.exit_code == 0, result.output
    assert "helm is not on PATH" in result.output
    uninstall_mock.assert_not_called()
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_skips_helm_check_without_kubeconfig(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=None,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="y\n",
        )

    assert result.exit_code == 0, result.output
    assert "no kubeconfig" in result.output
    uninstall_mock.assert_not_called()
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_declines_helm_then_unregisters(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    tmp_path: Path,
) -> None:
    client = MagicMock()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=INSTALLED_BYOC_RELEASES,
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="n\nn\ny\n",
        )

    assert result.exit_code == 0, result.output
    uninstall_mock.assert_not_called()
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_accepts_helm_then_unregisters(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    tmp_path: Path,
) -> None:
    client = MagicMock()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=INSTALLED_BYOC_RELEASES,
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="y\nn\ny\n",
        )

    assert result.exit_code == 0, result.output
    assert "checkpoints, metrics, and datasets" in result.output
    uninstall_mock.assert_called_once_with(AGENT_HELM_RELEASE, kubeconfig)
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_uninstalls_agent_outside_default_namespace(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    tmp_path: Path,
) -> None:
    client = MagicMock()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=[("arena-byoc-agent", "arena-agent")],
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster", "--uninstall-helm", "--yes"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    uninstall_mock.assert_called_once_with(
        [("arena-byoc-agent", "arena-agent")], kubeconfig
    )
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_accepts_helm_declines_arena_skips_uninstall(
    command_config: CommandConfig,
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=INSTALLED_BYOC_RELEASES,
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="y\nn\nn\n",
        )

    assert result.exit_code == 0, result.output
    assert "Aborted." in result.output
    uninstall_mock.assert_not_called()
    unregister_mock.assert_not_called()


def test_cluster_unregister_uninstall_helm_yes_skips_prompts(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    tmp_path: Path,
) -> None:
    client = MagicMock()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=INSTALLED_BYOC_RELEASES,
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster", "--uninstall-helm", "--yes"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert "Uninstall the Arena agent Helm release?" not in result.output
    assert "Unregister BYOC cluster" not in result.output
    assert "Keeping MinIO storage Helm release" in result.output
    uninstall_mock.assert_called_once_with(AGENT_HELM_RELEASE, kubeconfig)
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_uninstall_storage_yes_removes_minio(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    tmp_path: Path,
) -> None:
    client = MagicMock()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=INSTALLED_BYOC_RELEASES,
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
        ) as uninstall_mock,
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            [
                "--name",
                "prod-cluster",
                "--uninstall-helm",
                "--uninstall-storage",
                "--yes",
            ],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert "Keeping MinIO storage Helm release" not in result.output
    uninstall_mock.assert_called_once_with(INSTALLED_BYOC_RELEASES, kubeconfig)
    unregister_mock.assert_called_once_with("prod-cluster")


def test_cluster_unregister_helm_uninstall_failure_skips_api(
    command_config: CommandConfig,
    tmp_path: Path,
) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.byoc.commands.resolve_cluster_kubeconfig",
            return_value=kubeconfig,
        ),
        patch("agilerl.arena.byoc.commands.helm_available", return_value=True),
        patch(
            "agilerl.arena.byoc.commands.list_installed_byoc_releases",
            return_value=INSTALLED_BYOC_RELEASES,
        ),
        patch(
            "agilerl.arena.byoc.commands.uninstall_byoc_releases",
            side_effect=click.ClickException("helm uninstall failed"),
        ),
        patch.object(ByocApi, "unregister_cluster") as unregister_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster", "--uninstall-helm", "--yes"],
            obj=command_config,
        )

    assert result.exit_code != 0
    assert "helm uninstall failed" in result.output
    unregister_mock.assert_not_called()


def test_cluster_rotate_token_requires_confirmation(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.arena.byoc.commands.run_cluster_rotate_token") as rotate_mock:
        result = CliRunner().invoke(
            build_cluster_rotate_token_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="\n",
        )

    assert result.exit_code == 0, result.output
    assert "Aborted." in result.output
    rotate_mock.assert_not_called()


def test_cluster_rotate_token_yes_skips_prompt(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch("agilerl.arena.byoc.commands.run_cluster_rotate_token") as rotate_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_rotate_token_command(),
            ["--name", "prod-cluster", "--yes"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert "Rotate the cluster token" not in result.output
    rotate_mock.assert_called_once()
    assert rotate_mock.call_args.args[0] is client
    assert rotate_mock.call_args.kwargs["name"] == "prod-cluster"
    assert rotate_mock.call_args.kwargs["namespace"] is None


def test_cluster_rotate_token_passes_agent_namespace(
    command_config: CommandConfig, client_context: Callable[[MagicMock], MagicMock]
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch("agilerl.arena.byoc.commands.run_cluster_rotate_token") as rotate_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_rotate_token_command(),
            [
                "--name",
                "prod-cluster",
                "--agent-namespace",
                "arena-agent",
                "--yes",
            ],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert rotate_mock.call_args.kwargs["namespace"] == "arena-agent"


def test_cluster_group_includes_hardcoded_commands() -> None:
    cluster = ClusterDynamicGroup()

    assert cluster.name == "cluster"
    assert "unregister" in cluster.commands
    assert "rotate-token" in cluster.commands
    assert "register" in cluster.commands
    assert "register-classes" in cluster.commands


def test_apply_verbosity_toggles_debug_level() -> None:
    arena_logger = logging.getLogger("agilerl.arena")
    original = arena_logger.level
    try:
        _apply_verbosity(verbose=False)
        assert arena_logger.level == original  # unchanged
        _apply_verbosity(verbose=True)
        assert arena_logger.level == logging.DEBUG
    finally:
        arena_logger.setLevel(original)


def _nebius_spec(**overrides: object) -> ClusterSpec:
    raw = dict(overrides)
    nebius: dict[str, object] = {
        "tenant_id": "tenant",
        "project_id": "project",
        "storage_project_id": "storage-project",
        "subnet_id": "subnet",
        "object_storage": {"size_gib": 1024},
    }
    nebius.update(raw)
    return ClusterSpec.model_validate(
        {
            "provider": "nebius",
            "name": "test",
            "terraform_state": {"bucket": "test-tfstate", "endpoint": "https://s3"},
            "arena": {
                "storage": {
                    "bucket": "bucket",
                    "endpoint": "https://storage.example.com",
                },
                "inference": {"domain": "inference.example.com"},
            },
            "nebius": nebius,
        }
    )


class TestNebiusByocProvider:
    def test_includes_state_endpoint(self) -> None:
        provider = registration_payload(_nebius_spec())

        assert provider["config"]["terraform_state"]["endpoint"] == "https://s3"

    def test_fills_project_ids_the_spec_omits(self) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        provider = registration_payload(
            spec, project_id="project-created", storage_project_id="storage-created"
        )

        assert provider["config"]["project_id"] == "project-created"
        assert provider["config"]["storage_project_id"] == "storage-created"
        assert "service_account_key" not in provider["config"]
        assert "profile" not in provider["config"]


class TestServiceAccountCredentials:
    def test_builds_key_and_profile_for_the_created_account(self) -> None:
        pem_label = "PRIVATE " + "KEY"
        private_key = f"-----BEGIN {pem_label}-----\nYWJj\n-----END {pem_label}-----"
        outputs = {
            "service_account_id": "serviceaccount-e00m886x7epbx082j3",
            "service_account_public_key_id": "publickey-e00abc",
            "service_account_private_key": private_key + "\n",
            "project_id": "project-e00",
        }

        documents = service_account_credentials(outputs)

        assert documents is not None
        key, profile = documents
        parsed = json.loads(key)
        assert parsed["subject-credentials"] == {
            "type": "JWT",
            "alg": "RS256",
            "private-key": private_key,
            "kid": "publickey-e00abc",
            "iss": "serviceaccount-e00m886x7epbx082j3",
            "sub": "serviceaccount-e00m886x7epbx082j3",
        }
        assert profile == "\n".join(
            (
                f"default: {NEBIUS_PROFILE_NAME}",
                "profiles:",
                f"  {NEBIUS_PROFILE_NAME}:",
                "    endpoint: api.nebius.cloud",
                "    parent-id: project-e00",
                f"    service-account-credentials-file-path: {NEBIUS_CREDENTIALS_PATH}",
                "",
            )
        )

    def test_omits_credentials_when_the_account_was_not_created(self) -> None:
        payload = registration_payload(_nebius_spec())

        attach_service_account_credentials(
            payload,
            {"project_id": "project-e00", "service_account_id": None},
        )

        assert "service_account_key" not in payload["config"]
        assert "profile" not in payload["config"]

    def test_register_body_stores_the_created_credentials(self) -> None:
        payload = _byoc_provider_with_module(
            _nebius_spec(),
            ClusterOutputs(
                cluster_name="test",
                context="test",
                kubeconfig_path=Path("kubeconfig"),
                storage_access_key_id="access",
                storage_secret_access_key="secret",
                worker_node_group_ids={"gpu": "nodegroup-1"},
                project_id="project-e00",
                service_account_key='{"subject-credentials":{}}',
                service_account_profile="default: nebius-default\n",
            ),
            None,
        )

        assert payload["config"]["service_account_key"] == '{"subject-credentials":{}}'
        assert payload["config"]["profile"] == "default: nebius-default\n"


class TestStoredSpecHelpers:
    def test_stored_subnet_id_none_when_project_missing(self) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        recovered = recover_stored_config(spec, {"subnet_id": "subnet-2"})

        assert recovered.nebius.subnet_id is None

    def test_stored_state_location_empty_when_not_a_mapping(self) -> None:
        spec = _nebius_spec()

        assert recover_stored_config(spec, {"terraform_state": "nope"}) is spec


class TestGpuResourceClasses:
    def test_requires_node_group_id(self) -> None:
        with pytest.raises(click.ClickException, match="missing node group id"):
            resource_classes(_nebius_spec(), worker_node_group_ids={})

    def test_wraps_invalid_preset(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "2vcpu-8gb"}])

        with pytest.raises(click.ClickException, match="GPU worker preset"):
            resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

    def test_registers_the_preset_machine_size(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "1gpu-4vcpu-16gb"}])

        classes = resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 4
        assert compute["memoryBytes"] == "16 GB"

    def test_keeps_a_one_vcpu_preset(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "1gpu-1vcpu-1gb"}])

        classes = resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 1
        assert compute["memoryBytes"] == "1 GB"


class TestRegistrationValuesFromSpec:
    def test_rejects_non_list_parent_refs(self) -> None:
        with pytest.raises(click.ClickException, match="JSON list"):
            _registration_values_from_spec(
                _nebius_spec(), gateway_api_parent_refs={"name": "gw"}
            )


class TestKubeconfigEnvironment:
    def test_restores_previous_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KUBECONFIG", "/old/config")
        kubeconfig = Path("/new/config")

        with _kubeconfig_environment(kubeconfig):
            assert os.environ["KUBECONFIG"] == str(kubeconfig)

        assert os.environ["KUBECONFIG"] == "/old/config"


class TestLoadProvisioning:
    def test_wraps_invalid_spec(self, tmp_path: Path) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text("not: valid: yaml: [\n", encoding="utf-8")

        with pytest.raises(click.ClickException, match="Invalid YAML"):
            _load_provisioning(spec, None)


class TestUnregisterNameRequired:
    def test_blank_name_is_rejected(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "   "],
            obj=command_config,
            input="y\n",
        )

        assert result.exit_code != 0
        assert "--name is required" in result.output


class TestGenerateSpecCommand:
    def test_wraps_existing_file_without_force(self, tmp_path: Path) -> None:
        destination = tmp_path / "arena-nebius.yaml"
        destination.write_text("existing\n", encoding="utf-8")

        result = CliRunner().invoke(
            build_cluster_generate_spec_command(),
            ["--output", str(destination)],
        )

        assert result.exit_code != 0
        assert "already exists" in result.output


class TestStatusCommand:
    def test_prints_outputs(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch("agilerl.arena.byoc.commands.ClusterProvisioner") as provisioner_cls,
        ):
            provisioner_cls.return_value.status.return_value = {
                "cluster_id": "mk8s-1",
                "service_account_private_key": "pem-should-not-print",
                "storage_secret_access_key": "secret-access-should-not-print",
            }
            result = CliRunner().invoke(
                build_cluster_status_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "cluster_id: mk8s-1" in result.output
        assert "service_account_private_key: <sensitive>" in result.output
        assert "storage_secret_access_key: <sensitive>" in result.output
        assert "pem-should-not-print" not in result.output
        assert "secret-access-should-not-print" not in result.output


class TestProvisionCommand:
    def test_rejects_provider_mismatch(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        mismatched = MagicMock()
        mismatched.provider = "other"
        with (
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.arena.byoc.commands._registration_values_from_spec",
                return_value=(mismatched, {"byoc_provider": {}}),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_provision_command(),
                ["nebius", "--spec", str(spec), "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "does not match spec provider" in result.output


class TestLoadDestroyTarget:
    def test_wraps_invalid_stored_cluster(self, command_config: CommandConfig) -> None:
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
            ) as client_cm,
            patch.object(ByocApi, "byoc_cluster", return_value={"name": "broken"}),
        ):
            client = MagicMock()
            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            client_cm.return_value = cm
            with pytest.raises(click.ClickException, match="no stored cloud settings"):
                _load_destroy_target(command_config, None, "broken", None)


class TestRegistrationValuesFromSpecJson:
    def test_parses_json_string_parent_refs(self) -> None:
        spec, registration = _registration_values_from_spec(
            _nebius_spec(),
            gateway_api_parent_refs='[{"name": "arena", "namespace": "arena"}]',
        )

        assert registration["gateway_api_parent_refs"] == [
            {"name": "arena", "namespace": "arena"}
        ]
        assert spec.arena.gateway.parent_refs == [
            {"name": "arena", "namespace": "arena"}
        ]

    def test_defaults_placement_and_applies_override(self) -> None:
        spec, registration = _registration_values_from_spec(_nebius_spec())
        overridden, overridden_registration = _registration_values_from_spec(
            _nebius_spec(),
            placement="ephemeral",
        )

        assert spec.values.placement == "in-cluster"
        assert registration["placement"] == "in-cluster"
        assert overridden.values.placement == "ephemeral"
        assert overridden_registration["placement"] == "ephemeral"

    def test_name_only_register_defaults_placement(self) -> None:
        fields = _register_runtime_fields(
            None,
            {
                "placement": None,
                "domain": None,
                "gateway_api_parent_refs": None,
            },
        )

        assert fields["placement"] == "in-cluster"


class TestKubeconfigEnvironmentUnset:
    def test_clears_when_previously_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KUBECONFIG", raising=False)
        kubeconfig = Path("/new/config")

        with _kubeconfig_environment(kubeconfig):
            assert os.environ["KUBECONFIG"] == str(kubeconfig)

        assert "KUBECONFIG" not in os.environ


class TestRecoverNebiusSpec:
    def test_rejects_unknown_providers(self, command_config: CommandConfig) -> None:
        spec = ClusterSpec.model_construct(
            **{**_nebius_spec().model_dump(), "provider": "other"}
        )

        with pytest.raises(ValueError, match="Unsupported cluster provider"):
            _recover_spec(command_config, spec)

    def test_skips_lookup_when_both_project_ids_are_set(
        self, command_config: CommandConfig
    ) -> None:
        spec = _nebius_spec()

        with patch("agilerl.arena.byoc.commands.arena_client") as client_cm:
            assert _recover_spec(command_config, spec) is spec
            client_cm.assert_not_called()

    def test_fills_missing_ids_from_arena(self, command_config: CommandConfig) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        with (
            patch("agilerl.arena.byoc.commands.arena_client") as client_cm,
            patch.object(
                ByocApi,
                "stored_provider_config",
                return_value={
                    "project_id": "project-stored",
                    "storage_project_id": "storage-stored",
                    "terraform_state": {
                        "key": "clusters/test/terraform.tfstate",
                        "endpoint": "https://s3.stored",
                    },
                },
            ),
        ):
            client = MagicMock()
            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            client_cm.return_value = cm
            recovered = _recover_spec(command_config, spec)

        assert recovered.nebius.project_id == "project-stored"
        assert recovered.nebius.storage_project_id == "storage-stored"


class TestSpecWithStoredNebiusConfig:
    def test_returns_spec_when_nothing_is_missing(self) -> None:
        spec = _nebius_spec()

        assert recover_stored_config(spec, {"project_id": "other"}) is spec


class TestStoredStateLocationEndpoint:
    def test_fills_endpoint_when_spec_omits_it(self) -> None:
        spec = _nebius_spec()
        spec = spec.model_copy(
            update={
                "terraform_state": spec.terraform_state.model_copy(
                    update={"endpoint": None, "key": None}
                )
            }
        )

        recovered = recover_stored_config(
            spec, {"terraform_state": {"endpoint": "https://s3.stored"}}
        )

        assert recovered.terraform_state.endpoint == "https://s3.stored"


class TestRegisterCommand:
    def test_requires_name_without_spec(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_register_command(),
            [],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--name is required" in result.output

    def test_requires_spec_when_provider_is_given(
        self, command_config: CommandConfig
    ) -> None:
        result = CliRunner().invoke(
            build_cluster_register_command(),
            ["nebius", "--name", "prod-cluster"],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--spec is required" in result.output

    def test_wraps_invalid_spec(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text("not: valid: yaml: [\n", encoding="utf-8")

        result = CliRunner().invoke(
            build_cluster_register_command(),
            ["--spec", str(spec)],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "Invalid YAML" in result.output

    def test_aborts_when_provision_is_declined(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch("agilerl.arena.byoc.commands.ClusterProvisioner") as provisioner_cls,
            patch("agilerl.arena.byoc.commands.click.confirm", return_value=False),
        ):
            provisioner_cls.return_value.cluster_exists.return_value = False
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Aborted." in result.output

    def test_wraps_invalid_domain_without_spec(
        self, command_config: CommandConfig
    ) -> None:
        with patch("agilerl.arena.byoc.commands.run_cluster_register"):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "prod-cluster", "--domain", "https://bad.example.com"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "bare DNS name" in result.output

    def test_parses_parent_refs_without_spec(
        self, command_config: CommandConfig
    ) -> None:
        with patch("agilerl.arena.byoc.commands.run_cluster_register") as register_mock:
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--name",
                    "prod-cluster",
                    "--gateway-api-parent-refs",
                    '[{"name": "arena"}]',
                    "--skip-enable",
                    "--no-write",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert register_mock.call_args.kwargs["gateway_api_parent_refs"] == [
            {"name": "arena"}
        ]


class TestPlanCommand:
    def test_wraps_invalid_registration_overrides(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.arena.byoc.commands._registration_values_from_spec",
                side_effect=ValueError("bare DNS name"),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_plan_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "bare DNS name" in result.output


class TestProvisionCommandOverrides:
    def test_wraps_invalid_registration_overrides(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.arena.byoc.commands._registration_values_from_spec",
                side_effect=ValueError("bare DNS name"),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_provision_command(),
                ["nebius", "--spec", str(spec), "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "bare DNS name" in result.output


class TestRotateTokenCommand:
    def test_rejects_blank_name(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_rotate_token_command(),
            ["--name", "   "],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--name is required" in result.output


class TestRegisterProviderMismatch:
    def test_rejects_provider_that_does_not_match_spec(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        def recover(_config: CommandConfig, cluster_spec: ClusterSpec) -> ClusterSpec:
            object.__setattr__(cluster_spec, "provider", "other")
            return cluster_spec

        with patch("agilerl.arena.byoc.commands._recover_spec", side_effect=recover):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "does not match spec provider" in result.output


H100_PRESET = GpuPreset(
    platform="gpu-h100-sxm",
    preset="1gpu-16vcpu-200gb",
    gpu_count=1,
    vcpu_count=16,
    memory_gib=200,
    vram_gib=80,
)


class TestByocProviderModulePin:
    def test_adds_the_installed_module_pin(self, tmp_path: Path) -> None:
        write_module_pin(
            tmp_path,
            ModulePin(
                name="arena-cluster-nebius",
                version="1.0.0",
                digest="sha256:" + "ab" * 32,
                orchestration="nebius.v1",
                files=("main.tf",),
            ),
        )

        payload = _byoc_provider_with_module(
            _nebius_spec(),
            ClusterOutputs(
                cluster_name="test",
                context="test",
                kubeconfig_path=Path("kubeconfig"),
                storage_access_key_id="access",
                storage_secret_access_key="secret",
                worker_node_group_ids={},
            ),
            tmp_path,
        )

        assert payload["config"]["module"] == {
            "name": "arena-cluster-nebius",
            "version": "1.0.0",
            "digest": "sha256:" + "ab" * 32,
            "orchestration": "nebius.v1",
        }


class TestResourceClassesFromOutputs:
    def test_rejects_missing_worker_ids(self) -> None:
        with pytest.raises(click.ClickException, match="no worker_node_group_ids"):
            _resource_classes_from_outputs(_nebius_spec(), {})

    def test_rejects_a_non_map(self) -> None:
        with pytest.raises(click.ClickException, match="must be a string map"):
            _resource_classes_from_outputs(
                _nebius_spec(), {"worker_node_group_ids": "ng-1"}
            )


class TestRegisterGpuResourceClasses:
    def test_rejects_non_nebius_specs(self, command_config: CommandConfig) -> None:
        spec = ClusterSpec.model_validate(
            {
                "provider": "aws",
                "name": "eks-byoc",
                "terraform_state": {"bucket": "eks-tfstate"},
                "arena": {
                    "storage": {
                        "bucket": "eks-data",
                        "endpoint": "https://s3.eu-west-1.amazonaws.com",
                    },
                    "inference": {"domain": "inference.example.com"},
                },
                "aws": {
                    "region": "eu-west-1",
                    "object_storage": {"size_gib": 1024},
                },
            }
        )

        with pytest.raises(click.ClickException, match="supports Nebius"):
            _register_gpu_resource_classes(
                command_config,
                MagicMock(),
                spec,
                Path("cluster.yaml"),
                Path("terraform"),
                Path("out"),
                register_all=True,
                requested_presets=(),
                yes=True,
                module_version=None,
                register_node_group=False,
            )

    def test_rejects_an_empty_preset_list(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        runner = MagicMock()

        with (
            patch(
                "agilerl.arena.byoc.commands._cluster_provisioner",
                return_value=runner,
            ),
            patch(
                "agilerl.arena.byoc.commands.list_gpu_presets",
                return_value=[],
            ),
        ):
            with pytest.raises(click.ClickException, match="No GPU presets"):
                _register_gpu_resource_classes(
                    command_config,
                    api,
                    _nebius_spec(),
                    tmp_path / "cluster.yaml",
                    tmp_path / "terraform",
                    tmp_path / "out",
                    register_all=True,
                    requested_presets=(),
                    yes=True,
                    module_version=None,
                    register_node_group=False,
                )


class TestSelectedWorkerNames:
    def test_rejects_a_preset_missing_from_the_spec(self) -> None:
        with pytest.raises(click.ClickException, match="was not added"):
            _selected_worker_names(_nebius_spec(), [H100_PRESET])


class TestNodeGroupsForClasses:
    def test_returns_recorded_ids_when_every_group_exists(self, tmp_path: Path) -> None:
        spec = _nebius_spec(
            workers=[
                {
                    "name": "gpu-h100-sxm-1gpu-16vcpu-200gb",
                    "platform": "gpu-h100-sxm",
                    "preset": "1gpu-16vcpu-200gb",
                    "register_node_group": True,
                }
            ]
        )
        runner = MagicMock()
        ids = {"gpu-h100-sxm-1gpu-16vcpu-200gb": "ng-1"}

        group_ids = _node_groups_for_classes(
            spec,
            original=spec,
            spec_path=tmp_path / "cluster.yaml",
            state_dir=tmp_path / "terraform",
            output_dir=tmp_path / "out",
            new_workers=[],
            selected=[H100_PRESET],
            terraform_outputs={"worker_node_group_ids": ids},
            runner=runner,
        )

        assert group_ids == ids
        runner.status.assert_not_called()


class TestLoadDestroyTargetStoredRow:
    def test_wraps_invalid_stored_provider_config(
        self, command_config: CommandConfig
    ) -> None:
        with (
            patch("agilerl.arena.byoc.commands.arena_client") as client_cm,
            patch.object(
                ByocApi,
                "byoc_cluster",
                return_value={
                    "name": "broken",
                    "byocProvider": {"provider": "aws", "config": "nope"},
                },
            ),
        ):
            client = MagicMock()
            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            client_cm.return_value = cm
            with pytest.raises(click.ClickException, match="no stored AWS settings"):
                _load_destroy_target(command_config, None, "broken", None)


class TestProvisionPrintsModulePin:
    def test_echoes_the_installed_module(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )
        state_dir = tmp_path / "terraform"
        write_module_pin(
            state_dir,
            ModulePin(
                name="arena-cluster-nebius",
                version="1.0.0",
                digest="sha256:" + "ab" * 32,
                orchestration="nebius.v1",
                files=("main.tf",),
            ),
        )
        outputs = ClusterOutputs(
            cluster_name="arena-nebius",
            context="arena-nebius",
            kubeconfig_path=tmp_path / "kubeconfig",
            storage_access_key_id="AKIA-test",
            storage_secret_access_key="secret-test",
            worker_node_group_ids={"workers": "ng-1"},
        )

        with (
            patch(
                "agilerl.arena.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.provision",
                return_value=outputs,
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_provision_command(),
                [
                    "nebius",
                    "--spec",
                    str(spec),
                    "--state-dir",
                    str(state_dir),
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Module arena-cluster-nebius 1.0.0" in result.output
