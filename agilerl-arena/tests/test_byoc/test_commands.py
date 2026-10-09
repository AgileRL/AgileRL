# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``arena cluster`` Click commands and verbosity helper."""

from __future__ import annotations

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
    _kubeconfig_environment,
    _load_destroy_target,
    _load_provisioning,
    _recover_spec,
    _registration_values_from_spec,
    build_cluster_generate_spec_command,
    build_cluster_provision_command,
    build_cluster_status_command,
)
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    recover_stored_config,
    registration_payload,
    resource_classes,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
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

    def test_small_preset_keeps_room_for_the_workload(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "1gpu-4vcpu-16gb"}])

        classes = resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 3
        assert compute["memoryBytes"] == "14 GiB"

    def test_rejects_preset_too_small_for_headroom(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "1gpu-1vcpu-1gb"}])

        with pytest.raises(click.ClickException, match="too small"):
            resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})


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
            provisioner_cls.return_value.status.return_value = {"cluster_id": "mk8s-1"}
            result = CliRunner().invoke(
                build_cluster_status_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "cluster_id: mk8s-1" in result.output


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
            patch.object(ByocApi, "on_prem_cluster", return_value={"name": "broken"}),
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
