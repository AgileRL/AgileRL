# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for cloud cluster provisioning commands."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import click
import pytest
import yaml
from click.testing import CliRunner

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.cluster_register import ClusterResourceClass
from agilerl.arena.byoc.commands import (
    build_cluster_destroy_command,
    build_cluster_generate_spec_command,
    build_cluster_plan_command,
    build_cluster_provision_command,
)
from agilerl.arena.byoc.provisioning.gateway_api import (
    CILIUM_GATEWAY_API_CLUSTER_ROLE,
    CILIUM_GATEWAY_CLASS,
    CILIUM_GATEWAY_CONTROLLER,
    CILIUM_GATEWAY_SERVICES_ROLE,
    GATEWAY_API_EXPERIMENTAL_CRDS,
    GATEWAY_API_STANDARD_CRDS,
    HTTP_LISTENER_NAME,
    HTTPS_LISTENER_NAME,
    HTTPS_REDIRECT_ROUTE_NAME,
    _create_self_signed_tls_secret,
    _kubectl,
    build_cilium_gateway_api_rbac_manifests,
    build_gateway_api_parent_refs,
    configure_gateway_api,
    enable_cilium_gateway_api,
    ensure_cilium_gateway,
    ensure_cilium_gateway_class,
    ensure_https_redirect_route,
    ensure_inference_tls_secret,
)
from agilerl.arena.byoc.provisioning.inference import (
    DEFAULT_INFERENCE_TLS_SECRET_NAME,
    normalize_inference_domain,
    resolve_inference_hostname_template,
)
from agilerl.arena.byoc.provisioning.kubectl import require_kubectl
from agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig import (
    build_kubeconfig_argv,
    build_nebius_kubeconfig_argv,
    resolve_kubeconfig_executable,
)
from agilerl.arena.byoc.provisioning.providers.nebius.provider import (
    NEBIUS,
    NEBIUS_COMPUTE_PROJECT_RESOURCE,
    NEBIUS_OBJECT_STORAGE_RESOURCES,
    NEBIUS_PROVIDER_ENV_VARS,
    NEBIUS_PROVIDER_PROFILE,
    NEBIUS_PROVIDER_SERVICE_ACCOUNT,
    NEBIUS_STORAGE_PROJECT_RESOURCE,
    _worker_node_group_ids,
    resolve_nebius_service_account_id,
    terraform_project_id,
    validate_nebius_spec,
    write_nebius_provider_config,
)
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    NEBIUS_WORKER_NODE_GROUP_LABEL,
    nebius_gpu_vram_gib,
    parse_nebius_gpu_preset,
    resource_classes,
)
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    spec_from_stored_row as cluster_spec_from_on_prem_row,
)
from agilerl.arena.byoc.provisioning.providers.nebius.terraform import (
    TerraformStateCredentials,
    _credentials_from_fields,
    _ensure_state_access_key,
    _find_named_resource,
    _get_by_name,
    _member_id,
    _nebius_cmd,
    _next_page_token,
    _parse_nebius_json,
    _payload_items,
    _resource_id,
    _resource_name,
    _run_nebius_json,
    _storage_access_key_id,
    delete_cluster_terraform_state,
    delete_nebius_project,
    delete_terraform_state_bucket,
    ensure_terraform_state_bucket,
    experiment_storage_import_ids,
    load_terraform_state_credentials,
    nebius_region_from_endpoint,
    read_terraform_state_credentials,
    storage_project_id,
    terraform_state_bucket_exists,
)
from agilerl.arena.byoc.provisioning.provisioner import ClusterProvisioner
from agilerl.arena.byoc.provisioning.runtime_class import (
    build_gpu_runtime_class_manifest,
    ensure_gpu_runtime_class,
)
from agilerl.arena.byoc.provisioning.spec import (
    DEFAULT_LAB_STORAGE_SECRET_NAME,
    DEFAULT_STORAGE_SECRET_NAME,
    ClusterSpec,
    apply_cluster_spec_overrides,
    load_cluster_spec,
    render_default_cluster_spec,
    write_default_cluster_spec,
)
from agilerl.arena.byoc.provisioning.storage import (
    aws_region_from_s3_endpoint,
    ensure_storage_secret,
)
from agilerl.arena.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
    local_terraform_state_has_resources,
    terraform_address_matches,
)
from agilerl.arena.config import CommandConfig


def materialize_outputs(
    runner: TerraformRunner, work_dir: Path, output_dir: Path
) -> ClusterOutputs:
    """Read Terraform outputs and turn them into cluster outputs."""
    return NEBIUS.cluster_outputs(
        runner.terraform_outputs(work_dir),
        output_dir=output_dir,
        run=runner.run,
    )


def write_spec(path: Path) -> None:
    """Write a valid minimal Nebius cluster spec."""
    path.write_text(
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


def write_auto_projects_spec(path: Path) -> None:
    """Write a spec that omits both Nebius project ids."""
    path.write_text(
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
  region: eu-north1
  object_storage:
    size_gib: 1024
""",
        encoding="utf-8",
    )


def _terraform_project_output(_work_dir: Path, name: str) -> str:
    """Return named Terraform project outputs used by bootstrap tests."""
    return {
        "project_id": "project-created",
        "storage_project_id": "storage-project-created",
    }[name]


def test_load_cluster_spec_rejects_unknown_values(tmp_path: Path) -> None:
    """Cluster specs reject unexpected Nebius fields."""
    spec_path = tmp_path / "cluster.yaml"
    spec_path.write_text(
        """
provider: nebius
name: test
terraform_state:
  bucket: test-tfstate
arena:
  storage:
    bucket: bucket
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant
  project_id: project
  storage_project_id: storage-project
  subnet_id: subnet
  object_storage:
    size_gib: 1024
  unexpected: value
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid cluster spec"):
        load_cluster_spec(spec_path)


def _pop_arena(overrides: dict[str, object]) -> dict[str, object]:
    """Move Arena fields out of a flat override map."""
    storage: dict[str, object] = {
        "bucket": "bucket",
        "endpoint": "https://storage.example.com",
    }
    arena: dict[str, object] = {
        "storage": storage,
        "inference": {"domain": "inference.example.com"},
    }
    if "storage_bucket_name" in overrides:
        storage["bucket"] = overrides.pop("storage_bucket_name")
    if "storage_endpoint" in overrides:
        storage["endpoint"] = overrides.pop("storage_endpoint")
    if "inference" in overrides:
        arena["inference"] = overrides.pop("inference")
    gateway = overrides.pop("gateway_api", None)
    if isinstance(gateway, dict):
        mapped: dict[str, object] = {}
        if "enable" in gateway:
            mapped["enable"] = gateway["enable"]
        if "gateway_name" in gateway:
            mapped["name"] = gateway["gateway_name"]
        arena["gateway"] = mapped
    return arena


def _nebius_spec(**overrides: object) -> ClusterSpec:
    """Return a valid Nebius cluster spec, with optional cloud overrides."""
    raw = dict(overrides)
    arena = _pop_arena(raw)
    if "arena" in raw:
        raw["system"] = raw.pop("arena")
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
            "terraform_state": {"bucket": "test-tfstate"},
            "arena": arena,
            "nebius": nebius,
        }
    )


def _nebius_module_dir() -> Path:
    """Return the packaged Nebius Terraform module directory."""
    return (
        Path(__file__).resolve().parents[2]
        / "agilerl"
        / "arena"
        / "byoc"
        / "provisioning"
        / "terraform_modules"
        / "nebius"
    )


def _terraform_resource_block(resource_type: str, name: str) -> str:
    """Return the body of one resource block in the Nebius module."""
    text = (_nebius_module_dir() / "main.tf").read_text(encoding="utf-8")
    header = f'resource "{resource_type}" "{name}" {{'
    if header not in text:
        msg = f"{resource_type}.{name} is not declared in the Nebius module."
        raise AssertionError(msg)
    return text.split(header, maxsplit=1)[1].split("\n}\n", maxsplit=1)[0]


def _set_nebius_auth_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    """Write dummy auth files and export Nebius Terraform auth vars."""
    key_path = tmp_path / "authorized_key.pem"
    key_path.write_text("key\n", encoding="utf-8")
    nebius_cli = tmp_path / "nebius"
    nebius_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    nebius_cli.chmod(0o755)
    monkeypatch.setenv("AUTHKEY_PRIVATE_PATH", str(key_path))
    monkeypatch.setenv("AUTHKEY_PUBLIC_ID", "publickey-1")
    monkeypatch.setenv("SA_ID", "serviceaccount-1")
    return key_path, nebius_cli


def _state_bucket_run(existing: dict[tuple[str, str], str]) -> Callable[..., MagicMock]:
    """Return a fake Nebius CLI for Terraform state-bucket provisioning.

    ``existing`` maps a service such as ``("iam", "group")`` to the id a previous
    attempt left behind. Anything absent is reported as not found.
    """
    missing = MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
    empty = MagicMock(returncode=0, stdout="{}", stderr="")

    def found(resource_id: str) -> MagicMock:
        return MagicMock(
            returncode=0,
            stdout=json.dumps({"metadata": {"id": resource_id}}),
            stderr="",
        )

    def run(argv: list[str], **_kwargs: object) -> MagicMock:
        command = argv[1:]
        if command[:4] == ["iam", "v2", "access-key", "list-by-account"]:
            return empty
        if command[:4] == ["iam", "v2", "access-key", "create"]:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "status": {
                            "aws_access_key_id": "AKIA-created",
                            "secret": "created-secret",
                        }
                    }
                ),
                stderr="",
            )
        if command[2:3] == ["get-by-name"]:
            resource_id = existing.get((command[0], command[1]))
            return found(resource_id) if resource_id else missing
        if command[2:3] in (["list"], ["list-members"]):
            return empty
        if command[2:3] == ["create"]:
            return found(f"{command[1]}-created")
        if command[2:3] == ["delete"]:
            return empty
        return MagicMock(returncode=1, stdout="", stderr="unexpected")

    return run


@pytest.fixture(autouse=True)
def kubectl_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.shutil.which",
        lambda name: "/usr/bin/kubectl" if name == "kubectl" else None,
    )


@pytest.fixture
def terraform_state_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
        lambda *_args, **_kwargs: {
            "AWS_ACCESS_KEY_ID": "AKIA-tfstate",
            "AWS_SECRET_ACCESS_KEY": "tfstate-secret",
        },
    )
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.experiment_storage_import_ids",
        lambda *_args, **_kwargs: {},
    )


def test_nebius_spec_requires_authorized_key_env_when_sa_id_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service-account Terraform auth requires AUTHKEY env vars when SA_ID is set."""
    monkeypatch.setenv("SA_ID", "serviceaccount-1")
    for name in ("AUTHKEY_PRIVATE_PATH", "AUTHKEY_PUBLIC_ID"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        click.ClickException,
        match="AUTHKEY_PRIVATE_PATH, AUTHKEY_PUBLIC_ID",
    ):
        validate_nebius_spec(_nebius_spec())


def test_nebius_spec_requires_authorized_key_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AUTHKEY_PRIVATE_PATH must point at an existing private-key file."""
    monkeypatch.setenv("AUTHKEY_PRIVATE_PATH", str(tmp_path / "missing.pem"))
    monkeypatch.setenv("AUTHKEY_PUBLIC_ID", "publickey-1")
    monkeypatch.setenv("SA_ID", "serviceaccount-1")

    with pytest.raises(
        click.ClickException, match="AUTHKEY_PRIVATE_PATH is not a file"
    ):
        validate_nebius_spec(_nebius_spec())


def test_nebius_spec_requires_nebius_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nebius provisioning requires the Nebius CLI."""
    _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NEBIUS_CLI", raising=False)
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.shutil.which",
        lambda _name: None,
    )

    with pytest.raises(click.ClickException, match="Nebius CLI not found"):
        validate_nebius_spec(_nebius_spec())


def test_nebius_spec_accepts_nebius_cli_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """NEBIUS_CLI resolves the Nebius CLI when it is not on PATH."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

    validate_nebius_spec(_nebius_spec())

    assert resolve_kubeconfig_executable(["nebius"]) == nebius_cli


def test_nebius_spec_accepts_set_terraform_auth_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nebius Terraform proceeds when provider auth env vars are set."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

    validate_nebius_spec(_nebius_spec())


def test_nebius_spec_allows_missing_sa_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SA_ID is optional when the Nebius CLI profile can authenticate Terraform."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    for name in NEBIUS_PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    validate_nebius_spec(_nebius_spec())


class TestNebiusServiceAccountId:
    def test_defaults_to_unset(self) -> None:
        assert _nebius_spec().nebius.service_account_id is None
        assert _nebius_spec().terraform_values()["service_account_id"] == ""

    def test_accepts_explicit_id(self) -> None:
        spec = _nebius_spec(service_account_id="serviceaccount-explicit")

        assert spec.nebius.service_account_id == "serviceaccount-explicit"
        assert (
            spec.terraform_values()["service_account_id"] == "serviceaccount-explicit"
        )

    def test_blank_id_is_unset(self) -> None:
        assert _nebius_spec(service_account_id="  ").nebius.service_account_id is None


class TestNebiusResourceSelection:
    def test_creates_project_and_subnet_from_tenant_and_region(self) -> None:
        spec = _nebius_spec(project_id=None, subnet_id=None, region=" eu-north1 ")

        assert spec.nebius.project_id is None
        assert spec.nebius.subnet_id is None
        assert spec.terraform_values() == {
            **_nebius_spec().terraform_values(),
            "project_id": "",
            "region": "eu-north1",
            "subnet_id": "",
        }

    def test_creates_both_projects_when_ids_are_omitted(self) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        assert spec.nebius.storage_project_id is None
        assert spec.terraform_values()["project_id"] == ""
        assert spec.terraform_values()["storage_project_id"] == ""

    def test_passes_explicit_storage_project(self) -> None:
        spec = _nebius_spec(storage_project_id=" project-storage ")

        assert spec.nebius.storage_project_id == "project-storage"
        assert spec.terraform_values()["storage_project_id"] == "project-storage"

    def test_requires_region_without_project(self) -> None:
        with pytest.raises(ValueError, match=r"nebius\.region is required"):
            _nebius_spec(project_id=None, subnet_id=None, region=None)

    def test_requires_region_without_storage_project(self) -> None:
        with pytest.raises(ValueError, match=r"nebius\.region is required"):
            _nebius_spec(storage_project_id=None, region=None)

    def test_requires_project_with_subnet(self) -> None:
        with pytest.raises(ValueError, match=r"nebius\.subnet_id requires"):
            _nebius_spec(project_id=None, subnet_id="vpcsubnet-1", region="eu-north1")


class TestNebiusProviderAuthentication:
    def test_resolve_fills_from_sa_id_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SA_ID", "serviceaccount-from-env")

        resolved = resolve_nebius_service_account_id(_nebius_spec())

        assert resolved.nebius.service_account_id == "serviceaccount-from-env"

    def test_spec_id_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SA_ID", "serviceaccount-from-env")

        resolved = resolve_nebius_service_account_id(
            _nebius_spec(service_account_id="serviceaccount-from-spec")
        )

        assert resolved.nebius.service_account_id == "serviceaccount-from-spec"

    def test_provider_config_uses_profile_without_sa_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("SA_ID", raising=False)

        write_nebius_provider_config(tmp_path / "work")

        assert (tmp_path / "work" / "provider.tf").read_text(
            encoding="utf-8"
        ) == NEBIUS_PROVIDER_PROFILE

    def test_provider_config_uses_service_account_with_sa_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SA_ID", "serviceaccount-1")

        write_nebius_provider_config(tmp_path / "work")

        assert (tmp_path / "work" / "provider.tf").read_text(
            encoding="utf-8"
        ) == NEBIUS_PROVIDER_SERVICE_ACCOUNT

    def test_prepare_copies_env_sa_id_into_spec(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir

        ClusterProvisioner(runner=runner).plan(
            _nebius_spec(), state_dir=tmp_path / "state"
        )

        prepared = runner.prepare.call_args.args[0]
        assert prepared.nebius.service_account_id == "serviceaccount-1"
        runner.adopt_resources.assert_called_once_with(work_dir, {})
        assert (work_dir / "provider.tf").read_text(
            encoding="utf-8"
        ) == NEBIUS_PROVIDER_SERVICE_ACCOUNT

    def test_prepare_leaves_service_account_unset_without_sa_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        monkeypatch.delenv("SA_ID", raising=False)
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir

        ClusterProvisioner(runner=runner).plan(
            _nebius_spec(), state_dir=tmp_path / "state"
        )

        prepared = runner.prepare.call_args.args[0]
        assert prepared.nebius.service_account_id is None
        assert (work_dir / "provider.tf").read_text(
            encoding="utf-8"
        ) == NEBIUS_PROVIDER_PROFILE

    def test_prepare_bootstraps_auto_created_projects(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=None,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
                return_value={},
            ),
        ):
            prepared, prepared_dir = ClusterProvisioner(runner=runner)._prepare(
                spec, tmp_path / "state"
            )

        assert prepared_dir == work_dir
        assert prepared.nebius.project_id == "project-created"
        assert prepared.nebius.storage_project_id == "storage-project-created"
        runner.initialize_without_backend.assert_called_once_with(work_dir)
        runner.apply_targets.assert_called_once_with(
            work_dir,
            (
                "nebius_iam_v2_project.arena",
                "nebius_iam_v2_project.storage",
            ),
        )
        runner.terraform_outputs.assert_not_called()
        assert [call.args[1] for call in runner.terraform_output.call_args_list] == [
            "project_id",
            "storage_project_id",
        ]

    def test_prepare_bootstraps_only_the_missing_storage_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(storage_project_id=None, region="eu-north1")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=None,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
                return_value={},
            ),
        ):
            prepared, _ = ClusterProvisioner(runner=runner)._prepare(
                spec, tmp_path / "state"
            )

        assert prepared.nebius.project_id == "project-created"
        assert prepared.nebius.storage_project_id == "storage-project-created"
        runner.apply_targets.assert_called_once_with(
            work_dir, ("nebius_iam_v2_project.storage",)
        )
        runner.terraform_outputs.assert_not_called()
        assert [call.args[1] for call in runner.terraform_output.call_args_list] == [
            "project_id",
            "storage_project_id",
        ]

    def test_prepare_skips_bootstrap_when_both_projects_are_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"

        ClusterProvisioner(runner=runner)._prepare(_nebius_spec(), tmp_path / "state")

        runner.apply_targets.assert_not_called()
        runner.initialize_without_backend.assert_not_called()
        runner.terraform_output.assert_not_called()
        runner.terraform_outputs.assert_not_called()

    def test_bootstrap_runs_before_the_s3_backend_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Terraform rejects every command while an uninitialized backend is on disk.
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        modules_dir = tmp_path / "modules"
        (modules_dir / "nebius").mkdir(parents=True)
        state_dir = tmp_path / "state" / "terraform"
        commands: list[tuple[tuple[str, ...], bool]] = []

        def fake_run(argv: list[str], **_: object) -> MagicMock:
            commands.append((tuple(argv[1:]), (state_dir / "backend.tf").is_file()))
            if argv[1:3] == ["output", "-json"]:
                name = argv[3]
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(_terraform_project_output(state_dir, name)),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="", stderr="")

        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        with (
            patch(
                "agilerl.arena.byoc.provisioning.terraform.shutil.which",
                return_value="/usr/local/bin/terraform",
            ),
            patch(
                "agilerl.arena.byoc.provisioning.terraform.files",
                return_value=modules_dir,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=None,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
                return_value={},
            ),
        ):
            prepared, _ = ClusterProvisioner(
                runner=TerraformRunner(run=fake_run)
            )._prepare(spec, state_dir)

        assert prepared.nebius.project_id == "project-created"
        assert prepared.nebius.storage_project_id == "storage-project-created"
        assert commands == [
            (("init", "-input=false", "-backend=false"), False),
            (
                (
                    "apply",
                    "-input=false",
                    "-auto-approve",
                    "-target=nebius_iam_v2_project.arena",
                    "-target=nebius_iam_v2_project.storage",
                ),
                False,
            ),
            (("output", "-json", "project_id"), False),
            (("output", "-json", "storage_project_id"), False),
        ]

    def test_prepare_bootstraps_when_credentials_exist_but_projects_are_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir
        runner.state_list.return_value = ()
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )
        credentials = TerraformStateCredentials("AKIA-existing", "existing-secret")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=credentials,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
                return_value={},
            ) as ensure_bucket,
        ):
            prepared, _ = ClusterProvisioner(runner=runner)._prepare(
                spec, tmp_path / "state"
            )

        assert prepared.nebius.project_id == "project-created"
        assert prepared.nebius.storage_project_id == "storage-project-created"
        runner.initialize.assert_called_once()
        assert runner.initialize.call_args.args[0] == work_dir
        runner.initialize_without_backend.assert_not_called()
        runner.apply_targets.assert_called_once_with(
            work_dir,
            (
                "nebius_iam_v2_project.arena",
                "nebius_iam_v2_project.storage",
            ),
        )
        ensure_bucket.assert_called_once()
        assert ensure_bucket.call_args.kwargs == {
            "state_dir": tmp_path / "state",
            "create": False,
            "prompt": None,
        }
        assert ensure_bucket.call_args.args[0].nebius.project_id == "project-created"
        assert (
            ensure_bucket.call_args.args[0].nebius.storage_project_id
            == "storage-project-created"
        )
        assert any(
            call.args[0] == credentials.environ()
            for call in runner.extra_env.update.call_args_list
        )

    def test_prepare_skips_bootstrap_when_credentials_exist_and_projects_are_in_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir
        runner.state_list.return_value = (
            "nebius_iam_v2_project.arena",
            "nebius_iam_v2_project.storage",
        )
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )
        credentials = TerraformStateCredentials("AKIA-existing", "existing-secret")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=credentials,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
                return_value={},
            ) as ensure_bucket,
        ):
            prepared, _ = ClusterProvisioner(runner=runner)._prepare(
                spec, tmp_path / "state"
            )

        assert prepared.nebius.project_id == "project-created"
        assert prepared.nebius.storage_project_id == "storage-project-created"
        runner.initialize.assert_called_once()
        assert runner.initialize.call_args.args[0] == work_dir
        runner.apply_targets.assert_not_called()
        runner.initialize_without_backend.assert_not_called()
        ensure_bucket.assert_called_once()
        assert ensure_bucket.call_args.kwargs == {
            "state_dir": tmp_path / "state",
            "create": False,
            "prompt": None,
        }
        assert ensure_bucket.call_args.args[0].nebius.project_id == "project-created"
        assert (
            ensure_bucket.call_args.args[0].nebius.storage_project_id
            == "storage-project-created"
        )

    def test_prepare_bootstraps_only_projects_missing_from_remote_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir
        runner.state_list.return_value = ("nebius_iam_v2_project.arena",)
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )
        credentials = TerraformStateCredentials("AKIA-existing", "existing-secret")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=credentials,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_terraform_state_bucket",
                return_value={},
            ) as ensure_bucket,
        ):
            ClusterProvisioner(runner=runner)._prepare(spec, tmp_path / "state")

        runner.apply_targets.assert_called_once_with(
            work_dir, ("nebius_iam_v2_project.storage",)
        )
        ensure_bucket.assert_called_once()


def test_build_nebius_kubeconfig_argv_uses_cluster_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nebius kubeconfig retrieval uses cluster id, not cluster name."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    kubeconfig_path = tmp_path / "kubeconfig"

    argv = build_nebius_kubeconfig_argv(
        cluster_id="mk8scluster-test",
        context_name="arena-nebius",
        kubeconfig_path=kubeconfig_path,
    )

    assert argv[0] == str(nebius_cli)
    assert "--id" in argv
    assert "mk8scluster-test" in argv
    assert "--name" not in argv
    assert argv[argv.index("--kubeconfig") + 1] == str(kubeconfig_path)
    assert "--force" in argv


def test_nebius_spec_requires_kubectl_when_gateway_api_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gateway API provisioning requires kubectl on PATH."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.shutil.which",
        lambda name: None if name == "kubectl" else str(nebius_cli),
    )

    with pytest.raises(click.ClickException, match="kubectl not found"):
        validate_nebius_spec(_nebius_spec())


def test_nebius_spec_requires_kubectl_with_gateway_api_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The GPU RuntimeClass needs kubectl even when Gateway API is disabled."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.shutil.which",
        lambda _name: None,
    )

    with pytest.raises(click.ClickException, match="kubectl not found"):
        validate_nebius_spec(_nebius_spec(gateway_api={"enable": False}))


def test_build_gpu_runtime_class_manifest() -> None:
    """The RuntimeClass routes GPU pods to the containerd nvidia handler."""
    assert build_gpu_runtime_class_manifest() == {
        "apiVersion": "node.k8s.io/v1",
        "kind": "RuntimeClass",
        "metadata": {"name": "nvidia"},
        "handler": "nvidia",
        "scheduling": {
            "tolerations": [
                {
                    "key": "nvidia.com/gpu",
                    "operator": "Equal",
                    "value": "true",
                    "effect": "NoSchedule",
                }
            ]
        },
    }


def test_ensure_gpu_runtime_class_applies_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The RuntimeClass is applied against the provisioned kubeconfig."""
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.kubectl.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    run = MagicMock(return_value=MagicMock(returncode=0, stderr="", stdout=""))

    ensure_gpu_runtime_class(kubeconfig_path=kubeconfig, run=run)

    run.assert_called_once()
    assert run.call_args.args[0] == ["kubectl", "apply", "-f", "-"]
    assert list(yaml.safe_load_all(run.call_args.kwargs["input"])) == [
        build_gpu_runtime_class_manifest()
    ]
    assert run.call_args.kwargs["env"]["KUBECONFIG"] == str(kubeconfig.resolve())


def test_build_gateway_api_parent_refs() -> None:
    """Parent refs point at the configured Gateway resource."""
    refs = build_gateway_api_parent_refs(
        gateway_name="arena",
        gateway_namespace="arena",
    )

    assert refs == [
        {
            "group": "gateway.networking.k8s.io",
            "kind": "Gateway",
            "name": "arena",
            "namespace": "arena",
            "sectionName": HTTPS_LISTENER_NAME,
        }
    ]


def test_build_cilium_gateway_api_rbac_manifests() -> None:
    """Additive RBAC grants Gateway API access without patching managed roles."""
    manifests = build_cilium_gateway_api_rbac_manifests(
        gateway_namespace="arena",
    )

    cluster_role = manifests[0]
    cluster_role_binding = manifests[1]
    services_role = manifests[2]
    services_role_binding = manifests[3]

    assert cluster_role["kind"] == "ClusterRole"
    assert cluster_role["metadata"]["name"] == CILIUM_GATEWAY_API_CLUSTER_ROLE
    assert any(
        "gateways" in rule["resources"]
        and "listenersets" in rule["resources"]
        and "watch" in rule["verbs"]
        for rule in cluster_role["rules"]
        if rule["apiGroups"] == ["gateway.networking.k8s.io"]
    )
    assert any(
        rule["resources"] == ["configmaps"] and "list" in rule["verbs"]
        for rule in cluster_role["rules"]
        if rule["apiGroups"] == [""]
    )
    assert cluster_role_binding["subjects"][0]["name"] == "cilium-operator"
    assert services_role["metadata"]["namespace"] == "arena"
    assert services_role["metadata"]["name"] == CILIUM_GATEWAY_SERVICES_ROLE
    assert services_role["rules"][0]["resources"] == ["services", "endpoints"]
    assert "create" in services_role["rules"][0]["verbs"]
    assert any(
        rule["resources"] == ["endpointslices"] and "create" in rule["verbs"]
        for rule in services_role["rules"]
        if rule["apiGroups"] == ["discovery.k8s.io"]
    )
    assert services_role_binding["roleRef"]["name"] == CILIUM_GATEWAY_SERVICES_ROLE


def test_enable_cilium_gateway_api_patches_and_restarts_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cilium Gateway API and Envoy config are enabled through cilium-config."""
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
        lambda _name: "/usr/bin/kubectl",
    )
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        calls.append(argv)
        if argv[:2] == ["kubectl", "get"] and "cilium-config" in argv:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    enable_cilium_gateway_api(
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    patch_call = next(call for call in calls if "patch" in call)
    patch_payload = json.loads(patch_call[-1])
    assert patch_payload["data"]["enable-gateway-api"] == "true"
    assert patch_payload["data"]["enable-envoy-config"] == "true"
    assert any("rollout" in call and "restart" in call for call in calls)
    assert any("daemonset/cilium-envoy" in call for call in calls)


def test_enable_cilium_gateway_api_enables_envoy_when_gateway_api_is_on(
    tmp_path: Path,
) -> None:
    """Envoy config is turned on even when Gateway API is already enabled."""
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        calls.append(argv)
        if argv[:2] == ["kubectl", "get"] and "enable-gateway-api" in argv[-1]:
            return MagicMock(returncode=0, stdout="true", stderr="")
        if argv[:2] == ["kubectl", "get"] and "enable-envoy-config" in argv[-1]:
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    enable_cilium_gateway_api(
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    patch_call = next(call for call in calls if "patch" in call)
    patch_payload = json.loads(patch_call[-1])
    assert patch_payload["data"]["enable-envoy-config"] == "true"
    assert any("rollout" in call and "restart" in call for call in calls)


def test_ensure_cilium_gateway_class_uses_cilium_controller(tmp_path: Path) -> None:
    """The GatewayClass is named cilium and owned by the Cilium controller."""
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    captured: dict[str, str] = {}

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        if argv[:3] == ["kubectl", "apply", "-f"]:
            captured["manifest"] = _kwargs["input"]
        return MagicMock(returncode=0, stdout="", stderr="")

    ensure_cilium_gateway_class(
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    manifest = yaml.safe_load(captured["manifest"])
    assert manifest["kind"] == "GatewayClass"
    assert manifest["metadata"]["name"] == CILIUM_GATEWAY_CLASS
    assert manifest["spec"]["controllerName"] == CILIUM_GATEWAY_CONTROLLER


def test_configure_gateway_api_applies_crd_urls_and_returns_parent_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway API setup installs CRDs and returns registration parent refs."""
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
        lambda _name: "/usr/bin/kubectl",
    )
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []
    applied_manifests: list[str] = []

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        calls.append(argv)
        if argv[:2] == ["kubectl", "get"] and "enable-gateway-api" in argv[-1]:
            return MagicMock(returncode=0, stdout="true", stderr="")
        if argv[:2] == ["kubectl", "get"] and "enable-envoy-config" in argv[-1]:
            return MagicMock(returncode=0, stdout="true", stderr="")
        if argv[:2] == ["kubectl", "get"] and argv[2] == "crd":
            return MagicMock(returncode=0, stdout="", stderr="")
        if argv[:3] == ["kubectl", "apply", "-f"] and argv[-1] == "-":
            applied_manifests.append(_kwargs["input"])
        return MagicMock(returncode=0, stdout="", stderr="")

    refs = configure_gateway_api(
        kubeconfig_path=kubeconfig,
        gateway_name="arena",
        domain="inference.example.com",
        tls_secret_name=None,
        run=MagicMock(side_effect=run_side_effect),
    )

    assert refs[0]["name"] == "arena"
    applied_urls = [
        call[-1]
        for call in calls
        if call[:2] == ["kubectl", "apply"] and call[-1] != "-"
    ]
    expected_crds = [*GATEWAY_API_STANDARD_CRDS, *GATEWAY_API_EXPERIMENTAL_CRDS]
    assert applied_urls[: len(expected_crds)] == expected_crds
    rbac_documents = [
        document
        for payload in applied_manifests
        for document in yaml.safe_load_all(payload)
        if document is not None
        and document["kind"] == "ClusterRole"
        and document["metadata"]["name"] == CILIUM_GATEWAY_API_CLUSTER_ROLE
    ]
    gateway_classes = [
        document
        for payload in applied_manifests
        for document in yaml.safe_load_all(payload)
        if document is not None and document["kind"] == "GatewayClass"
    ]
    assert len(rbac_documents) == 1
    assert gateway_classes[0]["metadata"]["name"] == CILIUM_GATEWAY_CLASS
    assert gateway_classes[0]["spec"]["controllerName"] == CILIUM_GATEWAY_CONTROLLER
    redirect_routes = [
        document
        for payload in applied_manifests
        for document in yaml.safe_load_all(payload)
        if document is not None and document["kind"] == "HTTPRoute"
    ]
    assert redirect_routes[0]["metadata"]["name"] == HTTPS_REDIRECT_ROUTE_NAME
    assert (
        redirect_routes[0]["spec"]["parentRefs"][0]["sectionName"] == HTTP_LISTENER_NAME
    )
    assert (
        redirect_routes[0]["spec"]["rules"][0]["filters"][0]["type"]
        == "RequestRedirect"
    )


def test_ensure_cilium_gateway_uses_wildcard_http_and_https_listeners(
    tmp_path: Path,
) -> None:
    """The default Gateway exposes wildcard listeners for the inference domain."""
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    captured: dict[str, str] = {}

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        if argv[:3] == ["kubectl", "apply", "-f"]:
            captured["manifest"] = _kwargs["input"]
        return MagicMock(returncode=0, stdout="", stderr="")

    ensure_cilium_gateway(
        gateway_name="arena",
        gateway_namespace="arena",
        domain="inference.example.com",
        tls_secret_name="arena-inference-tls",
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    manifest = yaml.safe_load(captured["manifest"])
    listeners = manifest["spec"]["listeners"]
    assert listeners[0]["name"] == "http"
    assert listeners[0]["hostname"] == "*.inference.example.com"
    assert listeners[0]["protocol"] == "HTTP"
    assert listeners[1]["name"] == "https"
    assert listeners[1]["hostname"] == "*.inference.example.com"
    assert listeners[1]["protocol"] == "HTTPS"
    assert listeners[1]["tls"]["certificateRefs"][0]["name"] == "arena-inference-tls"
    assert manifest["metadata"]["namespace"] == "arena"


def test_ensure_https_redirect_route_sends_http_to_https(tmp_path: Path) -> None:
    """HTTP listener traffic is redirected to HTTPS with a 301."""
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    captured: dict[str, str] = {}

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        if argv[:3] == ["kubectl", "apply", "-f"]:
            captured["manifest"] = _kwargs["input"]
        return MagicMock(returncode=0, stdout="", stderr="")

    ensure_https_redirect_route(
        gateway_name="arena",
        gateway_namespace="arena",
        domain="inference.example.com",
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    manifest = yaml.safe_load(captured["manifest"])
    assert manifest["kind"] == "HTTPRoute"
    assert manifest["metadata"]["name"] == HTTPS_REDIRECT_ROUTE_NAME
    assert manifest["spec"]["hostnames"] == ["*.inference.example.com"]
    parent = manifest["spec"]["parentRefs"][0]
    assert parent["name"] == "arena"
    assert parent["sectionName"] == HTTP_LISTENER_NAME
    redirect = manifest["spec"]["rules"][0]["filters"][0]
    assert redirect["type"] == "RequestRedirect"
    assert redirect["requestRedirect"]["scheme"] == "https"
    assert redirect["requestRedirect"]["statusCode"] == 301


def test_ensure_inference_tls_secret_uses_existing_secret(tmp_path: Path) -> None:
    """A named TLS secret is used when it already exists."""
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        calls.append(argv)
        return MagicMock(returncode=0, stdout="", stderr="")

    name = ensure_inference_tls_secret(
        tls_secret_name="existing-tls",
        domain="inference.example.com",
        gateway_namespace="arena",
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    assert name == "existing-tls"
    assert any(call[:3] == ["kubectl", "get", "secret"] for call in calls)
    assert not any(call[:1] == ["openssl"] for call in calls)
    assert not any("create" in call and "secret" in call for call in calls)


def test_ensure_inference_tls_secret_creates_self_signed_for_named_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A named TLS secret is created as self-signed when it is missing."""
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
        lambda name: "/usr/bin/openssl" if name == "openssl" else None,
    )
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []
    openssl_config: dict[str, str] = {}

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        calls.append(argv)
        if argv[:3] == ["kubectl", "get", "secret"]:
            return MagicMock(returncode=1, stdout="", stderr="NotFound")
        if argv[:1] == ["openssl"]:
            openssl_config["text"] = Path(argv[argv.index("-config") + 1]).read_text(
                encoding="utf-8"
            )
            Path(argv[argv.index("-keyout") + 1]).write_bytes(b"key")
            Path(argv[argv.index("-out") + 1]).write_bytes(b"cert")
        return MagicMock(returncode=0, stdout="", stderr="")

    name = ensure_inference_tls_secret(
        tls_secret_name="star.inference.example.com-tls",
        domain="inference.example.com",
        gateway_namespace="arena",
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    assert name == "star.inference.example.com-tls"
    assert "DNS:*.inference.example.com" in openssl_config["text"]
    create_call = next(
        call for call in calls if call[:4] == ["kubectl", "create", "secret", "tls"]
    )
    assert "star.inference.example.com-tls" in create_call


def test_ensure_inference_tls_secret_creates_self_signed_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-signed TLS secret is created when none is specified."""
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
        lambda name: "/usr/bin/openssl" if name == "openssl" else None,
    )
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
    calls: list[list[str]] = []
    openssl_config: dict[str, str] = {}

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        calls.append(argv)
        if argv[:3] == ["kubectl", "get", "secret"]:
            return MagicMock(returncode=1, stdout="", stderr="NotFound")
        if argv[:1] == ["openssl"]:
            openssl_config["text"] = Path(argv[argv.index("-config") + 1]).read_text(
                encoding="utf-8"
            )
            Path(argv[argv.index("-keyout") + 1]).write_bytes(b"key")
            Path(argv[argv.index("-out") + 1]).write_bytes(b"cert")
        return MagicMock(returncode=0, stdout="", stderr="")

    name = ensure_inference_tls_secret(
        tls_secret_name=None,
        domain="inference.example.com",
        gateway_namespace="arena",
        env={"KUBECONFIG": str(kubeconfig)},
        run=MagicMock(side_effect=run_side_effect),
    )

    assert name == DEFAULT_INFERENCE_TLS_SECRET_NAME
    assert "DNS:*.inference.example.com" in openssl_config["text"]
    create_call = next(
        call for call in calls if call[:4] == ["kubectl", "create", "secret", "tls"]
    )
    assert DEFAULT_INFERENCE_TLS_SECRET_NAME in create_call


def test_nebius_spec_accepts_existing_inference_tls_secret(tmp_path: Path) -> None:
    """inference.tls_secret_name is accepted on the Nebius spec."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            "    domain: inference.example.com\n",
            "    domain: inference.example.com\n    tls_secret_name: existing-tls\n",
        ),
        encoding="utf-8",
    )

    spec = load_cluster_spec(spec_path)

    assert spec.arena.inference.tls_secret_name == "existing-tls"


def test_cluster_provisioner_configures_gateway_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terraform_state_ready: None,
) -> None:
    """Provisioning returns Gateway parent refs after Terraform apply."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    spec = _nebius_spec()
    outputs = ClusterOutputs(
        cluster_name="arena-nebius",
        context="arena-nebius",
        kubeconfig_path=tmp_path / "kubeconfig",
        storage_access_key_id="AKIA-test",
        storage_secret_access_key="secret-test",
        worker_node_group_ids={"workers": "mk8snodegroup-test"},
    )
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_storage_secret",
        MagicMock(),
    )
    runner = MagicMock()
    runner.prepare.return_value = tmp_path / "work"
    monkeypatch.setattr(NEBIUS, "cluster_outputs", lambda *_args, **_kwargs: outputs)
    parent_refs = build_gateway_api_parent_refs(
        gateway_name="arena",
        gateway_namespace="arena",
    )
    configure_mock = MagicMock(return_value=parent_refs)
    runtime_class_mock = MagicMock()
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.configure_gateway_api",
        configure_mock,
    )
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_gpu_runtime_class",
        runtime_class_mock,
    )

    result = ClusterProvisioner(runner=runner).provision(
        spec,
        state_dir=tmp_path / "state",
        output_dir=tmp_path / "output",
    )

    configure_mock.assert_called_once_with(
        kubeconfig_path=outputs.kubeconfig_path,
        gateway_name="arena",
        domain="inference.example.com",
        tls_secret_name=None,
    )
    runtime_class_mock.assert_called_once_with(kubeconfig_path=outputs.kubeconfig_path)
    assert result.gateway_api_parent_refs == parent_refs
    assert result.inference_domain == "inference.example.com"
    assert result.inference_tls_secret_name == DEFAULT_INFERENCE_TLS_SECRET_NAME
    assert result.inference_hostname_template == "inference-{deploymentId}"


class TestEnsureStorageSecret:
    def test_applies_namespace_and_credentials_secret(self, tmp_path: Path) -> None:
        # Arrange
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        applied: list[list[dict[str, object]]] = []

        def run_side_effect(argv: list[str], **kwargs: object) -> MagicMock:
            applied.append(list(yaml.safe_load_all(str(kwargs["input"]))))
            assert argv == ["kubectl", "apply", "-f", "-"]
            return MagicMock(returncode=0, stderr="")

        # Act
        ensure_storage_secret(
            kubeconfig_path=kubeconfig,
            secret_name="storage",
            access_key_id="AKIA-nebius",
            secret_access_key="nebius-secret",
            endpoint="https://storage.eu-north1.nebius.cloud",
            run=MagicMock(side_effect=run_side_effect),
        )

        # Assert
        namespace, secret = applied[0][0], applied[1][0]
        assert namespace["kind"] == "Namespace"
        assert namespace["metadata"] == {"name": "arena"}
        assert secret["metadata"] == {"name": "storage", "namespace": "arena"}
        assert secret["stringData"] == {
            "AWS_ACCESS_KEY_ID": "AKIA-nebius",
            "AWS_SECRET_ACCESS_KEY": "nebius-secret",
            "endpoint": "https://storage.eu-north1.nebius.cloud",
            "AWS_ENDPOINT_URL": "https://storage.eu-north1.nebius.cloud",
            "APP_AWS_CLIENT_ENDPOINT": "https://storage.eu-north1.nebius.cloud",
        }

    def test_sets_aws_region_from_the_s3_endpoint(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        applied: list[list[dict[str, object]]] = []

        def run_side_effect(argv: list[str], **kwargs: object) -> MagicMock:
            applied.append(list(yaml.safe_load_all(str(kwargs["input"]))))
            assert argv == ["kubectl", "apply", "-f", "-"]
            return MagicMock(returncode=0, stderr="")

        ensure_storage_secret(
            kubeconfig_path=kubeconfig,
            secret_name="storage",
            access_key_id="AKIA-aws",
            secret_access_key="aws-secret",
            endpoint="https://s3.eu-west-1.amazonaws.com",
            run=MagicMock(side_effect=run_side_effect),
        )

        secret = applied[1][0]
        assert secret["stringData"]["AWS_REGION"] == "eu-west-1"
        assert secret["stringData"]["AWS_DEFAULT_REGION"] == "eu-west-1"

    def test_reports_failed_apply(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        run = MagicMock(return_value=MagicMock(returncode=1, stderr="forbidden"))

        with pytest.raises(click.ClickException, match="forbidden"):
            ensure_storage_secret(
                kubeconfig_path=kubeconfig,
                secret_name="storage",
                access_key_id="AKIA-nebius",
                secret_access_key="nebius-secret",
                endpoint="https://storage.example.com",
                run=run,
            )


class TestAwsRegionFromS3Endpoint:
    """AWS region encoded in an S3 API endpoint host."""

    @pytest.mark.parametrize(
        ("endpoint", "region"),
        [
            ("https://s3.eu-west-1.amazonaws.com", "eu-west-1"),
            ("https://s3.dualstack.eu-west-1.amazonaws.com", "eu-west-1"),
            ("https://eks-byoc-wei-data.s3.eu-west-1.amazonaws.com", "eu-west-1"),
            ("https://s3.amazonaws.com", None),
            ("https://storage.eu-north1.nebius.cloud", None),
        ],
    )
    def test_reads_the_region_from_an_aws_s3_host(
        self, endpoint: str, region: str | None
    ) -> None:
        assert aws_region_from_s3_endpoint(endpoint) == region


class TestClusterProvisionerStorageSecret:
    def _provision(
        self,
        spec: ClusterSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> MagicMock:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"
        outputs = ClusterOutputs(
            cluster_name="arena-nebius",
            context="arena-nebius",
            kubeconfig_path=tmp_path / "kubeconfig",
            storage_access_key_id="AKIA-nebius",
            storage_secret_access_key="nebius-secret",
            worker_node_group_ids={"workers": "mk8snodegroup-test"},
        )
        monkeypatch.setattr(
            NEBIUS, "cluster_outputs", lambda *_args, **_kwargs: outputs
        )
        secret_mock = MagicMock()
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_storage_secret",
            secret_mock,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.configure_gateway_api",
            MagicMock(return_value=[]),
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_gpu_runtime_class",
            MagicMock(),
        )
        ClusterProvisioner(runner=runner).provision(
            spec,
            state_dir=tmp_path / "state",
            output_dir=tmp_path / "output",
        )
        return secret_mock

    def test_creates_secret_with_resolved_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        # Act
        secret_mock = self._provision(_nebius_spec(), monkeypatch, tmp_path)

        # Assert
        secret_mock.assert_called_once_with(
            kubeconfig_path=tmp_path / "kubeconfig",
            secret_name=DEFAULT_STORAGE_SECRET_NAME,
            access_key_id="AKIA-nebius",
            secret_access_key="nebius-secret",
            endpoint="https://storage.example.com",
        )

    def test_skips_secret_when_minio_chart_installs_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        spec = _nebius_spec()
        spec = spec.model_copy(
            update={
                "arena": spec.arena.model_copy(
                    update={
                        "storage": spec.arena.storage.model_copy(
                            update={"install": True}
                        )
                    }
                )
            }
        )

        secret_mock = self._provision(spec, monkeypatch, tmp_path)

        secret_mock.assert_not_called()


def test_nebius_spec_uses_node_and_storage_defaults(tmp_path: Path) -> None:
    """Nebius specs provide the default node groups and storage resources."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)

    spec = load_cluster_spec(spec_path)

    assert spec.nebius.system.min_node_count == 2
    assert spec.nebius.system.max_node_count == 10
    assert spec.nebius.system.platform == "cpu-e2"
    assert spec.nebius.system.preset == "2vcpu-8gb"
    assert len(spec.nebius.workers) == 1
    assert spec.nebius.workers[0].name == "workers"
    assert spec.nebius.workers[0].min_node_count == 0
    assert spec.nebius.workers[0].max_node_count == 10
    assert spec.nebius.workers[0].platform == "gpu-h100-sxm"
    assert spec.nebius.workers[0].preset == "1gpu-16vcpu-200gb"
    assert spec.nebius.workers[0].gpu_drivers_preset == "cuda13.0"
    assert spec.nebius.shared_filesystem.provision is True
    assert spec.nebius.shared_filesystem.size_gib == 256
    assert spec.arena.gateway.enable is True
    assert spec.arena.gateway.name == "arena"
    assert spec.nebius.service_account_id is None
    assert spec.arena.inference.domain == "inference.example.com"
    assert spec.arena.inference.tls_secret_name is None
    assert (
        resolve_inference_hostname_template(spec.arena.inference.hostname_template)
        == "inference-{deploymentId}"
    )


class TestParseNebiusGpuPreset:
    def test_parses_default_worker_preset(self) -> None:
        assert parse_nebius_gpu_preset("1gpu-16vcpu-200gb") == (16, 1, 200)

    def test_rejects_cpu_preset(self) -> None:
        with pytest.raises(ValueError, match="1gpu-16vcpu-200gb"):
            parse_nebius_gpu_preset("2vcpu-8gb")


class TestNebiusGpuVramGib:
    def test_returns_vram_for_h100_platform(self) -> None:
        assert nebius_gpu_vram_gib("gpu-h100-sxm") == 80

    def test_returns_vram_for_l40s_platform(self) -> None:
        assert nebius_gpu_vram_gib("gpu-l40s-d") == 48

    def test_rejects_unknown_platform(self) -> None:
        with pytest.raises(ValueError, match="Unknown Nebius GPU platform 'cpu-e2'"):
            nebius_gpu_vram_gib("cpu-e2")


class TestArena:
    def test_rejects_head_key(self) -> None:
        with pytest.raises(ValueError, match=r"nebius\.system"):
            _nebius_spec(head={"min_node_count": 2})

    def test_rejects_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError, match="min_node_count"):
            _nebius_spec(arena={"min_node_count": 5, "max_node_count": 2})


class TestWorkers:
    def test_rejects_singular_worker_field(self) -> None:
        with pytest.raises(ValueError, match=r"nebius\.workers must be a list"):
            _nebius_spec(worker={"node_count": 1})

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            _nebius_spec(
                workers=[
                    {"name": "gpu", "min_node_count": 1},
                    {"name": "gpu", "min_node_count": 2},
                ]
            )

    def test_rejects_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError, match="min_node_count"):
            _nebius_spec(workers=[{"min_node_count": 5, "max_node_count": 2}])

    def test_rejects_blank_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _nebius_spec(workers=[{"name": "  "}])

    def test_accepts_named_groups(self) -> None:
        spec = _nebius_spec(
            workers=[
                {"name": "h100", "platform": "gpu-h100-sxm"},
                {
                    "name": "l40s",
                    "platform": "gpu-l40s-d",
                    "preset": "2gpu-32vcpu-200gb",
                },
            ]
        )

        assert [worker.name for worker in spec.nebius.workers] == ["h100", "l40s"]

    def test_resource_classes_use_worker_names(self) -> None:
        spec = _nebius_spec(
            workers=[
                {"name": "h100", "platform": "gpu-h100-sxm"},
                {
                    "name": "l40s",
                    "platform": "gpu-l40s-d",
                    "preset": "2gpu-32vcpu-200gb",
                },
            ]
        )

        classes = resource_classes(
            spec,
            worker_node_group_ids={
                "h100": "mk8snodegroup-h100",
                "l40s": "mk8snodegroup-l40s",
            },
        )

        assert [item.name for item in classes] == ["test-h100", "test-l40s"]
        assert classes[0].node_selector == {
            NEBIUS_WORKER_NODE_GROUP_LABEL: "mk8snodegroup-h100"
        }
        assert classes[1].metadata["computeResource"]["numGpus"] == 2
        assert classes[1].metadata["computeResource"]["gramPerGpu"] == 48
        assert classes[0].metadata["computeResource"]["gpu"] == {
            "type": "gpu-h100-sxm",
            "count": 1,
            "driverVersion": "default",
        }
        assert classes[1].metadata["computeResource"]["gpu"] == {
            "type": "gpu-l40s-d",
            "count": 2,
            "driverVersion": "default",
        }


def test_cluster_spec_resolves_registration_settings(tmp_path: Path) -> None:
    """Registration uses explicit spec settings and provisioned storage defaults."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    text = spec_path.read_text(encoding="utf-8")
    text = text.replace(
        "    endpoint: https://storage.example.com\n",
        "    endpoint: https://storage.example.com\n"
        "    prefix: models/\n"
        "    secret_name: inference-storage\n",
    )
    text = text.replace(
        "    domain: inference.example.com\n",
        "    domain: inference.example.com\n"
        "  workloads:\n"
        "    ray_data_pvc_size: 20Gi\n",
    )
    spec_path.write_text(text, encoding="utf-8")

    arena = load_cluster_spec(spec_path).arena

    assert arena.storage.endpoint == "https://storage.example.com"
    assert arena.storage.bucket == "arena-data"
    assert arena.storage.prefix == "models/"
    assert arena.storage.resolved_secret_name() == "inference-storage"
    assert arena.storage.install is False
    assert arena.inference.domain == "inference.example.com"
    assert arena.workloads.ray_data_pvc_size == "20Gi"


def test_cluster_spec_cli_overrides_take_precedence(tmp_path: Path) -> None:
    """Explicit CLI values override settings from the cluster spec."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    spec = apply_cluster_spec_overrides(
        load_cluster_spec(spec_path),
        name="override-name",
        storage_bucket="override-bucket",
        domain="override.example.com",
    )

    assert spec.name == "override-name"
    assert spec.arena.storage.bucket == "override-bucket"
    assert spec.arena.inference.domain == "override.example.com"


class TestClusterSpecStorageSecretDefaults:
    def test_defaults_to_storage_without_minio(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)

        storage = load_cluster_spec(spec_path).arena.storage

        assert storage.install is False
        assert storage.resolved_secret_name() == DEFAULT_STORAGE_SECRET_NAME

    def test_defaults_to_arena_storage_with_minio(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        spec_path.write_text(
            spec_path.read_text(encoding="utf-8").replace(
                "    endpoint: https://storage.example.com\n",
                "    endpoint: https://storage.example.com\n    install: true\n",
            ),
            encoding="utf-8",
        )

        storage = load_cluster_spec(spec_path).arena.storage

        assert storage.install is True
        assert storage.resolved_secret_name() == DEFAULT_LAB_STORAGE_SECRET_NAME

    def test_explicit_secret_name_is_kept(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        spec_path.write_text(
            spec_path.read_text(encoding="utf-8").replace(
                "    endpoint: https://storage.example.com\n",
                "    endpoint: https://storage.example.com\n"
                "    install: true\n"
                "    secret_name: corp-s3\n",
            ),
            encoding="utf-8",
        )

        assert load_cluster_spec(spec_path).arena.storage.resolved_secret_name() == (
            "corp-s3"
        )


def test_cluster_spec_rejects_registration_section(tmp_path: Path) -> None:
    """The registration section is not part of the cluster spec."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8") + "registration: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="registration are not accepted"):
        load_cluster_spec(spec_path)


def test_terraform_runner_writes_custom_resource_settings(tmp_path: Path) -> None:
    """Terraform variables contain the configured Nebius resource settings."""
    spec = ClusterSpec.model_validate(
        {
            "provider": "nebius",
            "name": "arena-nebius",
            "terraform_state": {"bucket": "arena-nebius-tfstate"},
            "arena": {
                "storage": {
                    "bucket": "bucket",
                    "endpoint": "https://storage.example.com",
                },
                "inference": {"domain": "inference.example.com"},
            },
            "nebius": {
                "tenant_id": "tenant",
                "project_id": "project",
                "storage_project_id": "storage-project",
                "subnet_id": "subnet",
                "object_storage": {"size_gib": 1024},
                "system": {
                    "min_node_count": 3,
                    "max_node_count": 8,
                    "preset": "8vcpu-32gb",
                },
                "workers": [
                    {
                        "name": "l40s",
                        "min_node_count": 0,
                        "max_node_count": 4,
                        "platform": "gpu-l40s",
                        "preset": "2gpu-32vcpu-200gb",
                        "gpu_drivers_preset": "cuda12.8",
                    },
                    {
                        "name": "h100",
                        "min_node_count": 0,
                        "max_node_count": 2,
                        "platform": "gpu-h100-sxm",
                        "preset": "1gpu-16vcpu-200gb",
                        "gpu_drivers_preset": "cuda13.0",
                    },
                ],
                "shared_filesystem": {
                    "provision": False,
                    "type": "NETWORK_HDD",
                    "size_gib": 512,
                    "mount_tag": "shared-data",
                },
            },
        }
    )
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    runner = TerraformRunner()

    with (
        patch(
            "agilerl.arena.byoc.provisioning.terraform.shutil.which",
            return_value="/usr/local/bin/terraform",
        ),
        patch(
            "agilerl.arena.byoc.provisioning.terraform.files",
            return_value=tmp_path,
        ),
    ):
        work_dir = runner.prepare(
            spec, module_name="module", state_dir=tmp_path / "state"
        )

    tfvars = json.loads(
        (work_dir / "terraform.tfvars.json").read_text(encoding="utf-8")
    )
    assert "kubeconfig_command" not in tfvars
    assert tfvars["service_account_id"] == ""
    assert tfvars["arena_min_node_count"] == 3
    assert tfvars["arena_max_node_count"] == 8
    assert tfvars["arena_preset"] == "8vcpu-32gb"
    assert tfvars["workers"] == [
        {
            "name": "l40s",
            "min_node_count": 0,
            "max_node_count": 4,
            "platform": "gpu-l40s",
            "preset": "2gpu-32vcpu-200gb",
            "gpu_drivers_preset": "cuda12.8",
        },
        {
            "name": "h100",
            "min_node_count": 0,
            "max_node_count": 2,
            "platform": "gpu-h100-sxm",
            "preset": "1gpu-16vcpu-200gb",
            "gpu_drivers_preset": "cuda13.0",
        },
    ]
    assert tfvars["object_storage_size_gib"] == 1024
    assert tfvars["provision_shared_filesystem"] is False
    assert tfvars["filesystem_size_gib"] == 512
    assert tfvars["filesystem_mount_tag"] == "shared-data"
    assert not (work_dir / "backend.tf").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"system": {"min_node_count": 0}},
        {"system": {"max_node_count": 0}},
        {"workers": [{"max_node_count": 0}]},
        {"object_storage": {"size_gib": 0}},
        {"shared_filesystem": {"size_gib": 0}},
    ],
)
def test_nebius_spec_rejects_non_positive_resource_sizes(
    overrides: dict[str, object],
) -> None:
    """Nebius specs reject non-positive node counts and storage sizes."""
    with pytest.raises(ValueError, match="greater than 0"):
        _nebius_spec(**overrides)


def test_terraform_outputs_writes_kubeconfig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Terraform output values create a local kubeconfig file."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    output = {
        "cluster_name": {"value": "arena-nebius"},
        "context": {"value": "arena-nebius"},
        "cluster_id": {"value": "mk8scluster-test"},
        "storage_class_name": {"value": "arena-shared"},
        "storage_endpoint": {"value": "https://storage.example.com"},
        "storage_bucket": {"value": "arena-data"},
        "storage_access_key_id": {"value": "AKIA-nebius"},
        "storage_secret_access_key": {"value": "nebius-secret"},
        "worker_node_group_ids": {"value": {"workers": "mk8snodegroup-test"}},
    }
    terraform_result = MagicMock(returncode=0, stdout=json.dumps(output), stderr="")
    kubeconfig_result = MagicMock(returncode=0)

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        if argv[0] == "terraform":
            return terraform_result
        kubeconfig_path = Path(argv[argv.index("--kubeconfig") + 1])
        kubeconfig_path.write_text("apiVersion: v1\n", encoding="utf-8")
        return kubeconfig_result

    runner = TerraformRunner(run=MagicMock(side_effect=run_side_effect))

    outputs = materialize_outputs(runner, tmp_path, tmp_path / "output")

    assert outputs.cluster_name == "arena-nebius"
    assert outputs.kubeconfig_path.is_file()
    assert outputs.kubeconfig_path.read_text(encoding="utf-8") == "apiVersion: v1\n"
    assert outputs.storage_access_key_id == "AKIA-nebius"
    assert outputs.storage_secret_access_key == "nebius-secret"
    assert outputs.worker_node_group_ids == {"workers": "mk8snodegroup-test"}


def test_terraform_outputs_writes_kubeconfig_when_cwd_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Kubeconfig is written when getcwd() fails after Terraform apply."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    monkeypatch.setenv("PWD", str(tmp_path))
    output = {
        "cluster_name": {"value": "arena-nebius"},
        "context": {"value": "arena-nebius"},
        "cluster_id": {"value": "mk8scluster-test"},
        "storage_access_key_id": {"value": "AKIA-nebius"},
        "storage_secret_access_key": {"value": "nebius-secret"},
        "worker_node_group_ids": {"value": {"workers": "mk8snodegroup-test"}},
    }
    terraform_result = MagicMock(returncode=0, stdout=json.dumps(output), stderr="")
    kubeconfig_result = MagicMock(returncode=0)

    def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
        if argv[0] == "terraform":
            return terraform_result
        kubeconfig_path = Path(argv[argv.index("--kubeconfig") + 1])
        kubeconfig_path.write_text("apiVersion: v1\n", encoding="utf-8")
        return kubeconfig_result

    runner = TerraformRunner(run=MagicMock(side_effect=run_side_effect))

    def cwd_removed() -> Path:
        raise FileNotFoundError

    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.terraform.Path.cwd",
        cwd_removed,
    )

    outputs = materialize_outputs(runner, tmp_path, Path("arena-cluster"))

    assert (
        outputs.kubeconfig_path == (tmp_path / "arena-cluster" / "kubeconfig").resolve()
    )
    assert outputs.kubeconfig_path.is_file()


def test_terraform_output_reads_one_named_value(tmp_path: Path) -> None:
    run = MagicMock(
        return_value=MagicMock(
            returncode=0, stdout=json.dumps("project-created"), stderr=""
        )
    )
    runner = TerraformRunner(run=run)

    value = runner.terraform_output(tmp_path, "project_id")

    assert value == "project-created"
    assert run.call_args.args[0] == [
        "terraform",
        "output",
        "-json",
        "project_id",
    ]


def test_terraform_outputs_requires_storage_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Terraform state without object storage credentials fails before kubeconfig."""
    _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
    output = {
        "cluster_name": {"value": "arena-nebius"},
        "context": {"value": "arena-nebius"},
        "cluster_id": {"value": "mk8scluster-test"},
    }
    terraform_result = MagicMock(returncode=0, stdout=json.dumps(output), stderr="")
    runner = TerraformRunner(run=MagicMock(return_value=terraform_result))

    with pytest.raises(click.ClickException, match="storage_access_key_id"):
        materialize_outputs(runner, tmp_path, tmp_path / "output")


def test_terraform_outputs_reports_missing_nebius_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing Nebius CLI binaries surface a clear CLI error."""
    _set_nebius_auth_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NEBIUS_CLI", raising=False)
    monkeypatch.setattr(
        "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.shutil.which",
        lambda _name: None,
    )
    output = {
        "cluster_name": {"value": "arena-nebius"},
        "context": {"value": "arena-nebius"},
        "cluster_id": {"value": "mk8scluster-test"},
        "storage_access_key_id": {"value": "AKIA-nebius"},
        "storage_secret_access_key": {"value": "nebius-secret"},
        "worker_node_group_ids": {"value": {"workers": "mk8snodegroup-test"}},
    }
    terraform_result = MagicMock(returncode=0, stdout=json.dumps(output), stderr="")
    runner = TerraformRunner(run=MagicMock(return_value=terraform_result))

    with pytest.raises(click.ClickException, match="Nebius CLI not found"):
        materialize_outputs(runner, tmp_path, tmp_path / "output")


def test_render_default_cluster_spec_is_loadable() -> None:
    """The generated Nebius spec is a valid cluster spec with the requested name."""
    yaml_text = render_default_cluster_spec(provider="nebius", name="demo-cluster")

    spec = ClusterSpec.model_validate(yaml.safe_load(yaml_text))

    assert spec.provider == "nebius"
    assert spec.name == "demo-cluster"
    assert spec.arena.storage.bucket == "demo-cluster-data"
    assert spec.terraform_state.bucket == "demo-cluster-tfstate"
    assert spec.arena.inference.domain == "demo-cluster.inference.agilerl.rlops.ai"
    assert "__CLUSTER_NAME__" not in yaml_text
    assert "inference-{deploymentId}" in yaml_text


def test_write_default_cluster_spec_refuses_existing_file(tmp_path: Path) -> None:
    """Writing a spec does not overwrite an existing file unless forced."""
    spec_path = tmp_path / "cluster.yaml"
    spec_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        write_default_cluster_spec(spec_path, provider="nebius", name="demo-cluster")

    written = write_default_cluster_spec(
        spec_path, provider="nebius", name="demo-cluster", force=True
    )

    assert written == spec_path
    assert load_cluster_spec(spec_path).name == "demo-cluster"


def test_generate_spec_command_writes_yaml(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    """generate-spec writes a default cluster YAML to the requested path."""
    spec_path = tmp_path / "out" / "nebius.yaml"

    result = CliRunner().invoke(
        build_cluster_generate_spec_command(),
        ["--name", "demo-cluster", "--output", str(spec_path)],
        obj=command_config,
    )

    assert result.exit_code == 0, result.output
    assert spec_path.is_file()
    assert load_cluster_spec(spec_path).name == "demo-cluster"
    assert f"Wrote cluster spec to {spec_path}" in result.output


def test_generate_spec_command_defaults_name_to_arena_provider(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    """The default cluster name is arena-{provider}."""
    result = CliRunner().invoke(
        build_cluster_generate_spec_command(),
        ["--output", str(tmp_path / "spec.yaml")],
        obj=command_config,
    )

    assert result.exit_code == 0, result.output
    assert load_cluster_spec(tmp_path / "spec.yaml").name == "arena-nebius"


def test_plan_command_runs_without_confirmation(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    """Planning cloud infrastructure does not prompt or apply."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    with patch("agilerl.arena.byoc.commands.ClusterProvisioner.plan") as plan_mock:
        result = CliRunner().invoke(
            build_cluster_plan_command(), ["--spec", str(spec_path)], obj=command_config
        )

    assert result.exit_code == 0, result.output
    plan_mock.assert_called_once()
    assert plan_mock.call_args.kwargs["create_state_bucket"] is False


def test_plan_creates_state_bucket_when_spec_asks(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    text = spec_path.read_text(encoding="utf-8").replace(
        "  bucket: arena-nebius-tfstate\n",
        "  bucket: arena-nebius-tfstate\n  create_bucket: true\n",
    )
    spec_path.write_text(text, encoding="utf-8")
    with patch("agilerl.arena.byoc.commands.ClusterProvisioner.plan") as plan_mock:
        result = CliRunner().invoke(
            build_cluster_plan_command(),
            ["--spec", str(spec_path)],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert plan_mock.call_args.kwargs["create_state_bucket"] is True


def test_provision_requires_confirmation(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    """Provisioning does not apply if the user declines confirmation."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    with patch(
        "agilerl.arena.byoc.commands.ClusterProvisioner.provision"
    ) as provision_mock:
        result = CliRunner().invoke(
            build_cluster_provision_command(),
            ["nebius", "--spec", str(spec_path)],
            obj=command_config,
            input="n\n",
        )

    assert result.exit_code == 0, result.output
    assert "Aborted." in result.output
    provision_mock.assert_not_called()


def test_provision_prints_registration_command(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    """Provisioning prints the Helm registration handoff by default."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    outputs = ClusterOutputs(
        cluster_name="arena-nebius",
        context="arena-nebius",
        kubeconfig_path=tmp_path / "kubeconfig",
        storage_access_key_id="AKIA-test",
        storage_secret_access_key="secret-test",
        worker_node_group_ids={"workers": "mk8snodegroup-test"},
    )
    with patch(
        "agilerl.arena.byoc.commands.ClusterProvisioner.provision",
        return_value=outputs,
    ):
        result = CliRunner().invoke(
            build_cluster_provision_command(),
            ["nebius", "--spec", str(spec_path), "--yes"],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    assert f"arena cluster register --spec {spec_path} --install" in result.output


def test_provision_register_and_install_uses_spec_install_storage(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    tmp_path: Path,
) -> None:
    """Provision installs MinIO when the spec requests the lab storage chart."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            "    endpoint: https://storage.example.com\n",
            "    endpoint: https://storage.example.com\n    install: true\n",
        ),
        encoding="utf-8",
    )
    outputs = ClusterOutputs(
        cluster_name="arena-nebius",
        context="arena-nebius",
        kubeconfig_path=tmp_path / "kubeconfig",
        storage_access_key_id="AKIA-test",
        storage_secret_access_key="secret-test",
        worker_node_group_ids={"workers": "mk8snodegroup-test"},
        storage_endpoint="https://storage.example.com",
        storage_bucket="arena-data",
    )
    client = MagicMock()
    with (
        patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.provision",
            return_value=outputs,
        ),
        patch(
            "agilerl.arena.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch("agilerl.arena.byoc.commands.run_cluster_register") as run_mock,
    ):
        result = CliRunner().invoke(
            build_cluster_provision_command(),
            [
                "nebius",
                "--spec",
                str(spec_path),
                "--yes",
                "--register-and-install",
            ],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["install_storage"] is True
    assert kwargs["storage_secret_name"] == DEFAULT_LAB_STORAGE_SECRET_NAME
    assert kwargs["byoc_provider"] == {
        "provider": "nebius",
        "config": {
            "tenant_id": "tenant-1",
            "project_id": "project-1",
            "storage_project_id": "storage-project-1",
            "subnet_id": "subnet-1",
            "terraform_state": {"bucket": "arena-nebius-tfstate"},
            "object_storage_size_gib": 1024,
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
    assert kwargs["resource_classes"] == [
        ClusterResourceClass(
            name="arena-nebius-workers",
            num_nodes=10,
            node_selector={NEBIUS_WORKER_NODE_GROUP_LABEL: "mk8snodegroup-test"},
            metadata={
                "computeResource": {
                    "numCpus": 15,
                    "numGpus": 1,
                    "memoryBytes": "198 GiB",
                    "gramPerGpu": 80,
                    "gpuMemoryBytes": "80 GiB",
                    "gpu": {
                        "type": "gpu-h100-sxm",
                        "count": 1,
                        "driverVersion": "default",
                    },
                }
            },
        )
    ]


def test_plan_command_applies_spec_overrides(
    command_config: CommandConfig, tmp_path: Path
) -> None:
    """Plan passes explicit CLI overrides into the loaded cluster spec."""
    spec_path = tmp_path / "cluster.yaml"
    write_spec(spec_path)
    with patch("agilerl.arena.byoc.commands.ClusterProvisioner.plan") as plan_mock:
        result = CliRunner().invoke(
            build_cluster_plan_command(),
            [
                "--spec",
                str(spec_path),
                "--name",
                "override-name",
                "--storage-bucket",
                "override-bucket",
                "--domain",
                "override.example.com",
            ],
            obj=command_config,
        )

    assert result.exit_code == 0, result.output
    planned_spec = plan_mock.call_args.args[0]
    assert planned_spec.name == "override-name"
    assert planned_spec.arena.storage.bucket == "override-bucket"
    assert planned_spec.arena.inference.domain == "override.example.com"


class TestTerraformAddressMatches:
    def test_exact_address(self) -> None:
        assert terraform_address_matches(
            "nebius_storage_v1_bucket.arena_data",
            ("nebius_storage_v1_bucket.arena_data",),
        )

    def test_indexed_instance(self) -> None:
        assert terraform_address_matches(
            'nebius_mk8s_v1_node_group.worker["workers"]',
            ("nebius_mk8s_v1_node_group.worker",),
        )

    def test_does_not_match_sibling_resource_name(self) -> None:
        assert not terraform_address_matches(
            "nebius_iam_v1_service_account.storage_admin",
            ("nebius_iam_v1_service_account.storage",),
        )


class TestTerraformRunnerDestroy:
    def test_destroys_all_resources_when_exclude_empty(self, tmp_path: Path) -> None:
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.destroy(tmp_path)

        run.assert_called_once()
        assert run.call_args.args[0] == [
            "terraform",
            "destroy",
            "-input=false",
            "-auto-approve",
        ]

    def test_targets_non_excluded_resources(self, tmp_path: Path) -> None:
        run = MagicMock(
            return_value=MagicMock(
                returncode=0,
                stdout=(
                    "nebius_mk8s_v1_cluster.arena\n"
                    "nebius_storage_v1_bucket.arena_data\n"
                    'nebius_mk8s_v1_node_group.worker["workers"]\n'
                ),
                stderr="",
            )
        )
        runner = TerraformRunner(run=run)

        runner.destroy(tmp_path, exclude=("nebius_storage_v1_bucket.arena_data",))

        assert run.call_args_list[0].args[0] == ["terraform", "state", "list"]
        assert run.call_args_list[1].args[0] == [
            "terraform",
            "destroy",
            "-input=false",
            "-auto-approve",
            "-target=nebius_mk8s_v1_cluster.arena",
            '-target=nebius_mk8s_v1_node_group.worker["workers"]',
        ]

    def test_skips_destroy_when_only_excluded_resources_remain(
        self, tmp_path: Path
    ) -> None:
        run = MagicMock(
            return_value=MagicMock(
                returncode=0,
                stdout="nebius_storage_v1_bucket.arena_data\n",
                stderr="",
            )
        )
        runner = TerraformRunner(run=run)

        runner.destroy(tmp_path, exclude=("nebius_storage_v1_bucket.arena_data",))

        run.assert_called_once()
        assert run.call_args.args[0] == ["terraform", "state", "list"]


class TestClusterProvisionerDestroy:
    def test_keeps_object_storage_and_projects_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_terraform_state_bucket"
            ) as delete_bucket,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_nebius_project"
            ) as delete_project,
        ):
            ClusterProvisioner(runner=runner).destroy(
                _nebius_spec(), state_dir=tmp_path / "state"
            )

        runner.destroy.assert_called_once_with(
            work_dir,
            exclude=(
                *NEBIUS_OBJECT_STORAGE_RESOURCES,
                NEBIUS_COMPUTE_PROJECT_RESOURCE,
                NEBIUS_STORAGE_PROJECT_RESOURCE,
            ),
        )
        delete_bucket.assert_not_called()
        delete_project.assert_not_called()

    def test_delete_storage_destroys_object_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        work_dir = tmp_path / "work"
        runner.prepare.return_value = work_dir

        with patch(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_terraform_state_bucket"
        ) as delete_bucket:
            ClusterProvisioner(runner=runner).destroy(
                _nebius_spec(),
                state_dir=tmp_path / "state",
                delete_storage=True,
            )

        runner.destroy.assert_called_once_with(
            work_dir,
            exclude=(
                NEBIUS_COMPUTE_PROJECT_RESOURCE,
                NEBIUS_STORAGE_PROJECT_RESOURCE,
            ),
        )
        delete_bucket.assert_not_called()

    def test_auto_created_compute_project_is_deleted_without_storage_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        # Arrange
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"
        runner.state_list.return_value = (
            NEBIUS_COMPUTE_PROJECT_RESOURCE,
            NEBIUS_STORAGE_PROJECT_RESOURCE,
        )
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        # Act
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_terraform_state_bucket"
            ) as delete_bucket,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_nebius_project"
            ) as delete_project,
        ):
            ClusterProvisioner(runner=runner).destroy(
                spec, state_dir=tmp_path / "state"
            )

        # Assert
        assert runner.destroy.call_args.kwargs["exclude"] == (
            *NEBIUS_OBJECT_STORAGE_RESOURCES,
            NEBIUS_COMPUTE_PROJECT_RESOURCE,
            NEBIUS_STORAGE_PROJECT_RESOURCE,
        )
        delete_project.assert_called_once_with("project-created")
        delete_bucket.assert_not_called()

    def test_delete_storage_project_removes_state_bucket_and_storage_project(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        # Arrange
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"
        runner.state_list.return_value = (
            NEBIUS_COMPUTE_PROJECT_RESOURCE,
            NEBIUS_STORAGE_PROJECT_RESOURCE,
        )
        runner.terraform_output.side_effect = _terraform_project_output
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        # Act
        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_terraform_state_bucket"
            ) as delete_bucket,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.delete_nebius_project"
            ) as delete_project,
        ):
            ClusterProvisioner(runner=runner).destroy(
                spec, state_dir=tmp_path / "state", delete_storage_project=True
            )

        # Assert
        assert runner.destroy.call_args.kwargs["exclude"] == (
            NEBIUS_COMPUTE_PROJECT_RESOURCE,
            NEBIUS_STORAGE_PROJECT_RESOURCE,
        )
        assert (
            delete_bucket.call_args.args[0].nebius.storage_project_id
            == "storage-project-created"
        )
        assert [call.args[0] for call in delete_project.call_args_list] == [
            "project-created",
            "storage-project-created",
        ]

    def test_refuses_to_delete_a_storage_project_from_the_spec(
        self, tmp_path: Path
    ) -> None:
        runner = MagicMock()

        with pytest.raises(
            click.ClickException, match="only deletes a Nebius project that"
        ):
            ClusterProvisioner(runner=runner).destroy(
                _nebius_spec(),
                state_dir=tmp_path / "state",
                delete_storage_project=True,
            )

        runner.destroy.assert_not_called()


class TestDeleteTerraformStateBucket:
    def test_deletes_the_bucket_with_a_zero_ttl(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            side_effect=_state_bucket_run(
                existing={("storage", "bucket"): "storagebucket-1"}
            )
        )

        # Act
        delete_terraform_state_bucket(_nebius_spec(), run=run)

        # Assert
        delete_call = next(
            call.args[0]
            for call in run.call_args_list
            if call.args[0][1:4] == ["storage", "bucket", "delete"]
        )
        assert delete_call[4:] == [
            "--id",
            "storagebucket-1",
            "--ttl",
            "0s",
            "--format",
            "json",
        ]

    def test_skips_delete_when_the_bucket_is_gone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(side_effect=_state_bucket_run(existing={}))

        delete_terraform_state_bucket(_nebius_spec(), run=run)

        assert not any(
            call.args[0][1:4] == ["storage", "bucket", "delete"]
            for call in run.call_args_list
        )


class TestDeleteNebiusProject:
    def test_deletes_the_project_by_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="{}", stderr=""))

        delete_nebius_project("project-created", run=run)

        assert run.call_args.args[0][1:] == [
            "iam",
            "v2",
            "project",
            "delete",
            "--id",
            "project-created",
            "--format",
            "json",
        ]

    def test_raises_when_the_cli_fails(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(
                returncode=1, stdout="", stderr="ProjectDeletionNotSafe"
            )
        )

        with pytest.raises(click.ClickException, match="ProjectDeletionNotSafe"):
            delete_nebius_project("project-created", run=run)


class TestNebiusObjectStorageResources:
    def test_addresses_exist_in_module(self) -> None:
        module = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
            / "nebius"
            / "main.tf"
        )
        text = module.read_text(encoding="utf-8")

        for address in NEBIUS_OBJECT_STORAGE_RESOURCES:
            resource_type, name = address.split(".", 1)
            assert f'resource "{resource_type}" "{name}"' in text


class TestNebiusNetworkResources:
    def test_module_creates_missing_project_network_and_subnet(self) -> None:
        module = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
            / "nebius"
            / "main.tf"
        )
        text = module.read_text(encoding="utf-8")

        assert 'resource "nebius_iam_v2_project" "arena"' in text
        assert 'resource "nebius_vpc_v1_network" "arena"' in text
        assert 'resource "nebius_vpc_v1_subnet" "arena"' in text
        assert "parent_id = local.project_id" in text
        assert "subnet_id         = local.subnet_id" in text


class TestNebiusProjectSplit:
    @pytest.mark.parametrize(
        ("resource_type", "name"),
        [
            ("nebius_storage_v1_bucket", "arena_data"),
            ("nebius_iam_v1_service_account", "storage"),
            ("nebius_iam_v2_access_key", "storage"),
        ],
    )
    def test_storage_resources_live_in_the_storage_project(
        self, resource_type: str, name: str
    ) -> None:
        block = _terraform_resource_block(resource_type, name)

        assert "parent_id" in block
        assert "local.storage_project_id" in block
        assert "local.project_id" not in block

    @pytest.mark.parametrize(
        ("resource_type", "name"),
        [
            ("nebius_vpc_v1_network", "arena"),
            ("nebius_vpc_v1_subnet", "arena"),
            ("nebius_mk8s_v1_cluster", "arena"),
            ("nebius_compute_v1_filesystem", "shared"),
            ("nebius_iam_v1_service_account", "arena"),
        ],
    )
    def test_compute_resources_live_in_the_compute_project(
        self, resource_type: str, name: str
    ) -> None:
        block = _terraform_resource_block(resource_type, name)

        assert "local.project_id" in block
        assert "local.storage_project_id" not in block

    def test_storage_editor_group_stays_at_tenant_scope(self) -> None:
        block = _terraform_resource_block("nebius_iam_v1_group", "storage_editors")

        assert "parent_id = var.tenant_id" in block

    def test_storage_project_is_created_when_the_variable_is_empty(self) -> None:
        block = _terraform_resource_block("nebius_iam_v2_project", "storage")

        assert 'count     = var.storage_project_id == "" ? 1 : 0' in block
        assert "parent_id = var.tenant_id" in block

    def test_module_outputs_both_project_ids(self) -> None:
        outputs = (_nebius_module_dir() / "outputs.tf").read_text(encoding="utf-8")

        assert "value = local.project_id" in outputs
        assert "value = local.storage_project_id" in outputs

    def test_module_declares_the_storage_project_variable(self) -> None:
        variables = (_nebius_module_dir() / "variables.tf").read_text(encoding="utf-8")

        assert 'variable "storage_project_id"' in variables


class TestWorkerNodeGroupGpuTaint:
    def test_worker_template_taints_gpu_nodes(self) -> None:
        module = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
            / "nebius"
            / "main.tf"
        )
        text = module.read_text(encoding="utf-8")
        worker = text.split('resource "nebius_mk8s_v1_node_group" "worker"')[1]
        arena = text.split('resource "nebius_mk8s_v1_node_group" "arena"')[1].split(
            'resource "nebius_mk8s_v1_node_group" "worker"'
        )[0]

        assert 'key    = "nvidia.com/gpu"' in worker
        assert 'value  = "true"' in worker
        assert 'effect = "NO_SCHEDULE"' in worker
        assert "autoscaling = {" in worker
        assert "min_node_count = each.value.min_node_count" in worker
        assert "max_node_count = each.value.max_node_count" in worker
        assert "fixed_node_count" not in worker
        assert "taints" not in arena
        assert 'name      = "arena"' in arena
        assert "autoscaling = {" in arena
        assert "min_node_count = var.arena_min_node_count" in arena
        assert "max_node_count = var.arena_max_node_count" in arena
        assert "fixed_node_count" not in arena


class TestTerraformRequiredVersion:
    def test_modules_require_native_s3_locking(self) -> None:
        modules = (
            Path(__file__).resolve().parents[2]
            / "agilerl"
            / "arena"
            / "byoc"
            / "provisioning"
            / "terraform_modules"
        )

        aws = (modules / "aws" / "main.tf").read_text(encoding="utf-8")
        nebius = (modules / "nebius" / "main.tf").read_text(encoding="utf-8")

        assert 'required_version = ">= 1.10.0"' in aws
        assert 'required_version = ">= 1.10.0"' in nebius


class TestClusterDestroyCommand:
    @pytest.fixture(autouse=True)
    def _stub_arena_registration(
        self, client_context: Callable[[MagicMock], MagicMock]
    ) -> Iterator[None]:
        client = MagicMock()
        with (
            patch(
                "agilerl.arena.byoc.commands.arena_client",
                return_value=client_context(client),
            ),
            patch.object(ByocApi, "find_cluster", return_value=None) as find_mock,
            patch.object(ByocApi, "unregister_cluster") as unregister_mock,
        ):
            self.find_cluster = find_mock
            self.unregister_cluster = unregister_mock
            yield

    def test_requires_confirmation(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path)],
                obj=command_config,
                input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert "Aborted." in result.output
        assert "Object storage bucket 'arena-data' will be kept." in result.output
        destroy_mock.assert_not_called()

    def test_keeps_object_storage_by_default(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Kept object storage bucket 'arena-data'." in result.output
        assert "Kept Terraform state" in result.output
        destroy_mock.assert_called_once()
        assert destroy_mock.call_args.kwargs["delete_storage"] is False

    def test_delete_storage_destroys_object_storage(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
            ) as destroy_mock,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.NebiusProvider.delete_terraform_state"
            ) as delete_state_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--yes", "--delete-storage"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Kept object storage" not in result.output
        assert "Kept Terraform state" in result.output
        destroy_mock.assert_called_once()
        assert destroy_mock.call_args.kwargs["delete_storage"] is True
        assert destroy_mock.call_args.kwargs["delete_storage_project"] is False
        delete_state_mock.assert_not_called()

    def test_delete_storage_project_deletes_the_state_bucket(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_auto_projects_spec(spec_path)
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
            ) as destroy_mock,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.NebiusProvider.delete_terraform_state"
            ) as delete_state_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--yes", "--delete-storage-project"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert (
            "Deleted Terraform state bucket 'arena-nebius-tfstate' with its Nebius "
            "storage project." in result.output
        )
        spec = destroy_mock.call_args.args[0]
        assert spec.nebius.project_id is None
        assert spec.nebius.storage_project_id is None
        assert destroy_mock.call_args.kwargs["delete_storage"] is True
        assert destroy_mock.call_args.kwargs["delete_storage_project"] is True
        delete_state_mock.assert_not_called()

    def test_delete_storage_project_confirm_prompt(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_auto_projects_spec(spec_path)
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--delete-storage-project"],
                obj=command_config,
                input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert (
            "Destroy Nebius cluster 'arena-nebius', force-delete object storage "
            "bucket 'arena-data', and delete its Nebius storage project?"
        ) in result.output
        destroy_mock.assert_not_called()

    def test_delete_storage_project_refuses_a_spec_project(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        runner = MagicMock()
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner",
            return_value=ClusterProvisioner(runner=runner),
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--yes", "--delete-storage-project"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "only deletes a Nebius project that" in result.output
        runner.destroy.assert_not_called()

    def test_delete_storage_confirm_prompt(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--delete-storage"],
                obj=command_config,
                input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert (
            "Destroy Nebius cluster 'arena-nebius' and force-delete "
            "object storage bucket 'arena-data'?"
        ) in result.output
        destroy_mock.assert_not_called()

    def test_deletes_terraform_state_when_confirmed(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
            ) as destroy_mock,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.NebiusProvider.delete_terraform_state"
            ) as delete_state_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path)],
                obj=command_config,
                input="y\ny\n",
            )

        assert result.exit_code == 0, result.output
        destroy_mock.assert_called_once()
        delete_state_mock.assert_called_once()
        assert (
            "Deleted Terraform state 'clusters/arena-nebius/terraform.tfstate' "
            "from bucket 'arena-nebius-tfstate'."
        ) in result.output

    def test_keeps_terraform_state_when_declined(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with (
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
            ) as destroy_mock,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.NebiusProvider.delete_terraform_state"
            ) as delete_state_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path)],
                obj=command_config,
                input="y\nn\n",
            )

        assert result.exit_code == 0, result.output
        destroy_mock.assert_called_once()
        delete_state_mock.assert_not_called()
        assert (
            "Kept Terraform state 'clusters/arena-nebius/terraform.tfstate' "
            "in bucket 'arena-nebius-tfstate'."
        ) in result.output

    def test_skips_unregister_when_cluster_is_not_registered(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path)],
                obj=command_config,
                input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert "Unregister BYOC cluster" not in result.output
        self.unregister_cluster.assert_not_called()
        destroy_mock.assert_not_called()

    def test_aborts_when_unregister_is_declined(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        self.find_cluster.return_value = {"name": "arena-nebius"}
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path)],
                obj=command_config,
                input="n\n",
            )

        assert result.exit_code == 0, result.output
        assert (
            "Unregister BYOC cluster 'arena-nebius' from Arena "
            "before destroying the cloud cluster?"
        ) in result.output
        assert "Aborted." in result.output
        assert "Destroy Nebius cluster" not in result.output
        self.unregister_cluster.assert_not_called()
        destroy_mock.assert_not_called()

    def test_unregisters_before_terraform_confirmation(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        self.find_cluster.return_value = {"name": "arena-nebius"}
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path)],
                obj=command_config,
                input="y\nn\n",
            )

        assert result.exit_code == 0, result.output
        self.unregister_cluster.assert_called_once_with("arena-nebius")
        assert "Unregistered BYOC cluster 'arena-nebius'." in result.output
        assert "Aborted." in result.output
        destroy_mock.assert_not_called()

    def test_yes_unregisters_and_skips_prompts(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        self.find_cluster.return_value = {"name": "arena-nebius"}
        with patch(
            "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
        ) as destroy_mock:
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--spec", str(spec_path), "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert (
            "Unregister BYOC cluster 'arena-nebius' from Arena before"
            not in result.output
        )
        assert "Destroy Nebius cluster" not in result.output
        self.unregister_cluster.assert_called_once_with("arena-nebius")
        destroy_mock.assert_called_once()

    def test_name_loads_the_cluster_stored_in_arena(
        self, command_config: CommandConfig
    ) -> None:
        row = {
            "name": "arena-nebius",
            "storage_bucket": "arena-data",
            "storage_endpoint": "https://storage.example.com",
            "domain": "inference.example.com",
            "byoc_provider": {
                "provider": "nebius",
                "config": {
                    "tenant_id": "tenant-1",
                    "region": "eu-north1",
                    "project_id": "project-1",
                    "storage_project_id": "storage-project-1",
                    "terraform_state": {"bucket": "arena-nebius-tfstate"},
                    "object_storage_size_gib": 1024,
                    "workers": [
                        {
                            "name": "workers",
                            "platform": "gpu-h100-sxm",
                            "preset": "1gpu-16vcpu-200gb",
                        }
                    ],
                },
            },
        }
        with (
            patch.object(ByocApi, "on_prem_cluster", return_value=row),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
            ) as destroy_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--name", "arena-nebius", "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Using cluster 'arena-nebius' stored in Arena." in result.output
        spec = destroy_mock.call_args.args[0]
        assert spec.name == "arena-nebius"
        assert spec.terraform_state.bucket == "arena-nebius-tfstate"
        assert spec.arena.storage.bucket == "arena-data"
        assert spec.nebius.project_id == "project-1"
        assert spec.nebius.storage_project_id == "storage-project-1"
        assert spec.nebius.workers[0].platform == "gpu-h100-sxm"
        assert destroy_mock.call_args.kwargs["state_dir"].name == "terraform"

    def test_name_requires_a_stored_cluster(
        self, command_config: CommandConfig
    ) -> None:
        with (
            patch.object(ByocApi, "on_prem_cluster", return_value=None),
            patch(
                "agilerl.arena.byoc.commands.ClusterProvisioner.destroy"
            ) as destroy_mock,
        ):
            result = CliRunner().invoke(
                build_cluster_destroy_command(),
                ["--name", "missing", "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "No cluster named 'missing'" in result.output
        destroy_mock.assert_not_called()

    def test_rejects_spec_and_name_together(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec_path = tmp_path / "cluster.yaml"
        write_spec(spec_path)
        result = CliRunner().invoke(
            build_cluster_destroy_command(),
            ["--spec", str(spec_path), "--name", "arena-nebius", "--yes"],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "not both" in result.output

    def test_requires_spec_or_name(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_destroy_command(),
            ["--yes"],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "Pass --spec or --name." in result.output


class TestClusterSpecFromOnPremRow:
    def test_builds_a_nebius_spec(self) -> None:
        spec = cluster_spec_from_on_prem_row(
            {
                "name": "arena-nebius",
                "storage_bucket": "arena-data",
                "storage_endpoint": "https://storage.example.com",
                "domain": "inference.example.com",
                "hostname_template": "inference-{deploymentId}",
                "byoc_provider": {
                    "provider": "nebius",
                    "config": {
                        "tenant_id": "tenant-1",
                        "project_id": "project-1",
                        "storage_project_id": "storage-project-1",
                        "terraform_state": {
                            "bucket": "arena-nebius-tfstate",
                            "key": "clusters/arena-nebius/terraform.tfstate",
                        },
                        "object_storage_size_gib": 1024,
                    },
                },
            }
        )

        assert spec.terraform_state.bucket == "arena-nebius-tfstate"
        assert spec.terraform_state.key == "clusters/arena-nebius/terraform.tfstate"
        assert spec.nebius.tenant_id == "tenant-1"
        assert spec.nebius.workers[0].name == "workers"

    def test_rejects_a_cluster_without_nebius_settings(self) -> None:
        with pytest.raises(ValueError, match="no stored Nebius settings"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "byoc_provider": {"provider": "custom", "config": {}},
                }
            )


class TestTerraformStateSpec:
    def test_rejects_experiment_data_bucket(self) -> None:
        with pytest.raises(ValueError, match="must not be the experiment-data bucket"):
            ClusterSpec.model_validate(
                {
                    "provider": "nebius",
                    "name": "test",
                    "terraform_state": {"bucket": "bucket"},
                    "arena": {
                        "storage": {
                            "bucket": "bucket",
                            "endpoint": "https://storage.eu-north1.nebius.cloud",
                        },
                        "inference": {"domain": "inference.example.com"},
                    },
                    "nebius": {
                        "tenant_id": "tenant",
                        "project_id": "project",
                        "storage_project_id": "storage-project",
                        "subnet_id": "subnet",
                        "object_storage": {"size_gib": 1024},
                    },
                }
            )

    def test_defaults_object_key_and_endpoint(self) -> None:
        spec = _nebius_spec()

        assert spec.terraform_state.create_bucket is False
        assert spec.terraform_state.object_key(spec.name) == (
            "clusters/test/terraform.tfstate"
        )
        assert (
            spec.terraform_state.s3_endpoint(spec.arena.storage.endpoint)
            == spec.arena.storage.endpoint
        )


class TestNebiusRegionFromEndpoint:
    def test_parses_nebius_storage_host(self) -> None:
        assert (
            nebius_region_from_endpoint("https://storage.eu-north1.nebius.cloud")
            == "eu-north1"
        )

    def test_defaults_unknown_hosts(self) -> None:
        assert nebius_region_from_endpoint("https://storage.example.com") == "eu-north1"


class TestTerraformStateBucket:
    def test_exists_when_get_by_name_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="{}", stderr=""))

        assert terraform_state_bucket_exists(_nebius_spec(), run=run) is True
        assert run.call_args.args[0][1:4] == ["storage", "bucket", "get-by-name"]

    def test_missing_when_get_by_name_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
        )

        assert terraform_state_bucket_exists(_nebius_spec(), run=run) is False

    def test_missing_when_nebius_reports_nosuchbucket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(
                returncode=1,
                stdout="",
                stderr=(
                    "Switch to your browser to complete the authentication process.\n"
                    "Error: rpc error: code = NotFound desc = NoSuchBucket: "
                    "Bucket doesn't exist"
                ),
            )
        )

        assert terraform_state_bucket_exists(_nebius_spec(), run=run) is False

    def test_ensure_uses_env_credentials_when_bucket_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-existing")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "existing-secret")
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="{}", stderr=""))

        env = ensure_terraform_state_bucket(
            _nebius_spec(), state_dir=tmp_path / "state", run=run
        )

        assert env["AWS_ACCESS_KEY_ID"] == "AKIA-existing"
        assert env["AWS_SECRET_ACCESS_KEY"] == "existing-secret"

    def test_ensure_creates_bucket_when_requested(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        env = ensure_terraform_state_bucket(
            _nebius_spec(),
            state_dir=tmp_path / "terraform",
            create=True,
            run=MagicMock(side_effect=_state_bucket_run(existing={})),
        )

        assert env == {
            "AWS_ACCESS_KEY_ID": "AKIA-created",
            "AWS_SECRET_ACCESS_KEY": "created-secret",
        }
        creds = json.loads(
            (tmp_path / "tfstate-credentials.json").read_text(encoding="utf-8")
        )
        assert creds["aws_access_key_id"] == "AKIA-created"

    def test_ensure_reuses_iam_left_by_a_failed_attempt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange: the group lives at tenant scope, so it outlives a failed attempt.
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        run = MagicMock(
            side_effect=_state_bucket_run(
                existing={
                    ("iam", "service-account"): "sa-existing",
                    ("iam", "group"): "group-existing",
                }
            )
        )

        # Act
        env = ensure_terraform_state_bucket(
            _nebius_spec(),
            state_dir=tmp_path / "terraform",
            create=True,
            run=run,
        )

        # Assert
        assert env == {
            "AWS_ACCESS_KEY_ID": "AKIA-created",
            "AWS_SECRET_ACCESS_KEY": "created-secret",
        }
        created = [
            tuple(call.args[0][1:4])
            for call in run.call_args_list
            if "create" in call.args[0]
        ]
        assert ("iam", "service-account", "create") not in created
        assert ("iam", "group", "create") not in created
        assert ("iam", "group-membership", "create") in created
        assert ("storage", "bucket", "create") in created

    def test_ensure_refetches_credentials_when_file_is_gone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(returncode=0, stdout="{}", stderr="")
            if argv[1:4] == ["iam", "service-account", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "sa-tfstate"}}),
                    stderr="",
                )
            if argv[1:5] == ["iam", "v2", "access-key", "list-by-account"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {"metadata": {"id": "key-1", "name": "test-tfstate"}}
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:5] == ["iam", "v2", "access-key", "get-secret"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "aws_access_key_id": "AKIA-fetched",
                            "secret": "fetched-secret",
                        }
                    ),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="unexpected")

        # Act
        env = ensure_terraform_state_bucket(
            _nebius_spec(),
            state_dir=tmp_path / "terraform",
            run=MagicMock(side_effect=run_side_effect),
        )

        # Assert
        assert env == {
            "AWS_ACCESS_KEY_ID": "AKIA-fetched",
            "AWS_SECRET_ACCESS_KEY": "fetched-secret",
        }
        creds = json.loads(
            (tmp_path / "tfstate-credentials.json").read_text(encoding="utf-8")
        )
        assert creds["aws_secret_access_key"] == "fetched-secret"

    def test_ensure_fails_when_state_access_key_is_gone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(returncode=0, stdout="{}", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        # Act / Assert
        with pytest.raises(click.ClickException, match="no access key named"):
            ensure_terraform_state_bucket(
                _nebius_spec(),
                state_dir=tmp_path / "terraform",
                run=MagicMock(side_effect=run_side_effect),
            )

    def test_ensure_requires_bucket_when_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
        )

        with pytest.raises(click.ClickException, match="is required"):
            ensure_terraform_state_bucket(
                _nebius_spec(), state_dir=tmp_path / "state", run=run
            )


class TestTerraformRunnerInitialize:
    def test_initializes_without_backend(self, tmp_path: Path) -> None:
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.initialize_without_backend(tmp_path)

        assert run.call_args.args[0] == [
            "terraform",
            "init",
            "-input=false",
            "-backend=false",
        ]

    def test_bootstrap_removes_a_stale_backend(self, tmp_path: Path) -> None:
        backend = tmp_path / "backend.tf"
        backend.write_text('terraform {\n  backend "s3" {}\n}\n', encoding="utf-8")
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.initialize_without_backend(tmp_path)

        assert not backend.exists()

    def test_writes_the_s3_backend_before_init(self, tmp_path: Path) -> None:
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.initialize(
            tmp_path,
            _nebius_spec(storage_endpoint="https://storage.eu-north1.nebius.cloud"),
        )

        backend = (tmp_path / "backend.tf").read_text(encoding="utf-8")
        assert 'backend "s3"' in backend
        assert 'bucket                      = "test-tfstate"' in backend
        assert (
            'key                         = "clusters/test/terraform.tfstate"' in backend
        )
        assert 'region                      = "eu-north1"' in backend
        assert "skip_credentials_validation = true" in backend

    def test_migrates_local_state_with_resources(self, tmp_path: Path) -> None:
        (tmp_path / "terraform.tfstate").write_text(
            json.dumps(
                {
                    "version": 4,
                    "resources": [
                        {
                            "mode": "managed",
                            "type": "nebius_storage_v1_bucket",
                            "name": "arena_data",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.initialize(tmp_path, _nebius_spec())

        assert run.call_args.args[0] == [
            "terraform",
            "init",
            "-input=false",
            "-migrate-state",
            "-force-copy",
        ]

    def test_does_not_overwrite_remote_state_with_empty_local_file(
        self, tmp_path: Path
    ) -> None:
        local_state = tmp_path / "terraform.tfstate"
        local_state.write_text("{}\n", encoding="utf-8")
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.initialize(tmp_path, _nebius_spec())

        assert run.call_args.args[0] == ["terraform", "init", "-input=false"]
        assert not local_state.is_file()


class TestTerraformRunnerAdoptResources:
    def test_skips_when_no_resources(self, tmp_path: Path) -> None:
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.adopt_resources(tmp_path, {})

        run.assert_not_called()

    def test_imports_resources_missing_from_state(self, tmp_path: Path) -> None:
        run = MagicMock(
            return_value=MagicMock(
                returncode=0,
                stdout="nebius_iam_v1_service_account.storage\n",
                stderr="",
            )
        )
        runner = TerraformRunner(run=run)

        runner.adopt_resources(
            tmp_path,
            {
                "nebius_iam_v1_service_account.storage": "sa-1",
                "nebius_storage_v1_bucket.arena_data": "bucket-1",
            },
        )

        commands = [call.args[0] for call in run.call_args_list]
        assert commands[0] == ["terraform", "state", "list"]
        assert commands[1] == [
            "terraform",
            "import",
            "-input=false",
            "nebius_storage_v1_bucket.arena_data",
            "bucket-1",
        ]

    def test_imports_all_resources_when_state_object_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        # Arrange: a fresh backend has no state object, so state list exits 1.
        def _run(argv: list[str], **_: object) -> MagicMock:
            if argv[1:3] == ["state", "list"]:
                return MagicMock(
                    returncode=1,
                    stdout="",
                    stderr="Error: No state file was found!\n",
                )
            return MagicMock(returncode=0, stdout="", stderr="")

        runner = TerraformRunner(run=MagicMock(side_effect=_run))

        # Act
        runner.adopt_resources(
            tmp_path,
            {
                "nebius_iam_v1_service_account.storage": "sa-1",
                "nebius_storage_v1_bucket.arena_data": "bucket-1",
            },
        )

        # Assert
        commands = [call.args[0] for call in runner.run.call_args_list]
        assert commands[1:] == [
            [
                "terraform",
                "import",
                "-input=false",
                "nebius_iam_v1_service_account.storage",
                "sa-1",
            ],
            [
                "terraform",
                "import",
                "-input=false",
                "nebius_storage_v1_bucket.arena_data",
                "bucket-1",
            ],
        ]

    def test_raises_on_other_state_list_failures(self, tmp_path: Path) -> None:
        run = MagicMock(
            return_value=MagicMock(
                returncode=1,
                stdout="",
                stderr="Error: Backend initialization required",
            )
        )
        runner = TerraformRunner(run=run)

        with pytest.raises(click.ClickException, match="terraform state list failed"):
            runner.adopt_resources(
                tmp_path, {"nebius_storage_v1_bucket.arena_data": "bucket-1"}
            )


class TestExperimentStorageImportIds:
    def test_empty_when_bucket_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
        )

        assert experiment_storage_import_ids(_nebius_spec(), run=run) == {}

    def test_finds_storage_when_get_by_name_misses(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if "get-by-name" in argv:
                return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
            if argv[1:4] == ["storage", "bucket", "list"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {"items": [{"metadata": {"id": "bucket-1", "name": "bucket"}}]}
                    ),
                    stderr="",
                )
            if argv[1:4] == ["iam", "service-account", "list"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {
                                        "id": "sa-storage",
                                        "name": "test-storage",
                                    }
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group", "list"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {
                                        "id": "group-storage",
                                        "name": "test-storage-editors",
                                    }
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group-membership", "list-members"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "memberships": [
                                {
                                    "metadata": {"id": "membership-1"},
                                    "spec": {"member_id": "sa-storage"},
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:5] == ["iam", "v2", "access-key", "list-by-account"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {
                                        "id": "key-1",
                                        "name": "test-storage",
                                    }
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="unexpected")

        assert (
            experiment_storage_import_ids(
                _nebius_spec(), run=MagicMock(side_effect=run_side_effect)
            )["nebius_iam_v1_group.storage_editors"]
            == "group-storage"
        )

    def test_returns_ids_when_bucket_and_iam_exist(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "bucket-1"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "service-account", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "sa-storage"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "group-storage"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group-membership", "list-members"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "memberships": [
                                {
                                    "metadata": {"id": "membership-1"},
                                    "spec": {"member_id": "sa-storage"},
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:5] == ["iam", "v2", "access-key", "list-by-account"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {
                                        "id": "key-1",
                                        "name": "test-storage",
                                    }
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="unexpected")

        assert experiment_storage_import_ids(
            _nebius_spec(), run=MagicMock(side_effect=run_side_effect)
        ) == {
            "nebius_iam_v1_service_account.storage": "sa-storage",
            "nebius_iam_v1_group.storage_editors": "group-storage",
            "nebius_iam_v1_group_membership.storage_editor": "membership-1",
            "nebius_iam_v2_access_key.storage": "key-1",
            "nebius_storage_v1_bucket.arena_data": "bucket-1",
        }

    def test_follows_membership_page_tokens(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "bucket-1"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "service-account", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "sa-storage"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "group-storage"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group-membership", "list-members"]:
                if "--page-token" not in argv:
                    return MagicMock(
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "memberships": [
                                    {
                                        "metadata": {"id": "membership-other"},
                                        "spec": {"member_id": "sa-other"},
                                    }
                                ],
                                "next_page_token": "page-2",
                            }
                        ),
                        stderr="",
                    )
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "memberships": [
                                {
                                    "metadata": {"id": "membership-1"},
                                    "spec": {"member_id": "sa-storage"},
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:5] == ["iam", "v2", "access-key", "list-by-account"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {"metadata": {"id": "key-1", "name": "test-storage"}}
                            ]
                        }
                    ),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="unexpected")

        run = MagicMock(side_effect=run_side_effect)

        # Act
        ids = experiment_storage_import_ids(_nebius_spec(), run=run)

        # Assert
        assert ids["nebius_iam_v1_group_membership.storage_editor"] == "membership-1"
        membership_calls = [
            call.args[0]
            for call in run.call_args_list
            if call.args[0][1:4] == ["iam", "group-membership", "list-members"]
        ]
        assert len(membership_calls) == 2
        assert "--all" not in membership_calls[0]
        token_index = membership_calls[1].index("--page-token")
        assert membership_calls[1][token_index + 1] == "page-2"

    def test_empty_membership_page_reports_missing_membership(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange: Nebius returns `{}` for a list with no results.
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "bucket-1"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "service-account", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "sa-storage"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "group-storage"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "group-membership", "list-members"]:
                return MagicMock(returncode=0, stdout="{}", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="unexpected")

        # Act / Assert
        with pytest.raises(click.ClickException, match="has no membership"):
            experiment_storage_import_ids(
                _nebius_spec(), run=MagicMock(side_effect=run_side_effect)
            )

    def test_errors_when_bucket_exists_without_service_account(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "bucket-1"}}),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        with pytest.raises(click.ClickException, match="service account"):
            experiment_storage_import_ids(
                _nebius_spec(), run=MagicMock(side_effect=run_side_effect)
            )


class TestDeleteClusterTerraformState:
    def test_deletes_state_object_and_lockfile(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        requests: list[object] = []

        def opener(request: object, timeout: int = 30) -> MagicMock:
            requests.append(request)
            response = MagicMock()
            response.__enter__.return_value = response
            response.__exit__.return_value = None
            return response

        delete_cluster_terraform_state(
            _nebius_spec(), state_dir=tmp_path / "state", opener=opener
        )

        urls = [request.full_url for request in requests]
        assert len(urls) == 2
        assert urls[0].endswith("/test-tfstate/clusters/test/terraform.tfstate")
        assert urls[1].endswith("/test-tfstate/clusters/test/terraform.tfstate.tflock")
        assert all(request.get_method() == "DELETE" for request in requests)

    def test_missing_object_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        def opener(request: object, timeout: int = 30) -> MagicMock:
            raise HTTPError(
                getattr(request, "full_url", ""),
                404,
                "Not Found",
                hdrs={},
                fp=None,
            )

        delete_cluster_terraform_state(
            _nebius_spec(), state_dir=tmp_path / "state", opener=opener
        )


class TestInferenceDomain:
    def test_rejects_urls(self) -> None:
        with pytest.raises(ValueError, match="bare DNS name"):
            normalize_inference_domain("https://inference.example.com")

    def test_rejects_blank_hostname_template(self) -> None:
        with pytest.raises(ValueError, match="non-empty host pattern"):
            resolve_inference_hostname_template("   ")


class TestRequireKubectl:
    def test_raises_when_kubectl_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.kubectl.shutil.which",
            lambda _name: None,
        )

        with pytest.raises(click.ClickException, match="kubectl not found"):
            require_kubectl()


class TestKubeconfigExecutable:
    def test_empty_command_is_unresolved(self) -> None:
        assert resolve_kubeconfig_executable([]) is None

    def test_uses_an_existing_file(self, tmp_path: Path) -> None:
        exe = tmp_path / "nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")

        assert resolve_kubeconfig_executable([str(exe)]) == exe

    def test_absolute_missing_file_is_unresolved(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing-nebius"

        assert resolve_kubeconfig_executable([str(missing)]) is None

    def test_uses_path_lookup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.shutil.which",
            lambda _name: str(exe),
        )

        assert resolve_kubeconfig_executable(["nebius"]) == exe

    def test_uses_default_nebius_cli(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.shutil.which",
            lambda _name: None,
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.NEBIUS_DEFAULT_CLI",
            exe,
        )

        assert resolve_kubeconfig_executable(["nebius"]) == exe

    def test_uses_nebius_cli_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exe = tmp_path / "custom-nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("NEBIUS_CLI", str(exe))
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.shutil.which",
            lambda _name: None,
        )

        assert resolve_kubeconfig_executable(["nebius"]) == exe

    def test_unresolved_command_is_returned_unchanged(self) -> None:
        with patch(
            "agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig.resolve_kubeconfig_executable",
            return_value=None,
        ):
            assert build_kubeconfig_argv(["nebius", "mk8s"]) == ["nebius", "mk8s"]


class TestAbsolutePath:
    def test_uses_pwd_when_cwd_is_gone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PWD", str(tmp_path))

        with patch(
            "agilerl.arena.byoc.provisioning.terraform.Path.cwd",
            side_effect=FileNotFoundError,
        ):
            resolved = absolute_path(Path("work"))

        assert resolved == (tmp_path / "work").resolve()

    def test_reraises_when_cwd_and_pwd_are_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PWD", raising=False)

        with (
            patch(
                "agilerl.arena.byoc.provisioning.terraform.Path.cwd",
                side_effect=FileNotFoundError,
            ),
            pytest.raises(FileNotFoundError),
        ):
            absolute_path(Path("work"))

    def test_returns_unresolved_path_when_resolve_fails(self, tmp_path: Path) -> None:
        missing = tmp_path / "gone" / "work"

        with patch.object(Path, "resolve", side_effect=FileNotFoundError):
            assert absolute_path(missing) == missing


class TestLocalTerraformStateHasResources:
    def test_invalid_json_is_empty(self, tmp_path: Path) -> None:
        state = tmp_path / "terraform.tfstate"
        state.write_text("{not-json", encoding="utf-8")

        assert local_terraform_state_has_resources(state) is False

    def test_non_mapping_payload_is_empty(self, tmp_path: Path) -> None:
        state = tmp_path / "terraform.tfstate"
        state.write_text("[]\n", encoding="utf-8")

        assert local_terraform_state_has_resources(state) is False


class TestTerraformRunnerFailures:
    def test_prepare_requires_terraform(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.terraform.shutil.which",
            lambda _name: None,
        )

        with pytest.raises(click.ClickException, match="terraform not found"):
            TerraformRunner().prepare(
                _nebius_spec(), module_name="nebius", state_dir=tmp_path
            )

    def _outputs_payload(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "cluster_name": "test",
            "context": "ctx",
            "cluster_id": "mk8s-1",
            "storage_access_key_id": "AKIA",
            "storage_secret_access_key": "secret",
            "worker_node_group_ids": {"gpu": "ng-1"},
        }
        values.update(overrides)
        return values

    def test_outputs_require_worker_group_ids(self, tmp_path: Path) -> None:
        runner = TerraformRunner(run=MagicMock())
        payload = self._outputs_payload(worker_node_group_ids={"gpu": " "})
        runner.terraform_outputs = lambda _work_dir: payload

        with pytest.raises(click.ClickException, match="non-empty string map"):
            materialize_outputs(runner, tmp_path, tmp_path)

    def test_outputs_require_cluster_id_string(self, tmp_path: Path) -> None:
        runner = TerraformRunner(run=MagicMock())
        payload = self._outputs_payload(cluster_id="  ")
        runner.terraform_outputs = lambda _work_dir: payload

        with pytest.raises(click.ClickException, match="cluster_id must be"):
            materialize_outputs(runner, tmp_path, tmp_path)

    def test_outputs_require_nebius_cli(self, tmp_path: Path) -> None:
        runner = TerraformRunner(run=MagicMock())
        payload = self._outputs_payload()
        runner.terraform_outputs = lambda _work_dir: payload

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.resolve_kubeconfig_executable",
                return_value=None,
            ),
            pytest.raises(click.ClickException, match="Nebius CLI not found"),
        ):
            materialize_outputs(runner, tmp_path, tmp_path)

    def test_outputs_wrap_missing_cli_binary(self, tmp_path: Path) -> None:
        runner = TerraformRunner(run=MagicMock(side_effect=FileNotFoundError))
        payload = self._outputs_payload()
        runner.terraform_outputs = lambda _work_dir: payload
        exe = tmp_path / "nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.resolve_kubeconfig_executable",
                return_value=exe,
            ),
            pytest.raises(click.ClickException, match="Nebius CLI not found"),
        ):
            materialize_outputs(runner, tmp_path, tmp_path)

    def test_outputs_raise_when_get_credentials_fails(self, tmp_path: Path) -> None:
        runner = TerraformRunner(
            run=MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr=""))
        )
        payload = self._outputs_payload()
        runner.terraform_outputs = lambda _work_dir: payload
        exe = tmp_path / "nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.resolve_kubeconfig_executable",
                return_value=exe,
            ),
            pytest.raises(click.ClickException, match="Could not retrieve kubeconfig"),
        ):
            materialize_outputs(runner, tmp_path, tmp_path)

    def test_outputs_raise_when_kubeconfig_is_not_written(self, tmp_path: Path) -> None:
        runner = TerraformRunner(
            run=MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        )
        payload = self._outputs_payload()
        runner.terraform_outputs = lambda _work_dir: payload
        exe = tmp_path / "nebius"
        exe.write_text("#!/bin/sh\n", encoding="utf-8")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.resolve_kubeconfig_executable",
                return_value=exe,
            ),
            pytest.raises(click.ClickException, match="did not write a kubeconfig"),
        ):
            materialize_outputs(runner, tmp_path, tmp_path)

    def test_terraform_output_rejects_invalid_json(self, tmp_path: Path) -> None:
        runner = TerraformRunner(
            run=MagicMock(
                return_value=MagicMock(returncode=0, stdout="{nope", stderr="")
            )
        )

        with pytest.raises(click.ClickException, match="invalid JSON for output"):
            runner.terraform_output(tmp_path, "project_id")

    def test_terraform_outputs_reject_invalid_json(self, tmp_path: Path) -> None:
        runner = TerraformRunner(
            run=MagicMock(
                return_value=MagicMock(returncode=0, stdout="{nope", stderr="")
            )
        )

        with pytest.raises(click.ClickException, match="invalid JSON outputs"):
            runner.terraform_outputs(tmp_path)

    def test_worker_ids_ignore_blank_names(self) -> None:
        assert _worker_node_group_ids({"  ": "ng-1"}) == {}

    def test_worker_ids_ignore_blank_ids(self) -> None:
        assert _worker_node_group_ids({"gpu": "  "}) == {}

    def test_run_raises_on_nonzero(self, tmp_path: Path) -> None:
        runner = TerraformRunner(
            run=MagicMock(
                return_value=MagicMock(returncode=1, stdout="", stderr="boom")
            )
        )

        with pytest.raises(click.ClickException, match="terraform plan"):
            runner.plan(tmp_path)

    def test_apply_runs_the_saved_plan(self, tmp_path: Path) -> None:
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        runner = TerraformRunner(run=run)

        runner.apply(tmp_path)

        assert run.call_args.args[0] == [
            "terraform",
            "apply",
            "-input=false",
            "arena.tfplan",
        ]

    def test_worker_ids_ignore_non_mappings(self) -> None:
        assert _worker_node_group_ids(None) == {}
        assert _worker_node_group_ids([]) == {}


class TestTerraformProjectId:
    def test_rejects_blank_output(self) -> None:
        with pytest.raises(click.ClickException, match="did not return"):
            terraform_project_id("  ", "project id")


class TestClusterProvisionerState:
    def test_cluster_does_not_exist_without_state(self, tmp_path: Path) -> None:
        provisioner = ClusterProvisioner(runner=MagicMock())

        with patch.object(NEBIUS, "open_state", return_value=None):
            assert (
                provisioner.cluster_exists(_nebius_spec(), state_dir=tmp_path / "state")
                is False
            )

    def test_write_kubeconfig_requires_state(self, tmp_path: Path) -> None:
        provisioner = ClusterProvisioner(runner=MagicMock())

        with (
            patch.object(NEBIUS, "open_state", return_value=None),
            pytest.raises(click.ClickException, match="is not available"),
        ):
            provisioner.write_kubeconfig(
                _nebius_spec(),
                state_dir=tmp_path / "state",
                output_dir=tmp_path / "out",
            )

    def test_unknown_provider_is_rejected(self, tmp_path: Path) -> None:
        spec = ClusterSpec.model_construct(
            **{**_nebius_spec().model_dump(), "provider": "other"}
        )
        provisioner = ClusterProvisioner(runner=MagicMock())

        with pytest.raises(ValueError, match="Unsupported cluster provider"):
            provisioner.cluster_exists(spec, state_dir=tmp_path)

    def test_open_state_skips_when_storage_project_is_unknown(
        self, tmp_path: Path
    ) -> None:
        spec = _nebius_spec(storage_project_id=None, region="eu-north1")
        provisioner = ClusterProvisioner(runner=MagicMock())

        with patch(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
            return_value=None,
        ):
            assert (
                NEBIUS.open_state(spec, runner=provisioner.runner, state_dir=tmp_path)
                is None
            )

    def test_open_state_skips_when_bucket_is_missing(self, tmp_path: Path) -> None:
        provisioner = ClusterProvisioner(runner=MagicMock())

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=None,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.terraform_state_bucket_exists",
                return_value=False,
            ),
        ):
            assert (
                NEBIUS.open_state(
                    _nebius_spec(), runner=provisioner.runner, state_dir=tmp_path
                )
                is None
            )

    def test_cluster_exists_when_state_lists_cluster(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.state_list.return_value = ("nebius_mk8s_v1_cluster.arena",)
        provisioner = ClusterProvisioner(runner=runner)

        with patch.object(NEBIUS, "open_state", return_value=tmp_path):
            assert (
                provisioner.cluster_exists(_nebius_spec(), state_dir=tmp_path / "state")
                is True
            )

    def test_write_kubeconfig_returns_outputs_path(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        runner = MagicMock()
        provisioner = ClusterProvisioner(runner=runner)

        with (
            patch.object(NEBIUS, "open_state", return_value=tmp_path),
            patch.object(
                NEBIUS,
                "cluster_outputs",
                return_value=MagicMock(kubeconfig_path=kubeconfig),
            ),
        ):
            assert (
                provisioner.write_kubeconfig(
                    _nebius_spec(),
                    state_dir=tmp_path / "state",
                    output_dir=tmp_path / "out",
                )
                == kubeconfig
            )

    def test_open_state_fetches_credentials_when_bucket_exists(
        self, tmp_path: Path
    ) -> None:
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"
        runner.extra_env = {}
        creds = TerraformStateCredentials("AKIA", "secret")

        with (
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.read_terraform_state_credentials",
                return_value=None,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.terraform_state_bucket_exists",
                return_value=True,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.fetch_terraform_state_credentials",
                return_value=creds,
            ),
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.write_terraform_state_credentials"
            ) as write_mock,
            patch(
                "agilerl.arena.byoc.provisioning.providers.nebius.provider.write_nebius_provider_config"
            ),
        ):
            work = NEBIUS.open_state(_nebius_spec(), runner=runner, state_dir=tmp_path)

        assert work == tmp_path / "work"
        write_mock.assert_called_once()
        runner.initialize.assert_called_once()

    def test_status_returns_terraform_outputs(self, tmp_path: Path) -> None:
        runner = MagicMock()
        runner.terraform_outputs.return_value = {"cluster_id": "mk8s-1"}
        provisioner = ClusterProvisioner(runner=runner)

        with patch.object(
            provisioner, "_prepare", return_value=(_nebius_spec(), tmp_path)
        ):
            assert provisioner.status(_nebius_spec(), state_dir=tmp_path) == {
                "cluster_id": "mk8s-1"
            }

    def test_provision_skips_gateway_when_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        terraform_state_ready: None,
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        spec = _nebius_spec(gateway_api={"enable": False})
        outputs = ClusterOutputs(
            cluster_name="test",
            context="test",
            kubeconfig_path=tmp_path / "kubeconfig",
            storage_access_key_id="AKIA",
            storage_secret_access_key="secret",
            worker_node_group_ids={"workers": "ng-1"},
        )
        runner = MagicMock()
        runner.prepare.return_value = tmp_path / "work"
        monkeypatch.setattr(
            NEBIUS, "cluster_outputs", lambda *_args, **_kwargs: outputs
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_storage_secret",
            MagicMock(),
        )
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.ensure_gpu_runtime_class",
            MagicMock(),
        )
        configure = MagicMock()
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.provider.configure_gateway_api",
            configure,
        )

        result = ClusterProvisioner(runner=runner).provision(
            spec, state_dir=tmp_path / "state", output_dir=tmp_path / "out"
        )

        configure.assert_not_called()
        assert result.gateway_api_parent_refs is None


class TestGatewayApiFailures:
    def test_restarts_cilium_when_crds_are_missing(self, tmp_path: Path) -> None:
        kubeconfig = tmp_path / "kubeconfig"
        kubeconfig.write_text("apiVersion: v1\n", encoding="utf-8")
        calls: list[list[str]] = []

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            calls.append(argv)
            if argv[:2] == ["kubectl", "get"] and "enable-gateway-api" in argv[-1]:
                return MagicMock(returncode=0, stdout="true", stderr="")
            if argv[:2] == ["kubectl", "get"] and "enable-envoy-config" in argv[-1]:
                return MagicMock(returncode=0, stdout="true", stderr="")
            if argv[:2] == ["kubectl", "get"] and argv[2] == "crd":
                return MagicMock(returncode=1, stdout="", stderr="NotFound")
            return MagicMock(returncode=0, stdout="", stderr="")

        enable_cilium_gateway_api(
            env={"KUBECONFIG": str(kubeconfig)},
            run=MagicMock(side_effect=run_side_effect),
        )

        assert any("rollout" in call and "restart" in call for call in calls)

    def test_self_signed_secret_requires_openssl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
            lambda _name: None,
        )

        with pytest.raises(click.ClickException, match="openssl not found"):
            _create_self_signed_tls_secret(
                secret_name="tls",
                domain="inference.example.com",
                gateway_namespace="arena",
                env={},
                run=MagicMock(),
            )

    def test_self_signed_secret_raises_when_openssl_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
            lambda name: "/usr/bin/openssl" if name == "openssl" else None,
        )

        with pytest.raises(click.ClickException, match="self-signed TLS"):
            _create_self_signed_tls_secret(
                secret_name="tls",
                domain="inference.example.com",
                gateway_namespace="arena",
                env={},
                run=MagicMock(
                    return_value=MagicMock(
                        returncode=1, stdout="", stderr="openssl failed"
                    )
                ),
            )

    def test_self_signed_secret_raises_when_create_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.gateway_api.shutil.which",
            lambda name: "/usr/bin/openssl" if name == "openssl" else None,
        )

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[:1] == ["openssl"]:
                Path(argv[argv.index("-keyout") + 1]).write_bytes(b"key")
                Path(argv[argv.index("-out") + 1]).write_bytes(b"cert")
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=1, stdout="", stderr="forbidden")

        with pytest.raises(click.ClickException, match="Could not create TLS secret"):
            _create_self_signed_tls_secret(
                secret_name="tls",
                domain="inference.example.com",
                gateway_namespace="arena",
                env={},
                run=MagicMock(side_effect=run_side_effect),
            )

    def test_gateway_class_apply_failure(self) -> None:
        with pytest.raises(click.ClickException, match="Could not create GatewayClass"):
            ensure_cilium_gateway_class(
                env={},
                run=MagicMock(
                    return_value=MagicMock(returncode=1, stdout="", stderr="fail")
                ),
            )

    def test_gateway_apply_failure(self) -> None:
        with pytest.raises(click.ClickException, match="Could not create Gateway"):
            ensure_cilium_gateway(
                gateway_name="arena",
                domain="inference.example.com",
                tls_secret_name="tls",
                gateway_namespace="arena",
                env={},
                run=MagicMock(
                    return_value=MagicMock(returncode=1, stdout="", stderr="fail")
                ),
            )

    def test_https_redirect_apply_failure(self) -> None:
        with pytest.raises(click.ClickException, match="Could not create HTTPRoute"):
            ensure_https_redirect_route(
                gateway_name="arena",
                domain="inference.example.com",
                gateway_namespace="arena",
                env={},
                run=MagicMock(
                    return_value=MagicMock(returncode=1, stdout="", stderr="fail")
                ),
            )

    def test_cilium_config_read_failure(self) -> None:
        with pytest.raises(click.ClickException, match="Could not read Cilium"):
            enable_cilium_gateway_api(
                env={},
                run=MagicMock(
                    return_value=MagicMock(returncode=1, stdout="", stderr="denied")
                ),
            )

    def test_kubectl_wrapper_failure(self) -> None:
        with pytest.raises(click.ClickException, match="kubectl patch"):
            _kubectl(
                ["patch", "configmap", "cilium-config"],
                env={},
                run=MagicMock(
                    return_value=MagicMock(returncode=1, stdout="", stderr="denied")
                ),
            )


class TestTerraformStateHelpers:
    def test_storage_project_id_requires_a_value(self) -> None:
        with pytest.raises(click.ClickException, match="storage project is unknown"):
            storage_project_id(
                _nebius_spec(storage_project_id=None, region="eu-north1")
            )

    def test_bucket_lookup_failure_is_not_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(returncode=1, stdout="", stderr="permission denied")
        )

        with pytest.raises(click.ClickException, match="Could not look up"):
            terraform_state_bucket_exists(_nebius_spec(), run=run)

    def test_import_ids_require_bucket_when_iam_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["iam", "service-account", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "sa-storage"}}),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        with pytest.raises(click.ClickException, match="bucket"):
            experiment_storage_import_ids(
                _nebius_spec(), run=MagicMock(side_effect=run_side_effect)
            )

    def test_import_ids_require_group(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:4] == ["storage", "bucket", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "bucket-1"}}),
                    stderr="",
                )
            if argv[1:4] == ["iam", "service-account", "get-by-name"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"metadata": {"id": "sa-storage"}}),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        with pytest.raises(click.ClickException, match="group"):
            experiment_storage_import_ids(
                _nebius_spec(), run=MagicMock(side_effect=run_side_effect)
            )

    def test_read_credentials_from_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        state_dir = tmp_path / "terraform"
        state_dir.mkdir()
        (tmp_path / "tfstate-credentials.json").write_text(
            json.dumps(
                {
                    "aws_access_key_id": "AKIA-file",
                    "aws_secret_access_key": "file-secret",
                }
            ),
            encoding="utf-8",
        )

        creds = read_terraform_state_credentials(state_dir)

        assert creds is not None
        assert creds.access_key_id == "AKIA-file"

    def test_load_credentials_requires_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        with pytest.raises(click.ClickException, match="AWS_ACCESS_KEY_ID"):
            load_terraform_state_credentials(tmp_path / "terraform")

    def test_delete_state_raises_on_http_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

        def opener(request: object, timeout: int = 30) -> MagicMock:
            raise HTTPError(
                getattr(request, "full_url", ""),
                500,
                "Server Error",
                hdrs={},
                fp=None,
            )

        with pytest.raises(click.ClickException, match="Could not delete"):
            delete_cluster_terraform_state(
                _nebius_spec(), state_dir=tmp_path / "state", opener=opener
            )

    def test_nebius_cmd_requires_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.providers.nebius.terraform.resolve_kubeconfig_executable",
            lambda _command: None,
        )

        with pytest.raises(click.ClickException, match="Nebius CLI not found"):
            _nebius_cmd("storage", "bucket", "list")

    def test_run_nebius_json_requires_an_object(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="[]", stderr=""))

        with pytest.raises(click.ClickException, match="invalid JSON"):
            _run_nebius_json(["storage", "bucket", "list"], run=run)

    def test_get_by_name_requires_an_object(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="[]", stderr=""))

        with pytest.raises(click.ClickException, match="invalid JSON"):
            _get_by_name(["storage", "bucket", "get-by-name"], run=run)

    def test_get_by_name_raises_on_unexpected_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(
            return_value=MagicMock(returncode=1, stdout="", stderr="denied")
        )

        with pytest.raises(click.ClickException, match=r"nebius .* failed"):
            _get_by_name(["storage", "bucket", "get-by-name"], run=run)

    def test_find_named_resource_falls_back_to_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if "get-by-name" in argv:
                return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {"items": [{"metadata": {"id": "bucket-1", "name": "bucket"}}]}
                ),
                stderr="",
            )

        found = _find_named_resource(
            ["storage", "bucket"],
            name="bucket",
            parent_id="project",
            run=MagicMock(side_effect=run_side_effect),
        )

        assert found is not None
        assert _resource_id(found) == "bucket-1"

    def test_resource_name_reads_top_level_name(self) -> None:
        assert _resource_name({"name": " bucket "}) == "bucket"

    def test_parse_nebius_json_rejects_invalid_payload(self) -> None:
        with pytest.raises(click.ClickException, match="invalid JSON"):
            _parse_nebius_json("{nope")

    def test_payload_items_accepts_a_list(self) -> None:
        assert _payload_items([{"id": "1"}, "skip"]) == [{"id": "1"}]
        with pytest.raises(click.ClickException, match="invalid JSON"):
            _payload_items(None)
        with pytest.raises(click.ClickException, match="invalid JSON"):
            _payload_items(12)

    def test_member_id_reads_spec(self) -> None:
        assert _member_id({"spec": {"member_id": " sa-1 "}}) == "sa-1"

    def test_storage_access_key_id_requires_a_match(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="{}", stderr=""))

        with pytest.raises(click.ClickException, match="No access key named"):
            _storage_access_key_id("sa-1", "missing", run=run)

    def test_resource_id_requires_metadata(self) -> None:
        with pytest.raises(click.ClickException, match=r"missing metadata\.id"):
            _resource_id({})

    def test_credentials_from_fields_require_both_keys(self) -> None:
        with pytest.raises(click.ClickException, match="missing AWS credentials"):
            _credentials_from_fields({"aws_access_key_id": "AKIA"})

    def test_resource_name_returns_empty_when_missing(self) -> None:
        assert _resource_name({}) == ""

    def test_next_page_token_ignores_non_mappings(self) -> None:
        assert _next_page_token([]) == ""

    def test_member_id_reads_top_level_field(self) -> None:
        assert _member_id({"member_id": " sa-1 "}) == "sa-1"

    def test_member_id_returns_empty_when_missing(self) -> None:
        assert _member_id({}) == ""

    def test_list_items_raise_on_unexpected_errors(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if "get-by-name" in argv:
                return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")
            return MagicMock(returncode=1, stdout="", stderr="permission denied")

        with pytest.raises(click.ClickException, match=r"nebius .* failed"):
            _find_named_resource(
                ["storage", "bucket"],
                name="bucket",
                parent_id="project",
                run=MagicMock(side_effect=run_side_effect),
            )

    def test_existing_access_key_is_fetched(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, nebius_cli = _set_nebius_auth_env(monkeypatch, tmp_path)
        monkeypatch.setenv("NEBIUS_CLI", str(nebius_cli))

        def run_side_effect(argv: list[str], **_kwargs: object) -> MagicMock:
            if argv[1:5] == ["iam", "v2", "access-key", "list-by-account"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "items": [
                                {
                                    "metadata": {
                                        "id": "key-1",
                                        "name": "test-tfstate",
                                    }
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            if argv[1:5] == ["iam", "v2", "access-key", "get-secret"]:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "aws_access_key_id": "AKIA-existing",
                            "secret": "existing-secret",
                        }
                    ),
                    stderr="",
                )
            return MagicMock(returncode=1, stdout="", stderr="NOT_FOUND")

        creds = _ensure_state_access_key(
            "sa-1",
            name="test-tfstate",
            project_id="storage-project",
            run=MagicMock(side_effect=run_side_effect),
        )

        assert creds.access_key_id == "AKIA-existing"


def _spec_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": "nebius",
        "name": "test",
        "terraform_state": {"bucket": "test-tfstate"},
        "arena": {
            "storage": {
                "bucket": "bucket",
                "endpoint": "https://storage.example.com",
            },
            "inference": {"domain": "inference.example.com"},
        },
        "nebius": {
            "tenant_id": "tenant",
            "project_id": "project",
            "storage_project_id": "storage-project",
            "subnet_id": "subnet",
            "object_storage": {"size_gib": 1024},
        },
    }
    payload.update(updates)
    return payload


class TestClusterSpecErrors:
    def test_rejects_blank_state_bucket(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            ClusterSpec.model_validate(_spec_payload(terraform_state={"bucket": "   "}))

    def test_blank_optional_state_fields_are_unset(self) -> None:
        spec = ClusterSpec.model_validate(
            _spec_payload(
                terraform_state={
                    "bucket": "test-tfstate",
                    "key": "  ",
                    "endpoint": "  ",
                }
            )
        )

        assert spec.terraform_state.key is None
        assert spec.terraform_state.endpoint is None

    def test_blank_tls_secret_is_unset(self) -> None:
        spec = _nebius_spec(
            inference={"domain": "inference.example.com", "tls_secret_name": "  "}
        )

        assert spec.arena.inference.tls_secret_name is None

    def test_blank_service_account_is_unset(self) -> None:
        spec = _nebius_spec(service_account_id="  ")

        assert spec.nebius.service_account_id is None

    def test_rejects_values_section(self) -> None:
        with pytest.raises(
            ValueError, match="values and registration are not accepted"
        ):
            ClusterSpec.model_validate(_spec_payload(values={"tenant_id": "tenant"}))

    def test_rejects_aws_block_for_nebius(self) -> None:
        with pytest.raises(ValueError, match="aws block is not allowed"):
            ClusterSpec.model_validate(
                _spec_payload(
                    aws={
                        "region": "eu-west-1",
                        "object_storage": {"size_gib": 1024},
                    }
                )
            )

    def test_requires_the_nebius_block(self) -> None:
        payload = _spec_payload()
        del payload["nebius"]

        with pytest.raises(ValueError, match="nebius block is required"):
            ClusterSpec.model_validate(payload)

    def test_render_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unsupported cluster spec provider"):
            render_default_cluster_spec(provider="gcp", name="arena")

    def test_render_rejects_blank_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            render_default_cluster_spec(provider="nebius", name="  ")

    def test_write_rejects_existing_file_without_force(self, tmp_path: Path) -> None:
        path = tmp_path / "cluster.yaml"
        path.write_text("existing\n", encoding="utf-8")

        with pytest.raises(ValueError, match="already exists"):
            write_default_cluster_spec(path, provider="nebius", name="arena")

    def test_load_wraps_os_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Could not read"):
            load_cluster_spec(tmp_path / "missing.yaml")

    def test_load_wraps_yaml_errors(self, tmp_path: Path) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text("not: valid: yaml: [\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_cluster_spec(spec)

    def test_load_requires_a_mapping(self, tmp_path: Path) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text("- item\n", encoding="utf-8")

        with pytest.raises(ValueError, match="YAML mapping"):
            load_cluster_spec(spec)

    def test_on_prem_row_requires_a_name(self) -> None:
        with pytest.raises(ValueError, match="missing a name"):
            cluster_spec_from_on_prem_row({})

    def test_on_prem_row_requires_config_mapping(self) -> None:
        with pytest.raises(ValueError, match="no stored Nebius"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "byoc_provider": {"provider": "nebius", "config": "nope"},
                }
            )

    def test_on_prem_row_requires_state_bucket(self) -> None:
        with pytest.raises(ValueError, match="no Terraform state bucket"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {"tenant_id": "t", "terraform_state": {}},
                    },
                }
            )

    def test_on_prem_row_requires_storage_settings(self) -> None:
        with pytest.raises(ValueError, match="tenant or object storage"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {"terraform_state": {"bucket": "state"}},
                    },
                }
            )

    def test_on_prem_row_requires_domain(self) -> None:
        with pytest.raises(ValueError, match="inference domain"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "storage_bucket": "arena-data",
                    "storage_endpoint": "https://storage.example.com",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {
                            "tenant_id": "tenant-1",
                            "terraform_state": {"bucket": "state"},
                            "object_storage_size_gib": 1024,
                        },
                    },
                }
            )

    def test_on_prem_row_requires_storage_size(self) -> None:
        with pytest.raises(ValueError, match="object storage size"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "storage_bucket": "arena-data",
                    "storage_endpoint": "https://storage.example.com",
                    "domain": "inference.example.com",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {
                            "tenant_id": "tenant-1",
                            "terraform_state": {"bucket": "state"},
                        },
                    },
                }
            )

    def test_on_prem_row_rejects_non_list_workers(self) -> None:
        with pytest.raises(ValueError, match="invalid stored worker"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "storage_bucket": "arena-data",
                    "storage_endpoint": "https://storage.example.com",
                    "domain": "inference.example.com",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {
                            "tenant_id": "tenant-1",
                            "terraform_state": {"bucket": "state"},
                            "object_storage_size_gib": 1024,
                            "workers": {"name": "gpu"},
                        },
                    },
                }
            )

    def test_on_prem_row_rejects_non_dict_worker(self) -> None:
        with pytest.raises(ValueError, match="invalid stored worker"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "storage_bucket": "arena-data",
                    "storage_endpoint": "https://storage.example.com",
                    "domain": "inference.example.com",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {
                            "tenant_id": "tenant-1",
                            "terraform_state": {"bucket": "state"},
                            "object_storage_size_gib": 1024,
                            "workers": ["gpu"],
                        },
                    },
                }
            )

    def test_on_prem_row_wraps_invalid_stored_settings(self) -> None:
        with pytest.raises(ValueError, match="stored Nebius settings are invalid"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "storage_bucket": "arena-data",
                    "storage_endpoint": "https://storage.example.com",
                    "domain": "not a domain",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {
                            "tenant_id": "tenant-1",
                            "terraform_state": {"bucket": "state"},
                            "object_storage_size_gib": 1024,
                        },
                    },
                }
            )

    def test_blank_optional_ids_are_unset(self) -> None:
        spec = ClusterSpec.model_validate(
            _spec_payload(
                terraform_state={
                    "bucket": "test-tfstate",
                    "key": None,
                    "endpoint": None,
                },
                nebius={
                    "tenant_id": "tenant",
                    "project_id": "project",
                    "storage_project_id": "storage-project",
                    "subnet_id": "subnet",
                    "service_account_id": None,
                    "object_storage": {"size_gib": 1024},
                },
                arena={
                    "storage": {
                        "bucket": "bucket",
                        "endpoint": "https://storage.example.com",
                    },
                    "inference": {
                        "domain": "inference.example.com",
                        "tls_secret_name": None,
                    },
                },
            )
        )

        assert spec.terraform_state.key is None
        assert spec.nebius.service_account_id is None
        assert spec.arena.inference.tls_secret_name is None

    def test_on_prem_row_rejects_zero_storage_size(self) -> None:
        with pytest.raises(ValueError, match="object storage size"):
            cluster_spec_from_on_prem_row(
                {
                    "name": "arena-nebius",
                    "storage_bucket": "arena-data",
                    "storage_endpoint": "https://storage.example.com",
                    "domain": "inference.example.com",
                    "byoc_provider": {
                        "provider": "nebius",
                        "config": {
                            "tenant_id": "tenant-1",
                            "terraform_state": {"bucket": "state"},
                            "object_storage_size_gib": 0,
                        },
                    },
                }
            )

    def test_on_prem_row_keeps_state_endpoint_and_tls_secret(self) -> None:
        spec = cluster_spec_from_on_prem_row(
            {
                "name": "arena-nebius",
                "storage_bucket": "arena-data",
                "storage_endpoint": "https://storage.example.com",
                "domain": "inference.example.com",
                "tls_secret_name": "arena-tls",
                "byoc_provider": {
                    "provider": "nebius",
                    "config": {
                        "tenant_id": "tenant-1",
                        "project_id": "project-1",
                        "storage_project_id": "storage-project-1",
                        "terraform_state": {
                            "bucket": "state",
                            "endpoint": "https://s3.stored",
                        },
                        "object_storage_size_gib": 1024,
                    },
                },
            }
        )

        assert spec.terraform_state.endpoint == "https://s3.stored"
        assert spec.arena.inference.tls_secret_name == "arena-tls"
