# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nebius Terraform auth, kubeconfig, and post-apply cluster setup."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import click

from agilerl.arena.byoc.cluster_register import ClusterResourceClass
from agilerl.arena.byoc.provisioning.gateway_api import configure_gateway_api
from agilerl.arena.byoc.provisioning.inference import (
    DEFAULT_INFERENCE_TLS_SECRET_NAME,
    normalize_inference_domain,
    resolve_inference_hostname_template,
)
from agilerl.arena.byoc.provisioning.providers.nebius.kubeconfig import (
    build_nebius_kubeconfig_argv,
    resolve_kubeconfig_executable,
)
from agilerl.arena.byoc.provisioning.providers.nebius.spec import (
    needs_stored_config,
    recover_stored_config,
    registration_payload,
    resource_classes,
    spec_from_stored_row,
)
from agilerl.arena.byoc.provisioning.providers.nebius.terraform import (
    delete_cluster_terraform_state,
    delete_nebius_project,
    delete_terraform_state_bucket,
    ensure_terraform_state_bucket,
    experiment_storage_import_ids,
    fetch_terraform_state_credentials,
    nebius_region_from_endpoint,
    read_terraform_state_credentials,
    terraform_state_bucket_exists,
    write_terraform_state_credentials,
)
from agilerl.arena.byoc.provisioning.runtime_class import ensure_gpu_runtime_class
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.storage import ensure_storage_secret
from agilerl.arena.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
    terraform_address_matches,
)

NEBIUS_SA_AUTH_ENV_VARS = (
    "AUTHKEY_PRIVATE_PATH",
    "AUTHKEY_PUBLIC_ID",
)
NEBIUS_PROVIDER_ENV_VARS = (*NEBIUS_SA_AUTH_ENV_VARS, "SA_ID")
NEBIUS_PROVIDER_SERVICE_ACCOUNT = """\
provider "nebius" {
  service_account = {
    private_key_file_env = "AUTHKEY_PRIVATE_PATH"
    public_key_id_env    = "AUTHKEY_PUBLIC_ID"
    account_id_env       = "SA_ID"
  }
}
"""
NEBIUS_PROVIDER_PROFILE = """\
provider "nebius" {
  profile = {}
}
"""

NEBIUS_COMPUTE_PROJECT_RESOURCE = "nebius_iam_v2_project.arena"
NEBIUS_STORAGE_PROJECT_RESOURCE = "nebius_iam_v2_project.storage"
NEBIUS_MK8S_CLUSTER_RESOURCE = "nebius_mk8s_v1_cluster.arena"

# Experiment data, metrics, and checkpoints live in these resources.
NEBIUS_OBJECT_STORAGE_RESOURCES = (
    "nebius_storage_v1_bucket.arena_data",
    "nebius_iam_v2_access_key.storage",
    "nebius_iam_v1_group_membership.storage_editor",
    "nebius_iam_v1_group.storage_editors",
    "nebius_iam_v1_service_account.storage",
)


def nebius_service_account_id() -> str:
    """Return the Terraform provider service-account ID, or empty if unset."""
    return os.environ.get("SA_ID", "").strip()


def resolve_nebius_service_account_id(spec: ClusterSpec) -> ClusterSpec:
    """Fill ``nebius.service_account_id`` from ``SA_ID`` when the spec omits it."""
    cloud = spec.nebius_cloud()
    if cloud.service_account_id:
        return spec
    sa_id = nebius_service_account_id()
    if not sa_id:
        return spec
    return spec.model_copy(
        update={"nebius": cloud.model_copy(update={"service_account_id": sa_id})}
    )


def write_nebius_provider_config(work_dir: Path) -> None:
    """Write Nebius provider auth: service-account keys, or the CLI profile."""
    work_dir.mkdir(parents=True, exist_ok=True)
    content = (
        NEBIUS_PROVIDER_SERVICE_ACCOUNT
        if nebius_service_account_id()
        else NEBIUS_PROVIDER_PROFILE
    )
    (work_dir / "provider.tf").write_text(content, encoding="utf-8")


def validate_nebius_spec(spec: ClusterSpec) -> None:
    """Validate inputs required by the packaged Nebius module."""
    if nebius_service_account_id():
        missing = [
            name
            for name in NEBIUS_SA_AUTH_ENV_VARS
            if not os.environ.get(name, "").strip()
        ]
        if missing:
            msg = (
                "Nebius Terraform service-account auth env vars are not set: "
                f"{', '.join(missing)}. "
                "Export AUTHKEY_PRIVATE_PATH (private key file) and "
                "AUTHKEY_PUBLIC_ID, or unset SA_ID to authenticate with the "
                "Nebius CLI profile."
            )
            raise click.ClickException(msg)
        private_key = Path(os.environ["AUTHKEY_PRIVATE_PATH"]).expanduser()
        if not private_key.is_file():
            msg = (
                "AUTHKEY_PRIVATE_PATH is not a file: "
                f"{private_key}. Point it at the Nebius authorized-key PEM."
            )
            raise click.ClickException(msg)
    if resolve_kubeconfig_executable(["nebius"]) is None:
        msg = (
            "Nebius CLI not found. Install it, set NEBIUS_CLI to its path, "
            "or add ~/.nebius/bin to PATH."
        )
        raise click.ClickException(msg)
    if shutil.which("kubectl") is None:
        msg = (
            "kubectl not found on PATH; install kubectl before provisioning "
            "the GPU RuntimeClass and Gateway API."
        )
        raise click.ClickException(msg)


def nebius_bootstrap_targets(spec: ClusterSpec) -> tuple[str, ...]:
    """Return Nebius project resources the spec does not already name."""
    targets = []
    cloud = spec.nebius_cloud()
    if cloud.project_id is None:
        targets.append(NEBIUS_COMPUTE_PROJECT_RESOURCE)
    if cloud.storage_project_id is None:
        targets.append(NEBIUS_STORAGE_PROJECT_RESOURCE)
    return tuple(targets)


def terraform_project_id(value: object, name: str) -> str:
    """Return a project id from one Terraform output."""
    if not isinstance(value, str) or not value.strip():
        msg = f"Terraform did not return the Nebius {name}."
        raise click.ClickException(msg)
    return value.strip()


def _worker_node_group_ids(value: object) -> dict[str, str]:
    """Return a name-to-id map of worker node groups."""
    if not isinstance(value, dict) or not value:
        return {}
    ids: dict[str, str] = {}
    for name, group_id in value.items():
        if not isinstance(name, str) or not name.strip():
            return {}
        if not isinstance(group_id, str) or not group_id.strip():
            return {}
        ids[name] = group_id
    return ids


def _optional_string(value: object) -> str | None:
    """Return non-empty output strings, otherwise None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


class NebiusProvider:
    """Provision Arena clusters on Nebius."""

    name = "nebius"
    display_name = "Nebius"
    terraform_module = "nebius"
    cluster_address = NEBIUS_MK8S_CLUSTER_RESOURCE
    provisioned_addresses = (NEBIUS_MK8S_CLUSTER_RESOURCE,)

    def validate(self, spec: ClusterSpec) -> None:
        """Reject a spec this provider cannot apply."""
        validate_nebius_spec(spec)

    def needs_stored_config(self, spec: ClusterSpec) -> bool:
        """Return whether Arena may still hold ids the spec omitted."""
        return needs_stored_config(spec)

    def recover(self, spec: ClusterSpec, stored: dict[str, object]) -> ClusterSpec:
        """Fill omitted ids from the config Arena stored for this cluster."""
        return recover_stored_config(spec, stored)

    def registration_payload(
        self,
        spec: ClusterSpec,
        project_id: str | None = None,
        storage_project_id: str | None = None,
    ) -> dict[str, object]:
        """Return the ``byocProvider`` body sent when registering the cluster."""
        return registration_payload(
            spec,
            project_id=project_id,
            storage_project_id=storage_project_id,
        )

    def resource_classes(
        self, spec: ClusterSpec, worker_node_group_ids: dict[str, str]
    ) -> list[ClusterResourceClass]:
        """Build GPU resource classes for provisioned Nebius worker groups."""
        return resource_classes(spec, worker_node_group_ids=worker_node_group_ids)

    def spec_from_stored_row(self, row: dict[str, object]) -> ClusterSpec:
        """Rebuild a Nebius cluster spec from an Arena on-prem cluster record."""
        return spec_from_stored_row(row)

    def prepare(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
        create_state_bucket: bool = False,
        prompt_create: bool = False,
    ) -> tuple[ClusterSpec, Path]:
        """Validate the spec, write provider auth, and open a Terraform work dir."""
        spec = resolve_nebius_service_account_id(spec)
        validate_nebius_spec(spec)
        work_dir = runner.prepare(
            spec, module_name=self.terraform_module, state_dir=state_dir
        )
        write_nebius_provider_config(work_dir)
        credentials = None
        bootstrap_targets = nebius_bootstrap_targets(spec)
        if bootstrap_targets:
            credentials = read_terraform_state_credentials(state_dir)
            if credentials is None:
                runner.initialize_without_backend(work_dir)
                runner.apply_targets(work_dir, bootstrap_targets)
            else:
                runner.extra_env.update(credentials.environ())
                runner.initialize(work_dir, spec)
                present = runner.state_list(work_dir)
                missing_targets = tuple(
                    target
                    for target in bootstrap_targets
                    if not any(
                        terraform_address_matches(address, (target,))
                        for address in present
                    )
                )
                if missing_targets:
                    runner.apply_targets(work_dir, missing_targets)
            spec = spec.model_copy(
                update={
                    "nebius": spec.nebius_cloud().model_copy(
                        update={
                            "project_id": terraform_project_id(
                                runner.terraform_output(work_dir, "project_id"),
                                "project_id",
                            ),
                            "storage_project_id": terraform_project_id(
                                runner.terraform_output(work_dir, "storage_project_id"),
                                "storage_project_id",
                            ),
                        }
                    )
                }
            )
        prompt: Callable[[str], bool] | None = click.confirm if prompt_create else None
        runner.extra_env.update(
            ensure_terraform_state_bucket(
                spec,
                state_dir=state_dir,
                create=create_state_bucket,
                prompt=prompt,
            )
        )
        return spec, work_dir

    def open_state(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
    ) -> Path | None:
        """Initialize remote Terraform state, or ``None`` when nothing was provisioned."""
        credentials = read_terraform_state_credentials(state_dir)
        if credentials is None:
            if spec.nebius_cloud().storage_project_id is None:
                return None
            if not terraform_state_bucket_exists(spec):
                return None
            credentials = fetch_terraform_state_credentials(spec)
            write_terraform_state_credentials(state_dir, credentials)
        runner.extra_env.update(credentials.environ())
        work_dir = runner.prepare(
            spec, module_name=self.terraform_module, state_dir=state_dir
        )
        write_nebius_provider_config(work_dir)
        runner.initialize(work_dir, spec)
        return work_dir

    def write_backend(self, work_dir: Path, spec: ClusterSpec) -> None:
        """Write the Nebius Object Storage S3 backend for Terraform state."""
        state = spec.terraform_state
        endpoint = state.s3_endpoint(spec.arena.storage.endpoint)
        region = nebius_region_from_endpoint(endpoint)
        key = state.object_key(spec.name)
        (work_dir / "backend.tf").write_text(
            "\n".join(
                [
                    "terraform {",
                    '  backend "s3" {',
                    f"    bucket                      = {json.dumps(state.bucket)}",
                    f"    key                         = {json.dumps(key)}",
                    f"    region                      = {json.dumps(region)}",
                    f"    endpoints                   = {{ s3 = {json.dumps(endpoint)} }}",
                    "    use_lockfile                = true",
                    "    skip_credentials_validation = true",
                    "    skip_region_validation      = true",
                    "    skip_requesting_account_id  = true",
                    "    skip_metadata_api_check     = true",
                    "  }",
                    "}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def cluster_outputs(
        self,
        values: dict[str, object],
        output_dir: Path,
        run: Callable[..., subprocess.CompletedProcess[str]],
    ) -> ClusterOutputs:
        """Turn Terraform outputs into a kubeconfig and storage credentials."""
        required = (
            "cluster_name",
            "context",
            "cluster_id",
            "storage_access_key_id",
            "storage_secret_access_key",
            "worker_node_group_ids",
        )
        missing = [name for name in required if not values.get(name)]
        if missing:
            msg = f"Terraform outputs missing required values: {', '.join(missing)}."
            raise click.ClickException(msg)
        worker_ids = _worker_node_group_ids(values.get("worker_node_group_ids"))
        if not worker_ids:
            msg = (
                "Terraform output worker_node_group_ids must be a non-empty string map."
            )
            raise click.ClickException(msg)
        kubeconfig_path = absolute_path(output_dir) / "kubeconfig"
        kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_id = values["cluster_id"]
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            msg = "Terraform output cluster_id must be a non-empty string."
            raise click.ClickException(msg)
        context = str(values["context"])
        if resolve_kubeconfig_executable(["nebius"]) is None:
            msg = (
                "Nebius CLI not found. Install it, set NEBIUS_CLI to its path, "
                "or add ~/.nebius/bin to PATH."
            )
            raise click.ClickException(msg)
        argv = build_nebius_kubeconfig_argv(
            cluster_id=cluster_id,
            context_name=context,
            kubeconfig_path=kubeconfig_path,
        )
        try:
            result = run(argv, check=False, text=True)
        except FileNotFoundError as exc:
            msg = (
                f"Nebius CLI not found: {argv[0]!r}. "
                "Install it, set NEBIUS_CLI to its path, or add ~/.nebius/bin to PATH."
            )
            raise click.ClickException(msg) from exc
        if result.returncode != 0:
            msg = (
                "Could not retrieve kubeconfig from Nebius "
                f"({argv[0]} exited {result.returncode})."
            )
            raise click.ClickException(msg)
        if not kubeconfig_path.is_file():
            msg = "Nebius CLI did not write a kubeconfig file."
            raise click.ClickException(msg)
        return ClusterOutputs(
            cluster_name=str(values["cluster_name"]),
            context=str(values["context"]),
            kubeconfig_path=kubeconfig_path,
            storage_access_key_id=str(values["storage_access_key_id"]),
            storage_secret_access_key=str(values["storage_secret_access_key"]),
            worker_node_group_ids=worker_ids,
            storage_class_name=_optional_string(values.get("storage_class_name")),
            storage_endpoint=_optional_string(values.get("storage_endpoint")),
            storage_bucket=_optional_string(values.get("storage_bucket")),
        )

    def install_cni(
        self, _spec: ClusterSpec, _work_dir: Path, _runner: TerraformRunner
    ) -> None:
        """Nebius nodes join with the cluster CNI already running."""

    def post_apply(self, spec: ClusterSpec, outputs: ClusterOutputs) -> ClusterOutputs:
        """Install storage credentials, the GPU RuntimeClass, and Gateway API."""
        arena = spec.arena
        cloud = spec.nebius_cloud()
        inference = arena.inference
        inference_domain = normalize_inference_domain(inference.domain)
        hostname_template = resolve_inference_hostname_template(
            inference.hostname_template,
        )
        tls_secret_name = inference.tls_secret_name
        outputs = replace(
            outputs,
            project_id=cloud.project_id,
            storage_project_id=cloud.storage_project_id,
            inference_domain=inference_domain,
            inference_hostname_template=hostname_template,
            inference_tls_secret_name=tls_secret_name
            or DEFAULT_INFERENCE_TLS_SECRET_NAME,
        )
        if not arena.storage.install:
            ensure_storage_secret(
                kubeconfig_path=outputs.kubeconfig_path,
                secret_name=arena.storage.resolved_secret_name(),
                access_key_id=outputs.storage_access_key_id,
                secret_access_key=outputs.storage_secret_access_key,
                endpoint=arena.storage.endpoint,
            )
        ensure_gpu_runtime_class(kubeconfig_path=outputs.kubeconfig_path)
        parent_refs = self.ensure_gateway(spec, outputs.kubeconfig_path, {})
        if parent_refs is None:
            return outputs
        return replace(outputs, gateway_api_parent_refs=parent_refs)

    def ensure_gateway(
        self,
        spec: ClusterSpec,
        kubeconfig_path: Path,
        _terraform_outputs: dict[str, object],
    ) -> list[dict[str, str]] | None:
        """Create the Cilium Gateway when the spec enables it."""
        arena = spec.arena
        if not arena.gateway.enable:
            return None
        return configure_gateway_api(
            kubeconfig_path=kubeconfig_path,
            gateway_name=arena.gateway.name,
            domain=normalize_inference_domain(arena.inference.domain),
            tls_secret_name=arena.inference.tls_secret_name,
        )

    def import_ids(self, spec: ClusterSpec) -> dict[str, str]:
        """Return existing experiment-storage resources Terraform should adopt."""
        return experiment_storage_import_ids(spec)

    def assert_can_delete_storage_project(
        self, spec: ClusterSpec, delete_storage_project: bool
    ) -> None:
        """Reject deletion of a storage project the spec did not create."""
        storage_project_id = spec.nebius_cloud().storage_project_id
        if delete_storage_project and storage_project_id is not None:
            msg = (
                "--delete-storage-project only deletes a Nebius project that "
                "provisioning created. Remove nebius.storage_project_id from the "
                f"spec, or delete project {storage_project_id!r} in Nebius."
            )
            raise click.ClickException(msg)

    def created_projects(self, spec: ClusterSpec) -> tuple[bool, bool]:
        """Return whether compute and storage projects were omitted from the spec."""
        cloud = spec.nebius_cloud()
        return (
            cloud.project_id is None,
            cloud.storage_project_id is None,
        )

    def delete_gateway(
        self,
        _spec: ClusterSpec,
        _work_dir: Path,
        _runner: TerraformRunner,
    ) -> None:
        """The Gateway is deleted with the Kubernetes cluster."""

    def destroy_exclusions(
        self, _spec: ClusterSpec, delete_storage: bool
    ) -> tuple[str, ...]:
        """Return Terraform addresses ``destroy`` must leave in place."""
        exclude = () if delete_storage else NEBIUS_OBJECT_STORAGE_RESOURCES
        return (
            *exclude,
            NEBIUS_COMPUTE_PROJECT_RESOURCE,
            NEBIUS_STORAGE_PROJECT_RESOURCE,
        )

    def finish_destroy(
        self,
        spec: ClusterSpec,
        delete_compute_project: bool,
        delete_storage_project: bool,
    ) -> None:
        """Delete projects and the state bucket after Terraform destroy."""
        cloud = spec.nebius_cloud()
        if delete_compute_project:
            project_id = cloud.project_id
            if project_id is None:
                msg = "Terraform did not return the Nebius project_id."
                raise click.ClickException(msg)
            delete_nebius_project(project_id)
        if delete_storage_project:
            storage_project_id = cloud.storage_project_id
            if storage_project_id is None:
                msg = "Terraform did not return the Nebius storage_project_id."
                raise click.ClickException(msg)
            delete_terraform_state_bucket(spec)
            delete_nebius_project(storage_project_id)

    def delete_terraform_state(self, spec: ClusterSpec, state_dir: Path) -> None:
        """Delete this cluster's Terraform state object. The state bucket is kept."""
        delete_cluster_terraform_state(spec, state_dir=state_dir)


NEBIUS = NebiusProvider()
