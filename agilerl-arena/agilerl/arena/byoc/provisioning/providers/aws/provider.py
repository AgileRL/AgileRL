# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""AWS Terraform auth, kubeconfig, and post-apply cluster setup."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import click

from agilerl.arena.byoc.provisioning.inference import (
    DEFAULT_INFERENCE_TLS_SECRET_NAME,
    normalize_inference_domain,
    resolve_inference_hostname_template,
)
from agilerl.arena.byoc.provisioning.providers.aws.autoscaler import (
    ensure_cluster_autoscaler,
)
from agilerl.arena.byoc.provisioning.providers.aws.cilium import (
    api_server_host,
    ensure_cilium,
    ensure_coredns_addon,
)
from agilerl.arena.byoc.provisioning.providers.aws.filesystem import (
    ensure_efs_storage_class,
)
from agilerl.arena.byoc.provisioning.providers.aws.gateway import (
    configure_eks_gateway,
    delete_eks_gateway,
)
from agilerl.arena.byoc.provisioning.providers.aws.gpu import (
    ensure_gpu_startup_taint,
    ensure_nvidia_device_plugin,
)
from agilerl.arena.byoc.provisioning.providers.aws.kubeconfig import (
    build_eks_kubeconfig_argv,
    resolve_aws_executable,
)
from agilerl.arena.byoc.provisioning.providers.aws.spec import (
    AwsSpec,
    registration_payload,
    resource_classes,
    spec_from_stored_row,
)
from agilerl.arena.byoc.provisioning.providers.aws.terraform import (
    TFSTATE_BACKEND_CREDENTIALS_FILE,
    TFSTATE_BACKEND_PROFILE,
    credentials_path,
    delete_cluster_terraform_state,
    ensure_terraform_state_bucket,
    ensure_terraform_state_credentials,
    read_terraform_state_credentials,
    terraform_state_bucket_exists,
    write_terraform_state_profile,
)
from agilerl.arena.byoc.provisioning.runtime_class import ensure_gpu_runtime_class
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.storage import ensure_storage_secret
from agilerl.arena.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
)

AWS_EKS_CLUSTER_RESOURCE = "aws_eks_cluster.arena"
AWS_EKS_SYSTEM_NODE_GROUP_RESOURCE = "aws_eks_node_group.arena"
AWS_OBJECT_STORAGE_RESOURCES = (
    "aws_s3_bucket.arena_data",
    "aws_s3_bucket_public_access_block.arena_data",
    "aws_iam_user.storage",
    "aws_iam_access_key.storage",
    "aws_iam_user_policy.storage",
)


def _aws_values(spec: ClusterSpec) -> AwsSpec:
    values = spec.aws
    if values is not None:
        return values
    msg = f"Cluster {spec.name!r} is not an AWS spec."
    raise click.ClickException(msg)


def _worker_node_group_ids(value: object) -> dict[str, str]:
    """Return worker node group ids, or an empty map when there are none."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "Terraform output worker_node_group_ids must be a string map."
        raise click.ClickException(msg)
    ids: dict[str, str] = {}
    for name, group_id in value.items():
        if not isinstance(name, str) or not name.strip():
            msg = "Terraform output worker_node_group_ids must be a string map."
            raise click.ClickException(msg)
        if not isinstance(group_id, str) or not group_id.strip():
            msg = "Terraform output worker_node_group_ids must be a string map."
            raise click.ClickException(msg)
        ids[name] = group_id
    return ids


def _optional_string(value: object) -> str | None:
    """Return non-empty output strings, otherwise None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


class AwsProvider:
    """Provision Arena clusters on AWS."""

    name = "aws"
    display_name = "AWS"
    terraform_module = "aws"
    cluster_address = AWS_EKS_CLUSTER_RESOURCE
    # The CNI install applies only the cluster. This node group marks a finished apply.
    provisioned_addresses = (
        AWS_EKS_CLUSTER_RESOURCE,
        AWS_EKS_SYSTEM_NODE_GROUP_RESOURCE,
    )

    def validate(self, spec: ClusterSpec) -> None:
        """Reject a spec this provider cannot apply."""
        _aws_values(spec)
        if resolve_aws_executable() is None:
            msg = "AWS CLI not found. Install it and add aws to PATH."
            raise click.ClickException(msg)

    def needs_stored_config(self, spec: ClusterSpec) -> bool:
        """Return whether Arena may still hold ids the spec omitted."""
        _aws_values(spec)
        return False

    def recover(self, spec: ClusterSpec, stored: dict[str, object]) -> ClusterSpec:
        """Fill omitted ids from the config Arena stored for this cluster."""
        _aws_values(spec)
        return spec

    def registration_payload(
        self,
        spec: ClusterSpec,
        project_id: str | None = None,
        storage_project_id: str | None = None,
    ) -> dict[str, object]:
        """Return the ``byocProvider`` body sent when registering the cluster."""
        return registration_payload(spec)

    def resource_classes(
        self, spec: ClusterSpec, worker_node_group_ids: dict[str, str]
    ) -> list:
        """Build GPU resource classes for provisioned EKS worker groups."""
        return resource_classes(spec, worker_node_group_ids=worker_node_group_ids)

    def spec_from_stored_row(self, row: dict[str, object]) -> ClusterSpec:
        """Rebuild an AWS cluster spec from an Arena on-prem cluster record."""
        return spec_from_stored_row(row)

    def prepare(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
        create_state_bucket: bool = False,
        prompt_create: bool = False,
    ) -> tuple[ClusterSpec, Path]:
        """Validate the spec and open a Terraform work dir."""
        self.validate(spec)
        work_dir = runner.prepare(
            spec, module_name=self.terraform_module, state_dir=state_dir
        )
        prompt: Callable[[str], bool] | None = click.confirm if prompt_create else None
        ensure_terraform_state_bucket(
            spec,
            state_dir=state_dir,
            create=create_state_bucket,
            prompt=prompt,
        )
        return spec, work_dir

    def open_state(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
    ) -> Path | None:
        """Initialize remote Terraform state, or ``None`` when nothing was provisioned."""
        if not terraform_state_bucket_exists(spec):
            return None
        ensure_terraform_state_credentials(spec, state_dir=state_dir)
        work_dir = runner.prepare(
            spec, module_name=self.terraform_module, state_dir=state_dir
        )
        runner.initialize(work_dir, spec)
        return work_dir

    def write_backend(self, work_dir: Path, spec: ClusterSpec) -> None:
        """Write the S3 backend for Terraform state, authenticated as the state IAM user."""
        values = _aws_values(spec)
        state = spec.terraform_state
        key = state.object_key(spec.name)
        credentials = read_terraform_state_credentials(work_dir)
        if credentials is None:
            msg = (
                "Terraform state credentials are missing. Restore "
                f"{credentials_path(work_dir)}."
            )
            raise click.ClickException(msg)
        # Keys stay out of backend.tf so rotating them does not change the backend config.
        write_terraform_state_profile(work_dir, credentials)
        lines = [
            "terraform {",
            '  backend "s3" {',
            f"    bucket       = {json.dumps(state.bucket)}",
            f"    key          = {json.dumps(key)}",
            f"    region       = {json.dumps(values.region)}",
            "    use_lockfile = true",
            f"    profile      = {json.dumps(TFSTATE_BACKEND_PROFILE)}",
            (
                "    shared_credentials_files = "
                f"[{json.dumps(TFSTATE_BACKEND_CREDENTIALS_FILE)}]"
            ),
        ]
        if state.endpoint:
            endpoint = state.s3_endpoint(spec.arena.storage.endpoint)
            lines.extend(
                [
                    f"    endpoints                   = {{ s3 = {json.dumps(endpoint)} }}",
                    "    skip_credentials_validation = true",
                    "    skip_region_validation      = true",
                    "    skip_requesting_account_id  = true",
                    "    skip_metadata_api_check     = true",
                ]
            )
        lines.extend(["  }", "}", ""])
        (work_dir / "backend.tf").write_text("\n".join(lines), encoding="utf-8")

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
            "region",
            "cluster_endpoint",
            "storage_access_key_id",
            "storage_secret_access_key",
        )
        missing = [name for name in required if not values.get(name)]
        if missing or "worker_node_group_ids" not in values:
            listed = missing or ["worker_node_group_ids"]
            msg = f"Terraform outputs missing required values: {', '.join(listed)}."
            raise click.ClickException(msg)
        worker_ids = _worker_node_group_ids(values.get("worker_node_group_ids"))
        kubeconfig_path = absolute_path(output_dir) / "kubeconfig"
        kubeconfig_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_name = values["cluster_name"]
        region = values["region"]
        if not isinstance(cluster_name, str) or not isinstance(region, str):
            msg = "Terraform outputs cluster_name and region must be strings."
            raise click.ClickException(msg)
        if resolve_aws_executable() is None:
            msg = "AWS CLI not found. Install it and add aws to PATH."
            raise click.ClickException(msg)
        argv = build_eks_kubeconfig_argv(
            cluster_name=cluster_name,
            region=region,
            kubeconfig_path=kubeconfig_path,
        )
        try:
            result = run(argv, check=False, text=True)
        except FileNotFoundError as exc:
            msg = "AWS CLI not found. Install it and add aws to PATH."
            raise click.ClickException(msg) from exc
        if result.returncode != 0:
            msg = (
                "Could not retrieve kubeconfig from EKS "
                f"({argv[0]} exited {result.returncode})."
            )
            raise click.ClickException(msg)
        if not kubeconfig_path.is_file():
            msg = "AWS CLI did not write a kubeconfig file."
            raise click.ClickException(msg)
        return ClusterOutputs(
            cluster_name=cluster_name,
            context=str(values["context"]),
            kubeconfig_path=kubeconfig_path,
            storage_access_key_id=str(values["storage_access_key_id"]),
            storage_secret_access_key=str(values["storage_secret_access_key"]),
            worker_node_group_ids=worker_ids,
            storage_class_name=_optional_string(values.get("storage_class_name")),
            storage_endpoint=_optional_string(values.get("storage_endpoint")),
            storage_bucket=_optional_string(values.get("storage_bucket")),
            efs_file_system_id=_optional_string(values.get("efs_file_system_id")),
            vpc_id=_optional_string(values.get("vpc_id")),
            cluster_endpoint=str(values["cluster_endpoint"]),
        )

    def install_cni(
        self, spec: ClusterSpec, work_dir: Path, runner: TerraformRunner
    ) -> None:
        """Install Cilium before node groups so nodes can become Ready."""
        runner.apply_targets(work_dir, (AWS_EKS_CLUSTER_RESOURCE,))
        endpoint = runner.terraform_output(work_dir, "cluster_endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            msg = "Terraform did not return the EKS API endpoint."
            raise click.ClickException(msg)
        values = _aws_values(spec)
        kubeconfig_path = work_dir / "kubeconfig"
        argv = build_eks_kubeconfig_argv(
            cluster_name=spec.name,
            region=values.region,
            kubeconfig_path=kubeconfig_path,
        )
        result = runner.run(argv, check=False, text=True)
        if result.returncode != 0:
            msg = "Could not retrieve kubeconfig from EKS."
            raise click.ClickException(msg)
        ensure_cilium(
            kubeconfig_path=kubeconfig_path,
            api_server_host=api_server_host(endpoint),
            wait=False,
        )

    def post_apply(self, spec: ClusterSpec, outputs: ClusterOutputs) -> ClusterOutputs:
        """Install Cilium, CoreDNS, storage, the device plugin, and Gateway API."""
        values = _aws_values(spec)
        endpoint = outputs.cluster_endpoint
        if not endpoint:
            msg = "Terraform did not return the EKS API endpoint."
            raise click.ClickException(msg)
        ensure_cilium(
            kubeconfig_path=outputs.kubeconfig_path,
            api_server_host=api_server_host(endpoint),
        )
        ensure_coredns_addon(cluster_name=outputs.cluster_name, region=values.region)
        arena = spec.arena
        inference = arena.inference
        inference_domain = normalize_inference_domain(inference.domain)
        hostname_template = resolve_inference_hostname_template(
            inference.hostname_template,
        )
        tls_secret_name = inference.tls_secret_name
        outputs = replace(
            outputs,
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
        if any(worker.num_gpus > 0 for worker in values.workers):
            ensure_nvidia_device_plugin(kubeconfig_path=outputs.kubeconfig_path)
            ensure_gpu_startup_taint(kubeconfig_path=outputs.kubeconfig_path)
        if values.workers:
            ensure_cluster_autoscaler(
                kubeconfig_path=outputs.kubeconfig_path,
                cluster_name=outputs.cluster_name,
                region=values.region,
            )
        if values.shared_filesystem.provision:
            file_system_id = outputs.efs_file_system_id
            if not file_system_id:
                msg = "Terraform did not return the EFS file system id."
                raise click.ClickException(msg)
            ensure_efs_storage_class(
                kubeconfig_path=outputs.kubeconfig_path,
                file_system_id=file_system_id,
            )
        parent_refs = self.ensure_gateway(
            spec,
            outputs.kubeconfig_path,
            {"vpc_id": outputs.vpc_id},
        )
        if parent_refs is None:
            return outputs
        return replace(outputs, gateway_api_parent_refs=parent_refs)

    def ensure_gateway(
        self,
        spec: ClusterSpec,
        kubeconfig_path: Path,
        terraform_outputs: dict[str, object],
    ) -> list[dict[str, str]] | None:
        """Create the Cilium Gateway when the spec enables it."""
        arena = spec.arena
        if not arena.gateway.enable:
            return None
        vpc_id = terraform_outputs.get("vpc_id")
        if not isinstance(vpc_id, str) or not vpc_id:
            msg = "Terraform did not return the VPC id."
            raise click.ClickException(msg)
        values = _aws_values(spec)
        return configure_eks_gateway(
            kubeconfig_path=kubeconfig_path,
            gateway_name=arena.gateway.name,
            domain=normalize_inference_domain(arena.inference.domain),
            cluster_name=spec.name,
            region=values.region,
            vpc_id=vpc_id,
            tls_secret_name=arena.inference.tls_secret_name
            or DEFAULT_INFERENCE_TLS_SECRET_NAME,
        )

    def import_ids(self, spec: ClusterSpec) -> dict[str, str]:
        """Return existing resources Terraform should adopt."""
        _aws_values(spec)
        return {}

    def assert_can_delete_storage_project(
        self, spec: ClusterSpec, delete_storage_project: bool
    ) -> None:
        """Reject deletion of a storage project AWS does not have."""
        _aws_values(spec)
        if delete_storage_project:
            msg = (
                "--delete-storage-project deletes a Nebius storage project. "
                "AWS clusters have no storage project. Pass --delete-storage to "
                "delete the experiment bucket."
            )
            raise click.ClickException(msg)

    def created_projects(self, spec: ClusterSpec) -> tuple[bool, bool]:
        """Return whether compute and storage projects were omitted from the spec."""
        _aws_values(spec)
        return (False, False)

    def delete_gateway(
        self,
        spec: ClusterSpec,
        work_dir: Path,
        runner: TerraformRunner,
    ) -> None:
        """Delete the Cilium Gateway so its NLB is released before Terraform destroy."""
        if not spec.arena.gateway.enable:
            return
        values = _aws_values(spec)
        kubeconfig_path = work_dir / "kubeconfig"
        argv = build_eks_kubeconfig_argv(
            cluster_name=spec.name,
            region=values.region,
            kubeconfig_path=kubeconfig_path,
        )
        result = runner.run(argv, check=False, text=True)
        if result.returncode != 0:
            msg = "Could not retrieve kubeconfig from EKS."
            raise click.ClickException(msg)
        delete_eks_gateway(
            kubeconfig_path=kubeconfig_path,
            gateway_name=spec.arena.gateway.name,
        )

    def destroy_exclusions(
        self, spec: ClusterSpec, delete_storage: bool
    ) -> tuple[str, ...]:
        """Return Terraform addresses ``destroy`` must leave in place."""
        _aws_values(spec)
        if delete_storage:
            return ()
        return AWS_OBJECT_STORAGE_RESOURCES

    def finish_destroy(
        self,
        spec: ClusterSpec,
        delete_compute_project: bool,
        delete_storage_project: bool,
    ) -> None:
        """AWS has no projects to delete after Terraform destroy."""
        _aws_values(spec)

    def delete_terraform_state(self, spec: ClusterSpec, state_dir: Path) -> None:
        """Delete this cluster's Terraform state object. The state bucket is kept."""
        delete_cluster_terraform_state(spec, state_dir=state_dir)


AWS = AwsProvider()
