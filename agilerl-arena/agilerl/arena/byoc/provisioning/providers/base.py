# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Cloud provider hooks used by cluster provisioning and ``arena cluster``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Imported only for annotations. Loading these from the spec module would
    # cycle through this package.
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec
    from agilerl.arena.byoc.provisioning.terraform import (
        ClusterOutputs,
        TerraformRunner,
    )


class CloudProvider(Protocol):
    """One cloud that can provision an Arena Kubernetes cluster."""

    name: str
    display_name: str
    terraform_module: str
    cluster_address: str
    provisioned_addresses: tuple[str, ...]

    def validate(self, spec: ClusterSpec) -> None:
        """Reject a spec this provider cannot apply."""

    def needs_stored_config(self, spec: ClusterSpec) -> bool:
        """Return whether Arena may still hold ids the spec omitted."""

    def recover(self, spec: ClusterSpec, stored: dict[str, object]) -> ClusterSpec:
        """Fill omitted ids from the config Arena stored for this cluster."""

    def registration_payload(
        self,
        spec: ClusterSpec,
        project_id: str | None = None,
        storage_project_id: str | None = None,
    ) -> dict[str, object]:
        """Return the ``byocProvider`` body sent when registering the cluster."""

    def resource_classes(
        self, spec: ClusterSpec, worker_node_group_ids: dict[str, str]
    ) -> list[ClusterResourceClass]:
        """Build GPU resource classes for provisioned worker groups."""

    def spec_from_stored_row(self, row: dict[str, object]) -> ClusterSpec:
        """Rebuild a cluster spec from an Arena on-prem cluster record."""

    def prepare(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
        create_state_bucket: bool = False,
        prompt_create: bool = False,
    ) -> tuple[ClusterSpec, Path]:
        """Validate the spec, write provider auth, and open a Terraform work dir."""

    def open_state(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
    ) -> Path | None:
        """Initialize remote Terraform state, or ``None`` when nothing was provisioned."""

    def write_backend(self, work_dir: Path, spec: ClusterSpec) -> None:
        """Write the Terraform state backend file."""

    def install_cni(
        self, spec: ClusterSpec, work_dir: Path, runner: TerraformRunner
    ) -> None:
        """Install the CNI before Terraform creates nodes."""

    def ensure_gateway(
        self,
        spec: ClusterSpec,
        kubeconfig_path: Path,
        terraform_outputs: dict[str, object],
    ) -> list[dict[str, str]] | None:
        """Create the cluster Gateway when the spec enables it."""

    def cluster_outputs(
        self,
        values: dict[str, object],
        output_dir: Path,
        run: object,
    ) -> ClusterOutputs:
        """Turn Terraform outputs into kubeconfig and storage credentials."""

    def post_apply(self, spec: ClusterSpec, outputs: ClusterOutputs) -> ClusterOutputs:
        """Configure the cluster after Terraform apply."""

    def import_ids(self, spec: ClusterSpec) -> dict[str, str]:
        """Return existing resources Terraform should adopt."""

    def assert_can_delete_storage_project(
        self, spec: ClusterSpec, delete_storage_project: bool
    ) -> None:
        """Reject deletion of a storage project the spec did not create."""

    def created_projects(self, spec: ClusterSpec) -> tuple[bool, bool]:
        """Return whether compute and storage projects were omitted from the spec."""

    def delete_gateway(
        self,
        spec: ClusterSpec,
        work_dir: Path,
        runner: TerraformRunner,
    ) -> None:
        """Delete the cluster Gateway before Terraform destroy."""

    def destroy_exclusions(
        self, spec: ClusterSpec, delete_storage: bool
    ) -> tuple[str, ...]:
        """Return Terraform addresses ``destroy`` must leave in place."""

    def finish_destroy(
        self,
        spec: ClusterSpec,
        delete_compute_project: bool,
        delete_storage_project: bool,
    ) -> None:
        """Delete projects and the state bucket after Terraform destroy."""

    def delete_terraform_state(self, spec: ClusterSpec, state_dir: Path) -> None:
        """Delete this cluster's Terraform state object."""
