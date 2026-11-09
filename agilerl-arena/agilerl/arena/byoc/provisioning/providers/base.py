# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Cloud provider hooks used by cluster provisioning and ``arena cluster``."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CreatedProjectsQuery:
    """Inputs for deciding whether this cluster created its cloud projects."""

    spec: ClusterSpec
    # Resource addresses from Terraform state.
    addresses: tuple[str, ...] = ()
    # The state bucket is gone, so project names stand in for those addresses.
    state_missing: bool = False


@dataclass(frozen=True)
class InstallCni:
    """Terraform work dir for installing the CNI before nodes are created."""

    spec: ClusterSpec
    work_dir: Path
    runner: TerraformRunner


@dataclass(frozen=True)
class EnsureGateway:
    """Gateway settings and the kubeconfig that receives them."""

    spec: ClusterSpec
    kubeconfig_path: Path
    terraform_outputs: dict[str, object]


@dataclass(frozen=True)
class DeleteGateway:
    """Terraform work dir for deleting the gateway before destroy."""

    spec: ClusterSpec
    work_dir: Path
    runner: TerraformRunner


@dataclass(frozen=True)
class DestroyExclusions:
    """Which Terraform addresses destroy must leave in place."""

    spec: ClusterSpec
    delete_storage: bool


@dataclass(frozen=True)
class DestroyDeferred:
    """Resources the first Terraform destroy left for a second pass."""

    spec: ClusterSpec
    work_dir: Path
    runner: TerraformRunner
    delete_storage: bool


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
        """Rebuild a cluster spec from an Arena byoc cluster record."""

    def prepare(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
        create_state_bucket: bool = False,
        prompt_create: bool = False,
        require_state_bucket: bool = True,
    ) -> tuple[ClusterSpec, Path]:
        """Validate the spec, write provider auth, and open a Terraform work dir.

        :param require_state_bucket: When false, a missing state bucket is left alone.
        """

    def open_state(
        self,
        spec: ClusterSpec,
        runner: TerraformRunner,
        state_dir: Path,
    ) -> Path | None:
        """Initialize remote Terraform state, or ``None`` when nothing was provisioned."""

    def write_backend(self, work_dir: Path, spec: ClusterSpec) -> None:
        """Write the Terraform state backend file."""

    def install_cni(self, query: InstallCni) -> None:
        """Install the CNI before Terraform creates nodes."""

    def ensure_gateway(self, query: EnsureGateway) -> list[dict[str, str]] | None:
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
        self,
        spec: ClusterSpec,
        delete_storage_project: bool,
        created_storage_project: bool = False,
    ) -> None:
        """Reject deletion of a storage project the spec did not create."""

    def created_projects(self, query: CreatedProjectsQuery) -> tuple[bool, bool]:
        """Return whether compute and storage projects were created by this cluster."""

    def delete_gateway(self, query: DeleteGateway) -> None:
        """Delete the cluster Gateway before Terraform destroy."""

    def destroy_exclusions(self, query: DestroyExclusions) -> tuple[str, ...]:
        """Return Terraform addresses ``destroy`` must leave in place."""

    def destroy_deferred(self, query: DestroyDeferred) -> None:
        """Destroy resources left out of the first Terraform destroy."""

    def remote_state_available(self, spec: ClusterSpec) -> bool:
        """Return whether destroy can open this cluster's remote Terraform state."""

    def finish_destroy(
        self,
        spec: ClusterSpec,
        delete_compute_project: bool,
        delete_storage_project: bool,
        delete_all: bool,
        state_dir: Path,
    ) -> None:
        """Delete projects, cluster IAM, and the state bucket after Terraform destroy."""

    def delete_terraform_state(self, spec: ClusterSpec, state_dir: Path) -> None:
        """Delete this cluster's Terraform state object."""
