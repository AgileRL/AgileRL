# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle orchestration across cloud providers."""

from __future__ import annotations

from pathlib import Path

import click

from agilerl.arena.byoc.provisioning.providers import get_provider
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
    terraform_address_matches,
)


class ClusterProvisioner:
    """Provision cloud Kubernetes clusters with Terraform."""

    def __init__(self, runner: TerraformRunner | None = None) -> None:
        self.runner = runner or TerraformRunner()

    def cluster_exists(self, spec: ClusterSpec, state_dir: Path) -> bool:
        """Return whether Terraform state records a finished cluster."""
        provider = get_provider(spec.provider)
        work_dir = provider.open_state(spec, runner=self.runner, state_dir=state_dir)
        if work_dir is None:
            return False
        addresses = self.runner.state_list(work_dir)
        return all(
            any(
                terraform_address_matches(address, (required,)) for address in addresses
            )
            for required in provider.provisioned_addresses
        )

    def write_kubeconfig(
        self, spec: ClusterSpec, state_dir: Path, output_dir: Path
    ) -> Path:
        """Write the kubeconfig for a cluster already recorded in Terraform state."""
        provider = get_provider(spec.provider)
        work_dir = provider.open_state(spec, runner=self.runner, state_dir=state_dir)
        if work_dir is None:
            msg = f"Terraform state for {spec.name!r} is not available."
            raise click.ClickException(msg)
        values = self.runner.terraform_outputs(work_dir)
        return provider.cluster_outputs(
            values, output_dir=output_dir, run=self.runner.run
        ).kubeconfig_path

    def plan(
        self, spec: ClusterSpec, state_dir: Path, create_state_bucket: bool = False
    ) -> None:
        """Initialize and plan a cluster without changing infrastructure."""
        spec, work_dir = self._prepare(
            spec, state_dir, create_state_bucket=create_state_bucket, prompt_create=True
        )
        self._init_state(spec, work_dir)
        self.runner.plan(work_dir)

    def provision(
        self,
        spec: ClusterSpec,
        state_dir: Path,
        output_dir: Path,
        create_state_bucket: bool = False,
    ) -> ClusterOutputs:
        """Apply a cluster plan and retrieve its kubeconfig."""
        # Resolve while getcwd() still works. Terraform can outlive the process directory.
        output_dir = absolute_path(output_dir)
        state_dir = absolute_path(state_dir)
        provider = get_provider(spec.provider)
        spec, work_dir = self._prepare(
            spec, state_dir, create_state_bucket=create_state_bucket, prompt_create=True
        )
        self._init_state(spec, work_dir)
        provider.install_cni(spec, work_dir, self.runner)
        self.runner.plan(work_dir)
        self.runner.apply(work_dir)
        values = self.runner.terraform_outputs(work_dir)
        outputs = provider.cluster_outputs(
            values, output_dir=output_dir, run=self.runner.run
        )
        return provider.post_apply(spec, outputs)

    def status(
        self,
        spec: ClusterSpec,
        state_dir: Path,
    ) -> dict[str, object]:
        """Return raw Terraform outputs for the configured cluster."""
        spec, work_dir = self._prepare(spec, state_dir)
        self.runner.initialize(work_dir, spec)
        return self.runner.terraform_outputs(work_dir)

    def destroy(
        self,
        spec: ClusterSpec,
        state_dir: Path,
        delete_storage: bool = False,
        delete_storage_project: bool = False,
    ) -> None:
        """Destroy the cluster. Keep experiment object storage unless ``delete_storage`` is set."""
        provider = get_provider(spec.provider)
        delete_compute_project, _ = provider.created_projects(spec)
        provider.assert_can_delete_storage_project(
            spec, delete_storage_project=delete_storage_project
        )
        delete_storage = delete_storage or delete_storage_project
        spec, work_dir = self._prepare(spec, state_dir)
        self.runner.initialize(work_dir, spec)
        provider.delete_gateway(spec, work_dir, self.runner)
        self.runner.destroy(
            work_dir,
            exclude=provider.destroy_exclusions(spec, delete_storage=delete_storage),
        )
        provider.finish_destroy(
            spec,
            delete_compute_project=delete_compute_project,
            delete_storage_project=delete_storage_project,
        )

    def _prepare(
        self,
        spec: ClusterSpec,
        state_dir: Path,
        create_state_bucket: bool = False,
        prompt_create: bool = False,
    ) -> tuple[ClusterSpec, Path]:
        return get_provider(spec.provider).prepare(
            spec,
            runner=self.runner,
            state_dir=state_dir,
            create_state_bucket=create_state_bucket,
            prompt_create=prompt_create,
        )

    def _init_state(self, spec: ClusterSpec, work_dir: Path) -> None:
        self.runner.initialize(work_dir, spec)
        self.runner.adopt_resources(
            work_dir, get_provider(spec.provider).import_ids(spec)
        )
