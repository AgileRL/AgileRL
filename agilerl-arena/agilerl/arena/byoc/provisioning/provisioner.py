# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle orchestration across cloud providers."""

from __future__ import annotations

from pathlib import Path

import click

from agilerl.arena.byoc.provisioning.providers import get_provider
from agilerl.arena.byoc.provisioning.providers.base import (
    CloudProvider,
    CreatedProjectsQuery,
    DeleteGateway,
    DestroyDeferred,
    DestroyExclusions,
    InstallCni,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.terraform import (
    ClusterOutputs,
    TerraformRunner,
    absolute_path,
    local_terraform_state_has_resources,
    terraform_address_matches,
)

POST_APPLY_COMPLETE = "post-apply-complete"


def post_apply_marker(state_dir: Path) -> Path:
    """Return the file written after cluster setup finishes."""
    return absolute_path(state_dir).parent / POST_APPLY_COMPLETE


def post_apply_is_complete(state_dir: Path) -> bool:
    """Return whether cluster setup finished for this state directory."""
    return post_apply_marker(state_dir).is_file()


def mark_post_apply_complete(state_dir: Path) -> None:
    """Record that cluster setup finished."""
    post_apply_marker(state_dir).write_text("ok\n", encoding="utf-8")


def _resolve_project_deletion(
    provider: CloudProvider,
    spec: ClusterSpec,
    created_projects: tuple[bool, bool],
    delete_storage_project: bool,
    delete_all: bool,
) -> tuple[bool, bool]:
    """Return whether to delete the compute project and the storage project."""
    delete_compute_project, created_storage_project = created_projects
    if delete_all and created_storage_project:
        delete_storage_project = True
    provider.assert_can_delete_storage_project(
        spec,
        delete_storage_project=delete_storage_project,
        created_storage_project=created_storage_project,
    )
    return delete_compute_project, delete_storage_project


def _finish_missing_remote_state(
    provider: CloudProvider,
    spec: ClusterSpec,
    state_dir: Path,
    remote_state_missing: bool,
    has_local_state: bool,
    known_created: bool,
    delete_compute_project: bool,
    delete_storage_project: bool,
    delete_all: bool,
) -> bool:
    """Finish destroy when the state bucket is gone and local state is empty."""
    if not remote_state_missing or has_local_state:
        return False
    if not known_created:
        delete_compute_project, delete_storage_project = _resolve_project_deletion(
            provider,
            spec,
            provider.created_projects(CreatedProjectsQuery(spec, state_missing=True)),
            delete_storage_project,
            delete_all,
        )
    click.echo(
        f"Terraform state bucket {spec.terraform_state.bucket!r} is already gone."
    )
    provider.finish_destroy(
        spec,
        delete_compute_project=delete_compute_project,
        delete_storage_project=delete_storage_project,
        delete_all=delete_all,
        state_dir=state_dir,
    )
    return True


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
        provider.install_cni(InstallCni(spec, work_dir, self.runner))
        self.runner.plan(work_dir)
        self.runner.apply(work_dir)
        values = self.runner.terraform_outputs(work_dir)
        outputs = provider.cluster_outputs(
            values, output_dir=output_dir, run=self.runner.run
        )
        outputs = provider.post_apply(spec, outputs)
        mark_post_apply_complete(state_dir)
        return outputs

    def ensure_post_apply(
        self, spec: ClusterSpec, state_dir: Path, output_dir: Path
    ) -> None:
        """Finish cluster setup when it did not complete after Terraform apply."""
        state_dir = absolute_path(state_dir)
        output_dir = absolute_path(output_dir)
        if post_apply_is_complete(state_dir) or not self.cluster_exists(
            spec, state_dir
        ):
            return
        provider = get_provider(spec.provider)
        work_dir = provider.open_state(spec, runner=self.runner, state_dir=state_dir)
        if work_dir is None:
            return
        click.echo("Finishing cluster setup.")
        values = self.runner.terraform_outputs(work_dir)
        outputs = provider.cluster_outputs(
            values, output_dir=output_dir, run=self.runner.run
        )
        provider.post_apply(spec, outputs)
        mark_post_apply_complete(state_dir)

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
        delete_all: bool = False,
        created_projects: tuple[bool, bool] | None = None,
    ) -> None:
        """Destroy the cluster. Keep experiment object storage unless asked to delete it."""
        provider = get_provider(spec.provider)
        known_created = created_projects is not None
        delete_compute_project = False
        if created_projects is not None:
            delete_compute_project, delete_storage_project = _resolve_project_deletion(
                provider, spec, created_projects, delete_storage_project, delete_all
            )
        local_state = absolute_path(state_dir) / "terraform.tfstate"
        has_local_state = local_state.is_file() and local_terraform_state_has_resources(
            local_state
        )
        remote_state_missing = (
            delete_all or delete_storage_project
        ) and not provider.remote_state_available(spec)
        if _finish_missing_remote_state(
            provider,
            spec,
            state_dir,
            remote_state_missing,
            has_local_state,
            known_created,
            delete_compute_project,
            delete_storage_project,
            delete_all,
        ):
            return
        spec, work_dir = self._prepare(
            spec, state_dir, require_state_bucket=not remote_state_missing
        )
        if remote_state_missing:
            # Terraform state is the local file. The remote bucket is gone.
            self.runner.initialize_without_backend(work_dir)
        else:
            self.runner.initialize(work_dir, spec)
        if not known_created:
            delete_compute_project, delete_storage_project = _resolve_project_deletion(
                provider,
                spec,
                provider.created_projects(
                    CreatedProjectsQuery(spec, self.runner.state_list(work_dir))
                ),
                delete_storage_project,
                delete_all,
            )
        delete_storage = delete_storage or delete_storage_project or delete_all
        provider.delete_gateway(DeleteGateway(spec, work_dir, self.runner))
        self.runner.destroy(
            work_dir,
            exclude=provider.destroy_exclusions(
                DestroyExclusions(spec, delete_storage=delete_storage)
            ),
        )
        provider.destroy_deferred(
            DestroyDeferred(spec, work_dir, self.runner, delete_storage)
        )
        provider.finish_destroy(
            spec,
            delete_compute_project=delete_compute_project,
            delete_storage_project=delete_storage_project,
            delete_all=delete_all,
            state_dir=state_dir,
        )

    def _prepare(
        self,
        spec: ClusterSpec,
        state_dir: Path,
        create_state_bucket: bool = False,
        prompt_create: bool = False,
        require_state_bucket: bool = True,
    ) -> tuple[ClusterSpec, Path]:
        return get_provider(spec.provider).prepare(
            spec,
            runner=self.runner,
            state_dir=state_dir,
            create_state_bucket=create_state_bucket,
            prompt_create=prompt_create,
            require_state_bucket=require_state_bucket,
        )

    def _init_state(self, spec: ClusterSpec, work_dir: Path) -> None:
        self.runner.initialize(work_dir, spec)
        self.runner.adopt_resources(
            work_dir, get_provider(spec.provider).import_ids(spec)
        )
