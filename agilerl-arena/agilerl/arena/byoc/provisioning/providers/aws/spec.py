# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""AWS cluster values and the Arena registration payload built from them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agilerl.arena.byoc.provisioning.kubelet_reserved import (
    KUBELET_RESERVED_CPUS,
    KUBELET_RESERVED_MEMORY_GIB,
)

if TYPE_CHECKING:
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec

AWS_WORKER_NODE_GROUP_LABEL = "eks.amazonaws.com/nodegroup"


class AwsSystemNodeGroupSpec(BaseModel):
    """EKS system node group settings."""

    model_config = ConfigDict(extra="forbid")

    min_node_count: int = Field(default=2, gt=0)
    max_node_count: int = Field(default=10, gt=0)
    instance_type: str = "t3.xlarge"

    @model_validator(mode="after")
    def validate_node_count_range(self) -> AwsSystemNodeGroupSpec:
        """Autoscaling min must not exceed max."""
        if self.min_node_count > self.max_node_count:
            msg = "system min_node_count must be less than or equal to max_node_count."
            raise ValueError(msg)
        return self


class AwsWorkerNodeGroupSpec(BaseModel):
    """EKS worker node group settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "workers"
    min_node_count: int = Field(default=0, ge=0)
    max_node_count: int = Field(default=10, gt=0)
    instance_type: str = Field(min_length=1)
    num_cpus: int | None = Field(default=None, gt=0)
    num_gpus: int = Field(default=0, ge=0)
    memory_gib: int | None = Field(default=None, gt=0)
    gpu_type: str | None = None
    vram_gib: int | None = Field(default=None, gt=0)

    @field_validator("name", "instance_type")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Require a non-empty string."""
        stripped = value.strip()
        if not stripped:
            msg = "must be a non-empty string."
            raise ValueError(msg)
        return stripped

    @field_validator("gpu_type")
    @classmethod
    def strip_gpu_type(cls, value: str | None) -> str | None:
        """Treat a blank GPU type as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_worker(self) -> AwsWorkerNodeGroupSpec:
        """Autoscaling min must not exceed max, and GPU workers name their device."""
        if self.min_node_count > self.max_node_count:
            msg = "worker min_node_count must be less than or equal to max_node_count."
            raise ValueError(msg)
        if self.num_gpus > 0 and (
            self.num_cpus is None
            or self.memory_gib is None
            or self.vram_gib is None
            or not self.gpu_type
        ):
            msg = (
                f"GPU worker {self.name!r} requires num_cpus, memory_gib, "
                "gpu_type, and vram_gib."
            )
            raise ValueError(msg)
        return self


class AwsObjectStorageSpec(BaseModel):
    """EKS experiment bucket capacity."""

    model_config = ConfigDict(extra="forbid")

    size_gib: int = Field(gt=0)


class AwsSharedFilesystemSpec(BaseModel):
    """Whether provisioning creates an EFS file system."""

    model_config = ConfigDict(extra="forbid")

    provision: bool = False


class AwsSpec(BaseModel):
    """AWS cloud infrastructure settings."""

    model_config = ConfigDict(extra="forbid")

    region: str = Field(min_length=1)
    kubernetes_version: str = "1.36"
    system: AwsSystemNodeGroupSpec = Field(default_factory=AwsSystemNodeGroupSpec)
    workers: list[AwsWorkerNodeGroupSpec] = Field(default_factory=list)
    object_storage: AwsObjectStorageSpec
    shared_filesystem: AwsSharedFilesystemSpec = Field(
        default_factory=AwsSharedFilesystemSpec
    )

    @field_validator("region", "kubernetes_version")
    @classmethod
    def strip_required(cls, value: str) -> str:
        """Require a non-empty string."""
        stripped = value.strip()
        if not stripped:
            msg = "must be a non-empty string."
            raise ValueError(msg)
        return stripped

    @model_validator(mode="before")
    @classmethod
    def reject_singular_worker(cls, value: object) -> object:
        """Require the workers list instead of a single worker mapping."""
        if isinstance(value, dict) and "worker" in value:
            msg = "aws.workers must be a list of worker node groups."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_aws_values(self) -> AwsSpec:
        """Reject duplicate worker names."""
        names = [worker.name for worker in self.workers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            listed = ", ".join(repr(name) for name in duplicates)
            msg = f"aws.workers names must be unique: {listed}."
            raise ValueError(msg)
        return self

    def terraform_values(
        self,
        *,
        storage_bucket_name: str,
        storage_endpoint: str,
        enable_gateway_api: bool,
    ) -> dict[str, object]:
        """Return the module variables for this EKS cluster."""
        return {
            "region": self.region,
            "storage_bucket_name": storage_bucket_name,
            "storage_endpoint": storage_endpoint,
            "kubernetes_version": self.kubernetes_version,
            "arena_min_node_count": self.system.min_node_count,
            "arena_max_node_count": self.system.max_node_count,
            "arena_instance_type": self.system.instance_type,
            "provision_shared_filesystem": self.shared_filesystem.provision,
            "enable_gateway_api": enable_gateway_api,
            "workers": [
                {
                    "name": worker.name,
                    "min_node_count": worker.min_node_count,
                    "max_node_count": worker.max_node_count,
                    "instance_type": worker.instance_type,
                    "num_gpus": worker.num_gpus,
                }
                for worker in self.workers
            ],
        }


def registration_payload(spec: ClusterSpec) -> dict[str, object]:
    """Return the CLI register ``byocProvider`` body for an AWS cluster spec."""
    values = spec.aws
    if values is None:
        msg = f"Cluster {spec.name!r} is not an AWS spec."
        raise click.ClickException(msg)
    arena = spec.arena
    state = spec.terraform_state
    terraform_state: dict[str, str] = {"bucket": state.bucket}
    if state.key:
        terraform_state["key"] = state.key
    if state.endpoint:
        terraform_state["endpoint"] = state.endpoint
    config: dict[str, object] = {
        "region": values.region,
        "storage_bucket_name": arena.storage.bucket,
        "storage_endpoint": arena.storage.endpoint,
        "kubernetes_version": values.kubernetes_version,
        "terraform_state": terraform_state,
        "arena": {
            "min_node_count": values.system.min_node_count,
            "max_node_count": values.system.max_node_count,
            "instance_type": values.system.instance_type,
        },
        "workers": [
            {
                "name": worker.name,
                "min_node_count": worker.min_node_count,
                "max_node_count": worker.max_node_count,
                "instance_type": worker.instance_type,
                "num_cpus": worker.num_cpus,
                "num_gpus": worker.num_gpus,
                "memory_gib": worker.memory_gib,
                "gpu_type": worker.gpu_type,
                "vram_gib": worker.vram_gib,
            }
            for worker in values.workers
        ],
        "object_storage_size_gib": values.object_storage.size_gib,
        "shared_filesystem": {"provision": values.shared_filesystem.provision},
        "inference": {
            "domain": arena.inference.domain,
            "hostname_template": arena.inference.hostname_template,
            "tls_secret_name": arena.inference.tls_secret_name,
        },
        "gateway_api": {
            "enable": arena.gateway.enable,
            "gateway_name": arena.gateway.name,
        },
    }
    return {"provider": "aws", "config": config}


def resource_classes(
    spec: ClusterSpec, worker_node_group_ids: dict[str, str]
) -> list[ClusterResourceClass]:
    """Build GPU resource classes for provisioned EKS worker groups."""
    # Cycle: cluster_register imports provisioning while this module is loading.
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass

    values = spec.aws
    if values is None:
        msg = f"Cluster {spec.name!r} is not an AWS spec."
        raise click.ClickException(msg)
    classes: list[ClusterResourceClass] = []
    for worker in values.workers:
        if worker.num_gpus < 1:
            continue
        group_id = worker_node_group_ids.get(worker.name)
        if not group_id:
            msg = f"Terraform output missing node group id for worker {worker.name!r}."
            raise click.ClickException(msg)
        num_cpus = worker.num_cpus
        memory_gib = worker.memory_gib
        vram_gib = worker.vram_gib
        gpu_type = worker.gpu_type
        if (
            num_cpus is None
            or memory_gib is None
            or vram_gib is None
            or gpu_type is None
        ):
            msg = (
                f"GPU worker {worker.name!r} requires num_cpus, memory_gib, "
                "gpu_type, and vram_gib."
            )
            raise click.ClickException(msg)
        schedulable_cpus = num_cpus - KUBELET_RESERVED_CPUS
        schedulable_memory_gib = memory_gib - KUBELET_RESERVED_MEMORY_GIB
        if schedulable_cpus < 1 or schedulable_memory_gib < 1:
            msg = (
                f"Worker {worker.name!r} instance {worker.instance_type!r} is too "
                f"small: after reserving {KUBELET_RESERVED_CPUS} vCPU and "
                f"{KUBELET_RESERVED_MEMORY_GIB} GiB for kubelet reservations, nothing "
                "is left for the workload."
            )
            raise click.ClickException(msg)
        classes.append(
            ClusterResourceClass(
                name=f"{spec.name}-{worker.name}",
                num_nodes=worker.max_node_count,
                node_selector={AWS_WORKER_NODE_GROUP_LABEL: group_id},
                metadata={
                    "computeResource": {
                        "numCpus": schedulable_cpus,
                        "numGpus": worker.num_gpus,
                        "memoryBytes": f"{schedulable_memory_gib} GiB",
                        "gramPerGpu": vram_gib,
                        "gpuMemoryBytes": f"{vram_gib * worker.num_gpus} GiB",
                        "gpu": {
                            "type": gpu_type,
                            "count": worker.num_gpus,
                            "driverVersion": "default",
                        },
                    }
                },
            )
        )
    return classes


def _stored_text(value: object) -> str | None:
    """Return a stripped string, or ``None`` when *value* is blank."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def spec_from_stored_row(row: dict[str, object]) -> ClusterSpec:
    """Build an AWS cluster spec from an Arena on-prem cluster record."""
    # Cycle: provisioning.spec imports AwsSpec from this module.
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec

    name = row.get("name")
    if not isinstance(name, str) or not name.strip():
        msg = "Arena cluster record is missing a name."
        raise ValueError(msg)
    provider = row.get("byoc_provider")
    if not isinstance(provider, dict):
        provider = row.get("byocProvider")
    if not isinstance(provider, dict) or provider.get("provider") != "aws":
        msg = f"Cluster {name!r} has no stored AWS settings."
        raise ValueError(msg)
    config = provider.get("config")
    if not isinstance(config, dict):
        msg = f"Cluster {name!r} has no stored AWS settings."
        raise ValueError(msg)
    inference = config.get("inference")
    state = config.get("terraform_state")
    if not isinstance(inference, dict) or not isinstance(state, dict):
        msg = f"Cluster {name!r} is missing stored AWS inference or state settings."
        raise ValueError(msg)
    shared = config.get("shared_filesystem")
    gateway = config.get("gateway_api")
    provision = False
    if isinstance(shared, dict):
        raw_provision = shared.get("provision", False)
        provision = raw_provision if isinstance(raw_provision, bool) else False
    gateway_values: dict[str, object] = {}
    if isinstance(gateway, dict):
        if isinstance(gateway.get("enable"), bool):
            gateway_values["enable"] = gateway["enable"]
        gateway_name = _stored_text(gateway.get("gateway_name"))
        if gateway_name is not None:
            gateway_values["name"] = gateway_name
    bucket = _stored_text(config.get("storage_bucket_name"))
    endpoint = _stored_text(config.get("storage_endpoint"))
    if bucket is None or endpoint is None:
        msg = f"Cluster {name!r} is missing stored AWS object storage settings."
        raise ValueError(msg)
    system = config.get("arena")
    return ClusterSpec.model_validate(
        {
            "provider": "aws",
            "name": name.strip(),
            "terraform_state": state,
            "arena": {
                "storage": {"bucket": bucket, "endpoint": endpoint},
                "inference": inference,
                "gateway": gateway_values,
            },
            "aws": {
                "region": config.get("region"),
                "kubernetes_version": config.get("kubernetes_version", "1.36"),
                "system": system if isinstance(system, dict) else {},
                "workers": config.get("workers") or [],
                "object_storage": {
                    "size_gib": config.get("object_storage_size_gib", 1024)
                },
                "shared_filesystem": {"provision": provision},
            },
        }
    )
