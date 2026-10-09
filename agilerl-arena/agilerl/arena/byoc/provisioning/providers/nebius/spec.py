# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nebius cluster values and the Arena registration payload built from them."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import click
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agilerl.arena.byoc.provisioning.kubelet_reserved import (
    KUBELET_RESERVED_CPUS,
    KUBELET_RESERVED_MEMORY_GIB,
)

if TYPE_CHECKING:
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec

NEBIUS_WORKER_NODE_GROUP_LABEL = "nebius.com/node-group-id"
NEBIUS_GPU_WORKER_PRESET = re.compile(
    r"^(?P<gpus>\d+)gpu-(?P<cpus>\d+)vcpu-(?P<memory_gb>\d+)gb$"
)
NEBIUS_GPU_VRAM_GIB = {
    "gpu-b300-sxm": 288,
    "gpu-b200-sxm": 180,
    "gpu-b200-sxm-a": 180,
    "gpu-h200-sxm": 141,
    "gpu-h100-sxm": 80,
    "gpu-rtx6000": 96,
    "gpu-l40s-a": 48,
    "gpu-l40s-d": 48,
}


def parse_nebius_gpu_preset(preset: str) -> tuple[int, int, int]:
    """Return ``(num_cpus, num_gpus, memory_gib)`` for a Nebius GPU worker preset."""
    match = NEBIUS_GPU_WORKER_PRESET.fullmatch(preset.strip().lower())
    if match is None:
        msg = f"GPU worker preset must look like '1gpu-16vcpu-200gb' (got {preset!r})."
        raise ValueError(msg)
    return (
        int(match["cpus"]),
        int(match["gpus"]),
        int(match["memory_gb"]),
    )


def nebius_gpu_vram_gib(platform: str) -> int:
    """Return VRAM per GPU in GiB for a Nebius GPU platform."""
    vram = NEBIUS_GPU_VRAM_GIB.get(platform.strip().lower())
    if vram is None:
        supported = ", ".join(sorted(NEBIUS_GPU_VRAM_GIB))
        msg = (
            f"Unknown Nebius GPU platform {platform!r}; VRAM per GPU is required for "
            f"the resource class. Supported platforms: {supported}."
        )
        raise ValueError(msg)
    return vram


class SystemNodeGroupSpec(BaseModel):
    """Nebius Kubernetes system node group settings."""

    model_config = ConfigDict(extra="forbid")

    min_node_count: int = Field(default=2, gt=0)
    max_node_count: int = Field(default=10, gt=0)
    platform: str = "cpu-e2"
    preset: str = "2vcpu-8gb"

    @model_validator(mode="after")
    def validate_node_count_range(self) -> SystemNodeGroupSpec:
        """Autoscaling min must not exceed max."""
        if self.min_node_count > self.max_node_count:
            msg = "system min_node_count must be less than or equal to max_node_count."
            raise ValueError(msg)
        return self


class WorkerNodeGroupSpec(BaseModel):
    """Nebius Kubernetes GPU worker node group settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "workers"
    min_node_count: int = Field(default=0, ge=0)
    max_node_count: int = Field(default=10, gt=0)
    platform: str = "gpu-h100-sxm"
    preset: str = "1gpu-16vcpu-200gb"
    gpu_drivers_preset: str = "cuda13.0"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require a non-empty node group name."""
        stripped = value.strip()
        if not stripped:
            msg = "worker node group name must be a non-empty string."
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def validate_node_count_range(self) -> WorkerNodeGroupSpec:
        """Autoscaling min must not exceed max."""
        if self.min_node_count > self.max_node_count:
            msg = "worker min_node_count must be less than or equal to max_node_count."
            raise ValueError(msg)
        return self


class ObjectStorageSpec(BaseModel):
    """Nebius object storage settings."""

    model_config = ConfigDict(extra="forbid")

    size_gib: int = Field(gt=0)


class SharedFilesystemSpec(BaseModel):
    """Nebius shared filesystem settings."""

    model_config = ConfigDict(extra="forbid")

    provision: bool = True
    type: str = "NETWORK_SSD"
    size_gib: int = Field(default=256, gt=0)
    mount_tag: str = "csi-storage"


class NebiusSpec(BaseModel):
    """Nebius cloud infrastructure settings."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(min_length=1)
    region: str | None = None
    project_id: str | None = None
    storage_project_id: str | None = None
    subnet_id: str | None = None
    service_account_id: str | None = None
    kubernetes_version: str = "1.35"
    etcd_cluster_size: int = Field(default=3, gt=0)
    system: SystemNodeGroupSpec = Field(default_factory=SystemNodeGroupSpec)
    workers: list[WorkerNodeGroupSpec] = Field(
        default_factory=lambda: [WorkerNodeGroupSpec()],
        min_length=1,
    )
    object_storage: ObjectStorageSpec
    shared_filesystem: SharedFilesystemSpec = Field(
        default_factory=SharedFilesystemSpec
    )

    @field_validator("service_account_id")
    @classmethod
    def validate_service_account_id(cls, value: str | None) -> str | None:
        """Treat a blank service account ID as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("region", "project_id", "storage_project_id", "subnet_id")
    @classmethod
    def strip_optional_id(cls, value: str | None) -> str | None:
        """Treat blank resource IDs and regions as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="before")
    @classmethod
    def reject_singular_worker(cls, value: object) -> object:
        """Require the workers list instead of a single worker mapping."""
        if isinstance(value, dict) and "worker" in value:
            msg = "nebius.workers must be a list of worker node groups."
            raise ValueError(msg)
        if isinstance(value, dict) and "head" in value:
            msg = "nebius.system must be used instead of head."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_unique_worker_names(self) -> NebiusSpec:
        """Validate resource selection and unique worker names."""
        if self.region is None and (
            self.project_id is None or self.storage_project_id is None
        ):
            msg = (
                "nebius.region is required when nebius.project_id or "
                "nebius.storage_project_id is omitted."
            )
            raise ValueError(msg)
        if self.subnet_id is not None and self.project_id is None:
            msg = "nebius.subnet_id requires nebius.project_id."
            raise ValueError(msg)
        names = [worker.name for worker in self.workers]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            listed = ", ".join(repr(name) for name in duplicates)
            msg = f"nebius.workers names must be unique: {listed}."
            raise ValueError(msg)
        return self

    def terraform_values(
        self, *, storage_bucket_name: str, storage_endpoint: str
    ) -> dict[str, object]:
        """Return the module variables for this Nebius cluster."""
        return {
            "tenant_id": self.tenant_id,
            "region": self.region or "",
            "project_id": self.project_id or "",
            "storage_project_id": self.storage_project_id or "",
            "subnet_id": self.subnet_id or "",
            "service_account_id": self.service_account_id or "",
            "storage_bucket_name": storage_bucket_name,
            "storage_endpoint": storage_endpoint,
            "kubernetes_version": self.kubernetes_version,
            "etcd_cluster_size": self.etcd_cluster_size,
            "arena_min_node_count": self.system.min_node_count,
            "arena_max_node_count": self.system.max_node_count,
            "arena_platform": self.system.platform,
            "arena_preset": self.system.preset,
            "workers": [
                {
                    "name": worker.name,
                    "min_node_count": worker.min_node_count,
                    "max_node_count": worker.max_node_count,
                    "platform": worker.platform,
                    "preset": worker.preset,
                    "gpu_drivers_preset": worker.gpu_drivers_preset,
                }
                for worker in self.workers
            ],
            "object_storage_size_gib": self.object_storage.size_gib,
            "provision_shared_filesystem": self.shared_filesystem.provision,
            "filesystem_type": self.shared_filesystem.type,
            "filesystem_size_gib": self.shared_filesystem.size_gib,
            "filesystem_mount_tag": self.shared_filesystem.mount_tag,
        }


def _stored_text(value: object) -> str | None:
    """Return a stripped string, or ``None`` when *value* is blank."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def registration_payload(
    spec: ClusterSpec,
    project_id: str | None = None,
    storage_project_id: str | None = None,
) -> dict[str, object]:
    """Return the CLI register ``byocProvider`` body for a Nebius cluster spec."""
    config: dict[str, object] = {}
    values = spec.nebius_cloud()
    resolved_project_id = values.project_id or project_id
    resolved_storage_project_id = values.storage_project_id or storage_project_id
    for key, raw in (
        ("tenant_id", values.tenant_id),
        ("region", values.region),
        ("project_id", resolved_project_id),
        ("storage_project_id", resolved_storage_project_id),
        ("subnet_id", values.subnet_id),
    ):
        if isinstance(raw, str) and raw.strip():
            config[key] = raw.strip()
    state = spec.terraform_state
    terraform_state: dict[str, str] = {"bucket": state.bucket}
    if state.key:
        terraform_state["key"] = state.key
    if state.endpoint:
        terraform_state["endpoint"] = state.endpoint
    config["terraform_state"] = terraform_state
    config["object_storage_size_gib"] = values.object_storage.size_gib
    config["workers"] = [
        {
            "name": worker.name,
            "min_node_count": worker.min_node_count,
            "max_node_count": worker.max_node_count,
            "platform": worker.platform,
            "preset": worker.preset,
            "gpu_drivers_preset": worker.gpu_drivers_preset,
        }
        for worker in values.workers
    ]
    return {"provider": "nebius", "config": config}


def needs_stored_config(spec: ClusterSpec) -> bool:
    """Return whether omitted Nebius project ids should be loaded from Arena."""
    cloud = spec.nebius_cloud()
    return cloud.project_id is None or cloud.storage_project_id is None


def recover_stored_config(spec: ClusterSpec, config: dict[str, object]) -> ClusterSpec:
    """Fill Nebius ids and state location the spec left unset."""
    cloud = spec.nebius_cloud()
    values_update: dict[str, str] = {}
    if cloud.project_id is None:
        project_id = _stored_text(config.get("project_id"))
        if project_id is not None:
            values_update["project_id"] = project_id
    if cloud.storage_project_id is None:
        storage_project_id = _stored_text(config.get("storage_project_id"))
        if storage_project_id is not None:
            values_update["storage_project_id"] = storage_project_id
    project_id = values_update.get("project_id", cloud.project_id)
    if cloud.subnet_id is None and project_id is not None:
        subnet_id = _stored_text(config.get("subnet_id"))
        if subnet_id is not None:
            values_update["subnet_id"] = subnet_id
    stored_state = config.get("terraform_state")
    state_update: dict[str, str] = {}
    if isinstance(stored_state, dict):
        if spec.terraform_state.key is None:
            key = _stored_text(stored_state.get("key"))
            if key is not None:
                state_update["key"] = key
        if spec.terraform_state.endpoint is None:
            endpoint = _stored_text(stored_state.get("endpoint"))
            if endpoint is not None:
                state_update["endpoint"] = endpoint
    if not values_update and not state_update:
        return spec
    values = cloud.model_copy(update=values_update) if values_update else cloud
    state = (
        spec.terraform_state.model_copy(update=state_update)
        if state_update
        else spec.terraform_state
    )
    return spec.model_copy(update={"nebius": values, "terraform_state": state})


def resource_classes(
    spec: ClusterSpec, worker_node_group_ids: dict[str, str]
) -> list[ClusterResourceClass]:
    """Build GPU resource classes for provisioned Nebius worker groups."""
    # Cycle: cluster_register imports provisioning while this module is loading.
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass

    classes: list[ClusterResourceClass] = []
    for worker in spec.nebius_cloud().workers:
        group_id = worker_node_group_ids.get(worker.name)
        if not group_id:
            msg = f"Terraform output missing node group id for worker {worker.name!r}."
            raise click.ClickException(msg)
        try:
            num_cpus, num_gpus, memory_gib = parse_nebius_gpu_preset(worker.preset)
            vram_gib = nebius_gpu_vram_gib(worker.platform)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        schedulable_cpus = num_cpus - KUBELET_RESERVED_CPUS
        schedulable_memory_gib = memory_gib - KUBELET_RESERVED_MEMORY_GIB
        if schedulable_cpus < 1 or schedulable_memory_gib < 1:
            msg = (
                f"Worker {worker.name!r} preset {worker.preset!r} is too small: "
                f"after reserving {KUBELET_RESERVED_CPUS} vCPU and "
                f"{KUBELET_RESERVED_MEMORY_GIB} GiB for kubelet reservations, "
                "nothing is left for the workload."
            )
            raise click.ClickException(msg)
        classes.append(
            ClusterResourceClass(
                name=f"{spec.name}-{worker.name}",
                num_nodes=worker.max_node_count,
                node_selector={NEBIUS_WORKER_NODE_GROUP_LABEL: group_id},
                metadata={
                    "computeResource": {
                        "numCpus": schedulable_cpus,
                        "numGpus": num_gpus,
                        "memoryBytes": f"{schedulable_memory_gib} GiB",
                        "gramPerGpu": vram_gib,
                        "gpuMemoryBytes": f"{vram_gib * num_gpus} GiB",
                        "gpu": {
                            "type": worker.platform,
                            "count": num_gpus,
                            "driverVersion": "default",
                        },
                    }
                },
            )
        )
    return classes


def spec_from_stored_row(row: dict[str, object]) -> ClusterSpec:
    """Build a Nebius cluster spec from an Arena on-prem cluster record."""
    # Cycle: provisioning.spec imports NebiusSpec from this module.
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec

    name = _stored_text(row.get("name"))
    if name is None:
        msg = "Arena cluster record is missing a name."
        raise ValueError(msg)
    provider = row.get("byoc_provider")
    if not isinstance(provider, dict):
        provider = row.get("byocProvider")
    if not isinstance(provider, dict) or provider.get("provider") != "nebius":
        msg = f"Cluster {name!r} has no stored Nebius settings."
        raise ValueError(msg)
    config = provider.get("config")
    if not isinstance(config, dict):
        msg = f"Cluster {name!r} has no stored Nebius settings."
        raise ValueError(msg)
    stored_state = config.get("terraform_state")
    bucket = (
        _stored_text(stored_state.get("bucket"))
        if isinstance(stored_state, dict)
        else None
    )
    if bucket is None:
        msg = f"Cluster {name!r} has no Terraform state bucket stored in Arena."
        raise ValueError(msg)
    tenant_id = _stored_text(config.get("tenant_id"))
    storage_bucket = _stored_text(row.get("storage_bucket"))
    storage_endpoint = _stored_text(row.get("storage_endpoint"))
    domain = _stored_text(row.get("domain"))
    if tenant_id is None or storage_bucket is None or storage_endpoint is None:
        msg = f"Cluster {name!r} is missing Nebius tenant or object storage settings."
        raise ValueError(msg)
    if domain is None:
        msg = f"Cluster {name!r} is missing an inference domain."
        raise ValueError(msg)
    size = config.get("object_storage_size_gib")
    if isinstance(size, bool) or not isinstance(size, int | float) or int(size) != size:
        msg = f"Cluster {name!r} is missing an object storage size."
        raise ValueError(msg)
    if int(size) < 1:
        msg = f"Cluster {name!r} is missing an object storage size."
        raise ValueError(msg)
    state: dict[str, str] = {"bucket": bucket}
    if isinstance(stored_state, dict):
        key = _stored_text(stored_state.get("key"))
        endpoint = _stored_text(stored_state.get("endpoint"))
        if key is not None:
            state["key"] = key
        if endpoint is not None:
            state["endpoint"] = endpoint
    inference: dict[str, str] = {"domain": domain}
    hostname = _stored_text(row.get("hostname_template"))
    tls_secret_name = _stored_text(row.get("tls_secret_name"))
    if hostname is not None:
        inference["hostname_template"] = hostname
    if tls_secret_name is not None:
        inference["tls_secret_name"] = tls_secret_name
    nebius: dict[str, object] = {
        "tenant_id": tenant_id,
        "object_storage": {"size_gib": int(size)},
    }
    for field in ("region", "project_id", "storage_project_id", "subnet_id"):
        filled = _stored_text(config.get(field))
        if filled is not None:
            nebius[field] = filled
    raw_workers = config.get("workers")
    if raw_workers is not None:
        if not isinstance(raw_workers, list):
            msg = f"Cluster {name!r} has invalid stored worker node groups."
            raise ValueError(msg)
        worker_fields = (
            "name",
            "min_node_count",
            "max_node_count",
            "platform",
            "preset",
            "gpu_drivers_preset",
        )
        workers: list[dict[str, object]] = []
        for item in raw_workers:
            if not isinstance(item, dict):
                msg = f"Cluster {name!r} has invalid stored worker node groups."
                raise ValueError(msg)
            workers.append({key: item[key] for key in worker_fields if key in item})
        if workers:
            nebius["workers"] = workers
    try:
        return ClusterSpec.model_validate(
            {
                "provider": "nebius",
                "name": name,
                "terraform_state": state,
                "arena": {
                    "storage": {
                        "bucket": storage_bucket,
                        "endpoint": storage_endpoint,
                    },
                    "inference": inference,
                },
                "nebius": nebius,
            }
        )
    except ValidationError as exc:
        msg = f"Cluster {name!r} stored Nebius settings are invalid: {exc}"
        raise ValueError(msg) from exc
