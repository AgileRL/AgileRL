# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Nebius cluster values and the Arena registration payload built from them."""

from __future__ import annotations

import json
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

if TYPE_CHECKING:
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec

NEBIUS_WORKER_NODE_GROUP_LABEL = "nebius.com/node-group-id"
NEBIUS_PROFILE_NAME = "nebius-default"
# The agent chart mounts credentials.json here and selects NEBIUS_PROFILE_NAME.
NEBIUS_CREDENTIALS_PATH = "/etc/arena-byoc-nebius/credentials.json"
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
    "gpu-rtx6000-a": 96,
    "gpu-l40s-a": 48,
    "gpu-l40s-d": 48,
}
NEBIUS_GPU_NAMES = {
    "gpu-b300-sxm": "NVIDIA® B300",
    "gpu-b200-sxm": "NVIDIA® B200",
    "gpu-b200-sxm-a": "NVIDIA® B200",
    "gpu-h200-sxm": "NVIDIA® H200",
    "gpu-h100-sxm": "NVIDIA® H100",
    "gpu-rtx6000": "NVIDIA® RTX PRO™ 6000",
    "gpu-rtx6000-a": "NVIDIA® RTX PRO™ 6000",
    "gpu-l40s-a": "NVIDIA® L40S Intel",
    "gpu-l40s-d": "NVIDIA® L40S AMD",
}


def parse_nebius_gpu_preset(preset: str) -> tuple[int, int, int]:
    """Return ``(num_cpus, num_gpus, memory_gb)`` for a Nebius GPU worker preset."""
    match = NEBIUS_GPU_WORKER_PRESET.fullmatch(preset.strip().lower())
    if match is None:
        msg = f"GPU worker preset must look like '1gpu-16vcpu-200gb' (got {preset!r})."
        raise ValueError(msg)
    return (
        int(match["cpus"]),
        int(match["gpus"]),
        int(match["memory_gb"]),
    )


def nebius_gpu_description(
    platform: str, num_gpus: int, num_cpus: int, memory_gb: int
) -> str:
    """Return the Arena resource class description for a Nebius GPU preset."""
    name = NEBIUS_GPU_NAMES.get(platform.strip().lower(), platform)
    gpus = "GPU" if num_gpus == 1 else "GPUs"
    cpus = "CPU" if num_cpus == 1 else "CPUs"
    return f"{name} {num_gpus} {gpus} - {num_cpus} {cpus} - {memory_gb} GB RAM."


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
    platform: str = "gpu-l40s-a"
    preset: str = "1gpu-16vcpu-64gb"
    gpu_drivers_preset: str = "cuda13.0"
    register_node_group: bool = False

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
        self, storage_bucket_name: str, storage_endpoint: str
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


def _output_text(value: object) -> str | None:
    """Return a Terraform output string, or ``None`` when it is blank."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def service_account_credentials(
    outputs: dict[str, object],
) -> tuple[str, str] | None:
    """Return the created Arena service-account key and profile, or ``None``."""
    service_account_id = _output_text(outputs.get("service_account_id"))
    public_key_id = _output_text(outputs.get("service_account_public_key_id"))
    private_key = _output_text(outputs.get("service_account_private_key"))
    parent_id = _output_text(outputs.get("project_id"))
    if (
        service_account_id is None
        or public_key_id is None
        or private_key is None
        or parent_id is None
    ):
        return None
    key = json.dumps(
        {
            "subject-credentials": {
                "type": "JWT",
                "alg": "RS256",
                "private-key": private_key.strip(),
                "kid": public_key_id.strip(),
                "iss": service_account_id.strip(),
                "sub": service_account_id.strip(),
            }
        },
        indent=2,
    )
    profile = "\n".join(
        (
            f"default: {NEBIUS_PROFILE_NAME}",
            "profiles:",
            f"  {NEBIUS_PROFILE_NAME}:",
            "    endpoint: api.nebius.cloud",
            f"    parent-id: {parent_id.strip()}",
            f"    service-account-credentials-file-path: {NEBIUS_CREDENTIALS_PATH}",
            "",
        )
    )
    return key, profile


def attach_service_account_credentials(
    payload: dict[str, object], outputs: dict[str, object]
) -> None:
    """Add ``service_account_key`` and ``profile`` when Terraform created them."""
    config = payload.get("config")
    if not isinstance(config, dict):
        return
    documents = service_account_credentials(outputs)
    if documents is None:
        return
    config["service_account_key"], config["profile"] = documents


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
    spec: ClusterSpec,
    worker_node_group_ids: dict[str, str],
    vram_gib_by_platform: dict[str, int] | None = None,
    allow_missing_node_group: bool = False,
) -> list[ClusterResourceClass]:
    """Build GPU resource classes from each worker preset's machine size.

    :param vram_gib_by_platform: VRAM per GPU from a preset listing, by platform.
    :param allow_missing_node_group: When the worker's ``register_node_group`` is
        false and Terraform has no id, select ``nebius.com/platform`` and
        ``nebius.com/preset`` instead of the node group id.
    """
    # Cycle: cluster_register imports provisioning while this module is loading.
    from agilerl.arena.byoc.cluster_register import ClusterResourceClass

    vram_overrides = vram_gib_by_platform or {}
    classes: list[ClusterResourceClass] = []
    for worker in spec.nebius_cloud().workers:
        group_id = worker_node_group_ids.get(worker.name)
        if not group_id and (
            worker.register_node_group or not allow_missing_node_group
        ):
            msg = f"Terraform output missing node group id for worker {worker.name!r}."
            raise click.ClickException(msg)
        if group_id:
            node_selector = {NEBIUS_WORKER_NODE_GROUP_LABEL: group_id}
        else:
            node_selector = {
                "nebius.com/platform": worker.platform,
                "nebius.com/preset": worker.preset,
            }
        try:
            num_cpus, num_gpus, memory_gb = parse_nebius_gpu_preset(worker.preset)
            vram_gib = vram_overrides.get(worker.platform) or nebius_gpu_vram_gib(
                worker.platform
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        classes.append(
            ClusterResourceClass(
                name=f"{spec.name}-{worker.name}",
                num_nodes=worker.max_node_count,
                node_selector=node_selector,
                description=nebius_gpu_description(
                    worker.platform, num_gpus, num_cpus, memory_gb
                ),
                metadata={
                    "computeResource": {
                        "numCpus": num_cpus,
                        "numGpus": num_gpus,
                        "memoryBytes": f"{memory_gb} GB",
                        "gramPerGpu": vram_gib,
                        "machineType": worker.preset,
                        "spot": False,
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


def _nebius_stored_config(row: dict[str, object], name: str) -> dict[str, object]:
    """Return the stored Nebius provider config, or raise if it is missing."""
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
    return config


def _nebius_state_from_config(config: dict[str, object], name: str) -> dict[str, str]:
    """Return Terraform state fields stored on a Nebius cluster record."""
    stored_state = config.get("terraform_state")
    bucket = (
        _stored_text(stored_state.get("bucket"))
        if isinstance(stored_state, dict)
        else None
    )
    if bucket is None:
        msg = f"Cluster {name!r} has no Terraform state bucket stored in Arena."
        raise ValueError(msg)
    state: dict[str, str] = {"bucket": bucket}
    if isinstance(stored_state, dict):
        key = _stored_text(stored_state.get("key"))
        endpoint = _stored_text(stored_state.get("endpoint"))
        if key is not None:
            state["key"] = key
        if endpoint is not None:
            state["endpoint"] = endpoint
    return state


def _nebius_inference_from_row(row: dict[str, object], domain: str) -> dict[str, str]:
    """Return inference fields stored on a Nebius cluster record."""
    inference: dict[str, str] = {"domain": domain}
    hostname = _stored_text(row.get("hostname_template"))
    tls_secret_name = _stored_text(row.get("tls_secret_name"))
    if hostname is not None:
        inference["hostname_template"] = hostname
    if tls_secret_name is not None:
        inference["tls_secret_name"] = tls_secret_name
    return inference


def _nebius_workers_from_config(
    config: dict[str, object], name: str
) -> list[dict[str, object]] | None:
    """Return stored worker node groups, or ``None`` when the record omits them."""
    raw_workers = config.get("workers")
    if raw_workers is None:
        return None
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
    return workers or None


def spec_from_stored_row(row: dict[str, object]) -> ClusterSpec:
    """Build a Nebius cluster spec from an Arena byoc cluster record."""
    # Cycle: provisioning.spec imports NebiusSpec from this module.
    from agilerl.arena.byoc.provisioning.spec import ClusterSpec, placement_values

    name = _stored_text(row.get("name"))
    if name is None:
        msg = "Arena cluster record is missing a name."
        raise ValueError(msg)
    config = _nebius_stored_config(row, name)
    state = _nebius_state_from_config(config, name)
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
    nebius: dict[str, object] = {
        "tenant_id": tenant_id,
        "object_storage": {"size_gib": int(size)},
    }
    for field in ("region", "project_id", "storage_project_id", "subnet_id"):
        filled = _stored_text(config.get(field))
        if filled is not None:
            nebius[field] = filled
    workers = _nebius_workers_from_config(config, name)
    if workers is not None:
        nebius["workers"] = workers
    stored_module = config.get("module")
    module_version = (
        _stored_text(stored_module.get("version"))
        if isinstance(stored_module, dict)
        else None
    )
    payload: dict[str, object] = {
        "provider": "nebius",
        "name": name,
        "terraform_state": state,
        "arena": {
            "storage": {
                "bucket": storage_bucket,
                "endpoint": storage_endpoint,
            },
            "inference": _nebius_inference_from_row(row, domain),
        },
        "nebius": nebius,
    }
    if module_version is not None:
        payload["module_version"] = module_version
    stored_placement = placement_values(row)
    if stored_placement is not None:
        payload["values"] = stored_placement
    try:
        return ClusterSpec.model_validate(payload)
    except ValidationError as exc:
        msg = f"Cluster {name!r} stored Nebius settings are invalid: {exc}"
        raise ValueError(msg) from exc
