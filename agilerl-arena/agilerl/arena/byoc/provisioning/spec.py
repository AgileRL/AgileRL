# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Cluster provisioning specification models."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Unpack

from agilerl.arena.byoc.provisioning.inference import (
    DEFAULT_INFERENCE_HOSTNAME_TEMPLATE,
    normalize_inference_domain,
)
from agilerl.arena.byoc.provisioning.module_catalog import MODULE_VERSION_RE
from agilerl.arena.byoc.provisioning.providers import provider_names
from agilerl.arena.byoc.provisioning.providers.aws.spec import AwsSpec
from agilerl.arena.byoc.provisioning.providers.nebius.spec import NebiusSpec

DEFAULT_STORAGE_SECRET_NAME = "storage"
DEFAULT_LAB_STORAGE_SECRET_NAME = "arena-storage"


class TerraformStateSpec(BaseModel):
    """Dedicated S3-compatible bucket for Terraform state."""

    model_config = ConfigDict(extra="forbid")

    bucket: str = Field(min_length=1)
    key: str | None = None
    endpoint: str | None = None
    create_bucket: bool = False

    @field_validator("bucket")
    @classmethod
    def strip_bucket(cls, value: str) -> str:
        """Require a non-empty bucket name."""
        stripped = value.strip()
        if not stripped:
            msg = "must be a non-empty string."
            raise ValueError(msg)
        return stripped

    @field_validator("key", "endpoint")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        """Treat blank optional strings as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def object_key(self, cluster_name: str) -> str:
        """Return the state object key, defaulting to ``clusters/<name>/terraform.tfstate``."""
        return self.key or f"clusters/{cluster_name}/terraform.tfstate"

    def s3_endpoint(self, storage_endpoint: str) -> str:
        """Return the S3 API endpoint for this state bucket."""
        return self.endpoint or storage_endpoint


class ArenaStorageSpec(BaseModel):
    """Object storage Arena registers for a cluster."""

    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    prefix: str | None = None
    secret_name: str | None = None
    install: bool = False

    @field_validator("endpoint", "bucket")
    @classmethod
    def strip_required(cls, value: str) -> str:
        """Require a non-empty string."""
        stripped = value.strip()
        if not stripped:
            msg = "must be a non-empty string."
            raise ValueError(msg)
        return stripped

    @field_validator("prefix", "secret_name")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        """Treat a blank optional string as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def resolved_secret_name(self) -> str:
        """Return the Kubernetes Secret name Arena should write."""
        if self.secret_name:
            return self.secret_name
        if self.install:
            return DEFAULT_LAB_STORAGE_SECRET_NAME
        return DEFAULT_STORAGE_SECRET_NAME


class ArenaInferenceSpec(BaseModel):
    """Inference routing settings for a cluster."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    hostname_template: str = DEFAULT_INFERENCE_HOSTNAME_TEMPLATE
    tls_secret_name: str | None = None

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        """Normalize and validate the inference domain."""
        return normalize_inference_domain(value)

    @field_validator("tls_secret_name")
    @classmethod
    def validate_tls_secret_name(cls, value: str | None) -> str | None:
        """Treat a blank TLS secret name as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ArenaGatewaySpec(BaseModel):
    """Gateway settings Arena and the cloud provider both read."""

    model_config = ConfigDict(extra="forbid")

    enable: bool = True
    name: str = "arena"
    ingress_class_name: str | None = None
    parent_refs: list[dict[str, Any]] | None = None

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        """Require a non-empty gateway name."""
        stripped = value.strip()
        if not stripped:
            msg = "arena.gateway.name must be a non-empty string."
            raise ValueError(msg)
        return stripped

    @field_validator("ingress_class_name")
    @classmethod
    def strip_ingress_class_name(cls, value: str | None) -> str | None:
        """Treat a blank ingress class as unset."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ArenaWorkloadsSpec(BaseModel):
    """Scheduling settings passed to Arena at registration."""

    model_config = ConfigDict(extra="forbid")

    preprocessing_resource_class: str | None = None
    ray_data_storage_class_name: str | None = None
    ray_data_pvc_size: str | None = None


class ClusterValuesSpec(BaseModel):
    """Registration values shared by every cloud."""

    model_config = ConfigDict(extra="forbid")

    placement: Literal["in-cluster", "ephemeral"] = "in-cluster"


class ArenaSpec(BaseModel):
    """Arena settings. The same keys for every cloud."""

    model_config = ConfigDict(extra="forbid")

    storage: ArenaStorageSpec
    inference: ArenaInferenceSpec
    gateway: ArenaGatewaySpec = Field(default_factory=ArenaGatewaySpec)
    workloads: ArenaWorkloadsSpec = Field(default_factory=ArenaWorkloadsSpec)


class ClusterSpec(BaseModel):
    """Cluster provisioning input: Arena settings plus one cloud block."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["nebius", "aws"]
    name: str = Field(min_length=1)
    module_version: str | None = None
    terraform_state: TerraformStateSpec
    arena: ArenaSpec
    values: ClusterValuesSpec = Field(default_factory=ClusterValuesSpec)
    nebius: NebiusSpec | None = None
    aws: AwsSpec | None = None

    @field_validator("module_version")
    @classmethod
    def strip_module_version(cls, value: str | None) -> str | None:
        """Require MAJOR.MINOR.PATCH when a module release is pinned in the spec."""
        if value is None:
            return None
        stripped = value.strip()
        if MODULE_VERSION_RE.fullmatch(stripped) is None:
            msg = "module_version must be MAJOR.MINOR.PATCH."
            raise ValueError(msg)
        return stripped

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_sections(cls, data: object) -> object:
        """Reject the old cloud values blob and the registration section."""
        if not isinstance(data, dict):
            return data
        if "registration" in data:
            msg = (
                "Use arena and a nebius or aws block. "
                "values and registration are not accepted."
            )
            raise ValueError(msg)
        values = data.get("values")
        if isinstance(values, dict) and set(values) - {"placement"}:
            msg = (
                "Use arena and a nebius or aws block. "
                "values and registration are not accepted."
            )
            raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def require_matching_cloud_block(self) -> ClusterSpec:
        """Require the cloud block named by provider and no other cloud block."""
        if self.provider == "nebius":
            if self.nebius is None:
                msg = "nebius block is required when provider is nebius."
                raise ValueError(msg)
            if self.aws is not None:
                msg = "aws block is not allowed when provider is nebius."
                raise ValueError(msg)
        else:
            if self.aws is None:
                msg = "aws block is required when provider is aws."
                raise ValueError(msg)
            if self.nebius is not None:
                msg = "nebius block is not allowed when provider is aws."
                raise ValueError(msg)
        if self.terraform_state.bucket == self.arena.storage.bucket:
            msg = (
                "terraform_state.bucket must not be the experiment-data bucket "
                f"{self.arena.storage.bucket!r}."
            )
            raise ValueError(msg)
        return self

    def nebius_cloud(self) -> NebiusSpec:
        """Return the Nebius block."""
        if self.nebius is None:
            msg = f"Cluster {self.name!r} has no nebius block."
            raise ValueError(msg)
        return self.nebius

    def aws_cloud(self) -> AwsSpec:
        """Return the AWS block."""
        if self.aws is None:
            msg = f"Cluster {self.name!r} has no aws block."
            raise ValueError(msg)
        return self.aws

    def terraform_values(self) -> dict[str, object]:
        """Return Terraform variables for this cluster."""
        storage = self.arena.storage
        if self.provider == "nebius":
            return self.nebius_cloud().terraform_values(
                storage_bucket_name=storage.bucket,
                storage_endpoint=storage.endpoint,
            )
        return self.aws_cloud().terraform_values(
            storage_bucket_name=storage.bucket,
            storage_endpoint=storage.endpoint,
            enable_gateway_api=self.arena.gateway.enable,
        )


class ClusterSpecOverrides(TypedDict, total=False):
    """Explicit CLI values applied on top of a cluster spec."""

    name: str | None
    storage_endpoint: str | None
    storage_bucket: str | None
    storage_prefix: str | None
    storage_secret_name: str | None
    ingress_class_name: str | None
    gateway_api_parent_refs: list[dict[str, Any]] | None
    preprocessing_resource_class: str | None
    ray_data_storage_class_name: str | None
    ray_data_pvc_size: str | None
    domain: str | None
    hostname_template: str | None
    tls_secret_name: str | None
    placement: str | None


def _present_fields(values: dict[str, object]) -> dict[str, object]:
    """Return the entries whose values were set."""
    return {key: value for key, value in values.items() if value is not None}


def _arena_with_overrides(
    spec: ClusterSpec, overrides: ClusterSpecOverrides
) -> ArenaSpec:
    """Return arena settings with explicit CLI values applied."""
    storage = spec.arena.storage.model_copy(
        update=_present_fields(
            {
                "endpoint": overrides.get("storage_endpoint"),
                "bucket": overrides.get("storage_bucket"),
                "prefix": overrides.get("storage_prefix"),
                "secret_name": overrides.get("storage_secret_name"),
            }
        )
    )
    inference = spec.arena.inference.model_copy(
        update=_present_fields(
            {
                "domain": overrides.get("domain"),
                "hostname_template": overrides.get("hostname_template"),
                "tls_secret_name": overrides.get("tls_secret_name"),
            }
        )
    )
    gateway = spec.arena.gateway.model_copy(
        update=_present_fields(
            {
                "ingress_class_name": overrides.get("ingress_class_name"),
                "parent_refs": overrides.get("gateway_api_parent_refs"),
            }
        )
    )
    workloads = spec.arena.workloads.model_copy(
        update=_present_fields(
            {
                "preprocessing_resource_class": overrides.get(
                    "preprocessing_resource_class"
                ),
                "ray_data_storage_class_name": overrides.get(
                    "ray_data_storage_class_name"
                ),
                "ray_data_pvc_size": overrides.get("ray_data_pvc_size"),
            }
        )
    )
    return spec.arena.model_copy(
        update={
            "storage": storage,
            "inference": inference,
            "gateway": gateway,
            "workloads": workloads,
        }
    )


def apply_cluster_spec_overrides(
    spec: ClusterSpec, **overrides: Unpack[ClusterSpecOverrides]
) -> ClusterSpec:
    """Return a copy of a cluster spec with explicit CLI values applied."""
    name = overrides.get("name")
    placement = overrides.get("placement")
    values = spec.values
    if placement is not None:
        try:
            values = ClusterValuesSpec.model_validate({"placement": placement})
        except ValidationError as exc:
            msg = "placement must be in-cluster or ephemeral."
            raise ValueError(msg) from exc
    return spec.model_copy(
        update={
            "name": name.strip() if name else spec.name,
            "values": values,
            "arena": _arena_with_overrides(spec, overrides),
        }
    )


SPEC_PLACEMENT_TO_API = {
    "in-cluster": "byoc_in_cluster",
    "ephemeral": "byoc_ephemeral",
}
API_PLACEMENT_TO_SPEC = {
    "byoc_in_cluster": "in-cluster",
    "in_cluster": "in-cluster",
    "in-cluster": "in-cluster",
    "byoc_ephemeral": "ephemeral",
    "ephemeral": "ephemeral",
}


def api_placement(placement: str) -> str:
    """Return the Arena placement enum for a cluster spec value.

    :param placement: ``in-cluster`` or ``ephemeral``.
    :return: ``byoc_in_cluster`` or ``byoc_ephemeral``.
    """
    mapped = SPEC_PLACEMENT_TO_API.get(placement)
    if mapped is None:
        msg = "placement must be in-cluster or ephemeral."
        raise ValueError(msg)
    return mapped


def placement_values(row: dict[str, object]) -> dict[str, str] | None:
    """Map a stored cluster placement onto ``in-cluster`` or ``ephemeral``."""
    raw = row.get("placement")
    if not isinstance(raw, str):
        return None
    placement = raw.strip()
    if not placement:
        return None
    return {"placement": API_PLACEMENT_TO_SPEC.get(placement, placement)}


CLUSTER_NAME_PLACEHOLDER = "__CLUSTER_NAME__"


def default_cluster_name(provider: str) -> str:
    """Return the default cluster name for a cloud provider."""
    return f"arena-{provider}"


def render_default_cluster_spec(provider: str, name: str) -> str:
    """Return a default cluster spec YAML for *provider*."""
    if provider not in provider_names():
        msg = f"Unsupported cluster spec provider {provider!r}."
        raise ValueError(msg)
    cluster_name = name.strip()
    if not cluster_name:
        msg = "Cluster name must be a non-empty string."
        raise ValueError(msg)
    template = (
        files("agilerl.arena.byoc.provisioning.templates")
        .joinpath(f"{provider}-cluster.yaml")
        .read_text(encoding="utf-8")
    )
    return template.replace(CLUSTER_NAME_PLACEHOLDER, cluster_name)


def write_default_cluster_spec(
    path: Path,
    provider: str,
    name: str,
    force: bool = False,
) -> Path:
    """Write a default cluster spec YAML to *path*."""
    destination = path.expanduser()
    if destination.exists() and not force:
        msg = f"{destination} already exists. Pass --force to overwrite."
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_default_cluster_spec(provider=provider, name=name),
        encoding="utf-8",
    )
    return destination


def load_cluster_spec(path: Path) -> ClusterSpec:
    """Load and validate a cluster provisioning YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Could not read cluster spec {path}: {exc}"
        raise ValueError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in cluster spec {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = "Cluster spec must contain a YAML mapping."
        raise ValueError(msg)
    try:
        return ClusterSpec.model_validate(raw)
    except ValidationError as exc:
        msg = f"Invalid cluster spec {path}: {exc}"
        raise ValueError(msg) from exc
