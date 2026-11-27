# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Account profile in ``setup.json``, its pools, and the Snowflake object names."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from agilerl.snowflake.source import SnowflakeSession

DATABASE = "arena"
SCHEMA = "training"
RUNS_STAGE = "runs"
FILE_FORMAT = "arena_json"
METRICS_TABLE = "arena_metrics"
ARENA_ORG_KEY_SECRET = "arena_org_key"
ARENA_EXTERNAL_USER_SECRET = "arena_external_user_id"
ARENA_BASE_URL_SECRET = "arena_base_url"
# Native App only: the claim secret of an open app_code.connect claim.
ARENA_CLAIM_SECRET = "arena_claim_secret"
ARENA_API_RULE = "arena_api_egress"
ARENA_API_INTEGRATION = "arena_api_eai"
SETUP_PREFIX = "@arena.training.runs/_arena"
SETUP_JSON = "@arena.training.runs/_arena/setup.json"
CPU_POOL_MIN_NODES = 1
CPU_POOL_MAX_NODES = 4
POOL_SUSPEND_SECS = 300
DEFAULT_GPU_NODES = 2

NO_POOLS = (
    "This app's setup.json has no compute pools.\n"
    "Run CALL <app>.app_code.set_pools(['<gpu family>']) first."
)


@dataclass(frozen=True)
class GpuFamilySpec:
    """One Snowflake GPU instance family.

    ``memory_gib`` is per GPU. ``host_memory_gib`` is the node's RAM.
    ``node_cap`` is Snowflake's default node limit; ``0`` means on request.
    """

    gpus: int
    memory_gib: int
    host_memory_gib: int
    node_cap: int
    gpu: str


# Snowflake's AWS, Azure, and Google Cloud instance family tables.
GPU_FAMILIES = {
    "GPU_NV_S": GpuFamilySpec(1, 24, 27, 150, "A10G"),
    "GPU_NV_M": GpuFamilySpec(4, 24, 178, 10, "A10G"),
    "GPU_NV_L": GpuFamilySpec(8, 40, 1112, 10, "A100"),
    "GPU_L40S_G1_8": GpuFamilySpec(1, 48, 58, 5, "L40S"),
    "GPU_L40S_G1_16": GpuFamilySpec(1, 48, 116, 5, "L40S"),
    "GPU_L40S_G1_48": GpuFamilySpec(4, 48, 368, 0, "L40S"),
    "GPU_L40S_G1_192": GpuFamilySpec(8, 48, 1436, 0, "L40S"),
    "GPU_R6K_G1_8": GpuFamilySpec(1, 96, 58, 5, "RTX PRO 6000"),
    "GPU_R6K_G1_16": GpuFamilySpec(1, 96, 116, 5, "RTX PRO 6000"),
    "GPU_R6K_G1_32": GpuFamilySpec(1, 96, 240, 5, "RTX PRO 6000"),
    "GPU_R6K_G1_48": GpuFamilySpec(2, 96, 490, 0, "RTX PRO 6000"),
    "GPU_R6K_G1_96": GpuFamilySpec(4, 96, 984, 0, "RTX PRO 6000"),
    "GPU_R6K_G1_192": GpuFamilySpec(8, 96, 1843, 0, "RTX PRO 6000"),
    "GPU_NV_XS": GpuFamilySpec(1, 16, 26, 10, "T4"),
    "GPU_NV_SM": GpuFamilySpec(1, 24, 424, 10, "A10"),
    "GPU_NV_2M": GpuFamilySpec(2, 24, 858, 5, "A10"),
    # "A100" is the 40GB row. 80 GiB families need "80" in the name.
    "GPU_NV_3M": GpuFamilySpec(2, 80, 424, 0, "A100 80GB"),
    "GPU_NV_SL": GpuFamilySpec(4, 80, 858, 0, "A100 80GB"),
    "GPU_GCP_NV_L4_1_24G": GpuFamilySpec(1, 24, 28, 10, "L4"),
    "GPU_GCP_NV_L4_4_24G": GpuFamilySpec(4, 24, 178, 10, "L4"),
    "GPU_A100_G1_12": GpuFamilySpec(1, 40, 77, 0, "A100"),
    "GPU_A100_G1_48": GpuFamilySpec(4, 40, 324, 0, "A100"),
    "GPU_GCP_NV_A100_8_40G": GpuFamilySpec(8, 40, 654, 0, "A100"),
}


@dataclass(frozen=True)
class CpuPool:
    """Broker or head pool saved in ``setup.json``."""

    pool: str
    family: str


@dataclass(frozen=True)
class GpuPool:
    """One GPU family's pool and the cap recorded by setup."""

    pool: str
    max_gpus: int
    gpus_per_node: int
    gpu_memory_gib: int


@dataclass(frozen=True)
class AccountPools:
    """Compute pools one account setup created."""

    broker: CpuPool
    head: CpuPool
    gpu: dict[str, GpuPool]


@dataclass(frozen=True)
class AccountProfile:
    """Pools and image ``setup.json`` records for the account."""

    image: str
    warehouse: str
    stage: str
    metrics_table: str
    external_access_integrations: tuple[str, ...]
    pools: AccountPools

    def as_dict(self) -> dict[str, object]:
        """Fields written to ``setup.json``. No key material."""
        return {
            "image": self.image,
            "warehouse": self.warehouse,
            "stage": self.stage,
            "metrics_table": self.metrics_table,
            "external_access_integrations": list(self.external_access_integrations),
            "pools": {
                "broker": {
                    "pool": self.pools.broker.pool,
                    "family": self.pools.broker.family,
                },
                "head": {
                    "pool": self.pools.head.pool,
                    "family": self.pools.head.family,
                },
                "gpu": {
                    family: {
                        "pool": item.pool,
                        "max_gpus": item.max_gpus,
                        "gpus_per_node": item.gpus_per_node,
                        "gpu_memory_gib": item.gpu_memory_gib,
                    }
                    for family, item in self.pools.gpu.items()
                },
            },
        }


def profile_from_mapping(section: Mapping[str, object]) -> AccountProfile:
    """Build a profile from a setup.json object.

    :param section: Saved fields.
    :return: The profile.
    :raises ValueError: The section has no compute pools.
    """
    raw_pools = section.get("pools")
    if not isinstance(raw_pools, Mapping):
        raise ValueError(NO_POOLS)
    raw_integrations = section.get("external_access_integrations")
    names: list[str] = []
    if isinstance(raw_integrations, (list, tuple)):
        names.extend(str(item) for item in raw_integrations)
    return AccountProfile(
        image=str(section.get("image") or ""),
        warehouse=str(section["warehouse"]),
        stage=str(section["stage"]),
        metrics_table=str(section["metrics_table"]),
        external_access_integrations=tuple(names),
        pools=_pools_from(raw_pools),
    )


def _pools_from(raw: Mapping[str, object]) -> AccountPools:
    broker = raw.get("broker")
    head = raw.get("head")
    gpu = raw.get("gpu")
    if (
        not isinstance(broker, Mapping)
        or not isinstance(head, Mapping)
        or not isinstance(gpu, Mapping)
    ):
        raise ValueError(NO_POOLS)
    return AccountPools(
        broker=CpuPool(str(broker["pool"]), str(broker["family"])),
        head=CpuPool(str(head["pool"]), str(head["family"])),
        gpu={str(family): _gpu_pool(item) for family, item in gpu.items()},
    )


def _gpu_pool(raw: object) -> GpuPool:
    if not isinstance(raw, Mapping):
        raise ValueError(NO_POOLS)
    return GpuPool(
        pool=str(raw["pool"]),
        max_gpus=int(str(raw["max_gpus"])),
        gpus_per_node=int(str(raw["gpus_per_node"])),
        gpu_memory_gib=int(str(raw["gpu_memory_gib"])),
    )


def put_setup_json(
    session: SnowflakeSession, profile: AccountProfile, prefix: str = SETUP_PREFIX
) -> None:
    """Upload ``setup.json`` to the runs stage. The document has no key material.

    :param session: Snowpark session.
    :param profile: Account record.
    :param prefix: Stage directory that holds ``setup.json``.
    """
    text = json.dumps(profile.as_dict(), indent=2) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "setup.json"
        path.write_text(text, encoding="utf-8")
        session.file.put(str(path), prefix, auto_compress=False, overwrite=True)


def sole_gpu_pool(profile: AccountProfile) -> str | None:
    """GPU pool name when the profile records one GPU family.

    :param profile: Account record.
    :return: The pool, or ``None`` when there are several GPU families.
    """
    pools = [item.pool for item in profile.pools.gpu.values()]
    if len(pools) == 1:
        return pools[0]
    return None
