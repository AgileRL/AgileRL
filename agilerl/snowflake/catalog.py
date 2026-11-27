# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Read-only account queries: GPU families, compute pools, and ``setup.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agilerl.snowflake.job import lowered
from agilerl.snowflake.profile import (
    DATABASE,
    FILE_FORMAT,
    GPU_FAMILIES,
    SCHEMA,
    SETUP_JSON,
    AccountProfile,
    profile_from_mapping,
)
from agilerl.snowflake.source import SnowflakeRow, SnowflakeSession


@dataclass(frozen=True)
class GpuFamily:
    """One row of ``SHOW COMPUTE POOL INSTANCE FAMILIES``."""

    name: str
    gpu_count: int
    gpu: str


@dataclass(frozen=True)
class PoolState:
    """One compute pool returned by ``SHOW COMPUTE POOLS``."""

    name: str
    family: str
    max_nodes: int
    active_nodes: int | None


def collect(session: SnowflakeSession, statement: str) -> list[SnowflakeRow]:
    """Run one statement and return its rows.

    :param session: Snowpark session.
    :param statement: SQL.
    :return: The rows.
    """
    return list(session.sql(statement).collect())


def missing_object(exc: Exception) -> bool:
    """Whether a server error says the object is not there.

    :param exc: Server error.
    :return: True for a missing object or an unknown user-defined function.
    """
    text = str(exc).lower()
    return (
        "does not exist" in text
        or "not found" in text
        or "no such" in text
        or "unknown user-defined function" in text
    )


def load_families(session: SnowflakeSession) -> list[GpuFamily]:
    """GPU and CPU families in this region.

    :param session: Admin session.
    :return: Rows from ``SHOW COMPUTE POOL INSTANCE FAMILIES``.
    """
    return families_from(collect(session, "SHOW COMPUTE POOL INSTANCE FAMILIES"))


def families_from(rows: list[SnowflakeRow]) -> list[GpuFamily]:
    """Parse ``SHOW COMPUTE POOL INSTANCE FAMILIES``.

    :param rows: Statement rows.
    :return: One entry per family.
    """
    found = []
    for row in rows:
        values = lowered(row)
        name = str(values.get("name") or values.get("instance_family") or "")
        gpu = str(values.get("gpu") or values.get("gpu_type") or "")
        spec = GPU_FAMILIES.get(name.upper())
        if gpu == "" and spec is not None:
            gpu = spec.gpu
        found.append(GpuFamily(name, whole_number(values.get("gpu_count")), gpu))
    return found


def family_sized(family: str, families: list[GpuFamily]) -> bool:
    """Whether this region has ``family`` with the GPUs per node :data:`GPU_FAMILIES` sizes.

    Snowflake reuses some family names across clouds for nodes with other GPU
    counts. A row without ``gpu_count`` reads as 0 GPUs, so it never matches.

    :param family: GPU instance family.
    :param families: Families in this region.
    :return: True when the family is known and the region's GPU count matches.
    """
    name = family.upper()
    spec = GPU_FAMILIES.get(name)
    return spec is not None and any(
        item.name.upper() == name and item.gpu_count == spec.gpus for item in families
    )


def whole_number(value: object) -> int:
    """Integer from a ``SHOW`` column. Empty means 0.

    :param value: Column value.
    :return: The number.
    """
    if value is None or value == "":
        return 0
    return int(float(str(value)))


def optional_whole_number(values: dict[str, object], key: str) -> int | None:
    """Integer column, or ``None`` when the row does not have it.

    :param values: Lower-cased row.
    :param key: Column name.
    :return: The number, or ``None``.
    """
    if key not in values or values[key] in (None, ""):
        return None
    return whole_number(values[key])


def schema_rows(session: SnowflakeSession, statement: str) -> list[SnowflakeRow]:
    """Rows of a ``SHOW ... IN SCHEMA arena.training`` statement.

    :param session: Snowpark session.
    :param statement: The statement.
    :return: The rows, or none when the database or schema is not there.
    """
    try:
        return collect(session, statement)
    except Exception as exc:
        if missing_object(exc):
            return []
        raise


def existing_pools(session: SnowflakeSession) -> dict[str, PoolState]:
    """Compute pools in the account, keyed by upper-case name.

    :param session: Admin session.
    :return: One entry per pool ``SHOW COMPUTE POOLS`` returned.
    """
    found: dict[str, PoolState] = {}
    for row in collect(session, "SHOW COMPUTE POOLS"):
        state = pool_state(row)
        found[state.name.upper()] = state
    return found


def pool_state(row: SnowflakeRow) -> PoolState:
    """One ``SHOW COMPUTE POOLS`` or ``DESCRIBE COMPUTE POOL`` row.

    :param row: Statement row.
    :return: The pool.
    """
    values = lowered(row)
    return PoolState(
        str(values["name"]),
        str(values.get("instance_family") or ""),
        whole_number(values.get("max_nodes")),
        optional_whole_number(values, "active_nodes"),
    )


def remote_profile(
    session: SnowflakeSession,
    setup_json: str = SETUP_JSON,
    file_format: str = f"{DATABASE}.{SCHEMA}.{FILE_FORMAT}",
) -> AccountProfile | None:
    """Read ``setup.json``. ``None`` when the file is not on the stage.

    :param session: A session that can read the runs stage.
    :param setup_json: Stage path of ``setup.json``.
    :param file_format: JSON file format that reads it.
    :return: The profile, or ``None``.
    """
    rows = schema_rows(
        session, f"SELECT $1 AS DOC FROM {setup_json} (FILE_FORMAT => {file_format})"
    )
    if not rows:
        return None
    return profile_from_mapping(json.loads(str(lowered(rows[0])["doc"])))
