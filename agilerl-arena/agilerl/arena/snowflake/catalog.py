# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Read-only account queries shared by the admin, submit, and publish commands."""

from __future__ import annotations

import json
from dataclasses import dataclass

from agilerl.arena.snowflake.job import lowered
from agilerl.arena.snowflake.profile import (
    BUDGET,
    DATABASE,
    GPU_FAMILIES,
    HF_INTEGRATION,
    METRICS_TABLE,
    RESOURCE_MONITOR,
    SCHEMA,
    SERVICE_USER,
    SETUP_JSON,
    WAREHOUSE,
    AccountProfile,
    profile_from_mapping,
)
from agilerl.arena.snowflake.source import SnowflakeRow, SnowflakeSession

QUALIFIED_BUDGET = f"{DATABASE}.{SCHEMA}.{BUDGET}"
SET_EVENT_TABLE = "ALTER ACCOUNT SET EVENT_TABLE = SNOWFLAKE.TELEMETRY.EVENTS"
NO_SETUP = (
    "This account has no Arena setup at @arena.training.runs/_arena/setup.json.\n"
    "Ask an account admin to run arena snowflake setup."
)
FOREIGN_DATABASE = (
    "A database named arena already exists and it is not an Arena account.\n"
    "Setup leaves it alone.\n"
    "Nothing was created."
)


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
    :return: True for "does not exist", "not found", or "no such" errors.
    """
    text = str(exc).lower()
    return "does not exist" in text or "not found" in text or "no such" in text


def one_line(exc: Exception) -> str:
    """First line of a server error.

    :param exc: Server error.
    :return: One line.
    """
    text = str(exc).strip()
    if text == "":
        return "unknown error"
    return text.splitlines()[0]


def accountadmin_message(role: str) -> str:
    """Message when a command cannot use ACCOUNTADMIN.

    :param role: The role the session is using.
    :return: The message.
    """
    return (
        f"This session cannot use ACCOUNTADMIN (current role: {role}).\n"
        "Setup creates the role, compute pool, integrations, and service user, "
        "which require ACCOUNTADMIN.\n"
        "Sign in as a user with that role and run the command again.\n"
        "Nothing was created."
    )


def require_accountadmin(session: SnowflakeSession) -> None:
    """Switch to ACCOUNTADMIN.

    :param session: Admin session.
    :raises ValueError: The session cannot use ACCOUNTADMIN.
    """
    role = read_identity(session)[1]
    try:
        collect(session, "USE ROLE ACCOUNTADMIN")
    except Exception as exc:
        raise ValueError(accountadmin_message(role)) from exc
    role = current_role(session)
    if role.upper() != "ACCOUNTADMIN":
        raise ValueError(accountadmin_message(role))


def use_arena_warehouse(session: SnowflakeSession) -> None:
    """Make ``arena_wh`` current when it exists. Stage reads and budget calls need it.

    :param session: Admin session.
    """
    if object_exists(session, f"SHOW WAREHOUSES LIKE '{WAREHOUSE}'", WAREHOUSE):
        collect(session, f"USE WAREHOUSE {WAREHOUSE}")


def read_identity(session: SnowflakeSession) -> tuple[str, str, str, str]:
    """``CURRENT_USER``, role, account, and region.

    :param session: Snowpark session.
    :return: The four values, in that order.
    """
    rows = collect(
        session,
        "SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_ACCOUNT(), CURRENT_REGION()",
    )
    values = lowered(rows[0])
    items = list(rows[0].as_dict().values())
    return (
        _field(values, items, 0, "current_user()", "current_user"),
        _field(values, items, 1, "current_role()", "current_role"),
        _field(values, items, 2, "current_account()", "current_account"),
        _field(values, items, 3, "current_region()", "current_region"),
    )


def current_role(session: SnowflakeSession) -> str:
    """Role the session is using now.

    :param session: Snowpark session.
    :return: ``CURRENT_ROLE()``.
    """
    rows = collect(session, "SELECT CURRENT_ROLE()")
    values = lowered(rows[0])
    items = list(rows[0].as_dict().values())
    return _field(values, items, 0, "current_role()", "current_role")


def _field(
    values: dict[str, object], items: list[object], index: int, *keys: str
) -> str:
    for key in keys:
        if key in values and values[key] is not None:
            return str(values[key])
    return str(items[index])


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


def object_exists(session: SnowflakeSession, statement: str, name: str) -> bool:
    """Whether ``statement`` returns a row named ``name``.

    :param session: Snowpark session.
    :param statement: A ``SHOW`` statement.
    :param name: Object name.
    :return: True when a row matches.
    """
    return object_named(collect(session, statement), name)


def object_named(rows: list[SnowflakeRow], name: str) -> bool:
    """Whether any row's ``name`` matches, ignoring case.

    :param rows: ``SHOW`` rows.
    :param name: Expected name.
    :return: True when one row matches.
    """
    target = name.upper()
    return any(str(lowered(row).get("name", "")).upper() == target for row in rows)


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


def existing_integrations(session: SnowflakeSession) -> dict[str, bool]:
    """Arena external access integrations, upper-case name to enabled.

    :param session: Snowpark session.
    :return: One entry per ``arena_*`` integration.
    """
    found = {}
    for row in collect(session, "SHOW EXTERNAL ACCESS INTEGRATIONS LIKE 'arena_%'"):
        values = lowered(row)
        name = str(values.get("name", "")).upper()
        found[name] = str(values.get("enabled", "")).strip().lower() == "true"
    return found


def secret_names(session: SnowflakeSession) -> set[str]:
    """Secrets in ``arena.training``, upper case.

    :param session: Snowpark session.
    :return: The names, or none when the schema is not there.
    """
    rows = schema_rows(session, f"SHOW SECRETS IN SCHEMA {DATABASE}.{SCHEMA}")
    return {str(lowered(row).get("name", "")).upper() for row in rows}


def setup_json_exists(session: SnowflakeSession) -> bool:
    """Whether ``@arena.training.runs/_arena/setup.json`` is on the stage.

    :param session: Admin session.
    :return: True when ``LIST`` returns a row.
    """
    return len(schema_rows(session, f"LIST {SETUP_JSON}")) > 0


def is_foreign_database(session: SnowflakeSession) -> bool:
    """A database named ``arena`` that setup did not create.

    An account that already has the Hugging Face integration, the metrics
    table, or ``setup.json`` is an Arena account.

    :param session: Admin session.
    :return: Whether setup and teardown must leave the database alone.
    """
    if not object_exists(session, "SHOW DATABASES LIKE 'arena'", DATABASE):
        return False
    if HF_INTEGRATION.upper() in existing_integrations(session):
        return False
    tables = schema_rows(
        session, "SHOW TABLES LIKE 'ARENA_METRICS' IN SCHEMA arena.training"
    )
    if object_named(tables, METRICS_TABLE):
        return False
    if setup_json_exists(session):
        return False
    return len(collect(session, "SHOW SCHEMAS IN DATABASE arena")) > 0


def service_user_key(session: SnowflakeSession) -> str | None:
    """Public key already on ``arena_job_user``.

    :param session: Admin session.
    :return: ``None`` when the user does not exist, or ``""`` when it has no key.
    """
    rows = collect(session, "SHOW USERS LIKE 'arena_job_user'")
    if not object_named(rows, SERVICE_USER):
        return None
    properties = {
        str(values.get("property", "")).upper(): values.get("value")
        for values in map(lowered, collect(session, "DESCRIBE USER arena_job_user"))
    }
    raw = properties.get("RSA_PUBLIC_KEY")
    if raw is None or str(raw).strip().lower() in {"", "null"}:
        return ""
    return str(raw)


def linked_resources(session: SnowflakeSession) -> str:
    """Lower-case text of the budget's linked resources.

    :param session: Admin session.
    :return: Row values joined, so a pool name can be searched in the text.
        Empty when the budget does not exist yet.
    """
    rows = schema_rows(session, f"CALL {QUALIFIED_BUDGET}!GET_LINKED_RESOURCES()")
    return " ".join(
        str(value).lower() for row in rows for value in row.as_dict().values()
    )


def monitor_quota(session: SnowflakeSession) -> int | None:
    """Monthly credit quota on ``arena_monitor``.

    :param session: Admin session.
    :return: The quota, or ``None`` when the monitor does not exist.
    """
    for row in collect(session, f"SHOW RESOURCE MONITORS LIKE '{RESOURCE_MONITOR}'"):
        values = lowered(row)
        if str(values.get("name", "")).upper() == RESOURCE_MONITOR.upper():
            return optional_whole_number(values, "credit_quota")
    return None


def budget_limit(session: SnowflakeSession) -> int | None:
    """Monthly spending limit on the Arena budget.

    :param session: Admin session.
    :return: The limit, or ``None`` when the budget does not exist.
    """
    rows = schema_rows(session, f"CALL {QUALIFIED_BUDGET}!GET_SPENDING_LIMIT()")
    if not rows:
        return None
    first = next(iter(rows[0].as_dict().values()), None)
    if first in (None, ""):
        return None
    return whole_number(first)


def remote_profile(session: SnowflakeSession) -> AccountProfile | None:
    """Read ``setup.json``. ``None`` when the file is not on the stage.

    :param session: A session that can read the runs stage.
    :return: The profile, or ``None``.
    """
    rows = schema_rows(
        session,
        "SELECT $1 AS DOC FROM @arena.training.runs/_arena/setup.json "
        "(FILE_FORMAT => arena.training.arena_json)",
    )
    if not rows:
        return None
    return profile_from_mapping(json.loads(str(lowered(rows[0])["doc"])))


def active_event_table(session: SnowflakeSession) -> str:
    """The account's ``EVENT_TABLE`` parameter.

    :param session: Snowpark session.
    :return: The table name, or ``""`` when none is set.
    """
    for row in collect(session, "SHOW PARAMETERS LIKE 'EVENT_TABLE' IN ACCOUNT"):
        values = lowered(row)
        if str(values.get("key", "")).upper() == "EVENT_TABLE":
            return str(values.get("value") or "").strip()
    return ""


def event_table_warning(table: str) -> str | None:
    """Line printed when the account has no active event table.

    :param table: :func:`active_event_table`.
    :return: The warning, or ``None`` when a table is set.
    """
    if table != "":
        return None
    return (
        "Warning: this account has no active event table, so arena snowflake "
        f"logs cannot read finished jobs. To keep their logs, run: {SET_EVENT_TABLE}"
    )
