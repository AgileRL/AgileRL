# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Drop every Arena object setup created, then the local Arena logins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agilerl.arena.snowflake.account import quoted_identifier
from agilerl.arena.snowflake.catalog import (
    collect,
    existing_integrations,
    existing_pools,
    is_foreign_database,
    object_exists,
    remote_profile,
    require_accountadmin,
)
from agilerl.arena.snowflake.profile import (
    ADMIN_CONNECTION,
    ARENA_API_INTEGRATION,
    BROKER_POOL,
    DATABASE,
    ENV_INTEGRATION,
    HEAD_POOL,
    HF_INTEGRATION,
    JOB_CONNECTION,
    RESOURCE_MONITOR,
    ROLE,
    SERVICE_USER,
    WAREHOUSE,
    default_key_path,
    profile_pool_names,
    remove_connection,
)
from agilerl.arena.snowflake.session import connection_section
from agilerl.arena.snowflake.setup import SetupIO
from agilerl.arena.snowflake.source import SnowflakeSession

INTEGRATIONS = (HF_INTEGRATION, ARENA_API_INTEGRATION, ENV_INTEGRATION)


@dataclass(frozen=True)
class TeardownPlan:
    """Account objects teardown drops. Absent objects are left out."""

    pools: tuple[str, ...]
    integrations: tuple[str, ...]
    database: bool
    foreign_database: bool
    user: bool
    role: bool
    warehouse: bool
    monitor: bool


def run_teardown(
    session: SnowflakeSession, account: str, yes: bool, io: SetupIO
) -> None:
    """List what would be dropped, confirm once, drop it, and clear local logins.

    :param session: ACCOUNTADMIN session.
    :param account: Account identifier.
    :param yes: Skip the confirmation.
    :param io: Prompts and output.
    :raises ValueError: The session is not ACCOUNTADMIN, or the answer was no.
    """
    require_accountadmin(session)
    plan = plan_teardown(session)
    io.echo(format_plan(account, plan))
    if not yes:
        answer = io.ask("Drop these objects? [y/N]", "", False).strip().lower()
        if answer not in {"y", "yes"}:
            msg = "Nothing was dropped."
            raise ValueError(msg)
    for statement in teardown_statements(plan):
        collect(session, statement)
    removed = remove_local(account)
    io.echo("\n".join(["Dropped.", *removed]))


def plan_teardown(session: SnowflakeSession) -> TeardownPlan:
    """Read which Arena objects exist.

    :param session: ACCOUNTADMIN session.
    :return: The plan.
    """
    foreign = is_foreign_database(session)
    named = {BROKER_POOL.upper(), HEAD_POOL.upper()}
    if not foreign:
        profile = remote_profile(session)
        if profile is not None:
            named.update(name.upper() for name in profile_pool_names(profile))
    pools = tuple(
        state.name for key, state in existing_pools(session).items() if key in named
    )
    found = existing_integrations(session)
    return TeardownPlan(
        pools=pools,
        integrations=tuple(name for name in INTEGRATIONS if name.upper() in found),
        database=not foreign
        and object_exists(session, f"SHOW DATABASES LIKE '{DATABASE}'", DATABASE),
        foreign_database=foreign,
        user=object_exists(session, f"SHOW USERS LIKE '{SERVICE_USER}'", SERVICE_USER),
        role=object_exists(session, f"SHOW ROLES LIKE '{ROLE}'", ROLE),
        warehouse=object_exists(
            session, f"SHOW WAREHOUSES LIKE '{WAREHOUSE}'", WAREHOUSE
        ),
        monitor=object_exists(
            session,
            f"SHOW RESOURCE MONITORS LIKE '{RESOURCE_MONITOR}'",
            RESOURCE_MONITOR,
        ),
    )


def teardown_statements(plan: TeardownPlan) -> list[str]:
    """Drops in dependency order: services and pools, integrations, then the rest.

    :param plan: Objects to drop.
    :return: The statements.
    """
    statements: list[str] = []
    for pool in plan.pools:
        quoted = quoted_identifier(pool)
        statements.append(f"ALTER COMPUTE POOL {quoted} STOP ALL")
        statements.append(f"DROP COMPUTE POOL {quoted}")
    statements.extend(
        f"DROP EXTERNAL ACCESS INTEGRATION IF EXISTS {name}"
        for name in plan.integrations
    )
    if plan.database:
        statements.append(f"DROP DATABASE IF EXISTS {DATABASE}")
    if plan.user:
        statements.append(f"DROP USER IF EXISTS {SERVICE_USER}")
    if plan.role:
        statements.append(f"DROP ROLE IF EXISTS {ROLE}")
    if plan.warehouse:
        statements.append(f"DROP WAREHOUSE IF EXISTS {WAREHOUSE}")
    if plan.monitor:
        statements.append(f"DROP RESOURCE MONITOR IF EXISTS {RESOURCE_MONITOR}")
    return statements


def format_plan(account: str, plan: TeardownPlan) -> str:
    """Block printed before the confirmation.

    :param account: Account identifier.
    :param plan: Objects to drop.
    :return: The block.
    """
    lines = [f"Account {account}. Teardown drops:"]
    lines.extend(f"  compute pool {pool} and its services" for pool in plan.pools)
    lines.extend(f"  integration {name}" for name in plan.integrations)
    if plan.database:
        lines.append(
            f"  database {DATABASE}: secrets, network rules, budget, stage, "
            "image repository, checkpoints, and metrics"
        )
    for present, label in (
        (plan.user, f"user {SERVICE_USER}"),
        (plan.role, f"role {ROLE}"),
        (plan.warehouse, f"warehouse {WAREHOUSE}"),
        (plan.monitor, f"resource monitor {RESOURCE_MONITOR}"),
    ):
        if present:
            lines.append(f"  {label}")
    if plan.foreign_database:
        lines.append(f"  Database {DATABASE} is not an Arena database. It stays.")
    lines.append(
        f"  The local {JOB_CONNECTION} and {ADMIN_CONNECTION} connections on this machine."
    )
    return "\n".join(lines)


def remove_local(account: str) -> list[str]:
    """Delete ``[arena]``, ``[arena-admin]``, and setup's key.

    A key passed with ``--key`` stays; only the key setup generated is deleted.

    :param account: Account identifier.
    :return: One line per local item removed.
    """
    removed = []
    job = connection_section(JOB_CONNECTION) or {}
    key = Path(str(job.get("private_key_file", ""))).expanduser()
    for name in (JOB_CONNECTION, ADMIN_CONNECTION):
        section = connection_section(name)
        if section is not None and str(section.get("account", "")) == account:
            remove_connection(name)
            removed.append(f"Removed the [{name}] connection.")
    if (
        str(job.get("account", "")) == account
        and key == default_key_path()
        and key.is_file()
    ):
        key.unlink()
        removed.append(f"Deleted {key}.")
    return removed
