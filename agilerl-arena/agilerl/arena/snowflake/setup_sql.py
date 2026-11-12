# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Statements ``arena snowflake setup`` runs. Every statement is safe to run again.

Objects a running head references (secrets, network rules, integrations) are
created when missing and altered in place, so a re-run never drops them.
"""

from __future__ import annotations

from dataclasses import dataclass

from agilerl.arena.snowflake.license import arena_api_hostname
from agilerl.arena.snowflake.profile import (
    ARENA_API_INTEGRATION,
    ARENA_API_RULE,
    ARENA_BASE_URL_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_ORG_KEY_SECRET,
    BUDGET,
    DATABASE,
    FILE_FORMAT,
    HF_INTEGRATION,
    HF_RULE,
    IMAGE_REPOSITORY,
    METRICS_TABLE,
    POOL_SUSPEND_SECS,
    RESOURCE_MONITOR,
    ROLE,
    RUNS_STAGE,
    SCHEMA,
    SERVICE_USER,
    WAREHOUSE,
)

HF_HOSTS = (
    "huggingface.co:443",
    "cas-server.xethub.hf.co:443",
    "cas-server.xethub-eu.hf.co:443",
    "transfer.xethub.hf.co:443",
    "transfer.xethub-eu.hf.co:443",
    "us.aws.cdn.hf.co:443",
    "us.gcp.cdn.hf.co:443",
    "cdn-lfs.hf.co:443",
    "cdn-lfs-us-1.hf.co:443",
    "cdn-lfs-eu-1.hf.co:443",
)
SCHEMA_NAME = f"{DATABASE}.{SCHEMA}"
QUALIFIED_BUDGET = f"{SCHEMA_NAME}.{BUDGET}"
PROBE_INTEGRATION = "arena_probe_eai"
# Snowflake-managed rule, so the probe needs no database of its own.
PROBE_RULE = "snowflake.external_access.pypi_rule"


@dataclass(frozen=True)
class PoolPlan:
    """One pool setup will create or resize."""

    name: str
    family: str
    role: str
    min_nodes: int
    max_nodes: int
    max_gpus: int | None


@dataclass(frozen=True)
class ArenaCredentials:
    """Values setup stores in the three Arena secrets."""

    org_key: str
    external_user_id: str
    base_url: str


@dataclass(frozen=True)
class SetupPlan:
    """Everything the setup statements depend on.

    ``existing_max_nodes`` maps upper-case pool names already in the account
    to their max nodes. ``stored_key`` is ``None`` when ``arena_job_user`` does
    not exist. ``arena`` is ``None`` when the secrets stay as they are.
    ``integrations`` holds upper-case names of integrations already created.
    ``linked`` is the budget's linked-resource text.
    """

    pools: tuple[PoolPlan, ...]
    existing_max_nodes: dict[str, int]
    warehouse_credits: int
    budget_credits: int
    notify_email: str | None
    public_key: str
    stored_key: str | None
    rotate_key: bool
    arena: ArenaCredentials | None
    integrations: frozenset[str]
    linked: str


def setup_statements(plan: SetupPlan) -> list[str]:
    """Every statement setup runs, in order.

    :param plan: Resolved setup.
    :return: The statements.
    """
    return [
        f"CREATE ROLE IF NOT EXISTS {ROLE}",
        f"CREATE DATABASE IF NOT EXISTS {DATABASE}",
        f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}",
        network_rule_sql(HF_RULE, HF_HOSTS),
        integration_sql(HF_INTEGRATION, HF_RULE, plan.integrations),
        *warehouse_statements(plan.warehouse_credits),
        *[
            statement
            for pool in plan.pools
            for statement in pool_statements(
                plan.existing_max_nodes.get(pool.name.upper()), pool
            )
        ],
        *budget_statements(plan),
        *storage_statements(),
        *user_statements(plan.public_key, plan.stored_key, plan.rotate_key),
        *arena_statements(plan.arena, plan.integrations),
        *grant_statements([pool.name for pool in plan.pools]),
    ]


def script(statements: list[str]) -> str:
    """Statements as one worksheet, each ending in ``;``.

    :param statements: SQL.
    :return: The worksheet text.
    """
    return "\n\n".join(f"{statement};" for statement in statements) + "\n"


def string_literal(value: str) -> str:
    """Single-quoted SQL string.

    :param value: Raw text.
    :return: The literal.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def network_rule_sql(name: str, hosts: tuple[str, ...]) -> str:
    """Egress rule in ``arena.training``. Re-running sets the host list.

    :param name: Rule name.
    :param hosts: ``host:port`` values.
    :return: The statement.
    """
    values = ",\n    ".join(string_literal(host) for host in hosts)
    return (
        f"CREATE OR ALTER NETWORK RULE {SCHEMA_NAME}.{name}\n"
        "  TYPE = HOST_PORT\n"
        "  MODE = EGRESS\n"
        "  VALUE_LIST = (\n"
        f"    {values}\n"
        "  )"
    )


def integration_sql(name: str, rule: str, existing: frozenset[str]) -> str:
    """Create the integration, or point an existing one at ``rule`` and enable it.

    :param name: Integration name.
    :param rule: Network rule in ``arena.training``.
    :param existing: Upper-case names of integrations already in the account.
    :return: The statement.
    """
    rules = f"ALLOWED_NETWORK_RULES = ({SCHEMA_NAME}.{rule})"
    if name.upper() in existing:
        return f"ALTER EXTERNAL ACCESS INTEGRATION {name} SET {rules} ENABLED = TRUE"
    return f"CREATE EXTERNAL ACCESS INTEGRATION {name}\n  {rules}\n  ENABLED = TRUE"


def probe_statements() -> tuple[str, str]:
    """Create and drop a disabled integration. Trial accounts reject the create.

    :return: The create statement, then the drop.
    """
    return (
        (
            f"CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {PROBE_INTEGRATION}\n"
            f"  ALLOWED_NETWORK_RULES = ({PROBE_RULE})\n"
            "  ENABLED = FALSE"
        ),
        f"DROP EXTERNAL ACCESS INTEGRATION IF EXISTS {PROBE_INTEGRATION}",
    )


def warehouse_statements(credits: int) -> list[str]:
    """Warehouse, its monthly resource monitor, then ``USE WAREHOUSE`` for the budget calls.

    :param credits: Monitor credit quota.
    :return: The statements.
    """
    return [
        (
            f"CREATE WAREHOUSE IF NOT EXISTS {WAREHOUSE}\n"
            "  WAREHOUSE_SIZE = XSMALL\n"
            "  AUTO_SUSPEND = 60\n"
            "  AUTO_RESUME = TRUE\n"
            "  INITIALLY_SUSPENDED = TRUE"
        ),
        (
            f"CREATE RESOURCE MONITOR IF NOT EXISTS {RESOURCE_MONITOR}\n"
            f"  WITH CREDIT_QUOTA = {credits}\n"
            "  FREQUENCY = MONTHLY\n"
            "  START_TIMESTAMP = IMMEDIATELY\n"
            "  TRIGGERS\n"
            "    ON 80 PERCENT DO NOTIFY\n"
            "    ON 100 PERCENT DO SUSPEND"
        ),
        f"ALTER RESOURCE MONITOR {RESOURCE_MONITOR} SET CREDIT_QUOTA = {credits}",
        f"ALTER WAREHOUSE {WAREHOUSE} SET RESOURCE_MONITOR = {RESOURCE_MONITOR}",
        f"USE WAREHOUSE {WAREHOUSE}",
    ]


def pool_statements(max_nodes: int | None, plan: PoolPlan) -> list[str]:
    """Create the pool, or change its max nodes.

    :param max_nodes: Max nodes of the existing pool, or ``None`` when absent.
    :param plan: Name, family, and max nodes.
    :return: The statements.
    """
    if max_nodes is None:
        return [create_pool_sql(plan.name, plan.family, plan.min_nodes, plan.max_nodes)]
    if max_nodes != plan.max_nodes:
        return [f"ALTER COMPUTE POOL {plan.name} SET MAX_NODES = {plan.max_nodes}"]
    return []


def create_pool_sql(name: str, family: str, min_nodes: int, max_nodes: int) -> str:
    """``CREATE COMPUTE POOL`` for one Arena pool.

    :param name: Pool name.
    :param family: Instance family.
    :param min_nodes: Minimum nodes.
    :param max_nodes: Maximum nodes.
    :return: The statement.
    """
    return (
        f"CREATE COMPUTE POOL IF NOT EXISTS {name}\n"
        f"  MIN_NODES = {min_nodes}\n"
        f"  MAX_NODES = {max_nodes}\n"
        f"  INSTANCE_FAMILY = {family}\n"
        "  AUTO_RESUME = TRUE\n"
        f"  AUTO_SUSPEND_SECS = {POOL_SUSPEND_SECS}\n"
        "  INITIALLY_SUSPENDED = TRUE"
    )


def budget_statements(plan: SetupPlan) -> list[str]:
    """Budget, its limit, optional email, and every pool not yet linked.

    :param plan: Resolved setup.
    :return: The statements.
    """
    statements = [
        f"CREATE SNOWFLAKE.CORE.BUDGET IF NOT EXISTS {QUALIFIED_BUDGET}()",
        f"CALL {QUALIFIED_BUDGET}!SET_SPENDING_LIMIT({plan.budget_credits})",
    ]
    if plan.notify_email is not None:
        statements.append(
            f"CALL {QUALIFIED_BUDGET}!SET_EMAIL_NOTIFICATIONS("
            f"{string_literal(plan.notify_email)})"
        )
    statements.extend(
        budget_add_sql(pool.name)
        for pool in plan.pools
        if pool.name.lower() not in plan.linked
    )
    return statements


def budget_add_sql(pool: str) -> str:
    """Link one compute pool to the budget.

    :param pool: Compute pool name.
    :return: The statement.
    """
    return (
        f"CALL {QUALIFIED_BUDGET}!ADD_RESOURCE(\n"
        f"  SELECT SYSTEM$REFERENCE('COMPUTE POOL', '{pool}', "
        "'SESSION', 'applybudget'))"
    )


def storage_statements() -> list[str]:
    """Image repository, runs stage, JSON file format, and metrics table.

    :return: The statements.
    """
    return [
        f"CREATE IMAGE REPOSITORY IF NOT EXISTS {SCHEMA_NAME}.{IMAGE_REPOSITORY}",
        (
            f"CREATE STAGE IF NOT EXISTS {SCHEMA_NAME}.{RUNS_STAGE}\n"
            "  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')"
        ),
        (
            f"CREATE FILE FORMAT IF NOT EXISTS {SCHEMA_NAME}.{FILE_FORMAT}\n"
            "  TYPE = JSON\n"
            "  STRIP_OUTER_ARRAY = FALSE"
        ),
        (
            f"CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{METRICS_TABLE} (\n"
            "  RUN_ID STRING,\n"
            "  SEQ NUMBER,\n"
            "  TIME TIMESTAMP_TZ,\n"
            "  METRICS VARIANT\n"
            ")"
        ),
    ]


def keys_match(stored: str, body: str) -> bool:
    """Whether two public key bodies are the same, ignoring whitespace.

    :param stored: Key on the Snowflake user.
    :param body: Key from the local file.
    :return: True when they match.
    """
    return "".join(stored.split()) == "".join(body.split())


def user_statements(
    public_body: str, stored: str | None, rotate_key: bool
) -> list[str]:
    """Create the service user, or set its public key.

    :param public_body: Key body.
    :param stored: Key already on the user, or ``None`` when the user is new.
    :param rotate_key: Replace a key that is already set.
    :return: The statements.
    """
    if stored is None:
        create = (
            f"CREATE USER IF NOT EXISTS {SERVICE_USER}\n"
            "  TYPE = SERVICE\n"
            f"  DEFAULT_ROLE = {ROLE}\n"
            f"  DEFAULT_WAREHOUSE = {WAREHOUSE}\n"
            f"  RSA_PUBLIC_KEY = {string_literal(public_body)}"
        )
        return [create]
    if stored == "" or rotate_key or not keys_match(stored, public_body):
        key = string_literal(public_body)
        return [f"ALTER USER {SERVICE_USER} SET RSA_PUBLIC_KEY = {key}"]
    return []


def secret_statements(name: str, value: str) -> list[str]:
    """Create the secret when missing, then set its value.

    :param name: Secret name in ``arena.training``.
    :param value: Secret string.
    :return: The statements.
    """
    literal = string_literal(value)
    qualified = f"{SCHEMA_NAME}.{name}"
    return [
        (
            f"CREATE SECRET IF NOT EXISTS {qualified}\n"
            "  TYPE = GENERIC_STRING\n"
            f"  SECRET_STRING = {literal}"
        ),
        f"ALTER SECRET {qualified} SET SECRET_STRING = {literal}",
    ]


def arena_statements(
    arena: ArenaCredentials | None, integrations: frozenset[str]
) -> list[str]:
    """Arena secrets, the one-host Arena network rule, and ``arena_api_eai``.

    The rule host always matches the base URL secret, so both change together.

    :param arena: Values to store, or ``None`` to keep the secrets, rule, and
        integration as they are.
    :param integrations: Upper-case names of integrations already created.
    :return: The statements.
    """
    if arena is None:
        return []
    host = f"{arena_api_hostname(arena.base_url)}:443"
    return [
        *secret_statements(ARENA_ORG_KEY_SECRET, arena.org_key),
        *secret_statements(ARENA_EXTERNAL_USER_SECRET, arena.external_user_id),
        *secret_statements(ARENA_BASE_URL_SECRET, arena.base_url),
        network_rule_sql(ARENA_API_RULE, (host,)),
        integration_sql(ARENA_API_INTEGRATION, ARENA_API_RULE, integrations),
    ]


def grant_statements(pools: list[str]) -> list[str]:
    """Grants for ``arena_trainer``. The image repository grant is READ only.

    Each pool gets USAGE and MONITOR. MODIFY and OPERATE stay with the admin.

    :param pools: Compute pool names.
    :return: The statements.
    """
    statements = [
        f"GRANT ROLE {ROLE} TO USER {SERVICE_USER}",
        f"GRANT USAGE ON WAREHOUSE {WAREHOUSE} TO ROLE {ROLE}",
        f"GRANT USAGE ON DATABASE {DATABASE} TO ROLE {ROLE}",
        f"GRANT USAGE ON SCHEMA {SCHEMA_NAME} TO ROLE {ROLE}",
        f"GRANT CREATE SERVICE ON SCHEMA {SCHEMA_NAME} TO ROLE {ROLE}",
        f"GRANT CREATE MODEL ON SCHEMA {SCHEMA_NAME} TO ROLE {ROLE}",
        f"GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE {ROLE}",
        f"GRANT READ, WRITE ON STAGE {SCHEMA_NAME}.{RUNS_STAGE} TO ROLE {ROLE}",
        (
            f"GRANT SELECT, INSERT, UPDATE ON TABLE {SCHEMA_NAME}.{METRICS_TABLE} "
            f"TO ROLE {ROLE}"
        ),
        f"GRANT USAGE ON FILE FORMAT {SCHEMA_NAME}.{FILE_FORMAT} TO ROLE {ROLE}",
        (
            f"GRANT READ ON IMAGE REPOSITORY {SCHEMA_NAME}.{IMAGE_REPOSITORY} "
            f"TO ROLE {ROLE}"
        ),
    ]
    for name in pools:
        statements.append(f"GRANT USAGE ON COMPUTE POOL {name} TO ROLE {ROLE}")
        statements.append(f"GRANT MONITOR ON COMPUTE POOL {name} TO ROLE {ROLE}")
    statements.append(f"GRANT USAGE ON INTEGRATION {HF_INTEGRATION} TO ROLE {ROLE}")
    statements.extend(
        f"GRANT READ ON SECRET {SCHEMA_NAME}.{secret} TO ROLE {ROLE}"
        for secret in (
            ARENA_ORG_KEY_SECRET,
            ARENA_EXTERNAL_USER_SECRET,
            ARENA_BASE_URL_SECRET,
        )
    )
    statements.append(
        f"GRANT USAGE ON INTEGRATION {ARENA_API_INTEGRATION} TO ROLE {ROLE}"
    )
    return statements
