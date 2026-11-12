# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""List pools, set a GPU cap, invite, join, grant, and check the account."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from agilerl.arena.snowflake.catalog import (
    NO_SETUP,
    PoolState,
    active_event_table,
    collect,
    event_table_warning,
    existing_integrations,
    existing_pools,
    linked_resources,
    load_families,
    object_named,
    one_line,
    remote_profile,
    require_accountadmin,
    schema_rows,
    secret_names,
    whole_number,
)
from agilerl.arena.snowflake.images import run_driver_check
from agilerl.arena.snowflake.job import TABLE_PART, lowered
from agilerl.arena.snowflake.profile import (
    ARENA_API_INTEGRATION,
    ARENA_BASE_URL_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_ORG_KEY_SECRET,
    DATABASE,
    ENV_INTEGRATION,
    ENV_RULE,
    FILE_FORMAT,
    GPU_FAMILIES,
    HF_INTEGRATION,
    IMAGE_REPOSITORY,
    METRICS_TABLE,
    QUALIFIED_METRICS,
    QUALIFIED_STAGE,
    ROLE,
    RUNS_STAGE,
    SCHEMA,
    WAREHOUSE,
    GpuPool,
    gpu_pool_name,
    profile_pool_names,
    put_setup_json,
    trainer_image_path,
)
from agilerl.arena.snowflake.setup import (
    family_listed,
    no_gpu_message,
    nodes_for_gpus,
    pool_family_message,
    unsupported_gpu_message,
)
from agilerl.arena.snowflake.setup_sql import (
    budget_add_sql,
    create_pool_sql,
    integration_sql,
    string_literal,
)
from agilerl.arena.snowflake.source import SnowflakeRow, SnowflakeSession

TERMINAL_SERVICE = frozenset({"DONE", "FAILED", "CANCELLED", "CANCELED", "SUSPENDED"})
TABLE_NAME = re.compile(rf"({TABLE_PART})\.({TABLE_PART})\.({TABLE_PART})")
HOST_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
HOST_NAME = re.compile(rf"{HOST_LABEL}(?:\.{HOST_LABEL})+")
REQUIRED_SECRETS = (
    ARENA_ORG_KEY_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_BASE_URL_SECRET,
)
NEXT_SUBMIT = "Next: arena snowflake submit MANIFEST.yaml"


@dataclass(frozen=True)
class RepoTag:
    """One ``snowflake-job`` tag in the account image repository."""

    name: str
    created_on: str


def list_pools(session: SnowflakeSession) -> str:
    """Each Arena pool, its family, its cap or CPU role, and its current nodes.

    :param session: Admin session.
    :return: One pool per line.
    :raises ValueError: Setup has not been run.
    """
    require_accountadmin(session)
    profile = remote_profile(session)
    if profile is None:
        raise ValueError(NO_SETUP)
    shown = existing_pools(session)
    rows = [
        (profile.pools.broker.pool, profile.pools.broker.family, "broker"),
        (profile.pools.head.pool, profile.pools.head.family, "head"),
        *(
            (item.pool, family, f"max {item.max_gpus} GPUs")
            for family, item in profile.pools.gpu.items()
        ),
    ]
    return "\n".join(
        format_pool_line(pool, family, capacity, shown, session)
        for pool, family, capacity in rows
    )


def format_pool_line(
    pool: str,
    family: str,
    capacity: str,
    shown: dict[str, PoolState],
    session: SnowflakeSession,
) -> str:
    """One pool line for ``arena snowflake pools``.

    :param pool: Pool name.
    :param family: Instance family.
    :param capacity: ``broker``, ``head``, or a max-GPU label.
    :param shown: Pools from ``SHOW COMPUTE POOLS``.
    :param session: Admin session, used when the show row has no node count.
    :return: The line.
    """
    state = shown.get(pool.upper())
    if state is None:
        return f"{pool}  {family}  {capacity}  not found"
    live_family = state.family or family
    count = active_node_count(session, state)
    noun = "node" if count == 1 else "nodes"
    return f"{pool}  {live_family}  {capacity}  {count} {noun}"


def active_node_count(session: SnowflakeSession, state: PoolState) -> int:
    """Nodes the pool is using now.

    ``SHOW COMPUTE POOLS`` supplies ``active_nodes``. A row without that column
    is read again with ``DESCRIBE COMPUTE POOL``.

    :param session: Admin session.
    :param state: Pool from ``SHOW``.
    :return: The node count.
    """
    if state.active_nodes is not None:
        return state.active_nodes
    rows = collect(session, f"DESCRIBE COMPUTE POOL {state.name}")
    return whole_number(lowered(rows[0]).get("active_nodes"))


def set_gpu_cap(session: SnowflakeSession, family: str, max_gpus: int) -> str:
    """Set one GPU pool's ``MAX_NODES`` and record the cap in ``setup.json``.

    A family that has no pool yet is created first. Snowflake enforces the
    account's node limit for the family.

    :param session: Admin session.
    :param family: GPU instance family.
    :param max_gpus: GPU cap.
    :return: The line to print.
    :raises ValueError: The family is unknown, or a running service blocks the change.
    """
    require_accountadmin(session)
    name = family.strip().upper()
    spec = GPU_FAMILIES.get(name)
    if spec is None:
        raise ValueError(unsupported_gpu_message(name))
    families = load_families(session)
    if not family_listed(name, families):
        raise ValueError(no_gpu_message(name, families))
    nodes = nodes_for_gpus(max_gpus, spec.gpus)
    profile = remote_profile(session)
    if profile is None:
        raise ValueError(NO_SETUP)
    pool = gpu_pool_name(name)
    state = existing_pools(session).get(pool.upper())
    if state is None:
        collect(session, create_pool_sql(pool, name, 1, nodes))
        for statement in pool_grant_statements(pool):
            collect(session, statement)
        if pool.lower() not in linked_resources(session):
            collect(session, budget_add_sql(pool))
    else:
        if state.family.upper() not in {"", name}:
            raise ValueError(pool_family_message(pool, state.family))
        blocked = blocking_services(session, pool, nodes)
        if blocked:
            raise ValueError(shrink_message(blocked, nodes))
        if state.max_nodes != nodes:
            collect(session, f"ALTER COMPUTE POOL {pool} SET MAX_NODES = {nodes}")
    gpu = dict(profile.pools.gpu)
    gpu[name] = GpuPool(pool, max_gpus, spec.gpus, spec.memory_gib)
    put_setup_json(session, replace(profile, pools=replace(profile.pools, gpu=gpu)))
    return f"{pool}  {name}  max {max_gpus} GPUs  MAX_NODES {nodes}"


def pool_grant_statements(pool: str) -> list[str]:
    """USAGE and MONITOR on one pool.

    :param pool: Compute pool name.
    :return: The statements.
    """
    return [
        f"GRANT USAGE ON COMPUTE POOL {pool} TO ROLE {ROLE}",
        f"GRANT MONITOR ON COMPUTE POOL {pool} TO ROLE {ROLE}",
    ]


def blocking_services(
    session: SnowflakeSession, pool: str, max_nodes: int
) -> list[tuple[str, int]]:
    """Running services on ``pool`` that use more nodes than ``max_nodes``.

    :param session: Admin session.
    :param pool: Compute pool name.
    :param max_nodes: Proposed ``MAX_NODES``.
    :return: Service name and the nodes it is using.
    """
    blocked = []
    for row in collect(session, f"SHOW SERVICES IN COMPUTE POOL {pool}"):
        values = lowered(row)
        status = str(values.get("status") or values.get("service_status") or "").upper()
        if status in TERMINAL_SERVICE:
            continue
        nodes = _service_nodes(values)
        if nodes > max_nodes:
            blocked.append((service_name(values), nodes))
    return blocked


def shrink_message(blocked: list[tuple[str, int]], max_nodes: int) -> str:
    """Message when a lower ``MAX_NODES`` would drop a running service.

    :param blocked: Service name and the nodes it is using.
    :param max_nodes: Proposed ``MAX_NODES``.
    :return: The message.
    """
    lines = [
        f"{name} is using {nodes} nodes. MAX_NODES {max_nodes} is below that."
        for name, nodes in blocked
    ]
    lines.append("The pool was not changed.")
    return "\n".join(lines)


def _service_nodes(values: dict[str, object]) -> int:
    for key in ("current_instances", "target_instances"):
        if values.get(key) not in (None, ""):
            return whole_number(values.get(key))
    return whole_number(values.get("min_instances"))


def service_name(values: dict[str, object]) -> str:
    """``database.schema.name`` of a ``SHOW SERVICES`` row.

    :param values: Lower-cased row.
    :return: The qualified name.
    """
    return f"{values['database_name']}.{values['schema_name']}.{values['name']}"


def quoted_identifier(name: str) -> str:
    """Double-quoted Snowflake identifier. Keeps case and any character.

    :param name: Stored object name.
    :return: The identifier.
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def invite(session: SnowflakeSession, user: str, account: str) -> str:
    """Grant ``arena_trainer`` to an existing user.

    :param session: Admin session.
    :param user: Snowflake user name, as shown by ``SHOW USERS``.
    :param account: Account identifier, for the join command.
    :return: The message to print.
    :raises ValueError: The user is missing, or the session is not ACCOUNTADMIN.
    """
    require_accountadmin(session)
    text = user.strip()
    rows = collect(session, f"SHOW USERS LIKE {string_literal(text)}")
    names = [str(lowered(row).get("name", "")) for row in rows]
    stored = next((name for name in names if name == text), None) or next(
        (name for name in names if name.upper() == text.upper()), None
    )
    if stored is None:
        raise ValueError(missing_user_message(text))
    collect(session, f"GRANT ROLE {ROLE} TO USER {quoted_identifier(stored)}")
    return invite_message(stored, account)


def invite_message(user: str, account: str) -> str:
    """Text printed after a grant.

    :param user: Snowflake user name.
    :param account: Account identifier.
    :return: The message.
    """
    return (
        f"Granted {ROLE} to {user}.\n"
        f"{user} runs these on their own machine:\n"
        '  pip install "agilerl-arena[snowflake]"\n'
        f"  arena snowflake join --account {account} --user <login name>\n"
        "join signs in with their Snowflake key pair or a programmatic access "
        "token. No Arena login is needed.\n"
        f"{NEXT_SUBMIT}"
    )


def missing_user_message(user: str) -> str:
    """Message when invite's user does not exist.

    :param user: Snowflake user name.
    :return: The message.
    """
    return (
        f"No user {user} in this account.\n"
        "Create that person in Snowsight, then run "
        f"arena snowflake invite {user} again."
    )


def join_account(session: SnowflakeSession, user: str) -> None:
    """Check that the user can use ``arena_trainer`` and read ``setup.json``.

    Creates nothing.

    :param session: The teammate's session.
    :param user: Snowflake user name, for the role message.
    :raises ValueError: The user lacks the role, or setup has not been run.
    """
    try:
        collect(session, f"USE ROLE {ROLE}")
    except Exception as exc:
        raise ValueError(join_role_message(user)) from exc
    if remote_profile(session) is None:
        raise ValueError(NO_SETUP)


def join_role_message(user: str) -> str:
    """Message when join's user does not have ``arena_trainer``.

    :param user: Snowflake user name.
    :return: The message.
    """
    return (
        f"{user} does not have the role {ROLE}.\n"
        "Ask an account admin to run:\n"
        f"  arena snowflake invite {user}\n"
        "The account is already set up."
    )


def table_parts(table: str) -> tuple[str, str, str]:
    """Database, schema, and table of a ``database.schema.table`` name.

    :param table: ``--table`` value. Parts may be double-quoted.
    :return: The three parts, as written.
    :raises ValueError: The name is not three dot-separated identifiers.
    """
    found = TABLE_NAME.fullmatch(table.strip())
    if found is None:
        msg = f"--table must be database.schema.table, got {table!r}"
        raise ValueError(msg)
    return found.group(1), found.group(2), found.group(3)


def grant_table(session: SnowflakeSession, table: str) -> str:
    """Let ``arena_trainer`` read one table.

    :param session: Admin session.
    :param table: ``database.schema.table``.
    :return: The line to print.
    :raises ValueError: The name is malformed or the session is not ACCOUNTADMIN.
    """
    database, schema, name = table_parts(table)
    require_accountadmin(session)
    for statement in (
        f"GRANT USAGE ON DATABASE {database} TO ROLE {ROLE}",
        f"GRANT USAGE ON SCHEMA {database}.{schema} TO ROLE {ROLE}",
        f"GRANT SELECT ON TABLE {database}.{schema}.{name} TO ROLE {ROLE}",
    ):
        collect(session, statement)
    return f"Granted SELECT on {database}.{schema}.{name} to {ROLE}."


def grant_host(session: SnowflakeSession, host: str) -> str:
    """Let jobs reach ``host`` on 443 through ``arena_env_eai``.

    :param session: Admin session.
    :param host: Host name, no scheme or port.
    :return: The line to print.
    :raises ValueError: The host is malformed, setup has not run, or the
        session is not ACCOUNTADMIN.
    """
    name = host.strip().lower()
    if HOST_NAME.fullmatch(name) is None:
        msg = f"--host must be a host name such as api.example.com, got {host!r}"
        raise ValueError(msg)
    require_accountadmin(session)
    profile = remote_profile(session)
    if profile is None:
        raise ValueError(NO_SETUP)
    rule = f"{DATABASE}.{SCHEMA}.{ENV_RULE}"
    value = string_literal(f"{name}:443")
    rules = schema_rows(
        session, f"SHOW NETWORK RULES LIKE '{ENV_RULE}' IN SCHEMA {DATABASE}.{SCHEMA}"
    )
    if object_named(rules, ENV_RULE):
        collect(session, f"ALTER NETWORK RULE {rule} ADD VALUE_LIST = ({value})")
    else:
        collect(
            session,
            f"CREATE NETWORK RULE {rule}\n"
            "  TYPE = HOST_PORT\n"
            "  MODE = EGRESS\n"
            f"  VALUE_LIST = ({value})",
        )
    integrations = frozenset(existing_integrations(session))
    collect(session, integration_sql(ENV_INTEGRATION, ENV_RULE, integrations))
    collect(session, f"GRANT USAGE ON INTEGRATION {ENV_INTEGRATION} TO ROLE {ROLE}")
    attached = tuple(
        dict.fromkeys((*profile.external_access_integrations, ENV_INTEGRATION))
    )
    put_setup_json(session, replace(profile, external_access_integrations=attached))
    return f"Jobs can reach {name}:443 through {ENV_INTEGRATION}."


def run_check(session: SnowflakeSession) -> str:
    """Read the account as ``arena_trainer``, save the image, and check the driver and Arena from the pool.

    :param session: ``arena`` connection.
    :return: The text to print. It ends with a warning when no event table is active.
    :raises ValueError: A required object is missing, or no image was pushed.
    """
    try:
        collect(session, f"USE ROLE {ROLE}")
    except Exception as exc:
        msg = f"The role {ROLE} cannot be used.\n{one_line(exc)}\nRe-run arena snowflake setup."
        raise ValueError(msg) from exc
    missing = missing_objects(session)
    if missing:
        raise ValueError("\n".join([*missing, "Re-run arena snowflake setup."]))
    images = collect(session, "SHOW IMAGES IN IMAGE REPOSITORY arena.training.images")
    tags = snowflake_job_tags(images)
    if not tags:
        msg = (
            "No snowflake-job image is in arena.training.images yet.\n"
            "Re-run arena snowflake setup. It copies the image."
        )
        raise ValueError(msg)
    base = remote_profile(session)
    if base is None:
        raise ValueError(NO_SETUP)
    pools = collect(session, "SHOW COMPUTE POOLS LIKE 'arena_%'")
    for name in profile_pool_names(base):
        if not object_named(pools, name):
            msg = f"Compute pool {name} was not found.\nRe-run arena snowflake setup."
            raise ValueError(msg)
    chosen = choose_tag(tags, base.image)
    updated = replace(base, image=trainer_image_path(chosen))
    put_setup_json(session, updated)
    gpu_pool = next(iter(updated.pools.gpu.values())).pool
    driver = run_driver_check(session, gpu_pool, updated.image)
    lines = [check_report(tags, updated.image), f"driver {driver} on {gpu_pool}"]
    warning = event_table_warning(active_event_table(session))
    if warning is not None:
        lines.append(warning)
    return "\n".join(lines)


def missing_objects(session: SnowflakeSession) -> list[str]:
    """Lines naming each Arena object ``arena_trainer`` cannot see.

    :param session: Session using ``arena_trainer``.
    :return: One line per missing or disabled object. Empty when all are there.
    """
    schema = f"{DATABASE}.{SCHEMA}"
    shown = {
        "warehouse": collect(session, f"SHOW WAREHOUSES LIKE '{WAREHOUSE}'"),
        "stage": schema_rows(
            session, f"SHOW STAGES LIKE '{RUNS_STAGE}' IN SCHEMA {schema}"
        ),
        "table": schema_rows(
            session, f"SHOW TABLES LIKE '{METRICS_TABLE.upper()}' IN SCHEMA {schema}"
        ),
        "format": schema_rows(
            session, f"SHOW FILE FORMATS LIKE '{FILE_FORMAT}' IN SCHEMA {schema}"
        ),
        "repository": schema_rows(
            session,
            f"SHOW IMAGE REPOSITORIES LIKE '{IMAGE_REPOSITORY}' IN SCHEMA {schema}",
        ),
    }
    expected = (
        ("warehouse", WAREHOUSE, f"Warehouse {WAREHOUSE}"),
        ("stage", RUNS_STAGE, f"Stage {QUALIFIED_STAGE}"),
        ("table", METRICS_TABLE, f"Table {QUALIFIED_METRICS}"),
        ("format", FILE_FORMAT, f"File format {schema}.{FILE_FORMAT}"),
        (
            "repository",
            IMAGE_REPOSITORY,
            f"Image repository {schema}.{IMAGE_REPOSITORY}",
        ),
    )
    lines = [
        f"{label} was not found."
        for key, name, label in expected
        if not object_named(shown[key], name)
    ]
    secrets = secret_names(session)
    lines.extend(
        f"Secret {schema}.{name} was not found."
        for name in REQUIRED_SECRETS
        if name.upper() not in secrets
    )
    integrations = existing_integrations(session)
    lines.extend(
        f"External access integration {name} is missing or not enabled."
        for name in (HF_INTEGRATION, ARENA_API_INTEGRATION)
        if not integrations.get(name.upper(), False)
    )
    return lines


def snowflake_job_tags(rows: list[SnowflakeRow]) -> list[RepoTag]:
    """``snowflake-job`` tags from ``SHOW IMAGES``.

    :param rows: Image rows.
    :return: Tags, in the order the rows returned them.
    """
    found: list[RepoTag] = []
    for row in rows:
        values = lowered(row)
        if str(values.get("image_name") or "").split("/")[-1] != "snowflake-job":
            continue
        created = str(values.get("created_on") or "")
        for tag in str(values.get("tags") or "").split(","):
            name = tag.strip()
            if name:
                found.append(RepoTag(name, created))
    return found


def choose_tag(tags: list[RepoTag], current: str) -> str:
    """Keep the saved tag when it is still present. Otherwise take the newest.

    :param tags: Tags in the repository.
    :param current: Image path already saved, or ``""``.
    :return: The tag to save.
    """
    names = [tag.name for tag in tags]
    current_tag = current.rsplit(":", 1)[-1] if current else ""
    if current_tag in names:
        return current_tag
    return max(tags, key=lambda tag: tag.created_on).name


def check_report(tags: list[RepoTag], image: str) -> str:
    """Text printed when check saves an image.

    :param tags: Tags that were present.
    :param image: Image path that was saved.
    :return: The message.
    """
    lines = []
    if len(tags) > 1:
        lines.append("snowflake-job tags:")
        lines.extend(f"  {tag.name}" for tag in tags)
    lines.append(f"image  {image}")
    return "\n".join(lines)
