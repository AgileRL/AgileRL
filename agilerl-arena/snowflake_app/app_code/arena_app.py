# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stored procedures of the Arena Snowflake Native App: setup, runs, and published models.

The setup script registers each handler with ``IMPORTS`` of this file and
``agilerl_arena.zip``, so these run as owner's-rights procedures inside the app.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from snowflake.snowpark.exceptions import SnowparkSQLException

from agilerl.arena.snowflake.account import blocking_services, shrink_message
from agilerl.arena.snowflake.billing import reconcile
from agilerl.arena.snowflake.catalog import (
    PoolState,
    collect,
    existing_pools,
    load_families,
    remote_profile,
)
from agilerl.arena.snowflake.cli import run_status, service_logs
from agilerl.arena.snowflake.job import (
    ROLES,
    SubmitSettings,
    cancel_run,
    choose_gpu_family,
    lowered,
    manifest_fits,
    new_run_id,
    runs_prefix,
    submit_job,
    write_job_manifest,
)
from agilerl.arena.snowflake.license import (
    NATIVE_APP_CHANNEL,
    SNOWFLAKE_DENIED,
    DeploymentClosedError,
    DeploymentRefusedError,
    SnowflakeDeployment,
    SnowflakeLicenseError,
    SnowflakeUsage,
    TrainingDisabledError,
    UnknownUsageError,
    arena_api_base_url,
    arena_api_hostname,
    close_snowflake_deployment,
    fetch_capabilities,
    heartbeat_snowflake_deployment,
    open_snowflake_deployment,
    partner_headers,
    snowflake_priced_families,
    snowflake_training_enabled,
    snowflake_usage,
)
from agilerl.arena.snowflake.profile import (
    ARENA_API_RULE,
    ARENA_BASE_URL_SECRET,
    ARENA_EXTERNAL_USER_SECRET,
    ARENA_ORG_KEY_SECRET,
    CPU_POOL_MAX_NODES,
    CPU_POOL_MIN_NODES,
    DEFAULT_GPU_NODES,
    FILE_FORMAT,
    GPU_FAMILIES,
    METRICS_TABLE,
    RUNS_STAGE,
    AccountPools,
    AccountProfile,
    CpuPool,
    GpuPool,
    put_setup_json,
    sole_gpu_pool,
)
from agilerl.arena.snowflake.publish import (
    PublishDocument,
    PublishSession,
    PublishSettings,
    registry_schema,
    served_version,
    service_endpoint,
    submit_publish,
    version_endpoint,
    wait_for_publish,
)
from agilerl.arena.snowflake.publish import unpublish as drop_served
from agilerl.arena.snowflake.setup import (
    ORG_KEY_PREFIX,
    family_listed,
    missing_family_message,
    nodes_for_gpus,
    unsupported_gpu_message,
)
from agilerl.arena.snowflake.setup_sql import (
    PoolPlan,
    pool_statements,
    secret_statements,
    string_literal,
)
from agilerl.arena.snowflake.source import QuerySource, SnowflakeSession

if TYPE_CHECKING:
    from collections.abc import Callable

APP_SCHEMA = "app_state"
SOURCE_REFERENCES = ("SOURCE_TABLE", "SOURCE_VIEW")
# snowflake-app-build replaces the tag with the release tag, as in manifest.yml.
IMAGE = "/arena_provider/images/repo/snowflake-job:0.0.0-py313-cu130"
SETUP_PREFIX = f"@{APP_SCHEMA}.{RUNS_STAGE}/_arena"

logger = logging.getLogger(__name__)

# Snowflake writes procedure logs to the event table from the root logger; the
# CLI's console handler would keep agilerl.arena records on stderr instead.
ARENA_LOGGER = logging.getLogger("agilerl.arena")
ARENA_LOGGER.handlers.clear()
ARENA_LOGGER.propagate = True

# Snowflake returns a procedure's formatted exception to the caller; without this
# the consumer sees the app's stack and file paths around the message.
sys.tracebacklimit = 0


@dataclass(frozen=True)
class AppNames:
    """Account-level names of this app's objects, ``<prefix><suffix>``.

    The setup script creates the warehouse and integrations with the same
    prefix and suffixes. The ``AGILERL_APP_`` start of the prefix keeps them
    apart from the CLI's ``arena_`` objects.
    """

    warehouse: str
    broker_pool: str
    head_pool: str
    gpu_pool_prefix: str
    hf_integration: str
    api_integration: str


def app_names(session: SnowflakeSession) -> AppNames:
    """Names from ``app_code.object_prefix()``, which derives them from the app name.

    :param session: Procedure session.
    :return: The names.
    """
    [row] = collect(session, "SELECT app_code.object_prefix() AS PREFIX")
    prefix = str(lowered(row)["prefix"])
    return AppNames(
        warehouse=f"{prefix}WH",
        broker_pool=f"{prefix}BROKER_POOL",
        head_pool=f"{prefix}HEAD_POOL",
        gpu_pool_prefix=f"{prefix}GPU_",
        hf_integration=f"{prefix}HF_EAI",
        api_integration=f"{prefix}ARENA_EAI",
    )


def configure(
    session: SnowflakeSession, org_key: str, external_user_id: str, base_url: str
) -> str:
    """Check the org key with Arena, then save the three Arena secrets in ``app_state``.

    No secret value is logged, returned, or quoted in an error.

    :param session: Procedure session.
    :param org_key: Arena organisation key.
    :param external_user_id: External user id the key acts for.
    :param base_url: Arena origin. Its host must be in the app's Arena network rule.
    :return: Confirmation.
    :raises ValueError: The key is not an org key, the user id is empty, or the
        base URL host is not in the Arena network rule.
    :raises SnowflakeLicenseError: Arena rejected the credentials, could not be
        reached, or does not allow Snowflake training for the organisation.
    :raises RuntimeError: Snowflake did not save a secret.
    """
    org_key = (org_key or "").strip()
    external_user_id = (external_user_id or "").strip()
    if not org_key.startswith(ORG_KEY_PREFIX):
        msg = (
            "That is not an Arena org key.\n"
            f"Ask AgileRL for a Snowflake org key (it starts with {ORG_KEY_PREFIX}).\n"
            "Nothing was saved."
        )
        raise ValueError(msg)
    if external_user_id == "":
        msg = "Pass the Arena external user id.\nNothing was saved."
        raise ValueError(msg)
    origin = arena_origin(session, base_url)
    caps = fetch_capabilities(origin, partner_headers(org_key, external_user_id))
    if not snowflake_training_enabled(caps):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)
    for name, value in (
        (ARENA_ORG_KEY_SECRET, org_key),
        (ARENA_EXTERNAL_USER_SECRET, external_user_id),
        (ARENA_BASE_URL_SECRET, origin),
    ):
        save_secret(session, name, value)
    priced = snowflake_priced_families(caps)
    lines = [
        "Saved the Arena org key, external user id, and base URL.",
        "Arena allows Snowflake training for this organisation.",
    ]
    if priced is not None:
        lines.append(f"Priced GPU families: {', '.join(sorted(priced)) or 'none'}.")
    return "\n".join(lines)


def arena_origin(session: SnowflakeSession, base_url: str) -> str:
    """``https://<host>`` for ``base_url``, refused when the Arena rule does not allow the host.

    :param session: Procedure session.
    :param base_url: Arena origin as the admin typed it.
    :return: The origin.
    :raises ValueError: The value does not name one host, or the host is not in the rule.
    """
    try:
        host = arena_api_hostname(base_url or "")
    except ValueError:
        # The library message quotes the value, which may be a pasted secret.
        msg = "The base URL does not name one host.\nNothing was saved."
        raise ValueError(msg) from None
    allowed = rule_hosts(session, f"{APP_SCHEMA}.{ARENA_API_RULE}")
    if f"{host.lower()}:443" not in allowed:
        msg = (
            "The base URL host is not in the app's Arena network rule "
            f"({', '.join(sorted(allowed))}), so the app cannot reach it.\n"
            "Nothing was saved."
        )
        raise ValueError(msg)
    return arena_api_base_url(host)


def rule_hosts(session: SnowflakeSession, rule: str) -> set[str]:
    """Lower-case ``host:port`` values of a network rule.

    :param session: Procedure session.
    :param rule: Qualified network rule name.
    :return: The values.
    """
    hosts: set[str] = set()
    for row in collect(session, f"DESCRIBE NETWORK RULE {rule}"):
        raw = str(lowered(row).get("value_list") or "")
        hosts.update(item.strip().lower() for item in raw.split(",") if item.strip())
    return hosts


def save_secret(session: SnowflakeSession, name: str, value: str) -> None:
    """Create the secret when missing, then set its value.

    :param session: Procedure session.
    :param name: Secret name in ``app_state``.
    :param value: Secret string.
    :raises RuntimeError: Snowflake rejected a statement. The error names the
        secret, not its value.
    """
    try:
        for statement in secret_statements(APP_SCHEMA, name, value):
            collect(session, statement)
    except SnowparkSQLException as exc:
        # A Snowflake error can quote the statement, which holds the value.
        msg = (
            f"Snowflake did not save {APP_SCHEMA}.{name} "
            f"(Snowflake error {exc.sql_error_code})."
        )
        raise RuntimeError(msg) from None


def register_reference(
    session: SnowflakeSession, ref_name: str, operation: str, ref_or_alias: str
) -> None:
    """Bind, unbind, or clear a consumer table or view on a source reference.

    Snowflake calls this when the consumer changes a ``source_table`` or
    ``source_view`` binding. Both references are multi-valued.

    :param session: Procedure session.
    :param ref_name: Reference name from the manifest.
    :param operation: ``ADD``, ``REMOVE``, or ``CLEAR``.
    :param ref_or_alias: Reference to add, or the alias to remove.
    :raises ValueError: The reference or operation is unknown.
    """
    name = (ref_name or "").upper()
    if name not in SOURCE_REFERENCES:
        msg = f"Unknown reference {ref_name}. Bind source_table or source_view."
        raise ValueError(msg)
    reference = string_literal(name)
    match (operation or "").upper():
        case "ADD":
            statement = (
                f"SELECT SYSTEM$ADD_REFERENCE({reference}, "
                f"{string_literal(ref_or_alias)})"
            )
        case "REMOVE":
            statement = (
                f"SELECT SYSTEM$REMOVE_REFERENCE({reference}, "
                f"{string_literal(ref_or_alias)})"
            )
        case "CLEAR":
            statement = f"SELECT SYSTEM$REMOVE_ALL_REFERENCES({reference})"
        case _:
            msg = f"Unknown operation {operation}. Use ADD, REMOVE, or CLEAR."
            raise ValueError(msg)
    collect(session, statement)


def set_pools(
    session: SnowflakeSession,
    gpu_families: list[str],
    max_gpus: int | None,
    broker_family: str | None,
    head_family: str | None,
) -> str:
    """Create or resize the broker, head, and GPU pools, and record them in ``setup.json``.

    App GPU pools for families not in ``gpu_families`` are dropped. Every check
    runs before any pool changes.

    :param session: Procedure session.
    :param gpu_families: GPU instance families to keep, one pool each.
    :param max_gpus: GPU cap for each listed family. ``None`` keeps a saved cap,
        or gives a new family two nodes.
    :param broker_family: Instance family of the Pulsar broker pool, or
        ``None`` for ``CPU_X64_L``.
    :param head_family: Instance family of the Ray head pool, or ``None`` for
        ``CPU_X64_SL``.
    :return: One line per pool.
    :raises ValueError: A family is unknown or not in this region, an existing
        pool uses another family, or a running service blocks the change.
    """
    names = gpu_names(gpu_families, max_gpus)
    # An explicit NULL from SQL skips the procedure's DEFAULT, so default here too.
    broker = (broker_family or "CPU_X64_L").strip().upper()
    head = (head_family or "CPU_X64_SL").strip().upper()
    families = load_families(session)
    for family, role in (
        (broker, "broker"),
        (head, "head"),
        *((n, "GPU") for n in names),
    ):
        if not family_listed(family, families):
            raise ValueError(missing_family_message(family, role))
    saved = remote_profile(
        session, f"{SETUP_PREFIX}/setup.json", f"{APP_SCHEMA}.{FILE_FORMAT}"
    )
    objects = app_names(session)
    plans = pool_plans(
        objects,
        broker,
        head,
        names,
        max_gpus,
        saved.pools.gpu if saved is not None else {},
    )
    existing = existing_pools(session)
    leftovers = checked_leftovers(session, objects, plans, existing)
    for plan in plans:
        state = existing.get(plan.name.upper())
        for statement in pool_statements(state.max_nodes if state else None, plan):
            collect(session, statement)
    for name in leftovers:
        collect(session, f"ALTER COMPUTE POOL {name} STOP ALL")
        collect(session, f"DROP COMPUTE POOL {name}")
    put_setup_json(session, app_profile(objects, plans, saved), SETUP_PREFIX)
    lines = [
        f"{plan.name}  {plan.family}  {plan.role}"
        if plan.max_gpus is None
        else f"{plan.name}  {plan.family}  max {plan.max_gpus} GPUs  MAX_NODES {plan.max_nodes}"
        for plan in plans
    ]
    lines.extend(f"{name}  dropped" for name in leftovers)
    return "\n".join(lines)


def gpu_names(gpu_families: list[str], max_gpus: int | None) -> list[str]:
    """Upper-case GPU families in order, without repeats.

    :param gpu_families: GPU instance families from the caller.
    :param max_gpus: GPU cap for each family, or ``None``.
    :return: The families.
    :raises ValueError: No family is named, a family is unknown, or
        ``max_gpus`` is below 1.
    """
    names = list(dict.fromkeys(str(item).strip().upper() for item in gpu_families))
    if not names:
        msg = "Name at least one GPU family.\nNo pool was changed."
        raise ValueError(msg)
    for name in names:
        if name not in GPU_FAMILIES:
            raise ValueError(unsupported_gpu_message(name))
    if max_gpus is not None and max_gpus < 1:
        msg = "max_gpus must be at least 1.\nNo pool was changed."
        raise ValueError(msg)
    return names


def pool_plans(
    objects: AppNames,
    broker: str,
    head: str,
    names: list[str],
    max_gpus: int | None,
    saved_gpu: dict[str, GpuPool],
) -> list[PoolPlan]:
    """Broker, head, then one GPU pool per family.

    :param objects: The app's account-level names.
    :param broker: Instance family of the broker pool.
    :param head: Instance family of the head pool.
    :param names: GPU families.
    :param max_gpus: GPU cap for each family. ``None`` keeps a saved cap, or
        gives a new family two nodes.
    :param saved_gpu: GPU pools already in ``setup.json``, keyed by family.
    :return: The plans.
    """
    plans = [
        PoolPlan(
            objects.broker_pool,
            broker,
            "broker",
            CPU_POOL_MIN_NODES,
            CPU_POOL_MAX_NODES,
            None,
        ),
        PoolPlan(
            objects.head_pool,
            head,
            "head",
            CPU_POOL_MIN_NODES,
            CPU_POOL_MAX_NODES,
            None,
        ),
    ]
    for name in names:
        spec = GPU_FAMILIES[name]
        if max_gpus is not None:
            cap = max_gpus
        elif name in saved_gpu:
            cap = saved_gpu[name].max_gpus
        else:
            cap = spec.gpus * DEFAULT_GPU_NODES
        plans.append(
            PoolPlan(
                f"{objects.gpu_pool_prefix}{name}",
                name,
                "gpu",
                1,
                nodes_for_gpus(cap, spec.gpus),
                cap,
            )
        )
    return plans


def checked_leftovers(
    session: SnowflakeSession,
    objects: AppNames,
    plans: list[PoolPlan],
    existing: dict[str, PoolState],
) -> list[str]:
    """App GPU pools no plan keeps, once every plan and drop is known to be safe.

    :param session: Procedure session.
    :param objects: The app's account-level names.
    :param plans: Pools to create or resize.
    :param existing: Compute pools in the account, keyed by upper-case name.
    :return: Upper-case names of the pools to drop.
    :raises ValueError: An existing pool uses another family, or a running
        service blocks a resize or drop.
    """
    for plan in plans:
        state = existing.get(plan.name.upper())
        if state is None:
            continue
        if state.family.upper() not in {"", plan.family}:
            msg = (
                f"Pool {plan.name} is {state.family}.\n"
                "The app will not replace a pool. Drop it when nothing is running, "
                "then call set_pools again."
            )
            raise ValueError(msg)
        blocked = blocking_services(session, plan.name, plan.max_nodes)
        if blocked:
            raise ValueError(shrink_message(blocked, plan.max_nodes))
    kept = {plan.name.upper() for plan in plans}
    leftovers = [
        name
        for name in existing
        if name.startswith(objects.gpu_pool_prefix) and name not in kept
    ]
    for name in leftovers:
        blocked = blocking_services(session, name, 0)
        if blocked:
            msg = f"{name} is no longer listed, so it would be dropped.\n"
            raise ValueError(msg + shrink_message(blocked, 0))
    return leftovers


def app_profile(
    objects: AppNames, plans: list[PoolPlan], saved: AccountProfile | None
) -> AccountProfile:
    """``setup.json`` fields for the app, in the shape the CLI writes.

    :param objects: The app's account-level names.
    :param plans: Broker, head, then GPU pools.
    :param saved: Profile already saved. Its extra integrations stay.
    :return: The profile.
    """
    broker, head, *gpu_plans = plans
    extra = saved.external_access_integrations if saved is not None else ()
    return AccountProfile(
        image=IMAGE,
        warehouse=objects.warehouse,
        stage=f"{APP_SCHEMA}.{RUNS_STAGE}",
        metrics_table=f"{APP_SCHEMA}.{METRICS_TABLE}",
        external_access_integrations=tuple(
            dict.fromkeys((objects.hf_integration, *extra))
        ),
        pools=AccountPools(
            broker=CpuPool(broker.name, broker.family),
            head=CpuPool(head.name, head.family),
            gpu={
                plan.family: GpuPool(
                    plan.name,
                    int(plan.max_gpus or 0),
                    GPU_FAMILIES[plan.family].gpus,
                    GPU_FAMILIES[plan.family].memory_gib,
                )
                for plan in gpu_plans
            },
        ),
    )


def submit(
    session: SnowflakeSession,
    manifest: str,
    gpus: int,
    source_reference: str | None,
    source_alias: str | None,
    gpu_family: str | None,
    reward_file: str | None,
    checkpoint_steps: int | None,
    resume_from: str | None,
) -> str:
    """Start a run's broker, head, and worker job services on the app's pools.

    :param session: Procedure session.
    :param manifest: Training manifest YAML.
    :param gpus: GPUs for the run.
    :param source_reference: ``source_table`` or ``source_view``, or ``None``
        when the manifest trains on no rows.
    :param source_alias: Alias of the bound object the run reads.
    :param gpu_family: GPU family. ``None`` picks one as ``arena snowflake submit`` does.
    :param reward_file: Python source of the reward function, or ``None``.
    :param checkpoint_steps: Training steps between checkpoints, or ``None``.
    :param resume_from: Run whose newest finished checkpoint the run starts
        from, or ``None``.
    :return: The run id, the service names, and any manifest warnings.
    :raises ValueError: An argument is out of range, the source is not a bound
        reference, the pools are not set, Arena has no price for the family,
        or the manifest cannot run on them.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    if gpus < 1:
        msg = "gpus must be at least 1.\nNothing was started."
        raise ValueError(msg)
    if checkpoint_steps is not None and checkpoint_steps < 1:
        msg = "checkpoint_steps must be at least 1.\nNothing was started."
        raise ValueError(msg)
    source = reference_source(source_reference, source_alias)
    profile = app_setup(session)
    pools = priced_pools(profile.pools.gpu, gpu_family)
    with (
        tempfile.TemporaryDirectory() as tmp,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        directory = Path(tmp)
        submitted = directory / "submitted.yaml"
        submitted.write_text(manifest, encoding="utf-8")
        reward_path = None
        if reward_file is not None:
            reward_path = directory / "reward.py"
            reward_path.write_text(reward_file, encoding="utf-8")
        job_manifest = write_job_manifest(
            submitted, source, reward_path is not None, directory
        )
        family = choose_gpu_family(
            pools,
            gpus,
            gpu_family,
            lambda pool: manifest_fits(job_manifest, pool, gpus),
        )
        settings = submit_settings(
            profile,
            app_names(session).api_integration,
            source,
            gpus,
            family,
            checkpoint_steps,
            reward_path,
            resume_from,
        )
        run_id = new_run_id()
        # Recorded before any service starts, so a started run is always billed.
        collect(
            session,
            "INSERT INTO app_state.usage_items (kind, usage_id, opened_at) "
            f"VALUES ('run', {string_literal(run_id)}, CURRENT_TIMESTAMP())",
        )
        _, services = submit_job(session, job_manifest, settings, run_id)
    lines = [
        f"run_id: {run_id}",
        f"broker: {services.broker}",
        f"head: {services.head}",
        f"workers: {services.workers}",
    ]
    lines.extend(
        f"warning: {item.message}"
        for item in caught
        if issubclass(item.category, UserWarning)
    )
    return "\n".join(lines)


def priced_pools(
    pools: dict[str, GpuPool], gpu_family: str | None
) -> dict[str, GpuPool]:
    """Keep the app's GPU pools whose family Arena has a Snowflake price for.

    Arena refuses an unpriced family only when the head opens the run, after
    the GPU nodes have started, so ``submit`` checks first.

    :param pools: GPU pools in ``setup.json``, keyed by family.
    :param gpu_family: Family the caller named, or ``None``.
    :return: The priced pools, or every pool when Arena does not list prices.
    :raises ValueError: Arena is not configured, the named family has no
        price, or no pool's family has one.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    base_url, headers = arena_partner()
    priced = snowflake_priced_families(fetch_capabilities(base_url, headers))
    if priced is None:
        return pools
    listed = ", ".join(sorted(priced)) or "none"
    if gpu_family is not None and gpu_family not in priced:
        msg = (
            f"Arena has no Snowflake price for {gpu_family}, so it would refuse "
            f"the run after the GPUs start.\nPriced families: {listed}.\n"
            "Nothing was started."
        )
        raise ValueError(msg)
    kept = {family: pool for family, pool in pools.items() if family in priced}
    if not kept:
        msg = (
            f"Arena has no Snowflake price for any of the app's GPU pools "
            f"({', '.join(sorted(pools))}).\nPriced families: {listed}. "
            "An app_admin calls set_pools with one of them.\nNothing was started."
        )
        raise ValueError(msg)
    return kept


def submit_settings(
    profile: AccountProfile,
    api_integration: str,
    source: QuerySource | None,
    gpus: int,
    family: str,
    checkpoint_steps: int | None,
    reward_path: Path | None,
    resume_from: str | None,
) -> SubmitSettings:
    """Build the settings of a run on the app's pools, secrets, and integrations.

    :param profile: The app's ``setup.json``.
    :param api_integration: The app's Arena integration.
    :param source: Query the run reads, or ``None``.
    :param gpus: GPUs for the run.
    :param family: GPU family of the run's pool.
    :param checkpoint_steps: Training steps between checkpoints, or ``None``.
    :param reward_path: Reward function file, or ``None``.
    :param resume_from: Run to resume from, or ``None``.
    :return: The settings.
    """
    pool = profile.pools.gpu[family]
    return SubmitSettings(
        image=profile.image,
        warehouse=profile.warehouse,
        runs_stage=profile.stage,
        metrics_table=profile.metrics_table,
        source=source,
        external_access_integrations=profile.external_access_integrations,
        checkpoint_steps=checkpoint_steps,
        reward_file=reward_path,
        gpus=gpus,
        gpu_family=family,
        gpus_per_node=pool.gpus_per_node,
        gpu_pool=pool.pool,
        broker_pool=profile.pools.broker.pool,
        broker_family=profile.pools.broker.family,
        head_pool=profile.pools.head.pool,
        head_family=profile.pools.head.family,
        resume_from=resume_from,
        secrets_schema=APP_SCHEMA,
        arena_api_integration=api_integration,
        service_schema=APP_SCHEMA,
        channel=NATIVE_APP_CHANNEL,
    )


def reference_source(reference: str | None, alias: str | None) -> QuerySource | None:
    """Query that reads one object bound to a source reference.

    :param reference: ``source_table`` or ``source_view``, or ``None``.
    :param alias: Alias of the bound object, or ``None``.
    :return: The query, or ``None`` when neither is given.
    :raises ValueError: Only one is given, or the reference is unknown.
    """
    if reference is None and alias is None:
        return None
    name = (reference or "").upper()
    if name not in SOURCE_REFERENCES or not alias:
        msg = (
            "Pass source_reference (source_table or source_view) together with "
            "the alias of an object bound to it.\nNothing was started."
        )
        raise ValueError(msg)
    return QuerySource(
        f"SELECT * FROM reference({string_literal(name)}, {string_literal(alias)})"
    )


def app_registry(session: SnowflakeSession) -> tuple[str, str]:
    """Database and schema of the app's Model Registry: the app and ``app_state``.

    :param session: Procedure session.
    :return: Both names.
    """
    database, _ = registry_schema(session, None, None)
    return database, APP_SCHEMA


def app_setup(session: SnowflakeSession) -> AccountProfile:
    """``setup.json`` that ``set_pools`` wrote.

    :param session: Procedure session.
    :return: The profile.
    :raises ValueError: ``set_pools`` has not run.
    """
    profile = remote_profile(
        session, f"{SETUP_PREFIX}/setup.json", f"{APP_SCHEMA}.{FILE_FORMAT}"
    )
    if profile is None:
        msg = "The app has no compute pools yet.\nAn app_admin calls app_code.set_pools first."
        raise ValueError(msg)
    return profile


def known_run(session: SnowflakeSession, run_id: str) -> None:
    """Refuse a run id that has no ``job.json`` on the app's runs stage.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :raises ValueError: The id is malformed, or the app has no such run.
    """
    prefix = runs_prefix(f"{APP_SCHEMA}.{RUNS_STAGE}", run_id)
    if not collect(session, f"LIST {prefix}/job.json"):
        msg = f"This app has no run {run_id}."
        raise ValueError(msg)


def status(session: SnowflakeSession, run_id: str) -> str:
    """State of a run's broker, head, and workers, their containers, and the run.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :return: One line per role and container, then ``state: <run state>``.
    :raises ValueError: The app has no such run.
    """
    known_run(session, run_id)
    return run_status(session, run_id, APP_SCHEMA)


def logs(session: SnowflakeSession, run_id: str, role: str, instance: int) -> str:
    """Read the main container logs of one role of a run.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :param role: ``broker``, ``head``, or ``worker``.
    :param instance: Instance id. ``0`` is the first instance.
    :return: The log text.
    :raises ValueError: The role or instance is invalid, the app has no such
        run, or the service was dropped.
    """
    if role not in ROLES:
        msg = f"Unknown role {role}. Use broker, head, or worker."
        raise ValueError(msg)
    if instance < 0:
        msg = "instance must be 0 or more."
        raise ValueError(msg)
    known_run(session, run_id)
    return service_logs(session, run_id, instance, role, APP_SCHEMA)


def cancel(session: SnowflakeSession, run_id: str) -> str:
    """Drop a run's broker, head, and workers.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :return: Confirmation.
    :raises ValueError: The app has no such run.
    """
    known_run(session, run_id)
    cancel_run(session, run_id, APP_SCHEMA)
    return f"cancelled: {run_id}"


def arena_partner() -> tuple[str, dict[str, str]]:
    """Arena origin and partner headers from the secrets ``configure`` saved.

    :return: The origin and the headers.
    :raises ValueError: No org key has been saved.
    """
    # Lazy import of an optional package: _snowflake exists only inside a Snowflake procedure.
    import _snowflake  # noqa: PLC0415 -- not installable outside Snowflake

    org_key = _snowflake.get_generic_secret_string("org_key")
    if not org_key:
        msg = "Arena is not configured. An app_admin calls app_code.configure first."
        raise ValueError(msg)
    external_user_id = _snowflake.get_generic_secret_string("external_user_id")
    base_url = _snowflake.get_generic_secret_string("base_url")
    return base_url, partner_headers(org_key, external_user_id)


def reconcile_usage(session: SnowflakeSession) -> str:
    """Meter served models, bill every unsettled run and deployment, and log the summary at INFO.

    :param session: Procedure session.
    :return: The :func:`meter_deployments` and :func:`reconcile` summaries,
        or why nothing ran.
    """
    summary = reconcile_summary(session)
    logger.info("reconcile_usage: %s", summary)
    return summary


def reconcile_summary(session: SnowflakeSession) -> str:
    """Meter served models, then bill every unsettled run and deployment.

    :param session: Procedure session.
    :return: The :func:`meter_deployments` and :func:`reconcile` summaries,
        or why nothing ran.
    """
    try:
        base_url, headers = arena_partner()
    except ValueError as exc:
        return str(exc)

    # One task run covers every item, so each request retries only briefly.
    def fetch(kind: str, usage_id: str) -> SnowflakeUsage:
        return snowflake_usage(kind, usage_id, base_url, headers, time.sleep, 30.0)

    def open_call(deployment: SnowflakeDeployment) -> None:
        open_snowflake_deployment(deployment, base_url, headers, time.sleep)

    def heartbeat(deployment_id: str, now: datetime) -> None:
        heartbeat_snowflake_deployment(
            deployment_id, now, base_url, headers, time.sleep, 30.0
        )

    def close(deployment_id: str, now: datetime) -> None:
        close_snowflake_deployment(
            deployment_id, now, base_url, headers, time.sleep, 30.0
        )

    metered = meter_deployments(
        session,
        DeploymentCalls(open_call, heartbeat, close),
        datetime.now(timezone.utc),
    )
    return f"{metered}; {reconcile(session, fetch, int(time.time() * 1000))}"


@dataclass(frozen=True)
class DeploymentCalls:
    """Arena deployment requests: open, heartbeat, and close."""

    open: Callable[[SnowflakeDeployment], None]
    heartbeat: Callable[[str, datetime], None]
    close: Callable[[str, datetime], None]


@dataclass(frozen=True)
class ServedModel:
    """A service ``publish`` started, as ``app_state.served_models`` records it.

    :param service: Upper-case service name in ``app_state``.
    :param model: Model Registry model name.
    :param version: Model Registry version name.
    :param run_id: Run the model was published from.
    :param gpu_family: Instance family of the serving pool.
    :param gpu_count: GPUs the service requests.
    """

    service: str
    model: str
    version: str
    run_id: str
    gpu_family: str
    gpu_count: int


def service_live(status: str | None) -> bool:
    """Whether a ``SHOW SERVICES`` status is billed as serving.

    :param status: Upper-case status, or ``None`` when the service is gone.
    :return: True while the service exists and is neither suspended nor ended.
    """
    stopped = {
        "SUSPENDING",
        "SUSPENDED",
        "DONE",
        "FAILED",
        "CANCELLED",
        "CANCELED",
        "DELETING",
        "DELETED",
    }
    return status is not None and status not in stopped


def meter_deployments(
    session: SnowflakeSession, calls: DeploymentCalls, now: datetime
) -> str:
    """Heartbeat each open deployment, close those whose service is not live, and open missing ones.

    A suspended service is not live, so suspending closes its deployment and
    a resumed service gets a new one. A heartbeat refused because Snowflake
    training is off drops the service and closes the deployment. The row is
    marked closed only after that close returns. A deployment Arena has closed
    is marked closed, so it settles from the usage read, and its service gets
    a new deployment. Every live service in ``app_state.served_models`` with
    no open deployment gets one; a refused open drops the service. A service
    dropped while metering its open deployment is not granted endpoint usage.
    Every other live service is, including one already metered. A 404 is
    logged and the deployment is left. A deployment Arena has never answered
    for does not count its service as metered, so a live service is opened.
    The first other Arena failure stops metering.

    :param session: Procedure session.
    :param calls: Arena deployment requests.
    :param now: Current time, timezone-aware.
    :return: Counts of open deployments, heartbeats, closes, opens, dropped
        services, deployments Arena answered 404 for, and failures.
    """
    served = served_models(session)
    rows = collect(
        session,
        "SELECT usage_id, service_name, first_seen_at FROM app_state.usage_items "
        "WHERE kind = 'deployment' AND settled_at IS NULL AND closed_at IS NULL "
        "AND service_name IS NOT NULL ORDER BY opened_at",
    )
    statuses = service_statuses(session) if rows or served else {}
    counts = dict.fromkeys(
        ("heartbeat", "closed", "opened", "dropped", "missing", "failed"), 0
    )
    metered: set[str] = set()
    dropped: set[str] = set()
    for row in rows:
        values = lowered(row)
        deployment_id, service = str(values["usage_id"]), str(values["service_name"])
        try:
            outcome = meter_deployment(
                session, deployment_id, service, statuses.get(service), calls, now
            )
        except UnknownUsageError:
            logger.warning(
                "Arena has no deployment %s of service %s. Leaving it.",
                deployment_id,
                service,
            )
            counts["missing"] += 1
            # Arena has answered for this deployment, so do not open another.
            if values["first_seen_at"] is not None:
                metered.add(service)
            continue
        except SnowflakeLicenseError as exc:
            logger.warning(
                "Could not meter deployment %s with Arena, so this run meters "
                "no more deployments: %s",
                deployment_id,
                exc,
            )
            counts["failed"] += 1
            return deployment_summary(len(rows), counts)
        except SnowparkSQLException:
            logger.exception("Could not drop service %s", service)
            counts["failed"] += 1
            metered.add(service)
            continue
        if outcome == "dropped":
            dropped.add(service)
        if outcome != "reopen":
            counts[outcome] += 1
            metered.add(service)
    for service, model in served.items():
        if service in dropped or not service_live(statuses.get(service)):
            continue
        grant_endpoint_usage(session, service)
        if service in metered:
            continue
        try:
            open_served(session, model, calls.open, now)
        except DeploymentRefusedError:
            counts["dropped"] += 1
            continue
        except SnowflakeLicenseError as exc:
            logger.warning(
                "Could not open a deployment for service %s with Arena, so this "
                "run opens no more: %s",
                service,
                exc,
            )
            counts["failed"] += 1
            break
        counts["opened"] += 1
    return deployment_summary(len(rows), counts)


def service_statuses(session: SnowflakeSession) -> dict[str, str]:
    """Upper-case ``SHOW SERVICES`` status of each service in ``app_state``, keyed by name.

    :param session: Procedure session.
    :return: The statuses.
    """
    return {
        str(values["name"]).upper(): str(values.get("status") or "").upper()
        for values in map(
            lowered, collect(session, f"SHOW SERVICES IN SCHEMA {APP_SCHEMA}")
        )
    }


def served_models(session: SnowflakeSession) -> dict[str, ServedModel]:
    """Rows of ``app_state.served_models``, keyed by service name.

    :param session: Procedure session.
    :return: The served models.
    """
    rows = collect(
        session,
        "SELECT service_name, model, version, run_id, gpu_family, "
        "gpu_count FROM app_state.served_models",
    )
    return {
        str(v["service_name"]): ServedModel(
            str(v["service_name"]),
            str(v["model"]),
            str(v["version"]),
            str(v["run_id"]),
            str(v["gpu_family"]),
            int(str(v["gpu_count"])),
        )
        for v in map(lowered, rows)
    }


def deployment_summary(deployments: int, counts: dict[str, int]) -> str:
    """One line of :func:`meter_deployments` counts.

    :param deployments: Open deployments read.
    :param counts: Count of each outcome.
    :return: The line.
    """
    return f"deployments: {deployments}, " + ", ".join(
        f"{name}: {count}" for name, count in counts.items()
    )


def meter_deployment(
    session: SnowflakeSession,
    deployment_id: str,
    service: str,
    status: str | None,
    calls: DeploymentCalls,
    now: datetime,
) -> str:
    """Heartbeat one open deployment while its service is live, or close it.

    :param session: Procedure session.
    :param deployment_id: Arena deployment id.
    :param service: Service name in ``app_state``.
    :param status: ``SHOW SERVICES`` status, or ``None`` when the service is gone.
    :param calls: Arena deployment requests.
    :param now: Current time, timezone-aware.
    :return: ``heartbeat``, ``closed``, ``dropped``, or ``reopen`` when Arena
        had closed it and the service needs a new deployment.
    :raises UnknownUsageError: Arena has no such deployment.
    :raises SnowflakeLicenseError: The Arena request failed.
    :raises SnowparkSQLException: Snowflake did not drop the service.
    """
    if not service_live(status):
        try:
            calls.close(deployment_id, now)
        except UnknownUsageError:
            logger.warning("Arena has no deployment %s to close.", deployment_id)
        mark_deployment(session, deployment_id, now, None)
        return "closed"
    try:
        calls.heartbeat(deployment_id, now)
    except TrainingDisabledError as exc:
        collect(session, f"DROP SERVICE IF EXISTS {APP_SCHEMA}.{service}")
        try:
            calls.close(deployment_id, now)
        except UnknownUsageError:
            logger.warning("Arena has no deployment %s to close.", deployment_id)
        mark_deployment(session, deployment_id, now, str(exc))
        return "dropped"
    except DeploymentClosedError as exc:
        mark_deployment(session, deployment_id, now, str(exc))
        return "reopen"
    return "heartbeat"


def mark_deployment(
    session: SnowflakeSession, deployment_id: str, now: datetime, error: str | None
) -> None:
    """Stamp a deployment closed, with Arena's reason when there is one.

    :param session: Procedure session.
    :param deployment_id: Arena deployment id.
    :param now: Current time, timezone-aware.
    :param error: Reason logged and stored in ``last_error``, or ``None``.
    """
    stamp = f"TO_TIMESTAMP_TZ({int(now.timestamp() * 1000)}, 3)"
    columns = f"closed_at = {stamp}"
    if error is not None:
        logger.warning("Deployment %s: %s", deployment_id, error)
        columns += f", last_error = {string_literal(error)}, last_error_at = {stamp}"
    collect(
        session,
        f"UPDATE app_state.usage_items SET {columns} WHERE kind = 'deployment' "
        f"AND usage_id = {string_literal(deployment_id)}",
    )


def publish(
    session: PublishSession,
    run_id: str,
    model_name: str,
    version_name: str | None,
    step: int | None,
    serve: bool,
    gpu_count: int,
    service_name: str | None,
    max_model_len: int | None,
) -> str:
    """Log a run's merged checkpoint to the app's Model Registry schema and wait for the job.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :param model_name: Model Registry model name.
    :param version_name: Registry version. ``None`` uses the checkpoint directory name.
    :param step: Checkpoint step. ``None`` takes the newest complete merged export.
    :param serve: Start a vLLM service with a public endpoint after logging the model.
    :param gpu_count: GPUs the service requests.
    :param service_name: Service name. ``None`` uses ``<model>_<version>``.
    :param max_model_len: vLLM ``--max-model-len``, or ``None``.
    :return: The job, model, version, and any service, its endpoint, and its
        Arena deployment.
    :raises ValueError: An argument is invalid, Arena is not configured for a
        served model, the app has no such run, or the run has no usable
        checkpoint.
    :raises RuntimeError: The publish job failed or outlived the wait. A
        service it starts later gets its deployment from the hourly task.
    :raises SnowflakeLicenseError: Arena did not open the deployment. The
        service was dropped.
    """
    if gpu_count < 1:
        msg = "gpu_count must be at least 1.\nNothing was published."
        raise ValueError(msg)
    partner = arena_partner() if serve else None
    known_run(session, run_id)
    profile = app_setup(session)
    database, schema = app_registry(session)
    job = submit_publish(
        session,
        PublishSettings(
            run_id=run_id,
            runs_stage=profile.stage,
            model_name=model_name,
            version_name=version_name,
            step=step,
            serve=serve,
            compute_pool=sole_gpu_pool(profile),
            image_build_pool=profile.pools.head.pool,
            image=profile.image,
            warehouse=profile.warehouse,
            gpu_count=gpu_count,
            max_model_len=max_model_len,
            service_name=service_name,
            database=database,
            schema=schema,
            job_schema=APP_SCHEMA,
        ),
    )
    document = job.document
    service = document["service_name"]
    served = None
    if service is not None:
        served = published_model(profile, document, service, run_id)
        record_served(session, served)
    wait_for_publish(session, job)
    lines = [
        f"job: {job.name}",
        f"model: {document['database']}.{document['schema']}.{document['model_name']}",
        f"version: {document['version_name']}",
    ]
    if served is not None and partner is not None:
        base_url, headers = partner
        try:
            grant_endpoint_usage(session, served.service)
            deployment_id = open_served(
                session,
                served,
                lambda deployment: open_snowflake_deployment(
                    deployment, base_url, headers, time.sleep
                ),
                datetime.now(timezone.utc),
            )
        except Exception:
            collect(session, f"DROP SERVICE IF EXISTS {APP_SCHEMA}.{service}")
            raise
        lines.append(f"service: {service}")
        lines.append(f"endpoint: {service_endpoint(session, document)}")
        lines.append(f"deployment: {deployment_id}")
    return "\n".join(lines)


def published_model(
    profile: AccountProfile, document: PublishDocument, service: str, run_id: str
) -> ServedModel:
    """Build the record of what a publish job's service serves.

    :param profile: The app's ``setup.json``.
    :param document: Publish job document.
    :param service: The document's service name.
    :param run_id: Run the model was published from.
    :return: The served model.
    """
    families = {pool.pool.upper(): family for family, pool in profile.pools.gpu.items()}
    return ServedModel(
        service.upper(),
        document["model_name"],
        document["version_name"],
        run_id,
        families[document["compute_pool"].upper()],
        document["gpu_count"],
    )


def grant_endpoint_usage(session: SnowflakeSession, service: str) -> None:
    """Let ``app_user`` call the endpoints of a service, which only the app can call otherwise.

    :param session: Procedure session.
    :param service: Service in ``app_state``.
    """
    collect(
        session,
        f"GRANT SERVICE ROLE {APP_SCHEMA}.{service}!ALL_ENDPOINTS_USAGE "
        "TO APPLICATION ROLE app_user",
    )


def record_served(session: SnowflakeSession, served: ServedModel) -> None:
    """Replace the ``app_state.served_models`` row of a service.

    Recorded before the service exists, so the hourly task can open its
    deployment when the publish wait ends first.

    :param session: Procedure session.
    :param served: What the service serves.
    """
    name = string_literal(served.service)
    collect(session, f"DELETE FROM app_state.served_models WHERE service_name = {name}")
    collect(
        session,
        "INSERT INTO app_state.served_models "
        "(service_name, model, version, run_id, gpu_family, gpu_count) VALUES "
        f"({name}, {string_literal(served.model)}, {string_literal(served.version)}, "
        f"{string_literal(served.run_id)}, {string_literal(served.gpu_family)}, "
        f"{served.gpu_count})",
    )


def open_served(
    session: SnowflakeSession,
    served: ServedModel,
    open_call: Callable[[SnowflakeDeployment], None],
    now: datetime,
) -> str:
    """Open a new Arena deployment for a live service; a refused open drops the service.

    The deployment is added to ``app_state.usage_items`` before Arena opens it,
    so the hourly task bills an open whose answer was lost.

    :param session: Procedure session.
    :param served: What the service serves.
    :param open_call: Arena open of a deployment.
    :param now: Start of the deployment, timezone-aware.
    :return: The deployment id.
    :raises DeploymentRefusedError: Arena refused the open. The service was
        dropped and the deployment marked closed with Arena's reason.
    :raises SnowflakeLicenseError: The open did not land.
    """
    deployment_id = deployment_name(served.service, now)
    collect(
        session,
        "INSERT INTO app_state.usage_items "
        "(kind, usage_id, opened_at, service_name) VALUES ('deployment', "
        f"{string_literal(deployment_id)}, CURRENT_TIMESTAMP(), "
        f"{string_literal(served.service)})",
    )
    [row] = collect(session, "SELECT CURRENT_ACCOUNT() AS ACCOUNT")
    try:
        open_call(
            SnowflakeDeployment(
                deployment_id=deployment_id,
                snowflake_account=str(lowered(row)["account"]),
                gpu_count=served.gpu_count,
                gpu_family=served.gpu_family,
                started_at=now,
                model=served.model,
                version=served.version,
                run_id=served.run_id,
            )
        )
    except DeploymentRefusedError as exc:
        collect(session, f"DROP SERVICE IF EXISTS {APP_SCHEMA}.{served.service}")
        mark_deployment(session, deployment_id, now, str(exc))
        raise
    return deployment_id


def deployment_name(service: str, started_at: datetime) -> str:
    """Arena deployment id ``<SERVICE>-<yyyymmddThhmmssZ>``.

    :param service: Service name.
    :param started_at: Start, timezone-aware.
    :return: The id, at most 64 characters.
    """
    stamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # bill_hours takes ids of at most 64 characters, and Arena refuses $.
    prefix = service.upper().replace("$", "_")[: 63 - len(stamp)]
    return f"{prefix}-{stamp}"


def endpoint(session: SnowflakeSession, model: str, version_name: str | None) -> str:
    """Ingress URL of the inference service on a published model version.

    :param session: Procedure session.
    :param model: Model Registry model name in ``app_state``.
    :param version_name: Version. ``None`` takes the one version that has a service.
    :return: The URL, or Snowflake's provisioning message while ingress starts.
    :raises ValueError: The model, version, or service is not there.
    """
    database, schema = app_registry(session)
    served = served_version(session, database, schema, model, version_name)
    return version_endpoint(session, served)


def unpublish(
    session: SnowflakeSession, model: str, version_name: str | None, drop_version: bool
) -> str:
    """Drop the inference service of a published version and, when asked, the version.

    Then closes the Arena deployment of each dropped service.

    :param session: Procedure session.
    :param model: Model Registry model name in ``app_state``.
    :param version_name: Version. ``None`` takes the one version that has a service.
    :param drop_version: Also drop the version. A model's only version is
        dropped with the model.
    :return: One line per dropped object and closed deployment.
    :raises ValueError: The model or version is not there, there is nothing to
        drop, or the version is the default of several; or Arena is not
        configured to close a deployment.
    :raises SnowflakeLicenseError: Arena did not close a deployment. The
        hourly task closes it.
    """
    database, schema = app_registry(session)
    served = served_version(session, database, schema, model, version_name)
    drop_served(session, served, drop_version)
    lines = [f"dropped service: {service}" for service in served.services]
    if drop_version and served.only_version:
        lines.append(f"dropped model: {served.model}")
    elif drop_version:
        lines.append(f"dropped version: {served.version_name}")
    lines.extend(
        f"closed deployment: {deployment_id}"
        for deployment_id in close_deployments(session, served.services)
    )
    return "\n".join(lines)


def close_deployments(
    session: SnowflakeSession, services: tuple[str, ...]
) -> list[str]:
    """Forget dropped services and close their open Arena deployments, stamped now.

    :param session: Procedure session.
    :param services: Fully qualified service names.
    :return: Deployment ids Arena closed. A deployment Arena has no record of
        is logged and left out.
    :raises ValueError: A deployment needs closing and Arena is not configured.
    :raises SnowflakeLicenseError: Arena did not close a deployment.
    """
    if not services:
        return []
    names = ", ".join(
        string_literal(service.rsplit(".", 1)[-1].upper()) for service in services
    )
    collect(
        session, f"DELETE FROM app_state.served_models WHERE service_name IN ({names})"
    )
    rows = collect(
        session,
        "SELECT usage_id FROM app_state.usage_items WHERE kind = 'deployment' "
        f"AND settled_at IS NULL AND closed_at IS NULL AND service_name IN ({names})",
    )
    if not rows:
        return []
    base_url, headers = arena_partner()
    ended_at = datetime.now(timezone.utc)
    closed = []
    for row in rows:
        deployment_id = str(lowered(row)["usage_id"])
        try:
            close_snowflake_deployment(
                deployment_id, ended_at, base_url, headers, time.sleep
            )
        except UnknownUsageError:
            logger.warning("Arena has no deployment %s to close.", deployment_id)
        else:
            closed.append(deployment_id)
        mark_deployment(session, deployment_id, ended_at, None)
    return closed
