# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stored procedures of the Arena Snowflake Native App: setup, runs, and published models.

The setup script registers each handler with ``IMPORTS`` of this file,
``app_billing.py``, ``app_keys.py``, ``app_trial.py`` and
``agilerl_arena.zip``, so these run as owner's-rights procedures inside the app.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app_billing import (
    APP_SCHEMA,
    BILLING_TASK,
    arena_partner,
    close_deployments,
    drop_served_service,
    grant_chat_function,
    open_served,
    published_model,
    reconcile_summary,
    record_served,
    saved_partner,
    suspend_idle_gpu_pools,
    sync_billing_task,
)
from app_keys import (
    app_capabilities,
    arena_origin,
    claim_account,
    collect_linked_key,
    priced_lines,
    rotate_if_due,
    save_arena_key,
)
from app_trial import refuse_ended_trial

from agilerl.models.algorithms.base import LLMAlgorithmSpec
from agilerl.models.algorithms.opd import OPDSpec
from agilerl.models.env import LLMEnvSpec
from agilerl.models.model_info import (
    REMOTE_CODE_MODEL_TYPES,
    SUPPORTED_MODEL_INFO,
)
from agilerl.models.rewards import SnowflakeFunctionReward
from agilerl.snowflake.account import blocking_services, shrink_message
from agilerl.snowflake.catalog import (
    PoolState,
    collect,
    existing_pools,
    family_sized,
    load_families,
    remote_profile,
)
from agilerl.snowflake.gpu_fit import gpu_counts_message, pool_gpus
from agilerl.snowflake.job import (
    ROLES,
    RunServices,
    ServiceLeftRunningError,
    SubmitSettings,
    cancel_run,
    choose_gpu_family,
    lowered,
    new_run_id,
    runs_prefix,
    submit_job,
    write_job_manifest,
)
from agilerl.snowflake.license import (
    NATIVE_APP_CHANNEL,
    SNOWFLAKE_DENIED,
    AppKey,
    ClaimAccount,
    SnowflakeLicenseError,
    fetch_capabilities,
    open_snowflake_deployment,
    partner_headers,
    snowflake_priced_families,
    snowflake_training_enabled,
)
from agilerl.snowflake.manifest_refusal import load_app_manifest
from agilerl.snowflake.profile import (
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
    profile_from_mapping,
    put_setup_json,
    sole_gpu_pool,
)
from agilerl.snowflake.publish import (
    PublishSession,
    PublishSettings,
    chat_example,
    chat_function,
    drop_chat_function_sql,
    registry_schema,
    select_checkpoint,
    served_version,
    submit_publish,
    version_service,
    wait_for_publish,
)
from agilerl.snowflake.publish import unpublish as drop_served
from agilerl.snowflake.rewards import REWARD_REFERENCE
from agilerl.snowflake.setup import (
    ORG_KEY_PREFIX,
    check_gpu_cap,
    family_listed,
    format_family_lines,
    missing_family_message,
    no_gpu_message,
    nodes_for_gpus,
    unsupported_gpu_message,
)
from agilerl.snowflake.setup_sql import (
    PoolPlan,
    pool_statements,
    string_literal,
)
from agilerl.snowflake.sizing import manifest_fits
from agilerl.snowflake.source import QuerySource, SnowflakeSession
from agilerl.snowflake.status import run_status, service_logs

if TYPE_CHECKING:
    from agilerl.models.manifest import TrainingManifest
    from agilerl.snowflake.catalog import GpuFamily

SOURCE_REFERENCES = ("SOURCE_TABLE", "SOURCE_VIEW")
# Single-valued, unlike the source references.
EXPORT_REFERENCE = "EXPORT_STAGE"
SINGLE_REFERENCES = (EXPORT_REFERENCE, REWARD_REFERENCE)
REWARDS_DOCS = "https://docs.arena.agilerl.com/partners/snowflake-native-app-rewards/"
REWARD_OPTIONS = (
    "Snowflake does not let a Native App run code from outside its package, so "
    "the app takes no reward file. Either pick a built-in reward in the manifest "
    "(environment.reward with type exact_match, numeric, regex, json_schema or "
    "tool_call), or write the reward as a function in your account, bind it to "
    "the app's reward_function reference, and set environment.reward to "
    f"{{type: snowflake_function}}. Examples: {REWARDS_DOCS}"
)
# snowflake-app-build replaces the tag with the release tag, as in manifest.yml.
IMAGE = "/arena_provider/images/repo/snowflake-job:0.0.0-py313-cu130"
SETUP_PREFIX = f"@{APP_SCHEMA}.{RUNS_STAGE}/_arena"

logger = logging.getLogger(__name__)


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
    session: SnowflakeSession,
    org_key: str,
    external_user_id: str,
    base_url: str | None,
) -> str:
    """Check an org key AgileRL issued with Arena, then save the three Arena secrets in ``app_state``.

    No secret value is logged, returned, or quoted in an error.

    :param session: Procedure session.
    :param org_key: Arena organisation key.
    :param external_user_id: External user id the key acts for.
    :param base_url: Arena origin, or ``None`` for ``https://arena.agilerl.com``.
        Its host must be in the app's Arena network rule.
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
            "Run CALL app_code.connect_arena() to connect, or paste a key AgileRL gave "
            f"you (it starts with {ORG_KEY_PREFIX}).\n"
            "Nothing was saved."
        )
        raise ValueError(msg)
    if external_user_id == "":
        msg = "Pass the Arena external user id.\nNothing was saved."
        raise ValueError(msg)
    # An explicit NULL from SQL skips the procedure's DEFAULT, so default here too.
    origin = arena_origin(session, base_url or "https://arena.agilerl.com")
    caps = fetch_capabilities(origin, partner_headers(org_key, external_user_id))
    if not snowflake_training_enabled(caps):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)
    save_arena_key(
        session, origin, AppKey(org_key, external_user_id, None, None), "configure"
    )
    lines = [
        "Saved the Arena org key, external user id, and base URL.",
        "Arena allows Snowflake training for this organisation.",
    ]
    return "\n".join([*lines, *priced_lines(caps)])


def register_reference(
    session: SnowflakeSession, ref_name: str, operation: str, ref_or_alias: str
) -> None:
    """Bind, unbind, or clear a consumer object on a reference.

    Snowflake calls this when the consumer changes a ``source_table``,
    ``source_view``, ``export_stage`` or ``reward_function`` binding. The
    source references are multi-valued; ``export_stage`` holds one stage and
    ``reward_function`` one function.

    :param session: Procedure session.
    :param ref_name: Reference name from the manifest.
    :param operation: ``ADD``, ``REMOVE``, or ``CLEAR``.
    :param ref_or_alias: Reference to add, or the alias to remove.
    :raises ValueError: The reference or operation is unknown.
    """
    name = (ref_name or "").upper()
    if name not in (*SOURCE_REFERENCES, *SINGLE_REFERENCES):
        msg = (
            f"Unknown reference {ref_name}. "
            "Bind source_table, source_view, export_stage or reward_function."
        )
        raise ValueError(msg)
    reference = string_literal(name)
    add = "SET_REFERENCE" if name in SINGLE_REFERENCES else "ADD_REFERENCE"
    match (operation or "").upper():
        case "ADD":
            statement = (
                f"SELECT SYSTEM${add}({reference}, {string_literal(ref_or_alias)})"
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
    :raises ValueError: A family is unknown, missing from this region or with
        another GPU count per node here, ``max_gpus`` is not whole nodes of a
        family, an existing pool uses another family, or a queued or running
        service blocks the change.
    """
    names = gpu_names(gpu_families, max_gpus)
    # An explicit NULL from SQL skips the procedure's DEFAULT, so default here too.
    broker = (broker_family or "CPU_X64_L").strip().upper()
    head = (head_family or "CPU_X64_SL").strip().upper()
    families = load_families(session)
    for family, role in ((broker, "broker"), (head, "head")):
        if not family_listed(family, families):
            raise ValueError(missing_family_message(family, role))
    for name in names:
        if not family_sized(name, families):
            rerun = "CALL <app>.app_code.set_pools(['<name>'])"
            raise ValueError(no_gpu_message(name, families, rerun))
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
        ``max_gpus`` is below 1 or not whole nodes of a family.
    """
    names = list(dict.fromkeys(str(item).strip().upper() for item in gpu_families))
    if not names:
        msg = "Name at least one GPU family.\nNo pool was changed."
        raise ValueError(msg)
    for name in names:
        if name not in GPU_FAMILIES:
            raise ValueError(unsupported_gpu_message(name))
    if max_gpus is None:
        return names
    if max_gpus < 1:
        msg = "max_gpus must be at least 1.\nNo pool was changed."
        raise ValueError(msg)
    for name in names:
        check_gpu_cap(name, max_gpus, GPU_FAMILIES[name].gpus)
    return names


def gpu_families(session: SnowflakeSession) -> str:
    """GPU families ``set_pools`` can use: in this region with the app's GPU count, and priced by Arena.

    :param session: Procedure session.
    :return: One line per family, or the region's GPU families when none qualifies.
    :raises ValueError: Arena is not configured.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    priced = snowflake_priced_families(app_capabilities(*arena_partner()))
    region = load_families(session)
    usable = [
        item
        for item in region
        if family_sized(item.name, region)
        and (priced is None or item.name.upper() in priced)
    ]
    if usable:
        return "GPU families set_pools can use here:\n" + format_family_lines(usable)
    return (
        "No GPU family in this region is both sized by the app and priced by Arena.\n"
        "GPU families SHOW COMPUTE POOL INSTANCE FAMILIES returned:\n"
        + format_family_lines([item for item in region if item.gpu_count > 0])
    )


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
    :raises ValueError: An existing pool uses another family, or a queued or
        running service blocks a resize or drop.
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
    :param gpu_family: GPU family. ``None`` picks one with ``choose_gpu_family``.
    :param reward_file: Refused unless ``None``: the app runs no code from
        outside its package.
    :param checkpoint_steps: Training steps between checkpoints, or ``None``.
    :param resume_from: Run whose newest finished checkpoint the run starts
        from, or ``None``.
    :return: The run id, the service names, and any manifest warnings.
    :raises ValueError: An argument is out of range, a reward file is passed,
        the source is not a bound reference, the manifest names a rubric file
        or a snowflake_function reward with no function bound, a model is
        outside the catalog or runs remote code, the
        pools are not set, Arena has no price for the family, the manifest
        cannot run on them, or the account's node limit is reached. A failed
        start leaves no service and no usage item.
    :raises PermissionError: The trial has ended and the listing is not purchased.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    if gpus < 1:
        msg = "gpus must be at least 1.\nNothing was started."
        raise ValueError(msg)
    if checkpoint_steps is not None and checkpoint_steps < 1:
        msg = "checkpoint_steps must be at least 1.\nNothing was started."
        raise ValueError(msg)
    if reward_file is not None:
        msg = (
            "reward_file is not available in the Snowflake app.\n"
            f"{REWARD_OPTIONS}\nNothing was started."
        )
        raise ValueError(msg)
    source = reference_source(source_reference, source_alias)
    with (
        tempfile.TemporaryDirectory() as tmp,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        directory = Path(tmp)
        submitted = directory / "submitted.yaml"
        submitted.write_text(manifest, encoding="utf-8")
        job_manifest = write_job_manifest(submitted, source, False, directory)
        parsed = load_app_manifest(job_manifest)
        refuse_remote_code(parsed)
        check_app_reward(session, parsed)
        if source_reference is not None and source_alias is not None:
            check_bound_alias(session, source_reference, source_alias)
        refuse_ended_trial(session)
        profile = app_setup(session)
        family = run_family(session, profile, job_manifest, gpus, gpu_family)
        settings = submit_settings(
            profile,
            app_names(session).api_integration,
            source,
            gpus,
            family,
            checkpoint_steps,
            resume_from,
            claim_account(session),
        )
        run_id = new_run_id()
        services = start_billed_run(session, job_manifest, settings, run_id)
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


def start_billed_run(
    session: SnowflakeSession, job_manifest: Path, settings: SubmitSettings, run_id: str
) -> RunServices:
    """Record the run for billing, then start its services.

    :param session: Procedure session.
    :param job_manifest: Manifest the run stages.
    :param settings: The run's submit settings.
    :param run_id: The new run's id.
    :return: The started services.
    :raises ValueError: The account's node limit is reached. A failed start
        leaves no usage item.
    :raises ServiceLeftRunningError: A service could not be dropped; its usage
        item stays so it is billed.
    """
    # Recorded before any service starts, so a started run is always billed.
    collect(
        session,
        "INSERT INTO app_state.usage_items (kind, usage_id, opened_at) "
        f"VALUES ('run', {string_literal(run_id)}, CURRENT_TIMESTAMP())",
    )
    collect(session, f"ALTER TASK {BILLING_TASK} RESUME")
    try:
        _, services = submit_job(session, job_manifest, settings, run_id)
    except ServiceLeftRunningError:
        raise
    except Exception:
        # submit_job dropped every service it started, so there is nothing to bill.
        collect(
            session,
            "DELETE FROM app_state.usage_items WHERE kind = 'run' "
            f"AND usage_id = {string_literal(run_id)}",
        )
        raise
    return services


def run_family(
    session: SnowflakeSession,
    profile: AccountProfile,
    job_manifest: Path,
    gpus: int,
    gpu_family: str | None,
) -> str:
    """GPU family a new run uses: an app pool that is priced, sized, holds ``gpus`` and fits the manifest.

    :param session: Procedure session.
    :param profile: The app's ``setup.json``.
    :param job_manifest: Manifest the run stages.
    :param gpus: GPUs for the run.
    :param gpu_family: Family the caller named in any case, or ``None``.
    :return: The family.
    :raises ValueError: No pool can run the request; the message says why.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    named = gpu_family.strip().upper() if gpu_family is not None else None
    # Rotate a due key first, so the run's head reads the rotated key.
    partner = rotate_if_due(session, arena_partner())
    pools = holding_pools(run_pools(session, profile, named, partner), gpus, named)
    return choose_gpu_family(
        pools,
        gpus,
        named,
        lambda family, pool: manifest_fits(job_manifest, family, pool, gpus, *partner),
    )


def run_pools(
    session: SnowflakeSession,
    profile: AccountProfile,
    gpu_family: str | None,
    partner: tuple[str, dict[str, str]],
) -> dict[str, GpuPool]:
    """GPU pools a new run may use: priced by Arena and sized for this region.

    :param session: Procedure session.
    :param profile: The app's ``setup.json``.
    :param gpu_family: Family the caller named, or ``None``.
    :param partner: Arena origin and partner headers.
    :return: The pools, keyed by family.
    :raises ValueError: The named family has no pool, or no pool, or not the
        named one, has a price and the GPUs the app sized.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    if gpu_family is not None and gpu_family not in profile.pools.gpu:
        raise ValueError(missing_pool_message(gpu_family, list(profile.pools.gpu)))
    priced = priced_pools(profile.pools.gpu, gpu_family, partner)
    return sized_pools(priced, load_families(session), gpu_family)


def holding_pools(
    pools: dict[str, GpuPool], gpus: int, gpu_family: str | None
) -> dict[str, GpuPool]:
    """Keep the pools, or the named one, that hold ``gpus`` on their ``MAX_NODES``.

    The app writes ``setup.json`` with every pool it sizes, so a run that
    needs more nodes is refused here, before the run is recorded for billing.

    :param pools: GPU pools a new run may use, keyed by family.
    :param gpus: GPUs for the run.
    :param gpu_family: Family the caller named, or ``None``. A named family
        is one of ``pools``.
    :return: The pools that hold ``gpus``.
    :raises ValueError: None of them does.
    """
    candidates = pools if gpu_family is None else {gpu_family: pools[gpu_family]}
    kept = {
        family: pool for family, pool in candidates.items() if gpus <= pool_gpus(pool)
    }
    if not kept:
        raise ValueError(gpu_counts_message(gpus, candidates, gpu_family))
    return kept


def missing_pool_message(gpu_family: str, families: list[str]) -> str:
    """Refusal for a GPU family the app has no pool for, with the ``set_pools`` call that adds it.

    :param gpu_family: Family the caller named.
    :param families: Families the app has GPU pools for.
    :return: The message.
    """
    kept = ", ".join(f"'{name}'" for name in [*families, gpu_family])
    return (
        f"The app has no GPU pool for {gpu_family}. "
        f"GPU pools it has: {', '.join(families)}.\n"
        "An app_admin adds the pool, keeping the ones it has, with\n"
        f"CALL <app>.app_code.set_pools([{kept}]);\n"
        "Nothing was started."
    )


def priced_pools(
    pools: dict[str, GpuPool],
    gpu_family: str | None,
    partner: tuple[str, dict[str, str]],
) -> dict[str, GpuPool]:
    """Keep the app's GPU pools whose family Arena has a Snowflake price for.

    Arena refuses an unpriced family only when the head opens the run, after
    the GPU nodes have started, so ``submit`` checks first.

    :param pools: GPU pools in ``setup.json``, keyed by family.
    :param gpu_family: Family the caller named, or ``None``.
    :param partner: Arena origin and partner headers.
    :return: The priced pools, or every pool when Arena does not list prices.
    :raises ValueError: The named family has no price, or no pool's family has one.
    :raises SnowflakeLicenseError: Arena could not be reached or refused the key.
    """
    priced = snowflake_priced_families(app_capabilities(*partner))
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


def sized_pools(
    pools: dict[str, GpuPool], region: list[GpuFamily], gpu_family: str | None
) -> dict[str, GpuPool]:
    """Keep the app's GPU pools whose family this region has with the GPUs the app sized.

    :param pools: GPU pools in ``setup.json``, keyed by family.
    :param region: Families in this region.
    :param gpu_family: Family the caller named, or ``None``.
    :return: The pools :func:`family_sized` accepts.
    :raises ValueError: The named family, or every pool's family, fails that check.
    """
    kept = {
        family: pool for family, pool in pools.items() if family_sized(family, region)
    }
    refused = sorted(set(pools) - set(kept))
    if gpu_family in refused or (refused and not kept):
        named = [gpu_family] if gpu_family in refused else refused
        msg = (
            f"This region does not have {', '.join(named)} with the GPUs per node "
            "the app sized, so the run would ask for GPUs the nodes lack.\n"
            "An app_admin lists usable families with gpu_families() and calls "
            "set_pools with one of them.\n"
            "Nothing was started."
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
    resume_from: str | None,
    install: ClaimAccount,
) -> SubmitSettings:
    """Build the settings of a run on the app's pools, secrets, and integrations.

    :param profile: The app's ``setup.json``.
    :param api_integration: The app's Arena integration.
    :param source: Query the run reads, or ``None``.
    :param gpus: GPUs for the run.
    :param family: GPU family of the run's pool.
    :param checkpoint_steps: Training steps between checkpoints, or ``None``.
    :param resume_from: Run to resume from, or ``None``.
    :param install: Region and purchased flag the head sends with the run open.
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
        reward_file=None,
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
        snowflake_region=install.region,
        listing_purchased=install.purchased,
    )


def refuse_remote_code(manifest: TrainingManifest) -> None:
    """Refuse a model whose run would execute code from outside the app package.

    Snowflake does not let a Native App run code from outside its package. A
    model outside the catalog is refused too, since the app cannot read its
    config before the run starts.

    :param manifest: The run's manifest.
    :raises ValueError: ``vllm_engine_args.trust_remote_code`` is true, or the
        policy or teacher model is outside the catalog or its family loads
        Python from its Hugging Face repo.
    """
    algorithm = manifest.algorithm
    if not isinstance(algorithm, LLMAlgorithmSpec):
        return
    if algorithm.vllm_engine_args.get("trust_remote_code"):
        msg = (
            "algorithm.vllm_engine_args.trust_remote_code is not available in "
            "the Snowflake app. It runs code from the model's Hugging Face "
            "repository, and Snowflake does not allow a Native App to run code "
            "from outside its package. Remove it from the manifest.\n"
            "Nothing was started."
        )
        raise ValueError(msg)
    hub_ids = [algorithm.pretrained_model_name_or_path]
    if isinstance(algorithm, OPDSpec) and algorithm.teacher.model is not None:
        hub_ids.append(algorithm.teacher.model.pretrained_model_name_or_path)
    for hub_id in hub_ids:
        info = SUPPORTED_MODEL_INFO.get(hub_id)
        if info is None:
            msg = (
                f"{hub_id} is not in the Arena model catalog, so the Snowflake "
                "app cannot check that it runs without outside code. Pick a "
                "model from 'arena models supported'.\nNothing was started."
            )
            raise ValueError(msg)
        if info.model_type in REMOTE_CODE_MODEL_TYPES:
            msg = (
                f"{hub_id} is not available in the Snowflake app. It runs code "
                "from its Hugging Face repository, and Snowflake does not allow "
                "a Native App to run code from outside its package. Train it on "
                "Arena or with the arena CLI, or pick another model from "
                "'arena models supported'.\nNothing was started."
            )
            raise ValueError(msg)


def check_app_reward(session: SnowflakeSession, manifest: TrainingManifest) -> None:
    """Refuse a rubric file, and a ``snowflake_function`` reward with no function bound.

    :param session: Procedure session.
    :param manifest: The run's manifest.
    :raises ValueError: The environment names a rubric file, or its reward is
        ``snowflake_function`` and nothing is bound to ``reward_function``.
    """
    environment = manifest.environment
    if not isinstance(environment, LLMEnvSpec):
        return
    if environment.rubric_file_path is not None:
        msg = (
            "environment.rubric_file_path is not available in the Snowflake app.\n"
            f"{REWARD_OPTIONS}\nNothing was started."
        )
        raise ValueError(msg)
    if not isinstance(environment.reward, SnowflakeFunctionReward):
        return
    [row] = collect(
        session,
        f"SELECT SYSTEM$GET_ALL_REFERENCES({string_literal(REWARD_REFERENCE)}) AS REFS",
    )
    if not json.loads(str(lowered(row)["refs"])):
        msg = (
            "environment.reward is snowflake_function, but no function is bound "
            "to the app's reward_function reference. Bind one in Snowsight on the "
            f"app's references, or with app_code.register_reference. Examples: "
            f"{REWARDS_DOCS}\nNothing was started."
        )
        raise ValueError(msg)


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


def check_bound_alias(session: SnowflakeSession, reference: str, alias: str) -> None:
    """Refuse a source alias that is not bound to its reference.

    ``reference()`` on an unbound alias fails only when the run reads its
    rows, after the GPU nodes have started, so ``submit`` checks first.

    :param session: Procedure session.
    :param reference: ``source_table`` or ``source_view``, as
        :func:`reference_source` accepted it.
    :param alias: Alias the caller passed.
    :raises ValueError: No object is bound to ``reference`` under ``alias``.
    """
    name = reference.upper()
    [row] = collect(
        session, f"SELECT SYSTEM$GET_ALL_REFERENCES({string_literal(name)}) AS REFS"
    )
    bound = [str(item) for item in json.loads(str(lowered(row)["refs"]))]
    if alias not in bound:
        raise ValueError(unbound_alias_message(name, alias, bound))


def unbound_alias_message(name: str, alias: str, bound: list[str]) -> str:
    """Refusal for a source alias that is not bound, with the bound ones and how to bind.

    :param name: ``SOURCE_TABLE`` or ``SOURCE_VIEW``.
    :param alias: Alias the caller passed.
    :param bound: Aliases bound to ``name``.
    :return: The message.
    """
    reference = name.lower()
    kind = "TABLE" if name == "SOURCE_TABLE" else "VIEW"
    listed = ", ".join(bound) if bound else "none"
    return (
        f"Nothing is bound to {reference} with the alias {alias}.\n"
        f"Aliases bound to {reference}: {listed}.\n"
        f"To bind a {kind.lower()}, call\n"
        f"CALL <app>.app_code.register_reference('{reference}', 'ADD', "
        f"SYSTEM$REFERENCE('{kind}', '<database>.<schema>.<name>', 'PERSISTENT', "
        "'SELECT'));\n"
        "then pass the alias SHOW REFERENCES IN APPLICATION <app> lists.\n"
        "Nothing was started."
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
        msg = (
            "The app has no compute pools yet.\n"
            "An app_admin calls app_code.set_pools first.\nNothing was started."
        )
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
    :return: The log text, or a line saying the container has not started or
        has written no logs yet.
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
    return service_logs(
        session,
        run_id,
        instance,
        role,
        APP_SCHEMA,
        status_command=f"CALL <app>.app_code.status('{run_id}')",
    )


def cancel(session: SnowflakeSession, run_id: str) -> str:
    """Drop a run's broker, head, and workers, then suspend GPU pools left with nothing on them.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :return: Confirmation and the suspended pools.
    :raises ValueError: The app has no such run.
    """
    known_run(session, run_id)
    cancel_run(session, run_id, APP_SCHEMA)
    lines = [f"cancelled: {run_id}"]
    lines.extend(f"suspended: {pool}" for pool in suspend_idle_gpu_pools(session))
    return "\n".join(lines)


def export_model(session: SnowflakeSession, run_id: str, step: int | None) -> str:
    """Copy a run's merged checkpoint to ``<run_id>/<checkpoint>/`` on the stage bound to ``export_stage``.

    Starts no service and adds no usage item, so it is not billed by Arena, and
    works after the trial ends: it only hands back a model trained in the app.

    :param session: Procedure session.
    :param run_id: Run id from ``submit``.
    :param step: Checkpoint step. ``None`` takes the newest complete merged export.
    :return: The checkpoint, the stage path, and how many files were copied.
    :raises ValueError: No stage is bound to ``export_stage``, the app has no
        such run, or the run has no complete merged export at that step.
    """
    [row] = collect(
        session,
        f"SELECT SYSTEM$GET_ALL_REFERENCES({string_literal(EXPORT_REFERENCE)}) AS REFS",
    )
    if json.loads(str(lowered(row)["refs"])) == []:
        msg = (
            "No stage is bound to export_stage.\n"
            "Bind one of your stages to it with WRITE, then call export_model again."
        )
        raise ValueError(msg)
    known_run(session, run_id)
    prefix = runs_prefix(f"{APP_SCHEMA}.{RUNS_STAGE}", run_id)
    checkpoint, _, _ = select_checkpoint(
        collect(session, f"LIST {prefix}/"), run_id, step
    )
    path = f"{run_id}/{checkpoint}/"
    copied = collect(
        session,
        f"COPY FILES INTO @reference({string_literal(EXPORT_REFERENCE)})/{path} "
        f"FROM {prefix}/{checkpoint}/merged/",
    )
    return "\n".join(
        [f"checkpoint: {checkpoint}", f"path: {path}", f"files: {len(copied)}"]
    )


def reconcile_usage(session: SnowflakeSession) -> str:
    """Collect a linked claim's key, stop everything after an unpurchased trial, meter and bill, rotate an old key, suspend idle GPU pools and an idle task, and log at INFO.

    :param session: Procedure session.
    :return: What happened to a claim, the :func:`end_trial`,
        :func:`meter_deployments` and :func:`reconcile` summaries, or why
        nothing ran, then the GPU pools and billing task it suspended.
    """
    claim = collect_linked_key(session)
    summary = reconcile_summary(session)
    if claim is not None:
        summary = f"{claim}; {summary}"
    partner = saved_partner()
    if partner is not None:
        rotate_if_due(session, partner)
    pools = suspend_idle_gpu_pools(session)
    if pools:
        summary += f"; suspended idle GPU pools {', '.join(pools)}"
    if not sync_billing_task(session):
        summary += f"; nothing left to bill, suspended {BILLING_TASK}"
    logger.info("reconcile_usage: %s", summary)
    return summary


def version_init(session: SnowflakeSession) -> str:
    """Point ``setup.json`` at this version's image, and resume the billing task only when there is work to bill.

    Snowflake calls this after install and upgrade, and the previous version's
    after a failed upgrade, whose setup script may already have suspended the
    task. A version accepts only its own image, so ``submit`` and ``publish``
    would refuse the image an earlier version saved.

    :param session: Procedure session.
    :return: What was done to the image and the task.
    """
    lines = []
    # LIST and GET need no warehouse; an upgrade Snowflake starts has none.
    setup_json = f"{SETUP_PREFIX}/setup.json"
    profile = (
        profile_from_mapping(json.loads(session.file.get_stream(setup_json).read()))
        if collect(session, f"LIST {setup_json}")
        else None
    )
    if profile is not None and profile.image != IMAGE:
        put_setup_json(session, replace(profile, image=IMAGE), SETUP_PREFIX)
        lines.append(f"image set to {IMAGE}")
    if sync_billing_task(session):
        lines.append(f"resumed {BILLING_TASK}")
    else:
        lines.append(f"nothing to bill, suspended {BILLING_TASK}")
    return "; ".join(lines)


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
    :param serve: Start a vLLM service after logging the model, with no
        endpoint outside Snowflake, and a SQL function ``app_user`` calls it with.
    :param gpu_count: GPUs the service requests.
    :param service_name: Service name. ``None`` uses ``<model>_<version>``.
    :param max_model_len: vLLM ``--max-model-len``, or ``None``.
    :return: The job, model, version, and any service, its chat function, and
        its Arena deployment.
    :raises ValueError: An argument is invalid, Arena is not configured for a
        served model, the app has no such run, or the run has no usable
        checkpoint.
    :raises RuntimeError: The publish job failed or outlived the wait. A
        service it starts later gets its deployment from the hourly task.
    :raises PermissionError: The trial has ended and the listing is not purchased.
    :raises SnowflakeLicenseError: Arena did not open the deployment. The
        service was dropped.
    """
    if gpu_count < 1:
        msg = "gpu_count must be at least 1.\nNothing was published."
        raise ValueError(msg)
    refuse_ended_trial(session)
    partner = rotate_if_due(session, arena_partner()) if serve else None
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
            ingress=False,
            compute_pool=sole_gpu_pool(profile),
            head_pool=profile.pools.head.pool,
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
        collect(session, f"ALTER TASK {BILLING_TASK} RESUME")
    wait_for_publish(session, job)
    lines = [
        f"job: {job.name}",
        f"model: {document['database']}.{document['schema']}.{document['model_name']}",
        f"version: {document['version_name']}",
    ]
    if served is not None and partner is not None:
        base_url, headers = partner
        try:
            grant_chat_function(session, served.service)
            deployment_id = open_served(
                session,
                served,
                lambda deployment: open_snowflake_deployment(
                    deployment, base_url, headers, time.sleep
                ),
                datetime.now(timezone.utc),
            )
        except Exception:
            drop_served_service(session, served.service)
            raise
        function = chat_function(f"{database}.{schema}.{served.service}")
        lines.append(f"service: {service}")
        lines.append(f"function: {function}")
        lines.append(f"deployment: {deployment_id}")
    return "\n".join(lines)


def endpoint(session: SnowflakeSession, model: str, version_name: str | None) -> str:
    """Name the SQL function that calls a published version's served model, with an example.

    The app creates the function once the service is up, at the end of
    ``publish`` or on the next hourly task run.

    :param session: Procedure session.
    :param model: Model Registry model name in ``app_state``.
    :param version_name: Version. ``None`` takes the one version that has a service.
    :return: The function's name and a ``SELECT`` that calls it.
    :raises ValueError: The model, version, or service is not there.
    """
    database, schema = app_registry(session)
    served = served_version(session, database, schema, model, version_name)
    function = chat_function(version_service(served))
    return f"function: {function}\nexample:\n{chat_example(function)}"


def unpublish(
    session: SnowflakeSession, model: str, version_name: str | None, drop_version: bool
) -> str:
    """Drop the inference service of a published version and, when asked, the version.

    Then drops the chat function of each dropped service and closes its Arena
    deployment.

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
    lines = []
    for service in served.services:
        collect(session, drop_chat_function_sql(service))
        lines.append(f"dropped service: {service}")
        lines.append(f"dropped function: {chat_function(service)}")
    if drop_version and served.only_version:
        lines.append(f"dropped model: {served.model}")
    elif drop_version:
        lines.append(f"dropped version: {served.version_name}")
    lines.extend(
        f"closed deployment: {deployment_id}"
        for deployment_id in close_deployments(session, served.services)
    )
    return "\n".join(lines)
