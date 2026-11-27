# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Billing for the Arena Snowflake Native App: the billing task, idle GPU pools, and served models.

``arena_app`` and ``app_keys`` import this module, and this module imports
``app_trial``. The setup script lists all four files in each procedure's
``IMPORTS``, which puts them on the path side by side.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app_trial import end_trial, trial_ended
from snowflake.snowpark.exceptions import SnowparkSQLException

from agilerl.snowflake.billing import reconcile, training_started
from agilerl.snowflake.catalog import collect, whole_number
from agilerl.snowflake.job import lowered
from agilerl.snowflake.license import (
    DeploymentClosedError,
    DeploymentRefusedError,
    SnowflakeDeployment,
    SnowflakeLicenseError,
    SnowflakeUsage,
    TrainingDisabledError,
    UnknownUsageError,
    close_snowflake_deployment,
    heartbeat_snowflake_deployment,
    open_snowflake_deployment,
    partner_headers,
    snowflake_usage,
)
from agilerl.snowflake.profile import METRICS_TABLE
from agilerl.snowflake.publish import (
    create_chat_function_sql,
    drop_chat_function_sql,
    grant_chat_function_sql,
    grant_service_function_sql,
)
from agilerl.snowflake.setup_sql import string_literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from agilerl.snowflake.profile import AccountProfile
    from agilerl.snowflake.publish import PublishDocument
    from agilerl.snowflake.source import SnowflakeSession

APP_SCHEMA = "app_state"
BILLING_TASK = f"{APP_SCHEMA}.reconcile_usage_task"

logger = logging.getLogger(__name__)


def saved_secret(name: str) -> str:
    """Read a secret the procedure's ``SECRETS`` clause names.

    :param name: Name in the ``SECRETS`` clause.
    :return: The value.
    """
    # Lazy import of an optional package: _snowflake exists only inside a Snowflake procedure.
    import _snowflake  # noqa: PLC0415 -- not installable outside Snowflake

    return _snowflake.get_generic_secret_string(name)


def saved_partner() -> tuple[str, dict[str, str]] | None:
    """Arena origin and partner headers from the secrets ``configure`` or ``connect_arena`` saved.

    :return: The origin and the headers, or ``None`` when no org key is saved.
    """
    org_key = saved_secret("org_key")
    if not org_key:
        return None
    return saved_secret("base_url"), partner_headers(
        org_key, saved_secret("external_user_id")
    )


def arena_partner() -> tuple[str, dict[str, str]]:
    """Arena origin and partner headers from the saved secrets.

    :return: The origin and the headers.
    :raises ValueError: No org key has been saved.
    """
    partner = saved_partner()
    if partner is None:
        msg = "Arena is not connected. An app_admin calls app_code.connect_arena first."
        raise ValueError(msg)
    return partner


def sync_billing_task(session: SnowflakeSession) -> bool:
    """Suspend the billing task while :func:`app_busy` is false, and resume it otherwise.

    Snowflake charges the listing's monthly fee in a month with a query on the
    app or a running app compute pool, so an idle app runs no task. The check
    repeats after suspending: ``submit`` and ``publish`` record their work
    before they resume the task, so work recorded meanwhile resumes it here.

    :param session: Procedure session.
    :return: Whether the task is resumed.
    """
    if not app_busy(session):
        collect(session, f"ALTER TASK {BILLING_TASK} SUSPEND")
        if not app_busy(session):
            return False
    collect(session, f"ALTER TASK {BILLING_TASK} RESUME")
    return True


def app_busy(session: SnowflakeSession) -> bool:
    """Whether a service in ``app_state`` is live, a usage item still needs billing, or a proven claim waits for its key.

    An item Arena has only answered 404 for does not count: a run that never
    reached Arena would keep the task running for the 30 days until it settles.
    A claim counts only until it expires, so a signup link nobody opens costs
    at most two hours of task runs.

    :param session: Procedure session.
    :return: True while the billing task has work.
    """
    if any(service_live(status) for status in service_statuses(session).values()):
        return True
    [row] = collect(
        session,
        "SELECT COUNT(*) AS N FROM app_state.usage_items WHERE settled_at IS NULL "
        "AND (first_seen_at IS NOT NULL OR first_404_at IS NULL)",
    )
    if int(str(lowered(row)["n"])) > 0:
        return True
    return bool(
        collect(
            session,
            "SELECT claim_id FROM app_state.arena_claim WHERE proven_at IS NOT NULL "
            "AND expires_at > CURRENT_TIMESTAMP()",
        )
    )


def suspend_idle_gpu_pools(session: SnowflakeSession) -> list[str]:
    """Suspend the app's GPU pools that are not suspended and have no service or job.

    A pool still waiting for a node, or idle before ``AUTO_SUSPEND_SECS``,
    counts as running for the listing's monthly fee.

    :param session: Procedure session.
    :return: Names of the pools suspended.
    """
    [row] = collect(session, "SELECT app_code.object_prefix() AS PREFIX")
    app_prefix = str(lowered(row)["prefix"])
    prefix = f"{app_prefix}GPU_"
    suspended = []
    for pool in collect(
        session, f"SHOW COMPUTE POOLS LIKE {string_literal(prefix + '%')}"
    ):
        values = lowered(pool)
        name = str(values["name"]).upper()
        if (
            not name.startswith(prefix.upper())
            or str(values.get("state") or "").upper() == "SUSPENDED"
            or whole_number(values.get("num_services"))
            or whole_number(values.get("num_jobs"))
        ):
            continue
        collect(session, f"ALTER COMPUTE POOL {name} SUSPEND")
        suspended.append(name)
    return suspended


def reconcile_summary(session: SnowflakeSession) -> str:
    """Drop every service once the trial has ended unpurchased, meter served models, then bill.

    :param session: Procedure session.
    :return: The :func:`end_trial`, :func:`meter_deployments` and
        :func:`reconcile` summaries, or why nothing ran.
    """
    try:
        base_url, headers = arena_partner()
    except ValueError as exc:
        return str(exc)
    ended = None
    if trial_ended(session):
        served = served_models(session)
        services = sorted(
            name
            for name, status in service_statuses(session).items()
            if service_live(status) or name in served
        )
        for name in served:
            collect(session, drop_chat_function_sql(f"{APP_SCHEMA}.{name}"))
        ended = end_trial(session, services)

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

    def trained(run_id: str) -> bool:
        return training_started(session, f"{APP_SCHEMA}.{METRICS_TABLE}", run_id)

    billed = reconcile(session, fetch, trained, int(time.time() * 1000))
    summary = f"{metered}; {billed}"
    return summary if ended is None else f"{ended}; {summary}"


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


def service_callable(status: str | None) -> bool:
    """Whether SQL can call a service: live, or suspended until its next call resumes it.

    :param status: Upper-case ``SHOW SERVICES`` status, or ``None`` when the service is gone.
    :return: True for a live, suspending or suspended service.
    """
    return service_live(status) or status in {"SUSPENDING", "SUSPENDED"}


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
    dropped while metering its open deployment gets no chat function. Every
    other live or suspended service does, including one already metered,
    since its first SQL call resumes it. A 404 is
    logged and the deployment is left. A deployment Arena has never answered
    for does not count its service as metered, so its open is sent again.
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
        if service in dropped or not service_callable(statuses.get(service)):
            continue
        grant_chat_function(session, service)
        if service in metered or not service_live(statuses.get(service)):
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
        drop_served_service(session, service)
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


def grant_chat_function(session: SnowflakeSession, service: str) -> None:
    """Create the chat function of a live service and let ``app_user`` call it.

    The service has no endpoint outside Snowflake, so SQL is the only way in.
    ``app_user`` also gets the service's own ``__CALL__`` method, which takes
    every chat option.

    :param session: Procedure session.
    :param service: Service in ``app_state``.
    """
    qualified = f"{APP_SCHEMA}.{service}"
    grantee = "APPLICATION ROLE app_user"
    collect(session, create_chat_function_sql(qualified))
    collect(session, grant_chat_function_sql(qualified, grantee))
    collect(session, grant_service_function_sql(qualified, grantee))


def drop_served_service(session: SnowflakeSession, service: str) -> None:
    """Drop a served model's chat function and its service.

    :param session: Procedure session.
    :param service: Service in ``app_state``.
    """
    collect(session, drop_chat_function_sql(f"{APP_SCHEMA}.{service}"))
    collect(session, f"DROP SERVICE IF EXISTS {APP_SCHEMA}.{service}")


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
    """Open the Arena deployment of a live service; a refused open drops the service.

    The deployment is added to ``app_state.usage_items`` before Arena opens it,
    so the hourly task bills an open whose answer was lost. A service that
    already has an open deployment keeps it, and Arena is sent the same open
    again, so ``publish`` and the hourly task never give one service two.

    :param session: Procedure session.
    :param served: What the service serves.
    :param open_call: Arena open of a deployment.
    :param now: Current time, timezone-aware: the start of a new deployment.
    :return: The deployment id.
    :raises DeploymentRefusedError: Arena refused the open. The service was
        dropped and the deployment marked closed with Arena's reason.
    :raises SnowflakeLicenseError: The open did not land.
    """
    # Whole seconds, so a start read back from opened_at repeats the open exactly.
    started_at = now.replace(microsecond=0)
    deployment_id = deployment_name(served.service, started_at)
    service = string_literal(served.service)
    # MERGE locks the table, so a concurrent caller sees this row and reuses it.
    [merged] = collect(
        session,
        f"MERGE INTO app_state.usage_items AS t USING (SELECT {service} AS "
        "service_name) AS s ON t.kind = 'deployment' AND t.service_name = "
        "s.service_name AND t.closed_at IS NULL AND t.settled_at IS NULL "
        "WHEN NOT MATCHED THEN INSERT (kind, usage_id, opened_at, service_name) "
        f"VALUES ('deployment', {string_literal(deployment_id)}, "
        f"TO_TIMESTAMP_TZ({int(started_at.timestamp() * 1000)}, 3), s.service_name)",
    )
    if int(str(lowered(merged)["number of rows inserted"])) == 0:
        existing = lowered(
            collect(
                session,
                "SELECT usage_id, DATE_PART(EPOCH_MILLISECOND, opened_at) AS "
                "OPENED_MS FROM app_state.usage_items WHERE kind = 'deployment' "
                f"AND service_name = {service} AND closed_at IS NULL AND "
                "settled_at IS NULL ORDER BY opened_at",
            )[0]
        )
        deployment_id = str(existing["usage_id"])
        started_at = datetime.fromtimestamp(
            int(str(existing["opened_ms"])) / 1000, timezone.utc
        )
    [row] = collect(
        session,
        "SELECT CURRENT_ACCOUNT() AS ACCOUNT, CURRENT_REGION() AS REGION, "
        "app_code.listing_purchased() AS PURCHASED",
    )
    account = lowered(row)
    try:
        open_call(
            SnowflakeDeployment(
                deployment_id=deployment_id,
                snowflake_account=str(account["account"]),
                snowflake_region=str(account["region"]),
                gpu_count=served.gpu_count,
                gpu_family=served.gpu_family,
                started_at=started_at,
                model=served.model,
                version=served.version,
                run_id=served.run_id,
                purchased=account["purchased"] is True,
            )
        )
    except DeploymentRefusedError as exc:
        drop_served_service(session, served.service)
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
