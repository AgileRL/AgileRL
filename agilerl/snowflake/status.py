# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Job service states and container logs for a Snowflake run."""

from __future__ import annotations

from agilerl.snowflake.job import (
    event_log_text,
    format_run_status,
    in_schema,
    job_services_sql,
    log_text,
    lowered,
    role_states,
    service_event_logs_sql,
    service_for_role,
    service_logs_sql,
    service_status_sql,
)
from agilerl.snowflake.source import SnowflakeRow, SnowflakeSession


def run_status(
    session: SnowflakeSession, run_id: str, schema: str | None = None
) -> str:
    """State of each role, its containers, and the run state.

    :param session: Snowpark session.
    :param run_id: Run id.
    :param schema: Schema of the services, or ``None`` for the session's current one.
    :return: The formatted lines.
    """
    services_sql = job_services_sql(run_id, schema)
    states = role_states(run_id, list(session.sql(services_sql).collect()))
    containers: dict[str, list[SnowflakeRow]] = {}
    for role, state in states.items():
        containers_sql = service_status_sql(
            in_schema(service_for_role(run_id, role), schema)
        )
        containers[role] = (
            [] if state == "gone" else list(session.sql(containers_sql).collect())
        )
    return format_run_status(states, containers)


def service_logs(
    session: SnowflakeSession,
    run_id: str,
    instance: int,
    role: str = "head",
    schema: str | None = None,
    *,
    status_command: str,
) -> str:
    """Logs of the main container of one role.

    A running service reads ``SYSTEM$GET_SERVICE_LOGS`` once its container has
    started. A finished one reads the event table, since its node may already
    be released.

    :param session: Snowpark session.
    :param run_id: Run id.
    :param instance: Instance id. ``0`` is the first instance.
    :param role: ``broker``, ``head``, or ``worker``.
    :param schema: Schema of the services, or ``None`` for the session's current one.
    :param status_command: Command that shows the run's status, named while
        the container has not started.
    :return: The log text, or a line saying the container has not started or
        has written no logs yet.
    :raises ValueError: The service was dropped.
    """
    service = in_schema(service_for_role(run_id, role), schema)
    rows = list(session.sql(job_services_sql(run_id, schema)).collect())
    state = role_states(run_id, rows)[role]
    if state == "running":
        containers = list(session.sql(service_status_sql(service)).collect())
        if not container_started(containers, instance):
            return (
                f"The {role} container has not started yet (state: PENDING). "
                "Its node can take several minutes to start; GPU nodes take longest. "
                f"Check progress with {status_command}."
            )
        live = session.sql(service_logs_sql(service, instance)).collect()
        return log_text(list(live)) or (
            f"The {role} container has started but has written no logs yet. "
            "Check again in a minute."
        )
    if state == "gone":
        msg = f"{service} was dropped, so its logs cannot be read."
        raise ValueError(msg)
    stored = session.sql(service_event_logs_sql(service, instance)).collect()
    return event_log_text(list(stored))


def container_started(containers: list[SnowflakeRow], instance: int) -> bool:
    """Whether a container instance has left ``PENDING``.

    :param containers: ``SHOW SERVICE CONTAINERS`` rows. A pending service
        reports one row with no instance.
    :param instance: Instance id.
    :return: False while the instance is unlisted or ``PENDING``.
    """
    return any(
        values.get("instance_id") is not None
        and int(values["instance_id"]) == instance
        and str(values.get("status", "")).upper() != "PENDING"
        for values in map(lowered, containers)
    )
