# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena snowflake`` commands."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import click

from agilerl.arena.snowflake.job import (
    SubmitSettings,
    format_status_rows,
    log_text,
    service_logs_sql,
    service_status_sql,
    submit_job,
)
from agilerl.arena.snowflake.session import open_session
from agilerl.arena.snowflake.source import QuerySource, SnowflakeSession, TableSource


@click.group("snowflake")
def snowflake_group() -> None:
    """Train inside a Snowflake account."""


@snowflake_group.command("submit")
@click.argument(
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--source-table",
    default=None,
    help="Table the job reads. Not used by an env-backed rollout.",
)
@click.option(
    "--source-query",
    default=None,
    help="Query the job reads. Not used by an env-backed rollout.",
)
@click.option("--compute-pool", required=True, help="GPU compute pool name.")
@click.option(
    "--image",
    required=True,
    help="Image path in the account, /db/schema/repo/snowflake-job:tag.",
)
@click.option("--warehouse", required=True, help="Warehouse for queries the job runs.")
@click.option("--runs-stage", required=True, help="Stage that receives checkpoints.")
@click.option("--metrics-table", required=True, help="Table that receives metric rows.")
@click.option(
    "--nodes",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help="Job instances. Each takes one node of the compute pool.",
)
@click.option(
    "--eai",
    "external_access_integrations",
    multiple=True,
    help="External access integration. Repeat for each one.",
)
@click.option(
    "--checkpoint-steps",
    default=None,
    type=click.IntRange(min=1),
    help="Training steps between checkpoints. Unset uses max_steps // 10.",
)
@click.option(
    "--reward-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Python reward function. Required for dataset-backed RL manifests.",
)
@click.option("--pulsar-url", required=True, help="Pulsar broker URL for this run.")
@click.option(
    "--pulsar-topic-prefix",
    required=True,
    help="Topic prefix for this run, for example non-persistent://customer/arena.",
)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help="Connection name in ~/.snowflake/connections.toml.",
)
def submit_command(
    manifest: Path,
    source_table: str | None,
    source_query: str | None,
    compute_pool: str,
    image: str,
    warehouse: str,
    runs_stage: str,
    metrics_table: str,
    nodes: int,
    external_access_integrations: tuple[str, ...],
    checkpoint_steps: int | None,
    reward_file: Path | None,
    pulsar_url: str,
    pulsar_topic_prefix: str,
    connection_name: str | None,
) -> None:
    """Submit MANIFEST as a Snowflake job service."""
    if source_table is not None and source_query is not None:
        msg = "Pass at most one of --source-table or --source-query."
        raise click.UsageError(msg)
    source = None
    if source_table is not None:
        source = TableSource(source_table)
    if source_query is not None:
        source = QuerySource(source_query)
    settings = SubmitSettings(
        compute_pool=compute_pool,
        image=image,
        warehouse=warehouse,
        runs_stage=runs_stage,
        metrics_table=metrics_table,
        source=source,
        nodes=nodes,
        external_access_integrations=external_access_integrations,
        checkpoint_steps=checkpoint_steps,
        pulsar_url=pulsar_url,
        pulsar_topic_prefix=pulsar_topic_prefix,
        reward_file=reward_file,
    )
    try:
        with closing(open_session(connection_name)) as session:
            run_id, name = submit_job(session, manifest, settings)
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"run_id: {run_id}")
    click.echo(f"service: {name}")


@snowflake_group.command("status")
@click.argument("service")
@click.option("--connection", "connection_name", default=None)
def status_command(service: str, connection_name: str | None) -> None:
    """Print one row per container, then the job state."""
    try:
        with closing(open_session(connection_name)) as session:
            click.echo(service_status(session, service))
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


@snowflake_group.command("logs")
@click.argument("service")
@click.option("--instance", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--connection", "connection_name", default=None)
def logs_command(service: str, instance: int, connection_name: str | None) -> None:
    """Print logs from the main container of one job instance."""
    try:
        with closing(open_session(connection_name)) as session:
            click.echo(service_logs(session, service, instance))
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


def service_status(session: SnowflakeSession, service: str) -> str:
    """Run ``SHOW SERVICE CONTAINERS`` and format one line per container.

    :param session: Snowpark session.
    :param service: Service name.
    :return: The formatted rows and the job state line.
    """
    statement = service_status_sql(service)
    rows = session.sql(statement).collect()
    return format_status_rows(list(rows))


def service_logs(session: SnowflakeSession, service: str, instance: int) -> str:
    """Run ``SYSTEM$GET_SERVICE_LOGS`` for the main container.

    :param session: Snowpark session.
    :param service: Service name.
    :param instance: Instance id. ``0`` is the first instance.
    :return: The log text.
    """
    statement = service_logs_sql(service, instance)
    rows = session.sql(statement).collect()
    return log_text(list(rows))
