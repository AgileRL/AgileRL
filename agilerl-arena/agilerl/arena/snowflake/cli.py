# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena snowflake`` commands."""

from __future__ import annotations

from pathlib import Path

import click

from agilerl.arena.snowflake.job import SubmitSettings, submit_job
from agilerl.arena.snowflake.session import open_session
from agilerl.arena.snowflake.source import QuerySource, TableSource


@click.group("snowflake")
def snowflake_group() -> None:
    """Train inside a Snowflake account."""


@snowflake_group.command("submit")
@click.argument(
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--source-table", default=None, help="Table the job reads.")
@click.option("--source-query", default=None, help="Query the job reads.")
@click.option("--compute-pool", required=True, help="GPU compute pool name.")
@click.option(
    "--runtime-environment",
    required=True,
    help="Custom runtime environment, for example cre@arena_runtime.",
)
@click.option("--payload-stage", required=True, help="Stage that receives the payload.")
@click.option("--runs-stage", required=True, help="Stage that receives checkpoints.")
@click.option("--metrics-table", required=True, help="Table that receives metric rows.")
@click.option("--nodes", default=1, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--eai",
    "external_access_integrations",
    multiple=True,
    help="External access integration. Repeat for each one.",
)
@click.option("--checkpoint-steps", default=None, type=click.IntRange(min=1))
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
    runtime_environment: str,
    payload_stage: str,
    runs_stage: str,
    metrics_table: str,
    nodes: int,
    external_access_integrations: tuple[str, ...],
    checkpoint_steps: int | None,
    pulsar_url: str,
    pulsar_topic_prefix: str,
    connection_name: str | None,
) -> None:
    """Submit MANIFEST as a Snowflake ML Job."""
    if source_table is not None and source_query is None:
        source = TableSource(source_table)
    elif source_query is not None and source_table is None:
        source = QuerySource(source_query)
    else:
        msg = "Pass exactly one of --source-table or --source-query."
        raise click.UsageError(msg)
    settings = SubmitSettings(
        compute_pool=compute_pool,
        runtime_environment=runtime_environment,
        payload_stage=payload_stage,
        runs_stage=runs_stage,
        metrics_table=metrics_table,
        source=source,
        nodes=nodes,
        external_access_integrations=external_access_integrations,
        checkpoint_steps=checkpoint_steps,
        pulsar_url=pulsar_url,
        pulsar_topic_prefix=pulsar_topic_prefix,
    )
    try:
        run_id, job_id = submit_job(open_session(connection_name), manifest, settings)
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"run_id: {run_id}")
    click.echo(f"job_id: {job_id}")


@snowflake_group.command("status")
@click.argument("job_id")
@click.option("--connection", "connection_name", default=None)
def status_command(job_id: str, connection_name: str | None) -> None:
    """Print the status of a Snowflake ML Job."""
    try:
        click.echo(ml_job(open_session(connection_name), job_id).status)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


@snowflake_group.command("logs")
@click.argument("job_id")
@click.option("--connection", "connection_name", default=None)
def logs_command(job_id: str, connection_name: str | None) -> None:
    """Print the head-node logs of a Snowflake ML Job."""
    try:
        click.echo(ml_job(open_session(connection_name), job_id).get_logs())
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


def ml_job(session: object, job_id: str) -> object:
    """Load one ML Job handle.

    :param session: Snowpark session.
    :param job_id: Snowflake ML Job id.
    :return: The job handle.
    """
    # Deferred optional extra: snowflake.
    try:
        from snowflake.ml.jobs import MLJob
    except ImportError as exc:
        msg = (
            "Snowflake support needs the snowflake extra: "
            "pip install 'agilerl-arena[snowflake]'"
        )
        raise ImportError(msg) from exc
    return MLJob(job_id, session=session)
