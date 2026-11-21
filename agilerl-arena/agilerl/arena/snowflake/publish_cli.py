# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena snowflake publish``, ``endpoint``, and ``unpublish`` commands."""

from __future__ import annotations

from contextlib import closing
from typing import cast

import click

from agilerl.arena.snowflake.catalog import remote_profile
from agilerl.arena.snowflake.profile import (
    HEAD_POOL,
    checked_job_connection,
    sole_gpu_pool,
)
from agilerl.arena.snowflake.publish import (
    PublishJob,
    PublishSession,
    PublishSettings,
    registry_schema,
    served_version,
    service_endpoint,
    submit_publish,
    unpublish,
    version_endpoint,
    wait_for_publish,
)
from agilerl.arena.snowflake.session import open_session
from agilerl.arena.snowflake.source import SnowflakeSession


@click.command("publish")
@click.argument("run_id")
@click.option(
    "--runs-stage",
    default=None,
    help="Stage that holds the run. Default: the account profile in setup.json. "
    "Pass it when the account has no setup.json.",
)
@click.option("--model-name", required=True, help="Model Registry model name.")
@click.option(
    "--version",
    "version_name",
    default=None,
    help="Model Registry version. Defaults to the checkpoint directory name.",
)
@click.option(
    "--step",
    type=click.IntRange(min=0),
    default=None,
    help="Checkpoint step. Omit to use the newest complete merged checkpoint.",
)
@click.option(
    "--serve/--no-serve",
    default=False,
    show_default=True,
    help="Start a vLLM service after logging the model. Its image builds on the "
    f"account profile's head pool, or {HEAD_POOL} without a profile.",
)
@click.option(
    "--compute-pool",
    default=None,
    help="Pool for the publish job and the service. Default: the account "
    "profile's GPU pool when it has one GPU family, else the run's pool.",
)
@click.option(
    "--image",
    default=None,
    help="Job image. Default: the account profile's image, else the run's.",
)
@click.option(
    "--warehouse",
    default=None,
    help="Warehouse for the job's queries. Default: the account profile's, "
    "else the run's.",
)
@click.option(
    "--gpu-count",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="GPUs the service requests. Above 1, also sets vLLM --tensor-parallel-size.",
)
@click.option(
    "--service-name",
    default=None,
    help="Service name. Defaults to <model-name>_<version>. Must not exist.",
)
@click.option(
    "--max-model-len",
    type=click.IntRange(min=1),
    default=None,
    help="vLLM --max-model-len for the service.",
)
@click.option(
    "--database",
    default=None,
    help="Registry database. Pass with --schema. Defaults to the session's.",
)
@click.option(
    "--schema",
    default=None,
    help="Registry schema. Pass with --database. Defaults to the session's.",
)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help="Connection name in ~/.snowflake/connections.toml. "
    "Default: arena when that section exists.",
)
def publish(
    run_id: str,
    runs_stage: str | None,
    model_name: str,
    version_name: str | None,
    step: int | None,
    serve: bool,
    compute_pool: str | None,
    image: str | None,
    warehouse: str | None,
    gpu_count: int,
    service_name: str | None,
    max_model_len: int | None,
    database: str | None,
    schema: str | None,
    connection_name: str | None,
) -> None:
    """Log RUN_ID's merged checkpoint to the Model Registry from a job in the account."""
    try:
        name = checked_job_connection(connection_name)
        # Snowpark sessions also provide file.get_stream.
        with closing(cast("PublishSession", open_session(name))) as session:
            profile = remote_profile(session)
            image_build_pool = HEAD_POOL
            if profile is not None:
                runs_stage = runs_stage or profile.stage
                compute_pool = compute_pool or sole_gpu_pool(profile)
                image = image or profile.image or None
                warehouse = warehouse or profile.warehouse
                image_build_pool = profile.pools.head.pool
            if runs_stage is None:
                msg = "This account has no setup.json. Pass --runs-stage."
                raise click.UsageError(msg)
            settings = PublishSettings(
                run_id=run_id,
                runs_stage=runs_stage,
                model_name=model_name,
                version_name=version_name,
                step=step,
                serve=serve,
                compute_pool=compute_pool,
                image_build_pool=image_build_pool,
                image=image,
                warehouse=warehouse,
                gpu_count=gpu_count,
                max_model_len=max_model_len,
                service_name=service_name,
                database=database,
                schema=schema,
            )
            job = submit_publish(session, settings)
            click.echo(f"job: {job.name}")
            wait_for_publish(session, job, echo=click.echo)
            echo_published(session, job)
    except (ValueError, RuntimeError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command("endpoint")
@click.argument("model")
@click.option(
    "--version",
    "version_name",
    default=None,
    help="Model Registry version. Default: the one version that has a service.",
)
@click.option(
    "--database",
    default=None,
    help="Registry database. Pass with --schema. Defaults to the session's.",
)
@click.option(
    "--schema",
    default=None,
    help="Registry schema. Pass with --database. Defaults to the session's.",
)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help="Connection name in ~/.snowflake/connections.toml. "
    "Default: arena when that section exists.",
)
def endpoint_command(
    model: str,
    version_name: str | None,
    database: str | None,
    schema: str | None,
    connection_name: str | None,
) -> None:
    """Print the ingress URL of MODEL's inference service."""
    try:
        name = checked_job_connection(connection_name)
        with closing(open_session(name)) as session:
            database, schema = registry_schema(session, database, schema)
            served = served_version(session, database, schema, model, version_name)
            click.echo(version_endpoint(session, served))
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command("unpublish")
@click.argument("model")
@click.option(
    "--version",
    "version_name",
    default=None,
    help="Model Registry version. Default: the one version that has a service.",
)
@click.option(
    "--drop-version",
    is_flag=True,
    default=False,
    help="Also drop the version. A model's only version is dropped with the model.",
)
@click.option(
    "--database",
    default=None,
    help="Registry database. Pass with --schema. Defaults to the session's.",
)
@click.option(
    "--schema",
    default=None,
    help="Registry schema. Pass with --database. Defaults to the session's.",
)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help="Connection name in ~/.snowflake/connections.toml. "
    "Default: arena when that section exists.",
)
def unpublish_command(
    model: str,
    version_name: str | None,
    drop_version: bool,
    database: str | None,
    schema: str | None,
    connection_name: str | None,
) -> None:
    """Drop the inference service of MODEL's version, and optionally the version."""
    try:
        name = checked_job_connection(connection_name)
        with closing(open_session(name)) as session:
            database, schema = registry_schema(session, database, schema)
            served = served_version(session, database, schema, model, version_name)
            unpublish(session, served, drop_version)
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    for service in served.services:
        click.echo(f"dropped service: {service}")
    if drop_version and served.only_version:
        click.echo(f"dropped model: {served.model}")
    elif drop_version:
        click.echo(f"dropped version: {served.version_name}")


def echo_published(session: SnowflakeSession, job: PublishJob) -> None:
    """Print the model, version, and any service and endpoint of a finished job.

    :param session: Snowpark session.
    :param job: The finished publish job.
    """
    document = job.document
    click.echo(
        f"model: {document['database']}.{document['schema']}.{document['model_name']}"
    )
    click.echo(f"version: {document['version_name']}")
    if document["service_name"] is not None:
        click.echo(f"service: {document['service_name']}")
        click.echo(f"endpoint: {service_endpoint(session, document)}")
