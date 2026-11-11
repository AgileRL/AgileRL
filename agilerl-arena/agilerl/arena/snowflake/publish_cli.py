# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena snowflake publish`` command."""

from __future__ import annotations

from contextlib import closing
from typing import cast

import click

from agilerl.arena.snowflake.publish import (
    PublishSession,
    PublishSettings,
    service_endpoint,
    submit_publish,
    wait_for_publish,
)
from agilerl.arena.snowflake.session import open_session


@click.command("publish")
@click.argument("run_id")
@click.option("--runs-stage", required=True, help="Stage that holds the run.")
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
    help="Start a vLLM service after logging the model.",
)
@click.option(
    "--compute-pool",
    default=None,
    help="Pool for the publish job and the service. Defaults to the run's pool.",
)
@click.option("--image", default=None, help="Job image. Defaults to the run's image.")
@click.option(
    "--warehouse",
    default=None,
    help="Warehouse for the job's queries. Defaults to the run's warehouse.",
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
    help="Connection name in ~/.snowflake/connections.toml.",
)
def publish(
    run_id: str,
    runs_stage: str,
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
    settings = PublishSettings(
        run_id=run_id,
        runs_stage=runs_stage,
        model_name=model_name,
        version_name=version_name,
        step=step,
        serve=serve,
        compute_pool=compute_pool,
        image=image,
        warehouse=warehouse,
        gpu_count=gpu_count,
        max_model_len=max_model_len,
        service_name=service_name,
        database=database,
        schema=schema,
    )
    try:
        # Snowpark sessions also provide file.get_stream.
        with closing(cast("PublishSession", open_session(connection_name))) as session:
            job = submit_publish(session, settings)
            click.echo(f"job: {job.name}")
            wait_for_publish(session, job.name)
            document = job.document
            click.echo(
                f"model: {document['database']}.{document['schema']}."
                f"{document['model_name']}"
            )
            click.echo(f"version: {document['version_name']}")
            if document["service_name"] is not None:
                click.echo(f"service: {document['service_name']}")
                click.echo(f"endpoint: {service_endpoint(session, document)}")
    except (ValueError, RuntimeError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
