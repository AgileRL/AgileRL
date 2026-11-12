# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena snowflake`` commands."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import click

from agilerl.arena.auth import (
    ArenaOAuth2,
    load_credentials,
    validate_partner_credentials,
)
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAuthError
from agilerl.arena.snowflake.catalog import NO_SETUP, remote_profile
from agilerl.arena.snowflake.images import BUNDLE_STAMP, check_image_matches_cli
from agilerl.arena.snowflake.job import (
    ROLES,
    SubmitSettings,
    cancel_run,
    choose_gpu_family,
    event_log_text,
    format_run_status,
    job_services_sql,
    log_text,
    manifest_fits,
    qualified_table,
    role_states,
    service_event_logs_sql,
    service_for_role,
    service_logs_sql,
    service_status_sql,
    submit_job,
    write_job_manifest,
)
from agilerl.arena.snowflake.license import (
    DEFAULT_BASE_URL,
    SNOWFLAKE_DENIED,
    SnowflakeLicenseError,
    fetch_capabilities,
    partner_headers,
    snowflake_training_enabled,
)
from agilerl.arena.snowflake.profile import (
    NO_IMAGE,
    AccountProfile,
    checked_job_connection,
    gpu_pool_name,
    job_connection_name,
)
from agilerl.arena.snowflake.publish_cli import (
    endpoint_command,
    publish,
    unpublish_command,
)
from agilerl.arena.snowflake.session import open_session, sql_errors
from agilerl.arena.snowflake.setup_cli import (
    check,
    grant_command,
    invite_command,
    join_command,
    pools_group,
    setup,
    teardown_command,
)
from agilerl.arena.snowflake.source import (
    QuerySource,
    SnowflakeRow,
    SnowflakeSession,
    TableSource,
)


@click.group("snowflake")
def snowflake_group() -> None:
    """Train inside a Snowflake account."""


snowflake_group.add_command(publish)
snowflake_group.add_command(endpoint_command)
snowflake_group.add_command(unpublish_command)
snowflake_group.add_command(setup)
snowflake_group.add_command(check)
snowflake_group.add_command(invite_command)
snowflake_group.add_command(join_command)
snowflake_group.add_command(pools_group)
snowflake_group.add_command(grant_command)
snowflake_group.add_command(teardown_command)


@snowflake_group.command("submit")
@click.argument(
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--source-table",
    default=None,
    help="Table the job reads, as database.schema.table. "
    "The manifest needs no dataset. Not used by an env-backed rollout.",
)
@click.option(
    "--source-query",
    default=None,
    help="Query the job reads. Not used by an env-backed rollout.",
)
@click.option(
    "--image",
    default=None,
    help="Image path in the account, /db/schema/repo/snowflake-job:tag. "
    "Default: the account profile.",
)
@click.option(
    "--warehouse",
    default=None,
    help="Warehouse for queries the job runs. Default: the account profile.",
)
@click.option(
    "--runs-stage",
    default=None,
    help="Stage that receives checkpoints. Default: the account profile.",
)
@click.option(
    "--metrics-table",
    default=None,
    help="Table that receives metric rows. Default: the account profile.",
)
@click.option(
    "--gpus",
    required=True,
    type=click.IntRange(min=1),
    help="GPUs for the run. Workers are this count divided by the family's "
    "GPUs per node.",
)
@click.option(
    "--gpu-family",
    default=None,
    help="GPU instance family. Default: the smallest GPU memory whose GPUs "
    "per node divide --gpus, on the fewest nodes.",
)
@click.option(
    "--eai",
    "external_access_integrations",
    multiple=True,
    help="External access integration added to the account profile's. "
    "Repeat for each one.",
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
    help="Python reward function. Required for RL over a table or query; "
    "the manifest needs no rubric_file_path.",
)
@click.option(
    "--resume",
    "resume_from",
    default=None,
    metavar="RUN_ID",
    help="Start from the newest finished checkpoint of RUN_ID.",
)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help="Connection name in ~/.snowflake/connections.toml. "
    "Default: arena when that section exists.",
)
def submit_command(
    manifest: Path,
    source_table: str | None,
    source_query: str | None,
    image: str | None,
    warehouse: str | None,
    runs_stage: str | None,
    metrics_table: str | None,
    gpus: int,
    gpu_family: str | None,
    external_access_integrations: tuple[str, ...],
    checkpoint_steps: int | None,
    reward_file: Path | None,
    resume_from: str | None,
    connection_name: str | None,
) -> None:
    """Submit MANIFEST as broker, head, and worker job services."""
    if source_table is not None and source_query is not None:
        msg = "Pass at most one of --source-table or --source-query."
        raise click.UsageError(msg)
    with snowflake_errors(), tempfile.TemporaryDirectory() as tmp:
        source = None
        if source_table is not None:
            source = TableSource(qualified_table(source_table))
        if source_query is not None:
            source = QuerySource(source_query)
        name = checked_job_connection(connection_name)
        job_manifest = write_job_manifest(
            manifest, source, reward_file is not None, Path(tmp)
        )
        _require_snowflake_training()
        with closing(open_session(name)) as session:
            settings = _submit_settings(
                remote_profile(session),
                image,
                warehouse,
                runs_stage,
                metrics_table,
                source,
                gpus,
                gpu_family,
                job_manifest,
                external_access_integrations,
                checkpoint_steps,
                reward_file,
                resume_from,
            )
            run_id, services = submit_job(session, job_manifest, settings)
    click.echo(f"run_id: {run_id}")
    click.echo(f"broker: {services.broker}")
    click.echo(f"head: {services.head}")
    click.echo(f"workers: {services.workers}")


@snowflake_group.command("status")
@click.argument("run_id")
@click.option("--connection", "connection_name", default=None)
def status_command(run_id: str, connection_name: str | None) -> None:
    """Print the broker, head, and worker states, their containers, and the run state."""
    with (
        snowflake_errors(),
        closing(open_session(job_connection_name(connection_name))) as session,
    ):
        click.echo(run_status(session, run_id))


@snowflake_group.command("logs")
@click.argument("run_id")
@click.option(
    "--role",
    type=click.Choice(list(ROLES)),
    default="head",
    show_default=True,
)
@click.option("--instance", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--connection", "connection_name", default=None)
def logs_command(
    run_id: str, role: str, instance: int, connection_name: str | None
) -> None:
    """Print logs from the main container of one role."""
    with (
        snowflake_errors(),
        closing(open_session(job_connection_name(connection_name))) as session,
    ):
        click.echo(service_logs(session, run_id, instance, role))


@snowflake_group.command("cancel")
@click.argument("run_id")
@click.option("--connection", "connection_name", default=None)
def cancel_command(run_id: str, connection_name: str | None) -> None:
    """Drop the broker, head, and workers for RUN_ID."""
    with (
        snowflake_errors(),
        closing(open_session(job_connection_name(connection_name))) as session,
    ):
        cancel_run(session, run_id)
    click.echo(f"cancelled: {run_id}")


@contextmanager
def snowflake_errors() -> Iterator[None]:
    """Show submit, licence, and Snowflake SQL errors as one ``ClickException``.

    A SQL error prints on one line.
    """
    try:
        yield
    except (ValueError, ImportError, SnowflakeLicenseError) as exc:
        raise click.ClickException(str(exc)) from exc
    except sql_errors() as exc:
        raise click.ClickException(" ".join(str(exc).split())) from exc


def _require_snowflake_training() -> None:
    """Stop submit when local Arena credentials say Snowflake training is off.

    Without local credentials the check is skipped. The head checks the
    licence when it starts.

    :raises SnowflakeLicenseError: Arena cannot be read, or the flag is not true.
    """
    arena = _arena_headers()
    if arena is None:
        return
    base_url, headers = arena
    if not snowflake_training_enabled(fetch_capabilities(base_url, headers)):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)


def _arena_headers() -> tuple[str, dict[str, str]] | None:
    config: CommandConfig = click.get_current_context().find_root().obj
    base_url = config.base_url or os.environ.get("ARENA_BASE_URL") or DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    org_key = config.org_key or os.environ.get("ARENA_ORG_KEY")
    external_user_id = config.external_user_id or os.environ.get(
        "ARENA_EXTERNAL_USER_ID"
    )
    try:
        validate_partner_credentials(org_key, external_user_id)
    except ArenaAuthError as exc:
        raise SnowflakeLicenseError(str(exc)) from exc
    if org_key and external_user_id:
        return base_url, partner_headers(org_key, external_user_id)
    api_key = config.api_key or os.environ.get("ARENA_API_KEY")
    if api_key:
        return base_url, {"Authorization": f"Bearer {api_key}"}
    stored = load_credentials(ArenaOAuth2.CREDENTIALS_FILE)
    token = None if stored is None else stored.get("access_token")
    if isinstance(token, str) and token != "":
        return base_url, {"Authorization": f"Bearer {token}"}
    return None


def _submit_settings(
    profile: AccountProfile | None,
    image: str | None,
    warehouse: str | None,
    runs_stage: str | None,
    metrics_table: str | None,
    source: TableSource | QuerySource | None,
    gpus: int,
    gpu_family: str | None,
    manifest: Path,
    external_access_integrations: tuple[str, ...],
    checkpoint_steps: int | None,
    reward_file: Path | None,
    resume_from: str | None,
) -> SubmitSettings:
    if profile is None:
        raise ValueError(NO_SETUP)
    if image is None:
        if profile.image == "":
            raise ValueError(NO_IMAGE)
        check_image_matches_cli(profile.image, BUNDLE_STAMP)
        image = profile.image
    if warehouse is None:
        warehouse = profile.warehouse
    if runs_stage is None:
        runs_stage = profile.stage
    if metrics_table is None:
        metrics_table = profile.metrics_table
    integrations = tuple(
        dict.fromkeys(
            (*profile.external_access_integrations, *external_access_integrations)
        )
    )
    family = choose_gpu_family(
        profile.pools.gpu,
        gpus,
        gpu_family,
        lambda pool: manifest_fits(manifest, pool, gpus),
    )
    pool = profile.pools.gpu[family]
    return SubmitSettings(
        image=image,
        warehouse=warehouse,
        runs_stage=runs_stage,
        metrics_table=metrics_table,
        source=source,
        external_access_integrations=integrations,
        checkpoint_steps=checkpoint_steps,
        reward_file=reward_file,
        gpus=gpus,
        gpu_family=family,
        gpus_per_node=pool.gpus_per_node,
        gpu_pool=gpu_pool_name(family),
        broker_pool=profile.pools.broker.pool,
        broker_family=profile.pools.broker.family,
        head_pool=profile.pools.head.pool,
        head_family=profile.pools.head.family,
        resume_from=resume_from,
    )


def run_status(session: SnowflakeSession, run_id: str) -> str:
    """State of each role, its containers, and the run state.

    :param session: Snowpark session.
    :param run_id: Run id.
    :return: The formatted lines.
    """
    services_sql = job_services_sql(run_id)
    states = role_states(run_id, list(session.sql(services_sql).collect()))
    containers: dict[str, list[SnowflakeRow]] = {}
    for role, state in states.items():
        containers_sql = service_status_sql(service_for_role(run_id, role))
        containers[role] = (
            [] if state == "gone" else list(session.sql(containers_sql).collect())
        )
    return format_run_status(states, containers)


def service_logs(
    session: SnowflakeSession, run_id: str, instance: int, role: str = "head"
) -> str:
    """Logs of the main container of one role.

    A running service reads ``SYSTEM$GET_SERVICE_LOGS``. A finished one reads
    the event table, since its node may already be released.

    :param session: Snowpark session.
    :param run_id: Run id.
    :param instance: Instance id. ``0`` is the first instance.
    :param role: ``broker``, ``head``, or ``worker``.
    :return: The log text.
    :raises ValueError: The service was dropped.
    """
    service = service_for_role(run_id, role)
    rows = list(session.sql(job_services_sql(run_id)).collect())
    state = role_states(run_id, rows)[role]
    if state == "running":
        live = session.sql(service_logs_sql(service, instance)).collect()
        return log_text(list(live))
    if state == "gone":
        msg = f"{service} was dropped, so its logs cannot be read."
        raise ValueError(msg)
    stored = session.sql(service_event_logs_sql(service, instance)).collect()
    return event_log_text(list(stored))
