from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import click
from rich.logging import RichHandler

from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig, build_client
from agilerl.arena.output import (
    StreamTableRenderer,
    build_stream_handler,
    emit_csv_preview,
    emit_result,
    handle_error,
)
from agilerl.arena.payloads import load_json_payload, resolve_metrics_output_path

logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
    handlers=[RichHandler(show_time=False, show_path=False, markup=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@contextmanager
def arena_client(
    config: CommandConfig,
) -> Generator[ArenaClient, None, None]:
    """Build an :class:`ArenaClient`, handle errors, and guarantee cleanup."""
    client = build_client(config)
    try:
        yield client
    except Exception as exc:
        handle_error(exc)
    finally:
        client.close()


@contextmanager
def streaming_client(
    config: CommandConfig,
) -> Generator[tuple[ArenaClient, StreamTableRenderer], None, None]:
    """Like :func:`arena_client` but also wires up a streaming event handler.

    Yields ``(client, renderer)``.  Callers should call ``renderer.close()``
    before emitting the final result so the live table is stopped first.
    """
    client = build_client(config)
    handler, renderer = build_stream_handler()
    client.set_stream_handler(handler)
    try:
        yield client, renderer
    except Exception as exc:
        handle_error(exc)
    finally:
        renderer.close()
        client.set_stream_handler(None)
        client.close()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--api-key", default=None, help="Arena API key.")
@click.option(
    "--base-url",
    default=None,
    help="Override Arena API base URL (for local/dev environments).",
)
@click.option(
    "--keycloak-url",
    default=None,
    help="Override Keycloak URL for authentication.",
)
@click.option("--realm", default=None, help="Override Keycloak realm.")
@click.option("--client-id", default=None, help="Override Keycloak client ID.")
@click.option(
    "--request-timeout",
    type=click.IntRange(1),
    default=30,
    show_default=True,
    help="Default timeout in seconds for API requests.",
)
@click.option(
    "--upload-timeout",
    type=click.IntRange(1),
    default=300,
    show_default=True,
    help="Timeout in seconds for file upload requests.",
)
@click.pass_context
def main(
    ctx: click.Context,
    api_key: str | None,
    base_url: str | None,
    keycloak_url: str | None,
    realm: str | None,
    client_id: str | None,
    request_timeout: int,
    upload_timeout: int,
) -> None:
    """Arena CLI - Interact with the Arena RLOps platform directly from the command-line."""
    ctx.obj = CommandConfig(
        api_key=api_key,
        base_url=base_url,
        keycloak_url=keycloak_url,
        realm=realm,
        client_id=client_id,
        request_timeout=request_timeout,
        upload_timeout=upload_timeout,
    )


@main.command()
@click.option(
    "--timeout",
    type=click.IntRange(1),
    default=300,
    show_default=True,
    help="Maximum seconds to wait for device authorization.",
)
@click.pass_obj
def login(
    config: CommandConfig,
    timeout: int,
) -> None:
    """Authenticate with Arena."""
    with arena_client(config) as client:
        client.login(timeout=timeout)


@main.command()
@click.pass_obj
def logout(
    config: CommandConfig,
) -> None:
    """Log out and clear persisted credentials."""
    with arena_client(config) as client:
        client.logout()


@main.group("user")
def user_group() -> None:
    """Retrieve user / account information."""


@user_group.command("profile")
@click.pass_obj
def user_profile(
    config: CommandConfig,
) -> None:
    """Get current authenticated user profile."""
    with arena_client(config) as client:
        user = client.get_current_user()
        click.echo(f"User: {user.get('first_name', '')} {user.get('last_name', '')}")
        click.echo(f"Email: {user.get('email', '')}")


@user_group.command("credits")
@click.pass_obj
def user_credits(
    config: CommandConfig,
) -> None:
    """Get remaining account credits."""
    with arena_client(config) as client:
        emit_result(client.get_user_credits())


@main.group("environments")
def environments() -> None:
    """Manage your custom Gym / PettingZoo environments in Arena."""


@environments.command("list")
@click.pass_obj
def env_list(
    config: CommandConfig,
) -> None:
    """List all available environments in Arena, and whether they have been validated and profiled."""
    with arena_client(config) as client:
        emit_result(client.list_environments())


@environments.command("exists")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_exists(
    config: CommandConfig,
    name: str,
    version: str,
) -> None:
    """Check if an environment version exists in Arena."""
    with arena_client(config) as client:
        exists = client.environment_exists(name=name, version=version)
        emit_result({"name": name, "version": version, "exists": exists})


@environments.command("list-entrypoints")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_entrypoints(
    config: CommandConfig,
    name: str,
    version: str,
) -> None:
    """List available entrypoints for an existing environment version."""
    with arena_client(config) as client:
        emit_result(client.list_environment_entrypoints(name=name, version=version))


@environments.command("validate")
@click.argument("name", required=False, default=None, type=str)
@click.option("--name", "name_opt", default=None, hidden=True)
@click.option("--version", default="latest", show_default=True)
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to environment source directory or .tar.gz archive.",
)
@click.option(
    "--env-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to env_config.yaml file.",
)
@click.option(
    "--requirements",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to requirements.txt file.",
)
@click.option(
    "--entrypoint",
    default=None,
    help="Optional entrypoint override. Useful when multiple entrypoints exist.",
)
@click.option(
    "--description",
    default=None,
    help="Optional description of the environment.",
)
@click.option("--multi-agent/--single-agent", default=False, show_default=True)
@click.option("--do-rollouts/--no-do-rollouts", default=True, show_default=True)
@click.pass_obj
def env_validate(
    config: CommandConfig,
    name: str | None,
    name_opt: str | None,
    version: str,
    source: Path | None,
    env_config: Path | None,
    requirements: Path | None,
    entrypoint: str | None,
    description: str | None,
    multi_agent: bool,
    do_rollouts: bool,
) -> None:
    """Validate an environment on Arena.

    Pass 'name' to validate an already-registered environment.  Pass --source
    to upload and validate in one step.  When using --source without a positional
    'name', the source directory/file name is used by default.
    """
    env_name = name or name_opt or (source.stem if source else None)
    if env_name is None:
        msg = "Provide an environment name or use --source to upload and validate."
        raise click.UsageError(msg)

    with streaming_client(config) as (client, renderer):
        stream_resp = client.validate_environment(
            name=env_name,
            version=version,
            source=source,
            env_config=env_config,
            requirements=requirements,
            entrypoint=entrypoint,
            description=description,
            multi_agent=multi_agent,
            do_rollouts=do_rollouts,
            stream=True,
        )
        stream_resp.collect()
        renderer.close()


@environments.command("profile")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_profile(
    config: CommandConfig,
    name: str,
    version: str,
) -> None:
    """Profile a validated environment in Arena and get its resource requirements."""
    with streaming_client(config) as (client, renderer):
        stream_resp = client.profile_environment(
            name=name,
            version=version,
            stream=True,
        )
        result = stream_resp.collect()
        renderer.close()
        if result:
            emit_result(result)


@environments.command("delete")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without interactive prompt.",
)
@click.pass_obj
def env_delete(
    config: CommandConfig,
    name: str,
    version: str,
    yes: bool,
) -> None:
    """Delete an environment version from Arena."""
    if not yes and not click.confirm(
        f"Delete environment {name}:{version}?", default=False
    ):
        click.echo("Aborted.")
        return

    with arena_client(config) as client:
        result = client.delete_environment(name=name, version=version)
        if result in ("", None):
            emit_result({"deleted": True, "name": name, "version": version})
            return
        emit_result(result)


@main.group("jobs")
def jobs() -> None:
    """Submit, validate, and inspect training jobs."""


@jobs.command("submit")
@click.option(
    "--resource-id",
    type=int,
    default=None,
    help="Arena cluster type to submit the experiment to.",
)
@click.option(
    "--manifest-json",
    default=None,
    help="Raw JSON manifest payload string.",
)
@click.option(
    "--manifest-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to JSON file containing manifest payload.",
)
@click.pass_obj
def experiments_submit(
    config: CommandConfig,
    resource_id: int | None,
    manifest_json: str | None,
    manifest_file: str | None,
) -> None:
    """Submit a training job from run spec and/or existing experiment IDs."""
    manifest: dict[str, Any] | None = None
    if manifest_json is not None or manifest_file is not None:
        manifest = load_json_payload(
            manifest_json,
            manifest_file,
            json_option_name="--manifest-json",
            file_option_name="--manifest-file",
        )

    with streaming_client(config) as (client, renderer):
        stream_resp = client.submit_experiment_job(
            manifest=manifest,
            resource_id=resource_id,
            stream=True,
        )
        result = stream_resp.collect()
        renderer.close()
        if result:
            emit_result(result)


@jobs.command("status")
@click.argument("experiment_id", type=int)
@click.pass_obj
def jobs_status(config: CommandConfig, experiment_id: int) -> None:
    """Get status/details for an experiment by ID."""
    with arena_client(config) as client:
        emit_result(client.get_experiment_status(experiment_id))


@jobs.command("validate-runspec")
@click.option("--runspec-json", default=None, help="Raw JSON runspec payload string.")
@click.option(
    "--runspec-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to JSON file containing runspec payload.",
)
@click.pass_obj
def jobs_validate_runspec(
    config: CommandConfig,
    runspec_json: str | None,
    runspec_file: str | None,
) -> None:
    """Validate whether a run spec is structurally valid for training."""
    run_spec = load_json_payload(
        runspec_json,
        runspec_file,
        json_option_name="--runspec-json",
        file_option_name="--runspec-file",
    )
    with arena_client(config) as client:
        emit_result(client.validate_job_run_spec(run_spec))


@jobs.command("get-metrics")
@click.argument("experiment_id", type=int)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    required=True,
    help="Metric to download. Repeat for multiple metrics.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination file path. Defaults to ./experiment_<id>_metrics.csv (or .zip).",
)
@click.option(
    "--preview-rows",
    type=click.IntRange(0),
    default=10,
    show_default=True,
    help="When CSV is returned, preview this many rows in a rich table.",
)
@click.pass_obj
def jobs_get_metrics(
    config: CommandConfig,
    experiment_id: int,
    metrics: tuple[str, ...],
    output_file: Path | None,
    preview_rows: int,
) -> None:
    """Download metrics data for an experiment as CSV (or zip)."""
    with arena_client(config) as client:
        payload, content_type, disposition = client.download_experiment_metrics(
            experiment_id=experiment_id, metrics=list(metrics)
        )
        target_path = resolve_metrics_output_path(
            experiment_id=experiment_id,
            payload=payload,
            content_type=content_type,
            disposition=disposition,
            output_file=output_file,
        )
        target_path.write_bytes(payload)
        emit_result(
            {
                "saved": str(target_path),
                "bytes": len(payload),
                "content_type": content_type or "unknown",
            },
        )

        if preview_rows > 0 and (content_type or "").startswith("text/csv"):
            emit_csv_preview(payload, max_rows=preview_rows)
