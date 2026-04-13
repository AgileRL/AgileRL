from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import click

from agilerl.arena.config import CommandConfig, OutputFormat, build_client
from agilerl.arena.output import (
    StreamTableRenderer,
    build_stream_handler,
    emit,
    emit_csv_preview,
    handle_error,
)
from agilerl.arena.payloads import load_json_payload, resolve_metrics_output_path


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
@click.option(
    "--output",
    type=click.Choice(["json", "text"]),
    default="text",
    show_default=True,
    help="Output format. Use text for rich tables; json for scripting.",
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
    output: OutputFormat,
) -> None:
    """CLI for interacting with AgileRL Arena."""
    ctx.obj = CommandConfig(
        api_key=api_key,
        base_url=base_url,
        keycloak_url=keycloak_url,
        realm=realm,
        client_id=client_id,
        request_timeout=request_timeout,
        upload_timeout=upload_timeout,
        output=output,
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
    client = build_client(config)
    try:
        client.login(timeout=timeout)
        click.echo("Login successful.")
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@main.command()
@click.pass_obj
def logout(
    config: CommandConfig,
) -> None:
    """Log out and clear persisted credentials."""
    client = build_client(config)
    try:
        client.logout()
        click.echo("Logout successful.")
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@main.group("user")
def user_group() -> None:
    """User/account commands."""


@user_group.command("profile")
@click.pass_obj
def user_profile(
    config: CommandConfig,
) -> None:
    """Get current authenticated user profile."""
    client = build_client(config)
    try:
        emit(client.get_current_user(), config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@user_group.command("credits")
@click.pass_obj
def user_credits(
    config: CommandConfig,
) -> None:
    """Get remaining account credits."""
    client = build_client(config)
    try:
        emit(client.get_user_credits(), config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@main.group("environments")
def environments() -> None:
    """Manage custom gym environments."""


@environments.command("list")
@click.pass_obj
def env_list(
    config: CommandConfig,
) -> None:
    """List all available environments with validated/profiled metadata."""
    client = build_client(config)
    try:
        emit(client.list_custom_environments(), config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@environments.command("exists")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_exists(
    config: CommandConfig,
    name: str,
    version: str,
) -> None:
    """Check if an environment version exists."""
    client = build_client(config)
    try:
        exists = client.custom_environment_exists(name=name, version=version)
        emit({"name": name, "version": version, "exists": exists}, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


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
    client = build_client(config)
    try:
        emit(
            client.list_custom_environment_entrypoints(name=name, version=version),
            config.output,
        )
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@environments.command("create-and-validate")
@click.option("--name", default=None, help="Environment name. Random UUID if omitted.")
@click.option("--version", default="ident", show_default=True)
@click.option(
    "--file-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to environment tar.gz archive.",
)
@click.option(
    "--env-config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to env_config.yaml file.",
)
@click.option(
    "--requirements-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to requirements.txt file.",
)
@click.option(
    "--entrypoint",
    default=None,
    help="Optional entrypoint. If omitted and multiple exist, backend returns options.",
)
@click.option("--multi-agent/--single-agent", default=False, show_default=True)
@click.option("--do-rollouts/--no-do-rollouts", default=True, show_default=True)
@click.option("--stream/--no-stream", default=True, show_default=True)
@click.pass_obj
def env_create_and_validate(
    config: CommandConfig,
    name: str | None,
    version: str,
    file_path: Path,
    env_config_path: Path,
    requirements_path: Path,
    entrypoint: str | None,
    multi_agent: bool,
    do_rollouts: bool,
    stream: bool,
) -> None:
    """Upload and automatically validate a custom environment."""
    client = build_client(config)
    env_name = name or str(uuid4())
    stream_renderer: StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = build_stream_handler(config.output)
        result = client.create_and_validate_custom_environment(
            name=env_name,
            version=version,
            file_path=file_path,
            env_config_path=env_config_path,
            requirements_path=requirements_path,
            entrypoint=entrypoint,
            multi_agent=multi_agent,
            do_rollouts=do_rollouts,
            stream=stream,
            on_chunk=on_chunk,
        )
        if stream and stream_renderer is not None:
            result = stream_renderer.finalize_result(result)
        elif stream:
            click.echo()
        if result is not None:
            emit(result, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        if stream_renderer is not None:
            stream_renderer.close()
        client.close()


@environments.command("validate")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.option(
    "--entrypoint",
    default=None,
    help="Optional entrypoint override. Useful when multiple entrypoints exist.",
)
@click.option("--do-rollouts/--no-do-rollouts", default=True, show_default=True)
@click.option("--stream/--no-stream", default=True, show_default=True)
@click.pass_obj
def env_validate(
    config: CommandConfig,
    name: str,
    version: str,
    entrypoint: str | None,
    do_rollouts: bool,
    stream: bool,
) -> None:
    """Validate an already-created environment version."""
    client = build_client(config)
    stream_renderer: StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = build_stream_handler(config.output)
        result = client.validate_custom_environment(
            name=name,
            version=version,
            entrypoint=entrypoint,
            do_rollouts=do_rollouts,
            stream=stream,
            on_chunk=on_chunk,
        )
        if stream and stream_renderer is not None:
            result = stream_renderer.finalize_result(result)
        elif stream:
            click.echo()
        if result is not None:
            emit(result, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        if stream_renderer is not None:
            stream_renderer.close()
        client.close()


@environments.command("profile")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.option(
    "--entrypoint",
    "custom_env_path",
    default=None,
    help="Entrypoint/custom env path to profile. Defaults to saved entrypoint.",
)
@click.option("--multi-agent/--single-agent", default=False, show_default=True)
@click.option("--stream/--no-stream", default=True, show_default=True)
@click.pass_obj
def env_profile(
    config: CommandConfig,
    name: str,
    version: str,
    custom_env_path: str | None,
    multi_agent: bool,
    stream: bool,
) -> None:
    """Profile a validated environment (returns cpu_per_env and ram_per_env)."""
    client = build_client(config)
    stream_renderer: StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = build_stream_handler(config.output)
        result = client.profile_custom_environment(
            name=name,
            version=version,
            custom_env_path=custom_env_path,
            multi_agent=multi_agent,
            stream=stream,
            on_chunk=on_chunk,
        )
        if stream and stream_renderer is not None:
            result = stream_renderer.finalize_result(result)
        elif stream:
            click.echo()
        if result is not None:
            emit(result, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        if stream_renderer is not None:
            stream_renderer.close()
        client.close()


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

    client = build_client(config)
    try:
        result = client.delete_custom_environment(name=name, version=version)
        if result in ("", None):
            emit({"deleted": True, "name": name, "version": version}, config.output)
            return
        emit(result, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


@main.group("jobs")
def jobs() -> None:
    """Submit, validate, and inspect training jobs."""


@jobs.command("submit")
@click.option(
    "--custom-gym-env-impl-id",
    type=int,
    default=None,
    help="Custom gym environment implementation ID.",
)
@click.option(
    "--experiment-id",
    type=int,
    default=None,
    help="Existing experiment ID to submit.",
)
@click.option(
    "--gym-env-id",
    type=int,
    default=None,
    help="Gym environment ID to submit against.",
)
@click.option(
    "--experiment-name",
    default=None,
    help="Optional experiment name for newly created jobs.",
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
@click.option("--stream/--no-stream", default=True, show_default=True)
@click.pass_obj
def experiments_submit(
    config: CommandConfig,
    custom_gym_env_impl_id: int | None,
    experiment_id: int | None,
    gym_env_id: int | None,
    experiment_name: str | None,
    manifest_json: str | None,
    manifest_file: str | None,
    stream: bool,
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

    client = build_client(config)
    stream_renderer: StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = build_stream_handler(config.output)
        result = client.submit_experiment_job(
            manifest=manifest,
            custom_gym_env_impl_id=custom_gym_env_impl_id,
            experiment_id=experiment_id,
            gym_env_id=gym_env_id,
            experiment_name=experiment_name,
            stream=stream,
            on_chunk=on_chunk,
        )
        if stream and stream_renderer is not None:
            result = stream_renderer.finalize_result(result)
        elif stream:
            click.echo()
        if result is not None:
            emit(result, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        if stream_renderer is not None:
            stream_renderer.close()
        client.close()


@jobs.command("status")
@click.argument("experiment_id", type=int)
@click.pass_obj
def jobs_status(config: CommandConfig, experiment_id: int) -> None:
    """Get status/details for an experiment by ID."""
    client = build_client(config)
    try:
        emit(client.get_experiment_status(experiment_id), config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


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
    client = build_client(config)
    try:
        result = client.validate_job_run_spec(run_spec)
        emit(result, config.output)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()


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
    client = build_client(config)
    try:
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
        emit(
            {
                "saved": str(target_path),
                "bytes": len(payload),
                "content_type": content_type or "unknown",
            },
            config.output,
        )

        if (
            config.output == "text"
            and preview_rows > 0
            and (content_type or "").startswith("text/csv")
        ):
            emit_csv_preview(payload, max_rows=preview_rows)
    except Exception as exc:
        handle_error(exc, config.output)
    finally:
        client.close()
