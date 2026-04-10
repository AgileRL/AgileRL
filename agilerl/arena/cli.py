from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaError,
    ArenaValidationError,
)

OutputFormat = Literal["json", "text"]
console = Console()
error_console = Console(stderr=True)


def _print_rich(renderable: Any, *, is_error: bool = False) -> None:
    if is_error:
        error_console.print(renderable)
        return
    console.print(renderable)


@dataclass(slots=True)
class CLIConfig:
    api_key: str | None
    base_url: str | None
    keycloak_url: str | None
    realm: str | None
    client_id: str | None
    request_timeout: int
    upload_timeout: int
    output: OutputFormat


def _build_client(config: CLIConfig) -> ArenaClient:
    ArenaClient.configure(
        base_url=config.base_url,
        keycloak_url=config.keycloak_url,
        realm=config.realm,
        client_id=config.client_id,
    )
    return ArenaClient(
        api_key=config.api_key,
        request_timeout=config.request_timeout,
        upload_timeout=config.upload_timeout,
    )


def _emit(result: Any, output: OutputFormat, *, is_error: bool = False) -> None:
    if output == "json":
        click.echo(json.dumps(result, indent=2, default=str), err=is_error)
        return

    if isinstance(result, dict):
        if _looks_like_environment_catalog(result):
            _emit_environment_catalog(result, is_error=is_error)
            return
        _emit_key_value_table(result, is_error=is_error)
        return

    if isinstance(result, list):
        if result and all(isinstance(item, dict) for item in result):
            _emit_list_of_dicts(result, is_error=is_error)
            return
        _emit_simple_list(result, is_error=is_error)
        return

    _print_rich(str(result), is_error=is_error)


def _emit_key_value_table(values: dict[str, Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(str(key), _format_cell(value))
    _print_rich(table, is_error=is_error)


def _emit_simple_list(values: list[Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Value")
    for value in values:
        table.add_row(_format_cell(value))
    _print_rich(table, is_error=is_error)


def _emit_list_of_dicts(values: list[dict[str, Any]], *, is_error: bool = False) -> None:
    columns: list[str] = []
    for row in values:
        for key in row:
            if key not in columns:
                columns.append(key)

    table = Table(show_header=True, header_style="bold")
    for column in columns:
        table.add_column(str(column))

    for row in values:
        table.add_row(*[_format_cell(row.get(column)) for column in columns])
    _print_rich(table, is_error=is_error)


def _looks_like_environment_catalog(values: dict[str, Any]) -> bool:
    if not values:
        return False
    for version_map in values.values():
        if not isinstance(version_map, dict):
            return False
        for metadata in version_map.values():
            if not isinstance(metadata, dict):
                return False
            if not {"validated", "profiled"}.issubset(metadata):
                return False
    return True


def _emit_environment_catalog(values: dict[str, Any], *, is_error: bool = False) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Environment")
    table.add_column("Version")
    table.add_column("Validated")
    table.add_column("Profiled")

    if not values:
        _print_rich("No environments found.", is_error=is_error)
        return

    for env_name, versions in values.items():
        if not isinstance(versions, dict):
            continue
        for version_name, metadata in versions.items():
            metadata_dict = metadata if isinstance(metadata, dict) else {}
            table.add_row(
                str(env_name),
                str(version_name),
                "yes" if bool(metadata_dict.get("validated")) else "no",
                "yes" if bool(metadata_dict.get("profiled")) else "no",
            )
    _print_rich(table, is_error=is_error)


def _format_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return str(value)


def _handle_error(exc: Exception, output: OutputFormat) -> None:
    if isinstance(exc, ArenaValidationError):
        _emit({"error": "validation_failed", "details": exc.errors}, output, is_error=True)
        raise click.exceptions.Exit(1)
    if isinstance(exc, ArenaAuthError):
        _emit(
            {"error": "authentication_failed", "details": str(exc)},
            output,
            is_error=True,
        )
        raise click.exceptions.Exit(1)
    if isinstance(exc, ArenaAPIError):
        _emit(
            {
                "error": "api_error",
                "status_code": exc.status_code,
                "details": exc.detail,
            },
            output,
            is_error=True,
        )
        raise click.exceptions.Exit(1)
    if isinstance(exc, ArenaError):
        _emit({"error": "arena_error", "details": str(exc)}, output, is_error=True)
        raise click.exceptions.Exit(1)

    raise exc


def _stream_chunk(chunk: str) -> None:
    click.echo(chunk, nl=False)


@dataclass(slots=True)
class _StreamRow:
    event_type: str
    name: str
    status: str
    details: str


class _StreamTableRenderer:
    """Incrementally render newline-delimited JSON chunks as a Rich table."""

    def __init__(self, *, is_error: bool = False) -> None:
        self._console = error_console if is_error else console
        self._buffer = ""
        self._rows: list[_StreamRow] = []
        self._live: Live | None = None
        self._completion_payload: dict[str, Any] | None = None
        self._final_status_payload: dict[str, Any] | None = None

    def on_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._ensure_live()
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._consume_line(line.strip())
        self._refresh()

    def finalize_result(self, result: Any) -> Any | None:
        self.close()
        if not isinstance(result, dict):
            return result

        cleaned = {
            key: value for key, value in result.items() if key not in {"stream", "events"}
        }
        if self._completion_payload is None:
            return cleaned or None

        merged = {**cleaned, **self._completion_payload}
        if self._final_status_payload is not None:
            merged.setdefault("final_status", self._final_status_payload.get("status"))
            merged.setdefault("final_stage", self._final_status_payload.get("stage"))
            merged.setdefault("final_message", self._final_status_payload.get("message"))
        return merged or None

    def close(self) -> None:
        if self._buffer.strip():
            self._consume_line(self._buffer.strip())
            self._buffer = ""
        if self._live is not None:
            self._refresh()
            self._live.stop()
            self._live = None

    def _ensure_live(self) -> None:
        if self._live is not None:
            return
        self._live = Live(
            self._build_table(),
            console=self._console,
            refresh_per_second=8,
        )
        self._live.start()

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_table())

    def _build_table(self) -> Table:
        table = Table(title="Stream Updates")
        table.add_column("Type", no_wrap=True)
        table.add_column("Name")
        table.add_column("Status", no_wrap=True)
        table.add_column("Details")
        for row in self._rows:
            table.add_row(row.event_type, row.name, row.status, row.details)
        return table

    def _consume_line(self, line: str) -> None:
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._rows.append(_StreamRow("log", "-", "-", line))
            return

        if not isinstance(payload, dict):
            self._rows.append(_StreamRow("event", "-", "-", _format_cell(payload)))
            return

        if payload.get("kind") == "status":
            stage = str(payload.get("stage", "-"))
            status = str(payload.get("status", "-"))
            raw_message = payload.get("message", "")
            message = str(raw_message)
            parsed_message = self._parse_json_message(message)
            if parsed_message is not None:
                message = self._summarize_payload(parsed_message)
            self._rows.append(_StreamRow("status", stage, status, message))
            if status == "completed":
                self._final_status_payload = payload
                if isinstance(parsed_message, dict):
                    self._completion_payload = parsed_message
            return

        if "check" in payload and isinstance(payload.get("result"), dict):
            check_name = str(payload["check"])
            result = payload["result"]
            success = result.get("success")
            status = "PASS" if success is True else "FAIL" if success is False else "UNKNOWN"
            error_msg = result.get("error msg") or result.get("error") or ""
            warnings = result.get("warnings")
            warning_text = ""
            if isinstance(warnings, list) and warnings:
                warning_text = f"warnings: {', '.join(str(item) for item in warnings)}"
            details = "; ".join(part for part in (str(error_msg), warning_text) if part).strip()
            self._rows.append(_StreamRow("check", check_name, status, details or "-"))
            return

        if payload.get("complete") is True:
            self._completion_payload = payload
            env_name = (
                payload.get("env_info", {}).get("env_name")
                if isinstance(payload.get("env_info"), dict)
                else None
            )
            details = f"env: {env_name}" if env_name else "Validation payload received"
            self._rows.append(_StreamRow("result", "validation", "complete", details))
            return

        self._rows.append(_StreamRow("event", "-", "-", _format_cell(payload)))

    @staticmethod
    def _parse_json_message(message: str) -> dict[str, Any] | list[Any] | None:
        message = message.strip()
        if not message or message[0] not in "{[":
            return None
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, (dict, list)):
            return parsed
        return None

    @staticmethod
    def _summarize_payload(payload: dict[str, Any] | list[Any]) -> str:
        if isinstance(payload, list):
            return f"items: {len(payload)}"
        accepted = payload.get("accepted")
        experiment_id = payload.get("experimentId") or payload.get("experiment_id")
        submissions = payload.get("submissions")
        parts: list[str] = []
        if accepted is not None:
            parts.append(f"accepted: {accepted}")
        if experiment_id is not None:
            parts.append(f"experiment_id: {experiment_id}")
        if isinstance(submissions, list):
            parts.append(f"submissions: {len(submissions)}")
        if parts:
            return ", ".join(parts)
        keys = ", ".join(str(key) for key in list(payload.keys())[:4])
        return f"result keys: {keys}" if keys else "completed"


def _build_stream_handler(
    output: OutputFormat,
) -> tuple[Callable[[str], None], _StreamTableRenderer | None]:
    if output == "json":
        return _stream_chunk, None
    renderer = _StreamTableRenderer()
    return renderer.on_chunk, renderer


def _load_json_payload(
    payload_json: str | None,
    payload_file: str | None,
    *,
    json_option_name: str,
    file_option_name: str,
) -> dict[str, Any]:
    if payload_json is None and payload_file is None:
        msg = f"Provide either {json_option_name} or {file_option_name}."
        raise click.UsageError(msg)
    if payload_json is not None and payload_file is not None:
        msg = f"Use only one of {json_option_name} or {file_option_name}."
        raise click.UsageError(msg)

    try:
        if payload_json is not None:
            parsed = json.loads(payload_json)
        else:
            parsed = json.loads(Path(payload_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        msg = f"Invalid JSON payload: {exc}"
        raise click.UsageError(msg) from exc

    if not isinstance(parsed, dict):
        msg = "Payload must be a JSON object."
        raise click.UsageError(msg)
    return parsed


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
    ctx.obj = CLIConfig(
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
    config: CLIConfig,
    timeout: int,
) -> None:
    """Authenticate with Arena."""
    client = _build_client(config)
    try:
        client.login(timeout=timeout)
        click.echo("Login successful.")
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@main.command()
@click.pass_obj
def logout(
    config: CLIConfig,
) -> None:
    """Log out and clear persisted credentials."""
    client = _build_client(config)
    try:
        client.logout()
        click.echo("Logout successful.")
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@main.group("user")
def user_group() -> None:
    """User/account commands."""


@user_group.command("profile")
@click.pass_obj
def user_profile(
    config: CLIConfig,
) -> None:
    """Get current authenticated user profile."""
    client = _build_client(config)
    try:
        _emit(client.get_current_user(), config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@user_group.command("credits")
@click.pass_obj
def user_credits(
    config: CLIConfig,
) -> None:
    """Get remaining account credits."""
    client = _build_client(config)
    try:
        _emit(client.get_user_credits(), config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@main.group("environments")
def environments() -> None:
    """Manage custom gym environments."""


@environments.command("list")
@click.pass_obj
def env_list(
    config: CLIConfig,
) -> None:
    """List all available environments with validated/profiled metadata."""
    client = _build_client(config)
    try:
        _emit(client.list_custom_environments(), config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@environments.command("exists")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_exists(
    config: CLIConfig,
    name: str,
    version: str,
) -> None:
    """Check if an environment version exists."""
    client = _build_client(config)
    try:
        exists = client.custom_environment_exists(name=name, version=version)
        _emit({"name": name, "version": version, "exists": exists}, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@environments.command("list-entrypoints")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_entrypoints(
    config: CLIConfig,
    name: str,
    version: str,
) -> None:
    """List available entrypoints for an existing environment version."""
    client = _build_client(config)
    try:
        _emit(
            client.list_custom_environment_entrypoints(name=name, version=version),
            config.output,
        )
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
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
    from uuid import uuid4

    client = _build_client(config)
    env_name = name or str(uuid4())
    stream_renderer: _StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = _build_stream_handler(config.output)
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
            _emit(result, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
    name: str,
    version: str,
    entrypoint: str | None,
    do_rollouts: bool,
    stream: bool,
) -> None:
    """Validate an already-created environment version."""
    client = _build_client(config)
    stream_renderer: _StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = _build_stream_handler(config.output)
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
            _emit(result, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
    name: str,
    version: str,
    custom_env_path: str | None,
    multi_agent: bool,
    stream: bool,
) -> None:
    """Profile a validated environment (returns cpu_per_env and ram_per_env)."""
    client = _build_client(config)
    stream_renderer: _StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = _build_stream_handler(config.output)
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
            _emit(result, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
    name: str,
    version: str,
    yes: bool,
) -> None:
    """Delete an environment version from Arena."""
    if not yes and not click.confirm(f"Delete environment {name}:{version}?", default=False):
        click.echo("Aborted.")
        return

    client = _build_client(config)
    try:
        result = client.delete_custom_environment(name=name, version=version)
        if result in ("", None):
            _emit({"deleted": True, "name": name, "version": version}, config.output)
            return
        _emit(result, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
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
        manifest = _load_json_payload(
            manifest_json,
            manifest_file,
            json_option_name="--manifest-json",
            file_option_name="--manifest-file",
        )

    client = _build_client(config)
    stream_renderer: _StreamTableRenderer | None = None
    try:
        on_chunk: Callable[[str], None] | None = None
        if stream:
            on_chunk, stream_renderer = _build_stream_handler(config.output)
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
            _emit(result, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        if stream_renderer is not None:
            stream_renderer.close()
        client.close()


@jobs.command("status")
@click.argument("experiment_id", type=int)
@click.pass_obj
def jobs_status(config: CLIConfig, experiment_id: int) -> None:
    """Get status/details for an experiment by ID."""
    client = _build_client(config)
    try:
        _emit(client.get_experiment_status(experiment_id), config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
    runspec_json: str | None,
    runspec_file: str | None,
) -> None:
    """Validate whether a run spec is structurally valid for training."""
    run_spec = _load_json_payload(
        runspec_json,
        runspec_file,
        json_option_name="--runspec-json",
        file_option_name="--runspec-file",
    )
    client = _build_client(config)
    try:
        result = client.validate_job_run_spec(run_spec)
        _emit(result, config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
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
    config: CLIConfig,
    experiment_id: int,
    metrics: tuple[str, ...],
    output_file: Path | None,
    preview_rows: int,
) -> None:
    """Download metrics data for an experiment as CSV (or zip)."""
    client = _build_client(config)
    try:
        payload, content_type, disposition = client.download_experiment_metrics(
            experiment_id=experiment_id, metrics=list(metrics)
        )
        target_path = _resolve_metrics_output_path(
            experiment_id=experiment_id,
            payload=payload,
            content_type=content_type,
            disposition=disposition,
            output_file=output_file,
        )
        target_path.write_bytes(payload)
        _emit(
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
            _emit_csv_preview(payload, max_rows=preview_rows)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


def _resolve_metrics_output_path(
    *,
    experiment_id: int,
    payload: bytes,
    content_type: str | None,
    disposition: str | None,
    output_file: Path | None,
) -> Path:
    if output_file is not None:
        return output_file

    suggested_name = _filename_from_disposition(disposition)
    if suggested_name:
        return Path(suggested_name)

    is_zip = payload.startswith(b"PK") or "zip" in (content_type or "").lower()
    suffix = ".zip" if is_zip else ".csv"
    return Path(f"experiment_{experiment_id}_metrics{suffix}")


def _filename_from_disposition(disposition: str | None) -> str | None:
    if not disposition:
        return None
    match = re.search(r'filename="?([^";]+)"?', disposition)
    return match.group(1) if match else None


def _emit_csv_preview(payload: bytes, *, max_rows: int) -> None:
    text = payload.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return

    header = rows[0]
    data_rows = rows[1 : max_rows + 1]
    table = Table(title=f"Metrics Preview (first {len(data_rows)} rows)")
    for column in header:
        table.add_column(column)
    for row in data_rows:
        padded = row + [""] * max(0, len(header) - len(row))
        table.add_row(*padded[: len(header)])
    console.print(table)


