from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import click

from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaAuthError,
    ArenaError,
    ArenaValidationError,
)

OutputFormat = Literal["json", "text"]


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
        for key, value in result.items():
            click.echo(f"{key}: {value}", err=is_error)
        return

    if isinstance(result, list):
        for item in result:
            click.echo(item, err=is_error)
        return

    click.echo(result, err=is_error)


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
    default="json",
    show_default=True,
    help="Output format.",
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


@main.command()
@click.pass_obj
def user(
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


@main.command()
@click.pass_obj
def credits(
    config: CLIConfig,
) -> None:
    """Get current authenticated user credits."""
    client = _build_client(config)
    try:
        _emit(client.get_user_credits(), config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@main.group()
def env() -> None:
    """Environment-related commands."""


@env.command("list")
@click.pass_obj
def env_list(
    config: CLIConfig,
) -> None:
    """List custom environments."""
    client = _build_client(config)
    try:
        _emit(client.list_custom_environments(), config.output)
    except Exception as exc:  # noqa: BLE001
        _handle_error(exc, config.output)
    finally:
        client.close()


@env.command("exists")
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


@env.command("entrypoints")
@click.argument("name")
@click.option("--version", default="latest", show_default=True)
@click.pass_obj
def env_entrypoints(
    config: CLIConfig,
    name: str,
    version: str,
) -> None:
    """List available environment entrypoints."""
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
