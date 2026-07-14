from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import click

from agilerl.arena.client import ArenaClient
from agilerl.arena.output import handle_error


@dataclass(slots=True)
class CommandConfig:
    """Configuration for the Arena CLI."""

    api_key: str | None
    base_url: str | None
    keycloak_url: str | None
    realm: str | None
    client_id: str | None
    request_timeout: int
    upload_timeout: int


def _resolve_root_command_config(ctx: click.Context) -> CommandConfig:
    """Build :class:`CommandConfig` for the Arena root group.

    ``main`` normally sets ``ctx.obj``, but eager ``-h`` / ``--help`` is handled
    during parsing before :meth:`click.Command.invoke`, so the callback has not
    run while the help page is formatted (``list_commands``, ``get_command``).
    Reconstruct from ``ctx.params`` so capability-driven commands appear on
    ``arena … --help``.
    """
    obj = ctx.obj
    if isinstance(obj, CommandConfig):
        return obj
    params = ctx.params
    if not isinstance(params, dict):
        params = {}
    return CommandConfig(
        api_key=params.get("api_key"),
        base_url=params.get("base_url"),
        keycloak_url=params.get("keycloak_url"),
        realm=params.get("realm"),
        client_id=params.get("client_id"),
        request_timeout=params.get("request_timeout") or 30,
        upload_timeout=params.get("upload_timeout") or 300,
    )


def build_client(config: CommandConfig) -> ArenaClient:
    """Build an :class:`ArenaClient` with the given configuration.

    :param config: The command configuration.
    :type config: CommandConfig
    :returns: An :class:`ArenaClient` instance.
    :rtype: ArenaClient
    """
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


@contextmanager
def arena_client(config: CommandConfig) -> Generator[ArenaClient, None, None]:
    """Build an :class:`ArenaClient`, handle errors, and guarantee cleanup.

    :param config: The command configuration.
    :type config: CommandConfig
    :returns: A generator that yields the ArenaClient and ensures it is closed.
    :rtype: Generator[ArenaClient, None, None]
    """
    client = build_client(config)
    try:
        yield client
    except Exception as exc:
        handle_error(exc)
    finally:
        client.close()
