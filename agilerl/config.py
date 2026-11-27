# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import click

from agilerl.client import ArenaClient
from agilerl.output import handle_error

ClientT = TypeVar("ClientT", bound=ArenaClient)


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
    org_key: str | None = None
    external_user_id: str | None = None
    module_catalog_url: str | None = None
    module_catalog_public_key: str | None = None


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
        org_key=params.get("org_key"),
        external_user_id=params.get("external_user_id"),
        base_url=params.get("base_url"),
        keycloak_url=params.get("keycloak_url"),
        realm=params.get("realm"),
        client_id=params.get("client_id"),
        request_timeout=params.get("request_timeout") or 30,
        upload_timeout=params.get("upload_timeout") or 300,
        module_catalog_url=params.get("module_catalog_url"),
        module_catalog_public_key=params.get("module_catalog_public_key"),
    )


def build_client(
    config: CommandConfig, client_cls: type[ClientT] = ArenaClient
) -> ClientT:
    """Build an :class:`ArenaClient` (or subclass) with the given configuration.

    :param config: The command configuration.
    :type config: CommandConfig
    :param client_cls: Client class to build, e.g. a plugin's subclass.
    :type client_cls: type[ArenaClient]
    :returns: A ``client_cls`` instance.
    :rtype: ArenaClient
    """
    client_cls.configure(
        base_url=config.base_url,
        keycloak_url=config.keycloak_url,
        realm=config.realm,
        client_id=config.client_id,
    )
    return client_cls(
        api_key=config.api_key,
        org_key=config.org_key,
        external_user_id=config.external_user_id,
        request_timeout=config.request_timeout,
        upload_timeout=config.upload_timeout,
    )


@contextmanager
def arena_client(
    config: CommandConfig, client_cls: type[ClientT] = ArenaClient
) -> Generator[ClientT, None, None]:
    """Build an :class:`ArenaClient` (or subclass), handle errors, and guarantee cleanup.

    :param config: The command configuration.
    :type config: CommandConfig
    :param client_cls: Client class to build, e.g. a plugin's subclass.
    :type client_cls: type[ArenaClient]
    :returns: A generator that yields the client and ensures it is closed.
    :rtype: Generator[ArenaClient, None, None]
    """
    client = build_client(config, client_cls)
    try:
        yield client
    except Exception as exc:
        handle_error(exc)
    finally:
        client.close()
