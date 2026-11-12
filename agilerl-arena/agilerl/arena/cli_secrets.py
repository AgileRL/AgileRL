# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena secrets`` commands."""

from __future__ import annotations

import os
import sys

import click

from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.output import emit_json_result, emit_result, json_output_enabled


@click.group("secrets")
def secrets_group() -> None:
    """Manage your organization's secrets in Arena.

    A manifest's ``judge.api_key_env`` names one of these secrets; Arena
    passes it to the training job as an environment variable of that name.
    """


def _stdin_is_interactive() -> bool:
    """Whether stdin is a terminal rather than a pipe or file."""
    return sys.stdin.isatty()


def _read_secret_value(from_env: str | None) -> str:
    """Read the secret value from ``from_env``, piped stdin, or a hidden prompt.

    :param from_env: Name of a local environment variable holding the value.
    :type from_env: str | None
    :return: The secret value.
    :rtype: str
    """
    if from_env is not None:
        value = os.environ.get(from_env)
        if not value:
            msg = f"Environment variable {from_env} is not set or is empty."
            raise click.UsageError(msg)
        return value
    if _stdin_is_interactive():
        value = click.prompt("Secret value", hide_input=True, err=True)
    else:
        value = sys.stdin.read().rstrip("\r\n")
    if not value:
        msg = "The secret value is empty."
        raise click.UsageError(msg)
    return value


@secrets_group.command("list")
@click.pass_obj
def secrets_list(config: CommandConfig) -> None:
    """List your organization's secrets. Values are never shown."""
    with arena_client(config) as client:
        emit_result(
            client.list_secrets(),
            columns=["Name", "Created", "Updated", "Updated by"],
            empty_message="No secrets.",
        )


@secrets_group.command("set")
@click.argument("name")
@click.option(
    "--from-env",
    "from_env",
    metavar="VAR",
    default=None,
    help="Read the value from this local environment variable.",
)
@click.pass_obj
def secrets_set(config: CommandConfig, name: str, from_env: str | None) -> None:
    """Create or replace secret NAME.

    The value comes from --from-env, from stdin when piped, or from a hidden
    prompt. It is never accepted as an argument.
    """
    value = _read_secret_value(from_env)
    with arena_client(config) as client:
        row = client.set_secret(name=name, value=value)
    emit_json_result(row)
    if not json_output_enabled():
        click.echo(f"Stored secret {row['name']}.")


@secrets_group.command("delete")
@click.argument("name")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without interactive prompt.",
)
@click.pass_obj
def secrets_delete(config: CommandConfig, name: str, yes: bool) -> None:
    """Delete secret NAME."""
    if not yes and not click.confirm(
        f"Delete secret {name!r}?", default=False, err=True
    ):
        click.echo("Aborted.", err=True)
        return
    with arena_client(config) as client:
        result = client.delete_secret(name=name)
    emit_json_result(result)
    if not json_output_enabled():
        click.echo(f"Deleted secret {result['name']}.")
