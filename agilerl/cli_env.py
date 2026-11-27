# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena env`` commands."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from agilerl.config import CommandConfig, arena_client
from agilerl.output import emit_json_result, emit_result


def env_version_option(
    help_text: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """The ``--version`` option of an ``arena env`` command.

    :param help_text: What the version selects for this command.
    :return: The click option decorator.
    """
    return click.option("--version", default=None, help=help_text)


@click.group("env")
def env_group() -> None:
    """Manage your custom GEM environments."""


@env_group.command("list")
@click.option("--name", default=None, help="Only list this environment.")
@click.option(
    "--include-arena/--no-include-arena",
    "include_arena",
    default=False,
    show_default=True,
    type=bool,
    help="Also list the environments Arena provides.",
)
@click.pass_obj
def env_list(
    config: CommandConfig,
    name: str | None,
    include_arena: bool,
) -> None:
    """List all available environments in Arena, and whether they have been validated."""
    with arena_client(config) as client:
        emit_result(client.list_environments(name=name, include_arena=include_arena))


@env_group.command("exists")
@click.argument("name")
@env_version_option("Version to check. Omit for the latest.")
@click.pass_obj
def env_exists(
    config: CommandConfig,
    name: str,
    version: str | None,
) -> None:
    """Check if an environment version exists in Arena."""
    with arena_client(config) as client:
        emit_result(client.environment_exists(name=name, version=version))


@env_group.command("entrypoints")
@click.argument("name")
@env_version_option("Version to inspect. Omit for the latest.")
@click.pass_obj
def env_entrypoints(
    config: CommandConfig,
    name: str,
    version: str | None,
) -> None:
    """List available entrypoints for an existing environment version."""
    with arena_client(config) as client:
        emit_result(
            client.list_environment_entrypoints(name=name, version=version),
            columns=["Entrypoints"],
        )


@env_group.command("validate")
@click.argument("name", type=str)
@env_version_option(
    "Version to validate. Omit for the latest, or v1 when --source registers a new environment."
)
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
@click.option(
    "--do-rollouts/--no-do-rollouts",
    default=False,
    show_default=True,
    help="Also run sample episodes against the environment and report them.",
)
@click.pass_obj
def env_validate(
    config: CommandConfig,
    name: str,
    version: str | None,
    source: Path | None,
    env_config: Path | None,
    requirements: Path | None,
    entrypoint: str | None,
    description: str | None,
    do_rollouts: bool,
) -> None:
    """Validate a GEM environment on Arena.

    Pass NAME to validate an already-registered environment.  Pass --source
    to register and validate in one step.  NAME is always required and is never
    inferred from --source.
    """
    with arena_client(config) as client:
        result = client.validate_environment(
            name=name,
            version=version,
            source=source,
            env_config=env_config,
            requirements=requirements,
            entrypoint=entrypoint,
            description=description,
            do_rollouts=do_rollouts,
        )
    emit_json_result(result)
    if "error_code" in result:  # Failed checks still complete the stream.
        sys.exit(1)


@env_group.command("delete")
@click.argument("name")
@env_version_option("Version to delete. Omit to delete every version.")
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
    version: str | None,
    yes: bool,
) -> None:
    """Delete an environment version (or all versions if version is None) from Arena."""
    with arena_client(config) as client:
        client.delete_environment(name=name, version=version, confirm=yes)


@env_group.command("duplicate")
@click.argument("name")
@click.argument("new_version")
@env_version_option("Version to copy. Omit for the latest.")
@click.pass_obj
def env_duplicate(
    config: CommandConfig,
    name: str,
    new_version: str,
    version: str | None,
) -> None:
    """Copy an existing environment version to a new version name. If no version is provided, the latest version will be duplicated."""
    with arena_client(config) as client:
        client.duplicate_environment_version(
            name=name,
            new_version=new_version,
            version=version,
        )
