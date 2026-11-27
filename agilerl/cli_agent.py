# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Deployment update and archive commands, and chat session choice, for ``arena agent``."""

from __future__ import annotations

import logging
import uuid

import click

from agilerl.config import CommandConfig, arena_client
from agilerl.inference.cache import load_active_session, save_active_session
from agilerl.output import emit_result

logger = logging.getLogger(__name__)


def register_deployment_commands(group: click.Group) -> None:
    """Attach deployment update and archive commands to *group*."""
    group.add_command(agent_update)
    group.add_command(agent_delete)


def session_for_prompt(
    deployment: str, session_id: str | None, new_session: bool
) -> str:
    """Pick the session a prompt runs in, starting and recording one if needed.

    A new conversation gets its id here rather than from the deployment, which is
    what lets the next prompt carry on from it: a streamed reply is plain text
    and has nowhere to hand an id back.
    """
    if session_id:
        return session_id.strip()
    if not new_session:
        resumed = load_active_session(deployment)
        if resumed:
            return resumed
    started = str(uuid.uuid4())
    save_active_session(deployment, started)
    logger.info("Started session %s.", started)
    return started


@click.command("update")
@click.argument("name")
@click.option("--new-name", default=None, help="Replacement display name.")
@click.option("--description", default=None, help="Replacement description.")
@click.option(
    "--enabled/--disabled",
    "enabled",
    default=None,
    help="Enable or disable the deployment.",
)
@click.option(
    "--experiment-name",
    default=None,
    help="Experiment name when several deployments share this name.",
)
@click.option(
    "--project-name",
    default=None,
    help="Project name when several deployments share this name.",
)
@click.pass_obj
def agent_update(
    config: CommandConfig,
    name: str,
    new_name: str | None,
    description: str | None,
    enabled: bool | None,
    experiment_name: str | None,
    project_name: str | None,
) -> None:
    """Rename, describe, or enable a deployment."""
    with arena_client(config) as client:
        emit_result(
            client.update_inference_deployment(
                name,
                new_name=new_name,
                description=description,
                enabled=enabled,
                experiment_name=experiment_name,
                project_name=project_name,
            )
        )


@click.command("delete")
@click.argument("name")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Archive without an interactive prompt.",
)
@click.option(
    "--experiment-name",
    default=None,
    help="Experiment name when several deployments share this name.",
)
@click.option(
    "--project-name",
    default=None,
    help="Project name when several deployments share this name.",
)
@click.pass_obj
def agent_delete(
    config: CommandConfig,
    name: str,
    yes: bool,
    experiment_name: str | None,
    project_name: str | None,
) -> None:
    """Archive a deployment."""
    if not yes and not click.confirm(f"Archive deployment {name}?"):
        return
    with arena_client(config) as client:
        emit_result(
            client.delete_inference_deployment(
                name,
                experiment_name=experiment_name,
                project_name=project_name,
            )
        )
