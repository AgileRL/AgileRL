# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena experiments status``, ``logs`` and ``checkpoints``."""

from __future__ import annotations

from typing import Any

import click

from agilerl.config import CommandConfig, arena_client
from agilerl.output import emit_log_line, emit_log_summary, emit_result


def register_experiment_log_commands(group: click.Group) -> None:
    """Attach the status, logs and checkpoints commands to the ``experiments`` *group*."""
    group.add_command(experiment_status)
    group.add_command(experiment_logs)
    group.add_command(experiment_checkpoints)


@click.command("status")
@click.argument("experiment_name")
@click.option(
    "--project",
    default=None,
    help="Project, when the experiment name exists in more than one; the default project is used only then.",
)
@click.pass_obj
def experiment_status(
    config: CommandConfig, experiment_name: str, project: str | None
) -> None:
    """Show an experiment's status and, if it failed, why."""
    with arena_client(config) as client:
        emit_result(client.get_experiment_status(experiment_name, project=project))


@click.command("logs")
@click.argument("experiment_name")
@click.option(
    "--tail",
    type=click.IntRange(1, 1000),
    default=None,
    help="Show only the newest N lines.",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    help="Keep printing new lines until the experiment ends.",
)
@click.option(
    "--level",
    type=click.Choice(["info", "warning", "error"]),
    default=None,
    help="Lowest severity to show (default info).",
)
@click.option(
    "--project",
    default=None,
    help="Project, when the experiment name exists in more than one; the default project is used only then.",
)
@click.pass_obj
def experiment_logs(
    config: CommandConfig,
    experiment_name: str,
    tail: int | None,
    follow: bool,
    level: str | None,
    project: str | None,
) -> None:
    """Print an experiment's training logs, then why it failed if it did.

    With --json, prints one JSON object per log line, then one object with the
    experiment's final status and failure_reason.
    """
    page: dict[str, Any] = {}
    printed = 0
    with arena_client(config) as client:
        for page in client.iter_experiment_logs(
            experiment_name, tail=tail, follow=follow, level=level, project=project
        ):
            for line in page["lines"]:
                emit_log_line(line)
            printed += len(page["lines"])
    emit_log_summary(page, printed=printed)


@click.command("checkpoints")
@click.argument("experiment_name")
@click.option(
    "--project",
    default=None,
    help="Project, when the experiment name exists in more than one; the default project is used only then.",
)
@click.pass_obj
def experiment_checkpoints(
    config: CommandConfig,
    experiment_name: str,
    project: str | None,
) -> None:
    """List checkpoints for an experiment."""
    with arena_client(config) as client:
        rows = client.list_checkpoints(experiment_name, project=project)
    emit_result(
        rows,
        display=format_checkpoint_rows,
        empty_message=f"No checkpoints for {experiment_name}.",
    )


def format_checkpoint_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Put ``steps`` first and round ``size_mb`` to two decimals for the table."""
    out: list[dict[str, Any]] = []
    for row in rows:
        ordered: dict[str, Any] = {}
        if "steps" in row:
            ordered["steps"] = row["steps"]
        for key, value in row.items():
            if key == "steps":
                continue
            if key == "size_mb" and isinstance(value, (int, float)):
                value = f"{value:.2f}"
            ordered[key] = value
        out.append(ordered)
    return out
