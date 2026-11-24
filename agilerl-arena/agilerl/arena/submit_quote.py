# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Price preview and confirmation for ``arena experiments submit`` and ``resume``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import click

from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.output import (
    emit_json_result,
    json_output_enabled,
    stdin_is_a_terminal,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def confirm_submit_price(client: ArenaClient, yes: bool, **submit_args: Any) -> None:
    """Print what a submit would run on and cost, then ask to go on.

    :raises click.Abort: The user declined.
    """
    confirm_price(lambda: client.quote_experiment(**submit_args), yes, "Submit")


def confirm_resume_price(client: ArenaClient, yes: bool, **resume_args: Any) -> None:
    """Print what a resume would run on and cost, then ask to go on.

    :raises click.Abort: The user declined.
    """
    confirm_price(lambda: client.quote_resume(**resume_args), yes, "Resume")


def confirm_price(quote: Callable[[], dict[str, Any]], yes: bool, action: str) -> None:
    """Print the quote, then ask "<action> and start billing?".

    ``yes`` skips the question. Without ``yes``, stdin must be a terminal.

    :raises click.Abort: The user declined.
    :raises click.UsageError: No terminal to ask on and no ``yes``.
    """
    try:
        lines = submit_quote_lines(quote())
    except ArenaAPIError as err:
        # Arena servers without the quote endpoint answer 404.
        if err.status_code != 404:
            raise
        click.echo(
            "This Arena server cannot price a run before it starts.",
            err=True,
        )
        return
    for line in lines:
        click.echo(line, err=True)
    if yes:
        return
    if not stdin_is_a_terminal():
        msg = f"No terminal to confirm the price on. Pass --yes to {action.lower()} anyway."
        raise click.UsageError(msg)
    click.confirm(f"{action} and start billing?", default=False, abort=True, err=True)


def print_quote(quote: dict[str, Any]) -> None:
    """Print a quote for ``--dry-run``: JSON in JSON mode, else the price lines."""
    if json_output_enabled():
        emit_json_result(quote)
        return
    click.echo("\n".join(submit_quote_lines(quote)))


def submit_quote_lines(quote: dict[str, Any]) -> list[str]:
    """Hardware, hourly rate and estimated total for a quote."""
    nodes = quote["num_nodes"]
    gpus = f"{quote['num_gpus']}x {quote['gpu_type']}" if quote["gpu_type"] else "CPU"
    hardware = (
        f"{gpus} on {nodes} node{'s' if nodes != 1 else ''} "
        f"({quote['resource_name']}, {quote['resource_id']})"
    )
    if quote["estimated_cost"] is None:
        estimate = "no duration estimate"
    else:
        estimate = (
            f"${quote['estimated_cost']:,.2f} for about {quote['estimated_hours']:.1f} h "
            f"({quote['max_steps']:,} steps)"
        )
    return [
        f"Hardware:        {hardware}",
        f"Hourly rate:     ${quote['hourly_cost']:,.2f} per hour",
        f"Estimated total: {estimate}",
    ]
