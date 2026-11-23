# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Price preview and confirmation for ``arena experiments submit``."""

from __future__ import annotations

from typing import Any

import click

from agilerl.arena.client import ArenaClient
from agilerl.arena.exceptions import ArenaAPIError
from agilerl.arena.output import stdin_is_a_terminal


def confirm_submit_price(client: ArenaClient, yes: bool, **submit_args: Any) -> None:
    """Print what a submit would run on and cost, then ask to go on.

    :raises click.Abort: The user declined.
    """
    try:
        quote = client.quote_experiment(**submit_args)
    except ArenaAPIError as err:
        # Arena servers without the quote endpoint answer 404.
        if err.status_code != 404:
            raise
        click.echo(
            "This Arena server cannot price a run before it starts. "
            "The hourly rate prints during submit.",
            err=True,
        )
        return
    for line in submit_quote_lines(quote):
        click.echo(line, err=True)
    if yes or not stdin_is_a_terminal():
        return
    click.confirm("Submit and start billing?", default=False, abort=True, err=True)


def submit_quote_lines(quote: dict[str, Any]) -> list[str]:
    """Hardware, hourly rate and estimated total for a submit quote."""
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
