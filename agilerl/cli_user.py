# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""``arena user`` commands: profile, balance with accrued cost, and the ledger."""

from __future__ import annotations

from typing import Any

import click

from agilerl.client import ArenaClient
from agilerl.config import CommandConfig, arena_client
from agilerl.exceptions import ArenaAPIError
from agilerl.output import emit_json_result, emit_result, json_output_enabled


@click.group("user")
def user_group() -> None:
    """Retrieve user / account information."""


@user_group.command("profile")
@click.pass_obj
def user_profile(
    config: CommandConfig,
) -> None:
    """Get current authenticated user profile."""
    with arena_client(config) as client:
        user = client.get_current_user()
        if json_output_enabled():
            emit_json_result(user)
            return
        click.echo(f"User: {user.get('first_name', '')} {user.get('last_name', '')}")
        click.echo(f"Email: {user.get('email', '')}")


@user_group.command("credits")
@click.option(
    "--ledger",
    is_flag=True,
    default=False,
    help="List charges and credits, newest first, with the balance after each.",
)
@click.option(
    "--job", "job_id", type=int, default=None, help="With --ledger: one job only."
)
@click.option(
    "--deployment",
    "deployment_id",
    type=int,
    default=None,
    help="With --ledger: one deployment only.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="With --ledger: most entries.",
)
@click.pass_obj
def user_credits(
    config: CommandConfig,
    ledger: bool,
    job_id: int | None,
    deployment_id: int | None,
    limit: int | None,
) -> None:
    """Show the balance and the cost running jobs have accrued but not been charged.

    --ledger lists each charge by job and deployment instead.
    """
    if not ledger and (
        job_id is not None or deployment_id is not None or limit is not None
    ):
        msg = "--job, --deployment and --limit need --ledger."
        raise click.UsageError(msg)
    with arena_client(config) as client:
        if ledger:
            show_credit_ledger(
                client, job_id=job_id, deployment_id=deployment_id, limit=limit
            )
        else:
            show_credit_balance(client)


def _money(value: float) -> str:
    return f"-${-value:,.2f}" if value < 0 else f"${value:,.2f}"


def _time(value: str) -> str:
    """Naive UTC ISO time to ``YYYY-MM-DD HH:MM`` UTC."""
    return f"{value[:16].replace('T', ' ')} UTC"


def show_credit_balance(client: ArenaClient) -> None:
    """Print the balance, what running jobs and deployments owe, and each of them."""
    try:
        balance = client.get_credit_balance()
    except ArenaAPIError as err:
        # Arena servers without accrued cost answer 404; they still report the balance.
        if err.status_code != 404:
            raise
        emit_result(client.get_user_credits())
        return
    if json_output_enabled():
        # Scripts read this as the bare balance; get_credit_balance() has the rest.
        emit_result(balance["balance"])
        return
    click.echo(f"Balance:                {_money(balance['balance'])}")
    click.echo(f"Accrued, not charged:   {_money(balance['accrued_cost'])}")
    click.echo(f"Balance after accrued:  {_money(balance['balance_after_accrued'])}")
    if balance["accrued"]:
        emit_result(
            [_accrued_row(item) for item in balance["accrued"]],
            columns=["kind", "id", "name", "since", "hourly_cost", "accrued_cost"],
        )


def show_credit_ledger(
    client: ArenaClient,
    *,
    job_id: int | None,
    deployment_id: int | None,
    limit: int | None,
) -> None:
    """Print the organization's charges and credits, newest first.

    :raises click.ClickException: The Arena server has no ledger.
    """
    try:
        entries = client.get_credit_ledger(
            job_id=job_id, deployment_id=deployment_id, limit=limit
        )
    except ArenaAPIError as err:
        if err.status_code != 404:
            raise
        msg = (
            "This Arena server has no credit ledger yet. Upgrade Arena or use the "
            "Usage page in the web app."
        )
        raise click.ClickException(msg) from err
    if json_output_enabled():
        emit_result(entries)
        return
    emit_result(
        [_ledger_row(entry) for entry in entries],
        columns=["time", "item", "kind", "amount", "balance", "description"],
        empty_message="No charges or credits.",
    )


def _accrued_row(item: dict[str, Any]) -> dict[str, Any]:
    # emit_result prints every key, in key order. columns does not drop fields.
    return {
        "kind": item["kind"],
        "id": item["id"],
        "name": item["name"],
        "since": _time(item["since"]),
        "hourly_cost": _money(item["hourly_cost"]),
        "accrued_cost": _money(item["accrued_cost"]),
    }


def _ledger_row(entry: dict[str, Any]) -> dict[str, Any]:
    # emit_result prints every key, in key order. columns does not drop fields.
    return {
        "time": _time(entry["time"]),
        "item": entry["item"],
        "kind": entry["kind"],
        "amount": _money(entry["amount"]),
        "balance": _money(entry["balance"]),
        "description": entry["description"],
    }
