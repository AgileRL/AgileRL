# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena snowflake setup``, ``check``, ``invite``, ``join``, ``pools``, ``grant``, ``teardown``."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

import click

from agilerl.arena.config import CommandConfig
from agilerl.arena.snowflake import session as snowflake_session
from agilerl.arena.snowflake.account import (
    NEXT_SUBMIT,
    grant_host,
    grant_table,
    invite,
    join_account,
    list_pools,
    run_check,
    set_gpu_cap,
)
from agilerl.arena.snowflake.catalog import one_line, use_arena_warehouse
from agilerl.arena.snowflake.license import (
    DEFAULT_BASE_URL,
    SnowflakeLicenseError,
    arena_api_base_url,
)
from agilerl.arena.snowflake.profile import (
    ADMIN_CONNECTION,
    DATABASE,
    DEFAULT_BROKER_FAMILY,
    DEFAULT_GPU_CREDITS,
    DEFAULT_HEAD_FAMILY,
    DEFAULT_WAREHOUSE_CREDITS,
    JOB_CONNECTION,
    ROLE,
    SCHEMA,
    WAREHOUSE,
    admin_connection_fields,
    admin_connection_rejection,
    missing_key_message,
    missing_key_rejection,
    replaced_connection,
    write_connection,
)
from agilerl.arena.snowflake.session import (
    connection_section,
    load_connection,
    open_admin_session,
    open_session,
    password_auth_message,
    sql_errors,
)
from agilerl.arena.snowflake.setup import SetupRequest, print_setup_sql, run_setup
from agilerl.arena.snowflake.source import SnowflakeSession
from agilerl.arena.snowflake.teardown import run_teardown

RERUN_SETUP = "Re-run arena snowflake setup."


class ClickIO:
    """:class:`~agilerl.arena.snowflake.setup.SetupIO` backed by Click."""

    def echo(self, text: str) -> None:
        """Print a line."""
        click.echo(text)

    def ask(self, text: str, default: str | None, hide: bool) -> str:
        """Prompt once.

        :param text: Prompt.
        :param default: Value used when the answer is empty. ``None`` requires an answer.
        :param hide: When true, the answer is not echoed.
        :return: The answer.
        """
        if default is None:
            return str(click.prompt(text, hide_input=hide))
        return str(
            click.prompt(text, default=default, show_default=False, hide_input=hide)
        )


@contextmanager
def cli_errors(hint: str | None) -> Iterator[None]:
    """Turn expected failures into one Click error.

    :param hint: Line added after a Snowflake SQL error, or ``None``.
    :raises click.ClickException: A setup, Arena, or Snowflake statement failed.
    """
    try:
        yield
    except (ValueError, ImportError, SnowflakeLicenseError) as exc:
        raise click.ClickException(str(exc)) from exc
    except sql_errors() as exc:
        lines = [f"Snowflake said: {one_line(exc)}"]
        if hint is not None:
            lines.append(hint)
        raise click.ClickException("\n".join(lines)) from exc


@contextmanager
def admin_session(connection_name: str) -> Iterator[SnowflakeSession]:
    """Open an admin session on ``arena_wh`` when it exists, and close it after.

    :param connection_name: Section name in the connections file.
    """
    with closing(open_admin_session(connection_name)) as session:
        use_arena_warehouse(session)
        yield session


@click.command("setup")
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help=f"Admin connection. Omit to use [{ADMIN_CONNECTION}] or sign in with a browser.",
)
@click.option(
    "--broker-family",
    default=None,
    help=f"Broker pool family. Default {DEFAULT_BROKER_FAMILY}.",
)
@click.option(
    "--head-family",
    default=None,
    help=f"Head pool family. Default {DEFAULT_HEAD_FAMILY}.",
)
@click.option(
    "--gpu-family",
    "gpu_families",
    multiple=True,
    help="GPU pool family. Repeat for each one. "
    "Default: the families already set up, else the smallest GPU family here.",
)
@click.option(
    "--max-gpus",
    default=None,
    type=click.IntRange(min=1),
    help="GPUs each GPU pool can hold. Default: the saved cap, else two nodes.",
)
@click.option(
    "--warehouse-credits",
    default=None,
    type=click.IntRange(min=1),
    help=f"Warehouse credit cap per month. Default {DEFAULT_WAREHOUSE_CREDITS}.",
)
@click.option(
    "--gpu-credits",
    default=None,
    type=click.IntRange(min=1),
    help=f"Credit budget per month for all Arena pools. Default {DEFAULT_GPU_CREDITS}.",
)
@click.option(
    "--notify-email",
    default=None,
    help="Address for budget alerts. Must be verified in Snowflake.",
)
@click.option(
    "--key",
    "key_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Existing PKCS#8 private key. Omit and setup creates one.",
)
@click.option(
    "--rotate-key",
    is_flag=True,
    help="Replace the public key on arena_job_user with the local key.",
)
@click.option("--org-key", default=None, help="Arena organisation key. Omit to prompt.")
@click.option(
    "--external-user-id", default=None, help="Arena external user id. Omit to prompt."
)
@click.option(
    "--image-tag",
    default=None,
    help="Trainer image tag. 32.6.0, ml/v32.6.0, or 32.6.0-py313-cu130.",
)
@click.option(
    "--print-sql",
    is_flag=True,
    help="Print the statements for a fresh account and exit. Creates nothing.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip prompts. Requires --connection; the first run also needs Arena credentials.",
)
def setup(
    connection_name: str | None,
    broker_family: str | None,
    head_family: str | None,
    gpu_families: tuple[str, ...],
    max_gpus: int | None,
    warehouse_credits: int | None,
    gpu_credits: int | None,
    notify_email: str | None,
    key_path: Path | None,
    rotate_key: bool,
    org_key: str | None,
    external_user_id: str | None,
    image_tag: str | None,
    print_sql: bool,
    yes: bool,
) -> None:
    """Create the Arena objects, copy the trainer image, and check the GPU driver."""
    config: CommandConfig = click.get_current_context().find_root().obj
    with cli_errors(RERUN_SETUP):
        base_url = arena_api_base_url(
            config.base_url or os.environ.get("ARENA_BASE_URL") or DEFAULT_BASE_URL
        )
        request = SetupRequest(
            account="",
            broker_family=broker_family,
            head_family=head_family,
            gpu_families=gpu_families,
            max_gpus=max_gpus,
            warehouse_credits=warehouse_credits,
            gpu_credits=gpu_credits,
            key_path=key_path.expanduser() if key_path is not None else None,
            rotate_key=rotate_key,
            yes=yes,
            notify_email=notify_email,
            org_key=org_key or config.org_key or os.environ.get("ARENA_ORG_KEY"),
            external_user_id=external_user_id
            or config.external_user_id
            or os.environ.get("ARENA_EXTERNAL_USER_ID"),
            base_url=base_url,
            image_tag=image_tag,
        )
        if print_sql:
            click.echo(print_setup_sql(request), nl=False)
            return
        if yes and connection_name is None:
            msg = "--yes requires --connection."
            raise click.UsageError(msg)
        account, name = _setup_connection(connection_name)
        with admin_session(name) as session:
            run_setup(session, replace(request, account=account), ClickIO())


def _setup_connection(connection_name: str | None) -> tuple[str, str]:
    if connection_name is not None:
        return _account_of(connection_name), connection_name
    section = connection_section(ADMIN_CONNECTION)
    if section is not None:
        account = _account_of(ADMIN_CONNECTION)
        click.echo(
            f"Signing in with [{ADMIN_CONNECTION}] from "
            f"{snowflake_session.connections_file()} "
            f"(user {section.get('user', '?')} on {account})."
        )
        return account, ADMIN_CONNECTION
    account = str(click.prompt("Account identifier")).strip()
    login_name = str(click.prompt("Snowflake login name")).strip()
    write_connection(ADMIN_CONNECTION, admin_connection_fields(account, login_name))
    click.echo("Opening a browser to sign in. Use a user that has ACCOUNTADMIN.")
    return account, ADMIN_CONNECTION


def _account_of(connection_name: str) -> str:
    account = str(load_connection(connection_name).get("account", "")).strip()
    if account == "":
        msg = f"Connection {connection_name} has no account."
        raise ValueError(msg)
    return account


def _admin_target(connection_name: str | None) -> str:
    name = connection_name or ADMIN_CONNECTION
    rejection = admin_connection_rejection(name)
    if rejection is not None:
        raise ValueError(rejection)
    return name


@click.command("check")
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help=f"Connection name. Default {JOB_CONNECTION}.",
)
def check(connection_name: str | None) -> None:
    """Check the Arena objects as arena_trainer, save the image, and run the driver check."""
    name = connection_name or JOB_CONNECTION
    with cli_errors(RERUN_SETUP):
        rejection = missing_key_rejection(name)
        if rejection is not None:
            raise ValueError(rejection)
        with closing(open_session(name)) as session:
            click.echo(run_check(session))


@click.command("invite")
@click.argument("user")
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help=f"Admin connection. Default {ADMIN_CONNECTION}.",
)
def invite_command(user: str, connection_name: str | None) -> None:
    """Grant arena_trainer to USER. Does not create the user."""
    with cli_errors(None):
        name = _admin_target(connection_name)
        account = _account_of(name)
        with admin_session(name) as session:
            click.echo(invite(session, user, account))


@click.command("join")
@click.option("--account", default=None, help="Account identifier.")
@click.option("--user", default=None, help="Snowflake login name.")
@click.option("--token", default=None, help="Programmatic access token.")
@click.option(
    "--key",
    "key_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="PKCS#8 private key for this user.",
)
@click.option("--yes", is_flag=True, help=f"Replace [{JOB_CONNECTION}] without asking.")
def join_command(
    account: str | None,
    user: str | None,
    token: str | None,
    key_path: Path | None,
    yes: bool,
) -> None:
    """Write the [arena] login for a user who already has arena_trainer."""
    if token is not None and key_path is not None:
        msg = "Pass one of --token or --key."
        raise click.UsageError(msg)
    with cli_errors(None):
        account_name = account or str(click.prompt("Account identifier")).strip()
        user_name = user or str(click.prompt("Snowflake login name")).strip()
        fields = _join_fields(account_name, user_name, _join_auth(token, key_path))
        replaced = replaced_connection(JOB_CONNECTION, fields)
        if replaced is not None:
            click.echo(replaced)
            if not yes and not click.confirm("Replace it?", default=False):
                msg = f"[{JOB_CONNECTION}] was left as it is."
                raise ValueError(msg)
        write_connection(JOB_CONNECTION, fields)
        with closing(open_session(JOB_CONNECTION)) as session:
            join_account(session, user_name)
    click.echo(f"Joined {account_name} as {user_name}.\n{NEXT_SUBMIT}")


def _join_auth(token: str | None, key_path: Path | None) -> dict[str, str]:
    if token is not None:
        return {"authenticator": "PROGRAMMATIC_ACCESS_TOKEN", "token": token}
    if key_path is not None:
        return {
            "authenticator": "SNOWFLAKE_JWT",
            "private_key_file": str(_join_key(key_path)),
        }
    method = str(click.prompt("Sign in with [token/key]")).strip().lower()
    if method == "password":
        raise ValueError(password_auth_message())
    if method == "token":
        return _join_auth(
            str(click.prompt("Programmatic access token", hide_input=True)), None
        )
    if method == "key":
        return _join_auth(None, Path(str(click.prompt("Private key path"))))
    msg = "Sign in with token or key."
    raise ValueError(msg)


def _join_key(raw: Path) -> Path:
    path = raw.expanduser()
    if not path.is_file():
        raise ValueError(missing_key_message(path))
    return path


def _join_fields(account: str, user: str, auth: dict[str, str]) -> dict[str, str]:
    return {
        "account": account,
        "user": user,
        "role": ROLE,
        "warehouse": WAREHOUSE,
        "database": DATABASE,
        "schema": SCHEMA,
        **auth,
    }


@click.group("pools", invoke_without_command=True)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help=f"Admin connection. Default {ADMIN_CONNECTION}.",
)
@click.pass_context
def pools_group(ctx: click.Context, connection_name: str | None) -> None:
    """List compute pools, or set a GPU family's cap."""
    ctx.obj = connection_name
    if ctx.invoked_subcommand is None:
        _run_pools(connection_name, None, None)


@pools_group.command("set")
@click.argument("family")
@click.option(
    "--max-gpus",
    required=True,
    type=click.IntRange(min=1),
    help="GPUs this family can hold. The pool's MAX_NODES is the node count that fits.",
)
@click.pass_context
def pools_set(ctx: click.Context, family: str, max_gpus: int) -> None:
    """Set FAMILY's GPU cap. Creates the pool when that family is new."""
    connection = ctx.obj if isinstance(ctx.obj, str) else None
    _run_pools(connection, family, max_gpus)


def _run_pools(
    connection_name: str | None, family: str | None, max_gpus: int | None
) -> None:
    with cli_errors(RERUN_SETUP):
        name = _admin_target(connection_name)
        with admin_session(name) as session:
            if family is None or max_gpus is None:
                click.echo(list_pools(session))
                return
            click.echo(set_gpu_cap(session, family, max_gpus))


@click.command("grant")
@click.option(
    "--table",
    "tables",
    multiple=True,
    help="database.schema.table that jobs read. Repeat for each one.",
)
@click.option(
    "--host",
    "hosts",
    multiple=True,
    help="Host jobs reach on 443, such as api.example.com. Repeat for each one.",
)
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help=f"Admin connection. Default {ADMIN_CONNECTION}.",
)
def grant_command(
    tables: tuple[str, ...], hosts: tuple[str, ...], connection_name: str | None
) -> None:
    """Let Arena jobs read a table or reach a host."""
    if not tables and not hosts:
        msg = "Pass --table or --host."
        raise click.UsageError(msg)
    with cli_errors(RERUN_SETUP):
        name = _admin_target(connection_name)
        with admin_session(name) as session:
            for table in tables:
                click.echo(grant_table(session, table))
            for host in hosts:
                click.echo(grant_host(session, host))


@click.command("teardown")
@click.option(
    "--connection",
    "connection_name",
    default=None,
    help=f"Admin connection. Default {ADMIN_CONNECTION}.",
)
@click.option("--yes", is_flag=True, help="Drop without asking.")
def teardown_command(connection_name: str | None, yes: bool) -> None:
    """Drop every Arena object in the account and the local Arena logins."""
    with cli_errors(None):
        name = _admin_target(connection_name)
        account = _account_of(name)
        with admin_session(name) as session:
            run_teardown(session, account, yes, ClickIO())
