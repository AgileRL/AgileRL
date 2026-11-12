# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowpark sessions for the Arena Snowflake connector name."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from agilerl.arena.snowflake import APPLICATION
from agilerl.arena.snowflake.source import SnowflakeSession

BROWSER_SIGN_IN = frozenset({"EXTERNALBROWSER", "OAUTH_AUTHORIZATION_CODE"})


def open_session(connection_name: str | None) -> SnowflakeSession:
    """Open a Snowpark session tagged with :data:`APPLICATION`.

    Outside Snowflake this reads :func:`connections_file` and refuses password
    auth. Inside an SPCS container it reads ``/snowflake/session/token``.

    :param connection_name: Named connection. ``None`` uses ``default``.
    :return: The Snowpark session.
    :raises ValueError: The connection file is missing, unknown, or uses a password.
    """
    session_cls = _session_class()
    if token_path().is_file():
        return session_cls.builder.configs(oauth_configs()).create()
    name = connection_name or "default"
    _reject_password_auth(name)
    return session_cls.builder.configs(named_configs(name)).create()


def named_configs(connection_name: str) -> dict[str, object]:
    """Snowpark settings that open one section of :func:`connections_file`.

    The connector caches the connections file on first read. Passing its path
    makes it read the file again, so a section written in this process is seen.

    :param connection_name: Section name.
    :return: Snowpark ``Session.builder.configs`` values.
    """
    return {
        "connection_name": connection_name,
        "connections_file_path": connections_file(),
        "application": APPLICATION,
    }


def token_path() -> Path:
    """OAuth token file Snowflake mounts in a container."""
    return Path("/snowflake/session/token")


def oauth_configs() -> dict[str, str]:
    """Connector settings for the SPCS OAuth token.

    The warehouse is left unset so the job's ``QUERY_WAREHOUSE`` applies.
    The token file is read on every call.

    :return: Snowpark ``Session.builder.configs`` values.
    :raises ValueError: A required environment variable or the token is empty.
    """
    token = token_path().read_text(encoding="utf-8").strip()
    if token == "":
        msg = "Snowflake session token file is empty"
        raise ValueError(msg)
    configs = {
        key.removeprefix("SNOWFLAKE_").lower(): _env(key)
        for key in (
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_HOST",
            "SNOWFLAKE_DATABASE",
            "SNOWFLAKE_SCHEMA",
        )
    }
    configs["authenticator"] = "oauth"
    configs["token"] = token
    configs["application"] = APPLICATION
    return configs


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value == "":
        msg = f"{name} is required inside a Snowflake container"
        raise ValueError(msg)
    return value


class SessionBuilder(Protocol):
    """Snowpark ``Session.builder`` chain."""

    def configs(self, options: Mapping[str, object]) -> SessionBuilder:
        """Apply connector options."""
        ...

    def create(self) -> SnowflakeSession:
        """Open the session."""
        ...


class SnowparkSessionClass(Protocol):
    """Snowpark ``Session`` class object."""

    builder: SessionBuilder


def _session_class() -> SnowparkSessionClass:
    # Deferred optional extra: snowflake.
    try:
        from snowflake.snowpark import Session
    except ImportError as exc:
        msg = (
            "Snowflake support needs the snowflake extra: "
            "pip install 'agilerl-arena[snowflake]'"
        )
        raise ImportError(msg) from exc
    return Session


def transient_errors() -> tuple[type[Exception], ...]:
    """Connector and network errors a stage upload retries.

    :return: Exception types worth another attempt. Others raise at once.
    """
    # Deferred optional extra: snowflake.
    from snowflake.connector.errors import InterfaceError, OperationalError

    return (InterfaceError, OperationalError, OSError)


def sql_errors() -> tuple[type[Exception], ...]:
    """Connector and Snowpark errors a failed statement raises.

    :return: Exception types. Empty without the snowflake extra, since no
        statement can have run.
    """
    # Deferred optional extra: snowflake.
    try:
        from snowflake.connector.errors import DatabaseError
        from snowflake.snowpark.exceptions import SnowparkSQLException
    except ImportError:
        return ()
    return (DatabaseError, SnowparkSQLException)


def connections_file() -> Path:
    """``connections.toml`` in the directory snowflake-connector-python reads.

    ``$SNOWFLAKE_HOME`` (default ``~/.snowflake``) when that directory exists,
    otherwise the platform config directory for ``snowflake``.

    :return: The file path. The file may not exist yet.
    """
    # Deferred optional extra: snowflake. platformdirs ships with the connector.
    from platformdirs import PlatformDirs

    home = Path(os.environ.get("SNOWFLAKE_HOME", "~/.snowflake/")).expanduser()
    if home.exists():
        return home / "connections.toml"
    config = PlatformDirs(appname="snowflake", appauthor=False).user_config_path
    return Path(config) / "connections.toml"


def connection_section(connection_name: str) -> dict[str, Any] | None:
    """One section of :func:`connections_file`.

    :param connection_name: Section name.
    :return: The section fields, or ``None`` when the file or section is absent.
    """
    path = connections_file()
    if not path.is_file():
        return None
    # Deferred optional extra: snowflake. tomlkit ships with the connector.
    import tomlkit

    document = tomlkit.parse(path.read_text(encoding="utf-8")).unwrap()
    section = document.get(connection_name)
    return section if isinstance(section, dict) else None


def load_connection(connection_name: str) -> dict[str, Any]:
    """Read one connection from :func:`connections_file`.

    :param connection_name: Section name.
    :return: The section fields.
    :raises ValueError: The file or the section is missing.
    """
    path = connections_file()
    if not path.is_file():
        msg = (
            f"No Snowflake connections file at {path}. "
            "Add a key pair or programmatic access token under "
            f"[{connection_name}]."
        )
        raise ValueError(msg)
    section = connection_section(connection_name)
    if section is None:
        msg = (
            f"No [{connection_name}] connection in {path}. "
            "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
        )
        raise ValueError(msg)
    return section


def open_admin_session(connection_name: str) -> SnowflakeSession:
    """Open a session that may use browser sign-in.

    Password auth is rejected. Snowflake OAuth and ``externalbrowser`` are
    accepted here and only here. :func:`open_session` still rejects them.

    :param connection_name: Section name in the connections file.
    :return: The Snowpark session.
    :raises ValueError: The connection is missing or uses a password.
    """
    session_cls = _session_class()
    section = load_connection(connection_name)
    if section.get("password"):
        raise ValueError(password_auth_message())
    authenticator = str(section.get("authenticator", "")).strip().upper()
    if authenticator not in BROWSER_SIGN_IN:
        message = _password_auth_message(section)
        if message is not None:
            raise ValueError(message)
    return session_cls.builder.configs(named_configs(connection_name)).create()


def _reject_password_auth(connection_name: str) -> None:
    message = _password_auth_message(load_connection(connection_name))
    if message is not None:
        raise ValueError(message)


def _password_auth_message(section: Mapping[str, Any]) -> str | None:
    authenticator = str(section.get("authenticator", "")).strip().upper()
    has_password = bool(section.get("password"))
    has_key = any(
        section.get(field)
        for field in ("private_key_file", "private_key_path", "private_key")
    )
    has_token = bool(section.get("token"))
    if authenticator == "PROGRAMMATIC_ACCESS_TOKEN" or (has_token and not has_password):
        return None
    if (authenticator == "SNOWFLAKE_JWT" or has_key) and not has_password:
        return None
    if has_password or authenticator in {"", "SNOWFLAKE", "USERNAME_PASSWORD_MFA"}:
        return password_auth_message()
    return (
        f"Authenticator {authenticator!r} is not supported. "
        "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
    )


def password_auth_message() -> str:
    """Sentence used wherever password auth is rejected."""
    return (
        "This connection uses password auth. "
        "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
    )
