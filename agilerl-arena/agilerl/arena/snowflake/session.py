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


def open_session(connection_name: str | None) -> SnowflakeSession:
    """Open a Snowpark session tagged with :data:`APPLICATION`.

    Outside Snowflake this reads ``~/.snowflake/connections.toml`` and refuses
    password auth. Inside an SPCS container it reads ``/snowflake/session/token``.

    :param connection_name: Named connection. ``None`` uses ``default``.
    :return: The Snowpark session.
    :raises ValueError: The connection file is missing, unknown, or uses a password.
    """
    session_cls = _session_class()
    if token_path().is_file():
        return session_cls.builder.configs(oauth_configs()).create()
    name = connection_name or "default"
    _reject_password_auth(name)
    return session_cls.builder.configs(
        {"connection_name": name, "application": APPLICATION}
    ).create()


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

    def configs(self, options: Mapping[str, str]) -> SessionBuilder:
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


def connections_file() -> Path:
    """Path of the Snowflake CLI connections file."""
    return Path.home() / ".snowflake" / "connections.toml"


def _reject_password_auth(connection_name: str) -> None:
    path = connections_file()
    if not path.is_file():
        msg = (
            f"No Snowflake connections file at {path}. "
            "Add a key pair or programmatic access token under "
            f"[{connection_name}]."
        )
        raise ValueError(msg)
    # Deferred optional extra: snowflake. tomlkit ships with the connector.
    import tomlkit

    document = tomlkit.parse(path.read_text(encoding="utf-8")).unwrap()
    section = document.get(connection_name)
    if not isinstance(section, dict):
        msg = (
            f"No [{connection_name}] connection in {path}. "
            "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
        )
        raise ValueError(msg)
    message = _password_auth_message(section)
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
        return _password_message()
    return (
        f"Authenticator {authenticator!r} is not supported. "
        "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
    )


def _password_message() -> str:
    return (
        "This connection uses password auth. "
        "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
    )
