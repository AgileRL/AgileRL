# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowpark sessions for the Arena Snowflake connector name."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agilerl.arena.snowflake import APPLICATION
from agilerl.arena.snowflake.source import SnowflakeSession

KEY_FIELDS = ("private_key_file", "private_key_path", "private_key")
PASSWORD_AUTHENTICATORS = {"", "SNOWFLAKE", "USERNAME_PASSWORD_MFA"}


def open_session(connection_name: str | None) -> SnowflakeSession:
    """Open a Snowpark session tagged with :data:`APPLICATION`.

    Outside Snowflake this reads ``~/.snowflake/connections.toml`` and refuses
    password auth. Inside an ML Job it returns the job's active session.

    :param connection_name: Named connection. ``None`` uses ``default``.
    :return: The Snowpark session.
    :raises ValueError: The connection file is missing, unknown, or uses a password.
    """
    session_cls, get_active_session, session_error = _snowpark()
    try:
        active = get_active_session()
    except session_error:
        active = None
    if active is not None:
        # `application` is a connector login field. An ML Job session is already open.
        return active
    name = connection_name or "default"
    _reject_password_auth(name)
    return session_cls.builder.configs(
        {"connection_name": name, "application": APPLICATION}
    ).create()


def _snowpark() -> tuple[Any, Any, type[BaseException]]:
    # Deferred optional extra: snowflake.
    try:
        from snowflake.snowpark import Session
        from snowflake.snowpark.context import get_active_session
        from snowflake.snowpark.exceptions import SnowparkSessionException
    except ImportError as exc:
        msg = (
            "Snowflake support needs the snowflake extra: "
            "pip install 'agilerl-arena[snowflake]'"
        )
        raise ImportError(msg) from exc
    return Session, get_active_session, SnowparkSessionException


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
    document = _load_toml(path.read_text(encoding="utf-8"))
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
    has_key = any(section.get(field) for field in KEY_FIELDS)
    has_token = bool(section.get("token"))
    if authenticator == "PROGRAMMATIC_ACCESS_TOKEN" or (has_token and not has_password):
        return None
    if (authenticator == "SNOWFLAKE_JWT" or has_key) and not has_password:
        return None
    if has_password or authenticator in PASSWORD_AUTHENTICATORS:
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


def _load_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:
        return _parse_connections_toml(text)
    loaded = tomllib.loads(text)
    if not isinstance(loaded, dict):
        msg = "connections.toml must be a table of connections"
        raise ValueError(msg)
    return loaded


def _parse_connections_toml(text: str) -> dict[str, dict[str, str]]:
    """Parse the flat connection tables Python 3.10 can read without tomllib."""
    document: dict[str, dict[str, str]] = {}
    section: dict[str, str] | None = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            section = {}
            document[name] = section
            continue
        if section is None or "=" not in line:
            msg = f"connections.toml line {line_number} is not a connection key"
            raise ValueError(msg)
        key, value = line.split("=", 1)
        section[key.strip()] = _toml_string(value.strip(), line_number)
    return document


def _toml_string(value: str, line_number: int) -> str:
    if value.startswith(('"""', "'''")):
        msg = (
            f"connections.toml line {line_number} spans lines. "
            "Point private_key_file at the key instead."
        )
        raise ValueError(msg)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        quote = value[0]
        body = value[1:-1]
        if quote == '"':
            body = (
                body.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\\\", "\\")
            )
        return body
    return value
