# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowpark session login and password rejection."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from agilerl.arena.snowflake import APPLICATION
from agilerl.arena.snowflake.session import (
    connections_file,
    open_session,
    transient_errors,
)


class Builder:
    seen: dict[str, str] | None = None

    def configs(self, cfg: dict[str, str]) -> Builder:
        Builder.seen = cfg
        return self

    def create(self) -> str:
        return "opened"


class Session:
    builder = Builder()


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    Builder.seen = None
    monkeypatch.setattr(
        "agilerl.arena.snowflake.session._session_class",
        lambda: Session,
    )


def _write_connections(
    tmp_path: Path, body: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "connections.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setattr(
        "agilerl.arena.snowflake.session.connections_file", lambda: path
    )


JWT = """
[default]
account = "acct"
user = "trainer"
authenticator = "SNOWFLAKE_JWT"
private_key_file = "/keys/arena.p8"
"""


class TestOpenSession:
    def test_application_is_set_on_a_key_pair_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(tmp_path, JWT, monkeypatch)

        assert open_session(None) == "opened"
        assert Builder.seen == {
            "connection_name": "default",
            "application": APPLICATION,
        }
        assert APPLICATION == "AgileRL_Arena"

    def test_programmatic_access_token_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [ci]
            account = "acct"
            user = "trainer"
            authenticator = "PROGRAMMATIC_ACCESS_TOKEN"
            token = "pat"
            """,
            monkeypatch,
        )

        assert open_session("ci") == "opened"
        assert Builder.seen is not None
        assert Builder.seen["connection_name"] == "ci"
        assert Builder.seen["application"] == APPLICATION

    def test_token_file_uses_oauth_and_is_reread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        token = tmp_path / "token"
        token.write_text("tok-1\n", encoding="utf-8")
        monkeypatch.setattr("agilerl.arena.snowflake.session.token_path", lambda: token)
        for name, value in {
            "SNOWFLAKE_ACCOUNT": "acct",
            "SNOWFLAKE_HOST": "acct.snowflakecomputing.com",
            "SNOWFLAKE_DATABASE": "ARENA",
            "SNOWFLAKE_SCHEMA": "TRAINING",
        }.items():
            monkeypatch.setenv(name, value)

        assert open_session(None) == "opened"
        assert Builder.seen == {
            "account": "acct",
            "host": "acct.snowflakecomputing.com",
            "database": "ARENA",
            "schema": "TRAINING",
            "authenticator": "oauth",
            "token": "tok-1",
            "application": APPLICATION,
        }
        assert "warehouse" not in Builder.seen
        token.write_text("tok-2\n", encoding="utf-8")

        open_session("ignored")

        assert Builder.seen is not None
        assert Builder.seen["token"] == "tok-2"

    def test_an_empty_token_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        token = tmp_path / "token"
        token.write_text("  \n", encoding="utf-8")
        monkeypatch.setattr("agilerl.arena.snowflake.session.token_path", lambda: token)
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")

        with pytest.raises(ValueError, match="token file is empty"):
            open_session(None)

    def test_a_missing_container_env_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        token = tmp_path / "token"
        token.write_text("tok\n", encoding="utf-8")
        monkeypatch.setattr("agilerl.arena.snowflake.session.token_path", lambda: token)
        monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
        monkeypatch.delenv("SNOWFLAKE_HOST", raising=False)

        with pytest.raises(ValueError, match="SNOWFLAKE_HOST"):
            open_session(None)

    def test_password_auth_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [default]
            account = "acct"
            user = "trainer"
            password = "secret"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="password auth"):
            open_session("default")

    def test_password_alongside_a_key_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [default]
            authenticator = "SNOWFLAKE_JWT"
            private_key_file = "/keys/arena.p8"
            password = "secret"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="password auth"):
            open_session(None)

    def test_missing_connections_file_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        monkeypatch.setattr(
            "agilerl.arena.snowflake.session.connections_file",
            lambda: tmp_path / "connections.toml",
        )

        with pytest.raises(ValueError, match=r"connections\.toml"):
            open_session(None)

    def test_unknown_connection_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(tmp_path, JWT, monkeypatch)

        with pytest.raises(ValueError, match="No \\[missing\\]"):
            open_session("missing")

    def test_other_authenticators_are_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [default]
            authenticator = "externalbrowser"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="not supported"):
            open_session(None)

    def test_missing_sdk_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", None)

        with pytest.raises(ImportError, match="agilerl-arena\\[snowflake\\]"):
            open_session(None)

    def test_a_partial_sdk_import_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snowpark = types.ModuleType("snowflake.snowpark")
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", snowpark)

        with pytest.raises(ImportError, match="agilerl-arena\\[snowflake\\]"):
            open_session(None)

    def test_snowpark_symbols_are_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snowpark = types.ModuleType("snowflake.snowpark")
        snowpark.Session = Session
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", snowpark)
        Builder.seen = None
        _write_connections(tmp_path, JWT, monkeypatch)

        assert open_session(None) == "opened"
        assert Builder.seen is not None
        assert Builder.seen["application"] == APPLICATION

    def test_hash_inside_a_quoted_value_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [default]
            authenticator = "PROGRAMMATIC_ACCESS_TOKEN"  # not "password"
            token = "pat#1"
            password = ""
            """,
            monkeypatch,
        )

        assert open_session(None) == "opened"

    def test_hash_password_after_a_comment_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            # key pair below
            [default]
            authenticator = "SNOWFLAKE_JWT"
            private_key_file = "/keys/arena.p8"
            password = "#secret"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="password auth"):
            open_session(None)


class TestTransientErrors:
    def test_connector_and_network_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        errors = types.ModuleType("snowflake.connector.errors")

        class InterfaceError(Exception):
            pass

        class OperationalError(Exception):
            pass

        errors.InterfaceError = InterfaceError
        errors.OperationalError = OperationalError
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(
            sys.modules, "snowflake.connector", types.ModuleType("snowflake.connector")
        )
        monkeypatch.setitem(sys.modules, "snowflake.connector.errors", errors)

        # Act
        retry_on = transient_errors()

        # Assert
        assert retry_on == (InterfaceError, OperationalError, OSError)


class TestConnectionsFile:
    def test_lives_under_the_home_snowflake_directory(self) -> None:
        assert connections_file() == Path.home() / ".snowflake" / "connections.toml"
