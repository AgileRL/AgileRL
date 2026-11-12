# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowpark session login and password rejection."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from platformdirs import PlatformDirs

from agilerl.arena.snowflake import APPLICATION
from agilerl.arena.snowflake.session import (
    connection_section,
    connections_file,
    open_admin_session,
    open_session,
    sql_errors,
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
            "connections_file_path": tmp_path / "connections.toml",
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


class TestOpenAdminSession:
    @pytest.mark.parametrize(
        "authenticator", ["OAUTH_AUTHORIZATION_CODE", "externalbrowser"]
    )
    def test_browser_sign_in_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authenticator: str
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            f"""
            [arena-admin]
            account = "ORGNAME-ACCOUNT"
            user = "NICK"
            authenticator = "{authenticator}"
            """,
            monkeypatch,
        )

        assert open_admin_session("arena-admin") == "opened"
        assert Builder.seen == {
            "connection_name": "arena-admin",
            "connections_file_path": tmp_path / "connections.toml",
            "application": APPLICATION,
        }

    def test_job_sessions_still_reject_oauth_browser_sign_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [arena-admin]
            account = "ORGNAME-ACCOUNT"
            authenticator = "OAUTH_AUTHORIZATION_CODE"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="not supported"):
            open_session("arena-admin")

    def test_a_password_is_still_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [arena-admin]
            account = "ORGNAME-ACCOUNT"
            user = "NICK"
            authenticator = "externalbrowser"
            password = "secret"
            role = "ACCOUNTADMIN"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="uses password auth"):
            open_admin_session("arena-admin")

    def test_a_key_pair_admin_connection_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [arena-admin]
            account = "ORGNAME-ACCOUNT"
            user = "NICK"
            authenticator = "SNOWFLAKE_JWT"
            private_key_file = "/keys/admin.p8"
            role = "ACCOUNTADMIN"
            """,
            monkeypatch,
        )

        assert open_admin_session("arena-admin") == "opened"

    def test_an_unsupported_authenticator_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch)
        _write_connections(
            tmp_path,
            """
            [arena-admin]
            account = "ORGNAME-ACCOUNT"
            user = "NICK"
            authenticator = "okta"
            """,
            monkeypatch,
        )

        with pytest.raises(ValueError, match="not supported") as raised:
            open_admin_session("arena-admin")

        assert str(raised.value) == (
            "Authenticator 'OKTA' is not supported. "
            "Use a key pair (SNOWFLAKE_JWT) or a programmatic access token."
        )
        assert Builder.seen is None

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


class TestSqlErrors:
    def test_connector_and_snowpark_statement_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        errors = types.ModuleType("snowflake.connector.errors")
        exceptions = types.ModuleType("snowflake.snowpark.exceptions")

        class DatabaseError(Exception):
            pass

        class SnowparkSQLException(Exception):
            pass

        errors.DatabaseError = DatabaseError
        exceptions.SnowparkSQLException = SnowparkSQLException
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(
            sys.modules, "snowflake.connector", types.ModuleType("snowflake.connector")
        )
        monkeypatch.setitem(sys.modules, "snowflake.connector.errors", errors)
        monkeypatch.setitem(
            sys.modules, "snowflake.snowpark", types.ModuleType("snowflake.snowpark")
        )
        monkeypatch.setitem(sys.modules, "snowflake.snowpark.exceptions", exceptions)

        # Act
        caught = sql_errors()

        # Assert
        assert caught == (DatabaseError, SnowparkSQLException)

    def test_nothing_is_caught_without_the_snowflake_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "snowflake.connector.errors", None)

        assert sql_errors() == ()


class TestConnectionsFile:
    def test_snowflake_home_wins_when_it_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "sf"
        home.mkdir()
        monkeypatch.setenv("SNOWFLAKE_HOME", str(home))

        assert connections_file() == home / "connections.toml"

    def test_home_snowflake_directory_when_it_exists(
        self, arena_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SNOWFLAKE_HOME", raising=False)
        (arena_home / ".snowflake").mkdir()

        assert connections_file() == arena_home / ".snowflake" / "connections.toml"

    def test_platform_config_directory_without_home_snowflake(
        self, arena_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SNOWFLAKE_HOME", raising=False)
        monkeypatch.setattr(
            PlatformDirs,
            "user_config_path",
            property(lambda dirs: arena_home / "config" / dirs.appname),
        )

        found = connections_file()

        assert found == arena_home / "config" / "snowflake" / "connections.toml"


class TestConnectionSection:
    def test_returns_the_named_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_connections(tmp_path, JWT, monkeypatch)

        assert connection_section("default") == {
            "account": "acct",
            "user": "trainer",
            "authenticator": "SNOWFLAKE_JWT",
            "private_key_file": "/keys/arena.p8",
        }

    def test_missing_file_or_section_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.snowflake.session.connections_file",
            lambda: tmp_path / "connections.toml",
        )
        assert connection_section("default") is None

        _write_connections(tmp_path, JWT, monkeypatch)
        assert connection_section("arena") is None
