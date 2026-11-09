# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowpark session login and password rejection."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest

from agilerl.arena.snowflake import APPLICATION
from agilerl.arena.snowflake.session import (
    _load_toml,
    _parse_connections_toml,
    open_session,
)


class SnowparkSessionException(Exception):
    pass


class Builder:
    seen: dict[str, str] | None = None

    def configs(self, cfg: dict[str, str]) -> Builder:
        Builder.seen = cfg
        return self

    def create(self) -> str:
        return "opened"


class Session:
    builder = Builder()


def _install(monkeypatch: pytest.MonkeyPatch, active: str | None) -> None:
    Builder.seen = None

    def get_active_session() -> str:
        if active is None:
            msg = "no session"
            raise SnowparkSessionException(msg)
        return active

    monkeypatch.setattr(
        "agilerl.arena.snowflake.session._snowpark",
        lambda: (Session, get_active_session, SnowparkSessionException),
    )


def _write_connections(
    tmp_path: Path, body: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    directory = tmp_path / ".snowflake"
    directory.mkdir()
    (directory / "connections.toml").write_text(body, encoding="utf-8")


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
        _install(monkeypatch, None)
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
        _install(monkeypatch, None)
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

    def test_in_job_session_is_the_active_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, "active")
        monkeypatch.setenv("HOME", str(tmp_path))
        Builder.seen = None

        assert open_session(None) == "active"
        assert Builder.seen is None

    def test_password_auth_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, None)
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
        _install(monkeypatch, None)
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
        _install(monkeypatch, None)
        monkeypatch.setenv("HOME", str(tmp_path))

        with pytest.raises(ValueError, match=r"connections\.toml"):
            open_session(None)

    def test_unknown_connection_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, None)
        _write_connections(tmp_path, JWT, monkeypatch)

        with pytest.raises(ValueError, match="No \\[missing\\]"):
            open_session("missing")

    def test_other_authenticators_are_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(monkeypatch, None)
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
        context = types.ModuleType("snowflake.snowpark.context")
        snowpark.Session = Session
        context.get_active_session = lambda: None
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", snowpark)
        monkeypatch.setitem(sys.modules, "snowflake.snowpark.context", context)
        monkeypatch.setitem(sys.modules, "snowflake.snowpark.exceptions", None)

        with pytest.raises(ImportError, match="agilerl-arena\\[snowflake\\]"):
            open_session(None)

    def test_snowpark_symbols_are_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snowpark = types.ModuleType("snowflake.snowpark")
        context = types.ModuleType("snowflake.snowpark.context")
        exceptions = types.ModuleType("snowflake.snowpark.exceptions")
        snowpark.Session = Session
        exceptions.SnowparkSessionException = SnowparkSessionException

        def get_active_session() -> str:
            msg = "no session"
            raise SnowparkSessionException(msg)

        context.get_active_session = get_active_session
        monkeypatch.setitem(sys.modules, "snowflake", types.ModuleType("snowflake"))
        monkeypatch.setitem(sys.modules, "snowflake.snowpark", snowpark)
        monkeypatch.setitem(sys.modules, "snowflake.snowpark.context", context)
        monkeypatch.setitem(sys.modules, "snowflake.snowpark.exceptions", exceptions)
        Builder.seen = None
        _write_connections(tmp_path, JWT, monkeypatch)

        assert open_session(None) == "opened"
        assert Builder.seen is not None
        assert Builder.seen["application"] == APPLICATION


class TestParseConnectionsToml:
    def test_quoted_and_bare_values(self) -> None:
        document = _parse_connections_toml(
            """
            # comment
            [default]
            account = acct
            user = "trainer"
            note = "line\\n"
            """
        )

        assert document["default"]["account"] == "acct"
        assert document["default"]["user"] == "trainer"
        assert document["default"]["note"] == "line\n"

    def test_a_multiline_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="spans lines"):
            _parse_connections_toml('[default]\nkey = """secret"""\n')

    def test_a_key_outside_a_section_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="line 1"):
            _parse_connections_toml("account = acct\n")

    def test_loader_falls_back_without_tomllib(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def guarded(name: str, *args: object, **kwargs: object) -> object:
            if name == "tomllib":
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guarded)

        loaded = _load_toml("[default]\ntoken = 'pat'\n")

        assert loaded["default"]["token"] == "pat"

    def test_a_non_table_document_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tomllib

        monkeypatch.setattr(tomllib, "loads", lambda text: ["not", "a", "table"])

        with pytest.raises(ValueError, match="table of connections"):
            _load_toml("[]")
