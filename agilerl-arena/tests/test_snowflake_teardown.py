# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""arena snowflake teardown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agilerl.arena.snowflake.profile import (
    admin_connection_fields,
    job_connection_fields,
    write_connection,
)
from agilerl.arena.snowflake.session import connection_section
from agilerl.arena.snowflake.setup_cli import teardown_command
from agilerl.arena.snowflake.teardown import (
    TeardownPlan,
    plan_teardown,
    remove_local,
    run_teardown,
    teardown_statements,
)
from tests.test_snowflake_setup import (
    ACCOUNT,
    PROFILE_DOC,
    FakeSession,
    Row,
    Scripted,
    admin_routes,
    paths,
)

FULL_DROP = [
    'ALTER COMPUTE POOL "ARENA_PULSAR_POOL" STOP ALL',
    'DROP COMPUTE POOL "ARENA_PULSAR_POOL"',
    'ALTER COMPUTE POOL "ARENA_HEAD_POOL" STOP ALL',
    'DROP COMPUTE POOL "ARENA_HEAD_POOL"',
    'ALTER COMPUTE POOL "ARENA_GPU_GPU_NV_S" STOP ALL',
    'DROP COMPUTE POOL "ARENA_GPU_GPU_NV_S"',
    "DROP EXTERNAL ACCESS INTEGRATION IF EXISTS arena_hf_eai",
    "DROP EXTERNAL ACCESS INTEGRATION IF EXISTS arena_api_eai",
    "DROP DATABASE IF EXISTS arena",
    "DROP USER IF EXISTS arena_job_user",
    "DROP ROLE IF EXISTS arena_trainer",
    "DROP WAREHOUSE IF EXISTS arena_wh",
    "DROP RESOURCE MONITOR IF EXISTS arena_monitor",
]


def arena_account(
    pools: list[Row] | None = None,
) -> list[tuple[object, object]]:
    """An account setup ran on, plus pools setup.json does not name."""
    if pools is None:
        pools = [
            Row(name="ARENA_PULSAR_POOL"),
            Row(name="ARENA_HEAD_POOL"),
            Row(name="ARENA_GPU_GPU_NV_S"),
            Row(name="ARENA_GPU_GPU_NV_M"),
            Row(name="ANALYTICS_POOL"),
        ]
    return admin_routes(
        [
            (lambda s: s.startswith("SHOW DATABASES"), [Row(name="ARENA")]),
            (
                lambda s: s.startswith("SHOW EXTERNAL ACCESS"),
                [
                    Row(name="ARENA_HF_EAI", enabled="true"),
                    Row(name="ARENA_API_EAI", enabled="true"),
                ],
            ),
            (lambda s: s.startswith("SELECT $1"), [Row(DOC=json.dumps(PROFILE_DOC))]),
            (lambda s: s == "SHOW COMPUTE POOLS", pools),
            (lambda s: s.startswith("SHOW USERS"), [Row(name="ARENA_JOB_USER")]),
            (lambda s: s.startswith("SHOW ROLES"), [Row(name="ARENA_TRAINER")]),
            (lambda s: s.startswith("SHOW WAREHOUSES"), [Row(name="ARENA_WH")]),
            (
                lambda s: s.startswith("SHOW RESOURCE MONITORS"),
                [Row(name="ARENA_MONITOR")],
            ),
        ]
    )


def foreign_account() -> list[tuple[object, object]]:
    """A database named arena that holds someone else's schemas."""
    return admin_routes(
        [
            (lambda s: s.startswith("SHOW DATABASES"), [Row(name="ARENA")]),
            (lambda s: s.startswith("SHOW SCHEMAS"), [Row(name="SALES")]),
            (
                lambda s: s == "SHOW COMPUTE POOLS",
                [Row(name="ARENA_HEAD_POOL"), Row(name="ARENA_GPU_GPU_NV_S")],
            ),
        ]
    )


def drops(session: FakeSession) -> list[str]:
    return [s for s in session.statements if s.startswith(("DROP ", "ALTER COMPUTE"))]


class TestTeardownStatements:
    def test_stops_and_drops_pools_before_the_rest(self) -> None:
        plan = TeardownPlan(
            pools=("ARENA_PULSAR_POOL", "ARENA_HEAD_POOL", "ARENA_GPU_GPU_NV_S"),
            integrations=("arena_hf_eai", "arena_api_eai"),
            database=True,
            foreign_database=False,
            user=True,
            role=True,
            warehouse=True,
            monitor=True,
        )

        assert teardown_statements(plan) == FULL_DROP

    def test_quotes_pool_names(self) -> None:
        plan = TeardownPlan(
            ('arena_gpu_"x" pool',), (), False, False, False, False, False, False
        )

        assert teardown_statements(plan) == [
            'ALTER COMPUTE POOL "arena_gpu_""x"" pool" STOP ALL',
            'DROP COMPUTE POOL "arena_gpu_""x"" pool"',
        ]

    def test_an_empty_plan_drops_nothing(self) -> None:
        plan = TeardownPlan((), (), False, False, False, False, False, False)

        assert teardown_statements(plan) == []


class TestPlanTeardown:
    def test_finds_only_the_cpu_pools_and_pools_in_setup_json(self) -> None:
        plan = plan_teardown(FakeSession(arena_account()))

        assert "ARENA_GPU_GPU_NV_M" not in plan.pools

        assert plan.pools == (
            "ARENA_PULSAR_POOL",
            "ARENA_HEAD_POOL",
            "ARENA_GPU_GPU_NV_S",
        )
        assert plan.integrations == ("arena_hf_eai", "arena_api_eai")
        assert plan.database is True
        assert plan.foreign_database is False

    def test_a_foreign_database_stays(self) -> None:
        plan = plan_teardown(FakeSession(foreign_account()))

        assert plan.database is False
        assert plan.foreign_database is True
        assert plan.pools == ("ARENA_HEAD_POOL",)
        assert plan.user is False


class TestRunTeardown:
    def test_confirmed_drops_everything_and_clears_local_logins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        connections, key = paths(tmp_path, monkeypatch)
        key.write_text("key", encoding="utf-8")
        write_connection("arena", job_connection_fields(ACCOUNT, key))
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(arena_account())
        io = Scripted(["y"])

        # Act
        run_teardown(session, ACCOUNT, False, io)

        # Assert
        assert io.prompts == ["Drop these objects? [y/N]"]
        assert drops(session) == FULL_DROP
        assert "ANALYTICS_POOL" not in "\n".join(session.statements)
        assert "ARENA_GPU_GPU_NV_M" not in "\n".join(drops(session))
        assert "  compute pool ARENA_GPU_GPU_NV_S and its services" in io.lines[0]
        assert io.lines[-1] == "\n".join(
            [
                "Dropped.",
                "Removed the [arena] connection.",
                "Removed the [arena-admin] connection.",
                f"Deleted {key}.",
            ]
        )
        assert (
            "  The local arena and arena-admin connections on this machine."
            in (io.lines[0])
        )
        assert connection_section("arena") is None
        assert connection_section("arena-admin") is None
        assert not key.exists()
        assert connections.is_file()

    @pytest.mark.parametrize("answer", ["", "n", "no"])
    def test_anything_but_yes_drops_nothing(self, answer: str) -> None:
        session = FakeSession(arena_account())

        with pytest.raises(ValueError, match=r"Nothing was dropped\."):
            run_teardown(session, ACCOUNT, False, Scripted([answer]))

        assert drops(session) == []

    def test_a_foreign_database_is_named_and_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        session = FakeSession(foreign_account())
        io = Scripted()

        run_teardown(session, ACCOUNT, True, io)

        assert io.prompts == []
        assert "Database arena is not an Arena database. It stays." in io.lines[0]
        assert drops(session) == [
            'ALTER COMPUTE POOL "ARENA_HEAD_POOL" STOP ALL',
            'DROP COMPUTE POOL "ARENA_HEAD_POOL"',
        ]

    def test_without_accountadmin_drops_nothing(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("SELECT CURRENT_USER"),
                    [Row(c1="U", c2="PUBLIC", c3="A", c4="R")],
                ),
                (lambda s: s.startswith("USE ROLE"), RuntimeError("no")),
            ]
        )

        with pytest.raises(ValueError, match="cannot use ACCOUNTADMIN"):
            run_teardown(session, ACCOUNT, True, Scripted())

        assert drops(session) == []


class TestRemoveLocal:
    def test_a_key_outside_the_default_path_and_other_accounts_stay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        own_key = tmp_path / "mine.p8"
        own_key.write_text("key", encoding="utf-8")
        write_connection("arena", job_connection_fields(ACCOUNT, own_key))
        write_connection(
            "arena-admin", admin_connection_fields("OTHER-ACCOUNT", "NICK")
        )

        removed = remove_local(ACCOUNT)

        assert removed == ["Removed the [arena] connection."]
        assert own_key.is_file()
        assert connection_section("arena-admin") is not None


class TestTeardownCommand:
    def test_drops_with_yes_through_the_admin_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(arena_account(pools=[]))
        opened: list[str] = []
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session",
            lambda name: opened.append(name) or session,
        )

        result = CliRunner().invoke(teardown_command, ["--yes"])

        assert result.exit_code == 0, result.output
        assert opened == ["arena-admin"]
        assert "DROP DATABASE IF EXISTS arena" in session.statements
        assert session.closed == 1
        assert "Removed the [arena-admin] connection." in result.output

    def test_no_at_the_prompt_exits_with_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))
        session = FakeSession(arena_account())
        monkeypatch.setattr(
            "agilerl.arena.snowflake.setup_cli.open_admin_session", lambda name: session
        )

        result = CliRunner().invoke(teardown_command, [], input="\n")

        assert result.exit_code == 1
        assert "Nothing was dropped." in result.output
        assert drops(session) == []
