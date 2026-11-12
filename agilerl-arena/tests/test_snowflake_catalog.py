# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Read-only account queries."""

from __future__ import annotations

import json

import pytest

from agilerl.arena.snowflake.catalog import (
    SET_EVENT_TABLE,
    GpuFamily,
    active_event_table,
    budget_limit,
    event_table_warning,
    existing_integrations,
    families_from,
    is_foreign_database,
    missing_object,
    monitor_quota,
    one_line,
    remote_profile,
    require_accountadmin,
    secret_names,
    service_user_key,
    setup_json_exists,
    use_arena_warehouse,
    whole_number,
)
from tests.test_snowflake_setup import (
    IMAGE,
    PROFILE_DOC,
    FakeSession,
    Row,
    admin_routes,
)


def raising(prefix: str, error: Exception) -> FakeSession:
    return FakeSession([(lambda s: s.startswith(prefix), error)])


class TestMissingObject:
    @pytest.mark.parametrize(
        "text",
        [
            "Database 'ARENA' does not exist or not authorized.",
            "Stage not found",
            "No such file",
        ],
    )
    def test_missing_object_errors(self, text: str) -> None:
        assert missing_object(RuntimeError(text)) is True

    def test_other_errors(self) -> None:
        assert missing_object(RuntimeError("Insufficient privileges")) is False


class TestOneLine:
    def test_keeps_the_first_line(self) -> None:
        assert one_line(RuntimeError("  bad SQL\n  at line 3 ")) == "bad SQL"

    def test_an_empty_error(self) -> None:
        assert one_line(RuntimeError("")) == "unknown error"


class TestRequireAccountadmin:
    def test_switches_to_accountadmin(self) -> None:
        session = FakeSession(admin_routes())

        require_accountadmin(session)

        assert "USE ROLE ACCOUNTADMIN" in session.statements

    def test_a_session_left_on_another_role_stops(self) -> None:
        session = FakeSession(
            admin_routes(
                [
                    (
                        lambda s: s == "SELECT CURRENT_ROLE()",
                        [Row(**{"CURRENT_ROLE()": "SYSADMIN"})],
                    )
                ]
            )
        )

        with pytest.raises(ValueError, match=r"current role: SYSADMIN"):
            require_accountadmin(session)


class TestUseArenaWarehouse:
    def test_uses_arena_wh_when_it_exists(self) -> None:
        session = FakeSession(
            [(lambda s: s.startswith("SHOW WAREHOUSES"), [Row(name="ARENA_WH")])]
        )

        use_arena_warehouse(session)

        assert session.statements == [
            "SHOW WAREHOUSES LIKE 'arena_wh'",
            "USE WAREHOUSE arena_wh",
        ]

    def test_a_missing_warehouse_runs_nothing_else(self) -> None:
        session = FakeSession()

        use_arena_warehouse(session)

        assert session.statements == ["SHOW WAREHOUSES LIKE 'arena_wh'"]


class TestFamiliesFrom:
    def test_reads_name_gpu_count_and_gpu(self) -> None:
        rows = [Row(name="GPU_NV_M", gpu_count="4", gpu="A10G")]

        assert families_from(rows) == [GpuFamily("GPU_NV_M", 4, "A10G")]

    def test_a_known_family_without_a_gpu_column_uses_arenas_table(self) -> None:
        rows = [Row(instance_family="GPU_NV_S", gpu_count=1)]

        assert families_from(rows) == [GpuFamily("GPU_NV_S", 1, "A10G")]

    def test_an_unknown_family_without_a_gpu_column_has_no_gpu(self) -> None:
        rows = [Row(name="CPU_X64_L", gpu_count=0)]

        assert families_from(rows) == [GpuFamily("CPU_X64_L", 0, "")]


class TestWholeNumber:
    @pytest.mark.parametrize(
        ("value", "expected"), [("4.00", 4), (7, 7), (None, 0), ("", 0)]
    )
    def test_reads_a_show_column(self, value: object, expected: int) -> None:
        assert whole_number(value) == expected


class TestSetupJsonExists:
    def test_a_listed_file(self) -> None:
        session = FakeSession(
            [(lambda s: s.startswith("LIST"), [Row(name="setup.json")])]
        )

        assert setup_json_exists(session) is True

    def test_a_missing_stage_is_false(self) -> None:
        session = raising(
            "LIST", RuntimeError("Stage 'ARENA.TRAINING.RUNS' does not exist")
        )

        assert setup_json_exists(session) is False

    def test_other_errors_propagate(self) -> None:
        session = raising("LIST", RuntimeError("Insufficient privileges"))

        with pytest.raises(RuntimeError, match="Insufficient privileges"):
            setup_json_exists(session)


class TestSecretNames:
    def test_upper_case_names(self) -> None:
        session = FakeSession(
            [(lambda s: s.startswith("SHOW SECRETS"), [Row(name="arena_org_key")])]
        )

        assert secret_names(session) == {"ARENA_ORG_KEY"}

    def test_a_missing_schema_has_none(self) -> None:
        session = raising("SHOW SECRETS", RuntimeError("Schema does not exist"))

        assert secret_names(session) == set()


class TestExistingIntegrations:
    def test_maps_name_to_enabled(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("SHOW EXTERNAL"),
                    [
                        Row(name="arena_hf_eai", enabled="true"),
                        Row(name="ARENA_API_EAI", enabled="false"),
                    ],
                )
            ]
        )

        assert existing_integrations(session) == {
            "ARENA_HF_EAI": True,
            "ARENA_API_EAI": False,
        }


class TestMonitorQuota:
    def test_reads_a_decimal_quota(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("SHOW RESOURCE MONITORS"),
                    [Row(name="ARENA_MONITOR", credit_quota="250.00")],
                )
            ]
        )

        assert monitor_quota(session) == 250

    def test_a_missing_monitor(self) -> None:
        assert monitor_quota(FakeSession()) is None


class TestBudgetLimit:
    def test_reads_the_limit(self) -> None:
        session = FakeSession(
            [(lambda s: "GET_SPENDING_LIMIT" in s, [Row(GET_SPENDING_LIMIT=700)])]
        )

        assert budget_limit(session) == 700

    def test_a_missing_budget(self) -> None:
        session = raising("CALL", RuntimeError("Budget does not exist"))

        assert budget_limit(session) is None

    def test_a_budget_without_a_limit(self) -> None:
        session = FakeSession(
            [(lambda s: "GET_SPENDING_LIMIT" in s, [Row(GET_SPENDING_LIMIT=None)])]
        )

        assert budget_limit(session) is None


class TestServiceUserKey:
    def test_a_missing_user(self) -> None:
        assert service_user_key(FakeSession()) is None

    def test_a_user_without_a_key(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW USERS"), [Row(name="ARENA_JOB_USER")]),
                (
                    lambda s: s.startswith("DESCRIBE USER"),
                    [Row(property="RSA_PUBLIC_KEY", value="null")],
                ),
            ]
        )

        assert service_user_key(session) == ""

    def test_the_stored_key(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW USERS"), [Row(name="ARENA_JOB_USER")]),
                (
                    lambda s: s.startswith("DESCRIBE USER"),
                    [Row(property="RSA_PUBLIC_KEY", value="MIIB")],
                ),
            ]
        )

        assert service_user_key(session) == "MIIB"

    def test_the_key_among_other_properties(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW USERS"), [Row(name="ARENA_JOB_USER")]),
                (
                    lambda s: s.startswith("DESCRIBE USER"),
                    [
                        Row(property="NAME", value="ARENA_JOB_USER"),
                        Row(property="RSA_PUBLIC_KEY", value="MIIB"),
                        Row(property="RSA_PUBLIC_KEY_2", value="null"),
                    ],
                ),
            ]
        )

        assert service_user_key(session) == "MIIB"


class TestIsForeignDatabase:
    def test_no_database(self) -> None:
        assert is_foreign_database(FakeSession()) is False

    def test_an_arena_account(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW DATABASES"), [Row(name="ARENA")]),
                (
                    lambda s: s.startswith("SHOW EXTERNAL"),
                    [Row(name="ARENA_HF_EAI", enabled="true")],
                ),
            ]
        )

        assert is_foreign_database(session) is False

    def test_a_database_with_the_metrics_table_is_arenas(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW DATABASES"), [Row(name="ARENA")]),
                (lambda s: s.startswith("SHOW TABLES"), [Row(name="ARENA_METRICS")]),
                (lambda s: s.startswith("SHOW SCHEMAS"), [Row(name="TRAINING")]),
            ]
        )

        assert is_foreign_database(session) is False

    def test_a_database_with_setup_json_is_arenas(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW DATABASES"), [Row(name="ARENA")]),
                (lambda s: s.startswith("LIST"), [Row(name="setup.json")]),
                (lambda s: s.startswith("SHOW SCHEMAS"), [Row(name="TRAINING")]),
            ]
        )

        assert is_foreign_database(session) is False

    def test_someone_elses_database(self) -> None:
        session = FakeSession(
            [
                (lambda s: s.startswith("SHOW DATABASES"), [Row(name="ARENA")]),
                (lambda s: s.startswith("SHOW SCHEMAS"), [Row(name="SALES")]),
            ]
        )

        assert is_foreign_database(session) is True


class TestRemoteProfile:
    def test_reads_setup_json(self) -> None:
        session = FakeSession(
            [(lambda s: s.startswith("SELECT $1"), [Row(DOC=json.dumps(PROFILE_DOC))])]
        )

        profile = remote_profile(session)

        assert profile is not None
        assert profile.image == IMAGE
        assert profile.pools.gpu["GPU_NV_S"].max_gpus == 2

    def test_a_missing_file(self) -> None:
        session = raising("SELECT $1", RuntimeError("File not found"))

        assert remote_profile(session) is None


class TestActiveEventTable:
    def test_reads_the_account_parameter(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s == "SHOW PARAMETERS LIKE 'EVENT_TABLE' IN ACCOUNT",
                    [Row(key="EVENT_TABLE", value="SNOWFLAKE.TELEMETRY.EVENTS")],
                )
            ]
        )

        assert active_event_table(session) == "SNOWFLAKE.TELEMETRY.EVENTS"

    def test_an_unset_parameter(self) -> None:
        session = FakeSession(
            [
                (
                    lambda s: s.startswith("SHOW PARAMETERS"),
                    [Row(key="EVENT_TABLE", value="")],
                )
            ]
        )

        assert active_event_table(session) == ""


class TestEventTableWarning:
    def test_names_the_sql_when_no_table_is_set(self) -> None:
        warning = event_table_warning("")

        assert warning is not None
        assert warning.endswith(f"run: {SET_EVENT_TABLE}")
        assert SET_EVENT_TABLE == (
            "ALTER ACCOUNT SET EVENT_TABLE = SNOWFLAKE.TELEMETRY.EVENTS"
        )

    def test_silent_when_a_table_is_set(self) -> None:
        assert event_table_warning("SNOWFLAKE.TELEMETRY.EVENTS") is None
