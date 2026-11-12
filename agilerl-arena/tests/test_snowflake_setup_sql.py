# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Statements arena snowflake setup runs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agilerl.arena.snowflake.setup_sql import (
    QUALIFIED_BUDGET,
    ArenaCredentials,
    PoolPlan,
    SetupPlan,
    arena_statements,
    budget_statements,
    integration_sql,
    pool_statements,
    script,
    secret_statements,
    setup_statements,
    string_literal,
    user_statements,
)

GPU_POOL = PoolPlan("arena_gpu_GPU_NV_S", "GPU_NV_S", "gpu", 0, 2, 2)
HEAD_POOL = PoolPlan("arena_head_pool", "CPU_X64_SL", "head", 0, 4, None)
CREDENTIALS = ArenaCredentials("arena_org_x", "user-1", "https://arena.agilerl.com")
PLAN = SetupPlan(
    pools=(HEAD_POOL, GPU_POOL),
    existing_max_nodes={},
    warehouse_credits=100,
    budget_credits=500,
    notify_email=None,
    public_key="KEY",
    stored_key=None,
    rotate_key=False,
    arena=CREDENTIALS,
    integrations=frozenset(),
    linked="",
)


class TestSetupStatements:
    def test_never_replaces_an_object(self) -> None:
        statements = setup_statements(PLAN)

        assert not any("OR REPLACE" in statement for statement in statements)
        assert statements[:3] == [
            "CREATE ROLE IF NOT EXISTS arena_trainer",
            "CREATE DATABASE IF NOT EXISTS arena",
            "CREATE SCHEMA IF NOT EXISTS arena.training",
        ]

    def test_uses_the_warehouse_after_creating_it_and_before_the_budget(
        self,
    ) -> None:
        statements = setup_statements(PLAN)

        create = next(
            i
            for i, s in enumerate(statements)
            if s.startswith("CREATE WAREHOUSE IF NOT EXISTS arena_wh")
        )
        budget = next(i for i, s in enumerate(statements) if QUALIFIED_BUDGET in s)
        use = statements.index("USE WAREHOUSE arena_wh")

        assert create < use < budget

    def test_links_the_cpu_and_gpu_pools_to_the_budget(self) -> None:
        sql = "\n".join(setup_statements(PLAN))

        assert "'COMPUTE POOL', 'arena_head_pool'" in sql
        assert "'COMPUTE POOL', 'arena_gpu_GPU_NV_S'" in sql

    def test_has_no_event_table(self) -> None:
        sql = "\n".join(setup_statements(PLAN)).upper()

        assert "EVENT TABLE" not in sql
        assert "EVENT_TABLE" not in sql


class TestScript:
    def test_ends_each_statement_with_a_semicolon(self) -> None:
        assert script(["SELECT 1", "SELECT 2"]) == "SELECT 1;\n\nSELECT 2;\n"


class TestStringLiteral:
    def test_escapes_quotes_and_backslashes(self) -> None:
        assert string_literal("it's a\\b") == "'it''s a\\\\b'"


class TestIntegrationSql:
    def test_creates_a_missing_integration(self) -> None:
        sql = integration_sql("arena_hf_eai", "arena_hf_egress", frozenset())

        assert sql == (
            "CREATE EXTERNAL ACCESS INTEGRATION arena_hf_eai\n"
            "  ALLOWED_NETWORK_RULES = (arena.training.arena_hf_egress)\n"
            "  ENABLED = TRUE"
        )

    def test_alters_an_existing_integration_in_place(self) -> None:
        sql = integration_sql(
            "arena_hf_eai", "arena_hf_egress", frozenset({"ARENA_HF_EAI"})
        )

        assert sql == (
            "ALTER EXTERNAL ACCESS INTEGRATION arena_hf_eai SET "
            "ALLOWED_NETWORK_RULES = (arena.training.arena_hf_egress) ENABLED = TRUE"
        )


class TestPoolStatements:
    def test_creates_a_missing_pool(self) -> None:
        (statement,) = pool_statements(None, GPU_POOL)

        assert statement.startswith(
            "CREATE COMPUTE POOL IF NOT EXISTS arena_gpu_GPU_NV_S"
        )
        assert "MAX_NODES = 2" in statement

    def test_resizes_a_pool_with_a_different_max(self) -> None:
        assert pool_statements(5, GPU_POOL) == [
            "ALTER COMPUTE POOL arena_gpu_GPU_NV_S SET MAX_NODES = 2"
        ]

    def test_leaves_a_matching_pool_alone(self) -> None:
        assert pool_statements(2, GPU_POOL) == []


class TestBudgetStatements:
    def test_sets_the_limit_email_and_links_only_new_pools(self) -> None:
        plan = replace(
            PLAN,
            notify_email="ops@example.com",
            linked="compute pool arena_head_pool",
        )

        statements = budget_statements(plan)

        assert statements[:3] == [
            "CREATE SNOWFLAKE.CORE.BUDGET IF NOT EXISTS arena.training.arena_gpu_budget()",
            "CALL arena.training.arena_gpu_budget!SET_SPENDING_LIMIT(500)",
            (
                "CALL arena.training.arena_gpu_budget!SET_EMAIL_NOTIFICATIONS("
                "'ops@example.com')"
            ),
        ]
        assert len(statements) == 4
        assert "arena_gpu_GPU_NV_S" in statements[3]

    def test_without_an_email_sets_no_notification(self) -> None:
        statements = budget_statements(PLAN)

        assert not any("SET_EMAIL_NOTIFICATIONS" in s for s in statements)


class TestUserStatements:
    def test_creates_a_missing_user(self) -> None:
        (statement,) = user_statements("KEY", None, False)

        assert statement.startswith("CREATE USER IF NOT EXISTS arena_job_user")
        assert "RSA_PUBLIC_KEY = 'KEY'" in statement

    @pytest.mark.parametrize(
        ("stored", "rotate"), [("", False), ("OTHER", False), ("KEY", True)]
    )
    def test_sets_a_missing_changed_or_rotated_key(
        self, stored: str, rotate: bool
    ) -> None:
        assert user_statements("KEY", stored, rotate) == [
            "ALTER USER arena_job_user SET RSA_PUBLIC_KEY = 'KEY'"
        ]

    def test_the_key_is_a_quoted_literal(self) -> None:
        (create,) = user_statements("K'EY", None, False)
        (alter,) = user_statements("K'EY", "", False)

        assert "RSA_PUBLIC_KEY = 'K''EY'" in create
        assert alter == "ALTER USER arena_job_user SET RSA_PUBLIC_KEY = 'K''EY'"

    def test_a_matching_key_is_left_alone(self) -> None:
        assert user_statements("K E\nY", "KEY", False) == []


class TestSecretStatements:
    def test_creates_when_missing_then_sets_the_value(self) -> None:
        assert secret_statements("arena_org_key", "o'k") == [
            (
                "CREATE SECRET IF NOT EXISTS arena.training.arena_org_key\n"
                "  TYPE = GENERIC_STRING\n"
                "  SECRET_STRING = 'o''k'"
            ),
            "ALTER SECRET arena.training.arena_org_key SET SECRET_STRING = 'o''k'",
        ]


class TestArenaStatements:
    def test_kept_secrets_run_nothing(self) -> None:
        assert arena_statements(None, frozenset()) == []

    def test_rule_host_follows_the_base_url(self) -> None:
        arena = replace(CREDENTIALS, base_url="https://staging.arena.agilerl.com")

        statements = arena_statements(arena, frozenset({"ARENA_API_EAI"}))

        assert len(statements) == 8
        assert "'staging.arena.agilerl.com:443'" in statements[6]
        assert statements[6].startswith("CREATE OR ALTER NETWORK RULE")
        assert statements[7].startswith(
            "ALTER EXTERNAL ACCESS INTEGRATION arena_api_eai"
        )
