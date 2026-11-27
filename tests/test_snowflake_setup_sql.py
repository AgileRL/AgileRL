# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Compute pool and secret statements."""

from __future__ import annotations

from agilerl.snowflake.setup_sql import (
    PoolPlan,
    pool_statements,
    secret_statements,
    string_literal,
)

GPU_POOL = PoolPlan("arena_gpu_GPU_NV_S", "GPU_NV_S", "gpu", 0, 2, 2)
HEAD_POOL = PoolPlan("arena_head_pool", "CPU_X64_SL", "head", 0, 4, None)


class TestStringLiteral:
    def test_escapes_quotes_and_backslashes(self) -> None:
        assert string_literal("it's a\\b") == "'it''s a\\\\b'"


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


class TestSecretStatements:
    def test_creates_when_missing_then_sets_the_value(self) -> None:
        assert secret_statements("arena.training", "arena_org_key", "o'k") == [
            (
                "CREATE SECRET IF NOT EXISTS arena.training.arena_org_key\n"
                "  TYPE = GENERIC_STRING\n"
                "  SECRET_STRING = 'o''k'"
            ),
            "ALTER SECRET arena.training.arena_org_key SET SECRET_STRING = 'o''k'",
        ]
