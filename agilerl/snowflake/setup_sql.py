# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Compute pool and secret statements. Every statement is safe to run again."""

from __future__ import annotations

from dataclasses import dataclass

from agilerl.snowflake.profile import (
    POOL_SUSPEND_SECS,
)


@dataclass(frozen=True)
class PoolPlan:
    """One pool setup will create or resize."""

    name: str
    family: str
    role: str
    min_nodes: int
    max_nodes: int
    max_gpus: int | None


def string_literal(value: str) -> str:
    """Single-quoted SQL string.

    :param value: Raw text.
    :return: The literal.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "''")
    return f"'{escaped}'"


def pool_statements(max_nodes: int | None, plan: PoolPlan) -> list[str]:
    """Create the pool, or change its max nodes.

    :param max_nodes: Max nodes of the existing pool, or ``None`` when absent.
    :param plan: Name, family, and max nodes.
    :return: The statements.
    """
    if max_nodes is None:
        return [create_pool_sql(plan.name, plan.family, plan.min_nodes, plan.max_nodes)]
    if max_nodes != plan.max_nodes:
        return [f"ALTER COMPUTE POOL {plan.name} SET MAX_NODES = {plan.max_nodes}"]
    return []


def create_pool_sql(name: str, family: str, min_nodes: int, max_nodes: int) -> str:
    """``CREATE COMPUTE POOL`` for one Arena pool.

    :param name: Pool name.
    :param family: Instance family.
    :param min_nodes: Minimum nodes.
    :param max_nodes: Maximum nodes.
    :return: The statement.
    """
    return (
        f"CREATE COMPUTE POOL IF NOT EXISTS {name}\n"
        f"  MIN_NODES = {min_nodes}\n"
        f"  MAX_NODES = {max_nodes}\n"
        f"  INSTANCE_FAMILY = {family}\n"
        "  AUTO_RESUME = TRUE\n"
        f"  AUTO_SUSPEND_SECS = {POOL_SUSPEND_SECS}\n"
        "  INITIALLY_SUSPENDED = TRUE"
    )


def secret_statements(schema: str, name: str, value: str) -> list[str]:
    """Create the secret when missing, then set its value.

    :param schema: Schema that holds the secret.
    :param name: Secret name.
    :param value: Secret string.
    :return: The statements.
    """
    literal = string_literal(value)
    qualified = f"{schema}.{name}"
    return [
        (
            f"CREATE SECRET IF NOT EXISTS {qualified}\n"
            "  TYPE = GENERIC_STRING\n"
            f"  SECRET_STRING = {literal}"
        ),
        f"ALTER SECRET {qualified} SET SECRET_STRING = {literal}",
    ]
