# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Services that would block shrinking or dropping a compute pool."""

from __future__ import annotations

from agilerl.snowflake.catalog import (
    collect,
    whole_number,
)
from agilerl.snowflake.job import lowered
from agilerl.snowflake.source import SnowflakeSession

TERMINAL_SERVICE = frozenset(
    {
        "DONE",
        "FAILED",
        "INTERNAL_ERROR",
        "CANCELLED",
        "CANCELED",
        "SUSPENDED",
        "DELETED",
    }
)


def blocking_services(
    session: SnowflakeSession, pool: str, max_nodes: int
) -> list[tuple[str, int]]:
    """Queued or running services on ``pool``, job services included, that need more nodes than ``max_nodes``.

    :param session: Admin session.
    :param pool: Compute pool name.
    :param max_nodes: Proposed ``MAX_NODES``. ``0`` for a pool about to be dropped.
    :return: Service name and the nodes it needs.
    """
    blocked = []
    for row in collect(session, f"SHOW SERVICES IN COMPUTE POOL {pool}"):
        values = lowered(row)
        status = str(values.get("status") or values.get("service_status") or "").upper()
        if status in TERMINAL_SERVICE:
            continue
        nodes = _service_nodes(values)
        if nodes > max_nodes:
            blocked.append((service_name(values), nodes))
    return blocked


def shrink_message(blocked: list[tuple[str, int]], max_nodes: int) -> str:
    """Message when a lower ``MAX_NODES`` would strand a queued or running service.

    :param blocked: Service name and the nodes it needs.
    :param max_nodes: Proposed ``MAX_NODES``.
    :return: The message.
    """
    lines = [
        f"{name} needs {nodes} {'node' if nodes == 1 else 'nodes'}. "
        f"MAX_NODES {max_nodes} is below that."
        for name, nodes in blocked
    ]
    lines.append("The pool was not changed.")
    return "\n".join(lines)


def _service_nodes(values: dict[str, object]) -> int:
    # A queued service reports 0 current instances; a live one needs at least a node.
    keys = ("current_instances", "target_instances", "min_instances")
    return max(1, *(whole_number(values.get(key)) for key in keys))


def service_name(values: dict[str, object]) -> str:
    """``database.schema.name`` of a ``SHOW SERVICES`` row.

    :param values: Lower-cased row.
    :return: The qualified name.
    """
    return f"{values['database_name']}.{values['schema_name']}.{values['name']}"
