# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Services that would block shrinking or dropping a compute pool."""

from __future__ import annotations

from agilerl.snowflake.account import blocking_services, shrink_message
from tests.test_snowflake_setup import FakeSession, Row

POOL = "AGILERL_APP_ARENA_APP_GPU_GPU_NV_M"


def service(name: str, status: str, **counts: object) -> Row:
    return Row(
        database_name="ARENA_APP",
        schema_name="APP_STATE",
        name=name,
        status=status,
        **counts,
    )


def pool_session(rows: list[Row]) -> FakeSession:
    return FakeSession(
        [
            (
                lambda statement: statement.startswith("SHOW SERVICES IN COMPUTE POOL"),
                rows,
            )
        ]
    )


class TestBlockingServices:
    def test_a_running_service_above_the_cap_blocks(self) -> None:
        # Arrange
        session = pool_session(
            [
                service("RUN_WORKERS", "RUNNING", current_instances=3),
                service("SMALL", "RUNNING", current_instances=1),
                service("OLD", "DONE", current_instances=4),
            ]
        )

        # Act
        blocked = blocking_services(session, POOL, 2)

        # Assert
        assert blocked == [("ARENA_APP.APP_STATE.RUN_WORKERS", 3)]
        assert session.statements == [f"SHOW SERVICES IN COMPUTE POOL {POOL}"]

    def test_a_queued_job_counts_the_nodes_it_asked_for(self) -> None:
        session = pool_session(
            [service("QUEUED", "PENDING", current_instances=0, target_instances=4)]
        )

        assert blocking_services(session, POOL, 2) == [
            ("ARENA_APP.APP_STATE.QUEUED", 4)
        ]

    def test_a_service_without_counts_needs_one_node(self) -> None:
        session = pool_session(
            [
                Row(
                    database_name="ARENA_APP",
                    schema_name="APP_STATE",
                    name="BARE",
                    service_status="RUNNING",
                )
            ]
        )

        assert blocking_services(session, POOL, 0) == [("ARENA_APP.APP_STATE.BARE", 1)]
        assert blocking_services(session, POOL, 1) == []


class TestShrinkMessage:
    def test_names_each_service_and_says_nothing_changed(self) -> None:
        text = shrink_message([("A.B.ONE", 1), ("A.B.MANY", 3)], 0)

        assert text == (
            "A.B.ONE needs 1 node. MAX_NODES 0 is below that.\n"
            "A.B.MANY needs 3 nodes. MAX_NODES 0 is below that.\n"
            "The pool was not changed."
        )
