# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow", reason="LLM dependencies not installed")
pq = pytest.importorskip("pyarrow.parquet", reason="LLM dependencies not installed")

from agilerl.llm_envs.call_record import to_row  # noqa: E402
from agilerl.llm_envs.call_trace import CALL_RECORD_SCHEMA  # noqa: E402
from tests.test_llm_envs.call_factory import assistant, make_call, user  # noqa: E402


class TestCallRecordSchema:
    def test_a_record_row_round_trips_through_parquet(self, tmp_path) -> None:
        # Arrange
        record = make_call("01A", [user("hi")], assistant("hello"))
        path = tmp_path / "calls.parquet"

        # Act
        pq.write_table(
            pa.Table.from_pylist([to_row(record)], schema=CALL_RECORD_SCHEMA), path
        )
        (row,) = pq.read_table(path).to_pylist()

        # Assert
        assert row == to_row(record)
