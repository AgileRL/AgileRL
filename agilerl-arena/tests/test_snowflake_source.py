# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowflake table and query export to parquet shards."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pyarrow.parquet as pq
import pytest

from agilerl.arena.snowflake.source import (
    QuerySource,
    TableSource,
    selected_columns,
    write_source,
)


class Frame:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def select(self, *names: str) -> Frame:
        return Frame([{name: row[name] for name in names} for row in self.rows])

    def to_pandas_batches(self):
        import pandas as pd

        if len(self.rows) < 2:
            yield pd.DataFrame(self.rows)
            return
        yield pd.DataFrame(self.rows[:1])
        yield pd.DataFrame(self.rows[1:])


class RecordingSession:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.table_name: str | None = None
        self.query: str | None = None

    def table(self, name: str) -> Frame:
        self.table_name = name
        return Frame(self.rows)

    def sql(self, text: str) -> Frame:
        self.query = text
        return Frame(self.rows)


ROWS = [
    {"PROMPT": "one", "RESPONSE": "a", "UNUSED": "drop"},
    {"PROMPT": "two", "RESPONSE": "b", "UNUSED": "drop"},
]


def _read(directory: Path):
    shards = sorted(directory.glob("*.parquet"))
    return shards, pq.read_table(directory)


class TestWriteSource:
    def test_table_keeps_mapped_column_names(self, tmp_path: Path) -> None:
        session = RecordingSession(ROWS)
        destination = tmp_path / "data"

        written = write_source(
            session,
            TableSource("arena.training.prompts"),
            {"prompt": "PROMPT", "response": "RESPONSE"},
            destination,
        )

        shards, table = _read(written)
        assert session.table_name == "arena.training.prompts"
        assert session.query is None
        assert written == destination
        assert len(shards) == 2
        assert table.column_names == ["PROMPT", "RESPONSE"]
        assert table.column("PROMPT").to_pylist() == ["one", "two"]
        assert table.column("RESPONSE").to_pylist() == ["a", "b"]

    def test_query_writes_every_column_when_unmapped(self, tmp_path: Path) -> None:
        session = RecordingSession(ROWS[:1])

        written = write_source(
            session,
            QuerySource("select prompt, response, unused from prompts"),
            None,
            tmp_path / "data",
        )

        table = pq.read_table(written)
        assert session.query == "select prompt, response, unused from prompts"
        assert table.column_names == ["PROMPT", "RESPONSE", "UNUSED"]
        assert table.num_rows == 1


class TestSelectedColumns:
    def test_duplicate_dataset_columns_are_kept_once(self) -> None:
        assert selected_columns({"prompt": "PROMPT", "text": "PROMPT"}) == ["PROMPT"]

    def test_empty_mapping_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no dataset columns"):
            selected_columns({})

    def test_unknown_source_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Unknown Snowflake source"):
            write_source(
                RecordingSession(ROWS),
                cast("TableSource", object()),
                None,
                tmp_path / "data",
            )
