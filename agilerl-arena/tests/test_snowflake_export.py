# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowflake table and query export to parquet shards."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agilerl.arena.snowflake.export import (
    selected_columns,
    source_columns,
    write_source,
)
from agilerl.arena.snowflake.source import QuerySource, TableSource


class Frame:
    """Snowpark DataFrame stand-in that yields one pandas batch per entry."""

    def __init__(self, batches: list[pd.DataFrame], columns: list[str]) -> None:
        self.batches = batches
        self._columns = columns
        self.selected: tuple[str, ...] | None = None

    @property
    def columns(self) -> list[str]:
        return self._columns

    def select(self, *names: str) -> Frame:
        plain = [name.strip('"') for name in names]
        frame = Frame([batch[plain] for batch in self.batches], list(names))
        frame.selected = names
        self.selected = names
        return frame

    def to_pandas_batches(self) -> Iterator[pd.DataFrame]:
        yield from self.batches


class RecordingSession:
    def __init__(self, frame: Frame) -> None:
        self.frame = frame
        self.table_name: str | None = None
        self.query: str | None = None

    def table(self, name: str) -> Frame:
        self.table_name = name
        return self.frame

    def sql(self, text: str) -> Frame:
        self.query = text
        return self.frame


def _frame() -> Frame:
    return Frame(
        [
            pd.DataFrame([{"PROMPT": "one", "RESPONSE": "a", "UNUSED": "drop"}]),
            pd.DataFrame([{"PROMPT": "two", "RESPONSE": "b", "UNUSED": "drop"}]),
        ],
        ["PROMPT", "RESPONSE", "UNUSED"],
    )


class TestWriteSource:
    def test_table_keeps_mapped_column_names(self, tmp_path: Path) -> None:
        session = RecordingSession(_frame())
        destination = tmp_path / "data"

        written = write_source(
            session,
            TableSource("arena.training.prompts"),
            {"prompt": "PROMPT", "response": "RESPONSE"},
            destination,
        )

        table = pq.read_table(written)
        assert session.table_name == "arena.training.prompts"
        assert written == destination
        assert sorted(path.name for path in destination.iterdir()) == [
            "shard-00000.parquet",
            "shard-00001.parquet",
        ]
        assert table.column_names == ["PROMPT", "RESPONSE"]
        assert table.column("PROMPT").to_pylist() == ["one", "two"]

    def test_query_writes_every_column_when_unmapped(self, tmp_path: Path) -> None:
        session = RecordingSession(_frame())

        written = write_source(
            session,
            QuerySource("select prompt, response, unused from prompts"),
            None,
            tmp_path / "data",
        )

        table = pq.read_table(written)
        assert session.query == "select prompt, response, unused from prompts"
        assert table.column_names == ["PROMPT", "RESPONSE", "UNUSED"]
        assert table.num_rows == 2

    def test_quoted_lowercase_columns_are_selected_quoted(self, tmp_path: Path) -> None:
        frame = Frame(
            [pd.DataFrame([{"prompt": "one"}])],
            ['"prompt"'],
        )

        written = write_source(
            RecordingSession(frame),
            TableSource("t"),
            {"prompt": "prompt"},
            tmp_path / "d",
        )

        assert frame.selected == ('"prompt"',)
        assert pq.read_table(written).column_names == ["prompt"]

    def test_mixed_batch_types_share_one_schema(self, tmp_path: Path) -> None:
        frame = Frame(
            [
                pd.DataFrame({"ID": pd.Series([1], dtype="int8"), "NOTE": [None]}),
                pd.DataFrame({"ID": pd.Series([300], dtype="int64"), "NOTE": ["x"]}),
            ],
            ["ID", "NOTE"],
        )

        written = write_source(
            RecordingSession(frame), TableSource("t"), None, tmp_path / "d"
        )

        schemas = [pq.read_schema(path) for path in sorted(written.iterdir())]
        assert schemas[0].field("ID").type == pa.int64()
        assert schemas[0].remove_metadata() == schemas[1].remove_metadata()
        table = pq.read_table(written)
        assert table.column("ID").to_pylist() == [1, 300]
        assert table.column("NOTE").to_pylist() == [None, "x"]

    def test_empty_source_is_rejected(self, tmp_path: Path) -> None:
        frame = Frame([], ["PROMPT"])

        with pytest.raises(ValueError, match="returned no rows"):
            write_source(
                RecordingSession(frame), TableSource("t"), None, tmp_path / "d"
            )

    def test_a_failed_export_leaves_no_shards(self, tmp_path: Path) -> None:
        class Broken(Frame):
            def to_pandas_batches(self) -> Iterator[pd.DataFrame]:
                yield pd.DataFrame([{"PROMPT": "one"}])
                msg = "warehouse suspended"
                raise RuntimeError(msg)

        destination = tmp_path / "d"

        with pytest.raises(RuntimeError, match="warehouse suspended"):
            write_source(
                RecordingSession(Broken([], ["PROMPT"])),
                TableSource("t"),
                None,
                destination,
            )

        assert list(destination.iterdir()) == []

    def test_unknown_source_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Unknown Snowflake source"):
            write_source(
                RecordingSession(_frame()),
                cast("TableSource", object()),
                None,
                tmp_path / "data",
            )


class TestSourceColumns:
    def test_exact_and_quoted_names(self) -> None:
        assert source_columns(["PROMPT", '"answer"'], ["answer", "PROMPT"]) == [
            '"answer"',
            "PROMPT",
        ]

    def test_lowercase_name_for_an_unquoted_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\['prompt'\].*\['PROMPT'\].*upper case"):
            source_columns(["PROMPT"], ["prompt"])

    def test_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="MISSING"):
            source_columns(["PROMPT"], ["PROMPT", "MISSING"])


class TestSelectedColumns:
    def test_duplicate_dataset_columns_are_kept_once(self) -> None:
        assert selected_columns({"prompt": "PROMPT", "text": "PROMPT"}) == ["PROMPT"]

    def test_unmapped_keeps_every_column(self) -> None:
        assert selected_columns(None) is None

    def test_empty_mapping_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no dataset columns"):
            selected_columns({})
