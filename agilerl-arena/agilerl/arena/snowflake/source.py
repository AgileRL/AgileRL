# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Copy a Snowflake table or query into a local parquet shard directory."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class TableSource:
    """One Snowflake table, named as the session addresses it."""

    name: str


@dataclass(frozen=True)
class QuerySource:
    """One SQL query whose result is the training source."""

    sql: str


class SnowflakeFrame(Protocol):
    """Snowpark operations used to export a source and run a merge."""

    def select(self, *columns: str) -> SnowflakeFrame:
        """Return a frame with these columns."""

    def to_pandas_batches(self) -> Iterator[object]:
        """Yield pandas batches."""

    def collect(self) -> list[object]:
        """Execute the statement."""


class SnowflakeFiles(Protocol):
    """Stage upload used for checkpoints, the manifest, and metrics."""

    def put(
        self,
        local_file_name: str,
        stage_location: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> object:
        """Upload one local file to a stage directory."""


class SnowflakeSession(Protocol):
    """Snowpark session operations this package calls."""

    file: SnowflakeFiles

    def table(self, name: str) -> SnowflakeFrame:
        """Open a table."""

    def sql(self, query: str) -> SnowflakeFrame:
        """Run a SQL statement."""


def write_source(
    session: SnowflakeSession,
    source: TableSource | QuerySource,
    columns: dict[str, str] | None,
    destination: Path,
) -> Path:
    """Write ``source`` to parquet shards, keeping the table's column names.

    :param session: Snowpark session with ``table``, ``sql``, and ``to_pandas_batches``.
    :param source: Table or query to read.
    :param columns: Manifest map of canonical name to dataset column.
        ``None`` writes every column. Names are not renamed.
    :param destination: Directory that receives ``shard-NNNNN.parquet`` files.
    :return: ``destination``.
    """
    frame = _frame(session, source)
    selected = selected_columns(columns)
    if selected is not None:
        frame = frame.select(*selected)
    destination.mkdir(parents=True, exist_ok=True)
    # Deferred: pyarrow ships with the snowflake extra, not the base install.
    import pyarrow as pa
    import pyarrow.parquet as pq

    for index, batch in enumerate(frame.to_pandas_batches()):
        table = pa.Table.from_pandas(batch, preserve_index=False)
        pq.write_table(table, destination / f"shard-{index:05d}.parquet")
    return destination


def selected_columns(columns: dict[str, str] | None) -> list[str] | None:
    """Dataset column names named by a manifest ``columns`` mapping.

    :param columns: Canonical name to dataset column, or ``None``.
    :return: Unique dataset column names, or ``None`` when every column is kept.
    :raises ValueError: The mapping names no dataset column.
    """
    if columns is None:
        return None
    names: list[str] = []
    for dataset_column in columns.values():
        if dataset_column not in names:
            names.append(dataset_column)
    if not names:
        msg = "columns mapping names no dataset columns"
        raise ValueError(msg)
    return names


def _frame(
    session: SnowflakeSession, source: TableSource | QuerySource
) -> SnowflakeFrame:
    if isinstance(source, TableSource):
        return session.table(source.name)
    if isinstance(source, QuerySource):
        return session.sql(source.sql)
    msg = f"Unknown Snowflake source {type(source).__name__}"
    raise TypeError(msg)
