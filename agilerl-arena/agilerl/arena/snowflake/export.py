# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Copy a Snowflake table or query into parquet shards inside the job image."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agilerl.arena.snowflake.source import (
    QuerySource,
    SnowflakeFrame,
    SnowflakeSession,
    TableSource,
)


def write_source(
    session: SnowflakeSession,
    source: TableSource | QuerySource,
    columns: dict[str, str] | None,
    destination: Path,
) -> Path:
    """Write ``source`` to parquet shards that share one schema.

    Column names are written as Snowflake reports them, so unquoted names are
    upper case.

    :param session: Snowpark session with ``table``, ``sql``, and ``to_pandas_batches``.
    :param source: Table or query to read.
    :param columns: Manifest map of canonical name to dataset column.
        ``None`` writes every column. Names are not renamed.
    :param destination: Directory that receives ``shard-NNNNN.parquet`` files.
    :return: ``destination``.
    :raises ValueError: A mapped column is missing, or the source has no rows.
    """
    frame = _frame(session, source)
    selected = selected_columns(columns)
    if selected is not None:
        frame = frame.select(*source_columns(frame.columns, selected))
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        batches = []
        for index, batch in enumerate(frame.to_pandas_batches()):
            path = Path(scratch) / f"shard-{index:05d}.parquet"
            pq.write_table(pa.Table.from_pandas(batch, preserve_index=False), path)
            batches.append(path)
        if not batches:
            msg = "The Snowflake source returned no rows"
            raise ValueError(msg)
        # Snowflake sizes integer columns per batch, and an all-null batch has
        # null-typed columns, so batches are widened to one shared schema.
        schema = pa.unify_schemas(
            [pq.read_schema(path) for path in batches],
            promote_options="permissive",
        )
        for path in batches:
            pq.write_table(pq.read_table(path).cast(schema), destination / path.name)
    return destination


def source_columns(available: list[str], wanted: list[str]) -> list[str]:
    """Select expressions for ``wanted`` dataset columns, matched exactly.

    :param available: ``frame.columns``. Quoted names keep their double quotes.
    :param wanted: Dataset column names from the manifest mapping.
    :return: One select expression per wanted name.
    :raises ValueError: A wanted name is not a column of the source.
    """
    by_name = {name.strip('"'): name for name in available}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        msg = (
            f"Mapped columns {missing} are not in the Snowflake source. "
            f"Its columns are {sorted(by_name)}. Snowflake stores unquoted "
            "names in upper case."
        )
        raise ValueError(msg)
    return [by_name[name] for name in wanted]


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
