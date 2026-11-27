# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Snowflake training sources and the Snowpark calls this package makes."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TableSource:
    """One Snowflake table, named as the session addresses it."""

    name: str


@dataclass(frozen=True)
class QuerySource:
    """One SQL query whose result is the training source."""

    sql: str


class SnowflakeRow(Protocol):
    """Snowpark ``Row`` read through ``as_dict``."""

    def as_dict(self) -> dict[str, Any]:
        """Column name to value."""


class SnowflakeFrame(Protocol):
    """Snowpark operations used to export a source and run a statement."""

    @property
    def columns(self) -> list[str]:
        """Column names as Snowflake reports them."""

    def select(self, *columns: str) -> SnowflakeFrame:
        """Return a frame with these columns."""

    def to_pandas_batches(self) -> Iterator[object]:
        """Yield pandas batches."""

    def collect(self) -> list[SnowflakeRow]:
        """Execute the statement."""


class SnowflakeFiles(Protocol):
    """Stage transfers used for run files, checkpoints, and metrics."""

    def put(
        self,
        local_file_name: str,
        stage_location: str,
        auto_compress: bool = False,
        overwrite: bool = False,
    ) -> object:
        """Upload one local file to a stage directory."""

    def get(self, stage_location: str, target_directory: str) -> object:
        """Download stage files into a local directory."""


class SnowflakeSession(Protocol):
    """Snowpark session operations this package calls."""

    file: SnowflakeFiles

    def table(self, name: str) -> SnowflakeFrame:
        """Open a table."""

    def sql(self, query: str) -> SnowflakeFrame:
        """Run a SQL statement."""

    def close(self) -> None:
        """Close the connection."""
