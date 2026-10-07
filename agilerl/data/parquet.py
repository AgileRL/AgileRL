# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Load local parquet files or shard directories into a Hugging Face Dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = ["load_parquet_dataset", "parquet_shard_paths"]


def parquet_shard_paths(path: str | Path) -> list[Path]:
    """Return parquet files for a local file or directory, in filename order.

    :param path: A ``.parquet`` / ``.pq`` file, or a directory of ``*.parquet`` shards.
    :type path: str | Path
    :return: Shard paths in filename order.
    :rtype: list[Path]
    :raises ValueError: If *path* is an empty directory or unsupported file.
    :raises FileNotFoundError: If *path* does not exist.
    """
    resolved = Path(path)
    if resolved.is_file():
        if resolved.suffix not in {".parquet", ".pq"}:
            msg = f"Unsupported parquet path: {resolved}"
            raise ValueError(msg)
        return [resolved]
    if resolved.is_dir():
        files = sorted(resolved.glob("*.parquet"))
        if not files:
            msg = f"No .parquet files in directory: {resolved}"
            raise ValueError(msg)
        return files
    msg = f"Parquet path does not exist: {resolved}"
    raise FileNotFoundError(msg)


def _rename_columns(
    dataset: Dataset, column_rename: Mapping[str, str] | None
) -> Dataset:
    if not column_rename:
        return dataset
    rename = {
        old: new
        for old, new in column_rename.items()
        if old in dataset.column_names and old != new
    }
    if not rename:
        return dataset
    return dataset.rename_columns(rename)


def _assert_matching_schemas(shards: list[Dataset], files: list[Path]) -> None:
    expected = shards[0].features
    for shard, file in zip(shards[1:], files[1:], strict=True):
        if shard.features != expected:
            msg = (
                f"Parquet shard {file.name} schema does not match {files[0].name}: "
                f"{shard.features} != {expected}"
            )
            raise ValueError(msg)


def load_parquet_dataset(
    path: str | Path,
    column_rename: Mapping[str, str] | None = None,
) -> Dataset:
    """Load a local parquet file or directory of shards into one Dataset.

    Directory loads every ``*.parquet`` file in that directory, non-recursive,
    in filename order. Shards are concatenated as Hugging Face datasets, not
    as one pandas DataFrame. Mismatched schemas raise.

    :param path: Local ``.parquet`` / ``.pq`` file, or a directory of shards.
    :type path: str | Path
    :param column_rename: Existing column name to new name.
    :type column_rename: Mapping[str, str] | None
    :return: One Hugging Face Dataset. Callers split train/test.
    :rtype: Dataset
    """
    files = parquet_shard_paths(path)
    from datasets import Dataset, concatenate_datasets  # optional extra: llm

    shards = [Dataset.from_parquet(str(file)) for file in files]
    _assert_matching_schemas(shards, files)
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    return _rename_columns(dataset, column_rename)
