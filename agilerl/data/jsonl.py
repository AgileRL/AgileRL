# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Load local JSONL files or shard directories into a Hugging Face Dataset."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = [
    "is_local_jsonl_dataset",
    "jsonl_shard_paths",
    "load_jsonl_dataset",
    "load_jsonl_train_eval_from_splits",
    "training_jsonl_paths",
]


def _immediate_jsonl_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jsonl"))


def _one_level_jsonl_split_dirs(path: Path) -> list[Path]:
    """Child directories that contain JSONL files.

    Empty when *path* is not a directory, or that directory already has
    JSONL files directly inside it.
    """
    if not path.is_dir():
        return []
    if _immediate_jsonl_files(path):
        return []
    return [
        child
        for child in sorted(path.iterdir(), key=lambda item: item.name)
        if child.is_dir() and _immediate_jsonl_files(child)
    ]


def is_local_jsonl_dataset(path: str | Path) -> bool:
    """Return whether *path* is a JSONL file or a directory of JSONL shards.

    A directory counts when it holds ``*.jsonl`` files, or when those files
    live in immediate child directories.

    :param path: Local path or a string that may be a JSONL suffix.
    :type path: str | Path
    :return: ``True`` for a JSONL source.
    :rtype: bool
    """
    if str(path).endswith(".jsonl"):
        return True
    resolved = Path(path)
    if not resolved.is_dir():
        return False
    if _immediate_jsonl_files(resolved):
        return True
    return bool(_one_level_jsonl_split_dirs(resolved))


def jsonl_shard_paths(path: str | Path) -> list[Path]:
    """Return JSONL files for a local file or directory.

    A directory that already contains ``*.jsonl`` files returns those files
    only, in filename order. A directory with none returns the JSONL files
    in each immediate child directory, split name then filename.

    :param path: A ``.jsonl`` file, or a directory of shards.
    :type path: str | Path
    :return: Shard paths.
    :rtype: list[Path]
    :raises ValueError: If *path* is an empty directory or unsupported file.
    :raises FileNotFoundError: If *path* does not exist.
    """
    resolved = Path(path)
    if resolved.is_file():
        if resolved.suffix != ".jsonl":
            msg = f"Unsupported JSONL path: {resolved}"
            raise ValueError(msg)
        return [resolved]
    if resolved.is_dir():
        files = _immediate_jsonl_files(resolved)
        if files:
            return files
        nested: list[Path] = []
        for split_dir in _one_level_jsonl_split_dirs(resolved):
            nested.extend(_immediate_jsonl_files(split_dir))
        if not nested:
            msg = f"No .jsonl files in directory: {resolved}"
            raise ValueError(msg)
        return nested
    msg = f"JSONL path does not exist: {resolved}"
    raise FileNotFoundError(msg)


def training_jsonl_paths(path: str | Path) -> list[Path]:
    """Return JSONL files whose rows are training steps.

    A directory with a ``train`` split (any ASCII case) and at least one other
    split returns the JSONL files in the train directories only. A file, a
    flat shard directory, a train-only directory, or split directories with no
    train directory return the same files as :func:`jsonl_shard_paths`.

    :param path: A ``.jsonl`` file, or a directory of shards.
    :type path: str | Path
    :return: JSONL files counted as training rows.
    :rtype: list[Path]
    :raises ValueError: If *path* is an empty directory or unsupported file.
    :raises FileNotFoundError: If *path* does not exist.
    """
    resolved = Path(path)
    splits = _one_level_jsonl_split_dirs(resolved)
    train_dirs = [
        directory for directory in splits if directory.name.lower() == "train"
    ]
    other_dirs = [
        directory for directory in splits if directory.name.lower() != "train"
    ]
    if train_dirs and other_dirs:
        files: list[Path] = []
        for directory in train_dirs:
            files.extend(_immediate_jsonl_files(directory))
        return files
    return jsonl_shard_paths(path)


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
                f"JSONL shard {file.name} schema does not match {files[0].name}: "
                f"{shard.features} != {expected}"
            )
            raise ValueError(msg)


def load_jsonl_dataset(
    path: str | Path,
    column_rename: Mapping[str, str] | None = None,
) -> Dataset:
    """Load a local JSONL file or directory of shards into one Dataset.

    A directory with ``*.jsonl`` files loads those files only, in filename
    order. A directory whose JSONL files live in split subdirectories loads
    every split. Shards are concatenated as Hugging Face datasets. Mismatched
    schemas raise.

    :param path: Local ``.jsonl`` file, or a directory of shards.
    :type path: str | Path
    :param column_rename: Existing column name to new name.
    :type column_rename: Mapping[str, str] | None
    :return: One Hugging Face Dataset.
    :rtype: Dataset
    """
    files = jsonl_shard_paths(path)
    from datasets import Dataset, concatenate_datasets  # optional extra: llm

    shards = [Dataset.from_json(str(file)) for file in files]
    _assert_matching_schemas(shards, files)
    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    return _rename_columns(dataset, column_rename)


def _load_split_dirs(
    directories: list[Path],
    column_rename: Mapping[str, str] | None,
) -> Dataset:
    from datasets import concatenate_datasets

    loaded = [load_jsonl_dataset(directory, column_rename) for directory in directories]
    if len(loaded) == 1:
        return loaded[0]
    return concatenate_datasets(loaded)


def load_jsonl_train_eval_from_splits(
    path: str | Path,
    column_rename: Mapping[str, str] | None = None,
) -> tuple[Dataset, Dataset] | None:
    """Load a train directory and every other split when both exist.

    A child directory named ``train`` (any ASCII case) is the training set,
    every row. Every other child directory that contains JSONL files is the
    eval set, concatenated in directory-name order, every row. Returns
    ``None`` when the caller should random-split: a file, a flat shard
    directory, only a train directory, or split directories with no train
    directory.

    :param path: Local dataset file or directory.
    :type path: str | Path
    :param column_rename: Existing column name to new name, applied to both sets.
    :type column_rename: Mapping[str, str] | None
    :return: ``(train, eval)``, or ``None`` when there is no separate eval split.
    :rtype: tuple[Dataset, Dataset] | None
    """
    splits = _one_level_jsonl_split_dirs(Path(path))
    train_dirs = [
        directory for directory in splits if directory.name.lower() == "train"
    ]
    eval_dirs = [directory for directory in splits if directory.name.lower() != "train"]
    if not train_dirs or not eval_dirs:
        return None
    return (
        _load_split_dirs(train_dirs, column_rename),
        _load_split_dirs(eval_dirs, column_rename),
    )
