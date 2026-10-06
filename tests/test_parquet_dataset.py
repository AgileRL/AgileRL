# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public local parquet loader."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset

from agilerl.data.parquet import (
    is_local_parquet_dataset,
    load_parquet_dataset,
    parquet_shard_paths,
    training_parquet_paths,
)
from agilerl.models.env import (
    LLMEnvSpec,
    LLMEnvType,
    _load_llm_dataset,
)


def _write_parquet(path: Path, rows: dict[str, list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    return path


def _write_split(root: Path, name: str, rows: list[str], *, column: str = "q") -> None:
    _write_parquet(root / name / "00000.parquet", {column: rows})


def _forbid_train_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid(self, *args, **kwargs):
        msg = "train_test_split must not run when a separate eval split exists"
        raise AssertionError(msg)

    monkeypatch.setattr("datasets.Dataset.train_test_split", _forbid)


class TestParquetShardPaths:
    def test_file_and_directory_order(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        later = _write_parquet(shard_dir / "00001.parquet", {"x": [3]})
        first = _write_parquet(shard_dir / "00000.parquet", {"x": [1, 2]})

        assert parquet_shard_paths(first) == [first]
        assert parquet_shard_paths(shard_dir) == [first, later]

    def test_immediate_files_skip_nested_directories(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "flat"
        nested = shard_dir / "extra"
        nested.mkdir(parents=True)
        top = _write_parquet(shard_dir / "00000.parquet", {"x": [1]})
        _write_parquet(nested / "00001.parquet", {"x": [2]})

        assert parquet_shard_paths(shard_dir) == [top]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match=r"No \.parquet files in directory"):
            parquet_shard_paths(empty)

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("nope")

        with pytest.raises(ValueError, match="Unsupported parquet path"):
            parquet_shard_paths(path)

    def test_pq_file(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "one.pq", {"x": [1]})

        assert parquet_shard_paths(path) == [path]

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            parquet_shard_paths(tmp_path / "missing.parquet")


class TestIsLocalParquetDataset:
    def test_parquet_file_and_flat_directory(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "one.parquet", {"x": [1]})
        shard_dir = tmp_path / "flat"
        shard_dir.mkdir()
        _write_parquet(shard_dir / "00000.parquet", {"x": [1]})

        assert is_local_parquet_dataset(path) is True
        assert is_local_parquet_dataset(shard_dir) is True

    def test_split_directories(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", ["t0"])

        assert is_local_parquet_dataset(root) is True

    def test_empty_directory_is_false(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        assert is_local_parquet_dataset(empty) is False


class TestTrainingParquetPaths:
    def test_train_and_other_splits_return_train_files_only(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "cfg"
        first = _write_parquet(root / "train" / "00000.parquet", {"q": ["t0"]})
        later = _write_parquet(root / "train" / "00001.parquet", {"q": ["t1"]})
        _write_parquet(root / "test" / "00000.parquet", {"q": ["e0"]})
        _write_parquet(root / "validation" / "00000.parquet", {"q": ["v0"]})

        assert training_parquet_paths(root) == [first, later]

    def test_train_directory_match_is_case_insensitive(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        train = _write_parquet(root / "Train" / "00000.parquet", {"q": ["t0"]})
        _write_parquet(root / "test" / "00000.parquet", {"q": ["e0"]})

        assert training_parquet_paths(root) == [train]

    def test_train_only_directory_returns_every_shard(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        train = _write_parquet(root / "train" / "00000.parquet", {"q": ["t0"]})

        assert training_parquet_paths(root) == [train]

    def test_splits_without_train_return_every_shard(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        test = _write_parquet(root / "test" / "00000.parquet", {"q": ["e0"]})
        validation = _write_parquet(
            root / "validation" / "00000.parquet", {"q": ["v0"]}
        )

        assert training_parquet_paths(root) == [test, validation]

    def test_flat_shard_directory_returns_immediate_files(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        first = _write_parquet(shard_dir / "00000.parquet", {"q": ["a"]})
        later = _write_parquet(shard_dir / "00001.parquet", {"q": ["b"]})

        assert training_parquet_paths(shard_dir) == [first, later]

    def test_file_returns_the_file(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "one.parquet", {"q": ["t0"]})

        assert training_parquet_paths(path) == [path]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match=r"No \.parquet files in directory"):
            training_parquet_paths(empty)


class TestLoadParquetDataset:
    def test_importable_without_make_llm_env(self) -> None:
        from agilerl.data.parquet import load_parquet_dataset as loader

        assert callable(loader)

    def test_single_file(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "one.parquet", {"q": ["hi"], "a": ["yo"]})

        dataset = load_parquet_dataset(path)

        assert list(dataset["q"]) == ["hi"]
        assert list(dataset["a"]) == ["yo"]

    def test_directory_loads_filename_order_without_pandas(
        self, tmp_path, monkeypatch
    ) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_parquet(shard_dir / "00001.parquet", {"q": ["b"], "a": ["2"]})
        _write_parquet(shard_dir / "00000.parquet", {"q": ["a"], "a": ["1"]})

        def _boom(*_args, **_kwargs):
            msg = "pandas must not load or concat parquet shards"
            raise AssertionError(msg)

        monkeypatch.setattr("pandas.read_parquet", _boom)
        monkeypatch.setattr("pandas.concat", _boom)

        dataset = load_parquet_dataset(shard_dir)

        assert list(dataset["q"]) == ["a", "b"]
        assert list(dataset["a"]) == ["1", "2"]

    def test_column_rename(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "cols.parquet", {"old": ["x"], "keep": [1]})

        dataset = load_parquet_dataset(path, column_rename={"old": "new"})

        assert set(dataset.column_names) == {"new", "keep"}
        assert "old" not in dataset.column_names
        assert dataset[0]["new"] == "x"

    def test_column_rename_ignores_missing_names(self, tmp_path: Path) -> None:
        path = _write_parquet(tmp_path / "keep.parquet", {"a": [1]})

        dataset = load_parquet_dataset(path, column_rename={"missing": "x"})

        assert dataset.column_names == ["a"]

    def test_split_directories_load_every_row(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", [f"t{i}" for i in range(10)])
        _write_split(root, "test", ["e0", "e1"])
        _write_split(root, "validation", ["v0"])

        dataset = load_parquet_dataset(root)

        assert len(dataset) == 13
        assert set(dataset["q"]) == {f"t{i}" for i in range(10)} | {"e0", "e1", "v0"}

    def test_mismatched_schemas_raise(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "bad"
        shard_dir.mkdir()
        _write_parquet(shard_dir / "00000.parquet", {"q": ["a"]})
        _write_parquet(shard_dir / "00001.parquet", {"q": ["b"], "extra": [1]})

        with pytest.raises(ValueError, match="schema does not match"):
            load_parquet_dataset(shard_dir)

    def test_load_requires_datasets(self, tmp_path: Path, monkeypatch) -> None:
        path = _write_parquet(tmp_path / "one.parquet", {"x": [1]})
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                message = "No module named 'datasets'"
                raise ImportError(message)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="datasets"):
            load_parquet_dataset(path)


class TestParquetModuleImport:
    def test_import_does_not_require_datasets(self, monkeypatch) -> None:
        import agilerl.data as data_pkg

        original = sys.modules.pop("agilerl.data.parquet")
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                message = "No module named 'datasets'"
                raise ImportError(message)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            module = importlib.import_module("agilerl.data.parquet")

            assert callable(module.load_parquet_dataset)
            assert callable(module.parquet_shard_paths)
            assert callable(module.training_parquet_paths)
            assert callable(module.is_local_parquet_dataset)
            assert "Dataset" not in module.__dict__
            assert "concatenate_datasets" not in module.__dict__
        finally:
            sys.modules["agilerl.data.parquet"] = original
            data_pkg.parquet = original


class TestLoadDatasetFileSplit:
    def test_same_seed_matches_across_calls(self, tmp_path: Path) -> None:
        path = _write_parquet(
            tmp_path / "rows.parquet",
            {
                "question": [f"q{i}" for i in range(40)],
                "answer": [f"a{i}" for i in range(40)],
            },
        )
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(path),
        )

        train_a, test_a = _load_llm_dataset(spec, seed=7)
        train_b, test_b = _load_llm_dataset(spec, seed=7)

        assert train_a["question"] == train_b["question"]
        assert test_a["question"] == test_b["question"]

    def test_hf_id_still_dispatches_to_hub(self, monkeypatch) -> None:
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset="org/dataset",
        )
        calls: list[str] = []

        def _hf(loaded_spec, dataset, *, seed=None):
            calls.append(dataset)
            return Dataset.from_dict({"x": [1]}), Dataset.from_dict({"x": [2]})

        monkeypatch.setattr("agilerl.models.env._load_dataset_hf", _hf)
        _load_llm_dataset(spec)

        assert calls == ["org/dataset"]


class TestSeparateEvalSplits:
    def test_train_and_test_use_every_row(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", [f"t{i}" for i in range(10)], column="old")
        _write_split(root, "test", ["e0", "e1"], column="old")
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(root),
            columns={"old": "question"},
        )
        _forbid_train_test_split(monkeypatch)

        train, eval_ds = _load_llm_dataset(spec, seed=7)

        assert list(train["question"]) == [f"t{i}" for i in range(10)]
        assert list(eval_ds["question"]) == ["e0", "e1"]
        assert "old" not in train.column_names

    def test_validation_rows_stay_in_eval(self, tmp_path: Path, monkeypatch) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", [f"t{i}" for i in range(10)])
        _write_split(root, "test", ["e0", "e1"])
        _write_split(root, "validation", ["v0", "v1", "v2"])
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(root),
        )
        _forbid_train_test_split(monkeypatch)

        train, eval_ds = _load_llm_dataset(spec)

        assert list(train["q"]) == [f"t{i}" for i in range(10)]
        assert list(eval_ds["q"]) == ["e0", "e1", "v0", "v1", "v2"]

    def test_train_directory_match_is_case_insensitive(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "Train", ["t0"])
        _write_split(root, "test", ["e0"])
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(root),
        )
        _forbid_train_test_split(monkeypatch)

        train, eval_ds = _load_llm_dataset(spec)

        assert list(train["q"]) == ["t0"]
        assert list(eval_ds["q"]) == ["e0"]

    def test_only_train_directory_keeps_random_holdout(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", [f"t{i}" for i in range(10)])
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(root),
        )
        calls: list[dict] = []
        original = Dataset.train_test_split

        def _record(self, *args, **kwargs):
            calls.append(kwargs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Dataset, "train_test_split", _record)

        train, eval_ds = _load_llm_dataset(spec, seed=7)

        assert calls
        assert len(train) + len(eval_ds) == 10
        assert len(train) < 10
        assert len(eval_ds) > 0

    def test_splits_without_train_keep_random_holdout(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "test", [f"e{i}" for i in range(6)])
        _write_split(root, "validation", [f"v{i}" for i in range(4)])
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(root),
        )
        calls: list[dict] = []
        original = Dataset.train_test_split

        def _record(self, *args, **kwargs):
            calls.append(kwargs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Dataset, "train_test_split", _record)

        train, eval_ds = _load_llm_dataset(spec, seed=7)

        assert calls
        combined = list(train["q"]) + list(eval_ds["q"])
        assert sorted(combined) == sorted(
            [f"e{i}" for i in range(6)] + [f"v{i}" for i in range(4)]
        )
