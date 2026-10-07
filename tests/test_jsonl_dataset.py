# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public local JSONL loader."""

from __future__ import annotations

import builtins
import importlib
import json
import sys
from pathlib import Path

import pytest

from agilerl.data.jsonl import (
    is_local_jsonl_dataset,
    jsonl_shard_paths,
    load_jsonl_dataset,
    training_jsonl_paths,
)
from agilerl.models.env import LLMEnvSpec, LLMEnvType, _load_llm_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _write_split(root: Path, name: str, rows: list[str], *, column: str = "q") -> None:
    _write_jsonl(root / name / "00000.jsonl", [{column: row} for row in rows])


def _forbid_train_test_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbid(self, *args, **kwargs):
        msg = "train_test_split must not run when a separate eval split exists"
        raise AssertionError(msg)

    monkeypatch.setattr("datasets.Dataset.train_test_split", _forbid)


class TestJsonlShardPaths:
    def test_file_and_directory_order(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        later = _write_jsonl(shard_dir / "00001.jsonl", [{"x": 3}])
        first = _write_jsonl(shard_dir / "00000.jsonl", [{"x": 1}, {"x": 2}])

        assert jsonl_shard_paths(first) == [first]
        assert jsonl_shard_paths(shard_dir) == [first, later]

    def test_immediate_files_skip_nested_directories(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "flat"
        nested = shard_dir / "extra"
        nested.mkdir(parents=True)
        top = _write_jsonl(shard_dir / "00000.jsonl", [{"x": 1}])
        _write_jsonl(nested / "00001.jsonl", [{"x": 2}])

        assert jsonl_shard_paths(shard_dir) == [top]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match=r"No \.jsonl files in directory"):
            jsonl_shard_paths(empty)

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("nope")

        with pytest.raises(ValueError, match="Unsupported JSONL path"):
            jsonl_shard_paths(path)

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            jsonl_shard_paths(tmp_path / "missing.jsonl")


class TestIsLocalJsonlDataset:
    def test_jsonl_file_and_flat_directory(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "one.jsonl", [{"x": 1}])
        shard_dir = tmp_path / "flat"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "00000.jsonl", [{"x": 1}])

        assert is_local_jsonl_dataset(path) is True
        assert is_local_jsonl_dataset(shard_dir) is True

    def test_split_directories(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", ["t0"])

        assert is_local_jsonl_dataset(root) is True

    def test_empty_directory_is_false(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        assert is_local_jsonl_dataset(empty) is False


class TestTrainingJsonlPaths:
    def test_train_and_other_splits_return_train_files_only(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "cfg"
        first = _write_jsonl(root / "train" / "00000.jsonl", [{"q": "t0"}])
        later = _write_jsonl(root / "train" / "00001.jsonl", [{"q": "t1"}])
        _write_jsonl(root / "test" / "00000.jsonl", [{"q": "e0"}])

        assert training_jsonl_paths(root) == [first, later]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match=r"No \.jsonl files in directory"):
            training_jsonl_paths(empty)


class TestLoadJsonlDataset:
    def test_importable_without_make_llm_env(self) -> None:
        from agilerl.data.jsonl import load_jsonl_dataset as loader

        assert callable(loader)

    def test_single_file(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "one.jsonl", [{"q": "hi", "a": "yo"}])

        dataset = load_jsonl_dataset(path)

        assert list(dataset["q"]) == ["hi"]
        assert list(dataset["a"]) == ["yo"]

    def test_directory_loads_filename_order(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "00001.jsonl", [{"q": "b", "a": "2"}])
        _write_jsonl(shard_dir / "00000.jsonl", [{"q": "a", "a": "1"}])

        dataset = load_jsonl_dataset(shard_dir)

        assert list(dataset["q"]) == ["a", "b"]
        assert list(dataset["a"]) == ["1", "2"]

    def test_column_rename(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "cols.jsonl", [{"old": "x", "keep": 1}])

        dataset = load_jsonl_dataset(path, column_rename={"old": "new"})

        assert set(dataset.column_names) == {"new", "keep"}
        assert "old" not in dataset.column_names
        assert dataset[0]["new"] == "x"

    def test_column_rename_ignores_missing_names(self, tmp_path: Path) -> None:
        path = _write_jsonl(tmp_path / "keep.jsonl", [{"a": 1}])

        dataset = load_jsonl_dataset(path, column_rename={"missing": "x"})

        assert dataset.column_names == ["a"]

    def test_split_directories_load_every_row(self, tmp_path: Path) -> None:
        root = tmp_path / "cfg"
        _write_split(root, "train", [f"t{i}" for i in range(10)])
        _write_split(root, "test", ["e0", "e1"])
        _write_split(root, "validation", ["v0"])

        dataset = load_jsonl_dataset(root)

        assert len(dataset) == 13
        assert set(dataset["q"]) == {f"t{i}" for i in range(10)} | {"e0", "e1", "v0"}

    def test_mismatched_schemas_raise(self, tmp_path: Path) -> None:
        shard_dir = tmp_path / "bad"
        shard_dir.mkdir()
        _write_jsonl(shard_dir / "00000.jsonl", [{"q": "a"}])
        _write_jsonl(shard_dir / "00001.jsonl", [{"q": "b", "extra": 1}])

        with pytest.raises(ValueError, match="schema does not match"):
            load_jsonl_dataset(shard_dir)

    def test_agent_messages_rows_keep_trajectory_columns(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "agent.jsonl",
            [
                {
                    "uuid": "Search_Agent_000001",
                    "messages": [
                        {"role": "system", "content": "You are a search agent."},
                        {"role": "user", "content": "Find the capital of France."},
                        {"role": "assistant", "content": "Paris"},
                    ],
                    "tools": [{"type": "function", "function": {"name": "search"}}],
                    "source": "openbmb/UltraData-SFT-Agent-2609",
                    "domain": "Search_Agent-en",
                }
            ],
        )

        dataset = load_jsonl_dataset(path)

        assert set(dataset.column_names) == {
            "uuid",
            "messages",
            "tools",
            "source",
            "domain",
        }
        assert dataset[0]["uuid"] == "Search_Agent_000001"
        assert dataset[0]["messages"][1]["content"] == "Find the capital of France."

    def test_load_requires_datasets(self, tmp_path: Path, monkeypatch) -> None:
        path = _write_jsonl(tmp_path / "one.jsonl", [{"x": 1}])
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                message = "No module named 'datasets'"
                raise ImportError(message)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="datasets"):
            load_jsonl_dataset(path)


class TestJsonlModuleImport:
    def test_import_does_not_require_datasets(self, monkeypatch) -> None:
        import agilerl.data as data_pkg

        original = sys.modules.pop("agilerl.data.jsonl")
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasets":
                message = "No module named 'datasets'"
                raise ImportError(message)
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            module = importlib.import_module("agilerl.data.jsonl")

            assert callable(module.load_jsonl_dataset)
            assert callable(module.jsonl_shard_paths)
            assert callable(module.training_jsonl_paths)
            assert callable(module.is_local_jsonl_dataset)
            assert "Dataset" not in module.__dict__
            assert "concatenate_datasets" not in module.__dict__
        finally:
            sys.modules["agilerl.data.jsonl"] = original
            data_pkg.jsonl = original


class TestLoadDatasetFileSplit:
    def test_same_seed_matches_across_calls(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "rows.jsonl",
            [{"question": f"q{i}", "answer": f"a{i}"} for i in range(40)],
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

    def test_jsonl_id_does_not_dispatch_to_hub(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        path = _write_jsonl(
            tmp_path / "rows.jsonl",
            [{"question": f"q{i}", "answer": f"a{i}"} for i in range(10)],
        )
        spec = LLMEnvSpec(
            env_type=LLMEnvType.DATASET,
            objective="sft",
            dataset=str(path),
        )

        def _hf(*_args, **_kwargs):
            msg = "JSONL path must not go to the Hub"
            raise AssertionError(msg)

        monkeypatch.setattr("agilerl.models.env._load_dataset_hf", _hf)
        train, eval_ds = _load_llm_dataset(spec, seed=7)

        assert len(train) + len(eval_ds) == 10


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
