# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for arena.utils — upload helpers and filename parsing."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from agilerl.arena.exceptions import ArenaFileNotFoundError
from agilerl.arena.utils import (
    discover_env_sidecars,
    extract_filename,
    multipart_text_fields,
    order_dataset_fields,
    prepare_env_upload,
    prepare_file_upload,
    sort_dataset_search_by_downloads,
)


class TestOrderDatasetFields:
    def test_name_and_hf_dataset_id_first(self):
        row = {
            "id": 1,
            "category": "reasoning",
            "name": "ds",
            "hf_dataset_id": "org/foo",
        }
        ordered = order_dataset_fields(row)
        assert list(ordered.keys()) == ["name", "hf_dataset_id", "id", "category"]

    def test_always_includes_name_and_hf_dataset_id_keys(self):
        row = {"id": 1, "name": "ds"}
        ordered = order_dataset_fields(row)
        assert list(ordered.keys()) == ["name", "hf_dataset_id", "id"]
        assert ordered["hf_dataset_id"] is None

    def test_leaves_other_keys_unchanged(self):
        row = {"hf_dataset_id": "org/bar", "name": "ds", "downloads": 10}
        ordered = order_dataset_fields(row)
        assert list(ordered.keys()) == ["name", "hf_dataset_id", "downloads"]
        assert ordered["hf_dataset_id"] == "org/bar"


class TestMultipartTextFields:
    def test_omits_none_values(self):
        fields = multipart_text_fields(
            {"name": "ds", "description": None, "category": "sft"},
        )
        assert fields == {"name": (None, "ds"), "category": (None, "sft")}


class TestSortDatasetSearchByDownloads:
    def test_sorts_descending(self):
        rows = [
            {"name": "a", "downloads": 100},
            {"name": "b", "downloads": 500},
            {"name": "c", "downloads": 50},
        ]
        assert sort_dataset_search_by_downloads(rows) == [
            {"name": "b", "downloads": 500},
            {"name": "a", "downloads": 100},
            {"name": "c", "downloads": 50},
        ]

    def test_missing_downloads_treated_as_zero(self):
        rows = [{"name": "a", "downloads": 10}, {"name": "b"}]
        assert sort_dataset_search_by_downloads(rows)[0]["name"] == "a"


class TestExtractFilename:
    def test_returns_none_for_none(self):
        assert extract_filename(None) is None

    def test_returns_none_for_empty_string(self):
        assert extract_filename("") is None

    def test_extracts_unquoted_filename(self):
        assert extract_filename("attachment; filename=report.csv") == "report.csv"

    def test_extracts_quoted_filename(self):
        assert extract_filename('attachment; filename="report.csv"') == "report.csv"

    def test_returns_none_when_no_filename_part(self):
        assert extract_filename("inline") is None

    def test_handles_extra_params(self):
        result = extract_filename("attachment; foo=bar; filename=data.csv; baz=qux")
        assert result == "data.csv"


def _read_payload(payload) -> bytes:
    """Read an upload payload (open binary handle or raw bytes) into bytes."""
    if isinstance(payload, bytes):
        return payload
    try:
        return payload.read()
    finally:
        payload.close()


class TestPrepareEnvUpload:
    @staticmethod
    def _archive_names(archive) -> list[str]:
        archive_bytes = _read_payload(archive)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            return tar.getnames()

    def test_directory_source(self, tmp_path: Path):
        src = tmp_path / "myenv"
        src.mkdir()
        (src / "env.py").write_text("class MyEnv: pass")
        (src / "utils.py").write_text("def helper(): pass")

        name, archive = prepare_env_upload(src)
        assert name == "myenv.tar.gz"
        names = self._archive_names(archive)
        assert "env.py" in names
        assert "utils.py" in names

    def test_single_file_source(self, tmp_path: Path):
        src = tmp_path / "single_env.py"
        src.write_text("class Env: pass")

        name, archive = prepare_env_upload(src)
        assert name == "single_env.tar.gz"
        names = self._archive_names(archive)
        assert "single_env.py" in names

    def test_nested_directory(self, tmp_path: Path):
        src = tmp_path / "myenv"
        src.mkdir()
        sub = src / "submodule"
        sub.mkdir()
        (sub / "helper.py").write_text("x = 1")

        _, archive = prepare_env_upload(src)
        names = self._archive_names(archive)
        assert "submodule/helper.py" in names

    def test_existing_tar_gz_read_as_is(self, tmp_path: Path):
        archive_path = tmp_path / "env.tar.gz"
        archive_path.write_bytes(b"fake-archive-content")

        name, archive = prepare_env_upload(archive_path)
        assert name == "env.tar.gz"
        assert _read_payload(archive) == b"fake-archive-content"

    def test_bytes_passthrough(self):
        raw = b"raw-bytes"
        name, archive = prepare_env_upload(raw)
        assert name == "environment.tar.gz"
        assert archive is raw

    def test_missing_path_raises(self):
        with pytest.raises(ArenaFileNotFoundError, match="not found"):
            prepare_env_upload("/does/not/exist.tar.gz")

    def test_output_is_valid_gzip(self, tmp_path: Path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "main.py").write_text("pass")

        _, archive = prepare_env_upload(src)
        with tarfile.open(
            fileobj=io.BytesIO(_read_payload(archive)), mode="r:gz"
        ) as tar:
            assert len(tar.getnames()) >= 1


class TestPrepareFileUpload:
    def test_path_upload(self, tmp_path: Path):
        path = tmp_path / "cfg.yaml"
        path.write_bytes(b"key: val\n")
        name, payload, content_type = prepare_file_upload(
            path,
            default_name="default.yaml",
            content_type="application/x-yaml",
        )
        assert name == "cfg.yaml"
        assert _read_payload(payload) == b"key: val\n"
        assert content_type == "application/x-yaml"

    def test_bytes_upload(self):
        name, payload, content_type = prepare_file_upload(
            b"raw",
            default_name="default.txt",
            content_type="text/plain",
        )
        assert name == "default.txt"
        assert payload == b"raw"
        assert content_type == "text/plain"

    def test_bytes_filename_override(self):
        name, payload, content_type = prepare_file_upload(
            b"raw",
            default_name="default.txt",
            content_type="text/plain",
            filename="custom.txt",
        )
        assert name == "custom.txt"
        assert payload == b"raw"
        assert content_type == "text/plain"

    def test_filename_override(self, tmp_path: Path):
        path = tmp_path / "train-00000-of-00001.parquet"
        path.write_bytes(b"PAR1")
        name, payload, content_type = prepare_file_upload(
            path,
            default_name="dataset.parquet",
            content_type="application/vnd.apache.parquet",
            filename="main/train-00000-of-00001.parquet",
        )
        assert name == "main/train-00000-of-00001.parquet"
        assert _read_payload(payload) == b"PAR1"
        assert content_type == "application/vnd.apache.parquet"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ArenaFileNotFoundError, match="Upload file not found"):
            prepare_file_upload(
                tmp_path / "missing.txt",
                default_name="default.txt",
                content_type="text/plain",
            )


class TestDiscoverEnvSidecars:
    def _dir(self, tmp_path: Path, *files: str) -> Path:
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("x = 1")
        for f in files:
            (src / f).write_text("data")
        return src

    def test_detects_requirements_txt_at_top_level(self, tmp_path: Path):
        src = self._dir(tmp_path, "requirements.txt")
        requirements, env_config = discover_env_sidecars(
            src, requirements=None, env_config=None
        )
        assert Path(requirements) == src / "requirements.txt"
        assert env_config is None

    def test_detects_env_config_yaml(self, tmp_path: Path):
        src = self._dir(tmp_path, "env_config.yaml")
        _, env_config = discover_env_sidecars(src, requirements=None, env_config=None)
        assert Path(env_config) == src / "env_config.yaml"

    def test_detects_env_config_json(self, tmp_path: Path):
        src = self._dir(tmp_path, "env_config.json")
        _, env_config = discover_env_sidecars(src, requirements=None, env_config=None)
        assert Path(env_config) == src / "env_config.json"

    def test_env_config_precedence_yaml_over_yml_over_json(self, tmp_path: Path):
        src = self._dir(
            tmp_path, "env_config.json", "env_config.yml", "env_config.yaml"
        )
        _, env_config = discover_env_sidecars(src, requirements=None, env_config=None)
        assert Path(env_config) == src / "env_config.yaml"

    def test_explicit_requirements_takes_precedence(self, tmp_path: Path):
        src = self._dir(tmp_path, "requirements.txt")
        explicit = tmp_path / "other-reqs.txt"
        explicit.write_text("numpy")
        requirements, _ = discover_env_sidecars(
            src, requirements=explicit, env_config=None
        )
        assert requirements == explicit

    def test_explicit_env_config_takes_precedence(self, tmp_path: Path):
        src = self._dir(tmp_path, "env_config.yaml")
        explicit = tmp_path / "other-cfg.json"
        explicit.write_text("{}")
        _, env_config = discover_env_sidecars(
            src, requirements=None, env_config=explicit
        )
        assert env_config == explicit

    def test_no_sidecars_returns_none(self, tmp_path: Path):
        src = self._dir(tmp_path)
        requirements, env_config = discover_env_sidecars(
            src, requirements=None, env_config=None
        )
        assert requirements is None
        assert env_config is None

    def test_only_top_level_not_nested(self, tmp_path: Path):
        src = self._dir(tmp_path)
        nested = src / "pkg"
        nested.mkdir()
        (nested / "requirements.txt").write_text("numpy")
        requirements, _ = discover_env_sidecars(src, requirements=None, env_config=None)
        assert requirements is None

    def test_non_directory_source_is_passthrough(self, tmp_path: Path):
        single = tmp_path / "env.py"
        single.write_text("x = 1")
        requirements, env_config = discover_env_sidecars(
            single, requirements=None, env_config=None
        )
        assert requirements is None
        assert env_config is None

    def test_bytes_source_is_passthrough(self):
        requirements, env_config = discover_env_sidecars(
            b"raw", requirements=None, env_config=None
        )
        assert requirements is None
        assert env_config is None
