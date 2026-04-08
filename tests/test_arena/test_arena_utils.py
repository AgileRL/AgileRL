"""Tests for agilerl.utils.arena_utils — file helpers and archive preparation."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from agilerl.utils.arena_utils import (
    file_to_bytes,
    prepare_env_upload,
    resolve_env_config,
    resolve_env_requirements,
)


# ---------------------------------------------------------------------------
# file_to_bytes
# ---------------------------------------------------------------------------
class TestFileToBytes:
    def test_reads_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes(b"hello world")
        assert file_to_bytes(f) == b"hello world"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            file_to_bytes(tmp_path / "missing.txt")

    def test_directory_raises(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(FileNotFoundError, match="File not found"):
            file_to_bytes(d)

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        assert file_to_bytes(str(f)) == b"\x00\x01\x02"


# ---------------------------------------------------------------------------
# resolve_env_config
# ---------------------------------------------------------------------------
class TestResolveEnvConfig:
    def test_none_returns_none(self):
        assert resolve_env_config(None) is None

    def test_dict_to_json_bytes(self):
        cfg = {"render_mode": "rgb_array", "max_steps": 100}
        result = resolve_env_config(cfg)
        parsed = json.loads(result.decode("utf-8"))
        assert parsed == cfg

    def test_file_path(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text('{"key": "val"}', encoding="utf-8")
        result = resolve_env_config(cfg_file)
        assert json.loads(result) == {"key": "val"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_env_config(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# resolve_env_requirements
# ---------------------------------------------------------------------------
class TestResolveEnvRequirements:
    def test_none_returns_none(self):
        assert resolve_env_requirements(None) is None

    def test_file_path(self, tmp_path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("numpy>=1.24\ntorch>=2.0\n", encoding="utf-8")
        result = resolve_env_requirements(req_file)
        assert b"numpy" in result
        assert b"torch" in result

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_env_requirements(tmp_path / "missing.txt")


# ---------------------------------------------------------------------------
# prepare_env_upload
# ---------------------------------------------------------------------------
class TestPrepareEnvUpload:
    def _archive_names(self, archive_bytes: bytes) -> list[str]:
        """Extract member names from a tar.gz archive."""
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            return tar.getnames()

    def _read_archive_member(self, archive_bytes: bytes, name: str) -> bytes:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            member = tar.getmember(name)
            f = tar.extractfile(member)
            return f.read()

    def test_directory_source(self, tmp_path):
        src = tmp_path / "myenv"
        src.mkdir()
        (src / "env.py").write_text("class MyEnv: pass")
        (src / "utils.py").write_text("def helper(): pass")

        result = prepare_env_upload(src)
        names = self._archive_names(result)
        assert "env.py" in names
        assert "utils.py" in names

    def test_single_file_source(self, tmp_path):
        src = tmp_path / "single_env.py"
        src.write_text("class Env: pass")

        result = prepare_env_upload(src)
        names = self._archive_names(result)
        assert "single_env.py" in names

    def test_nested_directory(self, tmp_path):
        src = tmp_path / "myenv"
        src.mkdir()
        sub = src / "submodule"
        sub.mkdir()
        (sub / "helper.py").write_text("x = 1")

        result = prepare_env_upload(src)
        names = self._archive_names(result)
        assert "submodule/helper.py" in names

    def test_with_config_dict(self, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")

        config = {"render_mode": "human"}
        result = prepare_env_upload(src, config=config)
        names = self._archive_names(result)
        assert "config.json" in names

        content = self._read_archive_member(result, "config.json")
        assert json.loads(content) == config

    def test_with_config_file(self, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"a": 1}', encoding="utf-8")

        result = prepare_env_upload(src, config=cfg)
        names = self._archive_names(result)
        assert "config.json" in names

    def test_with_requirements_file(self, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")
        req = tmp_path / "requirements.txt"
        req.write_text("gymnasium>=0.29\n", encoding="utf-8")

        result = prepare_env_upload(src, requirements=req)
        names = self._archive_names(result)
        assert "requirements.txt" in names

        content = self._read_archive_member(result, "requirements.txt")
        assert b"gymnasium" in content

    def test_with_description(self, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")

        result = prepare_env_upload(src, description="A test environment")
        names = self._archive_names(result)
        assert "description.txt" in names

        content = self._read_archive_member(result, "description.txt")
        assert content == b"A test environment"

    def test_all_extras(self, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "env.py").write_text("pass")
        req = tmp_path / "req.txt"
        req.write_text("torch\n", encoding="utf-8")

        result = prepare_env_upload(
            src,
            config={"x": 1},
            requirements=req,
            description="desc",
        )
        names = self._archive_names(result)
        assert "env.py" in names
        assert "config.json" in names
        assert "requirements.txt" in names
        assert "description.txt" in names

    def test_output_is_valid_gzip(self, tmp_path):
        src = tmp_path / "env"
        src.mkdir()
        (src / "main.py").write_text("pass")

        result = prepare_env_upload(src)
        # Should be parseable as gzip tar
        with tarfile.open(fileobj=io.BytesIO(result), mode="r:gz") as tar:
            assert len(tar.getnames()) >= 1
