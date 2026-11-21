# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agilerl.arena.traces_format import (
    MANIFEST_FIELDS,
    REQUIRED_MANIFEST_FIELDS,
    ExcludedCounts,
    ModelVersionCount,
    SnapshotFile,
    SnapshotManifest,
    TimeRange,
    read_manifest,
    write_manifest,
)

WHEN = datetime(2026, 11, 12, 9, tzinfo=timezone.utc)
MANIFEST = SnapshotManifest(
    snapshot_id="01SNAP",
    created_at=WHEN,
    created_by="user-9",
    row_count=2,
    session_count=1,
    model_versions=[ModelVersionCount("v1", "tiny", None, 2)],
    time_range=TimeRange(WHEN, WHEN),
    filter={"deployment": "dep-1"},
    parent_snapshot_id=None,
    score_names=[],
    label_score=None,
    excluded=ExcludedCounts(no_content=1, error_or_cancelled=0),
    organization_id=None,
    deployment_ids=["dep-1"],
    files=[SnapshotFile("calls.parquet", 2, "abc")],
)


class TestWriteManifest:
    def test_writes_every_field_and_reads_back(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "manifest.json"

        # Act
        write_manifest(path, MANIFEST)

        # Assert
        data = json.loads(path.read_text())
        assert sorted(data) == sorted(MANIFEST_FIELDS)
        assert set(REQUIRED_MANIFEST_FIELDS) <= set(MANIFEST_FIELDS)
        assert (data["format"], data["schema_version"]) == ("agilerl-traces", 1)
        assert data["created_at"] == "2026-11-12T09:00:00+00:00"
        assert data["excluded"] == {"no_content": 1, "error_or_cancelled": 0}
        assert read_manifest(path) == MANIFEST

    def test_rejects_a_value_json_cannot_encode(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        manifest = dataclasses.replace(MANIFEST, filter={"mark": object()})

        with pytest.raises(TypeError, match="not JSON serializable"):
            write_manifest(path, manifest)


class TestReadManifest:
    def test_rejects_a_non_object(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("[]")

        with pytest.raises(ValueError, match="not a JSON object"):
            read_manifest(path)

    def test_ignores_keys_the_format_does_not_define(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        path.write_text(json.dumps({**json.loads(path.read_text()), "note": "x"}))

        assert read_manifest(path) == MANIFEST

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("format", "other", "manifest format is 'other'"),
            ("schema_version", 2, "manifest schema_version is 2"),
        ],
    )
    def test_rejects_another_format_or_version(
        self, tmp_path: Path, key: str, value: object, match: str
    ) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        path.write_text(json.dumps({**json.loads(path.read_text()), key: value}))

        with pytest.raises(ValueError, match=match):
            read_manifest(path)

    def test_rejects_a_missing_field(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        data = json.loads(path.read_text())
        del data["files"]
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="missing files"):
            read_manifest(path)

    def test_defaults_the_optional_fields(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        data = json.loads(path.read_text())
        path.write_text(
            json.dumps({name: data[name] for name in REQUIRED_MANIFEST_FIELDS})
        )

        # Act
        manifest = read_manifest(path)

        # Assert
        assert manifest.snapshot_id == "01SNAP"
        assert manifest.created_at is None
        assert manifest.time_range is None
        assert manifest.filter == {}
        assert manifest.excluded == ExcludedCounts()
        assert manifest.deployment_ids == []

    def test_reads_z_timestamps(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        data = json.loads(path.read_text())
        data["created_at"] = "2026-11-12T09:00:00Z"
        data["time_range"] = {
            "start": "2026-11-12T09:00:00Z",
            "end": "2026-11-12T09:00:00Z",
        }
        path.write_text(json.dumps(data))

        assert read_manifest(path) == MANIFEST

    def test_ignores_unknown_nested_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        data = json.loads(path.read_text())
        data["model_versions"][0]["adapter"] = "lora-1"
        data["files"][0]["bytes"] = 10
        data["excluded"]["other"] = 3
        path.write_text(json.dumps(data))

        assert read_manifest(path) == MANIFEST

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        [
            ("row_count", "2", "row_count is str, not int"),
            ("row_count", True, "row_count is bool, not int"),
            ("files", {}, "files is dict, not list"),
            ("model_versions", ["v1"], "model_versions entry is str"),
            ("label_score", 3, "label_score is int, not str"),
        ],
    )
    def test_rejects_a_wrong_type(
        self, tmp_path: Path, key: str, value: object, match: str
    ) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        path.write_text(json.dumps({**json.loads(path.read_text()), key: value}))

        with pytest.raises(ValueError, match=match):
            read_manifest(path)

    @pytest.mark.parametrize(
        ("section", "key", "value", "match"),
        [
            (
                "model_versions",
                "row_count",
                "2",
                "model_versions entry row_count is str",
            ),
            (
                "model_versions",
                "checkpoint_id",
                7,
                "model_versions entry checkpoint_id is int",
            ),
            ("files", "sha256", None, "files entry sha256 is NoneType"),
            ("excluded", "no_content", "1", "excluded entry no_content is str"),
        ],
    )
    def test_rejects_a_wrong_nested_type(
        self, tmp_path: Path, section: str, key: str, value: object, match: str
    ) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        data = json.loads(path.read_text())
        entry = data[section] if section == "excluded" else data[section][0]
        entry[key] = value
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match=match):
            read_manifest(path)

    def test_rejects_a_nested_entry_missing_a_field(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        write_manifest(path, MANIFEST)
        data = json.loads(path.read_text())
        del data["files"][0]["sha256"]
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="manifest files entry"):
            read_manifest(path)


class TestTracesFormatImport:
    def test_loads_no_torch(self) -> None:
        # Act
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, agilerl.arena.traces_format; "
                    "print('torch' in sys.modules)"
                ),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Assert
        assert loaded == "False"
