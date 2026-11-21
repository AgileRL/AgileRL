# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App build helper: release tag stamping and the importable code zip."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from snowflake_app import bundle

APP_DIR = Path(__file__).resolve().parents[1] / "snowflake_app"
TAG = "1.38.0-py313-cu130"


def app_build(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    (build / "app_code").mkdir(parents=True)
    shutil.copy(APP_DIR / "manifest.yml", build / "manifest.yml")
    shutil.copy(
        APP_DIR / "app_code" / "arena_app.py", build / "app_code" / "arena_app.py"
    )
    return build


def fake_wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "agilerl_arena-1.38.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as out:
        out.writestr("agilerl/arena/__init__.py", "")
        out.writestr("agilerl/arena/snowflake/job.py", "X = 1\n")
        out.writestr("agilerl_arena-1.38.0.dist-info/METADATA", "Version: 1.38.0\n")
    return wheel


def vendor_dir(tmp_path: Path) -> Path:
    vendor = tmp_path / "vendor"
    (vendor / "keycloak" / "__pycache__").mkdir(parents=True)
    (vendor / "keycloak" / "__init__.py").write_text("")
    (vendor / "keycloak" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (vendor / "python_keycloak-7.1.1.dist-info").mkdir()
    (vendor / "python_keycloak-7.1.1.dist-info" / "METADATA").write_text("")
    return vendor


class TestSetTag:
    def test_replaces_the_placeholder_tag_in_the_manifest_and_procedure_code(
        self, tmp_path: Path
    ) -> None:
        build = app_build(tmp_path)

        bundle.set_tag(build, TAG)

        manifest = (build / "manifest.yml").read_text(encoding="utf-8")
        code = (build / "app_code" / "arena_app.py").read_text(encoding="utf-8")
        image = f"/arena_provider/images/repo/snowflake-job:{TAG}"
        assert f"- {image}\n" in manifest
        assert f'IMAGE = "{image}"' in code
        assert "0.0.0-py313-cu130" not in manifest + code


class TestWriteZip:
    def test_writes_directory_entries_for_every_package_directory(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "agilerl_arena.zip"

        bundle.write_zip(out, fake_wheel(tmp_path), vendor_dir(tmp_path))

        names = zipfile.ZipFile(out).namelist()
        assert [name for name in names if name.endswith("/")] == [
            "agilerl/",
            "agilerl/arena/",
            "agilerl/arena/snowflake/",
            "agilerl_arena-1.38.0.dist-info/",
            "keycloak/",
            "python_keycloak-7.1.1.dist-info/",
        ]

    def test_adds_vendored_files_without_bytecode(self, tmp_path: Path) -> None:
        out = tmp_path / "agilerl_arena.zip"

        bundle.write_zip(out, fake_wheel(tmp_path), vendor_dir(tmp_path))

        files = [n for n in zipfile.ZipFile(out).namelist() if not n.endswith("/")]
        assert files == [
            "agilerl/arena/__init__.py",
            "agilerl/arena/snowflake/job.py",
            "agilerl_arena-1.38.0.dist-info/METADATA",
            "keycloak/__init__.py",
            "python_keycloak-7.1.1.dist-info/METADATA",
        ]
