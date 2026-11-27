# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native App build helper: release tag stamping and the importable code zip."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from snowflake_app import bundle

APP_DIR = Path(__file__).resolve().parents[1] / "snowflake_app"
TAG = "1.38.0-py313-cu130"


def app_build(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    shutil.copy(APP_DIR / "manifest.yml", build / "manifest.yml")
    shutil.copytree(APP_DIR / "app_code", build / "app_code")
    return build


def fake_wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "agilerl-3.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as out:
        out.writestr("agilerl/__init__.py", "")
        out.writestr("agilerl/snowflake/__init__.py", "")
        out.writestr("agilerl/snowflake/job.py", "X = 1\n")
        out.writestr("agilerl-3.0.0.dist-info/METADATA", "Version: 3.0.0\n")
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
        shipped = "".join(
            path.read_text(encoding="utf-8")
            for path in (build / "app_code").glob("*.py")
        )
        image = f"/arena_provider/images/repo/snowflake-job:{TAG}"
        assert f"- {image}\n" in manifest
        assert f'IMAGE = "{image}"' in code
        assert "0.0.0-py313-cu130" not in manifest + shipped


class TestWriteZip:
    def test_agilerl_snowflake_imports_from_the_zip(self, tmp_path: Path) -> None:
        # Arrange
        out = tmp_path / "agilerl_arena.zip"
        bundle.write_zip(out, fake_wheel(tmp_path), vendor_dir(tmp_path))

        # Act
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                (
                    f"import sys; sys.path.insert(0, {str(out)!r}); "
                    "import agilerl.snowflake.job as job, keycloak; print(job.X)"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Assert
        assert result.returncode == 0, result.stderr
        assert result.stdout == "1\n"

    def test_adds_vendored_files_without_bytecode(self, tmp_path: Path) -> None:
        out = tmp_path / "agilerl_arena.zip"

        bundle.write_zip(out, fake_wheel(tmp_path), vendor_dir(tmp_path))

        files = zipfile.ZipFile(out).namelist()
        assert files == [
            "agilerl/__init__.py",
            "agilerl/snowflake/__init__.py",
            "agilerl/snowflake/job.py",
            "agilerl-3.0.0.dist-info/METADATA",
            "keycloak/__init__.py",
            "python_keycloak-7.1.1.dist-info/METADATA",
        ]
