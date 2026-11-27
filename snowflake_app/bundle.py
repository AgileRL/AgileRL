# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stamp the release tag into a Native App build and write ``agilerl_arena.zip``."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def set_tag(build: Path, tag: str) -> None:
    """Replace the placeholder image tag in ``manifest.yml`` and the procedure code.

    :param build: Build directory.
    :param tag: Release tag, like ``38.0.4-py313-cu130``.
    """
    for path in [build / "manifest.yml", *build.glob("app_code/*.py")]:
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("0.0.0-py313-cu130", tag), encoding="utf-8")


def write_zip(out: Path, wheel: Path, vendor: Path) -> None:
    """Write the wheel and the vendored packages into one zip.

    :param out: Zip to write.
    :param wheel: ``agilerl`` wheel.
    :param vendor: ``--target`` directory of packages the Snowflake channel lacks.
    """
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as src:
        for name in src.namelist():
            if not name.endswith("/"):
                files[name] = src.read(name)
    for path in sorted(vendor.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(vendor).as_posix()] = path.read_bytes()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zipped:
        for name, data in files.items():
            zipped.writestr(name, data)


def main(argv: list[str]) -> None:
    """Stamp ``build`` with ``tag`` and write ``build/agilerl_arena.zip``.

    :param argv: Build directory, tag, wheel path, vendor directory.
    """
    build, tag, wheel, vendor = argv
    set_tag(Path(build), tag)
    write_zip(Path(build) / "agilerl_arena.zip", Path(wheel), Path(vendor))


if __name__ == "__main__":
    main(sys.argv[1:])
