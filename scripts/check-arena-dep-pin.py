# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ensure agilerl's arena extra pins match agilerl-arena's package version."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT_PYPROJECT = REPO_ROOT / "pyproject.toml"
ARENA_PYPROJECT = REPO_ROOT / "agilerl-arena" / "pyproject.toml"

_VERSION_RE = re.compile(r'(?m)^version\s*=\s*"([^"]+)"')
_ARENA_EXTRA_RE = re.compile(r"(?ms)^arena\s*=\s*\[(.*?)\]")
# Exact pin: name==X.Y.Z, optional extras, optional environment marker.
_PIN_RE = re.compile(r"^agilerl-arena(?:\[[^\]]*\])?==(\d+\.\d+\.\d+)(?:\s*;.*)?$")


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _project_version(path: Path) -> str:
    match = _VERSION_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        _fail(f'No version = "..." found in {path}')
    return match.group(1)


def _arena_extra_reqs(path: Path) -> list[str]:
    match = _ARENA_EXTRA_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def main() -> None:
    """Exit non-zero if the arena extra pin drifts from agilerl-arena's version."""
    expected = _project_version(ARENA_PYPROJECT)
    arena_reqs = _arena_extra_reqs(PARENT_PYPROJECT)
    if not arena_reqs:
        _fail(f"Missing [project.optional-dependencies].arena in {PARENT_PYPROJECT}")

    pins: list[str] = []
    for req in arena_reqs:
        match = _PIN_RE.match(req.strip())
        if match:
            pins.append(match.group(1))

    if not pins:
        _fail(
            "No exact agilerl-arena==X.Y.Z pin found in "
            f"[project.optional-dependencies].arena ({PARENT_PYPROJECT})"
        )
    if len(pins) > 1:
        _fail(
            f"Multiple agilerl-arena== pins in arena extra: {pins} ({PARENT_PYPROJECT})"
        )

    pinned = pins[0]
    if pinned != expected:
        _fail(
            f"Arena dep pin mismatch: parent arena extra pins "
            f"agilerl-arena=={pinned}, but agilerl-arena version is "
            f"{expected} ({ARENA_PYPROJECT})"
        )

    print(f"OK: agilerl arena extra pins agilerl-arena=={pinned}")


if __name__ == "__main__":
    main()
