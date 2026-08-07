# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ensure the arena extra pins agilerl-arena's version, and ``all`` unions every extra."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # python 3.10
    import tomli as tomllib  # ty: ignore[unresolved-import]

REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT_PYPROJECT = REPO_ROOT / "pyproject.toml"
ARENA_PYPROJECT = REPO_ROOT / "agilerl-arena" / "pyproject.toml"

# Exact pin: name==X.Y.Z, optional extras, optional environment marker.
_PIN_RE = re.compile(r"^agilerl-arena(?:\[[^\]]*\])?==(\d+\.\d+\.\d+)(?:\s*;.*)?$")
# A self-reference such as ``agilerl[llm,arena]``, which stands for those extras.
_SELF_RE = re.compile(r"^agilerl\[([^\]]+)\]$")


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _extras(path: Path) -> dict[str, list[str]]:
    project = _pyproject(path).get("project", {})
    return project.get("optional-dependencies", {})


def _requirements(extra: str, extras: dict[str, list[str]], seen: set[str]) -> set[str]:
    """Resolve ``extra`` to its requirements, expanding any self-references."""
    if extra in seen:
        _fail(f"Extra {extra!r} references itself in a cycle ({PARENT_PYPROJECT})")
    seen = seen | {extra}
    if extra not in extras:
        _fail(f"Extra {extra!r} does not exist ({PARENT_PYPROJECT})")
    resolved: set[str] = set()
    for requirement in extras[extra]:
        match = _SELF_RE.match(requirement.strip())
        if match is None:
            resolved.add(requirement.strip())
            continue
        for referenced in match.group(1).split(","):
            resolved |= _requirements(referenced.strip(), extras, seen)
    return resolved


def _check_arena_pin(extras: dict[str, list[str]]) -> str:
    """Return the arena extra's agilerl-arena pin, failing unless it is the built version."""
    expected = _pyproject(ARENA_PYPROJECT).get("project", {}).get("version")
    if not expected:
        _fail(f"No project.version found in {ARENA_PYPROJECT}")
    if "arena" not in extras:
        _fail(f"Missing [project.optional-dependencies].arena in {PARENT_PYPROJECT}")

    pins = [
        match.group(1)
        for match in (_PIN_RE.match(req.strip()) for req in extras["arena"])
        if match
    ]
    if not pins:
        _fail(
            "No exact agilerl-arena==X.Y.Z pin found in "
            f"[project.optional-dependencies].arena ({PARENT_PYPROJECT})"
        )
    if len(pins) > 1:
        _fail(
            f"Multiple agilerl-arena== pins in arena extra: {pins} ({PARENT_PYPROJECT})"
        )
    if pins[0] != expected:
        _fail(
            f"Arena dep pin mismatch: parent arena extra pins "
            f"agilerl-arena=={pins[0]}, but agilerl-arena version is "
            f"{expected} ({ARENA_PYPROJECT})"
        )
    return pins[0]


def _check_all_extra(extras: dict[str, list[str]]) -> None:
    """Fail unless ``all`` resolves to exactly what every other extra asks for."""
    if "all" not in extras:
        _fail(f"Missing [project.optional-dependencies].all in {PARENT_PYPROJECT}")

    parts = sorted(name for name in extras if name != "all")
    expected: set[str] = set()
    for name in parts:
        expected |= _requirements(name, extras, seen=set())
    actual = _requirements("all", extras, seen=set())

    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            f"The 'all' extra must install exactly what {parts} do "
            f"({PARENT_PYPROJECT}).\n"
            f"  missing from all: {missing or 'none'}\n"
            f"  only in all:      {extra or 'none'}\n"
            "Write it as a self-reference so it cannot drift again: "
            f'all = ["agilerl[{",".join(parts)}]"]'
        )


def main() -> None:
    """Exit non-zero if the arena pin drifts, or ``all`` is not the union of the extras."""
    extras = _extras(PARENT_PYPROJECT)
    pinned = _check_arena_pin(extras)
    _check_all_extra(extras)
    print(f"OK: arena extra pins agilerl-arena=={pinned}; all unions every extra")


if __name__ == "__main__":
    main()
