# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Ensure the arena extra is a compatible range (or a build-time exact pin), and ``all`` unions every extra."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

try:
    import tomllib
except ModuleNotFoundError:  # python 3.10
    import tomli as tomllib  # ty: ignore[unresolved-import]

REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT_PYPROJECT = REPO_ROOT / "pyproject.toml"

# Exact pin used only while a framework +local wheel is being built.
_EXACT_RE = re.compile(
    r"^agilerl-arena(?:\[[^\]]*\])?==(\d+\.\d+\.\d+(?:\+[A-Za-z0-9._-]+)?)(?:\s*;.*)?$"
)
# Committed extra: compatible range, not an exact ==.
_RANGE_RE = re.compile(
    r"^agilerl-arena(?:\[[^\]]*\])?(>=\d+\.\d+\.\d+,<\d+\.\d+(?:\.\d+)?)(?:\s*;.*)?$"
)
_SELF_RE = re.compile(r"^agilerl\[([^\]]+)\]$")


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _extras(path: Path) -> dict[str, list[str]]:
    project = _pyproject(path).get("project", {})
    return project.get("optional-dependencies", {})


def _public_tuple(version: str) -> tuple[int, ...]:
    core = version.split("+", 1)[0]
    for sep in (".dev", "rc", "a", "b"):
        core = core.split(sep, 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    if len(parts) < 2:
        _fail(f"Cannot parse public version from {version!r}")
    return tuple(parts)


def _in_range(version: str, spec: str) -> bool:
    lower_s, upper_s = spec.split(",", 1)
    if not lower_s.startswith(">=") or not upper_s.startswith("<"):
        return False
    ver = _public_tuple(version)
    lower = _public_tuple(lower_s[2:])
    upper = _public_tuple(upper_s[1:])
    return lower <= ver < upper


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


def _last_namespaced_tag_version(prefix: str) -> str:
    """Latest ``{prefix}/vX.Y.Z`` tag in this clone. Fail closed if git or tags are missing."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "tag", "-l", f"{prefix}/v*", "--sort=-v:refname"],  # noqa: S607
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        _fail(
            f"git not found; cannot read {prefix}/v* tags "
            "(namespaced git tags are version SoT)"
        )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        _fail(f"git tag -l {prefix}/v* failed: {detail}")
    for line in result.stdout.splitlines():
        tag = line.strip()
        if tag.startswith(f"{prefix}/v"):
            return tag[len(prefix) + 2 :]
    _fail(f"No {prefix}/v* tag in this clone (namespaced git tags are version SoT)")


def _hub_semver_script() -> Path:
    """Hub-only ``scripts/lib/semver.sh``; absent in the public spoke."""
    return REPO_ROOT.parent.parent / "scripts" / "lib" / "semver.sh"


def _semver_bump(version: str, kind: str) -> str | None:
    """Bump with the hub helper compute-release-versions.sh uses; ``None`` if unusable."""
    script = _hub_semver_script()
    if not script.is_file():
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "bash",
                "-c",
                'set -euo pipefail; . "$1"; semver_bump "$2" "$3"',
                "semver_bump",
                str(script),
                version,
                kind,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _expected_arena_version(tagged: str) -> tuple[str, str | None]:
    """Arena version this release will cut, plus the kind behind it.

    Falls back to ``(tagged, None)`` — the stricter last-tag rule — when arena is
    untouched or the kind is unresolvable. Never skips the range check.
    """
    if os.environ.get("ML_REBUILD_ARENA", "").strip() != "1":
        return tagged, None
    kind = os.environ.get("ML_SEMVER_KIND", "").strip()
    if kind not in {"major", "minor", "patch"}:
        return tagged, None
    bumped = _semver_bump(tagged, kind)
    if bumped is None:
        return tagged, None
    return bumped, kind


def _check_arena_extra(extras: dict[str, list[str]]) -> str:
    """Return the arena extra specifier after checking range vs build-time pin."""
    if "arena" not in extras:
        _fail(f"Missing [project.optional-dependencies].arena in {PARENT_PYPROJECT}")

    pin_override = os.environ.get("AGILERL_ARENA_PIN", "").strip()
    reqs = [req.strip() for req in extras["arena"]]
    exact = [m.group(1) for m in (_EXACT_RE.match(req) for req in reqs) if m]
    ranges = [m.group(1) for m in (_RANGE_RE.match(req) for req in reqs) if m]

    if pin_override:
        if len(exact) != 1 or ranges:
            _fail(
                "AGILERL_ARENA_PIN requires exactly one agilerl-arena== pin in "
                f"the arena extra ({PARENT_PYPROJECT})"
            )
        if exact[0] != pin_override:
            _fail(
                f"Arena extra pins agilerl-arena=={exact[0]}, but AGILERL_ARENA_PIN="
                f"{pin_override}"
            )
        return f"=={exact[0]}"

    if exact:
        _fail(
            "Committed arena extra must be a compatible range "
            "(agilerl-arena>=X.Y.Z,<X+1), not an exact == pin. Exact pins belong "
            f"in the ML bundle / framework +local build ({PARENT_PYPROJECT})"
        )
    if len(ranges) != 1:
        _fail(
            "Expected one agilerl-arena>=X.Y.Z,<A.B range in "
            f"[project.optional-dependencies].arena ({PARENT_PYPROJECT})"
        )
    tagged = _last_namespaced_tag_version("agilerl-arena")
    expected, kind = _expected_arena_version(tagged)
    if kind is not None:
        print(
            f"arena paths changed: semver:{kind} cuts agilerl-arena "
            f"{tagged} -> {expected}",
            flush=True,
        )
    if not _in_range(expected, ranges[0]):
        if kind is None:
            _fail(
                f"agilerl-arena/v{tagged} is outside extra range {ranges[0]} "
                f"({PARENT_PYPROJECT})"
            )
        _fail(
            f"semver:{kind} -> agilerl-arena will be {expected}, outside the "
            f"committed range {ranges[0]} (last released agilerl-arena/v{tagged}; "
            f"{PARENT_PYPROJECT})"
        )
    return ranges[0]


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
    extras = _extras(PARENT_PYPROJECT)
    spec = _check_arena_extra(extras)
    _check_all_extra(extras)
    print(f"OK: arena extra {spec}; all unions every extra")


if __name__ == "__main__":
    main()
