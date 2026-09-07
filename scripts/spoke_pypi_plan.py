# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Decide which INTERNAL dists a spoke Publish-release tag must upload.

``just publish`` uploads agilerl-arena first, then agilerl, because the
framework extra is a range that must resolve on PyPI. This plan is that
order for INTERNAL bytes: an ``agilerl-arena/v*`` tag still uploads only
arena; a ``v*`` tag uploads the highest ``agilerl-arena/v*`` that satisfies
this tree's extra, then agilerl.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import NoReturn

try:
    import tomllib
except ModuleNotFoundError:  # python 3.10
    import tomli as tomllib  # ty: ignore[unresolved-import]

_PREV_PATH = Path(__file__).resolve().parent / "previous_spoke_tag.py"
_PREV_SPEC = importlib.util.spec_from_file_location("previous_spoke_tag", _PREV_PATH)
assert _PREV_SPEC is not None and _PREV_SPEC.loader is not None
previous_spoke_tag_mod = importlib.util.module_from_spec(_PREV_SPEC)
_PREV_SPEC.loader.exec_module(previous_spoke_tag_mod)
family_and_version = previous_spoke_tag_mod.family_and_version
previous_spoke_tag = previous_spoke_tag_mod.previous_spoke_tag

DIST_PREFIX = {"agilerl": "agilerl", "agilerl-arena": "agilerl_arena"}
RANGE_RE = re.compile(
    r"^agilerl-arena(?:\[[^\]]*\])?(>=\d+\.\d+\.\d+,<\d+\.\d+(?:\.\d+)?)(?:\s*;.*)?$"
)
PUBLIC_VERSION_MIN_PARTS = 2


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _public_tuple(version: str) -> tuple[int, ...]:
    core = version.split("+", 1)[0]
    for sep in (".dev", "rc", "a", "b"):
        core = core.split(sep, 1)[0]
    parts: list[int] = []
    for piece in core.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    if len(parts) < PUBLIC_VERSION_MIN_PARTS:
        _fail(f"Cannot parse public version from {version!r}")
    return tuple(parts)


def version_in_range(version: str, spec: str) -> bool:
    """Return True if ``version`` satisfies ``>=X.Y.Z,<A.B``."""
    lower_s, upper_s = spec.split(",", 1)
    if not lower_s.startswith(">=") or not upper_s.startswith("<"):
        return False
    ver = _public_tuple(version)
    lower = _public_tuple(lower_s[2:])
    upper = _public_tuple(upper_s[1:])
    return lower <= ver < upper


def arena_extra_range(pyproject: Path) -> str:
    """Return the committed ``agilerl-arena>=X.Y.Z,<A.B`` extra specifier."""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    reqs = extras.get("arena")
    if not isinstance(reqs, list):
        _fail(f"error: missing [project.optional-dependencies].arena in {pyproject}")
    ranges = [
        match.group(1)
        for match in (RANGE_RE.match(str(req).strip()) for req in reqs)
        if match
    ]
    if len(ranges) != 1:
        _fail(
            "error: expected one agilerl-arena>=X.Y.Z,<A.B range in "
            f"[project.optional-dependencies].arena ({pyproject})"
        )
    return ranges[0]


def latest_arena_tag_in_range(spec: str, tags: list[str]) -> str:
    """Highest ``agilerl-arena/vX.Y.Z`` whose version satisfies ``spec``."""
    matching: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        parsed = family_and_version(tag)
        if parsed is None:
            continue
        family, ver = parsed
        if family != "agilerl-arena":
            continue
        version = ".".join(str(part) for part in ver)
        if version_in_range(version, spec):
            matching.append((ver, tag))
    if not matching:
        _fail(f"error: no agilerl-arena/v* tag satisfies extra range {spec}")
    matching.sort()
    return matching[-1][1]


def _planned_dist(tag: str, tags: list[str]) -> dict[str, str]:
    parsed = family_and_version(tag)
    if parsed is None:
        _fail(f"error: {tag!r} is not a v* or agilerl-arena/v* release tag")
    family, ver = parsed
    version = ".".join(str(part) for part in ver)
    return {
        "package": family,
        "version": version,
        "dist_name": DIST_PREFIX[family],
        "git_tag": tag,
        "previous_tag": previous_spoke_tag(tag, tags),
    }


def build_plan(tag: str, pyproject: Path, tags: list[str]) -> dict[str, object]:
    """Return dispatch metadata plus fetch/publish rows in upload order."""
    parsed = family_and_version(tag)
    if parsed is None:
        _fail(f"error: {tag!r} is not a v* or agilerl-arena/v* release tag")
    family, _ver = parsed
    dispatch = _planned_dist(tag, tags)
    fetch = [dispatch]
    if family == "agilerl":
        arena_tag = latest_arena_tag_in_range(arena_extra_range(pyproject), tags)
        fetch = [_planned_dist(arena_tag, tags), dispatch]
    return {"dispatch": dispatch, "fetch": fetch}


def write_github_output(plan: dict, path: Path) -> None:
    """Append dispatch fields for GitHub Actions ``GITHUB_OUTPUT``."""
    dispatch = plan["dispatch"]
    with path.open("a", encoding="utf-8") as handle:
        for key, value in (
            ("tag", dispatch["git_tag"]),
            ("version", dispatch["version"]),
            ("package", dispatch["package"]),
            ("dist_name", dispatch["dist_name"]),
            ("previous_tag", dispatch["previous_tag"]),
        ):
            handle.write(f"{key}={value}\n")


def print_fetch_rows(plan: dict) -> None:
    """Print ``package\\tversion`` rows in publish order."""
    for row in plan["fetch"]:
        print(f"{row['package']}\t{row['version']}")


def main(argv: list[str] | None = None) -> None:
    """Read tag names on stdin, or print fetch rows from a saved plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--pyproject", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--print-fetch", action="store_true")
    args = parser.parse_args(argv)
    if args.print_fetch:
        if args.plan is None:
            _fail("error: --print-fetch requires --plan")
        print_fetch_rows(json.loads(args.plan.read_text(encoding="utf-8")))
        return
    if args.tag is None or args.pyproject is None:
        _fail("error: --tag and --pyproject are required")
    tags = [line.strip() for line in sys.stdin if line.strip()]
    plan = build_plan(args.tag, args.pyproject, tags)
    encoded = json.dumps(plan, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    if args.github_output is not None:
        write_github_output(plan, args.github_output)


if __name__ == "__main__":
    main()
