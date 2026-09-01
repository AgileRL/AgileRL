# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Pick the previous same-family spoke tag for GitHub release notes."""

from __future__ import annotations

import argparse
import re
import sys

FRAMEWORK_TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")
ARENA_TAG = re.compile(r"^agilerl-arena/v(\d+\.\d+\.\d+)$")


def family_and_version(tag: str) -> tuple[str, tuple[int, int, int]] | None:
    """Return (family, X.Y.Z) for a public spoke tag, else None."""
    for pattern, family in ((ARENA_TAG, "agilerl-arena"), (FRAMEWORK_TAG, "agilerl")):
        match = pattern.fullmatch(tag)
        if match:
            major, minor, patch = (int(part) for part in match.group(1).split("."))
            return family, (major, minor, patch)
    return None


def previous_spoke_tag(current: str, tags: list[str]) -> str:
    """Return the highest same-family tag strictly older than current, or empty."""
    parsed = family_and_version(current)
    if parsed is None:
        print(
            f"error: {current!r} is not a v* or agilerl-arena/v* release tag",
            file=sys.stderr,
        )
        raise SystemExit(1)
    family, current_ver = parsed
    earlier: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        other = family_and_version(tag)
        if other is None:
            continue
        other_family, other_ver = other
        if other_family != family or tag == current or other_ver >= current_ver:
            continue
        earlier.append((other_ver, tag))
    if not earlier:
        return ""
    earlier.sort()
    return earlier[-1][1]


def main(argv: list[str] | None = None) -> None:
    """Read tag names on stdin; print the previous same-family tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_tag")
    args = parser.parse_args(argv)
    tags = [line.strip() for line in sys.stdin if line.strip()]
    print(previous_spoke_tag(args.current_tag, tags))


if __name__ == "__main__":
    main()
