# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Pick the previous vX.Y.Z spoke tag for GitHub release notes."""

from __future__ import annotations

import argparse
import re
import sys

RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def tag_version(tag: str) -> tuple[int, int, int] | None:
    """Return X.Y.Z for a ``vX.Y.Z`` spoke tag, else None."""
    match = RELEASE_TAG.fullmatch(tag)
    if not match:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def previous_spoke_tag(current: str, tags: list[str]) -> str:
    """Return the highest release tag strictly older than current, or empty."""
    current_ver = tag_version(current)
    if current_ver is None:
        print(f"error: {current!r} is not a vX.Y.Z release tag", file=sys.stderr)
        raise SystemExit(1)
    earlier: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        other_ver = tag_version(tag)
        if other_ver is None or other_ver >= current_ver:
            continue
        earlier.append((other_ver, tag))
    if not earlier:
        return ""
    earlier.sort()
    return earlier[-1][1]


def main(argv: list[str] | None = None) -> None:
    """Read tag names on stdin; print the previous release tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_tag")
    args = parser.parse_args(argv)
    tags = [line.strip() for line in sys.stdin if line.strip()]
    print(previous_spoke_tag(args.current_tag, tags))


if __name__ == "__main__":
    main()
