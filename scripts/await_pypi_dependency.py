# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Wait until a dependency declared by a wheel is installable from PyPI.

The wheel about to be uploaded is the source of truth for what it requires.
Tag pushes for two packages race, and PyPI needs a moment to serve a new
release, so poll rather than fail on the first miss.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_POLL_SECONDS = 15

Fetcher = Callable[[str], dict | None]


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def wheel_metadata(wheel: Path) -> str:
    """Read the METADATA of a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            _fail(
                f"error: {wheel.name} holds {len(names)} .dist-info/METADATA, expected 1"
            )
        return archive.read(names[0]).decode("utf-8")


def declared_specifiers(metadata: str, name: str) -> tuple[SpecifierSet, ...]:
    """Distinct version ranges the metadata requires of ``name``.

    :param metadata: wheel METADATA text.
    :param name: distribution to look for; extras are folded together.
    :return: one entry per distinct range, in first-seen order.
    """
    wanted = canonicalize_name(name)
    found: dict[str, SpecifierSet] = {}
    for line in metadata.splitlines():
        field, _, value = line.partition(":")
        if field.strip().lower() != "requires-dist":
            continue
        try:
            requirement = Requirement(value.strip())
        except InvalidRequirement:
            _fail(f"error: unparsable Requires-Dist: {value.strip()!r}")
        if canonicalize_name(requirement.name) != wanted:
            continue
        found.setdefault(str(requirement.specifier), requirement.specifier)
    return tuple(found.values())


def _fetch_pypi(name: str) -> dict | None:
    request = urllib.request.Request(
        PYPI_JSON_URL.format(name=name),
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def published_versions(payload: dict | None) -> tuple[Version, ...]:
    """Versions PyPI can install: at least one file, none of them yanked."""
    if payload is None:
        return ()
    versions = []
    for raw, files in (payload.get("releases") or {}).items():
        if not files or all(f.get("yanked") for f in files):
            continue
        try:
            versions.append(Version(raw))
        except InvalidVersion:
            continue
    return tuple(sorted(versions))


def unsatisfied(
    specifiers: tuple[SpecifierSet, ...], versions: tuple[Version, ...]
) -> tuple[SpecifierSet, ...]:
    """Ranges that no published version satisfies."""
    return tuple(s for s in specifiers if not any(s.contains(v) for v in versions))


def await_dependency(
    specifiers: tuple[SpecifierSet, ...],
    name: str,
    *,
    fetch: Fetcher,
    timeout_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Block until every range resolves on PyPI, or fail on timeout."""
    if not specifiers:
        print(f"no {name} requirement declared; nothing to wait for")
        return
    deadline = monotonic() + timeout_seconds
    while True:
        versions = published_versions(fetch(name))
        missing = unsatisfied(specifiers, versions)
        if not missing:
            print(f"{name} satisfies {', '.join(str(s) for s in specifiers)} on PyPI")
            return
        ranges = ", ".join(f"{name}{s}" for s in missing)
        if monotonic() >= deadline:
            published = ", ".join(str(v) for v in versions) or "none"
            _fail(
                f"error: PyPI has no {name} matching {ranges} after "
                f"{timeout_seconds:.0f}s (published: {published}). "
                f"Publish the {name} release tag first."
            )
        print(f"waiting {poll_seconds:.0f}s for {ranges} on PyPI", flush=True)
        sleep(poll_seconds)


def sole_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        _fail(f"error: {dist_dir} holds {len(wheels)} wheels, expected 1")
    return wheels[0]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--requirement", required=True)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    wheel = sole_wheel(args.dist_dir)
    specifiers = declared_specifiers(wheel_metadata(wheel), args.requirement)
    await_dependency(
        specifiers,
        args.requirement,
        fetch=_fetch_pypi,
        timeout_seconds=args.timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
