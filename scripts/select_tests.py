#!/usr/bin/env python3
"""Pick the tests a diff can reach, and size the Linux 3.13 shards for them.

Maps changed paths onto the ``scripts/build_test_map.py`` output of the last
full run. Any change it cannot trace to test files selects every test.
Standard library only: it runs on the runner's python before ``uv sync``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SOURCE_ROOTS = ("agilerl/", "agilerl-arena/agilerl/")
TEST_ROOTS = ("tests/", "agilerl-arena/tests/")
NO_TEST_PREFIXES = ("docs/",)
NO_TEST_FILES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        ".pre-commit-config.yaml",
        ".readthedocs.yaml",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "agilerl-arena/README.md",
    }
)


def is_test_file(path: str) -> bool:
    """True for a pytest module under a test root."""
    name = path.rpartition("/")[2]
    return (
        path.startswith(TEST_ROOTS)
        and name.startswith("test_")
        and name.endswith(".py")
    )


def is_test_helper(path: str) -> bool:
    """True for a module under a test root that tests import or launch by name."""
    name = path.rpartition("/")[2]
    return (
        path.startswith(TEST_ROOTS)
        and path.endswith(".py")
        and not is_test_file(path)
        and name not in ("conftest.py", "__init__.py")
    )


def tests_using_helper(helper: str, test_tree: dict[str, str]) -> set[str] | None:
    """Test files that name ``helper``'s module, directly or through other helpers.

    :param helper: Changed helper path.
    :param test_tree: ``{path: source}`` of every ``.py`` file under a test root.
    :return: Test files, or None when a conftest uses it (fixtures reach tests
        without naming the helper).
    """
    pending = [helper]
    seen = {helper}
    found: set[str] = set()
    while pending:
        stem = pending.pop().rpartition("/")[2].removesuffix(".py")
        pattern = re.compile(rf"\b{re.escape(stem)}\b")
        for path, text in test_tree.items():
            if path in seen or not pattern.search(text):
                continue
            seen.add(path)
            if is_test_file(path):
                found.add(path)
            elif path.endswith("conftest.py"):
                return None
            else:
                pending.append(path)
    return found


def is_source_file(path: str) -> bool:
    """True for a Python module coverage measures."""
    return path.startswith(SOURCE_ROOTS) and path.endswith(".py")


def needs_no_tests(path: str) -> bool:
    """True for a path no test reads."""
    return path in NO_TEST_FILES or path.startswith(NO_TEST_PREFIXES)


def affected_tests(
    changed: list[str],
    sources: dict[str, list[str]] | None,
    test_tree: dict[str, str],
) -> tuple[set[str] | None, str]:
    """Return the selected test files, or None for every test, and why.

    :param changed: Changed paths, repo-relative, deletions included.
    :param sources: ``build_test_map.py`` output, or None without a map.
    :param test_tree: ``{path: source}`` of every ``.py`` file under a test root.
    :return: ``(test files or None, reason)``.
    """
    if sources is None:
        return None, "no test map"
    selected: set[str] = set()
    for path in changed:
        if needs_no_tests(path):
            continue
        if is_test_file(path):
            selected.add(path)
            continue
        if is_test_helper(path):
            users = tests_using_helper(path, test_tree)
            if users is None:
                return None, f"{path} is used by a conftest"
            selected.update(users)
            continue
        if not is_source_file(path):
            return None, f"{path} is not traced by the test map"
        if path not in sources:
            # A new module runs only through changed importers or new tests.
            continue
        if not sources[path]:
            return None, f"{path} runs only outside test contexts"
        selected.update(sources[path])
    return selected, f"{len(selected)} test files"


def load_durations(path: Path | None) -> dict[str, float]:
    """Read ``{node id: seconds}``; a missing file is an empty map."""
    if path is None or not path.is_file():
        return {}
    return {
        str(nodeid): float(seconds)
        for nodeid, seconds in json.loads(path.read_text()).items()
    }


def shard_count(
    selected: set[str],
    durations: dict[str, float],
    unknown: set[str],
    full_shards: int,
) -> int:
    """Shards for one family, in proportion to the full run's time.

    :param selected: Selected test files.
    :param durations: This family's ``{node id: seconds}`` from a full run.
    :param unknown: Selected files absent from every family's durations.
    :param full_shards: Shards the full run uses for this family.
    :return: 0 when the selection has no test of this family.
    """
    if not durations:
        return full_shards if selected else 0
    total = sum(durations.values())
    estimate = sum(
        seconds
        for nodeid, seconds in durations.items()
        if nodeid.partition("::")[0] in selected
    )
    if estimate == 0 and not unknown:
        return 0
    return min(full_shards, max(1, math.ceil(full_shards * estimate / total)))


@dataclass(frozen=True)
class Plan:
    """Tests to run (None for every test) and shards per Linux 3.13 family."""

    tests: list[str] | None
    cpu_shard_total: int
    gpu_shard_total: int
    reason: str


def plan(
    changed: list[str],
    sources: dict[str, list[str]] | None,
    test_tree: dict[str, str],
    cpu_durations: dict[str, float],
    gpu_durations: dict[str, float],
    cpu_full_shards: int,
    gpu_full_shards: int,
) -> Plan:
    """Selected tests and shard counts for the CPU and GPU 3.13 jobs.

    Selected paths absent from this checkout are excluded.
    """
    selected, reason = affected_tests(changed, sources, test_tree)
    if selected is None:
        return Plan(None, cpu_full_shards, gpu_full_shards, f"every test: {reason}")
    selected &= set(test_tree)
    known = {nodeid.partition("::")[0] for nodeid in (*cpu_durations, *gpu_durations)}
    unknown = selected - known
    return Plan(
        sorted(selected),
        shard_count(selected, cpu_durations, unknown, cpu_full_shards),
        shard_count(selected, gpu_durations, unknown, gpu_full_shards),
        reason,
    )


def github_outputs(result: Plan) -> str:
    """Render a plan as ``$GITHUB_OUTPUT`` lines."""
    tests = "" if result.tests is None else json.dumps(result.tests)
    lines = [f"tests={tests}"]
    for family, total in (
        ("cpu", result.cpu_shard_total),
        ("gpu", result.gpu_shard_total),
    ):
        lines.append(f"{family}_shards={json.dumps(list(range(1, total + 1)))}")
        lines.append(f"{family}_shard_total={total}")
    return "\n".join(lines) + "\n"


def read_test_tree() -> dict[str, str]:
    """``{path: source}`` of every ``.py`` file under a test root in the checkout."""
    return {
        path.as_posix(): path.read_text(encoding="utf-8")
        for root in TEST_ROOTS
        for path in Path(root).rglob("*.py")
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed", type=Path, required=True)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--cpu-durations", type=Path)
    parser.add_argument("--gpu-durations", type=Path)
    parser.add_argument("--cpu-shards", type=int, required=True)
    parser.add_argument("--gpu-shards", type=int, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    changed = [line for line in args.changed.read_text().splitlines() if line]
    sources = None
    if args.map is not None and args.map.is_file():
        sources = json.loads(args.map.read_text())["sources"]
    result = plan(
        changed,
        sources,
        read_test_tree(),
        load_durations(args.cpu_durations),
        load_durations(args.gpu_durations),
        args.cpu_shards,
        args.gpu_shards,
    )
    print(
        f"{len(changed)} changed paths → {result.reason}; "
        f"cpu shards {result.cpu_shard_total}, gpu shards {result.gpu_shard_total}",
        file=sys.stderr,
    )
    sys.stdout.write(github_outputs(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
