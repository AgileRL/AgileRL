#!/usr/bin/env python3
"""Run one duration-balanced shard of the oss/agilerl pytest suite.

Node ids stay in-process (and on a file between roots) so a large suite does
not hit the OS argument limit. With no durations file, shards are equal
contiguous slices. With one, each test is packed onto the lightest shard.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_ROOTS = ("tests", "agilerl-arena/tests")


def shard_nodeids(
    nodeids: list[str],
    total_shards: int,
    shard_index_one_based: int,
    durations: dict[str, float] | None = None,
) -> list[str]:
    """Return the node ids for this shard, in collection order."""
    count = len(nodeids)
    if count == 0:
        return []

    if not durations:
        base_size, extra_shards = divmod(count, total_shards)
        shard_zero_based = shard_index_one_based - 1
        start = shard_zero_based * base_size + min(shard_zero_based, extra_shards)
        size = base_size + (1 if shard_zero_based < extra_shards else 0)
        return nodeids[start : start + size]

    known = [duration for duration in durations.values() if duration >= 0]
    global_default = statistics.median(known) if known else 1.0
    file_totals: dict[str, list[float]] = {}
    for nodeid, duration in durations.items():
        file_totals.setdefault(nodeid.partition("::")[0], []).append(duration)
    file_default = {
        path: sum(values) / len(values) for path, values in file_totals.items()
    }

    def weight(nodeid: str) -> float:
        if nodeid in durations:
            return max(durations[nodeid], 0.0)
        return file_default.get(nodeid.partition("::")[0], global_default)

    order = sorted(
        range(count),
        key=lambda index: (-weight(nodeids[index]), index),
    )
    loads = [0.0] * total_shards
    assignment: list[list[int]] = [[] for _ in range(total_shards)]
    for index in order:
        shard = min(range(total_shards), key=lambda item: (loads[item], item))
        loads[shard] += weight(nodeids[index])
        assignment[shard].append(index)

    selected = assignment[shard_index_one_based - 1]
    return [nodeids[index] for index in sorted(selected)]


def load_durations(path: Path | None) -> dict[str, float]:
    """Read ``{node id: seconds}``. A missing or broken file means count split."""
    if path is None or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"Ignoring durations file {path}: {error}", file=sys.stderr)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(nodeid): float(duration)
        for nodeid, duration in raw.items()
        if isinstance(duration, (int, float))
    }


def strip_xdist_group(name: str) -> str:
    """Drop xdist's ``@group`` suffix. An ``@`` inside parametrisation stays."""
    at = name.rfind("@")
    if at == -1:
        return name
    if at > name.rfind("]"):
        return name[:at]
    return name


def nodeid_from_junit(classname: str, name: str) -> str | None:
    """Rebuild the pytest node id for one junit ``<testcase>``."""
    parts = classname.split(".")
    file_index = next(
        (
            index
            for index in reversed(range(len(parts)))
            if parts[index].startswith("test_")
        ),
        None,
    )
    if file_index is None:
        return None
    file_path = "/".join(parts[: file_index + 1]) + ".py"
    class_chain = parts[file_index + 1 :]
    return "::".join([file_path, *class_chain, strip_xdist_group(name)])


def collect_durations(paths: list[Path]) -> dict[str, float]:
    """Take the slower time when the same node id appears twice."""
    durations: dict[str, float] = {}
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            nodeid = nodeid_from_junit(
                case.get("classname") or "",
                case.get("name") or "",
            )
            if nodeid is None:
                continue
            seconds = round(float(case.get("time") or 0.0), 2)
            durations[nodeid] = max(durations.get(nodeid, 0.0), seconds)
    return durations


def collection_args(pytest_args: list[str]) -> list[str]:
    """Keep the marker and import mode. Drop xdist, coverage, and junit."""
    kept: list[str] = []
    index = 0
    while index < len(pytest_args):
        arg = pytest_args[index]
        if arg in {"-m", "--import-mode"} and index + 1 < len(pytest_args):
            kept.extend((arg, pytest_args[index + 1]))
            index += 2
            continue
        if arg.startswith("--import-mode="):
            kept.append(arg)
        index += 1
    return kept


def extract_long_option(pytest_args: list[str], *names: str) -> str | None:
    """Return the value of ``--flag`` or ``--flag=value`` from *pytest_args*."""
    prefixes = tuple(f"{name}=" for name in names)
    for index, arg in enumerate(pytest_args):
        if arg in names:
            if index + 1 < len(pytest_args):
                return pytest_args[index + 1]
            return None
        for prefix in prefixes:
            if arg.startswith(prefix):
                return arg.partition("=")[2]
    return None


def without_option(pytest_args: list[str], *names: str) -> list[str]:
    """Copy *pytest_args* without ``--flag`` / ``--flag=value`` entries."""
    prefixes = tuple(f"{name}=" for name in names)
    out: list[str] = []
    skip_next = False
    for arg in pytest_args:
        if skip_next:
            skip_next = False
            continue
        if arg in names:
            skip_next = True
            continue
        if any(arg.startswith(prefix) for prefix in prefixes):
            continue
        out.append(arg)
    return out


def with_cov_append(pytest_args: list[str]) -> list[str]:
    """Add ``--cov-append`` when coverage is already enabled."""
    if "--cov-append" in pytest_args:
        return pytest_args
    if any(arg == "--cov" or arg.startswith("--cov=") for arg in pytest_args):
        return [*pytest_args, "--cov-append"]
    return pytest_args


def coverage_enabled(pytest_args: list[str]) -> bool:
    """True when pytest-cov will write a coverage data file."""
    return any(arg == "--cov" or arg.startswith("--cov=") for arg in pytest_args)


def junit_inner(text: str) -> str:
    """Return ``<testsuite>`` bodies from a pytest junit XML document."""
    start = text.find("<testsuite")
    if start < 0:
        return ""
    close_suites = text.rfind("</testsuites>")
    if close_suites >= 0:
        return text[start:close_suites].strip()
    close_suite = text.rfind("</testsuite>")
    if close_suite >= 0:
        return text[start : close_suite + len("</testsuite>")].strip()
    return text[start:].strip()


def merge_junit_xml(sources: list[Path], dest: Path) -> None:
    """Write one ``<testsuites>`` document from per-root pytest reports."""
    present = [path for path in sources if path.is_file()]
    if not present:
        return
    if len(present) == 1:
        if present[0] != dest:
            dest.write_bytes(present[0].read_bytes())
        return
    inner = [
        chunk
        for chunk in (junit_inner(path.read_text(encoding="utf-8")) for path in present)
        if chunk
    ]
    dest.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<testsuites>\n'
        + "\n".join(inner)
        + "\n</testsuites>\n",
        encoding="utf-8",
    )


def combine_coverage() -> None:
    """Merge xdist ``.coverage.*`` files so the next root can ``--cov-append``."""
    subprocess.run(
        [sys.executable, "-m", "coverage", "combine"],
        check=False,
    )


def collect_nodeids(root: Path, pytest_args: list[str]) -> list[str]:
    """Collect one root with addopts cleared so xdist does not start."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:xdist",
        "-p",
        "no:cov",
        "-o",
        "addopts=",
        *collection_args(pytest_args),
        str(root),
    ]
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.stderr.write(result.stdout)
        raise RuntimeError(f"collection failed for {root} ({result.returncode})")
    return [line for line in result.stdout.splitlines() if "::" in line]


def group_by_root(
    nodeids: list[str], roots: list[Path]
) -> list[tuple[Path, list[str]]]:
    """Split node ids by the root they were collected from."""
    ordered = sorted(roots, key=lambda path: len(path.as_posix()), reverse=True)
    groups: list[tuple[Path, list[str]]] = [(root, []) for root in roots]
    by_root = {root: ids for root, ids in groups}
    for nodeid in nodeids:
        for root in ordered:
            prefix = root.as_posix().rstrip("/") + "/"
            if nodeid.startswith(prefix):
                by_root[root].append(nodeid)
                break
        else:
            raise RuntimeError(f"{nodeid} matches no root")
    return groups


def exec_nodeids(pytest_args: list[str], nodeid_file: Path) -> int:
    """Run pytest on the node ids in *nodeid_file*."""
    import pytest

    nodeids = [
        line for line in nodeid_file.read_text(encoding="utf-8").splitlines() if line
    ]
    if not nodeids:
        return 0
    return int(pytest.main([*pytest_args, *nodeids]))


def run_shard(
    roots: list[Path],
    total_shards: int,
    shard_index: int,
    durations_file: Path | None,
    pytest_args: list[str],
) -> int:
    """Collect every root, keep this shard, and run each root in its own process."""
    collected: list[str] = []
    for root in roots:
        collected.extend(collect_nodeids(root, pytest_args))
    selected = shard_nodeids(
        collected,
        total_shards,
        shard_index,
        load_durations(durations_file),
    )
    if not selected:
        print(f"shard {shard_index}/{total_shards} selected no tests")
        return 0

    active = [
        (root, nodeids) for root, nodeids in group_by_root(selected, roots) if nodeids
    ]
    isolated = len(active) > 1
    junit_path = extract_long_option(pytest_args, "--junitxml", "--junit-xml")
    junit_parts: list[Path] = []
    cov = coverage_enabled(pytest_args)
    worst = 0
    for session_index, (_root, nodeids) in enumerate(active):
        session_args = list(pytest_args)
        if session_index > 0:
            if cov:
                combine_coverage()
            session_args = with_cov_append(session_args)
        if junit_path is not None and isolated:
            dest = Path(junit_path)
            part = dest.with_name(f"{dest.stem}.root{session_index}{dest.suffix}")
            session_args = without_option(session_args, "--junitxml", "--junit-xml")
            session_args.append(f"--junitxml={part.as_posix()}")
            junit_parts.append(part)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            handle.write("\n".join(nodeids))
            handle.write("\n")
            nodeid_file = Path(handle.name)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "exec",
                "--",
                *session_args,
            ],
            check=False,
            env={**os.environ, "PYTEST_SHARD_NODEIDS": str(nodeid_file)},
        )
        nodeid_file.unlink(missing_ok=True)
        worst = max(worst, completed.returncode)
    if junit_path is not None and junit_parts:
        merge_junit_xml(junit_parts, Path(junit_path))
        for part in junit_parts:
            part.unlink(missing_ok=True)
    if cov:
        combine_coverage()
    return worst


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="collect, shard, and run pytest")
    run.add_argument("--root", action="append", type=Path, dest="roots")
    run.add_argument("--total-shards", type=int, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--durations-file", type=Path)
    run.add_argument("pytest_args", nargs=argparse.REMAINDER)

    merge = sub.add_parser("merge-junit", help="write {node id: seconds} JSON")
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("reports", nargs="+", type=Path)

    execute = sub.add_parser("exec", help=argparse.SUPPRESS)
    execute.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "merge-junit":
        payload = collect_durations(args.reports)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=0, sort_keys=True) + "\n")
        return 0
    if args.command == "exec":
        nodeid_file = os.environ.get("PYTEST_SHARD_NODEIDS")
        if not nodeid_file:
            print("PYTEST_SHARD_NODEIDS is required", file=sys.stderr)
            return 2
        pytest_args = list(args.pytest_args)
        if pytest_args[:1] == ["--"]:
            pytest_args = pytest_args[1:]
        return exec_nodeids(pytest_args, Path(nodeid_file))

    if args.total_shards < 1 or not 1 <= args.shard_index <= args.total_shards:
        print(
            f"shard-index must be in 1..{args.total_shards}",
            file=sys.stderr,
        )
        return 2
    pytest_args = list(args.pytest_args)
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    roots = args.roots or [Path(root) for root in DEFAULT_ROOTS]
    return run_shard(
        roots,
        args.total_shards,
        args.shard_index,
        args.durations_file,
        pytest_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
