#!/usr/bin/env python3
"""Write, for each source file, the test files that executed it.

Reads a coverage database recorded with ``--cov-context=test``. The output is
the ``--map`` input of ``scripts/select_tests.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coverage import CoverageData


def test_file_from_context(context: str) -> str | None:
    """Return the test file of a pytest-cov context, or None outside a test.

    :param context: Context label, e.g. ``tests/test_a.py::test_b|run``.
    :return: Test file path, or None for the empty context.
    """
    nodeid = context.partition("|")[0]
    if "::" not in nodeid:
        return None
    return nodeid.partition("::")[0]


def build_map(data_file: Path) -> dict[str, list[str]]:
    """Map every measured source file to the test files that executed it.

    :param data_file: Combined ``.coverage`` database with test contexts.
    :return: ``{source path: sorted test files}``.
    """
    data = CoverageData(basename=str(data_file))
    data.read()
    sources: dict[str, list[str]] = {}
    for path in sorted(data.measured_files()):
        tests: set[str] = set()
        for contexts in data.contexts_by_lineno(path).values():
            for context in contexts:
                test_file = test_file_from_context(context)
                if test_file is not None:
                    tests.add(test_file)
        sources[path] = sorted(tests)
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    sources = build_map(args.data)
    args.out.write_text(json.dumps({"sources": sources}, sort_keys=True) + "\n")
    print(f"mapped {len(sources)} source files to tests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
