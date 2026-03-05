# noqa: INP001
"""Run local CodeQL and fail on findings.

This script is designed for local developer use (including pre-commit).
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "agilerl-codeql-db"
DEFAULT_CONFIG_PATH = REPO_ROOT / ".github" / "codeql" / "codeql-config.yml"
DEFAULT_CODEQL_INSTALL_DIR = Path("/tmp/codeql")  # noqa: S108
INSTALL_SCRIPT_PATH = Path(__file__).with_name("install_codeql.sh")
CSV_FILE_COL_IDX = 4
PYTHON_QUERY_SUITE = (
    "codeql/python-queries:codeql-suites/python-security-and-quality.qls"
)


def resolve_codeql_bin() -> str | None:
    """Resolve CodeQL binary path from env/PATH/common local install."""
    env_bin = os.environ.get("CODEQL_BIN")
    if env_bin:
        return env_bin
    path_bin = shutil.which("codeql")
    if path_bin:
        return path_bin
    install_dir = Path(
        os.environ.get("CODEQL_INSTALL_DIR", str(DEFAULT_CODEQL_INSTALL_DIR))
    )
    candidates = (
        [install_dir]
        if install_dir.is_file()
        else [
            install_dir / "codeql",
            install_dir / "codeql.exe",
        ]
    )
    for fallback in candidates:
        if fallback.exists():
            return str(fallback)
    return None


def ensure_codeql_installed() -> str | None:
    """Install CodeQL via bootstrap script when not present."""
    codeql = resolve_codeql_bin()
    if codeql is not None:
        return codeql
    if not INSTALL_SCRIPT_PATH.exists():
        return None

    bash_bin = shutil.which("bash")
    if bash_bin is None:
        return None

    subprocess.run(  # noqa: S603
        [bash_bin, str(INSTALL_SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=True,
    )
    return resolve_codeql_bin()


def run(cmd: list[str], cwd: Path) -> None:
    """Run subprocess command and propagate failure."""
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def format_clickable_path(path: str) -> str:
    """Normalize finding file paths for IDE click-through."""
    return path.lstrip("/\\")


def main() -> int:
    """Execute local CodeQL scan and return process exit code."""
    codeql = ensure_codeql_installed()
    if codeql is None:
        print(
            "CodeQL binary not found. Set CODEQL_BIN or install CodeQL "
            "(e.g. /tmp/codeql/codeql).",
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory(prefix="agilerl-codeql-") as td:
        output_csv = Path(td) / "results.csv"

        run(
            [
                codeql,
                "database",
                "create",
                str(DEFAULT_DB_PATH),
                "--overwrite",
                "--language=python",
                f"--source-root={REPO_ROOT}",
            ],
            cwd=REPO_ROOT,
        )
        run(
            [
                codeql,
                "database",
                "analyze",
                str(DEFAULT_DB_PATH),
                PYTHON_QUERY_SUITE,
                "--download",
                "--format=csv",
                f"--output={output_csv}",
            ],
            cwd=REPO_ROOT,
        )

        with output_csv.open(newline="") as f:
            rows = list(csv.reader(f))

    actionable = [r for r in rows if len(r) > CSV_FILE_COL_IDX]
    if actionable:
        print(f"CodeQL found {len(actionable)} actionable issue(s):", file=sys.stderr)
        for row in actionable[:20]:
            file_path = format_clickable_path(row[4])
            print(f"- {row[0]} at {file_path}:{row[5]}", file=sys.stderr)
        return 1

    print("CodeQL passed (no actionable issues).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
