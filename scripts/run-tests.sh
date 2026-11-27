#!/usr/bin/env bash
# Run pytest on tests/ and snowflake_app/tests/, or on the paths given.
# Usage: run-tests.sh [no-parallel] [cov] [junit] [pytest args or test paths...]
set -euo pipefail

# Empty arrays expand as ${a[@]+"${a[@]}"}: bash 3.2 (macOS) treats "${a[@]}" on one as unbound under set -u.

flags=("--import-mode=importlib")
pytest_args=()
targets=()

for arg in "$@"; do
  case "$arg" in
    no-parallel) flags+=("-o" "addopts=") ;;
    cov) flags+=("--cov=agilerl" "--cov-report=term-missing") ;;
    junit) flags+=("--junitxml=junit.xml") ;;
    tests | tests/* | snowflake_app/tests | snowflake_app/tests/*) targets+=("$arg") ;;
    *) pytest_args+=("$arg") ;;
  esac
done

if ((${#targets[@]} == 0)); then
  targets=(tests snowflake_app/tests)
fi

# Native App tests import snowflake_app from this directory.
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  uv run python -m pytest "${targets[@]}" ${pytest_args[@]+"${pytest_args[@]}"} "${flags[@]}"
