#!/usr/bin/env bash
set -euo pipefail

arena_pkg_path="$PWD/agilerl-arena"

run_pytest() {
  PYTHONPATH="$arena_pkg_path${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python -m pytest "$@"
}

flags=("--import-mode=importlib")
pytest_args=()
cov_enabled=0
explicit_test_target=0

for arg in "$@"; do
  case "$arg" in
    no-parallel)
      flags+=("-o" "addopts=")
      ;;
    cov)
      # Use [tool.coverage.run] source from pyproject.toml (agilerl + agilerl-arena/agilerl).
      # --cov agilerl alone misses arena code in the namespace package split.
      flags+=("--cov")
      cov_enabled=1
      ;;
    tests | tests/* | agilerl-arena/tests | agilerl-arena/tests/*)
      explicit_test_target=1
      pytest_args+=("$arg")
      ;;
    *)
      pytest_args+=("$arg")
      ;;
  esac
done

if ((explicit_test_target)); then
  run_pytest "${pytest_args[@]}" "${flags[@]}"
  exit $?
fi

rc=0
if ((cov_enabled)); then
  # Collect core coverage quietly; print the merged report after arena tests.
  run_pytest tests "${pytest_args[@]}" "${flags[@]}" --cov-report= || rc=$?
  flags+=("--cov-append")
  has_cov_report=0
  for arg in "${pytest_args[@]}"; do
    case "$arg" in --cov-report*) has_cov_report=1 ;; esac
  done
  if (( ! has_cov_report )); then
    flags+=("--cov-report=term")
  fi
else
  run_pytest tests "${pytest_args[@]}" "${flags[@]}" || rc=$?
fi

run_pytest agilerl-arena/tests "${pytest_args[@]}" "${flags[@]}" || rc=$?
exit $rc
