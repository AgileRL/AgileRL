#!/usr/bin/env bash
set -euo pipefail

arena_pkg_path="$PWD/agilerl-arena"

run_pytest() {
  PYTHONPATH="$arena_pkg_path${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python -m pytest "$@"
}

combine_coverage() {
  # [tool.coverage.run] parallel = true writes per-worker .coverage.* files;
  # combine before --cov-append and before the final report (needed with xdist).
  uv run coverage combine >/dev/null 2>&1 || true
}

emit_coverage_reports() {
  local reports=("$@")
  if ((${#reports[@]} == 0)); then
    uv run coverage report
    return
  fi

  for report in "${reports[@]}"; do
    case "$report" in
      --cov-report=xml) uv run coverage xml ;;
      --cov-report=term-missing) uv run coverage report -m ;;
      --cov-report=term) uv run coverage report ;;
      --cov-report=) ;;
      *) uv run coverage report ;;
    esac
  done
}

flags=("--import-mode=importlib")
pytest_args=()
cov_enabled=0
explicit_test_target=0
cov_reports=()

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
    --cov-report*)
      cov_reports+=("$arg")
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
  if ((cov_enabled)); then
    run_pytest "${pytest_args[@]}" "${flags[@]}" --cov-report=
    combine_coverage
    emit_coverage_reports "${cov_reports[@]}"
  else
    run_pytest "${pytest_args[@]}" "${flags[@]}"
  fi
  exit $?
fi

rc=0
if ((cov_enabled)); then
  run_pytest tests "${pytest_args[@]}" "${flags[@]}" --cov-report= || rc=$?
  combine_coverage
  flags+=("--cov-append")
  run_pytest agilerl-arena/tests "${pytest_args[@]}" "${flags[@]}" --cov-report= || rc=$?
  combine_coverage
  emit_coverage_reports "${cov_reports[@]}"
else
  run_pytest tests "${pytest_args[@]}" "${flags[@]}" || rc=$?
  run_pytest agilerl-arena/tests "${pytest_args[@]}" "${flags[@]}" || rc=$?
fi
exit $rc
