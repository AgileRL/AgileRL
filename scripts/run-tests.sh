#!/usr/bin/env bash
set -euo pipefail

# How agilerl.arena becomes importable in each context:
#   - Wheel install:  files merge physically into site-packages; no extra config.
#   - Editable dev:   uv's editable workspace install + extend_path() in
#                     agilerl/__init__.py cooperate via a sys.meta_path hook.
#   - Test runner:    PYTHONPATH below is the belt-and-suspenders fallback so
#                     pytest --import-mode=importlib finds arena even if the
#                     editable hook hasn't fired yet (e.g. subprocess workers).
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
junit_enabled=0
explicit_test_target=0
cov_reports=()

for arg in "$@"; do
  case "$arg" in
    no-parallel)
      flags+=("-o" "addopts=")
      ;;
    cov)
      # Use [tool.coverage.run] source from pyproject.toml (agilerl +
      # agilerl-arena/agilerl).
      # --cov agilerl alone misses arena code in the namespace package split.
      flags+=("--cov")
      cov_enabled=1
      ;;
    junit)
      junit_enabled=1
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

junit_for() {
  junit_flags=()
  if ((junit_enabled == 0)); then
    return
  fi
  case "$1" in
    tests) junit_flags=(--junitxml=junit-tests.xml) ;;
    agilerl-arena/tests) junit_flags=(--junitxml=junit-arena.xml) ;;
    *) junit_flags=(--junitxml=junit.xml) ;;
  esac
}

if ((explicit_test_target)); then
  junit_for other
  if ((cov_enabled)); then
    run_pytest "${pytest_args[@]}" "${flags[@]}" --cov-report= "${junit_flags[@]}"
    combine_coverage
    emit_coverage_reports "${cov_reports[@]}"
  else
    run_pytest "${pytest_args[@]}" "${flags[@]}" "${junit_flags[@]}"
  fi
  exit $?
fi

rc=0
if ((cov_enabled)); then
  junit_for tests
  run_pytest tests "${pytest_args[@]}" "${flags[@]}" --cov-report= "${junit_flags[@]}" || rc=$?
  combine_coverage
  flags+=("--cov-append")
  junit_for agilerl-arena/tests
  run_pytest agilerl-arena/tests "${pytest_args[@]}" "${flags[@]}" --cov-report= "${junit_flags[@]}" || rc=$?
  combine_coverage
  emit_coverage_reports "${cov_reports[@]}"
else
  junit_for tests
  run_pytest tests "${pytest_args[@]}" "${flags[@]}" "${junit_flags[@]}" || rc=$?
  junit_for agilerl-arena/tests
  run_pytest agilerl-arena/tests "${pytest_args[@]}" "${flags[@]}" "${junit_flags[@]}" || rc=$?
fi
exit $rc
