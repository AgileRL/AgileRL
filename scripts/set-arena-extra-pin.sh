#!/usr/bin/env bash
# Rewrite the committed agilerl-arena extra to an exact pin for one wheel build.
#
# Used only when publish-framework-dev.sh builds a framework +local wheel that
# must Requires-Dist the matching arena +local from the same pipeline. Stables
# keep a compatible range in pyproject; the ML bundle records exact versions.
#
# Usage:
#   set-arena-extra-pin.sh apply <pep440>
#   set-arena-extra-pin.sh restore
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="${AGILERL_PYPROJECT:-${REPO_ROOT}/pyproject.toml}"
BACKUP="${PYPROJECT}.arena-pin.bak"

usage() {
  echo "usage: $0 apply <pep440> | restore" >&2
  exit 2
}

apply_pin() {
  local pin="${1:?pin required}"
  if [[ ! "${pin}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([+][A-Za-z0-9._-]+)?$ ]]; then
    echo "error: pin ${pin} is not a PEP 440 X.Y.Z or X.Y.Z+local" >&2
    exit 1
  fi
  if [[ ! -f "${PYPROJECT}" ]]; then
    echo "error: ${PYPROJECT} not found" >&2
    exit 1
  fi
  if [[ ! -f "${BACKUP}" ]]; then
    cp "${PYPROJECT}" "${BACKUP}"
  fi
  python3 - "${PYPROJECT}" "${pin}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
pin = sys.argv[2]
text = path.read_text(encoding="utf-8")
new, n = re.subn(
    r'^(\s*"agilerl-arena)(?:==[^"]+|>=[^"]+)(")',
    rf'\1=={pin}\2',
    text,
    count=1,
    flags=re.MULTILINE,
)
if n != 1:
    sys.exit(f"error: expected one agilerl-arena extra in {path}, replaced {n}")
path.write_text(new, encoding="utf-8")
print(f"pinned agilerl extra to agilerl-arena=={pin}")
PY
}

restore_pin() {
  if [[ -f "${BACKUP}" ]]; then
    mv "${BACKUP}" "${PYPROJECT}"
    echo "restored ${PYPROJECT}"
  fi
}

cmd="${1:-}"
case "${cmd}" in
  apply)
    apply_pin "${2:-}"
    ;;
  restore)
    restore_pin
    ;;
  *)
    usage
    ;;
esac
