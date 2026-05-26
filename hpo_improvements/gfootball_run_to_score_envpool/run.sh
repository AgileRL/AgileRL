#!/usr/bin/env bash
set -euo pipefail

device=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "--device" || "$prev" == "-d" ]]; then
        device="$arg"
    fi
    prev="$arg"
done

if [[ "$device" == "mps" && -z "${PYTORCH_ENABLE_MPS_FALLBACK:-}" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    echo "Enabled PYTORCH_ENABLE_MPS_FALLBACK=1 for MPS missing-op fallback."
fi

uv run python benchmarking/gfootball_run_to_score_envpool/train.py "$@"
