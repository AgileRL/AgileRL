#!/usr/bin/env bash
# Train PPO on Gymnasium Walker2d with AgileRL and Weights & Biases.
#
# Prerequisites:
#   uv pip install 'gymnasium[mujoco]' wandb
#
# Usage:
#   bash hpo-improvements/walker2d/train_walker2d.sh
#   bash hpo-improvements/walker2d/train_walker2d.sh --no-wb --device cpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec uv run python "${SCRIPT_DIR}/train_walker2d.py" "$@"
