#!/usr/bin/env bash
# Train PPO on Gymnasium Walker2d with AgileRL and Weights & Biases.
#
# Prerequisites:
#   uv pip install 'gymnasium[mujoco]' wandb
#
# Usage:
#   bash hpo_improvements/walker2d/train_walker2d.sh
#   bash hpo_improvements/walker2d/train_walker2d.sh --no-wb --device cpu
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/ppo_walker2d_v5.yaml"
WANDB_PROJECT="${WANDB_PROJECT:-AgileRL}"

exec uv run python -m agilerl.train "${MANIFEST}" \
  --wb \
  --wandb-project "${WANDB_PROJECT}" \
  "$@"
