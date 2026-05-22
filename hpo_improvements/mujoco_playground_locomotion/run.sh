#!/usr/bin/env bash
# Train PPO on MuJoCo Playground WalkerWalk with AgileRL evolutionary HPO.
#
# Uses the batched jax.vmap wrapper for fast parallel env stepping.
#
# Prerequisites:
#   pip install playground   # MuJoCo Playground (installs mujoco_playground + JAX deps)
#
# Usage:
#   bash hpo_improvements/mujoco_playground_locomotion/run.sh          # W&B enabled
#   bash hpo_improvements/mujoco_playground_locomotion/run.sh --no-wb  # W&B disabled
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

exec python -m hpo_improvements.mujoco_playground_locomotion.train "${SCRIPT_DIR}/ppo_walker_walk.yaml" "$@"
