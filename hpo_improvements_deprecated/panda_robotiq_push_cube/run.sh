#!/usr/bin/env bash
# Train PPO on EnvPool MuJoCo Playground PandaRobotiqPushCube.
#
# Usage:
#   bash hpo_improvements/panda_robotiq_push_cube/run.sh
#   bash hpo_improvements/panda_robotiq_push_cube/run.sh --device cpu --wb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

exec uv run python hpo_improvements/panda_robotiq_push_cube/train.py "$@"
