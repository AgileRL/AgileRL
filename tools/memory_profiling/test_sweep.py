# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Sweep planner test — lives with the rig it exercises."""

import json


def test_sweep_dry_run_prints_plan(capsys):
    from tools.memory_profiling.sweep import HOLDOUT_POINTS
    from tools.memory_profiling.sweep import main as sweep_main

    code = sweep_main(["--model", "any/model", "--device-name", "any", "--dry-run"])
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert len(out_lines) == 16 + len(HOLDOUT_POINTS)  # corners + holdouts
    assert json.loads(out_lines[0])["seq_len"] == 512
    # Utilization is a real sweep axis: some holdouts sit at other budgets.
    utilizations = {json.loads(line)["gpu_memory_utilization"] for line in out_lines}
    assert len(utilizations) > 1
