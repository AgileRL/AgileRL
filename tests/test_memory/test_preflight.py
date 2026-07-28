# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Preflight CLI smoke tests (CPU-only, no model downloads)."""

import json
from pathlib import Path

from agilerl.memory.preflight import main

TINY_CONFIG = str(Path(__file__).parent.parent / "assets" / "tiny_llm" / "config.json")


def test_preflight_report_with_local_config(capsys):
    code = main(
        [
            "--config",
            TINY_CONFIG,
            "--device-gb",
            "8",
            "--max-model-len",
            "512",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Training device" in out
    assert "Generation device" in out
    assert "colocated" in out


def test_preflight_over_budget_prints_fixes(capsys):
    code = main(
        [
            "--config",
            TINY_CONFIG,
            "--device-gb",
            "1",
            "--max-model-len",
            "8192",
            "--micro-batch",
            "32",
            "--gpu-memory-utilization",
            "0.9",
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "OVER BUDGET" in out
    assert "Cheapest fixes:" in out


def test_preflight_json_output(capsys):
    code = main(["--config", TINY_CONFIG, "--device-gb", "8", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"training", "generation"}
    training_keys = [c["key"] for c in payload["training"]["components"]]
    assert "base_weights" in training_keys


def test_preflight_list_models_runs(capsys):
    assert main(["--list-models"]) == 0


def test_sweep_dry_run_prints_plan(capsys):
    from agilerl.memory.profiling.sweep import HOLDOUT_POINTS
    from agilerl.memory.profiling.sweep import main as sweep_main

    code = sweep_main(["--model", "any/model", "--device-name", "any", "--dry-run"])
    out_lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0
    assert len(out_lines) == 16 + len(HOLDOUT_POINTS)  # corners + holdouts
    assert json.loads(out_lines[0])["seq_len"] == 512
    # Utilization is a real sweep axis: some holdouts sit at other budgets.
    utilizations = {json.loads(line)["gpu_memory_utilization"] for line in out_lines}
    assert len(utilizations) > 1
