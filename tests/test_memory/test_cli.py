# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Sizing CLI smoke tests (CPU-only, no model downloads).

The exit codes are a contract with agilerl-arena, which runs this before a
submission and refuses the job on a non-zero status, so they are asserted
directly rather than inferred from the report text.
"""

import json
from pathlib import Path

from agilerl.memory.cli import BAR_WIDTH, EXIT_OK, EXIT_OVER_BUDGET, main

TINY_CONFIG = str(Path(__file__).parent.parent / "assets" / "tiny_llm" / "config.json")
#: Knobs that will not fit a 1 GiB card, for exercising the blocked path.
OVERSIZE = [
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


def test_report_with_local_config(capsys):
    code = main(["--config", TINY_CONFIG, "--device-gb", "8", "--max-model-len", "512"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Training" in out
    assert "Generation" in out
    assert "colocated" in out


def test_over_budget_blocks_and_prints_fixes(capsys):
    code = main(OVERSIZE)
    out = capsys.readouterr().out
    assert code == EXIT_OVER_BUDGET
    assert "OVER BUDGET" in out
    assert "Cheapest fixes:" in out
    assert "--allow-oversize" in out


def test_allow_oversize_downgrades_the_gate(capsys):
    """The override returns 0 but must still show the shortfall, so a caller
    that submits anyway has it on the record rather than hidden.
    """
    assert main([*OVERSIZE, "--allow-oversize"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "SHORTFALL" in out
    assert "over budget, submitting anyway" in out


def test_json_carries_the_gate_decision(capsys):
    code = main([*OVERSIZE, "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OVER_BUDGET
    assert payload["fits"] is False
    assert payload["blocked"] is True
    assert payload["training"]["components"]


def test_json_fits_case(capsys):
    code = main(
        [
            "--config",
            TINY_CONFIG,
            "--device-gb",
            "8",
            "--max-model-len",
            "512",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["fits"] is True
    assert payload["blocked"] is False


def test_bar_never_exceeds_its_width(capsys):
    """An over-budget bar truncates at capacity rather than running off the
    line -- renormalising would make a 3x overshoot look like a perfect fit.
    """
    main([*OVERSIZE, "--no-color"])
    bars = [
        line.split("|")[0][11:]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(("Training  ", "Generation"))
    ]
    assert bars, "no phase bars rendered"
    for bar in bars:
        assert len(bar) == BAR_WIDTH, f"bar was {len(bar)} cells, not {BAR_WIDTH}"


def test_missing_model_and_config_is_a_usage_error(capsys):
    assert main([]) == 2
