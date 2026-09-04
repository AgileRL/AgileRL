# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Sizing CLI smoke tests (CPU-only, no model downloads).

The exit codes are a contract with the submission gate, which runs this
before a submission and refuses the job on a non-zero status, so they are
asserted directly rather than inferred from the report text.
"""

import json
import sys
import types
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from agilerl.arena.memory.cli import (
    BAR_WIDTH,
    EXIT_OK,
    EXIT_OVER_BUDGET,
    EXIT_USAGE,
    checkpoint_param_count,
    main,
    memory_group,
    solve_main,
)
from agilerl.arena.memory.specs import ModelArch

TINY_CONFIG = str(Path(__file__).parent / "assets" / "tiny_llm" / "config.json")

MANIFEST = {
    "algorithm": {"name": "GRPO", "group_size": 4, "batch_size": 2},
    "environment": {
        "env_type": "rollout",
        "dataset": "openai/gsm8k",
        "reward_file_path": "reward.py",
        "prompt_template": {"user_0": "{question}"},
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_context_length": 512,
    },
    "training": {"max_steps": 100},
}

#: A manifest that will not fit a 1 GiB card, for exercising the blocked path.
OVERSIZE = {
    **MANIFEST,
    "algorithm": {
        "name": "GRPO",
        "group_size": 4,
        "batch_size": 2,
        "micro_batch_size_per_gpu": 32,
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_context_length": 8192,
    },
}


@pytest.fixture
def manifest_path(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(MANIFEST))
    return str(path)


@pytest.fixture
def oversize_path(tmp_path):
    path = tmp_path / "oversize.yaml"
    path.write_text(yaml.safe_dump(OVERSIZE))
    return str(path)


def _fake_hub(monkeypatch, get_safetensors_metadata):
    module = types.ModuleType("huggingface_hub")
    module.get_safetensors_metadata = get_safetensors_metadata
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def _metadata(total, tensor_names):
    return types.SimpleNamespace(
        parameter_count={"BF16": total},
        weight_map=dict.fromkeys(tensor_names, "model.safetensors"),
    )


TIED = ModelArch(
    n_layers=28,
    hidden_size=1024,
    intermediate_size=3072,
    n_heads=16,
    n_kv_heads=8,
    head_dim=128,
    vocab_size=151936,
    tied_embeddings=True,
)


def test_checkpoint_param_count_reads_safetensors_metadata(monkeypatch):
    _fake_hub(
        monkeypatch, lambda _m: _metadata(494_032_768, ["model.embed_tokens.weight"])
    )
    assert checkpoint_param_count("Qwen/Qwen2.5-0.5B-Instruct", TIED) == 494_032_768


def test_checkpoint_param_count_drops_a_tied_checkpoints_duplicate_lm_head(monkeypatch):
    # Qwen3-0.6B ships lm_head.weight *and* embed_tokens.weight at
    # tie_word_embeddings=true. from_pretrained re-ties them, so counting the
    # stored pair invents 311 MiB and blocks runs that fit.
    stored = ["model.embed_tokens.weight", "lm_head.weight"]
    _fake_hub(monkeypatch, lambda _m: _metadata(751_632_384, stored))
    resident = checkpoint_param_count("Qwen/Qwen3-0.6B", TIED)
    assert resident == 751_632_384 - TIED.vocab_size * TIED.hidden_size

    # An untied model genuinely has both, and param_counts already counts them.
    untied = TIED.model_copy(update={"tied_embeddings": False})
    assert checkpoint_param_count("Qwen/Qwen3-8B", untied) == 751_632_384


def test_checkpoint_param_count_falls_back_to_the_analytic_count(monkeypatch):
    # An exact count is an improvement, never a requirement: an offline box, a
    # gated repo or a repo that publishes no safetensors index must all still
    # produce an estimate.
    def boom(_model_id):
        msg = "no network"
        raise OSError(msg)

    _fake_hub(monkeypatch, boom)
    assert checkpoint_param_count("whatever", TIED) is None

    _fake_hub(monkeypatch, lambda _m: _metadata(0, []))
    assert checkpoint_param_count("whatever", TIED) is None


def test_report_with_local_config(manifest_path, capsys):
    code = main(manifest_path, config_path=TINY_CONFIG, device_gb=24)
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Training" in out
    assert "Generation" in out
    assert "colocated" in out


def test_gpu_name_resolves_the_device(manifest_path, capsys):
    code = main(manifest_path, config_path=TINY_CONFIG, gpu="NVIDIA L4")
    assert code == EXIT_OK
    assert "24 GiB card" in capsys.readouterr().out


def test_over_budget_blocks_and_prints_fixes(oversize_path, capsys):
    code = main(oversize_path, config_path=TINY_CONFIG, device_gb=1)
    out = capsys.readouterr().out
    assert code == EXIT_OVER_BUDGET
    assert "OVER BUDGET" in out
    assert "Cheapest fixes:" in out
    assert "--allow-oversize" in out


def test_allow_oversize_downgrades_the_gate(oversize_path, capsys):
    """The override returns 0 but must still show the shortfall, so a caller
    that submits anyway has it on the record rather than hidden.
    """
    code = main(
        oversize_path, config_path=TINY_CONFIG, device_gb=1, allow_oversize=True
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "SHORTFALL" in out
    assert "over budget, submitting anyway" in out


def test_json_carries_the_gate_decision(oversize_path, capsys):
    code = main(oversize_path, config_path=TINY_CONFIG, device_gb=1, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OVER_BUDGET
    assert payload["fits"] is False
    assert payload["blocked"] is True
    assert payload["training"]["components"]


def test_json_fits_case(manifest_path, capsys):
    code = main(manifest_path, config_path=TINY_CONFIG, device_gb=24, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["fits"] is True
    assert payload["blocked"] is False


def test_bar_never_exceeds_its_width(oversize_path, capsys):
    """An over-budget bar truncates at capacity rather than running off the
    line -- renormalising would make a 3x overshoot look like a perfect fit.
    """
    main(oversize_path, config_path=TINY_CONFIG, device_gb=1, no_color=True)
    bars = [
        line.split("|")[0][11:]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(("Training  ", "Generation"))
    ]
    assert bars, "no phase bars rendered"
    for bar in bars:
        assert len(bar) == BAR_WIDTH, f"bar was {len(bar)} cells, not {BAR_WIDTH}"


def test_missing_device_is_a_usage_error(manifest_path, capsys):
    assert main(manifest_path, config_path=TINY_CONFIG) == EXIT_USAGE


def test_a_classic_rl_manifest_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "dqn.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "DQN"},
                "environment": {"name": "LunarLander-v3", "num_envs": 8},
                "training": {"max_steps": 1000, "pop_size": 2},
            }
        )
    )
    assert main(str(path), config_path=TINY_CONFIG, device_gb=24) == EXIT_USAGE
    assert "does not need a gate" in capsys.readouterr().err


def test_invalid_manifest_is_a_usage_error(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({**MANIFEST, "algorithm": {"name": "GRPO"}}))
    assert main(str(path), config_path=TINY_CONFIG, device_gb=24) == EXIT_USAGE
    assert "could not validate" in capsys.readouterr().err


def test_solve_inference_max_model_len_on_l4(capsys):
    code = solve_main(
        "max_model_len",
        inference=True,
        gpu="NVIDIA L4",
        model_id="tiny",
        config_path=TINY_CONFIG,
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Solved max_model_len = 32768" in out
    assert "checkpoint / --hi cap" in out
    assert "Generation" in out
    assert "Training" not in out


def test_solve_inference_json(capsys):
    code = solve_main(
        "max_model_len",
        inference=True,
        gpu="NVIDIA L4",
        model_id="tiny",
        config_path=TINY_CONFIG,
        as_json=True,
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == EXIT_OK
    assert payload["value"] == 32768
    assert payload["limited_by"] == "bound"
    assert payload["mode"] == "inference"
    assert "training" not in payload
    assert payload["generation"]["components"]


def test_solve_inference_rejects_a_manifest(manifest_path, capsys):
    assert (
        solve_main(
            "max_model_len",
            manifest_path,
            inference=True,
            gpu="NVIDIA L4",
            config_path=TINY_CONFIG,
        )
        == EXIT_USAGE
    )
    assert "does not take a training manifest" in capsys.readouterr().err


def test_solve_without_inference_needs_a_manifest(capsys):
    assert solve_main("max_model_len", gpu="NVIDIA L4") == EXIT_USAGE
    assert "Pass a MANIFEST" in capsys.readouterr().err


def test_solve_from_manifest(manifest_path, capsys):
    code = solve_main(
        "max_model_len",
        manifest_path,
        gpu="NVIDIA L4",
        config_path=TINY_CONFIG,
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "Solved max_model_len" in out
    assert "Training" in out
    assert "Generation" in out


def test_solve_click_command_on_l4():
    runner = CliRunner()
    result = runner.invoke(
        memory_group,
        [
            "solve",
            "max_model_len",
            "--inference",
            "--gpu",
            "NVIDIA L4",
            "--model",
            "tiny",
            "--config",
            TINY_CONFIG,
            "--no-color",
        ],
    )
    assert result.exit_code == EXIT_OK
    assert "32768" in result.output


def test_click_command_exits_with_the_gate_code(oversize_path):
    runner = CliRunner()
    result = runner.invoke(
        memory_group,
        [
            "estimate",
            oversize_path,
            "--config",
            TINY_CONFIG,
            "--device-gb",
            "1",
            "--no-color",
        ],
    )
    assert result.exit_code == EXIT_OVER_BUDGET
