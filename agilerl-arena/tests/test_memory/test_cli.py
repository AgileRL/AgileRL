# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Sizing CLI smoke tests (CPU-only, no model downloads).

Exit codes are a contract with the submission gate: non-zero refuses the job.
"""

import builtins
import io
import json
import logging
import runpy
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
    _load_model_config,
    checkpoint_param_count,
    main,
    memory_group,
)
from agilerl.arena.memory.specs import ModelArch
from agilerl.arena.models.manifest import TrainingManifest

TINY_CONFIG = str(Path(__file__).parent / "assets" / "tiny_llm" / "config.json")

MANIFEST = {
    "algorithm": {
        "name": "GRPO",
        "group_size": 4,
        "batch_size": 2,
        "micro_batch_size_per_gpu": 1,
        "vllm_config": {"gpu_memory_utilization": 0.85},
    },
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
    "training": {
        "max_steps": 100,
        "rollout_mode": "async",
        "rollout_engines_per_agent": 1,
    },
    "replay_buffer": {"kind": "llm"},
}

#: A manifest that will not fit a 1 GiB card, for exercising the blocked path.
OVERSIZE = {
    **MANIFEST,
    "algorithm": {
        "name": "GRPO",
        "group_size": 4,
        "batch_size": 2,
        "micro_batch_size_per_gpu": 1,
        "vllm_config": {"gpu_memory_utilization": 0.9},
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
    errors = types.ModuleType("huggingface_hub.errors")
    errors.NotASafetensorsRepoError = type("NotASafetensorsRepoError", (Exception,), {})
    errors.SafetensorsParsingError = type("SafetensorsParsingError", (Exception,), {})
    module.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return module


def _metadata(tensors: dict[str, int]):
    """Safetensors metadata for one file holding ``tensors`` (name -> parameter count)."""
    infos = {
        name: types.SimpleNamespace(parameter_count=count)
        for name, count in tensors.items()
    }
    return types.SimpleNamespace(
        parameter_count={"BF16": sum(tensors.values())},
        weight_map=dict.fromkeys(tensors, "model.safetensors"),
        files_metadata={"model.safetensors": types.SimpleNamespace(tensors=infos)},
    )


BODY = 596_049_920
EMBED = 151_936 * 1024


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


class TestCheckpointParamCount:
    def test_reads_safetensors_metadata(self, monkeypatch):
        _fake_hub(
            monkeypatch,
            lambda _m: _metadata(
                {"model.embed_tokens.weight": EMBED, "model.layers": BODY}
            ),
        )
        assert (
            checkpoint_param_count("Qwen/Qwen2.5-0.5B-Instruct", TIED) == EMBED + BODY
        )

    def test_drops_a_tied_checkpoints_duplicate_lm_head(self, monkeypatch):
        # Tied checkpoints store lm_head and embed_tokens; from_pretrained re-ties them, so one copy is resident.
        stored = {
            "model.embed_tokens.weight": EMBED,
            "model.layers": BODY,
            "lm_head.weight": EMBED,
        }
        _fake_hub(monkeypatch, lambda _m: _metadata(stored))
        assert checkpoint_param_count("Qwen/Qwen3-0.6B", TIED) == EMBED + BODY

    def test_keeps_both_heads_of_an_untied_checkpoint(self, monkeypatch):
        stored = {
            "model.embed_tokens.weight": EMBED,
            "model.layers": BODY,
            "lm_head.weight": EMBED,
        }
        _fake_hub(monkeypatch, lambda _m: _metadata(stored))
        # An untied model keeps both, and param_counts already counts them.
        untied = TIED.model_copy(update={"tied_embeddings": False})
        assert checkpoint_param_count("Qwen/Qwen3-8B", untied) == 2 * EMBED + BODY

    def test_skips_multi_token_prediction_heads(self, monkeypatch):
        # Arrange: Nemotron 3.5 ships an mtp.* head the trainer never loads.
        stored = {
            "model.embed_tokens.weight": EMBED,
            "model.layers": BODY,
            "mtp.layers.0": 1_335_000_000,
        }
        _fake_hub(monkeypatch, lambda _m: _metadata(stored))
        untied = TIED.model_copy(update={"tied_embeddings": False})

        # Act
        count = checkpoint_param_count("nvidia/Nemotron", untied)

        # Assert
        assert count == EMBED + BODY

    def test_falls_back_to_the_analytic_count_and_says_why(self, monkeypatch, caplog):
        # Offline, gated, or index-less repos still estimate without a safetensors count.
        def boom(_model_id):
            msg = "no network"
            raise OSError(msg)

        _fake_hub(monkeypatch, boom)

        with caplog.at_level(logging.WARNING):
            count = checkpoint_param_count("org/model", TIED)

        assert count is None
        assert "org/model" in caplog.text
        assert "no network" in caplog.text
        assert "analytic parameter count" in caplog.text

    def test_skips_a_repo_without_safetensors(self, monkeypatch):
        # Arrange
        hub = _fake_hub(monkeypatch, lambda _m: None)

        def no_index(_model_id):
            msg = "no safetensors index"
            raise hub.errors.NotASafetensorsRepoError(msg)

        hub.get_safetensors_metadata = no_index

        # Act
        count = checkpoint_param_count("org/pytorch-bin-only", TIED)

        # Assert
        assert count is None

    def test_skips_metadata_with_no_parameters(self, monkeypatch):
        _fake_hub(monkeypatch, lambda _m: _metadata({}))
        assert checkpoint_param_count("whatever", TIED) is None

    def test_without_huggingface_hub_there_is_no_count(self, monkeypatch):
        real_import = builtins.__import__

        def block_hub(name, *args, **kwargs):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                msg = "no hub"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_hub)

        assert checkpoint_param_count("org/model", TIED) is None


class TestLoadModelConfig:
    def test_fetch_keeps_the_raw_config_keys(self, tmp_path):
        # Arrange: NemotronHConfig.to_dict() drops num_hidden_layers, which it
        # derives from layers_block_type; the estimator needs the raw key.
        raw = {
            "model_type": "nemotron_h",
            "num_hidden_layers": 2,
            "layers_block_type": ["mamba", "attention"],
            "hidden_size": 64,
        }
        (tmp_path / "config.json").write_text(json.dumps(raw))

        # Act
        config = _load_model_config(str(tmp_path), config_path=None)

        # Assert
        assert config["num_hidden_layers"] == 2
        assert config["layers_block_type"] == ["mamba", "attention"]

    def test_missing_checkpoint_is_a_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="could not fetch the config"):
            _load_model_config(str(tmp_path / "absent"), config_path=None)

    def test_without_huggingface_hub_a_hub_id_needs_a_config_file(self, monkeypatch):
        real_import = builtins.__import__

        def block_hub(name, *args, **kwargs):
            if name == "huggingface_hub" or name.startswith("huggingface_hub."):
                msg = "no hub"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_hub)

        with pytest.raises(ValueError, match="huggingface_hub is not installed"):
            _load_model_config("Qwen/Qwen2.5-0.5B-Instruct", config_path=None)


class TestMain:
    def test_report_with_local_config(self, manifest_path, capsys):
        code = main(manifest_path, config_path=TINY_CONFIG, device_gb=24)
        out = capsys.readouterr().out
        assert code == EXIT_OK
        assert "Training" in out
        assert "Generation" in out
        assert "Memory estimate" in out

    def test_gpu_name_resolves_the_device(self, manifest_path, capsys):
        code = main(manifest_path, config_path=TINY_CONFIG, gpu="NVIDIA L4")
        assert code == EXIT_OK
        assert "24 GiB card" in capsys.readouterr().out

    def test_device_gb_overrides_the_named_gpu(self, manifest_path, capsys):
        code = main(
            manifest_path, config_path=TINY_CONFIG, gpu="NVIDIA L4", device_gb=20
        )
        assert code == EXIT_OK
        assert "20 GiB card" in capsys.readouterr().out

    def test_local_config_notes_the_reconciliation_skip(self, manifest_path, capsys):
        code = main(manifest_path, config_path=TINY_CONFIG, device_gb=24)
        assert code == EXIT_OK
        assert "skips the Hub parameter count" in capsys.readouterr().err

    def test_json_fits_case(self, manifest_path, capsys):
        code = main(manifest_path, config_path=TINY_CONFIG, device_gb=24, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert code == EXIT_OK
        assert payload["fits"] is True

    def test_over_budget_blocks_and_prints_fixes(self, oversize_path, capsys, caplog):
        with caplog.at_level(logging.INFO):
            code = main(oversize_path, config_path=TINY_CONFIG, device_gb=1)
        out = capsys.readouterr().out
        assert code == EXIT_OVER_BUDGET
        assert "OVER BUDGET" in out
        assert "Fixes by memory saved" in out
        assert "Blocked: apply a fix above" in caplog.text

    def test_json_carries_the_gate_decision(self, oversize_path, capsys):
        code = main(oversize_path, config_path=TINY_CONFIG, device_gb=1, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert code == EXIT_OVER_BUDGET
        assert payload["fits"] is False
        assert payload["training"]["components"]

    def test_json_carries_ranked_advice(self, oversize_path, capsys):
        code = main(oversize_path, config_path=TINY_CONFIG, device_gb=1, as_json=True)
        payload = json.loads(capsys.readouterr().out)
        assert code == EXIT_OVER_BUDGET

        advice = payload["advice"]
        assert advice
        savings = [a["saves_bytes"] for a in advice]
        assert savings == sorted(savings, reverse=True)
        assert all(set(a) >= {"phase", "action", "saves_bytes"} for a in advice)

    def test_bar_never_exceeds_its_width(self, oversize_path, capsys):
        """Over-budget bar truncates at capacity."""
        main(oversize_path, config_path=TINY_CONFIG, device_gb=1, no_color=True)
        lines = capsys.readouterr().out.splitlines()
        # Legacy Windows draws square borders; a non-UTF console draws ASCII.
        bars = []
        for index, line in enumerate(lines):
            if not line.startswith(("╭", "┌", "+")):
                continue
            if "Training" not in line and "Generation" not in line:
                continue
            body = lines[index + 1].removeprefix("│ ").removeprefix("| ")
            bars.append(body.split(" ")[0])
        assert len(bars) == 2
        for bar in bars:
            assert len(bar) == BAR_WIDTH, f"bar was {len(bar)} cells, not {BAR_WIDTH}"

    def test_missing_device_is_a_usage_error(self, manifest_path):
        assert main(manifest_path, config_path=TINY_CONFIG) == EXIT_USAGE

    def test_a_manifest_without_a_model_is_a_usage_error(
        self, manifest_path, monkeypatch, capsys
    ):
        manifest = TrainingManifest.model_validate(MANIFEST)
        algo = manifest.algorithm.model_copy(
            update={"pretrained_model_name_or_path": None}
        )
        broken = manifest.model_copy(update={"algorithm": algo})
        monkeypatch.setattr(
            "agilerl.arena.memory.cli.TrainingManifest.get_validated",
            lambda *_args, **_kwargs: broken,
        )

        code = main(manifest_path, config_path=TINY_CONFIG, device_gb=24)

        assert code == EXIT_USAGE
        assert "no pretrained model" in capsys.readouterr().err

    def test_a_config_load_failure_is_a_usage_error(
        self, manifest_path, monkeypatch, capsys
    ):
        def boom(*_args, **_kwargs):
            msg = "no config"
            raise ValueError(msg)

        monkeypatch.setattr("agilerl.arena.memory.cli._load_model_config", boom)

        code = main(manifest_path, device_gb=24)

        assert code == EXIT_USAGE
        assert "could not load the model config" in capsys.readouterr().err

    def test_colocated_rollout_is_a_usage_error(self, tmp_path, capsys):
        doc = {
            **MANIFEST,
            "training": {**MANIFEST["training"], "rollout_mode": "colocated"},
        }
        path = tmp_path / "colocated.yaml"
        path.write_text(yaml.safe_dump(doc))

        code = main(str(path), config_path=TINY_CONFIG, device_gb=24)

        assert code == EXIT_USAGE
        assert "async rollout only" in capsys.readouterr().err

    def test_python_m_invokes_the_memory_command(self, monkeypatch):
        monkeypatch.setattr("agilerl.arena.memory.cli.memory_group", lambda: None)

        runpy.run_module("agilerl.arena.memory", run_name="__main__")

    def test_unknown_gpu_is_a_usage_error(self, manifest_path, capsys):
        assert (
            main(manifest_path, config_path=TINY_CONFIG, gpu="not-a-gpu") == EXIT_USAGE
        )
        assert "unknown GPU" in capsys.readouterr().err

    def test_unknown_gen_gpu_is_a_usage_error(self, manifest_path, capsys):
        assert (
            main(
                manifest_path,
                config_path=TINY_CONFIG,
                gpu="NVIDIA L4",
                gen_gpu="not-a-gpu",
            )
            == EXIT_USAGE
        )
        err = capsys.readouterr().err
        assert "unknown GPU" in err
        assert "--gen-device-gb" in err

    def test_a_classic_rl_manifest_is_a_usage_error(self, tmp_path, capsys):
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
        assert "is not sized" in capsys.readouterr().err

    def test_invalid_manifest_is_a_usage_error(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump(
                {**MANIFEST, "algorithm": {"name": "GRPO", "not_a_field": True}}
            )
        )
        assert main(str(path), config_path=TINY_CONFIG, device_gb=24) == EXIT_USAGE
        assert "could not validate" in capsys.readouterr().err

    def test_malformed_yaml_is_a_usage_error(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text("{ invalid yaml: [")
        assert main(str(path), config_path=TINY_CONFIG, device_gb=24) == EXIT_USAGE
        assert "could not validate" in capsys.readouterr().err

    @pytest.mark.parametrize("capacity", [0, -4, float("inf"), float("nan")])
    def test_nonpositive_capacity_is_a_usage_error(
        self, manifest_path, capsys, capacity
    ):
        code = main(manifest_path, config_path=TINY_CONFIG, device_gb=capacity)
        assert code == EXIT_USAGE
        assert "--device-gb" in capsys.readouterr().err

    def test_malformed_model_config_is_a_usage_error(
        self, manifest_path, tmp_path, capsys
    ):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "qwen2"}))
        code = main(manifest_path, config_path=str(config_path), device_gb=24)
        assert code == EXIT_USAGE
        assert "missing" in capsys.readouterr().err


class TestReportColour:
    def test_plain_output_has_no_escape_codes_and_keeps_glyphs(
        self, manifest_path, capsys
    ):
        main(manifest_path, config_path=TINY_CONFIG, device_gb=24, no_color=True)

        out = capsys.readouterr().out
        assert "\x1b[" not in out
        assert "█" not in out
        assert "#  Base weights (frozen)" in out

    def test_ascii_only_stdout_prints_an_ascii_report(self, oversize_path, monkeypatch):
        # Arrange
        raw = io.BytesIO()
        monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="ascii"))

        # Act
        code = main(oversize_path, config_path=TINY_CONFIG, device_gb=1, no_color=True)
        sys.stdout.flush()

        # Assert
        out = raw.getvalue().decode("ascii")
        assert code == EXIT_OVER_BUDGET
        assert "+- Training" in out
        assert "GiB usable | 1 GiB card | SHORTFALL" in out

    def test_forced_colour_draws_solid_coloured_blocks(
        self, manifest_path, capsys, monkeypatch
    ):
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("FORCE_COLOR", "1")
        monkeypatch.setenv("TERM", "xterm-256color")

        main(manifest_path, config_path=TINY_CONFIG, device_gb=24)

        out = capsys.readouterr().out
        assert "\x1b[" in out
        assert "█" in out
        assert "FITS" in out


class TestMemoryEstimateCommand:
    def test_exits_with_the_gate_code(self, oversize_path):
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
