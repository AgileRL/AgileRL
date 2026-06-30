"""Tests for the pure / lightly-dependent helpers in ``benchmark.py``.

These cover argument parsing, manifest helpers, the device resolver (torch
availability mocked), the log-capture machinery (``_Tee`` / ``_LogFileWriter`` /
``tee_to_file``), the W&B history-completeness predicate (fake run objects), and
the small filesystem helpers — no training, network, GPU, or W&B.
"""

from __future__ import annotations

import argparse
import io
import logging
from types import SimpleNamespace

import pandas as pd
import pytest

import benchmark


# --------------------------------------------------------------------------- #
# Trivial helpers                                                              #
# --------------------------------------------------------------------------- #
class TestTrivialHelpers:
    def test_elite_filename(self):
        assert benchmark.elite_filename("PPO") == "elite_PPO.pt"
        assert benchmark.elite_filename("DQN") == "elite_DQN.pt"

    def test_mutable_hp_names(self):
        manifest = {"mutation": {"rl_hp_selection": {"lr": {}, "batch_size": {}}}}
        assert benchmark.mutable_hp_names(manifest) == ["lr", "batch_size"]

    def test_mutable_hp_names_missing_blocks(self):
        assert benchmark.mutable_hp_names({}) == []
        assert benchmark.mutable_hp_names({"mutation": None}) == []
        assert benchmark.mutable_hp_names({"mutation": {}}) == []

    @pytest.mark.parametrize(
        "algo,expected",
        [("IPPO", True), ("ippo", True), ("PPO", False), ("DQN", False)],
    )
    def test_is_multi_agent_algo(self, algo, expected):
        assert benchmark.is_multi_agent_algo(algo) is expected

    def test_ale_render_id(self):
        assert benchmark._ale_render_id("Pong-v5") == "ALE/Pong-v5"
        assert benchmark._ale_render_id("ALE/Pong-v5") == "ALE/Pong-v5"


# --------------------------------------------------------------------------- #
# parse_args                                                                   #
# --------------------------------------------------------------------------- #
class TestParseArgs:
    def test_full(self):
        args = benchmark.parse_args(
            [
                "--project",
                "p",
                "--name",
                "n",
                "--config",
                "c.yaml",
                "--seeds",
                "1,2",
                "--device",
                "cpu",
                "--envs",
                "all",
                "--max-steps",
                "100",
                "--move-replay-buffer-to-cpu",
            ]
        )
        assert args.project == "p"
        assert args.seeds == "1,2"
        assert args.max_steps == 100
        assert args.move_replay_buffer_to_cpu is True

    def test_defaults(self):
        args = benchmark.parse_args([])
        assert args.seeds is None
        assert args.seed is None
        assert args.max_steps is None
        assert args.wandb_api_key is None
        assert args.move_replay_buffer_to_cpu is False


# --------------------------------------------------------------------------- #
# _prompt / _parse_seeds                                                       #
# --------------------------------------------------------------------------- #
class TestPromptAndSeeds:
    def test_prompt_passthrough(self):
        assert benchmark._prompt("x", "msg") == "x"

    def test_prompt_zero_is_truthy_string(self):
        assert benchmark._prompt(0, "msg") == "0"

    def test_prompt_falls_back_to_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "  typed  ")
        assert benchmark._prompt(None, "msg") == "typed"
        assert benchmark._prompt("", "msg") == "typed"

    def test_parse_seeds_from_seeds_string(self):
        ns = argparse.Namespace(seeds="1, 2 3", seed=None)
        assert benchmark._parse_seeds(ns) == [1, 2, 3]

    def test_parse_seeds_from_single_seed(self):
        ns = argparse.Namespace(seeds=None, seed=7)
        assert benchmark._parse_seeds(ns) == [7]

    def test_parse_seeds_prompt(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "4 5")
        ns = argparse.Namespace(seeds=None, seed=None)
        assert benchmark._parse_seeds(ns) == [4, 5]


# --------------------------------------------------------------------------- #
# resolve_device                                                              #
# --------------------------------------------------------------------------- #
class TestResolveDevice:
    def test_cpu(self):
        assert benchmark.resolve_device("CPU ") == "cpu"

    def test_unknown_falls_back(self):
        assert benchmark.resolve_device("tpu") == "cpu"

    def test_cuda_available_preserves_index(self, monkeypatch):
        monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
        assert benchmark.resolve_device("cuda") == "cuda"
        assert benchmark.resolve_device("cuda:1") == "cuda:1"

    def test_cuda_unavailable_falls_back(self, monkeypatch):
        monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
        assert benchmark.resolve_device("cuda") == "cpu"

    def test_mps_available(self, monkeypatch):
        monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)
        assert benchmark.resolve_device("mps") == "mps"

    def test_mps_unavailable_falls_back(self, monkeypatch):
        monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)
        assert benchmark.resolve_device("mps") == "cpu"


# --------------------------------------------------------------------------- #
# load_manifest_dict                                                          #
# --------------------------------------------------------------------------- #
class TestFileHelpers:
    def test_load_manifest_dict(self, tmp_path):
        import yaml

        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump({"algorithm": {"name": "PPO"}, "x": [1, 2]}))
        out = benchmark.load_manifest_dict(p)
        assert out["algorithm"]["name"] == "PPO"
        assert out["x"] == [1, 2]


# --------------------------------------------------------------------------- #
# _Tee / _LogFileWriter / tee_to_file                                         #
# --------------------------------------------------------------------------- #
class TestLogCapture:
    def test_tee_writes_all_streams_and_returns_len(self):
        a, b = io.StringIO(), io.StringIO()
        n = benchmark._Tee(a, b).write("hi")
        assert a.getvalue() == "hi"
        assert b.getvalue() == "hi"
        assert n == 2

    def test_logfilewriter_resolves_carriage_return(self):
        buf = io.StringIO()
        w = benchmark._LogFileWriter(buf)
        w.write("foo\rbar\n")
        assert buf.getvalue() == "bar\n"

    def test_logfilewriter_strips_ansi_and_backspace(self):
        buf = io.StringIO()
        w = benchmark._LogFileWriter(buf)
        w.write("\x1b[31mred\x1b[0m\b\n")
        assert buf.getvalue() == "red\n"

    def test_logfilewriter_close_flushes_partial_line(self):
        buf = io.StringIO()
        w = benchmark._LogFileWriter(buf)
        w.write("tail")
        assert buf.getvalue() == ""
        w.close()
        assert buf.getvalue() == "tail\n"

    def test_logfilewriter_returns_original_length(self):
        buf = io.StringIO()
        w = benchmark._LogFileWriter(buf)
        data = "\x1b[31mx\x1b[0m"
        assert w.write(data) == len(data)

    def test_tee_to_file(self, tmp_path):
        path = tmp_path / "sub" / "train.log"
        orig = __import__("sys").stdout
        with benchmark.tee_to_file(path):
            print("captured line")
        import sys

        assert sys.stdout is orig  # restored
        assert "captured line" in path.read_text()


# --------------------------------------------------------------------------- #
# _history_is_complete                                                        #
# --------------------------------------------------------------------------- #
class TestHistoryIsComplete:
    def _run(self, *, state="finished", summary=None, last=None):
        return SimpleNamespace(state=state, summary=summary, lastHistoryStep=last)

    def test_empty_df_false(self):
        run = self._run()
        assert benchmark._history_is_complete(run, pd.DataFrame()) is False

    def test_running_state_false(self):
        run = self._run(state="running", summary={"train/global_step": 3})
        df = pd.DataFrame({"train/global_step": [1, 2, 3]})
        assert benchmark._history_is_complete(run, df) is False

    def test_summary_target_reached(self):
        run = self._run(summary={"train/global_step": 3})
        df = pd.DataFrame({"train/global_step": [1, 2, 3]})
        assert benchmark._history_is_complete(run, df) is True

    def test_summary_target_not_reached(self):
        run = self._run(summary={"train/global_step": 5})
        df = pd.DataFrame({"train/global_step": [1, 2, 3]})
        assert benchmark._history_is_complete(run, df) is False

    def test_last_history_step_fallback(self):
        run = self._run(summary=None, last=3)
        df = pd.DataFrame({"_step": [0, 1, 2, 3]})
        assert benchmark._history_is_complete(run, df) is True

    def test_terminal_without_target_accepts(self):
        run = self._run(summary=None, last=None)
        df = pd.DataFrame({"other": [1]})
        assert benchmark._history_is_complete(run, df) is True


# --------------------------------------------------------------------------- #
# _move_replay_buffer_to_cpu                                                  #
# --------------------------------------------------------------------------- #
class TestMoveReplayBuffer:
    def test_moves_both_buffers(self):
        mem = SimpleNamespace(device="cuda")
        nmem = SimpleNamespace(device="cuda")
        trainer = SimpleNamespace(memory=mem, n_step_memory=nmem)
        benchmark._move_replay_buffer_to_cpu(trainer)
        assert mem.device == "cpu"
        assert nmem.device == "cpu"

    def test_noop_when_none(self):
        trainer = SimpleNamespace(memory=None, n_step_memory=None)
        benchmark._move_replay_buffer_to_cpu(trainer)  # must not raise


def test_logging_handler_removed_after_tee(tmp_path):
    # tee_to_file must remove its handler from the root logger on exit.
    before = list(logging.getLogger().handlers)
    with benchmark.tee_to_file(tmp_path / "t.log"):
        pass
    assert list(logging.getLogger().handlers) == before
