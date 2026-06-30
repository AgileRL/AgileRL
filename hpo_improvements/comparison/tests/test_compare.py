"""Hermetic tests for ``comparison/compare.py``.

The conftest has already imported ``compare`` with its module-global ``plotting``
bound to the *comparison* plotting module, so ``compare.plotting`` exposes
``plot_iqm_difference`` etc. that we monkeypatch to no-ops.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import compare


# --------------------------------------------------------------------------- #
# parse_args                                                                   #
# --------------------------------------------------------------------------- #
def test_parse_args_defaults_all_none():
    args = compare.parse_args([])
    assert args.studied is None
    assert args.baseline is None
    assert args.reps is None
    assert args.out_name is None


def test_parse_args_values():
    args = compare.parse_args(
        ["--studied", "S", "--baseline", "B", "--reps", "37", "--out-name", "OUT"]
    )
    assert args.studied == "S"
    assert args.baseline == "B"
    assert args.reps == 37
    assert isinstance(args.reps, int)
    assert args.out_name == "OUT"


# --------------------------------------------------------------------------- #
# save_results                                                                 #
# --------------------------------------------------------------------------- #
def _noop_plots(monkeypatch):
    monkeypatch.setattr(compare.plotting, "plot_iqm_difference", lambda *a, **k: None)
    monkeypatch.setattr(
        compare.plotting, "plot_probability_of_improvement", lambda *a, **k: None
    )
    monkeypatch.setattr(
        compare.plotting, "plot_aggregate_and_profile", lambda *a, **k: None
    )


def test_save_results_writes_json_and_csv(
    make_comparison_result, tmp_path, monkeypatch
):
    _noop_plots(monkeypatch)
    result = make_comparison_result()  # x has size 3 by default
    out_dir = tmp_path / "out"
    compare.save_results(result, out_dir)

    json_path = out_dir / "probability_of_improvement.json"
    assert json_path.is_file()
    summary = json.loads(json_path.read_text())
    assert set(summary.keys()) == {
        "algorithm",
        "studied",
        "baseline",
        "common_environments",
        "common_seeds",
        "per_environment_seeds",
        "n_pairs",
        "interpolated_grid",
        "reps",
        "confidence_level",
        "probability_of_improvement",
        "probability_of_improvement_ci",
    }
    assert summary["algorithm"] == "PPO"
    assert summary["probability_of_improvement"] == pytest.approx(0.7)
    assert summary["probability_of_improvement_ci"] == [0.6, 0.8]

    final_csv = out_dir / "final_values.csv"
    assert final_csv.is_file()
    lines = final_csv.read_text().strip().splitlines()
    assert lines[0] == "env,seed,studied_final,baseline_final,difference"
    # one row per pair (default fixture has 2 pairs).
    assert len(lines) == 1 + result.n_pairs
    # first data row corresponds to pairs[0] = (Ant-v4, 1).
    assert lines[1].startswith("Ant-v4,1,")


def test_save_results_writes_iqm_csv_when_x_nonempty(
    make_comparison_result, tmp_path, monkeypatch
):
    _noop_plots(monkeypatch)
    result = make_comparison_result()  # default x has size 3
    out_dir = tmp_path / "out"
    compare.save_results(result, out_dir)
    iqm_csv = out_dir / "iqm_difference.csv"
    assert iqm_csv.is_file()
    lines = iqm_csv.read_text().strip().splitlines()
    assert lines[0] == "x,iqm_difference,ci_low,ci_high"
    assert len(lines) == 1 + result.x.size


def test_save_results_no_iqm_csv_when_x_empty(
    make_comparison_result, tmp_path, monkeypatch
):
    _noop_plots(monkeypatch)
    result = make_comparison_result(
        x=np.array([]),
        diff_iqm=np.array([]),
        diff_ci_low=np.array([]),
        diff_ci_high=np.array([]),
    )
    out_dir = tmp_path / "out"
    compare.save_results(result, out_dir)
    assert not (out_dir / "iqm_difference.csv").exists()
    # JSON and final_values still written.
    assert (out_dir / "probability_of_improvement.json").is_file()
    assert (out_dir / "final_values.csv").is_file()


def test_save_results_calls_all_three_plots(
    make_comparison_result, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        compare.plotting,
        "plot_iqm_difference",
        lambda *a, **k: calls.append("iqm"),
    )
    monkeypatch.setattr(
        compare.plotting,
        "plot_probability_of_improvement",
        lambda *a, **k: calls.append("prob"),
    )
    monkeypatch.setattr(
        compare.plotting,
        "plot_aggregate_and_profile",
        lambda *a, **k: calls.append("agg"),
    )
    compare.save_results(make_comparison_result(), tmp_path / "out")
    assert set(calls) == {"iqm", "prob", "agg"}


# --------------------------------------------------------------------------- #
# _prompt_benchmark                                                            #
# --------------------------------------------------------------------------- #
def test_prompt_benchmark_nonempty_bypasses_input(tmp_path, monkeypatch):
    bench_root = tmp_path / "bench_results"
    bench_root.mkdir()
    folder = bench_root / "studied_bench"
    folder.mkdir()
    monkeypatch.setattr(compare, "BENCH_RESULTS_DIR", bench_root)

    def _no_input(*a, **k):  # input must NOT be called
        raise AssertionError("input() should not be called for a non-empty value")

    monkeypatch.setattr("builtins.input", _no_input)
    resolved = compare._prompt_benchmark("studied_bench", "Studied")
    assert resolved == folder


def test_prompt_benchmark_explicit_path(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "BENCH_RESULTS_DIR", tmp_path / "unused")
    folder = tmp_path / "explicit"
    folder.mkdir()
    resolved = compare._prompt_benchmark(str(folder), "Baseline")
    assert resolved == folder


# --------------------------------------------------------------------------- #
# main: no-prompt path                                                         #
# --------------------------------------------------------------------------- #
def test_main_no_prompt_path(
    make_benchmark_dir, make_comparison_result, tmp_path, monkeypatch
):
    bench_root = tmp_path / "bench_results"
    bench_root.mkdir()
    make_benchmark_dir(
        bench_root / "studied",
        algo="PPO",
        name="studied",
        runs={("Ant-v4", 1): [1.0, 2.0]},
        layout="multi",
    )
    make_benchmark_dir(
        bench_root / "baseline",
        algo="PPO",
        name="baseline",
        runs={("Ant-v4", 1): [1.0, 2.0]},
        layout="multi",
    )
    monkeypatch.setattr(compare, "BENCH_RESULTS_DIR", bench_root)
    monkeypatch.setattr(compare, "RESULTS_DIR", tmp_path / "comparison_out")

    sentinel_result = make_comparison_result()
    captured = {}

    def fake_compare_benchmarks(studied, baseline, **kwargs):
        captured["studied"] = studied
        captured["baseline"] = baseline
        captured["reps"] = kwargs.get("reps")
        return sentinel_result

    def fake_save_results(result, out_dir):
        captured["result"] = result
        captured["out_dir"] = out_dir

    monkeypatch.setattr(compare, "compare_benchmarks", fake_compare_benchmarks)
    monkeypatch.setattr(compare, "save_results", fake_save_results)

    def _no_input(*a, **k):
        raise AssertionError("input() should not be called when args are provided")

    monkeypatch.setattr("builtins.input", _no_input)

    compare.main(
        ["--studied", "studied", "--baseline", "baseline", "--out-name", "myout"]
    )

    # studied/baseline dirs resolved and loaded.
    assert captured["studied"].name == "studied"
    assert captured["baseline"].name == "baseline"
    assert captured["studied"].algo == "PPO"
    # save_results was invoked with the comparison result and the out dir.
    assert captured["result"] is sentinel_result
    assert captured["out_dir"] == (tmp_path / "comparison_out" / "myout")


def test_main_reps_forwarded(
    make_benchmark_dir, make_comparison_result, tmp_path, monkeypatch
):
    bench_root = tmp_path / "bench_results"
    bench_root.mkdir()
    make_benchmark_dir(
        bench_root / "studied",
        algo="PPO",
        name="studied",
        runs={("Ant-v4", 1): [1.0, 2.0]},
        layout="multi",
    )
    make_benchmark_dir(
        bench_root / "baseline",
        algo="PPO",
        name="baseline",
        runs={("Ant-v4", 1): [1.0, 2.0]},
        layout="multi",
    )
    monkeypatch.setattr(compare, "BENCH_RESULTS_DIR", bench_root)
    monkeypatch.setattr(compare, "RESULTS_DIR", tmp_path / "comparison_out")

    sentinel_result = make_comparison_result()
    captured = {}

    def fake_compare_benchmarks(s, b, **kw):
        captured["reps"] = kw.get("reps")
        return sentinel_result

    monkeypatch.setattr(compare, "compare_benchmarks", fake_compare_benchmarks)
    monkeypatch.setattr(compare, "save_results", lambda r, o: None)

    compare.main(
        [
            "--studied",
            "studied",
            "--baseline",
            "baseline",
            "--out-name",
            "myout",
            "--reps",
            "55",
        ]
    )
    assert captured["reps"] == 55
