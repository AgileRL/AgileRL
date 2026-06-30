"""Hermetic tests for ``comparison/loading.py``."""

from __future__ import annotations

import numpy as np
import pytest

from loading import (
    BenchmarkResults,
    list_available,
    resolve_results_dir,
)


# --------------------------------------------------------------------------- #
# resolve_results_dir                                                          #
# --------------------------------------------------------------------------- #
def test_resolve_results_dir_explicit_path_passthrough(tmp_path):
    folder = tmp_path / "some_bench"
    folder.mkdir()
    resolved = resolve_results_dir(str(folder), tmp_path / "unused_root")
    assert resolved == folder


def test_resolve_results_dir_name_under_root(tmp_path):
    root = tmp_path / "results"
    bench = root / "my_bench"
    bench.mkdir(parents=True)
    resolved = resolve_results_dir("my_bench", root)
    assert resolved == bench


def test_resolve_results_dir_tilde_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Build ~/expanded_bench as a real dir so the expanded path is a dir.
    folder = tmp_path / "expanded_bench"
    folder.mkdir()
    resolved = resolve_results_dir("~/expanded_bench", tmp_path / "unused_root")
    assert resolved == folder


def test_resolve_results_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_results_dir("does_not_exist", tmp_path)


# --------------------------------------------------------------------------- #
# list_available                                                              #
# --------------------------------------------------------------------------- #
def test_list_available_only_config_dirs(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    # Valid benchmarks (have config.yaml).
    for name in ("b_bench", "a_bench"):
        d = root / name
        d.mkdir()
        (d / "config.yaml").write_text("algorithm:\n  name: PPO\n")
    # A dir without config.yaml — must be skipped.
    (root / "no_config").mkdir()
    # A plain file — must be skipped.
    (root / "loose_file.txt").write_text("x")

    available = list(list_available(root))
    # Only the config dirs, and sorted.
    assert available == ["a_bench", "b_bench"]


def test_list_available_non_dir_root_yields_empty(tmp_path):
    missing = tmp_path / "nope"
    assert list(list_available(missing)) == []
    a_file = tmp_path / "plain_file"
    a_file.write_text("x")
    assert list(list_available(a_file)) == []


# --------------------------------------------------------------------------- #
# BenchmarkResults discovery                                                   #
# --------------------------------------------------------------------------- #
def test_discovery_multi_seed_layout(make_benchmark_dir, tmp_path):
    runs = {
        ("Ant-v4", 1): [1.0, 2.0],
        ("Ant-v4", 2): [1.5, 2.5],
        ("HalfCheetah-v4", 1): [0.5, 1.5],
    }
    root = make_benchmark_dir(
        tmp_path / "bench",
        algo="PPO",
        pop_size=4,
        name="mybench",
        runs=runs,
        layout="multi",
    )
    br = BenchmarkResults(root)
    assert br.algo == "PPO"
    assert br.pop_size == 4
    assert br.name == "mybench"
    assert br.envs == {"Ant-v4", "HalfCheetah-v4"}
    assert br.seeds("Ant-v4") == {1, 2}
    assert br.seeds("HalfCheetah-v4") == {1}
    assert set(br.runs.keys()) == set(runs.keys())


def test_discovery_single_seed_layout_uses_default_seed(make_benchmark_dir, tmp_path):
    runs = {("Ant-v4", 7): [1.0, 2.0, 3.0]}
    root = make_benchmark_dir(
        tmp_path / "bench",
        layout="single",
        seed_for_single=7,
        runs=runs,
    )
    br = BenchmarkResults(root)
    # Single-seed layout: the seed comes from benchmark.seed (==7).
    assert br.runs.keys() == {("Ant-v4", 7)}
    assert br.seeds("Ant-v4") == {7}


def test_discovery_single_seed_layout_default_seed_zero(make_benchmark_dir, tmp_path):
    # No benchmark.seed set: _default_seed() falls back to 0.
    runs = {("Ant-v4", 0): [1.0, 2.0, 3.0]}
    root = make_benchmark_dir(
        tmp_path / "bench",
        layout="single",
        seed_for_single=None,
        runs=runs,
    )
    br = BenchmarkResults(root)
    assert br.runs.keys() == {("Ant-v4", 0)}


def test_discovery_multiseed_takes_precedence_bare_csv_ignored(tmp_path):
    """If any child matches ``s<digits>`` the bare csv is ignored."""
    import pandas as pd
    import yaml

    root = tmp_path / "bench"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "PPO"},
                "training": {"pop_size": 2},
                "benchmark": {"name": "b"},
            }
        )
    )
    env_dir = root / "Ant-v4"
    seed_dir = env_dir / "s5"
    seed_dir.mkdir(parents=True)
    df = pd.DataFrame({"train/global_step": [1, 2], "eval/best_fitness": [1.0, 2.0]})
    df.to_csv(seed_dir / "wandb_history.csv", index=False)
    # A bare csv directly under the env dir — must be ignored because s5 exists.
    df.to_csv(env_dir / "wandb_history.csv", index=False)

    br = BenchmarkResults(root)
    assert br.runs.keys() == {("Ant-v4", 5)}


def test_pop_size_zero_or_none_becomes_one(tmp_path):
    import yaml

    root = tmp_path / "bench"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "PPO"},
                "training": {"pop_size": 0},
                "benchmark": {"name": "b"},
            }
        )
    )
    import pandas as pd

    d = root / "Ant-v4" / "s1"
    d.mkdir(parents=True)
    pd.DataFrame({"train/global_step": [1], "eval/best_fitness": [1.0]}).to_csv(
        d / "wandb_history.csv", index=False
    )

    br = BenchmarkResults(root)
    assert br.pop_size == 1


def test_name_falls_back_to_root_name(tmp_path):
    import yaml
    import pandas as pd

    root = tmp_path / "fallback_name"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "PPO"},
                "training": {"pop_size": 2},
                # No benchmark.name.
                "benchmark": {},
            }
        )
    )
    d = root / "Ant-v4" / "s1"
    d.mkdir(parents=True)
    pd.DataFrame({"train/global_step": [1], "eval/best_fitness": [1.0]}).to_csv(
        d / "wandb_history.csv", index=False
    )

    br = BenchmarkResults(root)
    assert br.name == "fallback_name"


# --------------------------------------------------------------------------- #
# _default_seed precedence                                                     #
# --------------------------------------------------------------------------- #
def _make_results_with_bench(tmp_path, bench_block):
    import yaml
    import pandas as pd

    root = tmp_path / "bench"
    root.mkdir()
    config = {
        "algorithm": {"name": "PPO"},
        "training": {"pop_size": 2},
        "benchmark": bench_block,
    }
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    # single-seed layout so _default_seed is exercised.
    d = root / "Ant-v4"
    d.mkdir(parents=True)
    pd.DataFrame({"train/global_step": [1, 2], "eval/best_fitness": [1.0, 2.0]}).to_csv(
        d / "wandb_history.csv", index=False
    )
    return BenchmarkResults(root)


def test_default_seed_prefers_seed(tmp_path):
    br = _make_results_with_bench(tmp_path, {"name": "b", "seed": 11, "seeds": [99]})
    assert br._default_seed() == 11
    assert br.runs.keys() == {("Ant-v4", 11)}


def test_default_seed_uses_first_of_seeds(tmp_path):
    br = _make_results_with_bench(tmp_path, {"name": "b", "seeds": [3, 4, 5]})
    assert br._default_seed() == 3


def test_default_seed_defaults_to_zero(tmp_path):
    br = _make_results_with_bench(tmp_path, {"name": "b"})
    assert br._default_seed() == 0


# --------------------------------------------------------------------------- #
# curve()                                                                      #
# --------------------------------------------------------------------------- #
def test_curve_x_divisor_and_normalization(make_benchmark_dir, tmp_path):
    # Ant-v4 baselines: random=-59.9, expert=5846.4.
    runs = {("Ant-v4", 1): [0.0, 100.0]}
    root = make_benchmark_dir(
        tmp_path / "bench",
        algo="PPO",
        pop_size=4,
        runs=runs,
        layout="multi",
    )
    br = BenchmarkResults(root)
    x, y = br.curve("Ant-v4", 1)
    # steps auto-generated as [1, 2]; x = step / pop_size.
    np.testing.assert_allclose(x, np.array([1.0, 2.0]) / 4.0)
    denom = 5846.4 - (-59.9)
    expected_y = np.array([(0.0 - (-59.9)) / denom, (100.0 - (-59.9)) / denom])
    np.testing.assert_allclose(y, expected_y)


def test_curve_missing_run_returns_none(make_benchmark_dir, tmp_path):
    root = make_benchmark_dir(
        tmp_path / "bench",
        runs={("Ant-v4", 1): [1.0, 2.0]},
        layout="multi",
    )
    br = BenchmarkResults(root)
    assert br.curve("Ant-v4", 999) is None


def test_curve_missing_column_returns_none(tmp_path, monkeypatch):
    import yaml
    import pandas as pd

    root = tmp_path / "bench"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "PPO"},
                "training": {"pop_size": 1},
                "benchmark": {"name": "b"},
            }
        )
    )
    d = root / "Ant-v4" / "s1"
    d.mkdir(parents=True)
    # Missing eval/best_fitness column.
    pd.DataFrame({"train/global_step": [1, 2]}).to_csv(
        d / "wandb_history.csv", index=False
    )
    br = BenchmarkResults(root)
    assert br.curve("Ant-v4", 1) is None


def test_curve_all_nan_returns_none(tmp_path):
    import yaml
    import pandas as pd

    root = tmp_path / "bench"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "PPO"},
                "training": {"pop_size": 1},
                "benchmark": {"name": "b"},
            }
        )
    )
    d = root / "Ant-v4" / "s1"
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "train/global_step": [1.0, 2.0],
            "eval/best_fitness": [np.nan, np.nan],
        }
    ).to_csv(d / "wandb_history.csv", index=False)
    br = BenchmarkResults(root)
    assert br.curve("Ant-v4", 1) is None


# --------------------------------------------------------------------------- #
# Error paths                                                                  #
# --------------------------------------------------------------------------- #
def test_missing_config_raises(tmp_path):
    root = tmp_path / "bench"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        BenchmarkResults(root)


def test_no_history_csv_raises(tmp_path):
    import yaml

    root = tmp_path / "bench"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": {"name": "PPO"},
                "training": {"pop_size": 1},
                "benchmark": {"name": "b"},
            }
        )
    )
    # An env dir but no wandb_history.csv anywhere.
    (root / "Ant-v4").mkdir()
    with pytest.raises(FileNotFoundError):
        BenchmarkResults(root)


def test_missing_algorithm_name_raises_keyerror(tmp_path):
    import yaml
    import pandas as pd

    root = tmp_path / "bench"
    root.mkdir()
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                # algorithm block present but no name.
                "algorithm": {},
                "training": {"pop_size": 1},
                "benchmark": {"name": "b"},
            }
        )
    )
    d = root / "Ant-v4" / "s1"
    d.mkdir(parents=True)
    pd.DataFrame({"train/global_step": [1], "eval/best_fitness": [1.0]}).to_csv(
        d / "wandb_history.csv", index=False
    )
    with pytest.raises(KeyError):
        BenchmarkResults(root)
