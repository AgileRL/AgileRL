"""Hermetic tests for ``comparison/analysis.py``.

rliable bootstrap ``reps`` are kept tiny (~10) throughout so the suite is fast.
"""

from __future__ import annotations

import numpy as np
import pytest

import analysis
from analysis import (
    ComparisonResult,
    _common_grid,
    _iqm_with_ci,
    _probability_of_improvement,
    _probability_of_improvement_stratified,
    _task_improvement_prob,
    compare_benchmarks,
)


# --------------------------------------------------------------------------- #
# Duck-typed fake BenchmarkResults                                            #
# --------------------------------------------------------------------------- #
class FakeBenchmark:
    """Minimal stand-in for ``loading.BenchmarkResults``."""

    def __init__(self, algo, name, curves):
        # curves: {(env, seed): (x, y) | None}
        self.algo = algo
        self.name = name
        self._curves = curves

    @property
    def envs(self):
        return {env for (env, _seed) in self._curves}

    def seeds(self, env):
        return {seed for (e, seed) in self._curves if e == env}

    def curve(self, env, seed):
        return self._curves.get((env, seed))


def _line(x_vals, y_vals):
    return np.array(x_vals, dtype=float), np.array(y_vals, dtype=float)


# --------------------------------------------------------------------------- #
# _common_grid                                                                #
# --------------------------------------------------------------------------- #
def test_common_grid_exact_shared_path():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    # 6-tuple (xs, ys, xb, yb, lo, hi); only indices 0 and 2 used.
    pair_curves = {
        ("E", 1): (xs, None, xs, None, 0.0, 3.0),
        ("E", 2): (xs, None, xs, None, 0.0, 3.0),
    }
    grid, interpolated = _common_grid(
        pair_curves, [("E", 1), ("E", 2)], 0.0, 3.0, grid_points=50
    )
    assert interpolated is False
    np.testing.assert_allclose(grid, np.array([0.0, 1.0, 2.0, 3.0]))


def test_common_grid_respects_lo_hi_bounds():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    pair_curves = {("E", 1): (xs, None, xs, None, 0.0, 3.0)}
    grid, interpolated = _common_grid(pair_curves, [("E", 1)], 1.0, 2.0, grid_points=50)
    assert interpolated is False
    # Only values within [1, 2] survive.
    np.testing.assert_allclose(grid, np.array([1.0, 2.0]))


def test_common_grid_fallback_when_fewer_than_two_common():
    # studied and baseline x's don't coincide => no shared exact x.
    xs = np.array([0.0, 1.0, 2.0])
    xb = np.array([0.5, 1.5, 2.5])
    pair_curves = {("E", 1): (xs, None, xb, None, 0.0, 2.5)}
    grid, interpolated = _common_grid(pair_curves, [("E", 1)], 0.5, 2.0, grid_points=7)
    assert interpolated is True
    np.testing.assert_allclose(grid, np.linspace(0.5, 2.0, 7))


def test_common_grid_fallback_single_common_value():
    # Exactly one common value (<2) triggers the linspace fallback.
    xs = np.array([0.0, 1.0])
    xb = np.array([1.0, 2.0])
    pair_curves = {("E", 1): (xs, None, xb, None, 0.0, 2.0)}
    grid, interpolated = _common_grid(pair_curves, [("E", 1)], 0.0, 2.0, grid_points=5)
    assert interpolated is True
    np.testing.assert_allclose(grid, np.linspace(0.0, 2.0, 5))


# --------------------------------------------------------------------------- #
# _task_improvement_prob                                                       #
# --------------------------------------------------------------------------- #
def test_task_improvement_prob_equal_arrays():
    a = np.array([1.0, 2.0, 3.0])
    assert _task_improvement_prob(a, a.copy()) == 0.5


def test_task_improvement_prob_strictly_greater():
    x = np.array([10.0, 11.0, 12.0])
    y = np.array([1.0, 2.0, 3.0])
    assert _task_improvement_prob(x, y) == 1.0


def test_task_improvement_prob_strictly_less():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 11.0, 12.0])
    assert _task_improvement_prob(x, y) == 0.0


def test_task_improvement_prob_asymmetric_known_case():
    # x = [2, 4], y = [1, 3]: U(x>y) counts pairs where x_i > y_j.
    # (2>1)=1, (2>3)=0, (4>1)=1, (4>3)=1 => 3 of 4 => 0.75.
    x = np.array([2.0, 4.0])
    y = np.array([1.0, 3.0])
    assert _task_improvement_prob(x, y) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# _probability_of_improvement_stratified                                       #
# --------------------------------------------------------------------------- #
def test_prob_strat_identical_inputs_point_half():
    env_studied = {"A": np.array([1.0, 2.0, 3.0])}
    env_baseline = {"A": np.array([1.0, 2.0, 3.0])}
    point, lo, hi = _probability_of_improvement_stratified(
        env_studied, env_baseline, reps=10, seed=0
    )
    assert point == 0.5
    assert lo <= point <= hi


def test_prob_strat_deterministic_for_fixed_seed():
    env_studied = {"A": np.array([5.0, 6.0, 7.0, 8.0])}
    env_baseline = {"A": np.array([1.0, 2.0, 3.0, 9.0])}
    r1 = _probability_of_improvement_stratified(
        env_studied, env_baseline, reps=20, seed=3
    )
    r2 = _probability_of_improvement_stratified(
        env_studied, env_baseline, reps=20, seed=3
    )
    assert r1 == r2


def test_prob_strat_ci_brackets_point():
    env_studied = {"A": np.array([5.0, 6.0, 7.0]), "B": np.array([4.0, 5.0])}
    env_baseline = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([1.0, 2.0])}
    point, lo, hi = _probability_of_improvement_stratified(
        env_studied, env_baseline, reps=30, seed=1
    )
    assert lo <= point <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


# --------------------------------------------------------------------------- #
# _iqm_with_ci                                                                 #
# --------------------------------------------------------------------------- #
def test_iqm_with_ci_shapes_and_determinism():
    rng = np.random.RandomState(0)
    scores = rng.rand(4, 2, 5)  # (n_runs, n_tasks, n_frames)
    iqm1, lo1, hi1 = _iqm_with_ci(scores, reps=10)
    assert iqm1.shape == (5,)
    assert lo1.shape == (5,)
    assert hi1.shape == (5,)
    iqm2, _lo2, _hi2 = _iqm_with_ci(scores, reps=10)
    # The IQM point estimate is deterministic (no randomness). The CI bands draw
    # from NumPy's global RNG (the arch bootstrap), which the conftest restores
    # only after the test, so band bounds are not required to match across two
    # in-test calls.
    np.testing.assert_array_equal(iqm1, iqm2)
    # CI brackets the point estimate.
    assert np.all(lo1 <= iqm1 + 1e-9)
    assert np.all(hi1 >= iqm1 - 1e-9)


def test_iqm_with_ci_forwards_to_bench_plotting(monkeypatch):
    sentinel_point = {analysis.DIFF_KEY: np.array([1.0, 2.0])}
    sentinel_cis = {analysis.DIFF_KEY: np.array([[0.5, 1.5], [1.5, 2.5]])}
    captured = {}

    def fake(score_dict, reps):
        captured["score_dict"] = score_dict
        captured["reps"] = reps
        return sentinel_point, sentinel_cis

    monkeypatch.setattr(analysis.bench_plotting, "_iqm_interval_estimates", fake)
    scores = np.ones((2, 1, 2))
    iqm, lo, hi = _iqm_with_ci(scores, reps=42)
    assert captured["reps"] == 42
    assert analysis.DIFF_KEY in captured["score_dict"]
    np.testing.assert_array_equal(iqm, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(lo, np.array([0.5, 1.5]))
    np.testing.assert_array_equal(hi, np.array([1.5, 2.5]))


# --------------------------------------------------------------------------- #
# _probability_of_improvement                                                  #
# --------------------------------------------------------------------------- #
def test_probability_of_improvement_shapes_and_determinism():
    studied = np.array([[0.6, 0.7], [0.65, 0.75]])  # (n_runs, n_tasks)
    baseline = np.array([[0.4, 0.5], [0.45, 0.55]])
    p1, lo1, hi1 = _probability_of_improvement(studied, baseline, reps=10, seed=0)
    p2, lo2, hi2 = _probability_of_improvement(studied, baseline, reps=10, seed=0)
    assert isinstance(p1, float)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)
    assert lo1 <= p1 <= hi1


def test_probability_of_improvement_forwards(monkeypatch):
    captured = {}

    def fake_get_interval(score_dict, fn, reps, random_state):
        captured["score_dict"] = score_dict
        captured["reps"] = reps
        point = {analysis.PROB_KEY: np.array([0.83])}
        cis = {analysis.PROB_KEY: np.array([[0.7], [0.9]])}
        return point, cis

    monkeypatch.setattr(analysis.rly, "get_interval_estimates", fake_get_interval)
    studied = np.array([[0.6]])
    baseline = np.array([[0.4]])
    p, lo, hi = _probability_of_improvement(studied, baseline, reps=7, seed=0)
    assert captured["reps"] == 7
    assert analysis.PROB_KEY in captured["score_dict"]
    assert p == pytest.approx(0.83)
    assert lo == pytest.approx(0.7)
    assert hi == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# compare_benchmarks: error paths                                             #
# --------------------------------------------------------------------------- #
def test_compare_algo_mismatch_raises():
    studied = FakeBenchmark("PPO", "s", {("Ant-v4", 1): _line([0, 1], [0.1, 0.2])})
    baseline = FakeBenchmark("DQN", "b", {("Ant-v4", 1): _line([0, 1], [0.1, 0.2])})
    with pytest.raises(ValueError, match="Algorithm mismatch"):
        compare_benchmarks(studied, baseline, reps=10)


def test_compare_no_shared_env_raises():
    studied = FakeBenchmark("PPO", "s", {("Ant-v4", 1): _line([0, 1], [0.1, 0.2])})
    baseline = FakeBenchmark(
        "PPO", "b", {("HalfCheetah-v4", 1): _line([0, 1], [0.1, 0.2])}
    )
    with pytest.raises(ValueError, match="share no environment"):
        compare_benchmarks(studied, baseline, reps=10)


def test_compare_no_shared_seed_raises():
    studied = FakeBenchmark("PPO", "s", {("Ant-v4", 1): _line([0, 1], [0.1, 0.2])})
    baseline = FakeBenchmark("PPO", "b", {("Ant-v4", 2): _line([0, 1], [0.1, 0.2])})
    with pytest.raises(ValueError, match="share no .*pair"):
        compare_benchmarks(studied, baseline, reps=10)


# --------------------------------------------------------------------------- #
# compare_benchmarks: rectangular path                                        #
# --------------------------------------------------------------------------- #
def _rect_curves(seeds, envs, studied_y, baseline_y):
    x = [0.0, 1.0, 2.0]
    s = {}
    b = {}
    for env in envs:
        for seed in seeds:
            s[(env, seed)] = _line(x, studied_y)
            b[(env, seed)] = _line(x, baseline_y)
    return s, b


def test_compare_rectangular_success():
    s_curves, b_curves = _rect_curves(
        seeds=[1, 2],
        envs=["Ant-v4"],
        studied_y=[0.5, 0.6, 0.7],
        baseline_y=[0.3, 0.4, 0.5],
    )
    studied = FakeBenchmark("PPO", "studied", s_curves)
    baseline = FakeBenchmark("PPO", "baseline", b_curves)
    result = compare_benchmarks(studied, baseline, reps=10)

    assert isinstance(result, ComparisonResult)
    assert result.algo == "PPO"
    assert result.studied_name == "studied"
    assert result.baseline_name == "baseline"
    assert result.common_envs == ["Ant-v4"]
    assert result.common_seeds == [1, 2]
    assert result.n_pairs == 2
    # rectangular layout: (n_seeds, n_tasks) finals.
    assert result.studied_final.shape == (2, 1)
    assert result.baseline_final.shape == (2, 1)
    # studied strictly above baseline => probability of improvement is 1.0.
    assert result.prob_improvement == pytest.approx(1.0)
    # diff curve over the shared exact grid.
    assert result.x.size > 0
    assert result.diff_iqm.shape == result.x.shape
    assert result.interpolated is False
    # final values aligned with pairs.
    assert len(result.studied_final_flat) == result.n_pairs
    assert result.pairs == [("Ant-v4", 1), ("Ant-v4", 2)]


def test_compare_rectangular_final_at_last_step():
    s_curves, b_curves = _rect_curves(
        seeds=[1, 2],
        envs=["Ant-v4"],
        studied_y=[0.5, 0.6, 0.9],
        baseline_y=[0.3, 0.4, 0.5],
    )
    studied = FakeBenchmark("PPO", "studied", s_curves)
    baseline = FakeBenchmark("PPO", "baseline", b_curves)
    result = compare_benchmarks(studied, baseline, reps=10)
    # Final value is taken at the largest shared step => last y values.
    np.testing.assert_allclose(result.studied_final.ravel(), [0.9, 0.9])
    np.testing.assert_allclose(result.baseline_final.ravel(), [0.5, 0.5])


# --------------------------------------------------------------------------- #
# compare_benchmarks: ragged path                                             #
# --------------------------------------------------------------------------- #
def test_compare_ragged_uneven_seeds():
    x = [0.0, 1.0, 2.0]
    sy = [0.5, 0.6, 0.7]
    by = [0.3, 0.4, 0.5]
    # Ant has seeds {1,2}; HalfCheetah has seed {1}. Unequal seed sets => ragged.
    s_curves = {
        ("Ant-v4", 1): _line(x, sy),
        ("Ant-v4", 2): _line(x, sy),
        ("HalfCheetah-v4", 1): _line(x, sy),
    }
    b_curves = {
        ("Ant-v4", 1): _line(x, by),
        ("Ant-v4", 2): _line(x, by),
        ("HalfCheetah-v4", 1): _line(x, by),
    }
    studied = FakeBenchmark("PPO", "studied", s_curves)
    baseline = FakeBenchmark("PPO", "baseline", b_curves)
    result = compare_benchmarks(studied, baseline, reps=10)

    assert result.n_pairs == 3
    # ragged: finals pooled to (n_pairs, 1).
    assert result.studied_final.shape == (3, 1)
    assert result.baseline_final.shape == (3, 1)
    assert sorted(result.common_envs) == ["Ant-v4", "HalfCheetah-v4"]
    assert result.per_env_seeds["Ant-v4"] == [1, 2]
    assert result.per_env_seeds["HalfCheetah-v4"] == [1]
    assert result.prob_improvement == pytest.approx(1.0)
    assert len(result.pairs) == 3


# --------------------------------------------------------------------------- #
# _collect_pair_curves drops non-overlapping x-ranges                          #
# --------------------------------------------------------------------------- #
def test_collect_pair_curves_drops_nonoverlapping():
    # studied x in [0,1], baseline x in [5,6]: hi <= lo so the pair is dropped.
    s_curves = {("Ant-v4", 1): _line([0.0, 1.0], [0.5, 0.6])}
    b_curves = {("Ant-v4", 1): _line([5.0, 6.0], [0.3, 0.4])}
    out = analysis._collect_pair_curves(
        FakeBenchmark("PPO", "s", s_curves),
        FakeBenchmark("PPO", "b", b_curves),
        [("Ant-v4", 1)],
    )
    assert out == {}


def test_collect_pair_curves_drops_none_curves():
    s_curves = {("Ant-v4", 1): None}
    b_curves = {("Ant-v4", 1): _line([0.0, 1.0], [0.3, 0.4])}
    out = analysis._collect_pair_curves(
        FakeBenchmark("PPO", "s", s_curves),
        FakeBenchmark("PPO", "b", b_curves),
        [("Ant-v4", 1)],
    )
    assert out == {}


def test_compare_rectangular_no_overlap_raises():
    # Both envs/seeds present and a shared pair exists, but the per-agent
    # x-ranges are disjoint => no usable task => ValueError.
    s_curves = {("Ant-v4", 1): _line([0.0, 1.0], [0.5, 0.6])}
    b_curves = {("Ant-v4", 1): _line([5.0, 6.0], [0.3, 0.4])}
    studied = FakeBenchmark("PPO", "s", s_curves)
    baseline = FakeBenchmark("PPO", "b", b_curves)
    with pytest.raises(ValueError, match="overlapping curve"):
        compare_benchmarks(studied, baseline, reps=10)


# --------------------------------------------------------------------------- #
# ComparisonResult is a trivially-constructible dataclass                      #
# --------------------------------------------------------------------------- #
def test_comparison_result_dataclass(make_comparison_result):
    res = make_comparison_result()
    assert isinstance(res, ComparisonResult)
    assert res.confidence_level == 0.95
    # default field can be overridden.
    res2 = make_comparison_result(confidence_level=0.9)
    assert res2.confidence_level == 0.9
