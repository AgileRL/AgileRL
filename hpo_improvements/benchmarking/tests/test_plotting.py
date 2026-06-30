"""Tests for ``plotting`` (the benchmarking figure module).

Two layers:

* the pure numeric helpers (grid alignment, score-array assembly, axis styling,
  the rliable interval wrappers) get tight value/shape/edge assertions;
* every public ``plot_*`` figure function gets a PNG-existence smoke test for
  both its real path and its degenerate/placeholder path, including the
  early-return-writes-no-file branches and the non-informative-lineage
  placeholder.

rliable bootstrap is sped up via the ``fast_bootstrap`` fixture (small ``reps``).
matplotlib is Agg (forced by the conftest); figures are closed by the autouse
isolation fixture.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import plotting
from plotting import DIVERSITY_SPECS


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def assert_png(path) -> None:
    assert os.path.exists(path), f"expected a PNG at {path}"
    assert os.path.getsize(path) > 0, f"PNG at {path} is empty"


def assert_absent(path) -> None:
    assert not os.path.exists(path), f"expected NO file at {path}"


@pytest.fixture
def fast_bootstrap(monkeypatch):
    """Force small rliable bootstrap replications for speed."""
    orig_iqm = plotting._iqm_interval_estimates
    orig_mean = plotting._mean_interval_estimates
    monkeypatch.setattr(plotting, "_BOOTSTRAP_REPS", 50)
    monkeypatch.setattr(
        plotting, "_iqm_interval_estimates", lambda sd, reps=50: orig_iqm(sd, reps=50)
    )
    monkeypatch.setattr(
        plotting, "_mean_interval_estimates", lambda v, reps=50: orig_mean(v, reps=50)
    )


def png(tmp_path, name="fig.png"):
    return str(tmp_path / name)


# --------------------------------------------------------------------------- #
# _per_agent_x                                                                  #
# --------------------------------------------------------------------------- #
class TestPerAgentX:
    def _df(self):
        return pd.DataFrame({"train/global_step": [4.0, 8.0, 12.0]})

    def test_divides_by_pop_size(self):
        np.testing.assert_allclose(plotting._per_agent_x(self._df(), 4), [1, 2, 3])

    def test_pop_size_zero_and_negative_clamp_to_one(self):
        np.testing.assert_allclose(plotting._per_agent_x(self._df(), 0), [4, 8, 12])
        np.testing.assert_allclose(plotting._per_agent_x(self._df(), -5), [4, 8, 12])


# --------------------------------------------------------------------------- #
# _common_overlap_grid                                                          #
# --------------------------------------------------------------------------- #
class TestCommonOverlapGrid:
    def test_empty_list(self):
        assert plotting._common_overlap_grid([]).size == 0

    def test_overlapping_range_and_coarsest_resolution(self):
        grid = plotting._common_overlap_grid(
            [np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0])]
        )
        # overlap [1, 2]; npts = min(4, 2) = 2.
        np.testing.assert_allclose(grid, [1.0, 2.0])

    def test_non_overlapping_returns_empty(self):
        grid = plotting._common_overlap_grid([np.array([0, 1]), np.array([5, 6])])
        assert grid.size == 0

    def test_touching_at_point_returns_empty(self):
        grid = plotting._common_overlap_grid([np.array([0, 1]), np.array([1, 2])])
        assert grid.size == 0  # hi == lo == 1, not hi > lo

    def test_min_two_points(self):
        grid = plotting._common_overlap_grid(
            [np.array([0.0, 4.0]), np.array([0.0, 4.0])]
        )
        assert grid.size == 2


# --------------------------------------------------------------------------- #
# _align_on_common_x                                                            #
# --------------------------------------------------------------------------- #
class TestAlignOnCommonX:
    def test_identical_x_is_exact(self):
        x = np.array([0.0, 1.0, 2.0])
        grid, stacked = plotting._align_on_common_x([(x, x * 10), (x, x * 100)])
        np.testing.assert_allclose(grid, x)
        np.testing.assert_allclose(stacked[0], x * 10)
        np.testing.assert_allclose(stacked[1], x * 100)

    def test_unsorted_input_handled(self):
        x = np.array([2.0, 0.0, 1.0])
        y = np.array([20.0, 0.0, 10.0])
        grid, stacked = plotting._align_on_common_x([(x, y), (x, y)])
        np.testing.assert_allclose(grid, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(stacked[0], [0.0, 10.0, 20.0])

    def test_linear_interp_midpoint(self):
        # Both curves have 3 points -> grid = linspace(0, 10, 3) = [0, 5, 10];
        # c1's samples don't include 5, so its value there is interpolated.
        c1 = (np.array([0.0, 3.0, 10.0]), np.array([0.0, 9.0, 100.0]))
        c2 = (np.array([0.0, 5.0, 10.0]), np.array([0.0, 5.0, 10.0]))
        grid, stacked = plotting._align_on_common_x([c1, c2])
        np.testing.assert_allclose(grid, [0.0, 5.0, 10.0])
        # c1 at x=5: 9 + (5-3)/(10-3)*(100-9) = 35.0
        np.testing.assert_allclose(stacked[0], [0.0, 35.0, 100.0])
        np.testing.assert_allclose(stacked[1], [0.0, 5.0, 10.0])

    def test_no_overlap_returns_empty_shapes(self):
        grid, stacked = plotting._align_on_common_x(
            [
                (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                (np.array([5.0, 6.0]), np.array([5.0, 6.0])),
            ]
        )
        assert grid.size == 0
        assert stacked.shape == (2, 0)


# --------------------------------------------------------------------------- #
# _aggregate_score_array                                                        #
# --------------------------------------------------------------------------- #
class TestAggregateScoreArray:
    def test_rectangularises_to_min_seed_count(self):
        curves = {
            "a": (np.array([0.0, 1.0]), np.ones((3, 2))),
            "b": (np.array([0.0, 1.0]), np.full((2, 2), 0.5)),
        }
        grid, scores = plotting._aggregate_score_array(curves)
        assert scores.shape == (2, 2, 2)  # min seeds=2, 2 envs, 2 frames

    def test_empty_returns_none(self):
        assert plotting._aggregate_score_array({}) is None

    def test_no_overlap_returns_none(self):
        curves = {
            "a": (np.array([0.0, 1.0]), np.ones((2, 2))),
            "b": (np.array([5.0, 6.0]), np.ones((2, 2))),
        }
        assert plotting._aggregate_score_array(curves) is None


# --------------------------------------------------------------------------- #
# small pure helpers                                                            #
# --------------------------------------------------------------------------- #
class TestSmallHelpers:
    def test_wide_positive(self):
        assert plotting._wide_positive(np.array([1.0, 200.0])) is True
        assert plotting._wide_positive(np.array([1.0, 50.0])) is False
        assert plotting._wide_positive(np.array([-1.0, 1000.0])) is False
        assert plotting._wide_positive(np.array([])) is False

    def test_category_color_known_and_fallback(self):
        assert (
            plotting._category_color("parameter", 0)
            == plotting.CATEGORY_COLORS["parameter"]
        )
        fb = plotting._FALLBACK_CATEGORY_COLORS
        assert plotting._category_color("unknown", 0) == fb[0]
        assert plotting._category_color("unknown", 5) == fb[5 % len(fb)]

    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 2), (3, 4), (4, 4)])
    def test_hp_panel_grid_axes_count(self, n, expected):
        import matplotlib.pyplot as plt

        fig, axes = plotting._hp_panel_grid(n)
        assert axes.size == expected
        plt.close(fig)

    def test_best_agent_indices(self):
        df = pd.DataFrame(
            {"eval/agent_0/fitness": [1.0, 5.0], "eval/agent_1/fitness": [3.0, 2.0]}
        )
        np.testing.assert_array_equal(plotting._best_agent_indices(df), [1, 0])

    def test_best_agent_indices_no_columns_zero(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        np.testing.assert_array_equal(plotting._best_agent_indices(df), [0, 0, 0])

    def test_best_agent_indices_nan_is_neg_inf(self):
        df = pd.DataFrame(
            {"eval/agent_0/fitness": [np.nan], "eval/agent_1/fitness": [1.0]}
        )
        np.testing.assert_array_equal(plotting._best_agent_indices(df), [1])

    def test_finalize_axis_styles(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_title("t")
        h = ax.axhline(0.0, label="ref")
        plotting._finalize_axis(ax, legend_handles=[h])
        assert all(s.get_visible() for s in ax.spines.values())
        assert ax.xaxis.label.get_fontweight() == "bold"
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_save_placeholder_writes_png(self, tmp_path):
        out = png(tmp_path)
        plotting._save_placeholder(out, "nothing here")
        assert_png(out)


# --------------------------------------------------------------------------- #
# rliable interval wrappers                                                     #
# --------------------------------------------------------------------------- #
class TestIntervalWrappers:
    def test_iqm_interval_shapes_and_determinism(self):
        sd = {"a": np.ones((2, 1, 3))}
        p1, c1 = plotting._iqm_interval_estimates(sd, reps=20)
        p2, c2 = plotting._iqm_interval_estimates(sd, reps=20)
        assert p1["a"].shape == (3,)
        assert c1["a"].shape == (2, 3)
        np.testing.assert_allclose(p1["a"], p2["a"])
        # constant scores -> IQM ~ 1 with collapsed CI.
        np.testing.assert_allclose(p1["a"], [1.0, 1.0, 1.0])

    def test_mean_interval_empty(self):
        m, lo, hi = plotting._mean_interval_estimates(np.array([]))
        assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)

    def test_mean_interval_single_zero_width(self):
        assert plotting._mean_interval_estimates(np.array([3.0])) == (3.0, 3.0, 3.0)

    def test_mean_interval_multiple(self):
        m, lo, hi = plotting._mean_interval_estimates(
            np.array([1.0, 2.0, 3.0]), reps=50
        )
        assert m == pytest.approx(2.0, abs=0.6)
        assert lo <= m <= hi


# --------------------------------------------------------------------------- #
# Per-run DataFrame figures                                                     #
# --------------------------------------------------------------------------- #
class TestPerRunFigures:
    def test_plot_fitness(self, tmp_path):
        from registry import NormalizationScores

        df = pd.DataFrame(
            {"train/global_step": [0, 10], "eval/best_fitness": [1.0, 2.0]}
        )
        out = png(tmp_path)
        plotting.plot_fitness(df, "Ant-v4", NormalizationScores(0.0, 1.0), 2, out)
        assert_png(out)

    def test_plot_fitness_equal_baselines_no_raise(self, tmp_path):
        from registry import NormalizationScores

        df = pd.DataFrame(
            {"train/global_step": [0, 10], "eval/best_fitness": [1.0, 2.0]}
        )
        out = png(tmp_path)
        plotting.plot_fitness(df, "Ant-v4", NormalizationScores(1.0, 1.0), 2, out)
        assert_png(out)

    def test_plot_dormant_fraction_real(self, tmp_path):
        df = pd.DataFrame(
            {"train/global_step": [0, 10], "eval/best_dormant_fraction": [0.1, 0.2]}
        )
        out = png(tmp_path)
        plotting.plot_dormant_fraction(df, "Ant-v4", 2, out)
        assert_png(out)

    def test_plot_dormant_fraction_placeholder(self, tmp_path):
        df = pd.DataFrame({"train/global_step": [0, 10]})
        out = png(tmp_path)
        plotting.plot_dormant_fraction(df, "Ant-v4", 2, out)
        assert_png(out)

    def test_plot_mutation_schedule_real(self, tmp_path):
        df = pd.DataFrame(
            {
                "train/global_step": [0, 10],
                "train/agent_0/fitness": [1.0, 2.0],
                "train/agent_0/learning_rate": [0.1, 0.2],
                "eval/agent_0/fitness": [1.0, 2.0],
            }
        )
        out = png(tmp_path)
        plotting.plot_mutation_schedule(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_mutation_schedule_placeholder(self, tmp_path):
        df = pd.DataFrame({"train/global_step": [0, 10]})
        out = png(tmp_path)
        plotting.plot_mutation_schedule(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_diversity_real_and_placeholder_panels(self, tmp_path):
        col = DIVERSITY_SPECS[0][0]
        df = pd.DataFrame({"train/global_step": [0, 10], col: [0.3, 0.4]})
        out = png(tmp_path)
        plotting.plot_diversity(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_diversity_all_placeholder(self, tmp_path):
        df = pd.DataFrame({"train/global_step": [0, 10]})
        out = png(tmp_path)
        plotting.plot_diversity(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_population_hp_trajectory_real(self, tmp_path):
        df = pd.DataFrame(
            {"train/global_step": [0, 10], "train/agent_0/learning_rate": [0.1, 0.2]}
        )
        out = png(tmp_path)
        plotting.plot_population_hp_trajectory(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_population_hp_trajectory_placeholder(self, tmp_path):
        df = pd.DataFrame({"train/global_step": [0, 10]})
        out = png(tmp_path)
        plotting.plot_population_hp_trajectory(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_hp_fitness_real(self, tmp_path):
        df = pd.DataFrame(
            {
                "train/global_step": [0, 10],
                "train/agent_0/learning_rate": [0.1, 0.2],
                "eval/agent_0/fitness": [1.0, 2.0],
            }
        )
        out = png(tmp_path)
        plotting.plot_hp_fitness(df, "Ant-v4", 1, out)
        assert_png(out)

    def test_plot_hp_fitness_placeholder(self, tmp_path):
        df = pd.DataFrame({"train/global_step": [0, 10]})
        out = png(tmp_path)
        plotting.plot_hp_fitness(df, "Ant-v4", 1, out)
        assert_png(out)


# --------------------------------------------------------------------------- #
# Over-seeds / aggregate figures                                                #
# --------------------------------------------------------------------------- #
class TestOverSeedsAndAggregate:
    def _two_seed_fitness(self):
        x = np.array([0.0, 1.0])
        return [
            (x, np.array([1.0, 2.0]), np.array([0.0, 1.0])),
            (x, np.array([1.5, 2.5]), np.array([0.2, 1.2])),
        ]

    def test_fitness_over_seeds_real(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        res = plotting.plot_fitness_over_seeds(self._two_seed_fitness(), "Ant-v4", out)
        assert_png(out)
        grid, stacked = res
        assert stacked.shape == (2, grid.size)

    def test_fitness_over_seeds_degenerate_returns_none(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        assert plotting.plot_fitness_over_seeds([], "Ant-v4", out) is None
        assert_absent(out)

    def test_plot_aggregate_real(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        curves = {"Ant-v4": (np.array([0.0, 1.0]), np.ones((2, 2)))}
        plotting.plot_aggregate(curves, out)
        assert_png(out)

    def test_plot_aggregate_degenerate_no_file(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_aggregate({}, out)
        assert_absent(out)

    def test_plot_performance_profile_real(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        curves = {"Ant-v4": (np.array([0.0, 1.0]), np.array([[0.2, 0.9], [0.4, 0.8]]))}
        plotting.plot_performance_profile(curves, out)
        assert_png(out)

    def test_plot_performance_profile_degenerate_no_file(
        self, tmp_path, fast_bootstrap
    ):
        out = png(tmp_path)
        plotting.plot_performance_profile({}, out)
        assert_absent(out)

    def test_diversity_over_seeds_real(self, tmp_path, fast_bootstrap):
        col = DIVERSITY_SPECS[0][0]
        seed_curves = [
            (np.array([0.0, 1.0]), {col: np.array([0.1, 0.2])}),
            (np.array([0.0, 1.0]), {col: np.array([0.3, 0.4])}),
        ]
        out = png(tmp_path)
        res = plotting.plot_diversity_over_seeds(seed_curves, "Ant-v4", out)
        assert_png(out)
        assert col in res

    def test_diversity_over_seeds_empty(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        res = plotting.plot_diversity_over_seeds([], "Ant-v4", out)
        assert_png(out)  # always writes a (placeholder) 4-panel PNG
        assert res == {}

    def test_diversity_aggregate_real(self, tmp_path, fast_bootstrap):
        col = DIVERSITY_SPECS[0][0]
        per_metric = {col: {"Ant-v4": (np.array([0.0, 1.0]), np.ones((2, 2)))}}
        out = png(tmp_path)
        plotting.plot_diversity_aggregate(per_metric, out)
        assert_png(out)

    def test_diversity_aggregate_empty_no_file(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_diversity_aggregate({}, out)
        assert_absent(out)

    def test_mechanism_population_over_seeds_real(self, tmp_path, fast_bootstrap):
        sel = [
            (np.array([0.0, 1.0]), np.array([0.5, 0.6])),
            (np.array([0.0, 1.0]), np.array([0.4, 0.7])),
        ]
        tur = [
            (np.array([0.0, 1.0]), np.array([0.2, 0.3])),
            (np.array([0.0, 1.0]), np.array([0.1, 0.4])),
        ]
        out = png(tmp_path)
        res = plotting.plot_mechanism_population_over_seeds(sel, tur, "Ant-v4", out)
        assert_png(out)
        assert "selection" in res and "turnover" in res

    def test_mechanism_population_over_seeds_empty(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        res = plotting.plot_mechanism_population_over_seeds([], [], "Ant-v4", out)
        assert_png(out)
        assert res == {}

    def test_mechanism_population_aggregate_real(self, tmp_path, fast_bootstrap):
        per_metric = {"selection": {"Ant-v4": (np.array([0.0, 1.0]), np.ones((2, 2)))}}
        out = png(tmp_path)
        plotting.plot_mechanism_population_aggregate(per_metric, out)
        assert_png(out)

    def test_mechanism_population_aggregate_empty_no_file(
        self, tmp_path, fast_bootstrap
    ):
        out = png(tmp_path)
        plotting.plot_mechanism_population_aggregate({}, out)
        assert_absent(out)


# --------------------------------------------------------------------------- #
# Mechanism efficacy / population figures                                       #
# --------------------------------------------------------------------------- #
class TestMechanismFigures:
    def _efficacy_df(self):
        return pd.DataFrame(
            {
                "mutation_category": ["parameter", "parameter", "no mutation"],
                "fitness_before": [1.0, 1.0, 5.0],
                "fitness_after": [2.0, 0.5, 6.0],
            }
        )

    def test_plot_mechanism_efficacy_real(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy(self._efficacy_df(), "Ant-v4", out)
        assert_png(out)

    def test_plot_mechanism_efficacy_placeholder(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy(pd.DataFrame(), "Ant-v4", out)
        assert_png(out)

    def test_plot_mechanism_efficacy_over_seeds(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy_over_seeds(
            {"parameter": np.array([1.0, -0.5, 0.3])}, "Ant-v4", out
        )
        assert_png(out)

    def test_plot_mechanism_efficacy_distribution(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy_distribution(
            self._efficacy_df(), "Ant-v4", out
        )
        assert_png(out)

    def test_plot_mechanism_efficacy_distribution_over_seeds(
        self, tmp_path, fast_bootstrap
    ):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy_distribution_over_seeds(
            {"parameter": np.array([1.0, -0.5, 0.3, 0.8])}, "Ant-v4", out
        )
        assert_png(out)

    def test_plot_mechanism_efficacy_aggregate(self, tmp_path, fast_bootstrap):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy_aggregate(
            {"parameter": np.array([0.1, -0.05, 0.2])}, out
        )
        assert_png(out)

    def test_plot_mechanism_efficacy_distribution_aggregate(
        self, tmp_path, fast_bootstrap
    ):
        out = png(tmp_path)
        plotting.plot_mechanism_efficacy_distribution_aggregate(
            {"parameter": np.array([0.1, -0.05, 0.2, 0.15])}, out
        )
        assert_png(out)

    def test_plot_mechanism_population_real(self, tmp_path, fast_bootstrap):
        # informative lineage: a post-gen-0 row whose parent differs from agent.
        df = pd.DataFrame(
            {
                "generation": [0, 0, 1, 1],
                "parent_id": [0, 1, 0, 0],
                "agent_id": [0, 1, 0, 1],
                "mutation_category": [
                    "no mutation",
                    "no mutation",
                    "parameter",
                    "parameter",
                ],
                "global_step": [0, 0, 400, 400],
            }
        )
        out = png(tmp_path)
        plotting.plot_mechanism_population(df, "Ant-v4", 4, out)
        assert_png(out)

    def test_plot_mechanism_population_noninformative_placeholder(self, tmp_path):
        # every agent is its own parent -> placeholder, no rliable needed.
        df = pd.DataFrame(
            {
                "generation": [0, 1],
                "parent_id": [0, 1],
                "agent_id": [0, 1],
                "mutation_category": ["no mutation", "parameter"],
                "global_step": [0, 400],
            }
        )
        out = png(tmp_path)
        plotting.plot_mechanism_population(df, "Ant-v4", 2, out)
        assert_png(out)
