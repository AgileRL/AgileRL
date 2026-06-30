"""Tests for ``mechanism`` (pure NumPy evolutionary-mechanism diagnostics).

Every function is exercised with tiny synthetic ``mutation_history.csv`` /
W&B-history DataFrames, covering the guard returns (``{}`` vs a pair of empty
arrays), the exact numeric contracts, and the degenerate-lineage paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mechanism
from mechanism import (
    CATEGORY_ORDER,
    GLOBAL_STEP_COL,
    category_deltas,
    efficacy_by_category,
    efficacy_from_deltas,
    hp_fitness_samples,
    infer_hp_names,
    lineage_is_informative,
    order_categories,
    population_hp_trajectory,
    selection_pressure,
    turnover,
)


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
def test_constants():
    assert GLOBAL_STEP_COL == "train/global_step"
    assert mechanism.MUT_GLOBAL_STEP_COL == "global_step"
    assert mechanism.CONTROL_CATEGORY == "no mutation"
    assert CATEGORY_ORDER[0] == "no mutation"
    assert mechanism._RESERVED_HP_SUFFIXES == {
        "fitness",
        "score",
        "local_steps",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "steps_per_second",
    }


# --------------------------------------------------------------------------- #
# order_categories                                                             #
# --------------------------------------------------------------------------- #
class TestOrderCategories:
    def test_empty(self):
        assert order_categories([]) == []

    def test_known_first_in_canonical_order(self):
        out = order_categories(["activation", "parameter", "no mutation"])
        assert out == ["no mutation", "parameter", "activation"]

    def test_unknown_appended_alphabetically(self):
        out = order_categories(["zeta", "parameter", "alpha"])
        assert out == ["parameter", "alpha", "zeta"]

    def test_known_absent_from_input_dropped(self):
        assert order_categories(["parameter"]) == ["parameter"]

    def test_duplicate_unknown_kept(self):
        assert order_categories(["x", "x"]) == ["x", "x"]


# --------------------------------------------------------------------------- #
# category_deltas                                                             #
# --------------------------------------------------------------------------- #
class TestCategoryDeltas:
    def _df(self, **extra):
        return pd.DataFrame(
            {
                "mutation_category": ["parameter", "parameter", "no mutation"],
                "fitness_before": [1.0, 2.0, 5.0],
                "fitness_after": [2.0, 1.0, 6.0],
                **extra,
            }
        )

    def test_none_returns_empty(self):
        assert category_deltas(None) == {}

    def test_empty_df_returns_empty(self):
        assert category_deltas(pd.DataFrame()) == {}

    def test_missing_columns_returns_empty(self):
        df = pd.DataFrame({"mutation_category": ["parameter"]})
        assert category_deltas(df) == {}

    def test_basic_deltas(self):
        out = category_deltas(self._df())
        assert set(out) == {"parameter", "no mutation"}
        np.testing.assert_allclose(sorted(out["parameter"]), [-1.0, 1.0])
        np.testing.assert_allclose(out["no mutation"], [1.0])

    def test_non_finite_rows_excluded(self):
        df = pd.DataFrame(
            {
                "mutation_category": ["parameter", "parameter", "parameter"],
                "fitness_before": [np.nan, 2.0, np.inf],
                "fitness_after": [1.0, 3.0, 4.0],
            }
        )
        out = category_deltas(df)
        np.testing.assert_allclose(out["parameter"], [1.0])

    def test_all_non_finite_returns_empty(self):
        df = pd.DataFrame(
            {
                "mutation_category": ["parameter"],
                "fitness_before": [np.nan],
                "fitness_after": [np.nan],
            }
        )
        assert category_deltas(df) == {}

    def test_category_only_in_nonfinite_rows_omitted(self):
        df = pd.DataFrame(
            {
                "mutation_category": ["parameter", "activation"],
                "fitness_before": [1.0, np.nan],
                "fitness_after": [2.0, 3.0],
            }
        )
        out = category_deltas(df)
        assert set(out) == {"parameter"}

    def test_affine_transform_scales_deltas(self):
        out = category_deltas(self._df(), transform=lambda a: 2.0 * a)
        np.testing.assert_allclose(sorted(out["parameter"]), [-2.0, 2.0])

    def test_additive_transform_leaves_deltas_unchanged(self):
        # A pure offset cancels in the difference.
        out = category_deltas(self._df(), transform=lambda a: a + 100.0)
        np.testing.assert_allclose(sorted(out["parameter"]), [-1.0, 1.0])


# --------------------------------------------------------------------------- #
# efficacy_from_deltas / efficacy_by_category                                  #
# --------------------------------------------------------------------------- #
class TestEfficacy:
    def test_empty(self):
        assert efficacy_from_deltas({}) == {}

    def test_win_rate_strict_positive(self):
        out = efficacy_from_deltas({"parameter": np.array([1.0, -1.0, 0.0, 2.0])})
        # 2 of 4 strictly > 0.
        assert out["parameter"]["win_rate"] == pytest.approx(0.5)
        assert out["parameter"]["n"] == 4
        assert out["parameter"]["mean_delta"] == pytest.approx(0.5)

    def test_nan_filtered(self):
        out = efficacy_from_deltas({"parameter": np.array([np.nan, 2.0])})
        assert out["parameter"]["n"] == 1

    def test_all_nan_category_skipped(self):
        out = efficacy_from_deltas({"parameter": np.array([np.nan, np.nan])})
        assert out == {}

    def test_order_follows_canonical(self):
        out = efficacy_from_deltas(
            {"activation": np.array([1.0]), "no mutation": np.array([1.0])}
        )
        assert list(out) == ["no mutation", "activation"]

    def test_returns_python_scalars(self):
        out = efficacy_from_deltas({"parameter": np.array([1.0, 2.0])})
        assert isinstance(out["parameter"]["n"], int)
        assert isinstance(out["parameter"]["win_rate"], float)

    def test_efficacy_by_category_keys_match_actual_not_docstring(self):
        # The docstring promises ci_low/ci_high; the implementation does not
        # produce them (they are added later in plotting). Pin actual behaviour.
        df = pd.DataFrame(
            {
                "mutation_category": ["parameter", "parameter"],
                "fitness_before": [1.0, 1.0],
                "fitness_after": [2.0, 0.5],
            }
        )
        out = efficacy_by_category(df)
        assert set(out["parameter"]) == {"n", "win_rate", "mean_delta"}

    def test_efficacy_by_category_empty(self):
        assert efficacy_by_category(pd.DataFrame()) == {}


# --------------------------------------------------------------------------- #
# lineage_is_informative                                                       #
# --------------------------------------------------------------------------- #
class TestLineageIsInformative:
    def test_none_empty_missing(self):
        assert lineage_is_informative(None) is False
        assert lineage_is_informative(pd.DataFrame()) is False
        assert lineage_is_informative(pd.DataFrame({"generation": [1]})) is False

    def test_only_generation_zero_false(self):
        df = pd.DataFrame(
            {"generation": [0, 0], "parent_id": [0, 1], "agent_id": [0, 1]}
        )
        assert lineage_is_informative(df) is False

    def test_all_self_parent_false(self):
        df = pd.DataFrame(
            {"generation": [0, 1, 1], "parent_id": [0, 0, 1], "agent_id": [0, 0, 1]}
        )
        assert lineage_is_informative(df) is False

    def test_differing_parent_true(self):
        df = pd.DataFrame(
            {"generation": [0, 1], "parent_id": [0, 0], "agent_id": [0, 1]}
        )
        assert lineage_is_informative(df) is True


# --------------------------------------------------------------------------- #
# selection_pressure                                                           #
# --------------------------------------------------------------------------- #
class TestSelectionPressure:
    def test_missing_columns_returns_empty_arrays(self):
        x, v = selection_pressure(pd.DataFrame(), 4)
        assert x.size == 0 and v.size == 0

    def _gen(self, parents, gen=1, step=400):
        n = len(parents)
        return pd.DataFrame(
            {
                "generation": [gen] * n,
                "parent_id": parents,
                "global_step": [step] * n,
            }
        )

    def test_single_lineage_strong_selection(self):
        x, v = selection_pressure(self._gen([7, 7, 7, 7]), 4)
        np.testing.assert_allclose(v, [0.25])

    def test_all_distinct_weak_selection(self):
        x, v = selection_pressure(self._gen([1, 2, 3, 4]), 4)
        np.testing.assert_allclose(v, [1.0])

    def test_two_lineages_half(self):
        x, v = selection_pressure(self._gen([1, 1, 2, 2]), 4)
        np.testing.assert_allclose(v, [0.5])

    def test_x_is_per_agent_step(self):
        x, v = selection_pressure(self._gen([1, 1, 2, 2], step=800), pop_size=4)
        np.testing.assert_allclose(x, [200.0])

    def test_pop_size_zero_clamped(self):
        x, _ = selection_pressure(self._gen([1, 1], step=10), pop_size=0)
        np.testing.assert_allclose(x, [10.0])


# --------------------------------------------------------------------------- #
# turnover                                                                     #
# --------------------------------------------------------------------------- #
class TestTurnover:
    def test_missing_columns_returns_empty(self):
        x, v = turnover(pd.DataFrame(), 4)
        assert x.size == 0 and v.size == 0

    def _gen(self, cats, parents=None, step=100):
        n = len(cats)
        parents = parents if parents is not None else list(range(n))
        return pd.DataFrame(
            {
                "generation": [1] * n,
                "mutation_category": cats,
                "parent_id": parents,
                "agent_id": list(range(n)),
                "global_step": [step] * n,
            }
        )

    def test_all_preserved_zero(self):
        _, v = turnover(self._gen(["no mutation"] * 3), 3)
        np.testing.assert_allclose(v, [0.0])

    def test_all_mutated_one(self):
        _, v = turnover(self._gen(["parameter", "activation"]), 2)
        np.testing.assert_allclose(v, [1.0])

    def test_mixed_fraction(self):
        _, v = turnover(self._gen(["parameter", "no mutation", "no mutation", "x"]), 4)
        np.testing.assert_allclose(v, [0.5])

    def test_unmutated_clone_not_counted_as_turnover(self):
        # parent_id != agent_id but category == "no mutation" -> NOT turnover.
        df = self._gen(["no mutation", "no mutation"], parents=[9, 8])
        _, v = turnover(df, 2)
        np.testing.assert_allclose(v, [0.0])


# --------------------------------------------------------------------------- #
# infer_hp_names                                                               #
# --------------------------------------------------------------------------- #
class TestInferHpNames:
    def test_no_agent0_columns(self):
        df = pd.DataFrame({"train/global_step": [1], "eval/best_fitness": [1.0]})
        assert infer_hp_names(df) == []

    def test_reserved_dropped_and_sorted(self):
        df = pd.DataFrame(
            columns=[
                "train/agent_0/lr",
                "train/agent_0/batch_size",
                "train/agent_0/loss",
                "train/agent_0/fitness",
            ]
        )
        assert infer_hp_names(df) == ["batch_size", "lr"]

    def test_only_agent0_inspected(self):
        df = pd.DataFrame(columns=["train/agent_1/lr", "train/agent_0/gamma"])
        assert infer_hp_names(df) == ["gamma"]


# --------------------------------------------------------------------------- #
# population_hp_trajectory                                                     #
# --------------------------------------------------------------------------- #
class TestPopulationHpTrajectory:
    def test_guards_return_empty(self):
        assert population_hp_trajectory(None, 4) == {}
        assert population_hp_trajectory(pd.DataFrame(), 4) == {}
        assert population_hp_trajectory(pd.DataFrame({"x": [1]}), 4) == {}

    def test_single_agent_band_collapses(self):
        df = pd.DataFrame({"train/global_step": [4, 8], "train/agent_0/lr": [0.1, 0.2]})
        out = population_hp_trajectory(df, pop_size=4, hp_names=["lr"])
        x, lo, med, hi = out["lr"]
        np.testing.assert_allclose(x, [1.0, 2.0])
        np.testing.assert_allclose(lo, med)
        np.testing.assert_allclose(med, hi)
        np.testing.assert_allclose(med, [0.1, 0.2])

    def test_multi_agent_band(self):
        df = pd.DataFrame(
            {
                "train/global_step": [10],
                "train/agent_0/lr": [0.1],
                "train/agent_1/lr": [0.3],
                "train/agent_2/lr": [0.2],
            }
        )
        out = population_hp_trajectory(df, pop_size=1, hp_names=["lr"])
        _, lo, med, hi = out["lr"]
        np.testing.assert_allclose(lo, [0.1])
        np.testing.assert_allclose(med, [0.2])
        np.testing.assert_allclose(hi, [0.3])

    def test_partial_nan_row_uses_nan_reductions(self):
        df = pd.DataFrame(
            {
                "train/global_step": [10],
                "train/agent_0/lr": [0.1],
                "train/agent_1/lr": [np.nan],
            }
        )
        out = population_hp_trajectory(df, pop_size=1, hp_names=["lr"])
        _, lo, med, hi = out["lr"]
        np.testing.assert_allclose(med, [0.1])

    def test_empty_hp_names_returns_empty(self):
        df = pd.DataFrame({"train/global_step": [1], "train/agent_0/lr": [0.1]})
        assert population_hp_trajectory(df, 4, hp_names=[]) == {}


# --------------------------------------------------------------------------- #
# hp_fitness_samples                                                           #
# --------------------------------------------------------------------------- #
class TestHpFitnessSamples:
    def test_guards_return_empty(self):
        assert hp_fitness_samples(None, 4) == {}
        assert hp_fitness_samples(pd.DataFrame(), 4) == {}

    def test_pairs_pooled_across_agents(self):
        df = pd.DataFrame(
            {
                "train/global_step": [4, 8],
                "train/agent_0/lr": [0.1, 0.2],
                "eval/agent_0/fitness": [10.0, 20.0],
                "train/agent_1/lr": [0.3, 0.4],
                "eval/agent_1/fitness": [30.0, 40.0],
            }
        )
        out = hp_fitness_samples(df, pop_size=4, hp_names=["lr"])
        hp_vals, fits, steps = out["lr"]
        assert hp_vals.size == 4
        np.testing.assert_allclose(sorted(hp_vals), [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(sorted(fits), [10.0, 20.0, 30.0, 40.0])
        # steps are per-agent: global_step / pop_size.
        np.testing.assert_allclose(sorted(set(steps)), [1.0, 2.0])

    def test_agent_without_fitness_column_skipped(self):
        df = pd.DataFrame(
            {
                "train/global_step": [4],
                "train/agent_0/lr": [0.1],
                "eval/agent_0/fitness": [10.0],
                "train/agent_1/lr": [0.3],  # no eval/agent_1/fitness
            }
        )
        out = hp_fitness_samples(df, pop_size=4, hp_names=["lr"])
        assert out["lr"][0].size == 1

    def test_non_finite_pairs_dropped(self):
        df = pd.DataFrame(
            {
                "train/global_step": [4, 8],
                "train/agent_0/lr": [0.1, np.nan],
                "eval/agent_0/fitness": [10.0, 20.0],
            }
        )
        out = hp_fitness_samples(df, pop_size=4, hp_names=["lr"])
        np.testing.assert_allclose(out["lr"][0], [0.1])
