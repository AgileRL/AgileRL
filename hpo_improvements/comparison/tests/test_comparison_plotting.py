"""Hermetic tests for ``comparison/plotting.py`` (imported via its package path).

``result.reps`` stays small (the fixture default is 10) so the rliable bootstrap
calls these figures make are fast.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from hpo_improvements.comparison import plotting as comp_plotting


def _png_ok(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


# --------------------------------------------------------------------------- #
# _finalize_axis                                                               #
# --------------------------------------------------------------------------- #
def test_finalize_axis_spines_and_bold_labels():
    fig, ax = plt.subplots()
    ax.set_title("My title")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    comp_plotting._finalize_axis(ax)
    for spine in ax.spines.values():
        assert spine.get_visible()
    assert ax.title.get_fontweight() == "bold"
    assert ax.xaxis.label.get_fontweight() == "bold"
    assert ax.yaxis.label.get_fontweight() == "bold"
    # No legend handles passed => no legend drawn.
    assert ax.get_legend() is None
    plt.close(fig)


def test_finalize_axis_with_legend():
    fig, ax = plt.subplots()
    handle = ax.axhline(0.0, label="ref")
    comp_plotting._finalize_axis(ax, legend_handles=[handle])
    assert ax.get_legend() is not None
    plt.close(fig)


# --------------------------------------------------------------------------- #
# plot_iqm_difference                                                          #
# --------------------------------------------------------------------------- #
def test_plot_iqm_difference_writes_png(make_comparison_result, tmp_path):
    result = make_comparison_result()  # x has size 3
    out = tmp_path / "iqm.png"
    comp_plotting.plot_iqm_difference(result, str(out))
    assert _png_ok(str(out))


def test_plot_iqm_difference_skips_when_x_empty(make_comparison_result, tmp_path):
    result = make_comparison_result(
        x=np.array([]),
        diff_iqm=np.array([]),
        diff_ci_low=np.array([]),
        diff_ci_high=np.array([]),
    )
    out = tmp_path / "iqm_empty.png"
    comp_plotting.plot_iqm_difference(result, str(out))
    assert not os.path.exists(out)


# --------------------------------------------------------------------------- #
# plot_probability_of_improvement                                             #
# --------------------------------------------------------------------------- #
def test_plot_probability_of_improvement_writes_png(make_comparison_result, tmp_path):
    result = make_comparison_result()
    out = tmp_path / "prob.png"
    comp_plotting.plot_probability_of_improvement(result, str(out))
    assert _png_ok(str(out))


def test_plot_probability_of_improvement_zero_width_interval(
    make_comparison_result, tmp_path
):
    # ci_low == ci_high == prob (degenerate interval) still writes a PNG.
    result = make_comparison_result(
        prob_improvement=0.5, prob_ci_low=0.5, prob_ci_high=0.5
    )
    out = tmp_path / "prob_zero.png"
    comp_plotting.plot_probability_of_improvement(result, str(out))
    assert _png_ok(str(out))


# --------------------------------------------------------------------------- #
# _plot_aggregate_panel                                                        #
# --------------------------------------------------------------------------- #
def test_aggregate_panel_real_path_draws_lines(make_comparison_result):
    result = make_comparison_result()  # x size 3, studied_scores non-empty
    fig, ax = plt.subplots()
    comp_plotting._plot_aggregate_panel(ax, result)
    # Real path: curves drawn and a legend present.
    assert len(ax.get_lines()) > 0
    assert ax.get_legend() is not None
    assert "no shared range" not in ax.get_title()
    plt.close(fig)


def test_aggregate_panel_placeholder_empty_x(make_comparison_result):
    result = make_comparison_result(
        x=np.array([]),
        studied_scores=np.empty((2, 1, 0)),
        baseline_scores=np.empty((2, 1, 0)),
    )
    fig, ax = plt.subplots()
    comp_plotting._plot_aggregate_panel(ax, result)
    assert "no shared range" in ax.get_title()
    assert len(ax.get_lines()) == 0
    plt.close(fig)


def test_aggregate_panel_placeholder_empty_scores(make_comparison_result):
    # x non-empty but scores empty also triggers the placeholder.
    result = make_comparison_result(studied_scores=np.empty((2, 1, 0)))
    fig, ax = plt.subplots()
    comp_plotting._plot_aggregate_panel(ax, result)
    assert "no shared range" in ax.get_title()
    assert len(ax.get_lines()) == 0
    plt.close(fig)


# --------------------------------------------------------------------------- #
# _plot_profile_panel                                                          #
# --------------------------------------------------------------------------- #
def test_profile_panel_real_path_draws_lines(make_comparison_result):
    result = make_comparison_result()  # finite studied/baseline finals
    fig, ax = plt.subplots()
    comp_plotting._plot_profile_panel(ax, result)
    assert len(ax.get_lines()) > 0
    assert ax.get_legend() is not None
    assert "no finite scores" not in ax.get_title()
    plt.close(fig)


def test_profile_panel_placeholder_all_nan(make_comparison_result):
    result = make_comparison_result(
        studied_final=np.full((1, 2), np.nan),
        baseline_final=np.full((1, 2), np.nan),
    )
    fig, ax = plt.subplots()
    comp_plotting._plot_profile_panel(ax, result)
    assert "no finite scores" in ax.get_title()
    assert len(ax.get_lines()) == 0
    plt.close(fig)


# --------------------------------------------------------------------------- #
# plot_aggregate_and_profile                                                   #
# --------------------------------------------------------------------------- #
def test_plot_aggregate_and_profile_real(make_comparison_result, tmp_path):
    result = make_comparison_result()
    out = tmp_path / "agg_real.png"
    comp_plotting.plot_aggregate_and_profile(result, str(out))
    assert _png_ok(str(out))


def test_plot_aggregate_and_profile_degenerate(make_comparison_result, tmp_path):
    # Fully-degenerate: no shared range AND all-NaN finals.
    result = make_comparison_result(
        x=np.array([]),
        studied_scores=np.empty((2, 1, 0)),
        baseline_scores=np.empty((2, 1, 0)),
        studied_final=np.full((1, 2), np.nan),
        baseline_final=np.full((1, 2), np.nan),
        diff_iqm=np.array([]),
        diff_ci_low=np.array([]),
        diff_ci_high=np.array([]),
    )
    out = tmp_path / "agg_degenerate.png"
    comp_plotting.plot_aggregate_and_profile(result, str(out))
    # Always writes a 2-panel PNG, even fully degenerate.
    assert _png_ok(str(out))
