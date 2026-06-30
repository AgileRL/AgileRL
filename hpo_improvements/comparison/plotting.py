"""Plots for the benchmark comparison tool.

Two figures, both built with rliable (Agarwal et al. 2021):

* :func:`plot_iqm_difference`: the IQM of the per-pair normalised-fitness
  difference ``f_studied - f_baseline`` over per-agent environment steps
  (``global_steps / pop_size``), with its stratified-bootstrap confidence band.
  A dashed line at zero marks "No preference".
* :func:`plot_probability_of_improvement`: the single ``P(studied > baseline)``
  point estimate with its stratified-bootstrap confidence interval, against the
  50% "No preference" reference.

A third, composite figure overlays the two benchmarks directly:

* :func:`plot_aggregate_and_profile`: a side-by-side figure whose left panel
  overlays the studied and baseline aggregate IQM of the best normalised fitness
  over per-agent environment steps, and whose right panel overlays their
  rliable performance profiles of the final best normalised fitness. Each panel
  has its own ``Studied``/``Baseline`` legend (plus the random/expert references).

All share the harness house style: a full box, a grid, default numeric ticks,
bold titles/labels, a blue main marker, and a top-right legend listing only the
reference line. The studied series is blue and the baseline orange. Text is in
UK English.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from rliable import library as rly
from rliable import plot_utils as rly_plot

# Reuse the harness's rliable IQM-with-CI helper and bootstrap settings so the
# studied/baseline aggregate curves are built exactly like the benchmarks' own.
from hpo_improvements.benchmarking import plotting as bench_plotting

if TYPE_CHECKING:
    from analysis import ComparisonResult

MAIN_COLOR = "tab:blue"

# The two overlaid series in the studied-vs-baseline figure. The studied curve
# keeps the harness's blue; the baseline is drawn in orange for contrast.
STUDIED_COLOR = "tab:blue"
BASELINE_COLOR = "tab:orange"

# Random/expert reference lines, matching the benchmarking harness house style.
RANDOM_COLOR = "red"
EXPERT_COLOR = "green"

X_LABEL = "Per-agent environment steps"


def _finalize_axis(
    ax,
    *,
    legend_handles: list | None = None,
    grid_axis: str = "both",
    legend_loc: str = "upper right",
) -> None:
    """Apply the shared house style to *ax*.

    Restores a default matplotlib box and numeric ticks (undoing rliable's
    despined style), adds a grid, bolds the title and axis labels, and draws a
    top-right legend from *legend_handles* only (the main blue marker/curve is
    left out).

    :param ax: The axes to style.
    :param legend_handles: Reference-line handles for the legend, or None.
    :param grid_axis: Which gridlines to draw (``"both"``/``"x"``/``"y"``).
    :param legend_loc: Location passed to ``ax.legend`` when a legend is drawn.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_position(("outward", 0))
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3.5,
        width=0.8,
        labelsize="medium",
        top=False,
        right=False,
    )
    ax.grid(True, axis=grid_axis, alpha=0.3)
    if ax.title.get_text():
        ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    if legend_handles:
        ax.legend(handles=legend_handles, loc=legend_loc)


def plot_iqm_difference(result: ComparisonResult, out_path: str) -> None:
    """Save the IQM-difference sample-efficiency curve with its CI band.

    :param result: The populated comparison result.
    :param out_path: Destination ``.png`` path.
    """
    if result.x.size == 0:
        return
    label = f"{result.studied_name} - {result.baseline_name}"
    point = {label: result.diff_iqm}
    cis = {label: np.stack([result.diff_ci_low, result.diff_ci_high])}

    fig, ax = plt.subplots(figsize=(10, 6))
    rly_plot.plot_sample_efficiency_curve(
        result.x,
        point,
        cis,
        algorithms=[label],
        colors={label: MAIN_COLOR},
        ax=ax,
        marker="",
        xlabel="Per-agent environment steps",
        ylabel="IQM of the best normalised fitness difference",
        labelsize="medium",
        ticklabelsize="medium",
    )
    ref = ax.axhline(
        0.0, color="grey", linestyle="--", linewidth=1.0, label="No preference"
    )
    ax.set_title(
        f"{result.algo}: IQM of the best normalised fitness difference "
        "(studied-baseline)"
    )
    _finalize_axis(ax, legend_handles=[ref])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability_of_improvement(result: ComparisonResult, out_path: str) -> None:
    """Save a compact interval plot of ``P(studied > baseline)``.

    :param result: The populated comparison result.
    :param out_path: Destination ``.png`` path.
    """
    prob = result.prob_improvement
    lo, hi = result.prob_ci_low, result.prob_ci_high

    fig, ax = plt.subplots(figsize=(8, 3.0))
    # x-axis is a percentage: scale the point and its CI to [0, 100].
    ax.errorbar(
        prob * 100.0,
        0,
        xerr=[[(prob - lo) * 100.0], [(hi - prob) * 100.0]],
        fmt="o",
        color=MAIN_COLOR,
        capsize=6,
        markersize=9,
        elinewidth=2,
    )
    ref = ax.axvline(
        50.0, color="grey", linestyle="--", linewidth=1.0, label="No preference"
    )
    ax.set_xlim(0.0, 100.0)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.set_yticks([])
    ax.set_xlabel("Probability to outperform the baseline")
    # Main title plus a (non-CI) subtitle of the point estimate itself.
    ax.set_title(f"{result.algo}: Probability of improvement", pad=22)
    ax.text(
        0.5,
        1.02,
        f"P({result.studied_name} > {result.baseline_name}) = {prob * 100:.1f}%",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize="small",
    )
    _finalize_axis(ax, legend_handles=[ref], grid_axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_aggregate_panel(ax, result: ComparisonResult) -> None:
    """Draw the overlaid studied/baseline aggregate normalised-fitness curves.

    Left panel of :func:`plot_aggregate_and_profile`. For both the studied and
    the baseline benchmark the IQM of the best normalised fitness is computed
    over **all shared ``(seed, env)`` pairs** at each per-agent step (the same
    rliable IQM-with-CI the harness's own ``plot_aggregate`` uses), and the two
    curves are overlaid with their stratified-bootstrap 95% confidence bands.

    :param ax: The axes to draw on.
    :param result: The populated comparison result.
    """
    if result.x.size == 0 or result.studied_scores.size == 0:
        ax.set_title(f"{result.algo}: Aggregate normalised fitness (no shared range)")
        ax.set_xlabel(X_LABEL)
        _finalize_axis(ax)
        return

    point, cis = bench_plotting._iqm_interval_estimates(
        {"studied": result.studied_scores, "baseline": result.baseline_scores},
        reps=result.reps,
    )
    rly_plot.plot_sample_efficiency_curve(
        result.x,
        point,
        cis,
        algorithms=["studied", "baseline"],
        colors={"studied": STUDIED_COLOR, "baseline": BASELINE_COLOR},
        ax=ax,
        marker="",
        xlabel=X_LABEL,
        ylabel="IQM of the best normalised fitness",
        labelsize="medium",
        ticklabelsize="medium",
    )
    # rliable labels the curves but draws no legend; collect the curve handles
    # and append the random/expert reference lines so the legend lists all four.
    curve_handles, _ = ax.get_legend_handles_labels()
    rnd = ax.axhline(
        0.0, color=RANDOM_COLOR, linestyle="--", linewidth=1.0, label="random"
    )
    exp = ax.axhline(
        1.0, color=EXPERT_COLOR, linestyle="--", linewidth=1.0, label="expert"
    )
    ax.set_title(f"{result.algo}: IQM of the best normalised fitness")
    _finalize_axis(
        ax, legend_handles=[*curve_handles, rnd, exp], legend_loc="lower right"
    )


def _plot_profile_panel(ax, result: ComparisonResult) -> None:
    """Draw the overlaid studied/baseline performance profiles.

    Right panel of :func:`plot_aggregate_and_profile`. From the **final** best
    normalised fitness of every shared ``(seed, env)`` pair (an
    ``(n_runs, n_tasks)`` matrix for each benchmark) the rliable score
    distribution is built: for each threshold ``tau`` the curve is the fraction
    of runs whose final normalised fitness exceeds ``tau``. Studied and baseline
    are overlaid with their stratified-bootstrap 95% confidence bands.

    :param ax: The axes to draw on.
    :param result: The populated comparison result.
    """
    studied_final = result.studied_final
    baseline_final = result.baseline_final
    finite = np.concatenate(
        [
            studied_final[np.isfinite(studied_final)].ravel(),
            baseline_final[np.isfinite(baseline_final)].ravel(),
        ]
    )
    if finite.size == 0:
        ax.set_title(f"{result.algo}: Performance profile (no finite scores)")
        _finalize_axis(ax)
        return

    tau_max = max(1.0, float(np.max(finite)))
    tau_list = np.linspace(0.0, tau_max, 100)

    # create_performance_profile draws from NumPy's global RNG (it has no
    # random_state argument), so seed it to keep the band reproducible.
    np.random.seed(bench_plotting._BOOTSTRAP_SEED)
    distributions, distribution_cis = rly.create_performance_profile(
        {"studied": studied_final, "baseline": baseline_final},
        tau_list,
        reps=result.reps,
    )
    rly_plot.plot_performance_profiles(
        distributions,
        tau_list,
        performance_profile_cis=distribution_cis,
        colors={"studied": STUDIED_COLOR, "baseline": BASELINE_COLOR},
        ax=ax,
        xlabel="Last best normalised fitness",
        ylabel=r"Fraction of runs with $\tau_{run} > \tau$",
        # Match the left panel's axis-label/tick size (rliable defaults to a
        # larger 'x-large' here, which would make the four labels mismatch).
        labelsize="medium",
        ticklabelsize="medium",
    )
    curve_handles, _ = ax.get_legend_handles_labels()
    rnd = ax.axvline(
        0.0, color=RANDOM_COLOR, linestyle="--", linewidth=1.0, label="random"
    )
    exp = ax.axvline(
        1.0, color=EXPERT_COLOR, linestyle="--", linewidth=1.0, label="expert"
    )
    ax.set_title(f"{result.algo}: Performance profile")
    _finalize_axis(ax, legend_handles=[*curve_handles, rnd, exp])


def plot_aggregate_and_profile(result: ComparisonResult, out_path: str) -> None:
    """Save the side-by-side studied-vs-baseline aggregate + profile figure.

    One figure with two panels in the harness house style:

    * **left** -- the aggregate IQM of the best normalised fitness over per-agent
      environment steps, with the studied (blue) and baseline (orange) curves
      overlaid (:func:`_plot_aggregate_panel`);
    * **right** -- the rliable performance profile of the final best normalised
      fitness, again with studied and baseline overlaid
      (:func:`_plot_profile_panel`).

    Each panel carries its own legend listing ``Studied`` and ``Baseline`` (plus
    the random/expert reference lines) so the two benchmarks are directly
    comparable. Text is in UK English.

    :param result: The populated comparison result.
    :param out_path: Destination ``.png`` path.
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
    _plot_aggregate_panel(ax_left, result)
    _plot_profile_panel(ax_right, result)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
