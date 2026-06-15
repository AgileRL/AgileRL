"""Plots for the benchmark comparison tool.

Two figures, both built with rliable (Agarwal et al. 2021):

* :func:`plot_iqm_difference`: the IQM of the per-pair normalised-fitness
  difference ``f_studied - f_baseline`` over per-agent environment steps
  (``global_steps / pop_size``), with its stratified-bootstrap confidence band.
  A dashed line at zero marks "No preference".
* :func:`plot_probability_of_improvement`: the single ``P(studied > baseline)``
  point estimate with its stratified-bootstrap confidence interval, against the
  50% "No preference" reference.

Both share the harness house style: a full box, a grid, default numeric ticks,
bold titles/labels, a blue main marker, and a top-right legend listing only the
reference line. Text is in UK English.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from rliable import plot_utils as rly_plot

if TYPE_CHECKING:
    from analysis import ComparisonResult

MAIN_COLOR = "tab:blue"


def _finalize_axis(
    ax, *, legend_handles: list | None = None, grid_axis: str = "both"
) -> None:
    """Apply the shared house style to *ax*.

    Restores a default matplotlib box and numeric ticks (undoing rliable's
    despined style), adds a grid, bolds the title and axis labels, and draws a
    top-right legend from *legend_handles* only (the main blue marker/curve is
    left out).

    :param ax: The axes to style.
    :param legend_handles: Reference-line handles for the legend, or None.
    :param grid_axis: Which gridlines to draw (``"both"``/``"x"``/``"y"``).
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
        ax.legend(handles=legend_handles, loc="upper right")


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
