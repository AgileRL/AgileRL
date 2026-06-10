"""Plots for the benchmark comparison tool.

Two figures, both built with rliable (Agarwal et al. 2021):

* :func:`plot_iqm_difference` — the IQM of the per-pair normalized-fitness
  difference ``f_studied - f_baseline`` over per-agent interactions
  (``global_steps / pop_size``), with its stratified-bootstrap confidence band.
  A dashed line at zero marks "no difference".
* :func:`plot_probability_of_improvement` — the single ``P(studied > baseline)``
  point estimate with its stratified-bootstrap confidence interval, against the
  0.5 "no preference" reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from rliable import plot_utils as rly_plot

if TYPE_CHECKING:
    from analysis import ComparisonResult


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
        ax=ax,
        marker="",
        xlabel="global steps / pop size",
        ylabel="IQM of normalized-fitness difference (studied - baseline)",
        labelsize="medium",
        ticklabelsize="medium",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, label="no difference")
    ax.set_title(
        f"{result.algo}: studied vs. baseline improvement\n"
        f"IQM of (studied - baseline) normalized fitness over "
        f"{len(result.common_envs)} tasks × {len(result.common_seeds)} seeds "
        "(95% stratified-bootstrap CI)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
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

    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.errorbar(
        prob,
        0,
        xerr=[[prob - lo], [hi - prob]],
        fmt="o",
        color="tab:blue",
        capsize=6,
        markersize=9,
        elinewidth=2,
    )
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.9, label="no preference")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("P(studied > baseline)")
    ax.set_title(
        f"{result.algo}: probability of improvement\n"
        f"P({result.studied_name} > {result.baseline_name}) = "
        f"{prob:.3f}  [{lo:.3f}, {hi:.3f}]"
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
