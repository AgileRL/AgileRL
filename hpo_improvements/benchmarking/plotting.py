"""Plotting helpers for the HPO benchmarking script.

All figures share one house style: a full box (all four spines), a grid,
default-style numeric ticks, bold titles and axis labels, and a top-right legend
that lists only the reference lines (the main data curve is always blue and is
deliberately left out of the legend). The x-axis is always
``global_steps / pop_size`` (per-agent environment steps), per the benchmark
specification. Text is in UK English.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rliable import library as rly  # noqa: E402
from rliable import metrics as rly_metrics  # noqa: E402
from rliable import plot_utils as rly_plot  # noqa: E402

# Make the sibling ``mechanism`` module importable whether this file is loaded as
# a bare script module (``import plotting``, with benchmarking/ on sys.path) or as
# the package submodule ``hpo_improvements.benchmarking.plotting`` (the comparison
# tool's import path, where benchmarking/ is not on sys.path) -- matching the
# harness's documented sys.path convention.
_PLOTTING_DIR = str(Path(__file__).resolve().parent)
if _PLOTTING_DIR not in sys.path:
    sys.path.insert(0, _PLOTTING_DIR)

import mechanism  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd
    from registry import NormalizationScores

GLOBAL_STEP_COL = "train/global_step"
BEST_FITNESS_COL = "eval/best_fitness"
DORMANT_FRACTION_COL = "eval/best_dormant_fraction"

# Population-diversity diagnostics: deliberately-separate [0, 1]-normalised curves
# logged each cycle (marginal HP spread plus its joint effective-dimensionality
# counterpart, architecture, activation). Ordered as (W&B column, label).
HP_DIVERSITY_COL = "eval/hp_diversity"
HP_EFFDIM_COL = "eval/hp_effective_dim"
ARCH_DIVERSITY_COL = "eval/arch_diversity"
ACTIVATION_DIVERSITY_COL = "eval/activation_diversity"
DIVERSITY_SPECS: list[tuple[str, str]] = [
    (HP_DIVERSITY_COL, "Hyperparameter diversity"),
    (HP_EFFDIM_COL, "Hyperparameter effective dimensionality"),
    (ARCH_DIVERSITY_COL, "Architecture diversity"),
    (ACTIVATION_DIVERSITY_COL, "Activation diversity"),
]

# Shared figure styling. The main data curve is always blue; the random/expert
# reference lines are red/green so they are easy to tell apart.
X_LABEL = "Per-agent environment steps"
MAIN_COLOR = "tab:blue"
RANDOM_COLOR = "red"
EXPERT_COLOR = "green"

# Stratified-bootstrap replications for rliable interval estimates. A fixed
# RandomState seed keeps the confidence intervals reproducible across re-plots.
_BOOTSTRAP_REPS = 2000
_BOOTSTRAP_SEED = 0


def _finalize_axis(ax, *, legend_handles: list | None = None) -> None:
    """Apply the shared house style to *ax*.

    Restores a default matplotlib box (all four spines, default-width) and
    default numeric ticks -- undoing rliable's despined style so rliable-drawn
    panels match the plain ``fitness.png`` figures -- adds a grid, makes the
    title and axis labels bold, and, when *legend_handles* are given, draws a
    top-right legend from those handles only (the main blue curve is omitted).

    :param ax: The axes to style.
    :param legend_handles: Reference-line handles to list in the legend, or None
        for no legend.
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
        left=True,
        bottom=True,
    )
    ax.grid(True, alpha=0.3)
    if ax.title.get_text():
        ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")


def _per_agent_x(df: pd.DataFrame, pop_size: int) -> np.ndarray:
    """Return the x-axis array (global_step / pop_size)."""
    return df[GLOBAL_STEP_COL].to_numpy(dtype=float) / max(pop_size, 1)


def plot_fitness(
    df: pd.DataFrame,
    env_name: str,
    scores: NormalizationScores,
    pop_size: int,
    out_path: str,
) -> None:
    """Save a two-panel figure: best fitness and best normalized fitness.

    :param df: W&B history dataframe for the run.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param scores: Random/expert normalization baselines.
    :type scores: NormalizationScores
    :param pop_size: Population size (x-axis divisor).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    """
    data = df.dropna(subset=[GLOBAL_STEP_COL, BEST_FITNESS_COL]).sort_values(
        GLOBAL_STEP_COL
    )
    x = _per_agent_x(data, pop_size)
    best = data[BEST_FITNESS_COL].to_numpy(dtype=float)
    normalized = np.array([scores.normalize(f) for f in best])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, best, color=MAIN_COLOR)
    ax1.set_title(f"{env_name}: Best fitness")
    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel("Best fitness (mean episodic return)")
    _finalize_axis(ax1)

    ax2.plot(x, normalized, color=MAIN_COLOR)
    rnd = ax2.axhline(
        0.0, color=RANDOM_COLOR, linestyle="--", linewidth=1.0, label="random"
    )
    exp = ax2.axhline(
        1.0, color=EXPERT_COLOR, linestyle="--", linewidth=1.0, label="expert"
    )
    ax2.set_title(f"{env_name}: Best normalised fitness")
    ax2.set_xlabel(X_LABEL)
    ax2.set_ylabel("Best normalised fitness")
    _finalize_axis(ax2, legend_handles=[rnd, exp])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dormant_fraction(
    df: pd.DataFrame,
    env_name: str,
    pop_size: int,
    out_path: str,
) -> None:
    """Save the best agent's gradient-dormant-neuron percentage over training.

    Plots the percentage of gradient-dormant neurons of the best individual (the
    GraMa metric, eq. 2 of "Measure gradients, not activations!") against per-agent
    environment steps, in the same house style as :func:`plot_fitness`. A neuron is
    gradient-dormant when its normalised mean absolute pre-activation gradient is
    at or below the threshold. A placeholder figure is emitted when the run logged
    no dormant-neuron values so the artifact always exists.

    :param df: W&B history dataframe for the run.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param pop_size: Population size (x-axis divisor).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if DORMANT_FRACTION_COL in df.columns:
        data = df.dropna(subset=[GLOBAL_STEP_COL, DORMANT_FRACTION_COL]).sort_values(
            GLOBAL_STEP_COL
        )
    else:
        data = df.iloc[0:0]

    if data.empty:
        ax.set_title(f"{env_name}: No dormant-neuron values logged")
        ax.set_xlabel(X_LABEL)
        _finalize_axis(ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    x = _per_agent_x(data, pop_size)
    y = data[DORMANT_FRACTION_COL].to_numpy(dtype=float) * 100.0

    ax.plot(x, y, color=MAIN_COLOR)
    ax.set_title(f"{env_name}: Gradient-dormant neurons (best agent)")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("% gradient-dormant neurons")
    _finalize_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _best_agent_indices(df: pd.DataFrame) -> np.ndarray:
    """Return, per row, the index of the agent with the highest fitness."""
    fit_cols = [
        c for c in df.columns if c.startswith("eval/agent_") and c.endswith("/fitness")
    ]
    if not fit_cols:
        return np.zeros(len(df), dtype=int)
    fit = df[fit_cols].to_numpy(dtype=float)
    fit = np.where(np.isnan(fit), -np.inf, fit)
    # Map argmax column position back to the agent index in the column name.
    agent_ids = [int(c.split("_")[1].split("/")[0]) for c in fit_cols]
    arg = np.argmax(fit, axis=1)
    return np.array([agent_ids[a] for a in arg], dtype=int)


def plot_mutation_schedule(
    df: pd.DataFrame,
    env_name: str,
    pop_size: int,
    out_path: str,
    hp_names: list[str] | None = None,
) -> None:
    """Save the per-cycle hyperparameter schedule of the best agent.

    For each cycle the best agent (highest per-agent fitness) is identified and
    its logged hyperparameters are plotted over time. With no HPO (single,
    never-mutated agent) the schedules are flat. Network-architecture
    characteristics are not logged to W&B and are therefore not plotted.

    :param df: W&B history dataframe for the run.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param pop_size: Population size (x-axis divisor).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    :param hp_names: Explicit mutable-hyperparameter names to plot (from the
        agent's ``hp_config``). If None, they are inferred from the dataframe,
        excluding known per-agent training metrics.
    :type hp_names: list[str] | None
    """
    data = df.dropna(subset=[GLOBAL_STEP_COL]).sort_values(GLOBAL_STEP_COL)
    x = _per_agent_x(data, pop_size)
    best_idx = _best_agent_indices(data)

    if hp_names is None:
        # Infer from agent_0 columns, excluding non-hyperparameter metrics.
        reserved = (
            "fitness",
            "score",
            "local_steps",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy_loss",
            "steps_per_second",
        )
        hp_names = sorted(
            {
                c.split("train/agent_0/")[1]
                for c in data.columns
                if c.startswith("train/agent_0/")
                and c.split("train/agent_0/")[1] not in reserved
            }
        )
    else:
        hp_names = list(hp_names)

    if not hp_names:
        # Nothing to plot; emit a placeholder so the artifact always exists.
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(f"{env_name}: No hyperparameter schedules logged")
        ax.set_xlabel(X_LABEL)
        _finalize_axis(ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    n = len(hp_names)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    rows = np.arange(len(data))
    for ax, hp in zip(axes, hp_names, strict=False):
        series = np.full(len(data), np.nan)
        for r, agent in zip(rows, best_idx, strict=False):
            col = f"train/agent_{agent}/{hp}"
            if col in data.columns:
                series[r] = data.iloc[r][col]
        ax.plot(x, series, color=MAIN_COLOR, drawstyle="steps-post")
        ax.set_title(f"{hp}")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(hp)
        _finalize_axis(ax)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _common_overlap_grid(xs: list[np.ndarray]) -> np.ndarray:
    """A shared, regular grid over the range covered by *every* x array.

    Used to stack curves that do not share exact x-values. The grid spans the
    overlapping range ``[max(min xᵢ), min(max xᵢ)]`` so no curve is ever
    extrapolated, and has as many points as the *coarsest* curve so resolution
    is never fabricated beyond what was actually logged.

    :param xs: List of (non-empty) x arrays.
    :return: The shared grid, or an empty array if the arrays do not overlap.
    """
    if not xs:
        return np.array([])
    lo = max(float(np.min(x)) for x in xs)
    hi = min(float(np.max(x)) for x in xs)
    if not hi > lo:
        return np.array([])
    npts = max(2, min(len(x) for x in xs))
    return np.linspace(lo, hi, npts)


def _align_on_common_x(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``(x, y)`` curves by interpolating onto a shared grid.

    Runs in this benchmark do **not** log at identical timesteps: a run's eval
    timesteps are the cumulative ``global_step`` reached at each eval, which
    drifts seed-to-seed because episode lengths are stochastic (so the
    exact-intersection of timesteps is typically empty). We therefore build one
    grid over the overlapping step range (see :func:`_common_overlap_grid`) and
    linearly interpolate each curve onto it. When every curve already shares the
    same timesteps the interpolation is exact, so no value is altered.

    :param curves: List of ``(x, y)`` arrays.
    :return: ``(grid, stacked)`` where ``grid`` is the shared, sorted x-values
        and ``stacked`` has shape ``(len(curves), len(grid))``. Both are empty
        when the curves share no overlapping range.
    """
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for x, y in curves:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        order = np.argsort(x_arr)
        prepared.append((x_arr[order], y_arr[order]))

    grid = _common_overlap_grid([x for x, _ in prepared])
    if grid.size == 0:
        return np.array([]), np.empty((len(prepared), 0))

    stacked = np.empty((len(prepared), grid.size))
    for i, (x_arr, y_arr) in enumerate(prepared):
        stacked[i] = np.interp(grid, x_arr, y_arr)
    return grid, stacked


def _iqm_interval_estimates(
    score_dict: dict[str, np.ndarray], reps: int = _BOOTSTRAP_REPS
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Rliable IQM point estimates + stratified-bootstrap CIs, per timestep.

    :param score_dict: Mapping label -> scores of shape
        ``(n_runs, n_tasks, n_frames)``. The IQM at each frame treats every
        ``(run, task)`` pair as one sample.
    :param reps: Number of stratified-bootstrap replications.
    :return: ``(point_estimates, interval_estimates)`` as returned by
        :func:`rliable.library.get_interval_estimates`; each interval array has
        shape ``(2, n_frames)`` (lower, upper).
    """

    def iqm(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [rly_metrics.aggregate_iqm(scores[..., f]) for f in range(scores.shape[-1])]
        )

    return rly.get_interval_estimates(
        score_dict,
        iqm,
        reps=reps,
        random_state=np.random.RandomState(_BOOTSTRAP_SEED),
    )


def plot_fitness_over_seeds(
    seed_curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    env_name: str,
    out_path: str,
    *,
    algo_label: str = "Algorithm",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Overlay every seed's fitness curves for one environment.

    Two panels (best fitness, best normalized fitness). Each seed is drawn as a
    light line; on top the interquartile mean (IQM) over seeds and its
    stratified-bootstrap 95% confidence band are drawn with rliable (Agarwal
    et al. 2021). Because all seeds are logged at the same timesteps the curves
    are stacked directly (no interpolation). Saved one level above the per-seed
    folders. The returned per-seed normalized stack is what feeds the
    cross-environment aggregate (over all tasks *and* all seeds).

    :param seed_curves: One ``(x, best_fitness, normalized_fitness)`` per seed.
    :type seed_curves: list[tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]]
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :return: ``(grid, stacked_norm)`` — the shared timesteps and a
        ``(n_seeds, n_frames)`` array of per-seed normalized fitness — for the
        aggregate plot, or None if there is no plottable data.
    :rtype: tuple[numpy.ndarray, numpy.ndarray] | None
    """
    seed_curves = [c for c in seed_curves if c is not None and len(c[0])]
    if not seed_curves:
        return None

    best_curves = [(x, best) for x, best, _ in seed_curves]
    norm_curves = [(x, norm) for x, _, norm in seed_curves]
    grid_best, stacked_best = _align_on_common_x(best_curves)
    grid, stacked_norm = _align_on_common_x(norm_curves)
    if grid.size == 0:
        return None

    # rliable IQM + stratified-bootstrap CIs over seeds (a single task here, so
    # scores have shape (n_seeds, 1, n_frames)).
    pe_best, ci_best = _iqm_interval_estimates({algo_label: stacked_best[:, None, :]})
    pe_norm, ci_norm = _iqm_interval_estimates({algo_label: stacked_norm[:, None, :]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    rly_plot.plot_sample_efficiency_curve(
        grid_best,
        pe_best,
        ci_best,
        algorithms=[algo_label],
        colors={algo_label: MAIN_COLOR},
        ax=ax1,
        marker="",
        xlabel=X_LABEL,
        ylabel="IQM of the best fitness (mean episodic return)",
        labelsize="medium",
        ticklabelsize="medium",
    )
    ax1.set_title(f"{env_name}: IQM of the best fitness")
    _finalize_axis(ax1)

    rly_plot.plot_sample_efficiency_curve(
        grid,
        pe_norm,
        ci_norm,
        algorithms=[algo_label],
        colors={algo_label: MAIN_COLOR},
        ax=ax2,
        marker="",
        xlabel=X_LABEL,
        ylabel="IQM of the best normalised fitness",
        labelsize="medium",
        ticklabelsize="medium",
    )
    rnd = ax2.axhline(
        0.0, color=RANDOM_COLOR, linestyle="--", linewidth=1.0, label="random"
    )
    exp = ax2.axhline(
        1.0, color=EXPERT_COLOR, linestyle="--", linewidth=1.0, label="expert"
    )
    ax2.set_title(f"{env_name}: IQM of the best normalised fitness")
    _finalize_axis(ax2, legend_handles=[rnd, exp])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return grid, stacked_norm


def _aggregate_score_array(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Assemble per-env ``(x, stacked_norm)`` into one rliable score tensor.

    Environments are aligned by interpolating each one's per-seed curves onto a
    shared grid over the overlapping step range (the per-env grids differ — see
    :func:`_align_on_common_x`) and truncated to the smallest shared seed count
    so the result is rectangular.

    :param curves: Mapping env_name -> ``(x, stacked_norm)`` with ``stacked_norm``
        of shape ``(n_seeds, n_frames)`` (per-seed normalized fitness).
    :return: ``(grid, scores)`` where ``scores`` has shape
        ``(n_runs, n_tasks, n_frames)``, or None if there is no overlapping range.
    """
    items = [
        (env, np.asarray(g, dtype=float), np.asarray(s, dtype=float))
        for env, (g, s) in curves.items()
        if np.asarray(g).size and np.asarray(s).size
    ]
    if not items:
        return None

    grid = _common_overlap_grid([g for _, g, _ in items])
    if grid.size == 0:
        return None

    n_runs = min(s.shape[0] for _, _, s in items)
    scores = np.empty((n_runs, len(items), grid.size))
    for t, (_, g, s) in enumerate(items):
        order = np.argsort(g)
        gx = g[order]
        for r in range(n_runs):
            scores[r, t, :] = np.interp(grid, gx, s[r][order])
    return grid, scores


def plot_aggregate(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
    *,
    algo_label: str = "Algorithm",
    suite_name: str = "Environment suite",
) -> None:
    """Save the cross-environment aggregate of best normalised fitness.

    The aggregate IQM is computed with rliable over **all tasks and all seeds**
    at each timestep (every ``(seed, env)`` pair is one sample), and the shaded
    band is the stratified-bootstrap 95% confidence interval around it. There is
    no IQR band.

    :param curves: Mapping env_name -> ``(x, stacked_norm)`` where
        ``stacked_norm`` has shape ``(n_seeds, n_frames)`` (per-seed normalized
        fitness for that environment).
    :type curves: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :param suite_name: Human-readable environment-suite name (for the title).
    :type suite_name: str
    """
    assembled = _aggregate_score_array(curves)
    if assembled is None:
        return
    grid, scores = assembled

    point, cis = _iqm_interval_estimates({algo_label: scores})

    fig, ax = plt.subplots(figsize=(10, 6))
    rly_plot.plot_sample_efficiency_curve(
        grid,
        point,
        cis,
        algorithms=[algo_label],
        colors={algo_label: MAIN_COLOR},
        ax=ax,
        marker="",
        xlabel=X_LABEL,
        ylabel="IQM of the best normalised fitness",
        labelsize="medium",
        ticklabelsize="medium",
    )
    rnd = ax.axhline(
        0.0, color=RANDOM_COLOR, linestyle="--", linewidth=1.0, label="random"
    )
    exp = ax.axhline(
        1.0, color=EXPERT_COLOR, linestyle="--", linewidth=1.0, label="expert"
    )
    ax.set_title(f"{suite_name}: IQM of the best normalised fitness")
    _finalize_axis(ax, legend_handles=[rnd, exp])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_performance_profile(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
    *,
    algo_label: str = "Algorithm",
    suite_name: str = "Environment suite",
) -> None:
    """Save the rliable performance profile of final normalised fitness.

    The score distribution (run-score profile) is built from the **last** value
    of the best normalized fitness of every ``(seed, env)`` pair, i.e. a
    ``(n_runs, n_tasks)`` matrix, and plotted with rliable together with its
    stratified-bootstrap 95% confidence band: for each threshold ``tau`` the
    curve is the fraction of runs whose final normalized fitness exceeds ``tau``.

    :param curves: Mapping env_name -> ``(x, stacked_norm)`` with ``stacked_norm``
        of shape ``(n_seeds, n_frames)``.
    :type curves: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :param suite_name: Human-readable environment-suite name (for the title).
    :type suite_name: str
    """
    assembled = _aggregate_score_array(curves)
    if assembled is None:
        return
    _, scores = assembled
    final = scores[..., -1]  # (n_runs, n_tasks): last normalized fitness

    finite = final[np.isfinite(final)]
    if finite.size == 0:
        return
    tau_max = max(1.0, float(np.max(finite)))
    tau_list = np.linspace(0.0, tau_max, 100)

    # create_performance_profile has no random_state argument (unlike
    # get_interval_estimates); it draws from NumPy's global RNG, so seed that
    # to keep the confidence band reproducible across re-plots.
    np.random.seed(_BOOTSTRAP_SEED)
    distributions, distribution_cis = rly.create_performance_profile(
        {algo_label: final},
        tau_list,
        reps=_BOOTSTRAP_REPS,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    rly_plot.plot_performance_profiles(
        distributions,
        tau_list,
        performance_profile_cis=distribution_cis,
        colors={algo_label: MAIN_COLOR},
        ax=ax,
        xlabel="Last best normalised fitness",
        ylabel=r"Fraction of runs with $\tau_{run} > \tau$",
    )
    rnd = ax.axvline(
        0.0, color=RANDOM_COLOR, linestyle="--", linewidth=1.0, label="random"
    )
    exp = ax.axvline(
        1.0, color=EXPERT_COLOR, linestyle="--", linewidth=1.0, label="expert"
    )
    ax.set_title(f"{suite_name}: Performance profile")
    _finalize_axis(ax, legend_handles=[rnd, exp])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Population-diversity figures
#
# Deliberately-separate normalised-diversity curves (hyperparameter marginal
# spread + effective dimensionality, architecture, activation), each in [0, 1].
# They share the fitness house style
# but omit the random/expert reference lines: diversity is already normalised and
# has no external reference -- 0 is a collapsed population, 1 is maximally spread
# across the mutation search space.
# --------------------------------------------------------------------------- #
def plot_diversity(
    df: pd.DataFrame,
    env_name: str,
    pop_size: int,
    out_path: str,
) -> None:
    """Save the normalised population-diversity curves for one run.

    One panel per diagnostic (hyperparameter spread / effective dimensionality /
    architecture / activation), plotted against per-agent environment steps with a
    fixed ``[0, 1]`` y-axis.
    A panel shows a placeholder when its column was not logged so the artifact
    always exists.

    :param df: W&B history dataframe for the run.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param pop_size: Population size (x-axis divisor).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    """
    fig, axes = plt.subplots(
        1, len(DIVERSITY_SPECS), figsize=(7 * len(DIVERSITY_SPECS), 5)
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, (col, label) in zip(axes, DIVERSITY_SPECS, strict=False):
        data = (
            df.dropna(subset=[GLOBAL_STEP_COL, col]).sort_values(GLOBAL_STEP_COL)
            if col in df.columns
            else df.iloc[0:0]
        )
        if data.empty:
            ax.set_title(f"{env_name}: No {label.lower()} logged")
            ax.set_xlabel(X_LABEL)
            _finalize_axis(ax)
            continue
        ax.plot(
            _per_agent_x(data, pop_size),
            data[col].to_numpy(dtype=float),
            color=MAIN_COLOR,
        )
        ax.set_title(f"{env_name}: {label}")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(label)
        ax.set_ylim(0.0, 1.0)
        _finalize_axis(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_diversity_over_seeds(
    seed_curves: list[tuple[np.ndarray, dict[str, np.ndarray]]],
    env_name: str,
    out_path: str,
    *,
    algo_label: str = "Algorithm",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Overlay every seed's diversity curves for one environment, with IQM + CI.

    One panel per diagnostic. As with :func:`plot_fitness_over_seeds`, all seeds
    are logged at identical timesteps so the curves are stacked directly (no
    interpolation) and reduced to an interquartile mean with a
    stratified-bootstrap 95% confidence band via rliable.

    :param seed_curves: One ``(x, {column: values})`` per seed.
    :type seed_curves: list[tuple[numpy.ndarray, dict[str, numpy.ndarray]]]
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :return: ``{column: (grid, stacked)}`` per diagnostic for the suite aggregate,
        with ``stacked`` of shape ``(n_seeds, n_frames)``.
    :rtype: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    """
    seed_curves = [c for c in seed_curves if c is not None and len(c[0])]
    fig, axes = plt.subplots(
        1, len(DIVERSITY_SPECS), figsize=(7 * len(DIVERSITY_SPECS), 5)
    )
    axes = np.atleast_1d(axes).ravel()
    stacks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ax, (col, label) in zip(axes, DIVERSITY_SPECS, strict=False):
        curves = [(x, d[col]) for x, d in seed_curves if d is not None and col in d]
        grid, stacked = _align_on_common_x(curves) if curves else (np.array([]), None)
        if grid.size == 0 or stacked is None:
            ax.set_title(f"{env_name}: No {label.lower()} logged")
            ax.set_xlabel(X_LABEL)
            _finalize_axis(ax)
            continue
        point, cis = _iqm_interval_estimates({algo_label: stacked[:, None, :]})
        rly_plot.plot_sample_efficiency_curve(
            grid,
            point,
            cis,
            algorithms=[algo_label],
            colors={algo_label: MAIN_COLOR},
            ax=ax,
            marker="",
            xlabel=X_LABEL,
            ylabel=f"IQM of {label.lower()}",
            labelsize="medium",
            ticklabelsize="medium",
        )
        ax.set_title(f"{env_name}: {label}")
        ax.set_ylim(0.0, 1.0)
        _finalize_axis(ax)
        stacks[col] = (grid, stacked)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return stacks


def plot_diversity_aggregate(
    per_metric_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: str,
    *,
    algo_label: str = "Algorithm",
    suite_name: str = "Environment suite",
) -> None:
    """Save the cross-environment aggregate of the diversity diagnostics.

    For each diagnostic the aggregate IQM is computed with rliable over **all
    tasks and all seeds** at each timestep (every ``(seed, env)`` pair is one
    sample), with a stratified-bootstrap 95% confidence band -- mirroring
    :func:`plot_aggregate` but without the random/expert reference lines.

    :param per_metric_curves: ``{column: {env_name: (grid, stacked)}}`` where
        ``stacked`` has shape ``(n_seeds, n_frames)``.
    :type per_metric_curves: dict[str, dict[str, tuple[numpy.ndarray, numpy.ndarray]]]
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :param suite_name: Human-readable environment-suite name (for the title).
    :type suite_name: str
    """
    fig, axes = plt.subplots(
        1, len(DIVERSITY_SPECS), figsize=(7 * len(DIVERSITY_SPECS), 5)
    )
    axes = np.atleast_1d(axes).ravel()
    drew = False
    for ax, (col, label) in zip(axes, DIVERSITY_SPECS, strict=False):
        assembled = _aggregate_score_array(per_metric_curves.get(col, {}))
        if assembled is None:
            ax.set_title(f"{suite_name}: {label}")
            ax.set_xlabel(X_LABEL)
            _finalize_axis(ax)
            continue
        grid, scores = assembled
        point, cis = _iqm_interval_estimates({algo_label: scores})
        rly_plot.plot_sample_efficiency_curve(
            grid,
            point,
            cis,
            algorithms=[algo_label],
            colors={algo_label: MAIN_COLOR},
            ax=ax,
            marker="",
            xlabel=X_LABEL,
            ylabel=f"IQM of {label.lower()}",
            labelsize="medium",
            ticklabelsize="medium",
        )
        ax.set_title(f"{suite_name}: {label}")
        ax.set_ylim(0.0, 1.0)
        _finalize_axis(ax)
        drew = True

    if not drew:
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Evolutionary-mechanism figures
#
# These read the per-generation evolutionary record (``mutation_history.csv``)
# and the per-agent W&B history rather than the aggregate fitness columns, and
# explain *why* an HPO regime behaves as it does: which mutation operators earn
# their keep, how hard selection bites, how fast the population churns, and where
# in hyperparameter space the search travels. The numerical work lives in
# :mod:`mechanism`; these functions only render it, in the shared house style.
# All emit a placeholder figure when their input is missing/degenerate so the
# artifact always exists (matching the fitness/diversity plots).
# --------------------------------------------------------------------------- #
# Colour of the 50%-win-rate reference line on the efficacy figure (purple, so it
# stands apart from the blue bars and the red/green normalisation references).
PCT_LINE_COLOR = "purple"

# Axis labels for the two lineage-based population diagnostics.
SELECTION_LABEL = "Effective surviving-lineage fraction"
TURNOVER_LABEL = "Population turnover"

# Per-category line colours for the mutation-effect distribution figure (one line
# per mutation category). Fixed per category name so a category keeps its colour
# across the per-seed and aggregate figures and across environments; unexpected
# categories fall back to the cycle below.
CATEGORY_COLORS = {
    "no mutation": "tab:gray",
    "parameter": "tab:blue",
    "hyperparameter": "tab:orange",
    "architecture": "tab:green",
    "activation": "tab:purple",
    "other": "tab:brown",
}
_FALLBACK_CATEGORY_COLORS = ["tab:red", "tab:pink", "tab:olive", "tab:cyan"]


def _category_color(category: str, fallback_index: int) -> str:
    """Stable colour for *category* (fixed map, else a fallback cycle)."""
    return CATEGORY_COLORS.get(
        category,
        _FALLBACK_CATEGORY_COLORS[fallback_index % len(_FALLBACK_CATEGORY_COLORS)],
    )


def _mean_interval_estimates(
    values: np.ndarray, reps: int = _BOOTSTRAP_REPS
) -> tuple[float, float, float]:
    """Mean of *values* with a stratified-bootstrap CI computed by rliable.

    The per-application fitness changes are treated as ``(n_samples, 1)`` scores
    (one task) and the mean is bootstrapped with the same rliable machinery and
    seeded ``RandomState`` the harness uses for its IQM curves, so the efficacy
    error bars are reproducible and consistent with the rest of the harness.

    :param values: 1-D array of samples (per-application ``Delta`` fitness).
    :param reps: Stratified-bootstrap replications.
    :return: ``(mean, ci_low, ci_high)``; all NaN for an empty sample, and a
        zero-width interval for a single sample (rliable needs >= 2 to resample).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0])
    point, cis = rly.get_interval_estimates(
        {"_": v.reshape(-1, 1)},
        lambda s: np.array([np.mean(s)]),
        reps=reps,
        random_state=np.random.RandomState(_BOOTSTRAP_SEED),
    )
    return float(point["_"][0]), float(cis["_"][0, 0]), float(cis["_"][1, 0])


def _save_placeholder(out_path: str, title: str, *, figsize=(8, 5)) -> None:
    """Emit a single-axis placeholder figure so the artifact always exists."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(X_LABEL)
    _finalize_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _wide_positive(values: np.ndarray) -> bool:
    """Whether *values* are strictly positive and span more than two decades.

    Used to pick a log scale for hyperparameters such as the learning rate, whose
    population values otherwise crush onto a single line.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return bool(v.size and (v > 0).all() and v.max() / v.min() > 100.0)


def _draw_efficacy(
    axes,
    deltas: dict[str, np.ndarray],
    title_prefix: str,
    *,
    delta_unit: str = "fitness",
) -> None:
    """Draw the win-rate and mean-Delta efficacy panels from raw ``Delta`` arrays.

    Shared by the per-run figure and the suite aggregate so both look identical.
    Every category (including ``no mutation``, the training-only control against
    which a real mutation's marginal effect is read) is drawn in the house blue.
    The mean-change error bars are stratified-bootstrap CIs from rliable
    (:func:`_mean_interval_estimates`). The win-rate is invariant to a positive
    rescaling of the deltas, so only the mean-change panel reflects *delta_unit*.

    :param axes: A length-2 sequence ``(win_rate_axis, mean_delta_axis)``.
    :param deltas: ``{category: deltas}`` per-application fitness changes
        (see :func:`mechanism.category_deltas`).
    :param title_prefix: Title prefix (environment id or suite name).
    :param delta_unit: Units of the change, for the mean-panel label (``"fitness"``
        for raw episodic return, ``"normalised fitness"`` for the suite aggregate).
    """
    ax_win, ax_delta = axes
    stats = mechanism.efficacy_from_deltas(deltas)
    cats = list(stats)
    x = np.arange(len(cats))
    labels = [c.replace(" ", "\n") for c in cats]

    # Win-rate panel.
    win = [stats[c]["win_rate"] for c in cats]
    ax_win.bar(x, win, color=MAIN_COLOR)
    ax_win.set_xticks(x)
    ax_win.set_xticklabels(labels)
    ax_win.set_ylim(0.0, 1.0)
    ax_win.set_ylabel(r"Win-rate $P(\Delta\,\mathrm{fitness} > 0)$")
    ax_win.set_title(f"{title_prefix}: Mutation win-rate")
    pct = ax_win.axhline(
        0.5, color=PCT_LINE_COLOR, linestyle="--", linewidth=1.0, label="50% win-rate"
    )
    _finalize_axis(ax_win, legend_handles=[pct])

    # Mean-Delta panel with rliable stratified-bootstrap CIs.
    estimates = [_mean_interval_estimates(deltas[c]) for c in cats]
    mean = np.array([m for m, _, _ in estimates], dtype=float)
    lo = np.array([lo_ for _, lo_, _ in estimates], dtype=float)
    hi = np.array([hi_ for _, _, hi_ in estimates], dtype=float)
    yerr = np.vstack([mean - lo, hi - mean])
    ax_delta.bar(x, mean, color=MAIN_COLOR, yerr=yerr, capsize=4, ecolor="black")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(labels)
    ax_delta.set_ylabel(rf"Mean $\Delta$ {delta_unit} per mutate-then-train cycle")
    ax_delta.set_title(f"{title_prefix}: Mutation effect")
    zero = ax_delta.axhline(
        0.0, color=PCT_LINE_COLOR, linestyle="--", linewidth=1.0, label="no change"
    )
    _finalize_axis(ax_delta, legend_handles=[zero])


def _efficacy_figure(
    deltas: dict[str, np.ndarray],
    title_prefix: str,
    out_path: str,
    *,
    delta_unit: str = "fitness",
) -> None:
    """Render the two-panel efficacy figure (or a placeholder) from *deltas*."""
    if not mechanism.efficacy_from_deltas(deltas):
        _save_placeholder(out_path, f"{title_prefix}: No mutation efficacy logged")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _draw_efficacy(axes, deltas, title_prefix, delta_unit=delta_unit)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mechanism_efficacy(df: pd.DataFrame, env_name: str, out_path: str) -> None:
    """Save the per-category mutation-efficacy figure for one run.

    Two panels: the win-rate ``P(Delta fitness > 0)`` and the mean fitness change
    per mutate-then-train cycle, by mutation category. The change conflates the
    mutation with one extra generation of training, so a category's *marginal*
    effect is its bar minus the ``no mutation`` control bar. The mean-change error
    bars are rliable stratified-bootstrap 95% confidence intervals.

    :param df: A parsed ``mutation_history.csv`` dataframe.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    """
    _efficacy_figure(mechanism.category_deltas(df), env_name, out_path)


def plot_mechanism_efficacy_over_seeds(
    deltas: dict[str, np.ndarray], env_name: str, out_path: str
) -> None:
    """Save the per-environment efficacy figure, pooling all of an env's seeds.

    Identical to :func:`plot_mechanism_efficacy` but driven by the per-application
    fitness changes pooled across every seed of one environment (so the win-rate
    and mean-change bars and their rliable CIs reflect the whole environment).

    :param deltas: ``{category: deltas}`` pooled across the environment's seeds.
    :type deltas: dict[str, numpy.ndarray]
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    """
    _efficacy_figure(deltas, env_name, out_path)


def _draw_delta_profile(
    ax, deltas: dict[str, np.ndarray], title: str, *, delta_unit: str = "fitness"
) -> None:
    """Draw the per-category probability density of the per-cycle fitness change.

    One Gaussian-KDE curve per mutation category over **plain** ``Delta`` fitness
    on the x-axis: the probability density of
    ``Delta = fitness_after - fitness_before`` for that mutation type (each curve
    integrates to 1). Where a category's mass sits and how widely it spreads shows
    how that operator changes fitness; a purple dashed line marks ``Delta = 0``
    (no change). The x-range is clipped to the central 99% of the pooled changes
    so a few extreme outliers do not crush the bulk of every distribution.

    :param ax: The axes to draw on.
    :param deltas: ``{category: deltas}`` per-application fitness changes.
    :param title: Axes title.
    :param delta_unit: Units of the change, for the x-axis label (``"fitness"`` or
        ``"normalised fitness"``).
    """
    from scipy.stats import gaussian_kde

    cats = list(mechanism.efficacy_from_deltas(deltas))
    arrays = {
        c: np.asarray(deltas[c], dtype=float)[np.isfinite(deltas[c])] for c in cats
    }
    arrays = {c: a for c, a in arrays.items() if a.size}
    if not arrays:
        return
    pooled = np.concatenate(list(arrays.values()))
    lo, hi = float(np.percentile(pooled, 0.5)), float(np.percentile(pooled, 99.5))
    if not hi > lo:
        lo, hi = float(pooled.min()), float(pooled.max())
        if not hi > lo:
            lo, hi = lo - 0.5, hi + 0.5
    grid = np.linspace(lo, hi, 400)

    handles = []
    for i, (cat, a) in enumerate(arrays.items()):
        color = _category_color(cat, i)
        if a.size >= 2 and np.ptp(a) > 0:
            # The KDE can over/underflow harmlessly in the far tails of the grid;
            # silence those numerical warnings (the plotted density is unaffected).
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                density = gaussian_kde(a)(grid)
            (line,) = ax.plot(grid, density, color=color, label=cat)
            handles.append(line)
    nochange = ax.axvline(
        0.0, color=PCT_LINE_COLOR, linestyle="--", linewidth=1.0, label="no change"
    )
    ax.set_xlim(lo, hi)
    ax.set_xlabel(rf"$\Delta$ {delta_unit} per mutate-then-train cycle")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    _finalize_axis(ax, legend_handles=[*handles, nochange])


def _distribution_figure(
    deltas: dict[str, np.ndarray],
    title_prefix: str,
    out_path: str,
    *,
    delta_unit: str = "fitness",
) -> None:
    """Render the mutation-effect distribution figure (or a placeholder)."""
    if not mechanism.efficacy_from_deltas(deltas):
        _save_placeholder(out_path, f"{title_prefix}: No mutation efficacy logged")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_delta_profile(
        ax,
        deltas,
        f"{title_prefix}: Mutation effect distribution",
        delta_unit=delta_unit,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mechanism_efficacy_distribution(
    df: pd.DataFrame, env_name: str, out_path: str
) -> None:
    """Save the per-category mutation-effect distribution figure for one run.

    The distribution counterpart of the right panel of
    :func:`plot_mechanism_efficacy`: instead of one mean bar per category, it
    overlays each category's probability density of the per-cycle fitness change
    (one line per mutation type) -- see :func:`_draw_delta_profile`.

    :param df: A parsed ``mutation_history.csv`` dataframe.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    """
    _distribution_figure(mechanism.category_deltas(df), env_name, out_path)


def plot_mechanism_efficacy_distribution_over_seeds(
    deltas: dict[str, np.ndarray], env_name: str, out_path: str
) -> None:
    """Save the per-environment mutation-effect distribution, pooling all seeds.

    Identical to :func:`plot_mechanism_efficacy_distribution` but driven by the
    per-application fitness changes pooled across every seed of one environment.

    :param deltas: ``{category: deltas}`` pooled across the environment's seeds.
    :type deltas: dict[str, numpy.ndarray]
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    """
    _distribution_figure(deltas, env_name, out_path)


def plot_mechanism_population(
    df: pd.DataFrame, env_name: str, pop_size: int, out_path: str
) -> None:
    """Save the lineage-based population-dynamics figure for one run.

    Two panels over per-agent environment steps, each in ``[0, 1]``: the effective
    surviving-lineage fraction (selection pressure) and the population turnover
    (the fraction of slots that underwent an actual mutation each generation -- a
    ``no mutation`` slot is a preserved agent, see :func:`mechanism.turnover`).
    A placeholder is emitted for regimes whose ``parent_id`` carries no selection
    signal (single-agent no-HPO, or MF-PBT, which keeps every agent as its own
    parent) -- see :func:`mechanism.lineage_is_informative`.

    :param df: A parsed ``mutation_history.csv`` dataframe.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param pop_size: Population size (x-axis divisor).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    """
    if not mechanism.lineage_is_informative(df):
        _save_placeholder(
            out_path,
            f"{env_name}: No lineage-based selection recorded for this regime",
            figsize=(14, 5),
        )
        return
    sel_x, sel_y = mechanism.selection_pressure(df, pop_size)
    tur_x, tur_y = mechanism.turnover(df, pop_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(sel_x, sel_y, color=MAIN_COLOR)
    ax1.set_title(f"{env_name}: Selection pressure")
    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel(SELECTION_LABEL)
    ax1.set_ylim(0.0, 1.0)
    _finalize_axis(ax1)

    ax2.plot(tur_x, tur_y, color=MAIN_COLOR)
    ax2.set_title(f"{env_name}: Population turnover")
    ax2.set_xlabel(X_LABEL)
    ax2.set_ylabel(TURNOVER_LABEL)
    ax2.set_ylim(0.0, 1.0)
    _finalize_axis(ax2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mechanism_population_over_seeds(
    sel_curves: list[tuple[np.ndarray, np.ndarray]],
    tur_curves: list[tuple[np.ndarray, np.ndarray]],
    env_name: str,
    out_path: str,
    *,
    algo_label: str = "Algorithm",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Overlay one environment's seeds for selection pressure and turnover.

    Mirrors :func:`plot_diversity_over_seeds` for the two lineage-based
    population diagnostics: each environment's per-seed curves are reduced to an
    interquartile mean with a stratified-bootstrap 95% confidence band (rliable),
    each panel in ``[0, 1]``. Only lineage-informative seeds are passed in (the
    caller filters them); a panel with no curve shows a placeholder.

    :param sel_curves: One ``(x, selection_pressure)`` per informative seed.
    :type sel_curves: list[tuple[numpy.ndarray, numpy.ndarray]]
    :param tur_curves: One ``(x, turnover)`` per informative seed.
    :type tur_curves: list[tuple[numpy.ndarray, numpy.ndarray]]
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :return: ``{metric: (grid, stacked)}`` per drawn diagnostic for the suite
        aggregate, ``stacked`` of shape ``(n_seeds, n_frames)``.
    :rtype: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    """
    specs = [
        ("selection", SELECTION_LABEL, sel_curves),
        ("turnover", TURNOVER_LABEL, tur_curves),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(7 * len(specs), 5))
    axes = np.atleast_1d(axes).ravel()
    stacks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ax, (key, label, curves) in zip(axes, specs, strict=False):
        grid, stacked = _align_on_common_x(curves) if curves else (np.array([]), None)
        if grid.size == 0 or stacked is None:
            ax.set_title(f"{env_name}: {label}")
            ax.set_xlabel(X_LABEL)
            _finalize_axis(ax)
            continue
        point, cis = _iqm_interval_estimates({algo_label: stacked[:, None, :]})
        rly_plot.plot_sample_efficiency_curve(
            grid,
            point,
            cis,
            algorithms=[algo_label],
            colors={algo_label: MAIN_COLOR},
            ax=ax,
            marker="",
            xlabel=X_LABEL,
            ylabel=f"IQM of {label.lower()}",
            labelsize="medium",
            ticklabelsize="medium",
        )
        ax.set_title(f"{env_name}: {label}")
        ax.set_ylim(0.0, 1.0)
        _finalize_axis(ax)
        stacks[key] = (grid, stacked)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return stacks


def _hp_panel_grid(n: int):
    """Create a (2-column) panel grid sized for *n* hyperparameters."""
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    return fig, np.atleast_1d(axes).ravel()


def plot_population_hp_trajectory(
    df: pd.DataFrame,
    env_name: str,
    pop_size: int,
    out_path: str,
    hp_names: list[str] | None = None,
) -> None:
    """Save the population's per-hyperparameter value band over training.

    One panel per hyperparameter: the population median (blue) with a shaded
    min--max band, in the hyperparameter's *actual* units (a log y-axis is used
    for strictly-positive, multi-decade ranges such as the learning rate). With a
    single agent the band collapses to that agent's trajectory. This complements
    the normalised ``hp_diversity`` curve: it shows *where* the search sits, not
    just how spread it is.

    :param df: W&B history dataframe for the run.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param pop_size: Population size (x-axis divisor).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    :param hp_names: Explicit hyperparameter names; inferred from the dataframe
        when None.
    :type hp_names: list[str] | None
    """
    traj = mechanism.population_hp_trajectory(df, pop_size, hp_names)
    if not traj:
        _save_placeholder(out_path, f"{env_name}: No hyperparameter values logged")
        return

    names = list(traj)
    fig, axes = _hp_panel_grid(len(names))
    for ax, hp in zip(axes, names, strict=False):
        x, lo, med, hi = traj[hp]
        band = ax.fill_between(
            x,
            lo,
            hi,
            color=MAIN_COLOR,
            alpha=0.25,
            step="post",
            label="population min-max",
        )
        (line,) = ax.plot(
            x, med, color=MAIN_COLOR, drawstyle="steps-post", label="population median"
        )
        if _wide_positive(np.concatenate([lo, hi])):
            ax.set_yscale("log")
        ax.set_title(f"{hp}")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(hp)
        _finalize_axis(ax, legend_handles=[line, band])
    for ax in axes[len(names) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hp_fitness(
    df: pd.DataFrame,
    env_name: str,
    pop_size: int,
    out_path: str,
    hp_names: list[str] | None = None,
) -> None:
    """Save the pooled hyperparameter-vs-fitness landscape for one run.

    One panel per hyperparameter: a scatter of every agent's hyperparameter value
    against its evaluation fitness, pooled over all agents and cycles and coloured
    by per-agent environment step (training progress). Reveals which regions of
    each hyperparameter score well and how the search drifts over time. A log
    x-axis is used for strictly-positive, multi-decade ranges.

    :param df: W&B history dataframe for the run.
    :type df: pandas.DataFrame
    :param env_name: Environment id (for titles).
    :type env_name: str
    :param pop_size: Population size (per-agent step normalisation for colour).
    :type pop_size: int
    :param out_path: Destination .png path.
    :type out_path: str
    :param hp_names: Explicit hyperparameter names; inferred when None.
    :type hp_names: list[str] | None
    """
    samples = mechanism.hp_fitness_samples(df, pop_size, hp_names)
    if not samples:
        _save_placeholder(out_path, f"{env_name}: No hyperparameter-fitness data")
        return

    names = list(samples)
    fig, axes = _hp_panel_grid(len(names))
    for ax, hp in zip(axes, names, strict=False):
        hp_vals, fit_vals, steps = samples[hp]
        sc = ax.scatter(
            hp_vals, fit_vals, c=steps, cmap="viridis", s=12, alpha=0.6, linewidths=0
        )
        if _wide_positive(hp_vals):
            ax.set_xscale("log")
        ax.set_title(f"{hp}")
        ax.set_xlabel(hp)
        ax.set_ylabel("Fitness (mean episodic return)")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(X_LABEL)
        _finalize_axis(ax)
    for ax in axes[len(names) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mechanism_efficacy_aggregate(
    pooled_deltas: dict[str, np.ndarray],
    out_path: str,
    *,
    algo_label: str = "Algorithm",
    suite_name: str = "Environment suite",
) -> None:
    """Save the cross-environment aggregate mutation-efficacy figure.

    The per-application changes are pooled across **all seeds and environments**
    per mutation category, then summarised like the per-run figure (win-rate plus
    mean change with an rliable stratified-bootstrap CI). Because environments have
    very different return scales, the pooled changes here are deltas in
    **expert-normalised** fitness (the caller passes normalised deltas) so no
    single environment dominates; the mean-change panel is labelled accordingly.

    :param pooled_deltas: ``{category: deltas}`` of **normalised**-fitness changes
        pooled over every run.
    :type pooled_deltas: dict[str, numpy.ndarray]
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :param suite_name: Human-readable environment-suite name (for the title).
    :type suite_name: str
    """
    _efficacy_figure(
        pooled_deltas,
        suite_name,
        out_path,
        delta_unit="normalised fitness",
    )


def plot_mechanism_efficacy_distribution_aggregate(
    pooled_deltas: dict[str, np.ndarray],
    out_path: str,
    *,
    algo_label: str = "Algorithm",
    suite_name: str = "Environment suite",
) -> None:
    """Save the cross-environment aggregate mutation-effect distribution figure.

    The distribution counterpart of :func:`plot_mechanism_efficacy_aggregate`:
    each mutation category's per-cycle changes, pooled across **all seeds and
    environments**, are shown as a probability density (one line per category).
    As with the aggregate bars, the pooled changes are deltas in
    **expert-normalised** fitness so environments share a comparable scale -- see
    :func:`_draw_delta_profile`.

    :param pooled_deltas: ``{category: deltas}`` of **normalised**-fitness changes
        pooled over every run.
    :type pooled_deltas: dict[str, numpy.ndarray]
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :param suite_name: Human-readable environment-suite name (for the title).
    :type suite_name: str
    """
    _distribution_figure(
        pooled_deltas,
        suite_name,
        out_path,
        delta_unit="normalised fitness",
    )


def plot_mechanism_population_aggregate(
    per_metric_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: str,
    *,
    algo_label: str = "Algorithm",
    suite_name: str = "Environment suite",
) -> None:
    """Save the cross-environment aggregate of the population-dynamics curves.

    For selection pressure and turnover the aggregate IQM is computed with
    rliable over **all tasks and all seeds** at each per-agent step (every
    ``(seed, env)`` pair is one sample), with a stratified-bootstrap 95%
    confidence band -- mirroring :func:`plot_diversity_aggregate`. Only
    lineage-informative seeds reach this function (the caller filters them).

    :param per_metric_curves: ``{metric: {env_name: (grid, stacked)}}`` for
        ``metric`` in ``{"selection", "turnover"}``, ``stacked`` of shape
        ``(n_seeds, n_frames)``.
    :type per_metric_curves: dict[str, dict[str, tuple[numpy.ndarray, numpy.ndarray]]]
    :param out_path: Destination .png path.
    :type out_path: str
    :param algo_label: Display label for the algorithm + HPO method.
    :type algo_label: str
    :param suite_name: Human-readable environment-suite name (for the title).
    :type suite_name: str
    """
    specs = [("selection", SELECTION_LABEL), ("turnover", TURNOVER_LABEL)]
    fig, axes = plt.subplots(1, len(specs), figsize=(7 * len(specs), 5))
    axes = np.atleast_1d(axes).ravel()
    drew = False
    for ax, (key, label) in zip(axes, specs, strict=False):
        assembled = _aggregate_score_array(per_metric_curves.get(key, {}))
        if assembled is None:
            ax.set_title(f"{suite_name}: {label}")
            ax.set_xlabel(X_LABEL)
            _finalize_axis(ax)
            continue
        grid, scores = assembled
        point, cis = _iqm_interval_estimates({algo_label: scores})
        rly_plot.plot_sample_efficiency_curve(
            grid,
            point,
            cis,
            algorithms=[algo_label],
            colors={algo_label: MAIN_COLOR},
            ax=ax,
            marker="",
            xlabel=X_LABEL,
            ylabel=f"IQM of {label.lower()}",
            labelsize="medium",
            ticklabelsize="medium",
        )
        ax.set_title(f"{suite_name}: {label}")
        ax.set_ylim(0.0, 1.0)
        _finalize_axis(ax)
        drew = True

    if not drew:
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
