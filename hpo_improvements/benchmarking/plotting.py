"""Plotting helpers for the HPO benchmarking script.

All figures share one house style: a full box (all four spines), a grid,
default-style numeric ticks, bold titles and axis labels, and a top-right legend
that lists only the reference lines (the main data curve is always blue and is
deliberately left out of the legend). The x-axis is always
``global_steps / pop_size`` (per-agent environment steps), per the benchmark
specification. Text is in UK English.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics as rly_metrics
from rliable import plot_utils as rly_plot

if TYPE_CHECKING:
    import pandas as pd
    from registry import NormalizationScores

GLOBAL_STEP_COL = "train/global_step"
BEST_FITNESS_COL = "eval/best_fitness"
DORMANT_FRACTION_COL = "eval/best_dormant_fraction"

# Population-diversity diagnostics: three deliberately-separate [0, 1]-normalised
# curves logged each cycle. Ordered as (W&B column, human-readable label).
HP_DIVERSITY_COL = "eval/hp_diversity"
ARCH_DIVERSITY_COL = "eval/arch_diversity"
ACTIVATION_DIVERSITY_COL = "eval/activation_diversity"
DIVERSITY_SPECS: list[tuple[str, str]] = [
    (HP_DIVERSITY_COL, "Hyperparameter diversity"),
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
    """Save the best agent's dormant-neuron percentage over training.

    Plots the percentage of dormant neurons of the best individual (Sokar et al.
    2023, Definition 3.1) against per-agent environment steps, in the same house
    style as :func:`plot_fitness`. A placeholder figure is emitted when the run
    logged no dormant-neuron values so the artifact always exists.

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
        ax.set_title(f"{env_name}: no dormant-neuron values logged")
        ax.set_xlabel(X_LABEL)
        _finalize_axis(ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    x = _per_agent_x(data, pop_size)
    y = data[DORMANT_FRACTION_COL].to_numpy(dtype=float) * 100.0

    ax.plot(x, y, color=MAIN_COLOR)
    ax.set_title(f"{env_name}: Dormant neurons (best agent)")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("% dormant neurons")
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
        ax.set_title(f"{env_name}: no hyperparameter schedules logged")
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


def _align_on_common_x(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``(x, y)`` curves reported at the **same timesteps** — no interp.

    Every run in this benchmark logs at identical timesteps (same manifest, eval
    frequency and pop size), so there is no need to interpolate onto an arbitrary
    grid: we take the intersection of the curves' x-values and index each curve
    onto it. This keeps every reported value exact instead of resampling it.

    :param curves: List of ``(x, y)`` arrays.
    :return: ``(grid, stacked)`` where ``grid`` is the shared, sorted x-values
        and ``stacked`` has shape ``(len(curves), len(grid))``. Both are empty
        when the curves share no common timestep.
    """
    keyed: list[tuple[np.ndarray, np.ndarray]] = []
    common: set[float] | None = None
    for x, y in curves:
        order = np.argsort(x)
        xr = np.round(np.asarray(x, dtype=float)[order], 6)
        yr = np.asarray(y, dtype=float)[order]
        keyed.append((xr, yr))
        xs = set(xr.tolist())
        common = xs if common is None else (common & xs)

    if not common:
        return np.array([]), np.empty((len(curves), 0))

    grid = np.array(sorted(common))
    stacked = np.empty((len(keyed), grid.size))
    for i, (xr, yr) in enumerate(keyed):
        idx = {v: j for j, v in enumerate(xr.tolist())}
        stacked[i] = np.array([yr[idx[v]] for v in grid])
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

    Aligns all environments on their common timesteps (no interpolation) and
    truncates to the smallest shared seed count so the result is rectangular.

    :param curves: Mapping env_name -> ``(x, stacked_norm)`` with ``stacked_norm``
        of shape ``(n_seeds, n_frames)`` (per-seed normalized fitness).
    :return: ``(grid, scores)`` where ``scores`` has shape
        ``(n_runs, n_tasks, n_frames)``, or None if there is no shared timestep.
    """
    items = [(env, g, s) for env, (g, s) in curves.items() if g.size and s.size]
    if not items:
        return None

    common: set[float] | None = None
    for _, g, _ in items:
        gs = set(np.round(np.asarray(g, dtype=float), 6).tolist())
        common = gs if common is None else (common & gs)
    if not common:
        return None
    grid = np.array(sorted(common))

    n_runs = min(s.shape[0] for _, _, s in items)
    scores = np.empty((n_runs, len(items), grid.size))
    for t, (_, g, s) in enumerate(items):
        gr = np.round(np.asarray(g, dtype=float), 6)
        idx = {v: j for j, v in enumerate(gr.tolist())}
        cols = [idx[v] for v in grid]
        scores[:, t, :] = s[:n_runs][:, cols]
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
# Three deliberately-separate normalised-diversity curves (hyperparameter,
# architecture, activation), each in [0, 1]. They share the fitness house style
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
    """Save the three normalised population-diversity curves for one run.

    One panel per diagnostic (hyperparameter / architecture / activation),
    plotted against per-agent environment steps with a fixed ``[0, 1]`` y-axis.
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
            ax.set_title(f"{env_name}: no {label.lower()} logged")
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
            ax.set_title(f"{env_name}: no {label.lower()} logged")
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
    """Save the cross-environment aggregate of the three diversity diagnostics.

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
