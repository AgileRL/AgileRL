"""Plotting helpers for the HPO benchmarking script.

All figures carry a title, a grid, and labelled axes. The x-axis is always
``global_steps / pop_size`` (per-agent environment interactions), per the
benchmark specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from registry import NormalizationScores

GLOBAL_STEP_COL = "train/global_step"
BEST_FITNESS_COL = "eval/best_fitness"


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

    ax1.plot(x, best, color="tab:blue")
    ax1.set_title(f"{env_name} — Best fitness")
    ax1.set_xlabel("global steps / pop size")
    ax1.set_ylabel("Best fitness (episodic return)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, normalized, color="tab:green")
    ax2.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, label="random")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="expert")
    ax2.set_title(f"{env_name} — Best normalized fitness")
    ax2.set_xlabel("global steps / pop size")
    ax2.set_ylabel("Normalized fitness (0=random, 1=expert)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(f"{env_name}: fitness vs. environment interactions")
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
        ax.set_title(f"{env_name} — no hyperparameter schedules logged")
        ax.set_xlabel("global steps / pop size")
        ax.grid(True, alpha=0.3)
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
        ax.plot(x, series, color="tab:purple", drawstyle="steps-post")
        ax.set_title(f"{hp}")
        ax.set_xlabel("global steps / pop size")
        ax.set_ylabel(hp)
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{env_name}: best-agent hyperparameter schedule")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _iqm(values: np.ndarray) -> float:
    """Interquartile mean: mean of values within the [25th, 75th] percentiles."""
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    if valid.size < 4:
        return float(np.mean(valid))
    lo, hi = np.percentile(valid, [25, 75])
    mask = (valid >= lo) & (valid <= hi)
    inner = valid[mask]
    return float(np.mean(inner)) if inner.size else float(np.mean(valid))


def plot_aggregate(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    out_path: str,
    *,
    n_points: int = 200,
) -> None:
    """Save the cross-environment aggregate of best normalized fitness.

    Each environment's normalized-fitness curve is interpolated onto a shared
    x-grid; the front line is the interquartile mean (IQM) across environments
    and the shaded band is the 25th-75th percentile range.

    :param curves: Mapping env_name -> (x, normalized_fitness) arrays.
    :type curves: dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    :param out_path: Destination .png path.
    :type out_path: str
    :param n_points: Number of points on the shared x-grid.
    :type n_points: int
    """
    if not curves:
        return

    x_min = max(float(np.min(x)) for x, _ in curves.values())
    x_max = min(float(np.max(x)) for x, _ in curves.values())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        # Fall back to the union range if envs don't overlap cleanly.
        x_min = min(float(np.min(x)) for x, _ in curves.values())
        x_max = max(float(np.max(x)) for x, _ in curves.values())
    grid = np.linspace(x_min, x_max, n_points)

    stacked = np.full((len(curves), n_points), np.nan)
    for i, (x, y) in enumerate(curves.values()):
        order = np.argsort(x)
        stacked[i] = np.interp(grid, x[order], y[order], left=np.nan, right=np.nan)

    iqm = np.array([_iqm(stacked[:, j]) for j in range(n_points)])
    q25 = np.nanpercentile(stacked, 25, axis=0)
    q75 = np.nanpercentile(stacked, 75, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(grid, q25, q75, alpha=0.25, color="tab:blue", label="IQR (25-75%)")
    ax.plot(grid, iqm, color="tab:blue", linewidth=2, label="IQM")
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(f"Aggregate normalized fitness across {len(curves)} environments")
    ax.set_xlabel("global steps / pop size")
    ax.set_ylabel("Best normalized fitness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
