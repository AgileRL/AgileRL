"""Compare two benchmarks over their shared ``(environment, seed)`` pairs.

Given a *studied* and a *baseline* :class:`~loading.BenchmarkResults` that used
the **same RL algorithm**, this module restricts attention to the environments
and seeds the two share and produces, with rliable (Agarwal et al. 2021):

#. **Probability of improvement** ``P(studied > baseline)`` over the *final* best
   normalized fitness of each shared pair, plus a stratified-bootstrap CI.
#. The **IQM of the per-pair difference** ``f_studied - f_baseline`` at every
   shared x value (``global_steps / pop_size``), plus stratified-bootstrap bands.

Both quantities treat every shared ``(env, seed)`` pair as one sample: seeds are
the *runs* and environments are the *tasks* in rliable's ``(n_runs, n_tasks)``
layout.

**Alignment.** The two benchmarks are aligned **per pair** on a shared per-agent
x-grid before differencing, so the comparison is fair even when the benchmarks
used different population sizes (the x-axis is per-agent interactions by design).
When both benchmarks log at the *same* per-agent steps (e.g. two regimes sharing
a manifest and population size) that shared grid is used exactly; when their
cadences differ (e.g. an HPO population vs. a single no-HPO agent) each curve is
linearly interpolated onto a common grid spanning the overlapping x-range. The
final value used for the probability of improvement is taken at the largest
per-agent step shared by all pairs, so studied and baseline are always compared
at an identical training budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from rliable import library as rly
from rliable import metrics as rly_metrics

# Reuse the harness's rliable IQM-with-CI helper so the difference curve is
# computed the same way the benchmarks' own aggregate curves are.
from hpo_improvements.benchmarking import plotting as bench_plotting

if TYPE_CHECKING:
    from loading import BenchmarkResults

logger = logging.getLogger("hpo_comparison")

# rliable score-dict keys (labels only; the comma form is rliable's pair idiom).
PROB_KEY = "studied,baseline"
DIFF_KEY = "studied - baseline"

# Stratified-bootstrap replications and seed for reproducible interval estimates.
DEFAULT_REPS = 2000
DEFAULT_SEED = 0

# Number of points on the common grid when the benchmarks' native per-agent
# cadences differ and interpolation is required.
DEFAULT_GRID_POINTS = 100

# x-values are matched after rounding to this many decimals (matches the
# harness's own alignment in ``benchmarking/plotting.py``).
_X_DECIMALS = 6


@dataclass
class ComparisonResult:
    """Everything the comparison produces, ready to save and plot."""

    algo: str
    studied_name: str
    baseline_name: str

    # Shared structure used for the analysis.
    common_envs: list[str]
    common_seeds: list[int]
    n_pairs: int

    # Per-pair final best normalized fitness, shape (n_seeds, n_envs).
    studied_final: np.ndarray
    baseline_final: np.ndarray

    # Full per-pair normalized-fitness tensors over the shared x-grid, each of
    # shape (n_seeds, n_envs, n_frames). These feed the side-by-side aggregate
    # (IQM over all pairs per frame) and performance-profile overlay of studied
    # vs baseline. Empty when the pairs share no common per-agent range.
    studied_scores: np.ndarray
    baseline_scores: np.ndarray

    # Probability of improvement P(studied > baseline) and its CI.
    prob_improvement: float
    prob_ci_low: float
    prob_ci_high: float

    # IQM of the per-pair difference over x (empty if no shared x-range).
    x: np.ndarray
    diff_iqm: np.ndarray
    diff_ci_low: np.ndarray
    diff_ci_high: np.ndarray

    # True if the shared grid was built by interpolation (cadences differed).
    interpolated: bool

    reps: int
    confidence_level: float = 0.95


def _common_grid(
    pair_curves: dict[tuple[str, int], tuple],
    keys: list[tuple[str, int]],
    lo: float,
    hi: float,
    grid_points: int,
) -> tuple[np.ndarray, bool]:
    """Return the shared per-agent x-grid for the difference curve.

    Prefers the x-values exactly shared by every pair (faithful, no resampling);
    falls back to a uniform grid over ``[lo, hi]`` when the pairs' native
    cadences do not coincide.

    :param pair_curves: ``(env, seed) -> (xs, ys, xb, yb, lo, hi)``.
    :param keys: The pairs to consider.
    :param lo: Lower bound of the globally overlapping x-range.
    :param hi: Upper bound of the globally overlapping x-range.
    :param grid_points: Number of points for the interpolation fallback.
    :return: ``(grid, interpolated)`` — the sorted grid and whether it required
        interpolation.
    """
    common: set[float] | None = None
    for key in keys:
        xs, _, xb, _, _, _ = pair_curves[key]
        shared = np.intersect1d(np.round(xs, _X_DECIMALS), np.round(xb, _X_DECIMALS))
        shared = shared[(shared >= lo) & (shared <= hi)]
        xs_set = set(shared.tolist())
        common = xs_set if common is None else (common & xs_set)
    if common and len(common) >= 2:
        return np.array(sorted(common)), False
    return np.linspace(lo, hi, grid_points), True


def _iqm_with_ci(
    scores: np.ndarray, reps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame IQM and stratified-bootstrap CI of a score tensor.

    :param scores: Shape ``(n_runs, n_tasks, n_frames)``.
    :param reps: Bootstrap replications.
    :return: ``(iqm, ci_low, ci_high)``, each of shape ``(n_frames,)``.
    """
    point, cis = bench_plotting._iqm_interval_estimates({DIFF_KEY: scores}, reps=reps)
    return point[DIFF_KEY], cis[DIFF_KEY][0], cis[DIFF_KEY][1]


def compare_benchmarks(
    studied: BenchmarkResults,
    baseline: BenchmarkResults,
    *,
    reps: int = DEFAULT_REPS,
    seed: int = DEFAULT_SEED,
    grid_points: int = DEFAULT_GRID_POINTS,
) -> ComparisonResult:
    """Compare *studied* against *baseline* over their shared pairs.

    :param studied: The benchmark whose improvement is being measured.
    :param baseline: The benchmark to improve upon.
    :param reps: Stratified-bootstrap replications.
    :param seed: Bootstrap RNG seed (probability-of-improvement CI).
    :param grid_points: Points on the common grid when cadences differ.
    :return: A populated :class:`ComparisonResult`.
    :raises ValueError: If the algorithms differ or there is no shared pair.
    """
    if studied.algo != baseline.algo:
        msg = (
            f"Algorithm mismatch: studied uses '{studied.algo}', baseline uses "
            f"'{baseline.algo}'. The comparison is only defined for a shared "
            "algorithm."
        )
        raise ValueError(msg)

    common_envs = sorted(studied.envs & baseline.envs)
    if not common_envs:
        msg = "The two benchmarks share no environment."
        raise ValueError(msg)

    # Per-env shared seeds, then the seeds shared by *every* common env so the
    # (run x task) score matrix is rectangular and the seeds are truly paired.
    per_env_seeds = {
        env: studied.seeds(env) & baseline.seeds(env) for env in common_envs
    }
    common_seeds = sorted(set.intersection(*per_env_seeds.values()))
    if not common_seeds:
        msg = (
            "The two benchmarks share no seed common to every shared "
            "environment. Per-environment shared seeds: "
            + ", ".join(f"{env}: {sorted(s)}" for env, s in per_env_seeds.items())
        )
        raise ValueError(msg)

    dropped_seeds = {
        env: sorted(per_env_seeds[env] - set(common_seeds))
        for env in common_envs
        if per_env_seeds[env] - set(common_seeds)
    }
    if dropped_seeds:
        logger.warning(
            "Dropping seeds not shared across all environments so the analysis "
            "is paired and rectangular: %s",
            dropped_seeds,
        )

    # Collect each shared pair's raw curves and overlapping x-range.
    pair_curves: dict[tuple[str, int], tuple] = {}
    for env in common_envs:
        for s in common_seeds:
            cs = studied.curve(env, s)
            cb = baseline.curve(env, s)
            if cs is None or cb is None:
                continue
            xs, ys = cs
            xb, yb = cb
            lo = max(float(xs.min()), float(xb.min()))
            hi = min(float(xs.max()), float(xb.max()))
            if hi <= lo:
                continue  # no overlapping per-agent range
            pair_curves[(env, s)] = (xs, ys, xb, yb, lo, hi)

    # Keep only environments with a usable pair for every shared seed.
    tasks = [
        env for env in common_envs if all((env, s) in pair_curves for s in common_seeds)
    ]
    if not tasks:
        msg = (
            "No environment has an overlapping curve for every shared seed (the "
            "benchmarks may cover disjoint per-agent step ranges)."
        )
        raise ValueError(msg)
    if len(tasks) < len(common_envs):
        logger.warning(
            "Dropping environments without a usable pair for every shared seed: %s",
            sorted(set(common_envs) - set(tasks)),
        )

    keys = [(env, s) for env in tasks for s in common_seeds]
    n_seeds, n_tasks = len(common_seeds), len(tasks)

    # Globally overlapping per-agent x-range across every retained pair.
    g_lo = max(pair_curves[k][4] for k in keys)
    g_hi = min(pair_curves[k][5] for k in keys)

    if g_hi > g_lo:
        grid, interpolated = _common_grid(pair_curves, keys, g_lo, g_hi, grid_points)
        studied_t = np.empty((n_seeds, n_tasks, grid.size))
        baseline_t = np.empty((n_seeds, n_tasks, grid.size))
        for j, env in enumerate(tasks):
            for i, s in enumerate(common_seeds):
                xs, ys, xb, yb, _, _ = pair_curves[(env, s)]
                # np.interp is exact at coincident x, so the exact-grid path
                # incurs no resampling error.
                studied_t[i, j] = np.interp(grid, xs, ys)
                baseline_t[i, j] = np.interp(grid, xb, yb)
        # Final value at the largest shared per-agent step (identical budget).
        studied_final = studied_t[..., -1]
        baseline_final = baseline_t[..., -1]
        studied_scores = studied_t
        baseline_scores = baseline_t
        diff_iqm, diff_lo, diff_hi = _iqm_with_ci(studied_t - baseline_t, reps)
    else:
        # No globally shared range: evaluate the finals at each pair's own last
        # shared step and skip the (now undefined) difference curve.
        logger.warning(
            "Pairs share no common per-agent range across all of them; using "
            "each pair's last shared step for the probability of improvement "
            "and skipping the difference curve."
        )
        interpolated = True
        grid = np.array([])
        studied_final = np.empty((n_seeds, n_tasks))
        baseline_final = np.empty((n_seeds, n_tasks))
        for j, env in enumerate(tasks):
            for i, s in enumerate(common_seeds):
                xs, ys, xb, yb, _, hi = pair_curves[(env, s)]
                studied_final[i, j] = float(np.interp(hi, xs, ys))
                baseline_final[i, j] = float(np.interp(hi, xb, yb))
        studied_scores = baseline_scores = np.empty((n_seeds, n_tasks, 0))
        diff_iqm = diff_lo = diff_hi = np.array([])

    prob, prob_lo, prob_hi = _probability_of_improvement(
        studied_final, baseline_final, reps=reps, seed=seed
    )

    return ComparisonResult(
        algo=studied.algo,
        studied_name=studied.name,
        baseline_name=baseline.name,
        common_envs=tasks,
        common_seeds=common_seeds,
        n_pairs=n_seeds * n_tasks,
        studied_final=studied_final,
        baseline_final=baseline_final,
        studied_scores=studied_scores,
        baseline_scores=baseline_scores,
        prob_improvement=prob,
        prob_ci_low=prob_lo,
        prob_ci_high=prob_hi,
        x=grid,
        diff_iqm=diff_iqm,
        diff_ci_low=diff_lo,
        diff_ci_high=diff_hi,
        interpolated=interpolated,
        reps=reps,
    )


def _probability_of_improvement(
    studied_final: np.ndarray,
    baseline_final: np.ndarray,
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """``P(studied > baseline)`` over tasks with a stratified-bootstrap CI.

    :param studied_final: Final scores, shape ``(n_runs, n_tasks)``.
    :param baseline_final: Final scores, shape ``(n_runs, n_tasks)``.
    :param reps: Bootstrap replications.
    :param seed: Bootstrap RNG seed.
    :return: ``(probability, ci_low, ci_high)``.
    """
    point, cis = rly.get_interval_estimates(
        {PROB_KEY: (studied_final, baseline_final)},
        rly_metrics.probability_of_improvement,
        reps=reps,
        random_state=np.random.RandomState(seed),
    )
    prob = float(np.asarray(point[PROB_KEY]).reshape(-1)[0])
    ci = np.asarray(cis[PROB_KEY]).reshape(2, -1)
    return prob, float(ci[0, 0]), float(ci[1, 0])
