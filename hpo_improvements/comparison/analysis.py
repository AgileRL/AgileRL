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
import scipy.stats
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

    # Every shared ``(env, seed)`` pair (env-major) actually used, and the shared
    # seeds per environment. With equal seeds across envs this is the rectangular
    # grid; with uneven seeds it is the full ragged set (no pair is discarded).
    pairs: list[tuple[str, int]]
    per_env_seeds: dict[str, list[int]]

    # Per-pair final best normalized fitness. The 2-D arrays keep the layout each
    # rliable call expects (``(n_seeds, n_envs)`` rectangular, or ``(n_pairs, 1)``
    # pooled in the ragged case); the flat arrays are aligned with ``pairs`` for
    # the per-pair CSV/report regardless of which layout was used.
    studied_final: np.ndarray
    baseline_final: np.ndarray
    studied_final_flat: np.ndarray
    baseline_final_flat: np.ndarray

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

    Every ``(environment, seed)`` pair the two benchmarks share is used. When the
    shared seeds are the **same set in every environment** the analysis is the
    classic rliable rectangular ``(n_seeds, n_envs)`` layout (unchanged
    behaviour). When environments share **different** seed sets — so no single
    rectangular grid exists — the analysis falls back to per-environment
    stratification: the probability of improvement is rliable's per-task
    Mann-Whitney probability averaged over environments (each env over its own
    shared seeds), with a stratified bootstrap that resamples seeds within each
    environment, and the difference/aggregate curves pool all shared pairs. This
    keeps every shared pair instead of intersecting seeds across environments.

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

    per_env_seeds = {
        env: sorted(studied.seeds(env) & baseline.seeds(env)) for env in common_envs
    }
    per_env_seeds = {env: s for env, s in per_env_seeds.items() if s}
    if not per_env_seeds:
        msg = (
            "The two benchmarks share no (environment, seed) pair. "
            "Per-environment seeds: studied "
            + ", ".join(f"{e}: {sorted(studied.seeds(e))}" for e in common_envs)
            + "; baseline "
            + ", ".join(f"{e}: {sorted(baseline.seeds(e))}" for e in common_envs)
        )
        raise ValueError(msg)

    seed_sets = {frozenset(s) for s in per_env_seeds.values()}
    if len(seed_sets) == 1:
        # Equal seeds across all shared envs: the original paired/rectangular path.
        return _compare_rectangular(
            studied,
            baseline,
            common_envs=sorted(per_env_seeds),
            common_seeds=sorted(next(iter(seed_sets))),
            reps=reps,
            seed=seed,
            grid_points=grid_points,
        )

    # Uneven shared seeds across environments: per-environment stratification so
    # no shared pair is thrown away to force a rectangular grid.
    n_shared = sum(len(s) for s in per_env_seeds.values())
    logger.info(
        "Per-environment shared seeds differ; using all %d shared (env, seed) "
        "pairs via per-environment stratification: %s",
        n_shared,
        {e: per_env_seeds[e] for e in sorted(per_env_seeds)},
    )
    return _compare_ragged(
        studied,
        baseline,
        per_env_seeds=per_env_seeds,
        reps=reps,
        seed=seed,
        grid_points=grid_points,
    )


def _collect_pair_curves(
    studied: BenchmarkResults,
    baseline: BenchmarkResults,
    candidate_pairs: list[tuple[str, int]],
) -> dict[tuple[str, int], tuple]:
    """Return ``(env, seed) -> (xs, ys, xb, yb, lo, hi)`` for usable pairs.

    A pair is usable when both benchmarks have a plottable curve for it and the
    two curves share a non-empty per-agent x-range.
    """
    pair_curves: dict[tuple[str, int], tuple] = {}
    for env, s in candidate_pairs:
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
    return pair_curves


def _compare_rectangular(
    studied: BenchmarkResults,
    baseline: BenchmarkResults,
    *,
    common_envs: list[str],
    common_seeds: list[int],
    reps: int,
    seed: int,
    grid_points: int,
) -> ComparisonResult:
    """Paired, rectangular ``(n_seeds, n_envs)`` comparison (equal seeds/env)."""
    candidate = [(env, s) for env in common_envs for s in common_seeds]
    pair_curves = _collect_pair_curves(studied, baseline, candidate)

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

    # Per-pair (env-major) view aligned with the flat finals for the report.
    pairs = [(env, s) for env in tasks for s in common_seeds]
    studied_final_flat = np.array(
        [studied_final[i, j] for j in range(n_tasks) for i in range(n_seeds)]
    )
    baseline_final_flat = np.array(
        [baseline_final[i, j] for j in range(n_tasks) for i in range(n_seeds)]
    )

    return ComparisonResult(
        algo=studied.algo,
        studied_name=studied.name,
        baseline_name=baseline.name,
        common_envs=tasks,
        common_seeds=common_seeds,
        n_pairs=n_seeds * n_tasks,
        pairs=pairs,
        per_env_seeds={env: list(common_seeds) for env in tasks},
        studied_final=studied_final,
        baseline_final=baseline_final,
        studied_final_flat=studied_final_flat,
        baseline_final_flat=baseline_final_flat,
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


def _compare_ragged(
    studied: BenchmarkResults,
    baseline: BenchmarkResults,
    *,
    per_env_seeds: dict[str, list[int]],
    reps: int,
    seed: int,
    grid_points: int,
) -> ComparisonResult:
    """Per-environment-stratified comparison when seeds differ across envs.

    Uses every shared ``(env, seed)`` pair. The probability of improvement is
    rliable's per-task Mann-Whitney probability averaged over environments (each
    over its own shared seeds); the difference/aggregate curves pool all pairs as
    independent runs of a single task.
    """
    candidate = [(env, s) for env in sorted(per_env_seeds) for s in per_env_seeds[env]]
    pair_curves = _collect_pair_curves(studied, baseline, candidate)

    env_seeds_used: dict[str, list[int]] = {}
    for env, s in candidate:
        if (env, s) in pair_curves:
            env_seeds_used.setdefault(env, []).append(s)
    tasks = [env for env in sorted(per_env_seeds) if env_seeds_used.get(env)]
    if not tasks:
        msg = (
            "No (environment, seed) pair has an overlapping per-agent range (the "
            "benchmarks may cover disjoint per-agent step ranges)."
        )
        raise ValueError(msg)
    dropped = {
        env: sorted(set(per_env_seeds[env]) - set(env_seeds_used.get(env, [])))
        for env in per_env_seeds
        if set(per_env_seeds[env]) - set(env_seeds_used.get(env, []))
    }
    if dropped:
        logger.warning(
            "Dropping (env, seed) pairs without an overlapping per-agent range: %s",
            dropped,
        )

    keys = [(env, s) for env in tasks for s in env_seeds_used[env]]  # env-major
    n_pairs = len(keys)

    # Globally overlapping per-agent x-range across every retained pair.
    g_lo = max(pair_curves[k][4] for k in keys)
    g_hi = min(pair_curves[k][5] for k in keys)

    if g_hi > g_lo:
        grid, interpolated = _common_grid(pair_curves, keys, g_lo, g_hi, grid_points)
        # Pool pairs as independent runs of a single task: (n_pairs, 1, n_frames).
        studied_t = np.empty((n_pairs, 1, grid.size))
        baseline_t = np.empty((n_pairs, 1, grid.size))
        for p, (env, s) in enumerate(keys):
            xs, ys, xb, yb, _, _ = pair_curves[(env, s)]
            studied_t[p, 0] = np.interp(grid, xs, ys)
            baseline_t[p, 0] = np.interp(grid, xb, yb)
        studied_final = studied_t[..., -1]
        baseline_final = baseline_t[..., -1]
        studied_scores = studied_t
        baseline_scores = baseline_t
        diff_iqm, diff_lo, diff_hi = _iqm_with_ci(studied_t - baseline_t, reps)
    else:
        logger.warning(
            "Pairs share no common per-agent range across all of them; using "
            "each pair's last shared step for the probability of improvement "
            "and skipping the difference curve."
        )
        interpolated = True
        grid = np.array([])
        studied_final = np.empty((n_pairs, 1))
        baseline_final = np.empty((n_pairs, 1))
        for p, (env, s) in enumerate(keys):
            xs, ys, xb, yb, _, hi = pair_curves[(env, s)]
            studied_final[p, 0] = float(np.interp(hi, xs, ys))
            baseline_final[p, 0] = float(np.interp(hi, xb, yb))
        studied_scores = baseline_scores = np.empty((n_pairs, 1, 0))
        diff_iqm = diff_lo = diff_hi = np.array([])

    # Slice the pooled finals back into per-environment arrays (keys are grouped
    # by env) for the task-stratified probability of improvement.
    env_studied: dict[str, np.ndarray] = {}
    env_baseline: dict[str, np.ndarray] = {}
    idx = 0
    for env in tasks:
        m = len(env_seeds_used[env])
        env_studied[env] = studied_final[idx : idx + m, 0]
        env_baseline[env] = baseline_final[idx : idx + m, 0]
        idx += m

    prob, prob_lo, prob_hi = _probability_of_improvement_stratified(
        env_studied, env_baseline, reps=reps, seed=seed
    )

    return ComparisonResult(
        algo=studied.algo,
        studied_name=studied.name,
        baseline_name=baseline.name,
        common_envs=tasks,
        common_seeds=sorted({s for _, s in keys}),
        n_pairs=n_pairs,
        pairs=keys,
        per_env_seeds={env: list(env_seeds_used[env]) for env in tasks},
        studied_final=studied_final,
        baseline_final=baseline_final,
        studied_final_flat=studied_final[:, 0].copy(),
        baseline_final_flat=baseline_final[:, 0].copy(),
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


def _task_improvement_prob(x: np.ndarray, y: np.ndarray) -> float:
    """rliable's single-task probability that a run of ``x`` exceeds one of ``y``.

    Mirrors :func:`rliable.metrics.probability_of_improvement` for one task: the
    Mann-Whitney ``U`` of ``x`` over ``y`` normalised by ``len(x) * len(y)``
    (0.5 when the two are identical).
    """
    if np.array_equal(x, y):
        return 0.5
    u, _ = scipy.stats.mannwhitneyu(x, y, alternative="greater")
    return float(u) / (len(x) * len(y))


def _probability_of_improvement_stratified(
    env_studied: dict[str, np.ndarray],
    env_baseline: dict[str, np.ndarray],
    *,
    reps: int,
    seed: int,
) -> tuple[float, float, float]:
    """Env-averaged ``P(studied > baseline)`` with a stratified-bootstrap CI.

    The point estimate is rliable's per-task probability of improvement averaged
    over environments (each environment uses its own shared seeds). The 95% CI
    comes from a stratified bootstrap that resamples seeds **within** each
    environment with replacement, recomputing the env-averaged probability each
    replication — the ragged generalisation of rliable's ``StratifiedBootstrap``.

    :param env_studied: env -> studied final scores (1-D, that env's seeds).
    :param env_baseline: env -> baseline final scores (1-D, that env's seeds).
    :param reps: Bootstrap replications.
    :param seed: Bootstrap RNG seed.
    :return: ``(probability, ci_low, ci_high)``.
    """
    envs = list(env_studied)

    def env_averaged(xs: dict[str, np.ndarray], ys: dict[str, np.ndarray]) -> float:
        return float(np.mean([_task_improvement_prob(xs[e], ys[e]) for e in envs]))

    point = env_averaged(env_studied, env_baseline)

    rng = np.random.RandomState(seed)
    boot = np.empty(reps)
    for r in range(reps):
        xs = {}
        ys = {}
        for e in envs:
            x, y = env_studied[e], env_baseline[e]
            xs[e] = x[rng.randint(0, len(x), len(x))]
            ys[e] = y[rng.randint(0, len(y), len(y))]
        boot[r] = env_averaged(xs, ys)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


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
