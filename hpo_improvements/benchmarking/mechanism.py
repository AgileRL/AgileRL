"""Evolutionary-mechanism diagnostics for the HPO benchmarking harness.

This module is the **pure-computation** half of the HPO-mechanism plots: it turns
the per-generation evolutionary record (``mutation_history.csv``, written by
:class:`agilerl.logger.MutationHistoryLogger`) and the W&B history dataframe into
small NumPy structures that :mod:`plotting` renders. It imports no plotting code
so it can be unit-tested on its own (see ``test_mechanism.py``).

The diagnostics derived here answer *why* an HPO regime behaves as it does rather
than merely *how well* it scores:

#. **Mutation efficacy by category** -- for each ``mutation_category`` the raw
   per-application fitness changes ``Delta = fitness_after - fitness_before``
   across a mutate-then-train cycle (:func:`category_deltas`), plus the win-rate
   ``P(Delta > 0)`` and mean ``Delta`` point estimates (:func:`efficacy_from_deltas`).
   These feed both the efficacy bars and the per-category ``Delta`` distribution
   (rliable score-distribution) figures. ``Delta`` conflates the mutation with one
   extra generation of training, so the ``no mutation`` category is the
   training-only **control**: a real category's marginal effect is its ``Delta``
   minus the ``no mutation`` ``Delta``.
#. **Selection pressure** -- the effective number of surviving lineages
   ``exp(H(parent shares))`` divided by the population size, in ``(0, 1]``. A value
   near 1 means weak selection (ancestry stays diverse); near 0 means a single
   lineage dominates (strong selection).
#. **Population turnover** -- the fraction of population slots that underwent an
   actual mutation each generation, in ``[0, 1]``. A ``no mutation`` slot is a
   **preserved** agent (genetically identical to its parent) whether it is an
   elitism survivor or an unmutated clone, so only slots whose
   ``mutation_category != "no mutation"`` count as turned over.
#. **Population hyperparameter trajectory / landscape** -- the *actual* per-agent
   hyperparameter values over training (median plus min/max band) and the pooled
   ``(hyperparameter, fitness)`` scatter, both from the W&B history. These show
   *where* in hyperparameter space the search sits and which regions score well --
   complementary to the normalised ``hp_diversity`` diagnostic, which shows only
   *how spread* the population is.

All confidence intervals on these diagnostics are computed with rliable
(stratified bootstrap), in :mod:`plotting`; this module returns only the raw
samples and point estimates.

Selection pressure is read off the ``parent_id``/``agent_id`` lineage. Some
regimes (single-agent no-HPO, and MF-PBT, which keeps every agent as its own
parent) never reassign parents, so the lineage carries no selection signal;
:func:`lineage_is_informative` detects this and the caller renders an honest
placeholder instead of a misleading flat curve. Mutation efficacy and the
hyperparameter diagnostics remain valid for every regime.

Text is in UK English to match the rest of the harness.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

# W&B history columns reused here (kept in sync with ``plotting``).
GLOBAL_STEP_COL = "train/global_step"

# Mutation-history columns this module relies on.
GENERATION_COL = "generation"
MUT_GLOBAL_STEP_COL = "global_step"
AGENT_ID_COL = "agent_id"
PARENT_ID_COL = "parent_id"
CATEGORY_COL = "mutation_category"
FITNESS_BEFORE_COL = "fitness_before"
FITNESS_AFTER_COL = "fitness_after"

# Display order for mutation categories; any category not listed is appended
# afterwards in alphabetical order so unexpected labels still appear.
CATEGORY_ORDER = (
    "no mutation",
    "parameter",
    "hyperparameter",
    "architecture",
    "activation",
    "other",
)

# The training-only control category (see the module docstring).
CONTROL_CATEGORY = "no mutation"

# Per-agent training metrics that are *not* mutable hyperparameters, excluded
# when hyperparameter names are inferred from the history (matches the reserved
# set used by ``plotting.plot_mutation_schedule``).
_RESERVED_HP_SUFFIXES = frozenset(
    {
        "fitness",
        "score",
        "local_steps",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "steps_per_second",
    }
)


def order_categories(categories: list[str]) -> list[str]:
    """Return *categories* in canonical display order.

    Known categories come first in :data:`CATEGORY_ORDER`; any others follow in
    alphabetical order so unexpected labels are never silently dropped.

    :param categories: The category labels present in the data.
    :return: The same labels, ordered for display.
    """
    known = [c for c in CATEGORY_ORDER if c in categories]
    extra = sorted(c for c in categories if c not in CATEGORY_ORDER)
    return known + extra


def category_deltas(
    mut_df: pd.DataFrame,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Per-category arrays of the fitness change across a mutate-then-train cycle.

    Only rows with a finite ``fitness_before`` *and* ``fitness_after`` contribute
    (generation 0 has no parent fitness, so it is excluded automatically). The
    change is ``fitness_after - fitness_before``.

    :param mut_df: A parsed ``mutation_history.csv`` dataframe.
    :param transform: Optional callable applied (vectorised) to the raw
        ``fitness_before`` and ``fitness_after`` arrays *before* differencing, so
        the deltas come out in transformed units. Pass an environment's
        expert-normalisation (``NormalizationScores.normalize``) to express the
        change as a delta in **normalised** fitness, which makes deltas comparable
        across environments. ``None`` (default) leaves the raw episodic return.
    :return: ``{category: deltas}`` with one ``deltas`` array per category that
        has at least one usable row (categories with none are omitted).
    """
    needed = {CATEGORY_COL, FITNESS_BEFORE_COL, FITNESS_AFTER_COL}
    if mut_df is None or mut_df.empty or not needed.issubset(mut_df.columns):
        return {}
    before = mut_df[FITNESS_BEFORE_COL].to_numpy(dtype=float)
    after = mut_df[FITNESS_AFTER_COL].to_numpy(dtype=float)
    finite = np.isfinite(before) & np.isfinite(after)
    if not finite.any():
        return {}
    if transform is not None:
        # The normalisation is affine, so the finiteness computed on the raw
        # values is preserved; transform the full arrays and mask afterwards.
        before = np.asarray(transform(before), dtype=float)
        after = np.asarray(transform(after), dtype=float)
    cats = mut_df[CATEGORY_COL].astype(str).to_numpy()
    delta = after - before
    out: dict[str, np.ndarray] = {}
    for cat in np.unique(cats[finite]):
        mask = finite & (cats == cat)
        out[str(cat)] = delta[mask]
    return out


def efficacy_from_deltas(
    deltas: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """Win-rate and mean fitness change per category from raw ``Delta`` arrays.

    Shared by the per-seed figure (deltas from one run) and the suite aggregate
    (deltas pooled across seeds and environments). Only the point estimates are
    returned here; the bootstrap confidence interval drawn on the mean-change bars
    is computed in :mod:`plotting` with rliable, working from the same raw
    ``deltas`` arrays.

    :param deltas: ``{category: deltas}`` arrays of fitness changes.
    :return: ``{category: {"n", "win_rate", "mean_delta"}}`` in canonical display
        order. Empty when *deltas* is empty.
    """
    out: dict[str, dict[str, float]] = {}
    for cat in order_categories(list(deltas)):
        d = np.asarray(deltas[cat], dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        out[cat] = {
            "n": int(d.size),
            "win_rate": float((d > 0.0).mean()),
            "mean_delta": float(d.mean()),
        }
    return out


def efficacy_by_category(mut_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Win-rate and mean fitness change per mutation category for one run.

    :param mut_df: A parsed ``mutation_history.csv`` dataframe.
    :return: ``{category: {"n", "win_rate", "mean_delta", "ci_low", "ci_high"}}``
        in canonical display order. Empty when no row is usable.
    """
    return efficacy_from_deltas(category_deltas(mut_df))


def _per_generation_steps(mut_df: pd.DataFrame, pop_size: int) -> np.ndarray:
    """Per-agent environment steps at each generation (sorted by generation).

    Every row in a generation shares the generation's ``global_step``, so the
    first value per generation is exact. Divided by the population size to match
    the harness's per-agent x-axis.
    """
    grouped = mut_df.groupby(GENERATION_COL)[MUT_GLOBAL_STEP_COL].first().sort_index()
    return grouped.to_numpy(dtype=float) / max(pop_size, 1)


def lineage_is_informative(mut_df: pd.DataFrame) -> bool:
    """Whether ``parent_id`` carries any selection signal.

    True when at least one post-initial-generation slot was filled by an
    individual whose parent differs from itself. False for single-agent no-HPO
    runs and for regimes (e.g. MF-PBT) that keep every agent as its own parent,
    where the selection-pressure and turnover curves would be degenerate.

    :param mut_df: A parsed ``mutation_history.csv`` dataframe.
    :return: True if lineage-based selection metrics are meaningful.
    """
    needed = {GENERATION_COL, PARENT_ID_COL, AGENT_ID_COL}
    if mut_df is None or mut_df.empty or not needed.issubset(mut_df.columns):
        return False
    later = mut_df[mut_df[GENERATION_COL] > 0]
    if later.empty:
        return False
    return bool((later[PARENT_ID_COL] != later[AGENT_ID_COL]).any())


def selection_pressure(
    mut_df: pd.DataFrame, pop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Effective surviving-lineage fraction per generation, in ``(0, 1]``.

    For each generation the share of the population descended from each parent is
    formed, and the effective number of lineages ``exp(H)`` (``H`` the Shannon
    entropy of those shares) is divided by the population size. A value near 1 is
    weak selection (diverse ancestry); near 0 is strong selection (one lineage
    dominates).

    :param mut_df: A parsed ``mutation_history.csv`` dataframe.
    :param pop_size: Population size (per-agent x-axis divisor).
    :return: ``(x, values)`` over per-agent environment steps. Empty arrays when
        the required columns are absent.
    """
    needed = {GENERATION_COL, PARENT_ID_COL, MUT_GLOBAL_STEP_COL}
    if mut_df is None or mut_df.empty or not needed.issubset(mut_df.columns):
        return np.array([]), np.array([])
    values: list[float] = []
    for _, grp in mut_df.groupby(GENERATION_COL):
        parents = grp[PARENT_ID_COL].to_numpy()
        n = parents.size
        if n == 0:
            values.append(float("nan"))
            continue
        _, counts = np.unique(parents, return_counts=True)
        p = counts / counts.sum()
        entropy = -(p * np.log(p)).sum()
        values.append(float(np.exp(entropy) / n))
    return _per_generation_steps(mut_df, pop_size), np.asarray(values, dtype=float)


def turnover(mut_df: pd.DataFrame, pop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of population slots genuinely changed each generation, in ``[0, 1]``.

    A slot counts as turned over only when it underwent an actual mutation
    (``mutation_category != "no mutation"``). A ``no mutation`` slot is a
    **preserved** agent -- genetically identical to its parent -- whether it is an
    elitism survivor (``parent_id == agent_id``) or an unmutated clone of a
    tournament-selected parent (``parent_id != agent_id``); both are excluded from
    turnover.

    :param mut_df: A parsed ``mutation_history.csv`` dataframe.
    :param pop_size: Population size (per-agent x-axis divisor).
    :return: ``(x, values)`` over per-agent environment steps. Empty arrays when
        the required columns are absent.
    """
    needed = {GENERATION_COL, CATEGORY_COL, MUT_GLOBAL_STEP_COL}
    if mut_df is None or mut_df.empty or not needed.issubset(mut_df.columns):
        return np.array([]), np.array([])
    values: list[float] = []
    for _, grp in mut_df.groupby(GENERATION_COL):
        mutated = (grp[CATEGORY_COL].astype(str) != CONTROL_CATEGORY).to_numpy()
        values.append(float(mutated.mean()) if mutated.size else float("nan"))
    return _per_generation_steps(mut_df, pop_size), np.asarray(values, dtype=float)


def infer_hp_names(history_df: pd.DataFrame) -> list[str]:
    """Infer mutable-hyperparameter names from per-agent W&B history columns.

    Mirrors the inference in ``plotting.plot_mutation_schedule``: takes the
    suffixes of ``train/agent_0/*`` columns and drops the reserved per-agent
    training metrics (losses, score, step counters).

    :param history_df: W&B history dataframe for a run.
    :return: Sorted hyperparameter names (possibly empty).
    """
    prefix = "train/agent_0/"
    names = {
        c[len(prefix) :]
        for c in history_df.columns
        if c.startswith(prefix) and c[len(prefix) :] not in _RESERVED_HP_SUFFIXES
    }
    return sorted(names)


def _agent_indices(history_df: pd.DataFrame, hp: str) -> list[int]:
    """Agent indices that have a ``train/agent_<i>/<hp>`` column, sorted."""
    pat = re.compile(rf"^train/agent_(\d+)/{re.escape(hp)}$")
    idx = {int(m.group(1)) for c in history_df.columns if (m := pat.match(c))}
    return sorted(idx)


def population_hp_trajectory(
    history_df: pd.DataFrame,
    pop_size: int,
    hp_names: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per-hyperparameter population band (min/median/max) over training.

    For each hyperparameter the per-agent columns are stacked and reduced, row by
    row, to the population minimum, median and maximum -- the *actual* values
    (e.g. a learning rate of 3e-4), not a normalised spread. With a single agent
    the band collapses to that agent's trajectory.

    :param history_df: W&B history dataframe for a run.
    :param pop_size: Population size (per-agent x-axis divisor).
    :param hp_names: Hyperparameter names to use; inferred when None.
    :return: ``{hp: (x, lo, median, hi)}`` for each hyperparameter with data.
    """
    if history_df is None or history_df.empty or GLOBAL_STEP_COL not in history_df:
        return {}
    if hp_names is None:
        hp_names = infer_hp_names(history_df)
    data = history_df.dropna(subset=[GLOBAL_STEP_COL]).sort_values(GLOBAL_STEP_COL)
    x = data[GLOBAL_STEP_COL].to_numpy(dtype=float) / max(pop_size, 1)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for hp in hp_names:
        cols = [f"train/agent_{i}/{hp}" for i in _agent_indices(data, hp)]
        if not cols:
            continue
        mat = data[cols].to_numpy(dtype=float)  # (n_rows, n_agents)
        if not np.isfinite(mat).any():
            continue
        lo = np.nanmin(mat, axis=1)
        med = np.nanmedian(mat, axis=1)
        hi = np.nanmax(mat, axis=1)
        valid = np.isfinite(med)
        if not valid.any():
            continue
        out[hp] = (x[valid], lo[valid], med[valid], hi[valid])
    return out


def hp_fitness_samples(
    history_df: pd.DataFrame,
    pop_size: int,
    hp_names: list[str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pooled ``(hyperparameter value, agent fitness, per-agent step)`` samples.

    For every agent and every logged cycle the agent's hyperparameter value is
    paired with its evaluation fitness (``eval/agent_<i>/fitness``), pooling all
    agents and cycles into one cloud per hyperparameter. The per-agent step is
    returned too so the scatter can be coloured by training progress.

    :param history_df: W&B history dataframe for a run.
    :param pop_size: Population size (per-agent x-axis divisor).
    :param hp_names: Hyperparameter names to use; inferred when None.
    :return: ``{hp: (hp_values, fitnesses, steps)}`` for each hyperparameter with
        at least one paired sample.
    """
    if history_df is None or history_df.empty or GLOBAL_STEP_COL not in history_df:
        return {}
    if hp_names is None:
        hp_names = infer_hp_names(history_df)
    data = history_df.dropna(subset=[GLOBAL_STEP_COL]).sort_values(GLOBAL_STEP_COL)
    step = data[GLOBAL_STEP_COL].to_numpy(dtype=float) / max(pop_size, 1)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for hp in hp_names:
        hp_vals: list[np.ndarray] = []
        fit_vals: list[np.ndarray] = []
        step_vals: list[np.ndarray] = []
        for i in _agent_indices(data, hp):
            fit_col = f"eval/agent_{i}/fitness"
            if fit_col not in data.columns:
                continue
            h = data[f"train/agent_{i}/{hp}"].to_numpy(dtype=float)
            f = data[fit_col].to_numpy(dtype=float)
            m = np.isfinite(h) & np.isfinite(f)
            if not m.any():
                continue
            hp_vals.append(h[m])
            fit_vals.append(f[m])
            step_vals.append(step[m])
        if not hp_vals:
            continue
        out[hp] = (
            np.concatenate(hp_vals),
            np.concatenate(fit_vals),
            np.concatenate(step_vals),
        )
    return out
