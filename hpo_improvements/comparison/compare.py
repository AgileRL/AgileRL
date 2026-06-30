"""Compare a studied benchmark against a baseline benchmark.

Interactive / CLI entry point for the comparison tool. It asks for (or accepts on
the command line) the results-folder names of a *studied* and a *baseline*
benchmark, restricts attention to the ``(environment, seed)`` pairs they share
(only when both used the same RL algorithm), and writes, under ``results/``:

* ``probability_of_improvement.json`` — ``P(studied > baseline)`` over final best
  normalized fitness with its stratified-bootstrap CI, plus the shared structure.
* ``final_values.csv`` — the per-pair final normalized fitness (studied, baseline,
  difference) the probability is computed from.
* ``iqm_difference.csv`` — the IQM of the per-pair difference and its CI per x.
* ``iqm_difference.png`` / ``probability_of_improvement.png`` — the two figures.
* ``aggregate_and_profile.png`` — a side-by-side figure overlaying the studied
  and baseline aggregate normalized-fitness curves (left) and their performance
  profiles (right), each with a ``Studied``/``Baseline`` legend.

Run from this folder::

    python compare.py --studied <studied_folder> --baseline <baseline_folder>

Both arguments are names of folders under
``hpo_improvements/benchmarking/results`` (absolute paths also work).
Omitted arguments are prompted for interactively.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running as a plain script: make sibling modules importable so the bare
# ``from loading import ...`` / ``from analysis import ...`` resolve whether this
# file is run as a script or imported as a package module (mirrors the
# benchmarking harness's convention).
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plotting  # noqa: E402
from analysis import DEFAULT_REPS, ComparisonResult, compare_benchmarks  # noqa: E402
from loading import (  # noqa: E402
    BenchmarkResults,
    list_available,
    resolve_results_dir,
)

logger = logging.getLogger("hpo_comparison")

# Benchmarks live under the sibling benchmarking harness; comparison outputs go
# under this package's own ``results`` directory.
BENCH_RESULTS_DIR = SCRIPT_DIR.parent / "benchmarking" / "results"
RESULTS_DIR = SCRIPT_DIR / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Missing values are prompted for interactively."""
    p = argparse.ArgumentParser(description="Compare two HPO benchmarks")
    p.add_argument(
        "--studied",
        help="Studied benchmark results folder (name under the benchmarking "
        "'results' dir, or a path)",
    )
    p.add_argument(
        "--baseline",
        help="Baseline benchmark results folder (name or path)",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Stratified-bootstrap replications (default: 2000)",
    )
    p.add_argument(
        "--out-name",
        default=None,
        help="Output subfolder name under results/ (default: "
        "'<studied>_vs_<baseline>-<timestamp>')",
    )
    return p.parse_args(argv)


def _prompt_benchmark(value: str | None, role: str) -> Path:
    """Resolve a benchmark folder from *value*, prompting (with a listing) if needed."""
    if value is None or str(value) == "":
        available = list(list_available(BENCH_RESULTS_DIR))
        if available:
            print(f"\nAvailable benchmarks under {BENCH_RESULTS_DIR}:")
            for name in available:
                print(f"  - {name}")
        value = input(f"\n{role} benchmark results folder: ").strip()
    return resolve_results_dir(value, BENCH_RESULTS_DIR)


def save_results(result: ComparisonResult, out_dir: Path) -> None:
    """Write the JSON/CSV artifacts and both figures for *result*.

    :param result: The populated comparison result.
    :param out_dir: Destination directory (created if missing).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "algorithm": result.algo,
        "studied": result.studied_name,
        "baseline": result.baseline_name,
        "common_environments": result.common_envs,
        "common_seeds": result.common_seeds,
        "per_environment_seeds": result.per_env_seeds,
        "n_pairs": result.n_pairs,
        "interpolated_grid": result.interpolated,
        "reps": result.reps,
        "confidence_level": result.confidence_level,
        "probability_of_improvement": result.prob_improvement,
        "probability_of_improvement_ci": [
            result.prob_ci_low,
            result.prob_ci_high,
        ],
    }
    with open(out_dir / "probability_of_improvement.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-pair final normalized fitness the probability is computed from. Driven
    # off the per-pair view so it is correct for both the rectangular and the
    # ragged (per-environment-stratified) layouts.
    final_lines = ["env,seed,studied_final,baseline_final,difference"]
    final_lines += [
        f"{env},{seed},{sf},{bf},{sf - bf}"
        for (env, seed), sf, bf in zip(
            result.pairs, result.studied_final_flat, result.baseline_final_flat
        )
    ]
    (out_dir / "final_values.csv").write_text("\n".join(final_lines) + "\n")

    # IQM-of-difference curve, point + CI per x.
    if result.x.size:
        curve_lines = ["x,iqm_difference,ci_low,ci_high"]
        curve_lines += [
            f"{result.x[k]},{result.diff_iqm[k]},"
            f"{result.diff_ci_low[k]},{result.diff_ci_high[k]}"
            for k in range(result.x.size)
        ]
        (out_dir / "iqm_difference.csv").write_text("\n".join(curve_lines) + "\n")

    plotting.plot_iqm_difference(result, str(out_dir / "iqm_difference.png"))
    plotting.plot_probability_of_improvement(
        result, str(out_dir / "probability_of_improvement.png")
    )
    plotting.plot_aggregate_and_profile(
        result, str(out_dir / "aggregate_and_profile.png")
    )


def main(argv: list[str] | None = None) -> None:
    """Run the comparison end to end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    studied_path = _prompt_benchmark(args.studied, "Studied")
    baseline_path = _prompt_benchmark(args.baseline, "Baseline")

    studied = BenchmarkResults(studied_path)
    baseline = BenchmarkResults(baseline_path)

    reps = args.reps or DEFAULT_REPS

    logger.info(
        "Comparing studied '%s' vs baseline '%s' (algorithm: %s)",
        studied.name,
        baseline.name,
        studied.algo,
    )

    result = compare_benchmarks(studied, baseline, reps=reps)

    out_name = args.out_name or (
        f"{studied.name}_vs_{baseline.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir = RESULTS_DIR / out_name
    save_results(result, out_dir)

    logger.info("")
    logger.info("Algorithm:              %s", result.algo)
    logger.info("Common environments:    %s", result.common_envs)
    logger.info("Common seeds:           %s", result.common_seeds)
    logger.info("Shared (env, seed):     %d pairs", result.n_pairs)
    logger.info(
        "P(studied > baseline):  %.4f  (95%% CI [%.4f, %.4f], %d bootstrap reps)",
        result.prob_improvement,
        result.prob_ci_low,
        result.prob_ci_high,
        result.reps,
    )
    logger.info("Results saved to:       %s", out_dir)


if __name__ == "__main__":
    main()
