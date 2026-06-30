"""Shared fixtures and import setup for the comparison test suite.

The comparison package mixes import styles: ``loading``/``analysis`` reach the
benchmarking harness through the *package* path
``hpo_improvements.benchmarking.plotting``, while ``compare.py`` uses the bare
``import plotting`` (its sibling ``comparison/plotting.py``). The bare name
``plotting`` is therefore contested with the benchmarking harness's own
``plotting`` module.

To make ``compare`` importable correctly regardless of what a shared xdist worker
loaded first, we import ``compare`` here under a scoped ``sys.modules['plotting']``
hijack (swap in comparison's plotting, import, restore). Once ``compare`` is
imported its module-global ``plotting`` is bound permanently, so the restore is
safe. This also means the comparison stack stays light: it pulls in
matplotlib/rliable/scipy/numpy/pandas but *not* torch/agilerl.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# --- bare-import path setup ------------------------------------------------- #
_COMPARISON_DIR = Path(__file__).resolve().parent.parent
_BENCH_DIR = _COMPARISON_DIR.parent / "benchmarking"
_REPO_ROOT = _COMPARISON_DIR.parent.parent
for _p in (str(_COMPARISON_DIR), str(_BENCH_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- import `compare` under a scoped bare-`plotting` hijack ----------------- #
# Importing comparison.plotting via its package path does NOT claim the bare
# `plotting` name; we temporarily point bare `plotting` at it so compare.py's
# `import plotting` resolves to the comparison sibling, then restore.
import hpo_improvements.comparison.plotting as _comp_plotting  # noqa: E402

_saved_plotting = sys.modules.get("plotting")
sys.modules["plotting"] = _comp_plotting
try:
    import compare  # noqa: E402,F401  (cached so test modules get the right one)
finally:
    if _saved_plotting is not None:
        sys.modules["plotting"] = _saved_plotting
    else:
        sys.modules.pop("plotting", None)


# --------------------------------------------------------------------------- #
# Global-state isolation                                                       #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_global_state():
    env_snapshot = dict(os.environ)
    np_state = np.random.get_state()
    yield
    os.environ.clear()
    os.environ.update(env_snapshot)
    np.random.set_state(np_state)
    plt.close("all")


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_benchmark_dir():
    """Return a builder for on-disk benchmark-results folders.

    ``runs`` maps ``(env, seed)`` to a list of ``eval/best_fitness`` values; the
    matching ``train/global_step`` column is auto-generated. ``layout`` chooses
    the multi-seed (``s<seed>/`` subdirs) or single-seed (flat) directory shape.
    """
    import pandas as pd
    import yaml

    def _build(
        root: Path,
        *,
        algo: str = "PPO",
        pop_size: int = 4,
        name: str = "bench",
        runs: dict | None = None,
        layout: str = "multi",
        seed_for_single: int | None = None,
    ) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        config = {
            "algorithm": {"name": algo},
            "training": {"pop_size": pop_size},
            "benchmark": {"name": name},
        }
        if seed_for_single is not None:
            config["benchmark"]["seed"] = seed_for_single
        (root / "config.yaml").write_text(yaml.safe_dump(config))

        runs = runs or {("Ant-v4", 42): [1.0, 2.0, 3.0]}
        for (env, seed), fitness in runs.items():
            steps = list(range(1, len(fitness) + 1))
            df = pd.DataFrame(
                {"train/global_step": steps, "eval/best_fitness": fitness}
            )
            if layout == "multi":
                d = root / env / f"s{seed}"
            else:
                d = root / env
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / "wandb_history.csv", index=False)
        return root

    return _build


@pytest.fixture
def make_comparison_result():
    """Return a builder for synthetic ``ComparisonResult`` objects."""
    from analysis import ComparisonResult

    def _build(**overrides):
        defaults = dict(
            algo="PPO",
            studied_name="studied",
            baseline_name="baseline",
            common_envs=["Ant-v4"],
            common_seeds=[1, 2],
            n_pairs=2,
            pairs=[("Ant-v4", 1), ("Ant-v4", 2)],
            per_env_seeds={"Ant-v4": [1, 2]},
            studied_final=np.array([[0.6, 0.7]]),
            baseline_final=np.array([[0.4, 0.5]]),
            studied_final_flat=np.array([0.6, 0.7]),
            baseline_final_flat=np.array([0.4, 0.5]),
            studied_scores=np.ones((2, 1, 3)),
            baseline_scores=np.full((2, 1, 3), 0.5),
            prob_improvement=0.7,
            prob_ci_low=0.6,
            prob_ci_high=0.8,
            x=np.array([0.0, 1.0, 2.0]),
            diff_iqm=np.array([0.1, 0.2, 0.3]),
            diff_ci_low=np.array([0.05, 0.15, 0.25]),
            diff_ci_high=np.array([0.15, 0.25, 0.35]),
            interpolated=False,
            reps=10,
        )
        defaults.update(overrides)
        return ComparisonResult(**defaults)

    return _build
