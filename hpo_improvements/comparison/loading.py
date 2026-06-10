"""Load a finished benchmark's results from disk.

A benchmark results folder (produced by the benchmarking harness) looks like::

    <results>/<benchmark_name>/
        config.yaml                       # the manifest + a `benchmark` block
        <Env-v5>/
            s42/wandb_history.csv         # multi-seed layout (Ray orchestrator)
            s43/wandb_history.csv
            ...
        <Env-v5>/wandb_history.csv        # single-seed layout (sequential runner)

This module discovers the ``(environment, seed)`` runs in such a folder and, for
each, reconstructs the **best normalized fitness** curve exactly the way the
benchmarking harness plots it: x-axis is ``train/global_step / pop_size`` (per-
agent environment interactions) and y is ``eval/best_fitness`` mapped through the
random/expert baselines in :mod:`registry`. Reusing the harness's column names,
normalization, and pop-size convention keeps the comparison tool and the
benchmarks it compares on a single source of truth.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

# Import the benchmarking harness as a package so its bare sibling imports
# (``import plotting`` / ``from registry import ...``) are never triggered: we
# only touch ``registry`` (pure) and the column-name constants on ``plotting``.
from hpo_improvements.benchmarking import plotting as bench_plotting
from hpo_improvements.benchmarking.registry import normalization_scores

if TYPE_CHECKING:
    from collections.abc import Iterable

GLOBAL_STEP_COL = bench_plotting.GLOBAL_STEP_COL
BEST_FITNESS_COL = bench_plotting.BEST_FITNESS_COL

# A per-seed subfolder is named ``s<seed>`` (e.g. ``s42``).
_SEED_DIR_RE = re.compile(r"^s(\d+)$")

HISTORY_FILENAME = "wandb_history.csv"
CONFIG_FILENAME = "config.yaml"


@dataclass
class BenchmarkResults:
    """A loaded benchmark results folder.

    :param root: Path to the benchmark folder (the one holding ``config.yaml``).
    """

    root: Path
    config: dict = field(init=False, repr=False)
    algo: str = field(init=False)
    pop_size: int = field(init=False)
    name: str = field(init=False)
    # (env_name, seed) -> path to that run's wandb_history.csv
    runs: dict[tuple[str, int], Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        config_path = self.root / CONFIG_FILENAME
        if not config_path.is_file():
            msg = f"No {CONFIG_FILENAME} found in {self.root}"
            raise FileNotFoundError(msg)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.algo = self.config["algorithm"]["name"]
        self.pop_size = int(self.config.get("training", {}).get("pop_size", 1) or 1)
        bench = self.config.get("benchmark", {}) or {}
        self.name = str(bench.get("name") or self.root.name)
        self.runs = self._discover()
        if not self.runs:
            msg = f"No '{HISTORY_FILENAME}' files found under {self.root}"
            raise FileNotFoundError(msg)

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def _default_seed(self) -> int:
        """Seed for the single-seed layout (no ``s<seed>`` subfolders).

        Older sequential runs store one seed per benchmark under
        ``benchmark.seed``; the parallel runner stores ``benchmark.seeds``.
        """
        bench = self.config.get("benchmark", {}) or {}
        if bench.get("seed") is not None:
            return int(bench["seed"])
        seeds = bench.get("seeds")
        if seeds:
            return int(seeds[0])
        return 0

    def _discover(self) -> dict[tuple[str, int], Path]:
        """Map every ``(env, seed)`` in the folder to its history CSV."""
        runs: dict[tuple[str, int], Path] = {}
        for env_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            env = env_dir.name
            seed_matches = [
                (d, _SEED_DIR_RE.match(d.name))
                for d in sorted(env_dir.iterdir())
                if d.is_dir()
            ]
            seed_dirs = [(d, m) for d, m in seed_matches if m is not None]
            if seed_dirs:
                for d, m in seed_dirs:
                    csv = d / HISTORY_FILENAME
                    if csv.is_file():
                        runs[(env, int(m.group(1)))] = csv
            else:
                csv = env_dir / HISTORY_FILENAME
                if csv.is_file():
                    runs[(env, self._default_seed())] = csv
        return runs

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def envs(self) -> set[str]:
        """Environments that have at least one run."""
        return {env for env, _ in self.runs}

    def seeds(self, env: str) -> set[int]:
        """Seeds available for *env*."""
        return {s for e, s in self.runs if e == env}

    def curve(self, env: str, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(x, normalized_best_fitness)`` for one ``(env, seed)`` run.

        ``x`` is ``global_step / pop_size`` (per-agent interactions) and the
        y-values are the best fitness normalized against *env*'s random/expert
        baselines. Returns ``None`` if the run has no plottable rows.

        :param env: Environment id.
        :param seed: Run seed.
        :return: ``(x, normalized)`` arrays, or ``None``.
        """
        csv = self.runs.get((env, seed))
        if csv is None:
            return None
        df = pd.read_csv(csv)
        if GLOBAL_STEP_COL not in df.columns or BEST_FITNESS_COL not in df.columns:
            return None
        data = df.dropna(subset=[GLOBAL_STEP_COL, BEST_FITNESS_COL]).sort_values(
            GLOBAL_STEP_COL
        )
        if data.empty:
            return None
        x = data[GLOBAL_STEP_COL].to_numpy(dtype=float) / max(self.pop_size, 1)
        best = data[BEST_FITNESS_COL].to_numpy(dtype=float)
        scores = normalization_scores(self.algo, env)
        y = np.array([scores.normalize(f) for f in best], dtype=float)
        return x, y


def resolve_results_dir(name_or_path: str, default_root: Path) -> Path:
    """Resolve a benchmark reference to an on-disk folder.

    Accepts either a folder *name* under *default_root* (the benchmarking
    ``results`` directory) or an explicit path to a benchmark folder.

    :param name_or_path: Folder name or path.
    :param default_root: Directory that benchmark-name references resolve under.
    :return: The resolved, existing benchmark folder.
    :raises FileNotFoundError: If neither interpretation points to a folder.
    """
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir():
        return candidate
    under_root = default_root / name_or_path
    if under_root.is_dir():
        return under_root
    msg = f"Benchmark '{name_or_path}' not found as a path or under {default_root}."
    raise FileNotFoundError(msg)


def list_available(default_root: Path) -> Iterable[str]:
    """Yield benchmark folder names under *default_root* (those with a config)."""
    if not default_root.is_dir():
        return
    for child in sorted(default_root.iterdir()):
        if child.is_dir() and (child / CONFIG_FILENAME).is_file():
            yield child.name


# Allow running sibling modules as plain scripts from this folder.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
