"""Shared fixtures and import setup for the benchmarking test suite.

The benchmarking harness uses a *bare-import* convention (``import benchmark``,
``import plotting``, ``from registry import ...``) with ``benchmarking/`` on
``sys.path``. We mirror that here by inserting the benchmarking directory (and
the repo root, for the occasional ``hpo_improvements.benchmarking.*`` package
import) onto ``sys.path`` before any test module is collected.

These tests are *hermetic*: no network, GPU, Ray cluster, real training, or
live Weights & Biases. Heavy library imports (torch, agilerl, matplotlib) still
happen — "hermetic" means no external side effects, not no imports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# --- bare-import path setup (must run at import time, before test collection) ---
_BENCH_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_DIR.parent.parent
for _p in (str(_BENCH_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Importing the plotting module pins matplotlib to the headless Agg backend.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Eagerly cache the benchmarking sibling modules under their bare names so the
# contested bare name ``plotting`` resolves to *this* package's plotting even
# when a sibling test suite (comparison/) has put its own directory on sys.path
# earlier in a shared xdist worker. ``benchmark.py``/``plotting.py`` already
# self-protect their own sys.path, but a benchmarking test doing a first bare
# ``import plotting`` would otherwise be at the mercy of sys.path ordering.
import mechanism  # noqa: E402,F401
import plotting  # noqa: E402,F401
import registry  # noqa: E402,F401


def pytest_configure(config):
    """Register the ``slow`` marker locally so this standalone suite needs no
    change to the repo-level ``pyproject.toml`` (which only registers the
    upstream ``vllm``/``gpu`` markers)."""
    config.addinivalue_line(
        "markers",
        "slow: slower hermetic test (e.g. constructing a real mpe2 environment).",
    )


# --------------------------------------------------------------------------- #
# Global-state isolation                                                       #
# --------------------------------------------------------------------------- #
# Several modules under test mutate process-global state (os.environ, NumPy's
# global RNG, torch thread counts, matplotlib's open-figure registry). Under
# xdist a worker runs many tests in one process, so we snapshot and restore that
# state around every test to stop leakage.
@pytest.fixture(autouse=True)
def _isolate_global_state():
    env_snapshot = dict(os.environ)
    np_state = np.random.get_state()
    try:
        import torch

        n_threads = torch.get_num_threads()
    except Exception:  # pragma: no cover - torch always present in this repo
        torch = None
        n_threads = None

    yield

    # Restore os.environ exactly.
    os.environ.clear()
    os.environ.update(env_snapshot)
    # Restore NumPy's global RNG.
    np.random.set_state(np_state)
    # Restore torch intra-op thread count (interop count cannot be reset).
    if torch is not None and n_threads is not None:
        try:
            torch.set_num_threads(n_threads)
        except Exception:  # pragma: no cover
            pass
    # Close any figures the test left open.
    plt.close("all")


# --------------------------------------------------------------------------- #
# Synthetic data builders                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_history_df():
    """Return a builder for tiny synthetic W&B-history DataFrames.

    Pass any columns as keyword arrays; ``train/global_step`` is supplied with a
    sensible default when omitted.
    """
    import pandas as pd

    def _build(n_rows: int = 3, **columns) -> "pd.DataFrame":
        data = {}
        if "global_step" not in columns and "train/global_step" not in columns:
            data["train/global_step"] = np.arange(1, n_rows + 1, dtype=float)
        for key, value in columns.items():
            col = key.replace("__", "/") if "/" not in key else key
            data[col] = value
        return pd.DataFrame(data)

    return _build


@pytest.fixture
def make_mutation_df():
    """Return a builder for tiny synthetic ``mutation_history.csv`` DataFrames."""
    import pandas as pd

    def _build(**columns) -> "pd.DataFrame":
        return pd.DataFrame(columns)

    return _build


@pytest.fixture
def out_png(tmp_path):
    """A path for a figure a plotting function should write."""
    return str(tmp_path / "figure.png")
