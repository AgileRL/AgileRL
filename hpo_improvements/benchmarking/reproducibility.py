"""Seeding helpers for reproducible benchmarking runs.

AgileRL's :func:`agilerl.hpo.mutation.set_global_seed` seeds numpy, torch,
cuda (``manual_seed``) and fastrand, but does **not** seed Python's ``random``
module nor all CUDA devices, and sets no cuDNN determinism flags. This module
fills those gaps locally (no library changes) so the whole benchmark is
reproducible from a single seed.

Note on environment seeding: callers should seed the environment **once**, on
its first ``reset(seed=seed)``, and never re-pass the seed on later resets —
otherwise every episode restarts from the same RNG state, which harms training.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from agilerl.hpo.mutation import set_global_seed


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for a reproducible run.

    :param seed: The global seed.
    :type seed: int
    :param deterministic: If True, set cuDNN to deterministic mode and disable
        its autotuner. Defaults to True.
    :type deterministic: bool
    """
    # numpy, torch (cpu + current cuda device), fastrand
    set_global_seed(seed)
    # Gaps not covered by set_global_seed:
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Pin CPU threading so results are invariant to the host's core count and
    # parallel load. Without this, PyTorch's intra-op parallelism splits float
    # reductions (matmuls, sums) across however many threads the host defaults
    # to, and that accumulation order is *not* identical across different thread
    # counts -- so a solo sequential run (all cores) and a Ray job under
    # concurrency (fewer effective threads), or two hosts with different core
    # counts, diverge from the first gradient step, and the evolutionary HPO loop
    # amplifies that sub-1% perturbation into large fitness gaps. Single-threaded
    # torch is deterministic regardless of host; for the small MLPs used here it
    # is also no slower (often faster) than the multi-threaded default. (Note this
    # does NOT give cross-architecture identity -- ARM vs x86 round floats
    # differently regardless of thread count.) ``set_num_threads`` is safe to call
    # repeatedly; ``set_num_interop_threads`` may only be set before any inter-op
    # work, so it is best-effort (a second call -- e.g. seed_everything per
    # (env, seed) in the sequential runner -- would raise).
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic cuBLAS GEMMs; must be set before the first
        # cuBLAS call (i.e. before training starts), which is why it lives here.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
