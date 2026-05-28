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

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
