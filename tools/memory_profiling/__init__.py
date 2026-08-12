# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""GPU profiling that pins the calibration constants for curated models.

The analytic core carries the shape of every memory component; profiling
only fits a small residual bundle per (model, device) — roughly 19 sweep
points rather than a combinatorial grid. See
:mod:`tools.memory_profiling.sweep` for the entry point.
"""

from tools.memory_profiling.harness import (
    SweepPoint,
    measure_point,
    measure_realised_weight_bytes,
)
from tools.memory_profiling.sweep import corner_plan, fit_residuals, run_sweep

__all__ = [
    "SweepPoint",
    "corner_plan",
    "fit_residuals",
    "measure_point",
    "measure_realised_weight_bytes",
    "run_sweep",
]
