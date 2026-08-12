# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Replay every stored measurement through the current analytic core.

The estimator ships with no fitted correction: a prediction is a pure
function of checkpoint geometry, run knobs and a small table of measured
per-device constants. That makes this the only thing standing between a
formula edit and silent drift — the estimator keeps returning numbers, they
are just wrong.

The fixtures are 406 measurements over 8 models and 2 devices, and they cost
GPU-hours to collect, so they are kept as validation data long after the
calibration layer they were originally fitted for was removed.

Usage::

    python -m tools.memory_profiling.validate            # all fixtures
    python -m tools.memory_profiling.validate --model Qwen/...
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

from agilerl.memory.estimator import estimate_generation, estimate_training
from tools.memory_profiling.calibration import (
    FIXTURES_DIR,
    ModelProfile,
    calibration_target_bytes,
    curated_profiles,
    load_profile,
)
from tools.memory_profiling.harness import SweepPoint

#: Mean error across a fixture's points, per phase. The analytic core sits
#: near 3-4%; this is a drift alarm, not the accuracy claim.
MEAN_BAND = 0.08
#: Worst single point. Corners where the true residual is convex in
#: batch x sequence cannot all be captured by a closed form, so this is
#: deliberately looser than the mean.
WORST_BAND = 0.20


def prediction_errors(profile: ModelProfile) -> dict[str, list[float]]:
    """Relative error of the analytic core on every stored point, by phase."""
    if profile.model_spec is None or profile.device is None:
        return {}
    model = profile.apply_realised_weights(profile.model_spec)
    device = profile.device.to_device_spec()
    baseline = profile.canonical_sleeping_baseline_bytes

    errors: dict[str, list[float]] = {"training": [], "generation": []}
    for measured in profile.measured:
        point = SweepPoint.from_dict(measured.knobs)
        if measured.phase == "training":
            breakdown = estimate_training(
                model, device, point.training_knobs(), colocated=True
            )
        else:
            breakdown = estimate_generation(
                model, device, point.generation_knobs(), colocated=True
            )
        target = calibration_target_bytes(measured, baseline)
        errors[measured.phase].append(abs(breakdown.total_bytes - target) / target)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tools.memory_profiling.validate", description=__doc__
    )
    parser.add_argument("--model", default=None, help="Validate one model id")
    parser.add_argument("--fixtures-dir", default=None)
    args = parser.parse_args(argv)

    fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else FIXTURES_DIR
    pairs = [(args.model, None)] if args.model else curated_profiles(fixtures_dir)
    if not pairs:
        print(f"No fixtures in {fixtures_dir}", file=sys.stderr)
        return 1

    failed = False
    pooled: dict[str, list[float]] = {"training": [], "generation": []}
    for model_id, device_name in pairs:
        profile = load_profile(model_id, fixtures_dir, device_name=device_name)
        if profile is None:
            print(f"{model_id}: no fixture", file=sys.stderr)
            return 1
        errors = prediction_errors(profile)
        print(f"{model_id} @ {device_name or 'unknown device'}")
        for phase in ("training", "generation"):
            values = errors.get(phase) or []
            if not values:
                continue
            pooled[phase].extend(values)
            mean, worst = st.mean(values), max(values)
            flag = "" if mean <= MEAN_BAND and worst <= WORST_BAND else "  <-- DRIFT"
            print(
                f"  {phase:10s} n={len(values):3d} "
                f"mean={mean:.2%} worst={worst:.2%}{flag}"
            )
            failed = failed or flag != ""

    print("\npooled")
    for phase in ("training", "generation"):
        values = pooled[phase]
        if values:
            print(
                f"  {phase:10s} n={len(values):3d} "
                f"mean={st.mean(values):.2%} worst={max(values):.2%}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
