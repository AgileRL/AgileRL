"""Re-fit checked-in profiles from their stored measurements — no GPU.

Every profile keeps its raw sweep points, so when the analytic core changes
(a new component, a corrected formula) the constants can be re-derived
without re-running the sweep. This is what makes formula work cheap: measure
once, re-fit freely.

Usage::

    python -m agilerl.memory.profiling.refit                    # all fixtures
    python -m agilerl.memory.profiling.refit --model Qwen/...   # just one
    python -m agilerl.memory.profiling.refit --check            # verify only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agilerl.memory.calibration import (
    FIXTURES_DIR,
    ModelProfile,
    curated_profiles,
    load_profile,
    save_profile,
)
from agilerl.memory.profiling.harness import SweepPoint
from agilerl.memory.profiling.sweep import HOLDOUT_POINTS, calibrate_phase
from agilerl.memory.specs import DeviceSpec

#: The headline accuracy claim: error on held-out knob combinations, which
#: is what "will my run fit" actually depends on.
ACCURACY_BAND = 0.10
#: Mean error across every measured point must also stay inside the band.
MEAN_ACCURACY_BAND = 0.10
#: Worst single point is allowed more room. The residual model is linear in
#: a handful of basis terms, so extreme corners — where the true residual is
#: convex in batch x sequence — cannot all be fitted simultaneously. This is
#: a drift alarm, not the accuracy claim.
WORST_POINT_BAND = 0.20


def _holdout_keys() -> set[tuple[int, int, int, int, float]]:
    return {
        (
            p.seq_len,
            p.micro_batch,
            p.group_size,
            p.lora_rank,
            p.gpu_memory_utilization,
        )
        for p in HOLDOUT_POINTS
    }


def refit(profile: ModelProfile) -> ModelProfile:
    """Return a copy of ``profile`` with both phases re-fitted from its
    stored measurements against the current analytic core.
    """
    if profile.model_spec is None:
        msg = (
            f"Profile {profile.model_id!r} has no embedded model_spec; "
            "cannot re-fit without the geometry it was measured against."
        )
        raise ValueError(msg)
    if profile.device is None:
        msg = f"Profile {profile.model_id!r} has no device fingerprint."
        raise ValueError(msg)

    model = profile.apply_realised_weights(profile.model_spec)
    major, _, minor = (profile.device.compute_capability or "8.0").partition(".")
    device = DeviceSpec.from_compute_capability(
        profile.device.total_bytes,
        int(major),
        int(minor or 0),
        name=profile.device.name,
    )

    holdout = _holdout_keys()
    fit_pairs: dict[str, list[tuple[SweepPoint, object]]] = {
        "training": [],
        "generation": [],
    }
    holdout_pairs: dict[str, list[tuple[SweepPoint, object]]] = {
        "training": [],
        "generation": [],
    }
    for measured in profile.measured:
        point = SweepPoint.from_dict(measured.knobs)
        key = (
            point.seq_len,
            point.micro_batch,
            point.group_size,
            point.lora_rank,
            point.gpu_memory_utilization,
        )
        target = holdout_pairs if key in holdout else fit_pairs
        target[measured.phase].append((point, measured))

    return profile.model_copy(
        update={
            "training": calibrate_phase(
                model,
                device,
                fit_pairs["training"],  # type: ignore[arg-type]
                holdout_pairs["training"],  # type: ignore[arg-type]
                "training",
            ),
            "generation": calibrate_phase(
                model,
                device,
                fit_pairs["generation"],  # type: ignore[arg-type]
                holdout_pairs["generation"],  # type: ignore[arg-type]
                "generation",
            ),
        }
    )


def prediction_errors(
    profile: ModelProfile, *, calibrated: bool = True
) -> dict[str, list[float]]:
    """Relative error of the estimator on every stored point, keyed by phase.

    With ``calibrated=False`` the fitted constants are withheld, measuring
    the analytic core on its own — i.e. the accuracy a model that has never
    been profiled would get.
    """
    from agilerl.memory import formulas
    from agilerl.memory.estimator import estimate_generation, estimate_training

    if profile.model_spec is None or profile.device is None:
        return {}
    model = profile.apply_realised_weights(profile.model_spec)
    major, _, minor = (profile.device.compute_capability or "8.0").partition(".")
    device = DeviceSpec.from_compute_capability(
        profile.device.total_bytes,
        int(major),
        int(minor or 0),
        name=profile.device.name,
    )
    applied = profile if calibrated else None
    measured_residual = profile.sleeping_engine_residual_bytes
    residual = (
        measured_residual
        if calibrated and measured_residual is not None
        else formulas.SLEEPING_ENGINE_RESIDUAL_BYTES
    )

    errors: dict[str, list[float]] = {"training": [], "generation": []}
    for measured in profile.measured:
        point = SweepPoint.from_dict(measured.knobs)
        if measured.phase == "training":
            breakdown = estimate_training(
                model,
                device,
                point.training_knobs(),
                colocated=True,
                profile=applied,
                colocated_engine_reservation_bytes=residual,
            )
        else:
            breakdown = estimate_generation(
                model,
                device,
                point.generation_knobs(),
                colocated=True,
                profile=applied,
            )
        errors[measured.phase].append(
            abs(breakdown.total_bytes - measured.device_peak_bytes)
            / measured.device_peak_bytes
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agilerl.memory.profiling.refit", description=__doc__
    )
    parser.add_argument("--model", default=None, help="Refit one model id")
    parser.add_argument("--fixtures-dir", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report accuracy without rewriting the fixtures",
    )
    args = parser.parse_args(argv)

    fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else FIXTURES_DIR
    pairs = [(args.model, None)] if args.model else curated_profiles(fixtures_dir)
    if not pairs:
        print(f"No fixtures in {fixtures_dir}", file=sys.stderr)
        return 1

    worst = 0.0
    for model_id, device_name in pairs:
        profile = load_profile(model_id, fixtures_dir, device_name=device_name)
        if profile is None:
            print(f"{model_id}: no fixture", file=sys.stderr)
            return 1
        updated = profile if args.check else refit(profile)
        if not args.check:
            save_profile(updated, fixtures_dir)
        errors = prediction_errors(updated)
        raw_errors = prediction_errors(updated, calibrated=False)
        label = f"{model_id} @ {device_name}" if device_name else model_id
        print(label)
        for phase in ("training", "generation"):
            phase_errors = errors.get(phase) or [0.0]
            raw = raw_errors.get(phase) or [0.0]
            calibration = getattr(updated, phase)
            worst = max(worst, max(phase_errors))
            print(
                f"  {phase:10s} n={calibration.n_points:2d} "
                f"calibrated max={max(phase_errors):.2%} "
                f"mean={sum(phase_errors) / len(phase_errors):.2%} | "
                f"holdout={(calibration.holdout_max_rel_error or 0):.2%} | "
                f"uncalibrated mean={sum(raw) / len(raw):.2%}"
            )
    if worst > WORST_POINT_BAND:
        print(f"\nWorst point {worst:.2%} exceeds the {WORST_POINT_BAND:.0%} band.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
