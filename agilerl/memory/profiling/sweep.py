r"""Profiling sweep: corner plan, residual fit, and fixture emission.

A model joins the curated list by running this sweep on a reference device:
roughly 16 corner points across ``{seq_len} x {micro_batch} x {group_size} x
{lora_rank}`` are measured, the analytic model's residuals are fitted as an
intercept plus named slopes, and centre points held out of the fit validate
interpolation across the knob space. The result is one self-contained JSON
fixture per model.

The planner and fitter are pure (testable without a GPU); only ``run_sweep``
touches CUDA.

Usage::

    python -m agilerl.memory.profiling.sweep --model Qwen/Qwen2.5-0.5B-Instruct \\
        --device-name "NVIDIA L4" --output-dir agilerl/memory/fixtures
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

from agilerl.memory.calibration import (
    DeviceFingerprint,
    MeasuredPoint,
    ModelProfile,
    PhaseCalibration,
    ResidualFit,
    generation_basis,
    save_profile,
    training_basis,
)
from agilerl.memory.estimator import estimate_generation, estimate_training
from agilerl.memory.profiling.harness import SweepPoint
from agilerl.memory.specs import DeviceSpec, ModelSpec

#: Corner levels per knob. Corners pin the fit; centre points validate
#: interpolation. Sequence length is the only effectively-continuous axis, so
#: it gets a mid level in the holdout.
CORNER_LEVELS: dict[str, tuple[int, ...]] = {
    "seq_len": (512, 4096),
    "micro_batch": (1, 8),
    "group_size": (4, 16),
    "lora_rank": (8, 64),
}
HOLDOUT_POINTS: tuple[SweepPoint, ...] = (
    SweepPoint(seq_len=1024, micro_batch=4, group_size=8, lora_rank=16),
    SweepPoint(seq_len=2048, micro_batch=2, group_size=8, lora_rank=32),
    SweepPoint(seq_len=2048, micro_batch=8, group_size=4, lora_rank=16),
)


def corner_plan() -> list[SweepPoint]:
    """The fit set: every corner of the knob space (16 points)."""
    return [
        SweepPoint(seq_len=s, micro_batch=b, group_size=g, lora_rank=r)
        for s, b, g, r in itertools.product(
            CORNER_LEVELS["seq_len"],
            CORNER_LEVELS["micro_batch"],
            CORNER_LEVELS["group_size"],
            CORNER_LEVELS["lora_rank"],
        )
    ]


def fit_residuals(
    basis_rows: list[dict[str, float]], residuals: list[float]
) -> ResidualFit:
    """Least-squares fit of ``residual = intercept + sum(slope * term)``.

    Slopes are kept only for terms that actually vary across the sweep;
    constant columns fold into the intercept.
    """
    import numpy as np

    terms = sorted({term for row in basis_rows for term in row})
    varying = [
        term for term in terms if len({row.get(term, 0.0) for row in basis_rows}) > 1
    ]
    design = np.array(
        [[1.0] + [row.get(term, 0.0) for term in varying] for row in basis_rows]
    )
    solution, *_ = np.linalg.lstsq(design, np.array(residuals, dtype=float), rcond=None)
    return ResidualFit(
        intercept_bytes=float(solution[0]),
        slopes={
            term: float(coeff)
            for term, coeff in zip(varying, solution[1:], strict=True)
            if coeff != 0.0
        },
    )


def _analytic_bytes(
    model: ModelSpec,
    device: DeviceSpec,
    point: SweepPoint,
    phase: str,
    gpu_memory_utilization: float,
) -> tuple[int, dict[str, float]]:
    if phase == "training":
        knobs = point.training_knobs()
        breakdown = estimate_training(
            model, device, knobs, colocated=True, profile=None
        )
        return breakdown.total_bytes, training_basis(model, knobs)
    gen_knobs = point.generation_knobs(gpu_memory_utilization)
    breakdown = estimate_generation(
        model, device, gen_knobs, colocated=True, profile=None
    )
    return breakdown.total_bytes, generation_basis(model, gen_knobs)


def calibrate_phase(
    model: ModelSpec,
    device: DeviceSpec,
    fit_points: list[tuple[SweepPoint, MeasuredPoint]],
    holdout_points: list[tuple[SweepPoint, MeasuredPoint]],
    phase: str,
    gpu_memory_utilization: float,
) -> PhaseCalibration:
    """Fit one phase's residual model and validate on held-out combinations."""
    basis_rows: list[dict[str, float]] = []
    residuals: list[float] = []
    for point, measured in fit_points:
        analytic, basis = _analytic_bytes(
            model, device, point, phase, gpu_memory_utilization
        )
        basis_rows.append(basis)
        residuals.append(measured.nvml_peak_bytes - analytic)
    fit = fit_residuals(basis_rows, residuals)

    max_rel_error = None
    if holdout_points:
        errors = []
        for point, measured in holdout_points:
            analytic, basis = _analytic_bytes(
                model, device, point, phase, gpu_memory_utilization
            )
            predicted = analytic + fit.correction_bytes(basis)
            errors.append(
                abs(predicted - measured.nvml_peak_bytes) / measured.nvml_peak_bytes
            )
        max_rel_error = max(errors)
    return PhaseCalibration(
        fit=fit,
        holdout_max_rel_error=max_rel_error,
        n_points=len(fit_points),
    )


def run_sweep(
    model_name: str,
    device_name: str,
    device_index: int = 0,
    gpu_memory_utilization: float = 0.45,
    quantizations: tuple[str, ...] = ("none",),
) -> ModelProfile:
    """Measure every plan point on the local GPU and build the profile."""
    import torch
    from transformers import AutoConfig

    import agilerl
    from agilerl.memory.profiling.harness import (
        measure_point,
        measure_realised_weight_bytes,
    )
    from agilerl.memory.specs import ModelArch, WeightVariant

    arch = ModelArch.from_hf_config(AutoConfig.from_pretrained(model_name).to_dict())
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    capability = torch.cuda.get_device_capability(device_index)
    device = DeviceSpec.from_compute_capability(
        total_bytes, *capability, name=device_name
    )

    realised: dict[str, int] = {}
    variants = []
    for quantization in quantizations:
        name = "base" if quantization == "none" else quantization
        realised[name] = measure_realised_weight_bytes(
            model_name, quantization, device_index
        )
        variants.append(
            WeightVariant(
                name=name,
                quantization=quantization,  # type: ignore[arg-type]
                realised_bytes=realised[name],
            )
        )
    model = ModelSpec(model_id=model_name, arch=arch, variants=tuple(variants))

    measured: list[MeasuredPoint] = []
    fit_pairs: dict[str, list[tuple[SweepPoint, MeasuredPoint]]] = {
        "training": [],
        "generation": [],
    }
    holdout_pairs: dict[str, list[tuple[SweepPoint, MeasuredPoint]]] = {
        "training": [],
        "generation": [],
    }
    plan = corner_plan()
    for i, point in enumerate(plan + list(HOLDOUT_POINTS)):
        held_out = i >= len(plan)
        print(
            f"[{i + 1}/{len(plan) + len(HOLDOUT_POINTS)}] {point}"
            f"{' (holdout)' if held_out else ''}",
            flush=True,
        )
        generation, training = measure_point(
            model_name,
            point,
            device_index=device_index,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        measured.extend([generation, training])
        target = holdout_pairs if held_out else fit_pairs
        target["generation"].append((point, generation))
        target["training"].append((point, training))

    return ModelProfile(
        model_id=model_name,
        model_spec=model,
        device=DeviceFingerprint(
            name=device_name,
            total_bytes=total_bytes,
            compute_capability=f"{capability[0]}.{capability[1]}",
        ),
        framework_versions={
            "agilerl": getattr(agilerl, "__version__", "unknown"),
            "torch": torch.__version__,
        },
        training=calibrate_phase(
            model,
            device,
            fit_pairs["training"],
            holdout_pairs["training"],
            "training",
            gpu_memory_utilization,
        ),
        generation=calibrate_phase(
            model,
            device,
            fit_pairs["generation"],
            holdout_pairs["generation"],
            "generation",
            gpu_memory_utilization,
        ),
        realised_weight_bytes=realised,
        measured=tuple(measured),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agilerl.memory.profiling.sweep", description=__doc__
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-name", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=["none"],
        choices=["none", "nf4", "int8"],
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep plan and exit (no GPU needed)",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        for point in corner_plan():
            print(json.dumps(point.as_dict()))
        for point in HOLDOUT_POINTS:
            print(json.dumps({**point.as_dict(), "holdout": True}))
        return 0

    profile = run_sweep(
        args.model,
        args.device_name,
        device_index=args.device_index,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantizations=tuple(args.quantizations),
    )
    from pathlib import Path

    output_dir = Path(args.output_dir) if args.output_dir else None
    path = save_profile(profile, output_dir)
    print(f"Wrote {path}")
    print(
        "Holdout max relative error — "
        f"training: {profile.training.holdout_max_rel_error}, "
        f"generation: {profile.generation.holdout_max_rel_error}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
