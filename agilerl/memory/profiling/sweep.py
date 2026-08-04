# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
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
import importlib.metadata
import itertools
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

from agilerl.memory.calibration import (
    FIXTURES_DIR,
    DeviceFingerprint,
    MeasuredPoint,
    ModelProfile,
    PhaseCalibration,
    ResidualFit,
    _slug,
    calibration_target_bytes,
    generation_basis,
    median_sleeping_baseline,
    save_profile,
    training_basis,
)
from agilerl.memory.estimator import estimate_generation, estimate_training
from agilerl.memory.profiling.harness import SweepPoint, variant_name
from agilerl.memory.profiling.nvml import wait_for_idle
from agilerl.memory.specs import DeviceSpec, ModelSpec, QuantizationMethod

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
    # Held out at other engine budgets: the corners are all measured at one
    # utilization, so these check that the utilization-dependent terms are
    # modelled analytically rather than baked into the fitted constants.
    SweepPoint(
        seq_len=1024,
        micro_batch=4,
        group_size=8,
        lora_rank=16,
        gpu_memory_utilization=0.30,
    ),
    SweepPoint(
        seq_len=1024,
        micro_batch=4,
        group_size=8,
        lora_rank=16,
        gpu_memory_utilization=0.60,
    ),
)


def corner_plan(seq_lens: tuple[int, ...] | None = None) -> list[SweepPoint]:
    """The fit set: every corner of the knob space (16 points)."""
    return [
        SweepPoint(seq_len=s, micro_batch=b, group_size=g, lora_rank=r)
        for s, b, g, r in itertools.product(
            seq_lens or CORNER_LEVELS["seq_len"],
            CORNER_LEVELS["micro_batch"],
            CORNER_LEVELS["group_size"],
            CORNER_LEVELS["lora_rank"],
        )
    ]


def budget_for(model: ModelSpec, device: DeviceSpec, point: SweepPoint) -> float:
    """Engine budget large enough to serve this point's KV demand.

    A fixed budget is why the two hardest models lost corners: a 14 GiB model
    plus long-context, high-concurrency KV does not fit 45% of a 40 GiB card,
    so exactly the corners that anchor the long-context slopes went missing
    and the fits were left extrapolating. Sizing the budget per point keeps
    them, and since the engine sleeps during training it costs the training
    phase nothing.
    """
    from agilerl.memory.estimator import recommend_engine_budget

    required, _ = recommend_engine_budget(model, device, point.generation_knobs())
    return min(max(required * 1.1, 0.2), 0.9)


def fit_residuals(
    basis_rows: list[dict[str, float]], residuals: list[float]
) -> ResidualFit:
    """Least-squares fit of ``residual = intercept + sum(slope * term)``.

    Coefficients are deliberately unconstrained. Constraining them to be
    non-negative is tempting — "the residual is unmodelled memory" — but it
    is wrong: the analytic core can over-count as well as under-count, and a
    negative coefficient is how the fit corrects that. Forcing non-negativity
    was measured to make the generation holdout 8x worse (0.49% -> 3.73%),
    because the engine-side model over-predicts and NNLS simply refuses to
    subtract.

    Slopes are kept only for terms that vary across the sweep; constant
    columns fold into the intercept.
    """
    import numpy as np

    if not basis_rows:
        msg = (
            "No measurements survived the sweep, so there is nothing to fit. "
            "Check the per-point logs: every point failed for the same reason."
        )
        raise ValueError(msg)

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
) -> tuple[int, dict[str, float]]:
    """Uncalibrated prediction and basis terms for one point.

    Passing ``profile=None`` keeps the colocated engine reservation at its
    analytic constant, so the residual being fitted is genuine unmodelled
    overhead rather than the reservation — which would otherwise pin the fit
    to one ``gpu_memory_utilization``.
    """
    if phase == "training":
        knobs = point.training_knobs()
        breakdown = estimate_training(
            model,
            device,
            knobs,
            trainer_variant=variant_name(point.quantization),
            colocated=True,
            profile=None,
        )
        return breakdown.total_bytes, training_basis(model, knobs)
    gen_knobs = point.generation_knobs()
    breakdown = estimate_generation(
        model, device, gen_knobs, colocated=True, profile=None
    )
    return breakdown.total_bytes, generation_basis(model, gen_knobs)


def _model_spec_for(model_name: str, quantizations: set[str]) -> ModelSpec:
    """Analytic spec for a model, without measuring weights on a GPU.

    The sweep proper measures realised weight bytes per variant; recovery
    from a checkpoint cannot, so it falls back to the analytic sizes.
    """
    from transformers import AutoConfig

    from agilerl.memory.specs import ModelArch, WeightVariant

    arch = ModelArch.from_hf_config(AutoConfig.from_pretrained(model_name).to_dict())
    variants = tuple(
        WeightVariant(
            name=variant_name(q),
            quantization=cast("QuantizationMethod", q),
        )
        for q in sorted(quantizations)
    )
    return ModelSpec(model_id=model_name, arch=arch, variants=variants)


def _append_checkpoint(path: Path, points: list[MeasuredPoint]) -> None:
    """Append measurements to the crash-safe sidecar, one JSON object a line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for point in points:
            handle.write(json.dumps(point.model_dump(mode="json")) + "\n")


def profile_from_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    device_name: str,
    device_index: int = 0,
) -> ModelProfile:
    """Rebuild a fitted profile from an interrupted sweep's measurements.

    The corners a partial sweep did reach still anchor a usable fit, so an
    hour of GPU time is not lost to the sweep being killed at point 12.
    Holdout error is left unset: which points were held out is a property of
    the plan, and a partial run has no guarantee of having reached them.
    """
    measured = [
        MeasuredPoint.model_validate(json.loads(line))
        for line in checkpoint_path.read_text().splitlines()
        if line.strip()
    ]
    if not measured:
        msg = f"{checkpoint_path} holds no measurements."
        raise ValueError(msg)

    total_bytes, capability = _device_identity(device_index)
    model = _model_spec_for(
        model_name, {str(p.knobs.get("quantization", "none")) for p in measured}
    )

    pairs: dict[str, list[tuple[SweepPoint, MeasuredPoint]]] = {
        "training": [],
        "generation": [],
    }
    for point in measured:
        pairs[point.phase].append((SweepPoint.from_dict(point.knobs), point))

    return _build_profile(
        model_name,
        model,
        device_name,
        total_bytes,
        capability,
        algorithm=str(measured[0].knobs.get("algorithm", "grpo")),
        quantization=str(measured[0].knobs.get("quantization", "none")),
        fit_pairs=pairs,
        holdout_pairs={"training": [], "generation": []},
        measured=measured,
    )


def calibrate_phase(
    model: ModelSpec,
    device: DeviceSpec,
    fit_points: list[tuple[SweepPoint, MeasuredPoint]],
    holdout_points: list[tuple[SweepPoint, MeasuredPoint]],
    phase: str,
    canonical_baseline_bytes: int = 0,
) -> PhaseCalibration:
    """Fit one phase's residual model and validate on held-out combinations."""
    basis_rows: list[dict[str, float]] = []
    residuals: list[float] = []
    for point, measured in fit_points:
        analytic, basis = _analytic_bytes(model, device, point, phase)
        basis_rows.append(basis)
        target = calibration_target_bytes(measured, canonical_baseline_bytes)
        residuals.append(target - analytic)
    fit = fit_residuals(basis_rows, residuals)

    max_rel_error = None
    if holdout_points:
        errors = []
        for point, measured in holdout_points:
            analytic, basis = _analytic_bytes(model, device, point, phase)
            predicted = analytic + fit.correction_bytes(basis)
            target = calibration_target_bytes(measured, canonical_baseline_bytes)
            errors.append(abs(predicted - target) / target)
        max_rel_error = max(errors)
    return PhaseCalibration(
        fit=fit,
        holdout_max_rel_error=max_rel_error,
        n_points=len(fit_points),
    )


def _build_profile(
    model_name: str,
    model: ModelSpec,
    device_name: str,
    total_bytes: int,
    capability: tuple[int, int],
    algorithm: str,
    quantization: str,
    fit_pairs: dict[str, list[tuple[SweepPoint, MeasuredPoint]]],
    holdout_pairs: dict[str, list[tuple[SweepPoint, MeasuredPoint]]],
    measured: list[MeasuredPoint],
    realised: dict[str, int] | None = None,
) -> ModelProfile:
    """Fit both phases and assemble the fixture from a sweep's measurements."""
    device = DeviceSpec.from_compute_capability(
        total_bytes, *capability, name=device_name
    )
    canonical = median_sleeping_baseline(measured)
    return ModelProfile(
        model_id=model_name,
        model_spec=model,
        algorithm=algorithm,
        quantization=quantization,
        device=DeviceFingerprint(
            name=device_name,
            total_bytes=total_bytes,
            compute_capability=f"{capability[0]}.{capability[1]}",
        ),
        framework_versions=_framework_versions(),
        training=calibrate_phase(
            model,
            device,
            fit_pairs["training"],
            holdout_pairs["training"],
            "training",
            canonical_baseline_bytes=canonical,
        ),
        generation=(
            calibrate_phase(
                model,
                device,
                fit_pairs["generation"],
                holdout_pairs["generation"],
                "generation",
            )
            # SFT and DPO never generate, so there is no engine phase to fit.
            if fit_pairs["generation"]
            else PhaseCalibration()
        ),
        realised_weight_bytes=realised or {},
        measured=tuple(measured),
    )


def _json_from_subprocess(make_cmd: Callable[[str], list[str]]) -> dict:
    """Run a subprocess that writes JSON to a temp path and return the parse."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = handle.name
    try:
        subprocess.run(make_cmd(out_path), check=True)
        return json.loads(Path(out_path).read_text())
    finally:
        Path(out_path).unlink(missing_ok=True)


def _device_identity(device_index: int) -> tuple[int, tuple[int, int]]:
    """Total bytes and compute capability of the device, read in a
    subprocess so the sweep parent never initialises CUDA.
    """
    script = (
        "import json,sys,torch;"
        "i=int(sys.argv[2]);"
        "p=torch.cuda.get_device_properties(i);"
        "c=torch.cuda.get_device_capability(i);"
        "open(sys.argv[1],'w').write("
        "json.dumps({'total':p.total_memory,'cc':list(c)}))"
    )
    data = _json_from_subprocess(
        lambda out: [sys.executable, "-c", script, out, str(device_index)]
    )
    return int(data["total"]), (int(data["cc"][0]), int(data["cc"][1]))


def _framework_versions() -> dict[str, str]:
    def _version(package: str) -> str | None:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    packages = ("agilerl", "torch", "vllm", "transformers", "peft")
    return {p: v for p in packages if (v := _version(p)) is not None}


def _measure_point_subprocess(
    model_name: str,
    point: SweepPoint,
    device_index: int,
    warmup_steps: int = 0,
) -> tuple[MeasuredPoint | None, MeasuredPoint]:
    """Measure one point in a fresh subprocess.

    vLLM's CuMem allocator is process-global and allows one engine per
    process, so each sleep-mode point needs its own process. That freshness
    is also why ``warmup_steps`` matters: every point starts from an agent
    that has never stepped its optimizer.
    """
    data = _json_from_subprocess(
        lambda out: [
            sys.executable,
            "-m",
            "agilerl.memory.profiling.harness",
            "--model",
            model_name,
            "--out",
            out,
            "--point-json",
            json.dumps(point.as_dict()),
            "--device-index",
            str(device_index),
            "--warmup-steps",
            str(warmup_steps),
        ]
    )
    return (
        MeasuredPoint.model_validate(data["generation"])
        if data.get("generation")
        else None,
        MeasuredPoint.model_validate(data["training"]),
    )


def _measure_weights_subprocess(
    model_name: str, quantization: str, device_index: int
) -> dict[str, int]:
    data = _json_from_subprocess(
        lambda out: [
            sys.executable,
            "-m",
            "agilerl.memory.profiling.harness",
            "--model",
            model_name,
            "--out",
            out,
            "--weights-only",
            "--quantization",
            quantization,
            "--device-index",
            str(device_index),
        ]
    )
    sizes = data["realised_weight_bytes"]
    if isinstance(sizes, int):  # profiles written before variant measurement
        return {"full": sizes}
    return {k: int(v) for k, v in sizes.items()}


def run_sweep(
    model_name: str,
    device_name: str,
    device_index: int = 0,
    gpu_memory_utilization: float = 0.45,
    quantizations: tuple[str, ...] = ("none",),
    point_quantization: str = "none",
    lora_target_scope: str | None = None,
    algorithm: str = "grpo",
    lora_target_modules: str = "all-linear",
    memory_efficient_params: bool = True,
    seq_lens: tuple[int, ...] | None = None,
    auto_budget: bool = False,
    checkpoint_path: Path | None = None,
    warmup_steps: int = 0,
) -> ModelProfile:
    """Measure every plan point on the local GPU and build the profile.

    Each measurement runs in its own subprocess (CuMem is process-global), so
    this parent process never touches CUDA. A point that fails (OOM on a
    corner too large for the device) is logged and skipped rather than
    aborting the sweep.

    Measurements are appended to ``checkpoint_path`` as they land. A sweep is
    an hour of GPU time and the profile is only written at the end, so
    without this a sweep interrupted at point 12 of 21 loses all twelve --
    which is exactly what happened when a long-context re-run was cut short.
    Rebuild a profile from a partial run with :func:`profile_from_checkpoint`.

    ``warmup_steps`` complete iterations precede the measured one; 0, the
    default, is what every checked-in fixture used. Treat a non-zero value as
    an experiment: a paired 21-point control on Qwen2.5-0.5B differing only
    in this knob left generation byte-identical but moved training peaks by
    -236 to +186 MiB with no consistent sign, and made calibrated training
    error worse (3.5% -> 12.9% mean). That control ran before
    :func:`~agilerl.memory.profiling.harness._warm_rollout_update` replayed
    the rollout, so it indicts the old warmup rather than the idea; the
    corrected version has not been measured. Whatever the outcome, a mixed
    corpus is worse than either choice, so change this for every profile at
    once or not at all.
    """
    from transformers import AutoConfig

    from agilerl.memory.specs import ModelArch, WeightVariant

    arch = ModelArch.from_hf_config(AutoConfig.from_pretrained(model_name).to_dict())

    # Every variant the checkpoint actually offers, measured rather than
    # derived: each quantization level requested, and — automatically, for
    # multimodal bases — the text-only size with the towers removed.
    realised: dict[str, int] = {}
    variants = []
    for quantization in quantizations:
        name = variant_name(quantization)
        sizes = _measure_weights_subprocess(model_name, quantization, device_index)
        realised[name] = sizes["full"]
        variants.append(
            WeightVariant(
                name=name,
                quantization=cast("QuantizationMethod", quantization),
                realised_bytes=sizes["full"],
            )
        )
        if "stripped" in sizes:
            stripped_name = f"{name}-stripped"
            realised[stripped_name] = sizes["stripped"]
            realised[f"{name}-towers"] = sizes["towers"]
            variants.append(
                WeightVariant(
                    name=stripped_name,
                    quantization=cast("QuantizationMethod", quantization),
                    stripped_multimodal=True,
                    realised_bytes=sizes["stripped"],
                )
            )
            print(
                f"    multimodal towers: {sizes['towers'] / 1024**3:.2f} GiB "
                f"({sizes['towers'] / sizes['full']:.0%} of the checkpoint)",
                flush=True,
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
    total_bytes, capability = _device_identity(device_index)
    device = DeviceSpec.from_compute_capability(
        total_bytes, *capability, name=device_name
    )

    plan = [
        replace(
            point,
            gpu_memory_utilization=gpu_memory_utilization,
            quantization=point_quantization,
            lora_target_scope=lora_target_scope,
            algorithm=algorithm,
            lora_target_modules=lora_target_modules,
            memory_efficient_params=memory_efficient_params,
        )
        for point in corner_plan(seq_lens)
    ]
    holdout_plan = [
        replace(
            point,
            quantization=point_quantization,
            lora_target_scope=lora_target_scope,
            algorithm=algorithm,
            lora_target_modules=lora_target_modules,
            memory_efficient_params=memory_efficient_params,
        )
        for point in HOLDOUT_POINTS
    ]
    if auto_budget:
        plan = [
            replace(p, gpu_memory_utilization=budget_for(model, device, p))
            for p in plan
        ]
        holdout_plan = [
            replace(p, gpu_memory_utilization=budget_for(model, device, p))
            for p in holdout_plan
        ]

    skipped: list[str] = []
    for i, point in enumerate(plan + holdout_plan):
        held_out = i >= len(plan)
        print(
            f"[{i + 1}/{len(plan) + len(holdout_plan)}] {point}"
            f"{' (holdout)' if held_out else ''}",
            flush=True,
        )
        floor = wait_for_idle(device_index)
        print(f"    device floor {floor / 1024**3:.2f} GiB", flush=True)
        try:
            generation, training = _measure_point_subprocess(
                model_name,
                point,
                device_index=device_index,
                warmup_steps=warmup_steps,
            )
        except subprocess.CalledProcessError as exc:
            # Usually a corner too large for this device, which is
            # information rather than a failure — but not always, so do not
            # assert a cause. Gemma 4 failed every point for an unrelated
            # reason (LoRA targeting its audio tower) while claiming OOM.
            print(
                f"    SKIPPED (exit {exc.returncode}) — see sweep log for the cause",
                flush=True,
            )
            skipped.append(str(point))
            continue
        measured.append(training)
        target = holdout_pairs if held_out else fit_pairs
        target["training"].append((point, training))
        if generation is not None:
            measured.append(generation)
            target["generation"].append((point, generation))
        if checkpoint_path is not None:
            _append_checkpoint(
                checkpoint_path,
                [p for p in (training, generation) if p is not None],
            )

    if skipped:
        print(f"\n{len(skipped)} point(s) skipped:", flush=True)
        for entry in skipped:
            print(f"  - {entry}", flush=True)

    return _build_profile(
        model_name,
        model,
        device_name,
        total_bytes,
        capability,
        algorithm=algorithm,
        quantization=point_quantization,
        fit_pairs=fit_pairs,
        holdout_pairs=holdout_pairs,
        measured=measured,
        realised=realised,
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
        help="Weight variants to measure realised sizes for",
    )
    parser.add_argument(
        "--point-quantization",
        default="none",
        choices=["none", "nf4", "int8"],
        help="Trainer quantization used for the swept points themselves",
    )
    parser.add_argument(
        "--lora-target-scope",
        default=None,
        help=(
            "Restrict LoRA targeting, e.g. language_model. Required for "
            "multimodal checkpoints: all-linear otherwise wraps the vision "
            "and audio towers and vLLM rejects the adapter."
        ),
    )
    parser.add_argument(
        "--algorithm", default="grpo", choices=["grpo", "ppo", "sft", "dpo"]
    )
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument(
        "--no-memory-efficient-params",
        action="store_true",
        help=(
            "Keep trainer weights on the device across the rollout. Needed "
            "for PPO, whose value head is not moved back and faults in Triton."
        ),
    )
    parser.add_argument(
        "--seq-lens",
        default=None,
        help="Comma-separated context lengths to use as the corner levels",
    )
    parser.add_argument(
        "--auto-budget",
        action="store_true",
        help=(
            "Size gpu_memory_utilization per point from the KV demand, so "
            "long-context and high-concurrency corners do not OOM away"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Append each measurement here as it lands, so an interrupted "
            "sweep is recoverable (default: <output-dir>/<model>.partial.jsonl)"
        ),
    )
    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        help="Rebuild a profile from an interrupted sweep's --checkpoint file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep plan and exit (no GPU needed)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Complete iterations to run before the measured one (default 0, "
            "matching every checked-in fixture). Change it for the whole "
            "corpus or not at all."
        ),
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else (output_dir or FIXTURES_DIR) / f"{_slug(args.model)}.partial.jsonl"
    )

    if args.from_checkpoint:
        recovered = profile_from_checkpoint(
            checkpoint, args.model, args.device_name, device_index=args.device_index
        )
        print(f"Recovered {len(recovered.measured)} measurements from {checkpoint}")
        print(f"Wrote {save_profile(recovered, output_dir)}")
        return 0

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
        point_quantization=args.point_quantization,
        lora_target_scope=args.lora_target_scope,
        algorithm=args.algorithm,
        lora_target_modules=args.lora_target_modules,
        memory_efficient_params=not args.no_memory_efficient_params,
        seq_lens=(
            tuple(int(s) for s in args.seq_lens.split(",")) if args.seq_lens else None
        ),
        auto_budget=args.auto_budget,
        checkpoint_path=checkpoint,
        warmup_steps=args.warmup_steps,
    )
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
