"""Per-(model, device) calibration of the analytic memory model.

The runtime object is the closed-form estimator; profiling produces a small
constant bundle per curated model that corrects it:

``predicted = analytic + intercept + sum(slope_i * basis_i(knobs))``

The basis terms are named, interpretable token counts so a fitted slope reads
as "extra bytes per gradient token on this (model, device)". Fitting happens
offline in :mod:`agilerl.memory.profiling`; this module only defines the
schema, evaluates basis terms, and applies a stored fit — all pure python so
it ports to the widget runtime unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agilerl.memory import formulas
from agilerl.memory.formulas import resolve_max_num_batched_tokens
from agilerl.memory.specs import (
    DeviceSpec,
    GenerationKnobs,
    ModelSpec,
    TrainingKnobs,
)

SCHEMA_VERSION = 1

Phase = Literal["training", "generation"]

#: Directory of checked-in profiles for curated models.
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def training_basis(model: ModelSpec, knobs: TrainingKnobs) -> dict[str, float]:
    """Basis terms for the training-phase residual model."""
    s = knobs.max_model_len
    return {
        "grad_tokens": float(knobs.grad_rows * s),
        "nograd_tokens": float(knobs.grad_rows * knobs.n_adapter_rows * s),
        # Total rows in the update, not just one micro-batch: the caching
        # allocator does not hand memory back between micro-batches, so an
        # update split into several of them peaks higher than one of them.
        "update_tokens": float(knobs.trajectories * s),
        "extra_micro_batches": float(knobs.n_micro_batches - 1),
        "seq_len": float(s),
    }


def generation_basis(model: ModelSpec, knobs: GenerationKnobs) -> dict[str, float]:
    """Basis terms for the generation-phase residual model."""
    batched = resolve_max_num_batched_tokens(
        knobs.max_num_seqs, knobs.max_model_len, knobs.max_num_batched_tokens
    )
    return {
        "batched_tokens": float(batched),
        "kv_tokens": float(knobs.concurrency * knobs.max_model_len),
        "seq_len": float(knobs.max_model_len),
    }


class ResidualFit(BaseModel):
    """Fitted correction: intercept plus named slopes over basis terms."""

    model_config = ConfigDict(frozen=True)

    intercept_bytes: float = 0.0
    slopes: dict[str, float] = Field(default_factory=dict)

    def correction_bytes(self, basis: dict[str, float]) -> float:
        return self.intercept_bytes + sum(
            coeff * basis.get(term, 0.0) for term, coeff in self.slopes.items()
        )


class MeasuredPoint(BaseModel):
    """One profiled sweep point, kept for refit and audit."""

    model_config = ConfigDict(frozen=True)

    knobs: dict[str, float | int | str | bool]
    phase: Phase
    #: Best device-level peak estimate and the calibration target. For
    #: generation this is the raw NVML poll (the only signal that sees vLLM's
    #: CuMem allocations). For training it is ``max`` of the poll and
    #: ``non-torch baseline + torch_max_reserved``, because NVML polling can
    #: miss the brief backward-pass spike that torch's exact high-water mark
    #: captures.
    device_peak_bytes: int
    #: Raw NVML poll, kept for audit even when ``device_peak_bytes`` is the
    #: torch-corrected value.
    nvml_polled_bytes: int | None = None
    #: Training phase only: device memory resident with the engine asleep and
    #: the trainer idle (CUDA context + engine structures).
    sleeping_baseline_bytes: int | None = None
    torch_max_allocated_bytes: int | None = None
    torch_max_reserved_bytes: int | None = None
    analytic_bytes: int | None = None


class PhaseCalibration(BaseModel):
    """Fit plus its validation record for one phase."""

    model_config = ConfigDict(frozen=True)

    fit: ResidualFit = Field(default_factory=ResidualFit)
    #: Relative error on held-out knob combinations, e.g. 0.08 for 8%.
    holdout_max_rel_error: float | None = None
    holdout_mean_rel_error: float | None = None
    n_points: int = 0


class DeviceFingerprint(BaseModel):
    """Identity of the device a profile was measured on. Calibration is
    per (model, device) — kernel selection and workspace sizes differ across
    architectures — so cross-device application is flagged, not silent.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    total_bytes: int
    compute_capability: str | None = None


class ModelProfile(BaseModel):
    """The checked-in calibration artefact for one curated model variant."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    model_id: str
    #: Full model spec (geometry + variants), making the fixture
    #: self-contained: the widget and the CLI load one JSON per curated model.
    model_spec: ModelSpec | None = None
    trainer_variant: str = "base"
    generation_variant: str = "base"
    device: DeviceFingerprint | None = None
    #: Versions the constants were measured under; drift in these moves the
    #: constants (e.g. a change to the fused-logprob chunking).
    framework_versions: dict[str, str] = Field(default_factory=dict)
    training: PhaseCalibration = Field(default_factory=PhaseCalibration)
    generation: PhaseCalibration = Field(default_factory=PhaseCalibration)
    #: Realised in-memory weight bytes measured at load, keyed by variant
    #: name. Feeds ``WeightVariant.realised_bytes``.
    realised_weight_bytes: dict[str, int] = Field(default_factory=dict)
    measured: tuple[MeasuredPoint, ...] = ()

    def measured_on(self, device: DeviceSpec) -> bool:
        """Whether this profile was measured on the given device.

        Cross-device transfer was measured on two L4/A100 model pairs and is
        unreliable in both directions: on Qwen2.5-1.5B the foreign training
        fit roughly halved the error (5.4% vs 10.4% unfitted), while on
        Qwen2.5-0.5B it was slightly worse than none (11.6% vs 10.7%).
        Applying it is a net win on average, so the estimator does — but
        reports it as ``other_device`` rather than as a calibrated number.
        """
        if self.device is None or device.name is None:
            return True
        return self.device.name == device.name

    @property
    def sleeping_engine_residual_bytes(self) -> int | None:
        """Measured device memory the sleeping engine leaves behind, net of
        the CUDA context the estimator already counts under overhead.

        ``None`` when the profile predates the measurement, in which case the
        estimator falls back to its analytic constant.
        """
        baselines = sorted(
            point.sleeping_baseline_bytes
            for point in self.measured
            if point.phase == "training" and point.sleeping_baseline_bytes
        )
        if not baselines:
            return None
        median = baselines[len(baselines) // 2]
        return max(median - formulas.CUDA_CONTEXT_BYTES, 0)

    def apply_realised_weights(self, model: ModelSpec) -> ModelSpec:
        """Return a copy of ``model`` with profiled realised sizes attached to
        the matching weight variants.
        """
        if not self.realised_weight_bytes:
            return model
        variants = tuple(
            v.model_copy(update={"realised_bytes": self.realised_weight_bytes[v.name]})
            if v.name in self.realised_weight_bytes
            else v
            for v in model.variants
        )
        return model.model_copy(update={"variants": variants})


def _slug(text: str) -> str:
    return text.replace("/", "__").replace(" ", "-")


def profile_path(
    model_id: str,
    fixtures_dir: Path | None = None,
    device_name: str | None = None,
) -> Path:
    """Canonical fixture path for a (model, device) pair.

    Calibration is per device, not per model, so the device belongs in the
    filename — otherwise profiling the same model on a second GPU silently
    overwrites the first.
    """
    directory = fixtures_dir or FIXTURES_DIR
    stem = _slug(model_id)
    if device_name:
        stem = f"{stem}@{_slug(device_name)}"
    return directory / f"{stem}.json"


def _parse_stem(stem: str) -> tuple[str, str | None]:
    model_part, _, device_part = stem.partition("@")
    return model_part.replace("__", "/"), (
        device_part.replace("-", " ") if device_part else None
    )


def load_profile(
    model_id: str,
    fixtures_dir: Path | None = None,
    device_name: str | None = None,
) -> ModelProfile | None:
    """Load a curated model's profile, preferring one measured on
    ``device_name``.

    Falls back to any profile for the model when the device has none, since
    a foreign-device fit still beats no fit on average — the estimator marks
    it ``other_device`` rather than passing it off as calibrated.
    """
    directory = fixtures_dir or FIXTURES_DIR
    if not directory.exists():
        return None
    candidates = [
        path
        for path in sorted(directory.glob("*.json"))
        if _parse_stem(path.stem)[0] == model_id
    ]
    if not candidates:
        return None
    if device_name:
        for path in candidates:
            if _parse_stem(path.stem)[1] == device_name:
                return ModelProfile.model_validate_json(path.read_text())
    return ModelProfile.model_validate_json(candidates[0].read_text())


def curated_models(fixtures_dir: Path | None = None) -> list[str]:
    """Model ids with a checked-in profile — the curated list is exactly the
    set of profiled models.
    """
    directory = fixtures_dir or FIXTURES_DIR
    if not directory.exists():
        return []
    return sorted({_parse_stem(p.stem)[0] for p in directory.glob("*.json")})


def curated_profiles(fixtures_dir: Path | None = None) -> list[tuple[str, str | None]]:
    """Every checked-in (model, device) pair."""
    directory = fixtures_dir or FIXTURES_DIR
    if not directory.exists():
        return []
    return sorted(_parse_stem(p.stem) for p in directory.glob("*.json"))


def save_profile(profile: ModelProfile, fixtures_dir: Path | None = None) -> Path:
    path = profile_path(
        profile.model_id,
        fixtures_dir,
        device_name=profile.device.name if profile.device else None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return path
