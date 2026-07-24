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

from agilerl.memory.formulas import resolve_max_num_batched_tokens
from agilerl.memory.specs import (
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
    torch_max_allocated_bytes: int | None = None
    torch_max_reserved_bytes: int | None = None
    analytic_bytes: int | None = None


class PhaseCalibration(BaseModel):
    """Fit plus its validation record for one phase."""

    model_config = ConfigDict(frozen=True)

    fit: ResidualFit = Field(default_factory=ResidualFit)
    #: Relative error on held-out knob combinations, e.g. 0.08 for 8%.
    holdout_max_rel_error: float | None = None
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


def profile_path(model_id: str, fixtures_dir: Path | None = None) -> Path:
    """Canonical fixture path for a model id (slashes become double
    underscores).
    """
    directory = fixtures_dir or FIXTURES_DIR
    return directory / f"{model_id.replace('/', '__')}.json"


def load_profile(
    model_id: str, fixtures_dir: Path | None = None
) -> ModelProfile | None:
    """Load the checked-in profile for a curated model, or ``None`` when the
    model has not been profiled.
    """
    path = profile_path(model_id, fixtures_dir)
    if not path.exists():
        return None
    return ModelProfile.model_validate_json(path.read_text())


def curated_models(fixtures_dir: Path | None = None) -> list[str]:
    """Model ids with a checked-in profile — the curated list is exactly the
    set of profiled models.
    """
    directory = fixtures_dir or FIXTURES_DIR
    if not directory.exists():
        return []
    return sorted(p.stem.replace("__", "/") for p in directory.glob("*.json"))


def save_profile(profile: ModelProfile, fixtures_dir: Path | None = None) -> Path:
    path = profile_path(profile.model_id, fixtures_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return path
