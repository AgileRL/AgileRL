# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Calibration schema, fixture round-trips, and the residual fitter."""

import pytest

from agilerl.memory.specs import (
    ModelSpec,
    WeightVariant,
)
from tests.test_memory.test_formulas import QWEN_05B
from tools.memory_profiling.calibration import (
    ModelProfile,
    PhaseCalibration,
    ResidualFit,
    curated_models,
    load_profile,
    save_profile,
)
from tools.memory_profiling.sweep import HOLDOUT_POINTS, corner_plan, fit_residuals


def test_residual_fit_correction_arithmetic():
    fit = ResidualFit(
        intercept_bytes=1000.0, slopes={"grad_tokens": 2.0, "unknown": 5.0}
    )
    assert fit.correction_bytes({"grad_tokens": 10.0}) == 1000.0 + 20.0


def test_profile_round_trip_and_curated_listing(tmp_path):
    profile = ModelProfile(
        model_id="org/model",
        model_spec=ModelSpec(model_id="org/model", arch=QWEN_05B),
        realised_weight_bytes={"base": 988_000_000},
        training=PhaseCalibration(
            fit=ResidualFit(intercept_bytes=1e9, slopes={"grad_tokens": 100.0}),
            n_points=16,
        ),
    )
    path = save_profile(profile, tmp_path)
    assert path.name == "org__model.json"
    loaded = load_profile("org/model", tmp_path)
    assert loaded == profile
    assert curated_models(tmp_path) == ["org/model"]
    assert load_profile("missing/model", tmp_path) is None


def test_apply_realised_weights_overrides_variant():
    model = ModelSpec(
        model_id="org/model",
        arch=QWEN_05B,
        variants=(WeightVariant(), WeightVariant(name="nf4", quantization="nf4")),
    )
    profile = ModelProfile(model_id="org/model", realised_weight_bytes={"nf4": 777})
    updated = profile.apply_realised_weights(model)
    assert updated.variant("nf4").realised_bytes == 777
    assert updated.variant("base").realised_bytes is None


def test_fit_residuals_recovers_synthetic_linear_model():
    pytest.importorskip("numpy")
    true_intercept, true_slope = 5e8, 12.5
    basis_rows = [
        {"grad_tokens": float(t), "constant_term": 3.0}
        for t in (1024, 4096, 16384, 65536)
    ]
    residuals = [true_intercept + true_slope * row["grad_tokens"] for row in basis_rows]
    fit = fit_residuals(basis_rows, residuals)
    assert fit.intercept_bytes == pytest.approx(true_intercept, rel=1e-6)
    assert fit.slopes["grad_tokens"] == pytest.approx(true_slope, rel=1e-6)
    # Constant columns fold into the intercept rather than getting a slope.
    assert "constant_term" not in fit.slopes


def test_corner_plan_covers_all_corners():
    plan = corner_plan()
    assert len(plan) == 16
    assert len(set(plan)) == 16
    seqs = {p.seq_len for p in plan}
    assert seqs == {512, 4096}
    # Holdout points sit strictly inside the corner ranges.
    for point in HOLDOUT_POINTS:
        assert 512 < point.seq_len < 4096 or 1 < point.micro_batch < 8
