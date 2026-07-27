"""Fixture regression: checked-in profiles must still predict their own
measurements.

This is the CPU-side drift check. The calibration constants are fitted
against a specific analytic core, so any change to the formulas (a component
added, a term corrected) invalidates them silently — the estimator keeps
returning numbers, they are just wrong. Replaying every stored sweep point
through the current core catches that in CI without a GPU.

When this fails because the formulas legitimately improved, re-fit rather
than loosen the band::

    python -m agilerl.memory.profiling.refit
"""

import pytest

from agilerl.memory.calibration import curated_models, load_profile
from agilerl.memory.profiling.refit import ACCURACY_BAND, prediction_errors

CURATED = curated_models()


@pytest.mark.skipif(not CURATED, reason="No calibration fixtures checked in yet.")
@pytest.mark.parametrize("model_id", CURATED)
def test_fixture_predicts_its_own_measurements(model_id):
    profile = load_profile(model_id)
    assert profile is not None
    errors = prediction_errors(profile)
    assert errors, f"{model_id}: fixture carries no replayable measurements"
    for phase, phase_errors in errors.items():
        if not phase_errors:
            continue
        worst = max(phase_errors)
        assert worst <= ACCURACY_BAND, (
            f"{model_id} {phase}: worst error {worst:.1%} exceeds the "
            f"{ACCURACY_BAND:.0%} band — the analytic core changed under the "
            f"fitted constants. Re-fit with "
            f"`python -m agilerl.memory.profiling.refit`."
        )


@pytest.mark.skipif(not CURATED, reason="No calibration fixtures checked in yet.")
@pytest.mark.parametrize("model_id", CURATED)
def test_fixture_is_self_describing(model_id):
    # A fixture has to carry everything the widget and a re-fit need: the
    # geometry, the device it was measured on, and the raw points.
    profile = load_profile(model_id)
    assert profile is not None
    assert profile.model_spec is not None
    assert profile.device is not None
    assert profile.measured
    assert profile.framework_versions.get("vllm")
    assert profile.training.n_points > 0
    assert profile.generation.n_points > 0


@pytest.mark.skipif(not CURATED, reason="No calibration fixtures checked in yet.")
@pytest.mark.parametrize("model_id", CURATED)
def test_fixture_holdout_within_band(model_id):
    profile = load_profile(model_id)
    assert profile is not None
    for phase in ("training", "generation"):
        error = getattr(profile, phase).holdout_max_rel_error
        if error is not None:
            assert error <= ACCURACY_BAND, (
                f"{model_id} {phase}: holdout error {error:.1%} exceeds the "
                f"{ACCURACY_BAND:.0%} band."
            )
