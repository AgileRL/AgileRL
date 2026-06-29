"""Tests for the MF-PBT Pydantic spec and its manifest integration.

Covers ``MFPBTSpec`` field validation (frequency ratios, bracket sizing, extra
fields) and the ``TrainingManifest``-level rules: ``mf_pbt`` is mutually exclusive
with ``tournament_selection`` and with an explicit ``pop_size``, and otherwise
derives ``training.pop_size = n_subpopulations * n_individuals_per_subpopulation``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.models.hpo import MFPBTSpec
from agilerl.models.manifest import TrainingManifest

VALID_MF_PBT = {
    "n_subpopulations": 4,
    "n_individuals_per_subpopulation": 4,
    "evolution_frequency_ratios": [1, 2, 4, 8],
    "n_winners": 1,
    "n_survivors": 1,
    "n_open_for_migration": 1,
    "n_losers": 1,
}


def _manifest(training: dict, **sections) -> dict:
    data = {
        "algorithm": {"name": "DQN"},
        "environment": {"name": "CartPole-v1"},
        "training": training,
    }
    data.update(sections)
    return data


# --------------------------------------------------------------------------- #
# MFPBTSpec field validation
# --------------------------------------------------------------------------- #
def test_valid_spec_constructs():
    spec = MFPBTSpec(**VALID_MF_PBT)
    assert spec.n_subpopulations == 4
    assert spec.evolution_frequency_ratios == [1, 2, 4, 8]


def test_ratios_length_must_match_n_subpopulations():
    bad = {**VALID_MF_PBT, "evolution_frequency_ratios": [1, 2, 4]}
    with pytest.raises(ValidationError, match="length"):
        MFPBTSpec(**bad)


def test_ratios_must_be_strictly_increasing():
    bad = {**VALID_MF_PBT, "evolution_frequency_ratios": [1, 2, 2, 8]}
    with pytest.raises(ValidationError, match="strictly increasing"):
        MFPBTSpec(**bad)


def test_ratios_must_be_at_least_one():
    bad = {**VALID_MF_PBT, "evolution_frequency_ratios": [0, 2, 4, 8]}
    with pytest.raises(ValidationError, match=">= 1"):
        MFPBTSpec(**bad)


def test_bracket_sizes_must_sum_to_individuals():
    bad = {**VALID_MF_PBT, "n_losers": 2}  # 1+1+1+2 = 5 != 4
    with pytest.raises(
        ValidationError, match="must equal n_individuals_per_subpopulation"
    ):
        MFPBTSpec(**bad)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        MFPBTSpec(**{**VALID_MF_PBT, "unexpected": 1})


def test_zero_sized_brackets_allowed_when_sum_matches():
    # Brackets may be 0; they need only sum to n_individuals_per_subpopulation.
    spec = MFPBTSpec(
        **{**VALID_MF_PBT, "n_survivors": 0, "n_losers": 2}  # 1 + 0 + 1 + 2 = 4
    )
    assert spec.n_survivors == 0
    assert spec.n_losers == 2


def test_frozen_subpopulation_all_survivors_allowed():
    # No winners/losers/migration -> a subpopulation that never evolves is valid.
    spec = MFPBTSpec(
        **{
            **VALID_MF_PBT,
            "n_winners": 0,
            "n_survivors": 4,
            "n_open_for_migration": 0,
            "n_losers": 0,
        }
    )
    assert spec.n_winners == 0


def test_zero_winners_with_losers_rejected():
    bad = {**VALID_MF_PBT, "n_winners": 0, "n_survivors": 1, "n_losers": 2}  # sum 4
    with pytest.raises(ValidationError, match="n_winners"):
        MFPBTSpec(**bad)


def test_zero_winners_with_open_for_migration_rejected():
    bad = {
        **VALID_MF_PBT,
        "n_winners": 0,
        "n_survivors": 2,
        "n_open_for_migration": 2,
        "n_losers": 0,
    }  # sum 4
    with pytest.raises(ValidationError, match="n_winners"):
        MFPBTSpec(**bad)


# --------------------------------------------------------------------------- #
# TrainingManifest integration
# --------------------------------------------------------------------------- #
def test_pop_size_derived_from_mf_pbt():
    data = _manifest({"max_steps": 1000, "evo_steps": 100}, mf_pbt=VALID_MF_PBT)
    manifest = TrainingManifest.model_validate(data)
    assert manifest.training.pop_size == 16


def test_mf_pbt_and_tournament_selection_are_mutually_exclusive():
    data = _manifest(
        {"max_steps": 1000, "evo_steps": 100},
        mf_pbt=VALID_MF_PBT,
        tournament_selection={"tournament_size": 2, "elitism": True},
    )
    with pytest.raises(ValidationError, match="tournament_selection"):
        TrainingManifest.model_validate(data)


def test_mf_pbt_rejects_conflicting_pop_size():
    data = _manifest(
        {"max_steps": 1000, "evo_steps": 100, "pop_size": 8}, mf_pbt=VALID_MF_PBT
    )
    with pytest.raises(ValidationError, match="conflicts"):
        TrainingManifest.model_validate(data)


def test_mf_pbt_allows_matching_pop_size():
    # A pop_size equal to the derived value is tolerated (so the manifest round-trips
    # through Trainer.to_manifest, which re-emits the derived pop_size).
    data = _manifest(
        {"max_steps": 1000, "evo_steps": 100, "pop_size": 16}, mf_pbt=VALID_MF_PBT
    )
    manifest = TrainingManifest.model_validate(data)
    assert manifest.training.pop_size == 16
