"""Tests for the multi-frequency selection Pydantic spec and its manifest integration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    TournamentSelectionSpec,
    resolve_multi_frequency_selection_pop_size,
)
from agilerl.models.manifest import TrainingManifest
from agilerl.models.training import TrainingSpec

# A fully-specified valid spec
VALID_MULTI_FREQUENCY = {
    "n_subpopulations": 4,
    "n_individuals_per_subpopulation": 4,
    "evolution_frequency_ratios": [1, 2, 4, 8],
    "n_winners": 1,
    "n_survivors": 1,
    "n_open_for_migration": 1,
    "n_losers": 1,
}

# The same spec as it appears under the unified ``tournament_selection`` manifest block
VALID_MULTI_FREQUENCY_BLOCK = {
    "selection_strategy": "multi_frequency",
    **VALID_MULTI_FREQUENCY,
}


def _manifest(training: dict, **sections) -> dict:
    data = {
        "algorithm": {"name": "DQN"},
        "environment": {"name": "CartPole-v1"},
        "training": training,
    }
    data.update(sections)
    return data


class TestMultiFrequencySelectionSpec:
    def test_default_spec_is_recommended_configuration(self):
        spec = MultiFrequencySelectionSpec()
        assert spec.n_subpopulations == 2
        assert spec.n_individuals_per_subpopulation == 8
        # round(0.25 * 8) = 2 winners / 2 open, 0 survivors, remaining 4 losers
        assert (spec.n_winners, spec.n_survivors, spec.n_open_for_migration) == (
            2,
            0,
            2,
        )
        assert spec.n_losers == 4
        assert spec.evolution_frequency_ratios == [1, 5]

    def test_ratios_default_scales_with_subpopulations(self):
        spec = MultiFrequencySelectionSpec(
            n_subpopulations=4, n_individuals_per_subpopulation=8
        )
        assert spec.evolution_frequency_ratios == [1, 5, 10, 15]

    def test_losers_default_fills_the_remainder(self):
        spec = MultiFrequencySelectionSpec(
            n_individuals_per_subpopulation=8,
            n_winners=3,
            n_survivors=1,
            n_open_for_migration=1,
        )
        assert spec.n_losers == 3

    def test_valid_full_spec_constructs(self):
        spec = MultiFrequencySelectionSpec(**VALID_MULTI_FREQUENCY)
        assert spec.n_subpopulations == 4
        assert spec.evolution_frequency_ratios == [1, 2, 4, 8]

    def test_ratios_length_must_match_n_subpopulations(self):
        bad = {**VALID_MULTI_FREQUENCY, "evolution_frequency_ratios": [1, 2, 4]}
        with pytest.raises(ValidationError, match="length"):
            MultiFrequencySelectionSpec(**bad)

    def test_ratios_must_be_strictly_increasing(self):
        bad = {**VALID_MULTI_FREQUENCY, "evolution_frequency_ratios": [1, 2, 2, 8]}
        with pytest.raises(ValidationError, match="strictly increasing"):
            MultiFrequencySelectionSpec(**bad)

    def test_ratios_must_be_at_least_one(self):
        bad = {**VALID_MULTI_FREQUENCY, "evolution_frequency_ratios": [0, 2, 4, 8]}
        with pytest.raises(ValidationError, match=">= 1"):
            MultiFrequencySelectionSpec(**bad)

    def test_bracket_sizes_must_sum_to_individuals(self):
        bad = {**VALID_MULTI_FREQUENCY, "n_losers": 2}  # 1+1+1+2 = 5 != 4
        with pytest.raises(
            ValidationError, match="must equal n_individuals_per_subpopulation"
        ):
            MultiFrequencySelectionSpec(**bad)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "unexpected": 1})

    def test_zero_winners_rejected(self):
        bad = {**VALID_MULTI_FREQUENCY, "n_winners": 0}
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**bad)

    def test_zero_open_for_migration_rejected(self):
        bad = {**VALID_MULTI_FREQUENCY, "n_open_for_migration": 0}
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**bad)

    def test_zero_losers_rejected(self):
        bad = {**VALID_MULTI_FREQUENCY, "n_losers": 0, "n_survivors": 2}
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**bad)

    @pytest.mark.parametrize(
        ("n_survivors", "derived_losers"),
        [(1, 0), (2, -1)],  # n_ind - n_winners - n_survivors - n_open = 4 - 2 - s - 1
    )
    def test_derived_losers_non_positive_rejected(self, n_survivors, derived_losers):
        with pytest.raises(
            ValidationError, match=f"n_losers must be >= 1, got {derived_losers}"
        ):
            MultiFrequencySelectionSpec(
                n_subpopulations=2,
                n_individuals_per_subpopulation=4,
                n_winners=2,
                n_survivors=n_survivors,
                n_open_for_migration=1,
            )

    def test_fewer_than_two_subpopulations_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            MultiFrequencySelectionSpec(
                n_subpopulations=1, n_individuals_per_subpopulation=4
            )

    @pytest.mark.parametrize("n_ind", [1, 2])
    def test_fewer_than_three_individuals_rejected(self, n_ind):
        with pytest.raises(ValidationError, match="greater than or equal to 3"):
            MultiFrequencySelectionSpec(n_individuals_per_subpopulation=n_ind)

    def test_three_individuals_is_the_minimum_allowed(self):
        spec = MultiFrequencySelectionSpec(n_individuals_per_subpopulation=3)
        assert spec.n_individuals_per_subpopulation == 3
        assert (spec.n_winners, spec.n_survivors, spec.n_open_for_migration) == (
            1,
            0,
            1,
        )
        assert spec.n_losers == 1

    def test_survivors_may_be_zero(self):
        spec = MultiFrequencySelectionSpec(
            n_subpopulations=2,
            n_individuals_per_subpopulation=4,
            evolution_frequency_ratios=[1, 2],
            n_winners=2,
            n_survivors=0,
            n_open_for_migration=1,
            n_losers=1,
        )
        assert spec.n_survivors == 0

    def test_open_for_migration_may_exceed_winners_plus_survivors(self):
        # Migration sources migrants from the frozen pre-evolution snapshot rather than
        # the live population, so a subpopulation may open more slots for migration than
        # it preserves natively
        spec = MultiFrequencySelectionSpec(
            **{
                **VALID_MULTI_FREQUENCY,
                "n_winners": 1,
                "n_survivors": 0,
                "n_open_for_migration": 2,
                "n_losers": 1,
            }
        )
        assert spec.n_open_for_migration == 2


class TestResolvePopSize:
    def test_resolve_pop_size_derives_when_unset(self):
        training = TrainingSpec(max_steps=1000, evo_steps=100)
        resolve_multi_frequency_selection_pop_size(
            MultiFrequencySelectionSpec(), training
        )  # 2 subpopulations with 8 agents each
        assert training.pop_size == 16

    def test_resolve_pop_size_tolerates_matching_value(self):
        training = TrainingSpec(pop_size=16)
        resolve_multi_frequency_selection_pop_size(
            MultiFrequencySelectionSpec(), training
        )
        assert training.pop_size == 16

    def test_resolve_pop_size_rejects_conflicting_value(self):
        training = TrainingSpec(pop_size=8)
        with pytest.raises(ValueError, match="derived value"):
            resolve_multi_frequency_selection_pop_size(
                MultiFrequencySelectionSpec(), training
            )

    def test_resolve_pop_size_rejects_conflicting_population_size_alias(self):
        # The alias must be honoured too: population_size=8 is an explicit conflict.
        training = TrainingSpec(population_size=8)
        with pytest.raises(ValueError, match="derived value"):
            resolve_multi_frequency_selection_pop_size(
                MultiFrequencySelectionSpec(), training
            )

    def test_resolve_pop_size_none_spec_leaves_training_untouched(self):
        training = TrainingSpec(pop_size=3)
        resolve_multi_frequency_selection_pop_size(None, training)
        assert training.pop_size == 3


class TestManifestIntegration:
    def test_tournament_is_the_default_selection_strategy(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
            tournament_selection={"tournament_size": 3, "elitism": False},
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.tournament_selection, TournamentSelectionSpec)
        assert manifest.tournament_selection.selection_strategy == "tournament"
        assert manifest.tournament_selection.tournament_size == 3

    def test_explicit_tournament_selection_strategy(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
            tournament_selection={
                "selection_strategy": "tournament",
                "tournament_size": 2,
            },
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.tournament_selection, TournamentSelectionSpec)

    def test_omitted_selection_block_leaves_none(self):
        # Omitting the block is how the no-HPO regime is selected
        data = _manifest({"max_steps": 1000, "evo_steps": 100, "pop_size": 2})
        manifest = TrainingManifest.model_validate(data)
        assert manifest.tournament_selection is None

    def test_multi_frequency_routes_to_spec(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.tournament_selection, MultiFrequencySelectionSpec)
        assert manifest.tournament_selection.selection_strategy == "multi_frequency"

    def test_pop_size_derived_from_multi_frequency(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.training.pop_size == 16

    def test_default_multi_frequency_derives_pop_size(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection={"selection_strategy": "multi_frequency"},
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.training.pop_size == 16  # 2 subpops x 8 individuals

    def test_invalid_selection_strategy_rejected(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection={"selection_strategy": "bogus"},
        )
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(data)

    def test_tournament_keys_rejected_under_multi_frequency(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection={
                "selection_strategy": "multi_frequency",
                "tournament_size": 2,
            },
        )
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(data)

    def test_multi_frequency_rejects_conflicting_pop_size(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 8},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        with pytest.raises(ValidationError, match="conflicts"):
            TrainingManifest.model_validate(data)

    def test_multi_frequency_allows_matching_pop_size(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.training.pop_size == 16

    def test_multi_frequency_manifest_dump_uses_unified_block(self):
        # Multi-frequency selection serializes under the single tournament_selection field, never a separate
        # multi_frequency_selection key
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        dumped = TrainingManifest.model_validate(data).model_dump(
            mode="json", exclude_none=True
        )
        assert dumped["tournament_selection"]["selection_strategy"] == "multi_frequency"
        assert "multi_frequency_selection" not in dumped
