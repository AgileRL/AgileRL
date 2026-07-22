"""Tests for the multi-frequency selection Pydantic spec and its manifest integration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    TournamentSelectionSpec,
    resolve_and_validate_multi_frequency_population,
)
from agilerl.models.manifest import TrainingManifest
from agilerl.models.training import TrainingSpec

# A fully-specified valid spec (for pop_size = 16)
VALID_MULTI_FREQUENCY = {
    "n_subpopulations": 4,
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
    """Field bounds and frequency-ratio validation (independent of population_size)."""

    def test_default_spec_resolves_ratios_and_leaves_brackets_unresolved(self):
        spec = MultiFrequencySelectionSpec()

        assert spec.n_subpopulations == 2
        assert spec.n_survivors == 0
        assert spec.evolution_frequency_ratios == [1, 5]
        # Bracket defaults need the population size, so they are not resolved here
        assert (spec.n_winners, spec.n_open_for_migration, spec.n_losers) == (
            None,
            None,
            None,
        )

    def test_ratios_default_scales_with_subpopulations(self):
        spec = MultiFrequencySelectionSpec(n_subpopulations=4)
        assert spec.evolution_frequency_ratios == [1, 5, 10, 15]

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

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "unexpected": 1})

    def test_removed_n_individuals_field_is_rejected(self):
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(n_individuals_per_subpopulation=8)

    def test_zero_winners_rejected(self):
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "n_winners": 0})

    def test_zero_open_for_migration_rejected(self):
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(
                **{**VALID_MULTI_FREQUENCY, "n_open_for_migration": 0}
            )

    def test_negative_survivors_rejected(self):
        with pytest.raises(ValidationError):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "n_survivors": -1})

    def test_fewer_than_two_subpopulations_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            MultiFrequencySelectionSpec(n_subpopulations=1)


class TestResolveAndValidateMultiFrequencyPopulation:
    """The pop_size-dependent finalization: mandatory pop_size + bracket resolution."""

    def test_resolves_bracket_defaults_onto_spec(self):
        spec = MultiFrequencySelectionSpec()
        training = TrainingSpec(pop_size=16)

        resolve_and_validate_multi_frequency_population(spec, training)

        assert spec.n_winners == 2
        assert spec.n_survivors == 0
        assert spec.n_open_for_migration == 2
        assert spec.n_losers == 4

    def test_accepts_matching_explicit_brackets(self):
        spec = MultiFrequencySelectionSpec(**VALID_MULTI_FREQUENCY)
        training = TrainingSpec(pop_size=16)  # 4 subpopulations of 4

        resolve_and_validate_multi_frequency_population(spec, training)

        assert spec.n_winners == 1
        assert spec.n_survivors == 1
        assert spec.n_open_for_migration == 1
        assert spec.n_losers == 1

    def test_rejects_missing_pop_size(self):
        spec = MultiFrequencySelectionSpec()
        training = TrainingSpec(max_steps=1000, evo_steps=100)

        with pytest.raises(ValueError, match="pop_size is required"):
            resolve_and_validate_multi_frequency_population(spec, training)

    def test_rejects_pop_size_below_six(self):
        spec = MultiFrequencySelectionSpec()
        training = TrainingSpec(pop_size=4)

        with pytest.raises(ValueError, match="population_size must be >= 6"):
            resolve_and_validate_multi_frequency_population(spec, training)

    def test_rejects_pop_size_not_divisible_by_subpopulations(self):
        spec = MultiFrequencySelectionSpec(n_subpopulations=4)
        training = TrainingSpec(pop_size=10)

        with pytest.raises(ValueError, match="must be divisible by n_subpopulations"):
            resolve_and_validate_multi_frequency_population(spec, training)

    def test_rejects_bracket_sizes_that_do_not_sum_to_subpopulation_size(self):
        spec = MultiFrequencySelectionSpec(
            n_subpopulations=2,
            n_winners=1,
            n_survivors=1,
            n_open_for_migration=1,
            n_losers=1,
        )
        training = TrainingSpec(pop_size=16)

        with pytest.raises(
            ValueError, match="must equal population_size // n_subpopulations"
        ):
            resolve_and_validate_multi_frequency_population(spec, training)

    def test_honors_population_size_alias(self):
        spec = MultiFrequencySelectionSpec()
        training = TrainingSpec(population_size=16)

        resolve_and_validate_multi_frequency_population(spec, training)

        assert spec.n_losers == 4

    def test_none_spec_leaves_training_untouched(self):
        training = TrainingSpec(pop_size=3)
        resolve_and_validate_multi_frequency_population(None, training)
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
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.tournament_selection, MultiFrequencySelectionSpec)
        assert manifest.tournament_selection.selection_strategy == "multi_frequency"

    def test_multi_frequency_resolves_brackets_from_explicit_pop_size(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            tournament_selection={"selection_strategy": "multi_frequency"},
        )
        manifest = TrainingManifest.model_validate(data)
        assert manifest.training.pop_size == 16
        assert manifest.tournament_selection.n_losers == 4  # 8 - 2 - 0 - 2

    def test_multi_frequency_requires_explicit_pop_size(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection={"selection_strategy": "multi_frequency"},
        )
        with pytest.raises(ValidationError, match="pop_size is required"):
            TrainingManifest.model_validate(data)

    def test_multi_frequency_rejects_pop_size_below_six(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 4},
            tournament_selection={"selection_strategy": "multi_frequency"},
        )
        with pytest.raises(ValidationError, match="population_size must be >= 6"):
            TrainingManifest.model_validate(data)

    def test_multi_frequency_rejects_pop_size_not_divisible(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 9},
            tournament_selection={
                "selection_strategy": "multi_frequency",
                "n_subpopulations": 2,
            },
        )
        with pytest.raises(ValidationError, match="must be divisible"):
            TrainingManifest.model_validate(data)

    def test_invalid_selection_strategy_rejected(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            tournament_selection={"selection_strategy": "bogus"},
        )
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(data)

    def test_tournament_keys_rejected_under_multi_frequency(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            tournament_selection={
                "selection_strategy": "multi_frequency",
                "tournament_size": 2,
            },
        )
        with pytest.raises(ValidationError):
            TrainingManifest.model_validate(data)

    def test_multi_frequency_manifest_dump_uses_unified_block(self):
        # Multi-frequency selection serializes under the single tournament_selection field, never a separate
        # multi_frequency_selection key
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        dumped = TrainingManifest.model_validate(data).model_dump(
            mode="json", exclude_none=True
        )
        assert dumped["tournament_selection"]["selection_strategy"] == "multi_frequency"
        assert "multi_frequency_selection" not in dumped
