# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the multi-frequency selection Pydantic spec and its manifest integration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agilerl.models.hpo import (
    MultiFrequencySelectionSpec,
    TournamentSelectionSpec,
)
from agilerl.models.manifest import TrainingManifest

# A fully-specified valid spec (for pop_size = 16)
VALID_MULTI_FREQUENCY = {
    "n_subpopulations": 4,
    "evolution_frequency_ratios": [1, 2, 4, 8],
    "n_winners": 1,
    "n_survivors": 1,
    "n_open_for_migration": 1,
    "n_losers": 1,
}

# The same spec as it appears under the unified ``selection_strategy`` manifest block
VALID_MULTI_FREQUENCY_BLOCK = {
    "strategy": "multi_frequency",
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


def _validated_multi_frequency(
    training: dict, **multi_frequency
) -> MultiFrequencySelectionSpec:
    """Validate a manifest under MF-PBT and return its finalized selection spec."""
    data = _manifest(
        training,
        selection_strategy={
            "strategy": "multi_frequency",
            **multi_frequency,
        },
    )
    return TrainingManifest.model_validate(data).selection_strategy


class TestMultiFrequencySelectionSpec:
    """Field bounds and frequency-ratio validation (independent of population_size)."""

    def test_default_spec_resolves_ratios_and_leaves_brackets_unresolved(self):
        spec = MultiFrequencySelectionSpec()

        assert spec.n_subpopulations == 2
        assert spec.evolution_frequency_ratios == [1, 5]
        assert (
            spec.n_winners,
            spec.n_survivors,
            spec.n_open_for_migration,
            spec.n_losers,
        ) == (None, None, None, None)

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
        with pytest.raises(
            ValidationError, match=r"unexpected\s+Extra inputs are not permitted"
        ):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "unexpected": 1})

    def test_removed_n_individuals_field_is_rejected(self):
        with pytest.raises(
            ValidationError,
            match=r"n_individuals_per_subpopulation\s+Extra inputs are not permitted",
        ):
            MultiFrequencySelectionSpec(n_individuals_per_subpopulation=8)

    def test_zero_winners_rejected(self):
        with pytest.raises(
            ValidationError,
            match=r"n_winners\s+Input should be greater than or equal to 1",
        ):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "n_winners": 0})

    def test_zero_open_for_migration_rejected(self):
        with pytest.raises(
            ValidationError,
            match=r"n_open_for_migration\s+Input should be greater than or equal to 1",
        ):
            MultiFrequencySelectionSpec(
                **{**VALID_MULTI_FREQUENCY, "n_open_for_migration": 0}
            )

    def test_negative_survivors_rejected(self):
        with pytest.raises(
            ValidationError,
            match=r"n_survivors\s+Input should be greater than or equal to 0",
        ):
            MultiFrequencySelectionSpec(**{**VALID_MULTI_FREQUENCY, "n_survivors": -1})

    def test_fewer_than_two_subpopulations_rejected(self):
        with pytest.raises(ValidationError, match="greater than or equal to 2"):
            MultiFrequencySelectionSpec(n_subpopulations=1)


class TestMultiFrequencyPopulationLayout:
    """The pop_size-dependent finalization: mandatory pop_size + bracket resolution.

    The manifest defers this to the operator's own resolver, so these cases pin the
    layout rules as the manifest applies them end to end.
    """

    def test_resolves_bracket_defaults_onto_spec(self):
        spec = _validated_multi_frequency({"max_steps": 1000, "pop_size": 16})

        assert spec.n_winners == 2
        assert spec.n_survivors == 0
        assert spec.n_open_for_migration == 2
        assert spec.n_losers == 4

    def test_accepts_matching_explicit_brackets(self):
        # 4 subpopulations of 4
        spec = _validated_multi_frequency(
            {"max_steps": 1000, "pop_size": 16}, **VALID_MULTI_FREQUENCY
        )

        assert spec.n_winners == 1
        assert spec.n_survivors == 1
        assert spec.n_open_for_migration == 1
        assert spec.n_losers == 1

    def test_rejects_missing_pop_size(self):
        with pytest.raises(ValidationError, match="pop_size is required"):
            _validated_multi_frequency({"max_steps": 1000, "evo_steps": 100})

    def test_rejects_pop_size_below_six(self):
        with pytest.raises(ValidationError, match="population_size must be >= 6"):
            _validated_multi_frequency({"max_steps": 1000, "pop_size": 4})

    def test_rejects_pop_size_not_divisible_by_subpopulations(self):
        with pytest.raises(
            ValidationError, match="must be divisible by n_subpopulations"
        ):
            _validated_multi_frequency(
                {"max_steps": 1000, "pop_size": 10}, n_subpopulations=4
            )

    @pytest.mark.parametrize(("pop_size", "n_subpopulations"), [(6, 3), (8, 4)])
    def test_rejects_subpopulation_size_below_three(self, pop_size, n_subpopulations):
        with pytest.raises(ValidationError, match="must be >= 3 so each subpopulation"):
            _validated_multi_frequency(
                {"max_steps": 1000, "pop_size": pop_size},
                n_subpopulations=n_subpopulations,
            )

    @pytest.mark.parametrize(
        ("pop_size", "n_subpopulations"), [(6, 2), (9, 3), (12, 4)]
    )
    def test_accepts_smallest_valid_subpopulation(self, pop_size, n_subpopulations):
        spec = _validated_multi_frequency(
            {"max_steps": 1000, "pop_size": pop_size},
            n_subpopulations=n_subpopulations,
        )

        assert (spec.n_winners, spec.n_open_for_migration, spec.n_losers) == (1, 1, 1)

    def test_rejects_bracket_sizes_that_do_not_sum_to_subpopulation_size(self):
        with pytest.raises(
            ValidationError, match="must equal population_size // n_subpopulations"
        ):
            _validated_multi_frequency(
                {"max_steps": 1000, "pop_size": 16},
                n_subpopulations=2,
                n_winners=1,
                n_survivors=1,
                n_open_for_migration=1,
                n_losers=1,
            )

    def test_rejects_negative_derived_n_losers(self):
        # With n_losers left to default, the remainder can go negative when the explicit
        # brackets already over-fill the subpopulation (4 - 2 - 2 - 1 = -1 here)
        with pytest.raises(ValidationError, match="n_losers must be >= 1, got -1"):
            _validated_multi_frequency(
                {"max_steps": 1000, "pop_size": 8},  # subpopulation_size = 4
                n_subpopulations=2,
                n_winners=2,
                n_survivors=2,
                n_open_for_migration=1,
            )

    def test_honors_population_size_alias(self):
        spec = _validated_multi_frequency({"max_steps": 1000, "population_size": 16})

        assert spec.n_losers == 4


class TestManifestIntegration:
    def test_tournament_is_the_default_selection_strategy(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
            selection_strategy={"tournament_size": 3, "elitism": False},
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)
        assert manifest.selection_strategy.strategy == "tournament"
        assert manifest.selection_strategy.tournament_size == 3

    def test_explicit_tournament_selection_strategy(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
            selection_strategy={
                "strategy": "tournament",
                "tournament_size": 2,
            },
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)

    def test_omitted_selection_block_leaves_none(self):
        # Omitting the block is how the no-HPO regime is selected
        data = _manifest({"max_steps": 1000, "evo_steps": 100, "pop_size": 2})
        manifest = TrainingManifest.model_validate(data)
        assert manifest.selection_strategy is None

    def test_multi_frequency_routes_to_spec(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            selection_strategy=VALID_MULTI_FREQUENCY_BLOCK,
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.selection_strategy, MultiFrequencySelectionSpec)
        assert manifest.selection_strategy.strategy == "multi_frequency"

    def test_invalid_selection_strategy_rejected(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100},
            selection_strategy={"strategy": "bogus"},
        )
        with pytest.raises(
            ValidationError,
            match=r"Input tag 'bogus' .* does not match any of the expected tags",
        ):
            TrainingManifest.model_validate(data)

    def test_tournament_keys_rejected_under_multi_frequency(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            selection_strategy={
                "strategy": "multi_frequency",
                "tournament_size": 2,
            },
        )
        with pytest.raises(
            ValidationError,
            match=r"tournament_size\s+Extra inputs are not permitted",
        ):
            TrainingManifest.model_validate(data)

    def test_multi_frequency_manifest_dump_uses_unified_block(self):
        # Multi-frequency selection serializes under the single selection_strategy
        # field, never a separate multi_frequency_selection key
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            selection_strategy=VALID_MULTI_FREQUENCY_BLOCK,
        )
        dumped = TrainingManifest.model_validate(data).model_dump(
            mode="json", exclude_none=True
        )
        assert dumped["selection_strategy"]["strategy"] == "multi_frequency"
        assert "multi_frequency_selection" not in dumped


class TestSelectionStrategyBackwardsCompatibility:
    """The block's pre-rename key tournament_selection is still accepted."""

    def test_legacy_manifest_key_resolves_to_selection_strategy(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 2},
            tournament_selection={"tournament_size": 3, "elitism": False},
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.selection_strategy, TournamentSelectionSpec)
        assert manifest.selection_strategy.tournament_size == 3

    def test_legacy_manifest_key_carries_multi_frequency(self):
        data = _manifest(
            {"max_steps": 1000, "evo_steps": 100, "pop_size": 16},
            tournament_selection=VALID_MULTI_FREQUENCY_BLOCK,
        )
        manifest = TrainingManifest.model_validate(data)
        assert isinstance(manifest.selection_strategy, MultiFrequencySelectionSpec)

    def test_both_manifest_keys_produce_equal_specs(self):
        training = {"max_steps": 1000, "evo_steps": 100, "pop_size": 16}
        legacy = TrainingManifest.model_validate(
            _manifest(training, tournament_selection=VALID_MULTI_FREQUENCY_BLOCK)
        )
        current = TrainingManifest.model_validate(
            _manifest(training, selection_strategy=VALID_MULTI_FREQUENCY_BLOCK)
        )
        assert legacy.selection_strategy == current.selection_strategy

    def test_keyword_construction_accepts_either_spelling(self):
        common = {
            "algorithm": {"name": "DQN"},
            "environment": {"name": "CartPole-v1"},
            "training": {"max_steps": 1000, "pop_size": 2},
        }
        legacy = TrainingManifest(**common, tournament_selection={"tournament_size": 3})
        current = TrainingManifest(**common, selection_strategy={"tournament_size": 3})
        assert legacy.selection_strategy == current.selection_strategy

    def test_from_trainer_specs_accepts_either_spelling(self):
        from agilerl.models.algorithms.dqn import DQNSpec
        from agilerl.models.env import GymEnvSpec
        from agilerl.models.training import TrainingSpec

        common = {
            "algorithm": DQNSpec(),
            "environment": GymEnvSpec(name="CartPole-v1"),
            "training": TrainingSpec(max_steps=1000, pop_size=2),
        }
        current = TrainingManifest.from_trainer_specs(
            **common, selection_strategy=TournamentSelectionSpec(tournament_size=3)
        )
        with pytest.warns(DeprecationWarning, match="'tournament_selection' argument"):
            legacy = TrainingManifest.from_trainer_specs(
                **common,
                tournament_selection=TournamentSelectionSpec(tournament_size=3),
            )
        assert legacy.selection_strategy == current.selection_strategy
