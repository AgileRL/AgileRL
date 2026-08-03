# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the multi-frequency selection evolution loop."""

from __future__ import annotations

import numpy as np
import pytest

import agilerl.hpo.multi_frequency as mf_module
from agilerl.hpo.multi_frequency import (
    MultiFrequencyOp,
    MultiFrequencySelection,
    resolve_and_validate_frequency_ratios,
)
from tests.helper_functions import (
    FakeOptimizerWrapper,
    FakeRegistry,
    FakeSelectionAgent,
    make_fake_selection_population,
    make_multi_frequency_selection,
    new_agents,
)


def run_migration(strategy, population, subpop, external_pool):
    """Bracket then migrate."""
    winners, _survivors, open_for_migration, _losers = strategy._bracket_subpopulation(
        population, subpop
    )
    return strategy._migrate(
        population, subpop, winners, open_for_migration, external_pool
    )


class TestResolveAndValidateFrequencyRatios:
    """The frequency-ratio invariant shared by the operator and its Pydantic spec."""

    def test_none_resolves_to_the_default_ladder(self):
        assert resolve_and_validate_frequency_ratios(None, 3) == [1, 5, 10]

    def test_empty_list_resolves_to_the_default_ladder(self):
        assert resolve_and_validate_frequency_ratios([], 2) == [1, 5]

    def test_explicit_ratios_are_returned_as_a_detached_copy(self):
        configured = [1, 3]

        resolved = resolve_and_validate_frequency_ratios(configured, 2)

        assert resolved == [1, 3]
        assert resolved is not configured

    @pytest.mark.parametrize(
        ("ratios", "match"),
        [
            ([1], "length n_subpopulations"),
            ([1, 2, 3], "length n_subpopulations"),
            ([1, 1], "strictly increasing"),
            ([2, 1], "strictly increasing"),
            ([0, 2], ">= 1"),
        ],
    )
    def test_rejects_invalid_ratios(self, ratios, match):
        with pytest.raises(ValueError, match=match):
            resolve_and_validate_frequency_ratios(ratios, 2)

    @pytest.mark.parametrize(
        ("ratios", "match"),
        [
            ([1], "length n_subpopulations"),
            ([1, 1], "strictly increasing"),
            ([0, 2], ">= 1"),
        ],
    )
    def test_operator_and_spec_reject_the_same_ratios(self, ratios, match):
        from pydantic import ValidationError

        from agilerl.models.hpo import MultiFrequencySelectionSpec

        with pytest.raises(ValueError, match=match):
            MultiFrequencySelection(
                population_size=8,
                n_subpopulations=2,
                evolution_frequency_ratios=ratios,
            )

        with pytest.raises(ValidationError, match=match):
            MultiFrequencySelectionSpec(
                n_subpopulations=2, evolution_frequency_ratios=ratios
            )

    def test_operator_and_spec_resolve_the_same_defaults(self):
        from agilerl.models.hpo import MultiFrequencySelectionSpec

        spec = MultiFrequencySelectionSpec(n_subpopulations=3)
        operator = MultiFrequencySelection(population_size=9, n_subpopulations=3)

        assert operator.deltas == spec.evolution_frequency_ratios


class TestMultiFrequencySelectionInit:
    def test_valid_construction_sets_derived_attributes(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 3]
        )
        assert strategy.population_size == 8
        assert strategy.subpopulation_size == 4
        assert strategy.deltas == [1, 3]
        assert strategy.bracket_sizes == (1, 1, 1, 1)
        assert strategy.counters == [0, 0]

    def test_defaults_resolve_to_recommended_configuration(self):
        strategy = MultiFrequencySelection(population_size=16)
        assert strategy.n_subpopulations == 2
        assert strategy.subpopulation_size == 8
        assert strategy.population_size == 16
        assert strategy.bracket_sizes == (2, 0, 2, 4)
        assert strategy.deltas == [1, 5]

    def test_ratios_default_scales_with_subpopulations(self):
        strategy = MultiFrequencySelection(population_size=32, n_subpopulations=4)
        assert strategy.deltas == [1, 5, 10, 15]

    def test_empty_ratios_list_resolves_to_default(self):
        strategy = MultiFrequencySelection(
            population_size=16,
            n_subpopulations=2,
            evolution_frequency_ratios=[],
        )
        assert strategy.deltas == [1, 5]

    def test_losers_default_fills_the_remainder(self):
        strategy = MultiFrequencySelection(
            population_size=16,
            n_subpopulations=2,
            n_winners=3,
            n_survivors=1,
            n_open_for_migration=1,
        )
        assert strategy.n_losers == 3  # 8 - 3 - 1 - 1

    @pytest.mark.parametrize("population_size", [3, 5])
    def test_init_rejects_population_size_below_six(self, population_size):
        with pytest.raises(ValueError, match="population_size must be >= 6"):
            MultiFrequencySelection(population_size=population_size, n_subpopulations=2)

    @pytest.mark.parametrize(
        ("population_size", "n_subpopulations"), [(6, 3), (8, 4), (10, 5), (12, 6)]
    )
    def test_init_rejects_subpopulation_size_below_three(
        self, population_size, n_subpopulations
    ):
        with pytest.raises(ValueError, match="must be >= 3 so each subpopulation"):
            MultiFrequencySelection(
                population_size=population_size, n_subpopulations=n_subpopulations
            )

    @pytest.mark.parametrize(
        ("population_size", "n_subpopulations"), [(6, 2), (9, 3), (12, 4)]
    )
    def test_init_accepts_smallest_valid_subpopulation(
        self, population_size, n_subpopulations
    ):
        strategy = MultiFrequencySelection(
            population_size=population_size, n_subpopulations=n_subpopulations
        )
        assert strategy.subpopulation_size == 3
        assert strategy.bracket_sizes == (1, 0, 1, 1)

    def test_init_rejects_population_size_not_divisible_by_subpopulations(self):
        with pytest.raises(ValueError, match="must be divisible by n_subpopulations"):
            MultiFrequencySelection(population_size=9, n_subpopulations=2)

    def test_init_rejects_zero_open_for_migration(self):
        with pytest.raises(ValueError, match="n_open_for_migration must be >= 1"):
            MultiFrequencySelection(
                population_size=8,
                n_subpopulations=2,
                n_winners=1,
                n_survivors=1,
                n_open_for_migration=0,
                n_losers=2,
            )

    def test_init_rejects_negative_survivors(self):
        with pytest.raises(ValueError, match="n_survivors must be >= 0"):
            MultiFrequencySelection(
                population_size=8,
                n_subpopulations=2,
                n_winners=1,
                n_survivors=-1,
                n_open_for_migration=1,
                n_losers=3,
            )

    @pytest.mark.parametrize(
        ("n_survivors", "derived_losers"),
        [
            (1, 0),
            (2, -1),
        ],  # subpop_size - n_winners - n_survivors - n_open = 4 - 2 - s - 1
    )
    def test_init_rejects_derived_losers_non_positive(
        self, n_survivors, derived_losers
    ):
        with pytest.raises(
            ValueError, match=f"n_losers must be >= 1, got {derived_losers}"
        ):
            MultiFrequencySelection(
                population_size=8,
                n_subpopulations=2,
                n_winners=2,
                n_survivors=n_survivors,
                n_open_for_migration=1,
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            (
                {"w": 0, "s": 1, "o": 1, "ln": 2},
                "n_winners must be >= 1",
            ),
            (
                {"w": 1, "s": 1, "o": 1, "ln": 2},
                "must equal population_size // n_subpopulations",
            ),  # sum 5 != 4
        ],
    )
    def test_init_rejects_invalid_brackets(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_multi_frequency_selection(
                n_subpop=2, population_size=8, ratios=[1, 2], **kwargs
            )

    def test_open_for_migration_may_exceed_winners_plus_survivors(self):
        # Migration sources migrants from the frozen pre-evolution snapshot, so a
        # subpopulation may open more slots for migration than it preserves natively
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2], w=1, s=0, o=2, ln=1
        )
        assert strategy.bracket_sizes == (1, 0, 2, 1)

    @pytest.mark.parametrize(
        ("ratios", "match"),
        [
            ([1], "length n_subpopulations"),
            ([1, 1], "strictly increasing"),
            ([0, 2], ">= 1"),
        ],
    )
    def test_init_rejects_invalid_ratios(self, ratios, match):
        with pytest.raises(ValueError, match=match):
            make_multi_frequency_selection(n_subpop=2, population_size=8, ratios=ratios)

    def test_init_rejects_negative_bracket(self):
        with pytest.raises(ValueError, match="n_losers must be >= 1, got -1"):
            MultiFrequencySelection(
                population_size=8,
                n_subpopulations=2,
                evolution_frequency_ratios=[1, 2],
                n_winners=2,
                n_survivors=2,
                n_open_for_migration=1,
                n_losers=-1,
            )

    def test_frozen_subpopulation_rejected(self):
        # A "frozen" subpopulation should raise an error
        with pytest.raises(ValueError, match="n_winners must be >= 1"):
            make_multi_frequency_selection(
                n_subpop=2, population_size=8, ratios=[1, 2], w=0, s=4, o=0, ln=0
            )

    def test_init_rejects_fewer_than_two_subpopulations(self):
        with pytest.raises(ValueError, match="n_subpopulations must be >= 2"):
            MultiFrequencySelection(
                population_size=6,
                n_subpopulations=1,
                evolution_frequency_ratios=[1],
                n_winners=1,
                n_survivors=1,
                n_open_for_migration=1,
                n_losers=1,
            )


class TestRanking:
    @pytest.mark.parametrize(
        "fitness",
        [
            [{"a": 9.0, "b": 0.0}, {"a": 5.0, "b": 5.0}, {"a": 1.0, "b": 3.0}],
            [np.array([9.0, 0.0]), np.array([5.0, 5.0]), np.array([1.0, 3.0])],
        ],
        ids=["dict", "array"],
    )
    def test_rank_orders_vector_fitness_by_its_mean(self, fitness):
        strategy = make_multi_frequency_selection()
        pop = [
            FakeSelectionAgent(index, 0, value) for index, value in enumerate(fitness)
        ]

        assert [a.index for a in strategy._rank(pop)] == [1, 0, 2]

    def test_rank_orders_scalar_fitness_highest_first(self):
        strategy = make_multi_frequency_selection()
        pop = [
            FakeSelectionAgent(index, 0, value)
            for index, value in enumerate([1.0, 5.0, 3.0])
        ]

        assert [a.index for a in strategy._rank(pop)] == [1, 2, 0]


class TestSubpopulationAssignment:
    def test_subpopulation_for_position_maps_contiguous_blocks(self):
        fn = MultiFrequencySelection._subpopulation_for_position
        assert [fn(i, 4) for i in range(12)] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_assign_initial_subpopulations_tags_by_position_not_index(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        indices = [10, 11, 12, 13, 20, 21, 22, 23]
        pop = [FakeSelectionAgent(idx, None, fitness=0.0) for idx in indices]

        strategy._assign_initial_subpopulations(pop)

        assert [a.subpopulation_id for a in pop] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_assign_initial_subpopulations_tags_only_untagged_agents(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = [FakeSelectionAgent(i, None, fitness=0.0) for i in range(8)]
        pop[0] = FakeSelectionAgent(0, 1, fitness=0.0)
        pop[5] = FakeSelectionAgent(5, 0, fitness=0.0)

        strategy._assign_initial_subpopulations(pop)

        assert [a.subpopulation_id for a in pop] == [1, 0, 0, 0, 1, 0, 1, 1]

    @pytest.mark.parametrize("size", [6, 10])
    def test_assign_initial_subpopulations_rejects_wrong_population_size(self, size):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8
        )  # pop_size == 8
        pop = [FakeSelectionAgent(i, None, fitness=0.0) for i in range(size)]

        with pytest.raises(ValueError, match=f"{size} agents, expected 8"):
            strategy._assign_initial_subpopulations(pop)

    def test_assign_initial_subpopulations_accepts_correct_population_size(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8
        )  # pop_size == 8
        pop = [FakeSelectionAgent(i, None, fitness=0.0) for i in range(8)]

        strategy._assign_initial_subpopulations(pop)

        assert [a.subpopulation_id for a in pop] == [0, 0, 0, 0, 1, 1, 1, 1]

    @pytest.mark.parametrize(
        "indices",
        [[3] * 8, [0, 1, 2, 3, 4, 5, 6, 6]],
        ids=["resumed-from-one-checkpoint", "single-duplicated-pair"],
    )
    def test_assign_initial_subpopulations_rejects_duplicate_indices(self, indices):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = [
            FakeSelectionAgent(idx, i // 4, fitness=0.0)
            for i, idx in enumerate(indices)
        ]

        with pytest.raises(ValueError, match="globally-unique agent indices"):
            strategy._assign_initial_subpopulations(pop)


class TestIndexAllocation:
    def test_next_index_issues_fresh_indices_above_the_populations_maximum(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )  # indices 0 to 7

        strategy._sync_index(pop)

        assert [strategy._next_index() for _ in range(3)] == [8, 9, 10]

    def test_sync_index_never_lowers_the_allocator(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )
        strategy._sync_index(pop)
        assert strategy._next_index() == 8

        strategy._sync_index(pop[:4])  # a lower-indexed subset

        assert strategy._next_index() == 9  # not 5, the subset's max index + 1

    def test_next_index_before_sync_index_raises(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)

        with pytest.raises(RuntimeError, match="must seed the allocator"):
            strategy._next_index()


class TestBrackets:
    @pytest.mark.parametrize(
        ("w", "s", "o", "ln", "expected"),
        [
            (1, 1, 1, 1, ([4.0], [3.0], [2.0], [1.0])),
            (2, 0, 1, 1, ([4.0, 3.0], [], [2.0], [1.0])),  # a zero-size slice is empty
            (1, 0, 1, 2, ([4.0], [], [3.0], [2.0, 1.0])),
        ],
    )
    def test_brackets_partition_subpop_by_descending_fitness(
        self, w, s, o, ln, expected
    ):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, w=w, s=s, o=o, ln=ln
        )
        pop = make_fake_selection_population(
            {0: [2.0, 4.0, 1.0, 3.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )

        winners, survivors, open_, losers = strategy._bracket_subpopulation(
            pop, subpop=0
        )

        assert [a.fitness[-1] for a in winners] == expected[0]
        assert [a.fitness[-1] for a in survivors] == expected[1]
        assert [a.fitness[-1] for a in open_] == expected[2]
        assert [a.fitness[-1] for a in losers] == expected[3]
        # Only the studied subpopulation's agents are ever bracketed
        for bracket in (winners, survivors, open_, losers):
            assert all(a.subpopulation_id == 0 for a in bracket)

    def test_brackets_select_the_requested_subpopulation(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )

        winners, survivors, open_, losers = strategy._bracket_subpopulation(
            pop, subpop=1
        )

        assert [a.fitness[-1] for a in winners] == [8.0]
        assert [a.fitness[-1] for a in survivors] == [7.0]
        assert [a.fitness[-1] for a in open_] == [6.0]
        assert [a.fitness[-1] for a in losers] == [5.0]
        for bracket in (winners, survivors, open_, losers):
            assert all(a.subpopulation_id == 1 for a in bracket)

    def test_brackets_reject_wrong_member_count(self):
        # A mis-tagged subpop is rejected up front rather than silently mis-sliced
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = [FakeSelectionAgent(i, 0, fitness=float(i)) for i in range(3)]

        with pytest.raises(
            ValueError, match="Subpopulation 0 has 3 members, expected 4"
        ):
            strategy._bracket_subpopulation(pop, subpop=0)


class TestCloneWinnersOverLosers:
    def test_clone_winners_over_losers_replaces_losers_with_fresh_winner_clones(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )
        max_index_before = max(a.index for a in pop)
        winners, _s, _o, losers = strategy._bracket_subpopulation(pop, subpop=0)

        new_pop, new_indices = strategy._clone_winners_over_losers(
            pop, winners, losers, subpop=0
        )

        assert len(new_pop) == len(pop)
        assert sum(a.subpopulation_id == 0 for a in new_pop) == 4
        # The loser (fitness 1.0) was replaced
        assert 1.0 not in [a.fitness[-1] for a in new_pop if a.subpopulation_id == 0]
        clones = new_agents(pop, new_pop)
        assert len(clones) == 1
        clone = clones[0]
        assert clone.index in new_indices
        assert clone.index > max_index_before
        assert clone.subpopulation_id == 0
        # The clone inherits its winner-parent's fitness (4.0)
        assert clone.fitness[-1] == 4.0

    def test_clone_winners_over_losers_returns_new_list(self):
        # Cloning must return a new population rather than mutating the one it's handed
        strategy = make_multi_frequency_selection()
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )
        pop_copy = list(pop)
        winners, _s, _o, losers = strategy._bracket_subpopulation(pop, subpop=0)

        strategy._clone_winners_over_losers(pop, winners, losers, subpop=0)

        assert pop == pop_copy

    def test_clone_winners_over_losers_clones_are_independent_of_parents(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, w=1, s=0, o=1, ln=2
        )
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )
        winner = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 4.0
        )
        winners, _s, _o, losers = strategy._bracket_subpopulation(pop, subpop=0)

        new_pop, _ = strategy._clone_winners_over_losers(pop, winners, losers, subpop=0)

        for clone in new_agents(pop, new_pop):
            assert clone is not winner
            assert clone.index != winner.index
            # Mutating the clone's HP must not touch the parent's
            clone.registry.hp_config["lr"].value = 999.0
            assert winner.registry.hp_config["lr"].value != 999.0

    def test_clone_winners_over_losers_with_zero_survivors_replaces_every_loser(self):
        # Brackets 1/0/1/2: one winner, no survivors, one open, two losers.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, w=1, s=0, o=1, ln=2
        )
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )
        winners, _s, _o, losers = strategy._bracket_subpopulation(pop, subpop=0)

        new_pop, new_indices = strategy._clone_winners_over_losers(
            pop, winners, losers, subpop=0
        )

        assert sum(a.subpopulation_id == 0 for a in new_pop) == 4
        clones = new_agents(pop, new_pop)
        assert len(clones) == 2
        assert {c.index for c in clones} == set(new_indices)

    def test_clone_winner_selection_is_reproducible(self):
        weights = {0: ["W_a", "W_b", "m", "x", "y"], 1: ["a", "b", "c", "d", "e"]}
        pop_kwargs = {
            "subpop_fitnesses": {0: [5.0, 4.0, 3.0, 2.0, 1.0], 1: [10, 9, 8, 7, 6]},
            "weights": weights,
        }

        def run():
            pop = make_fake_selection_population(**pop_kwargs)
            strategy = make_multi_frequency_selection(
                population_size=10, w=2, s=0, o=1, ln=2, seed=7
            )
            winners, _s, _o, losers = strategy._bracket_subpopulation(pop, subpop=0)
            new_pop, _ = strategy._clone_winners_over_losers(
                pop, winners, losers, subpop=0
            )
            return sorted(a.weights for a in new_agents(pop, new_pop))

        assert run() == run()  # identical winner picks for the same seed

    def test_clone_winner_selection_varies_with_seed(self):
        weights = {0: ["W_a", "W_b", "m", "x", "y"], 1: ["a", "b", "c", "d", "e"]}
        pop_kwargs = {
            "subpop_fitnesses": {0: [5.0, 4.0, 3.0, 2.0, 1.0], 1: [10, 9, 8, 7, 6]},
            "weights": weights,
        }

        def run(seed):
            picks = []
            # Several cloning rounds so the seeded stream can diverge
            for _ in range(6):
                pop = make_fake_selection_population(**pop_kwargs)
                strategy = make_multi_frequency_selection(
                    population_size=10, w=2, s=0, o=1, ln=2, seed=seed
                )
                winners, _s, _o, losers = strategy._bracket_subpopulation(pop, subpop=0)
                new_pop, _ = strategy._clone_winners_over_losers(
                    pop, winners, losers, subpop=0
                )
                picks.append(tuple(a.weights for a in new_agents(pop, new_pop)))
            return picks

        assert run(1) != run(2)


class TestSelect:
    def test_select_evolves_each_subpopulation_at_its_frequency(self):
        strategy = make_multi_frequency_selection(
            n_subpop=3, population_size=12, ratios=[1, 2, 3]
        )
        pop = make_fake_selection_population(
            {0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]}
        )
        fired = []  # (cycle, subpop)

        for cycle in range(1, 7):
            _elite, new_pop, _indices = strategy.select(pop)
            fired.extend(
                (cycle, subpop)
                for subpop in sorted(
                    {a.subpopulation_id for a in new_agents(pop, new_pop)}
                )
            )
            pop = new_pop

        assert [c for c, s in fired if s == 0] == [1, 2, 3, 4, 5, 6]  # delta=1
        assert [c for c, s in fired if s == 1] == [2, 4, 6]  # delta=2
        assert [c for c, s in fired if s == 2] == [3, 6]  # delta=3

    def test_select_returns_new_population_and_only_clone_indices(self):
        # Subpop 0 (delta 1) is due on cycle 1; subpop 1 (delta 2) is not.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )

        elite, new_pop, indices = strategy.select(pop)

        assert len(new_pop) == 8
        assert strategy.counters == [0, 1]
        # The global elite is the best-fitness agent
        assert elite.fitness[-1] == 8.0
        # Only the due subpopulation introduced fresh agents
        fresh = new_agents(pop, new_pop)
        assert fresh
        assert all(a.subpopulation_id == 0 for a in fresh)
        # Exactly the winner-clone is marked for mutation, never the migrant
        assert len(indices) == 1
        clone = next(a for a in fresh if a.index in indices)
        assert clone.fitness[-1] == 4.0
        # The not-due subpopulation is left entirely untouched
        original_subpop1 = [a for a in pop if a.subpopulation_id == 1]
        assert all(a in new_pop for a in original_subpop1)

    def test_select_rejects_duplicate_indices(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = [FakeSelectionAgent(3, i // 4, fitness=float(8 - i)) for i in range(8)]

        with pytest.raises(ValueError, match="globally-unique agent indices"):
            strategy.select(pop)

    def test_select_keeps_indices_unique_across_cycles(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]}
        )

        for _ in range(4):
            _elite, pop, _indices = strategy.select(pop)
            assert len({a.index for a in pop}) == len(pop)


class TestDeltaOf:
    def test_delta_of_returns_the_frequency_of_the_agents_subpopulation(self):
        strategy = make_multi_frequency_selection(
            n_subpop=3, population_size=12, ratios=[1, 4, 7]
        )
        agents = [FakeSelectionAgent(i, i, fitness=0.0) for i in range(3)]

        assert [strategy._delta_of(a) for a in agents] == [1, 4, 7]

    def test_delta_of_rejects_an_untagged_agent(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        agent = FakeSelectionAgent(0, None, fitness=0.0)

        with pytest.raises(ValueError, match="missing its subpopulation tag"):
            strategy._delta_of(agent)

    def test_migration_rejects_an_untagged_external_candidate(self):
        # An untagged agent in the external pool has no delta to compare against
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], None: [9.0, 8.0, 7.0, 6.0]}
        )

        with pytest.raises(ValueError, match="missing its subpopulation tag"):
            run_migration(strategy, pop, subpop=0, external_pool=pop)


class TestMigration:
    def test_migration_slow_to_fast_keeps_external_weights_but_elite_hps(self):
        # Studied subpop 1 (delta=2); external subpop 0 (delta=1) -> delta_ext < delta_studied.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]}
        )
        for a in pop:
            if a.subpopulation_id == 0:
                a.weights = "EXT"
                a.lr = 0.5
                a.optimizer = FakeOptimizerWrapper(0.5)
                a.registry.hp_config["lr"].value = 0.5
            else:
                a.lr = 0.001

        migrated = run_migration(strategy, pop, subpop=1, external_pool=pop)

        movers = new_agents(pop, migrated)
        assert len(movers) == 1
        mover = movers[0]
        assert mover.weights == "EXT"
        assert mover.lr == 0.001
        assert (
            mover.reinit_called is True
        )  # optimizer rebuilt at the elite's lr via reinit_optimizers
        assert mover.optimizer.lr == 0.001
        assert mover.optimizer.optimizer.param_groups[0]["lr"] == 0.001
        assert mover.registry.hp_config["lr"].value == 0.001

    def test_migration_full_clone_when_external_not_faster(self):
        # Studied subpop 0 (delta=1); external subpop 1 (delta=2) -> delta_ext > delta_studied.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [5.0, 4.0, 3.0, 2.0], 1: [9.0, 8.0, 7.0, 6.0]}
        )
        for a in pop:
            if a.subpopulation_id == 1:
                a.weights = "EXT"
                a.lr = 0.5

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        movers = new_agents(pop, migrated)
        assert len(movers) == 1
        assert movers[0].weights == "EXT"
        assert movers[0].lr == 0.5

    def test_migration_migrant_is_independent_of_external_parent(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [5.0, 4.0, 3.0, 2.0], 1: [9.0, 8.0, 7.0, 6.0]}
        )
        external = next(
            a for a in pop if a.subpopulation_id == 1 and a.fitness[-1] == 9.0
        )

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        mover = new_agents(pop, migrated)[0]
        assert mover is not external
        assert mover.index != external.index

    def test_migration_skips_when_open_agent_already_better(self):
        # The open agent of subpop 0 beats every external agent -> no migration
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [100.0, 99.0, 98.0, 1.0], 1: [9.0, 8.0, 7.0, 6.0]}
        )
        open_agent = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 98.0
        )

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        assert open_agent in migrated
        assert not new_agents(pop, migrated)

    def test_migration_sources_migrants_from_external_pool_not_live_population(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2], w=1, s=1, o=1, ln=1
        )
        pop = make_fake_selection_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [5.0, 4.0, 3.0, 2.0]}
        )
        for a in pop:  # subpop 0 already evolved this cycle -> its members are spent
            if a.subpopulation_id == 0:
                a.weights = "SPENT"
        frozen = make_fake_selection_population(
            {0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]}
        )
        for a in frozen:
            if a.subpopulation_id == 0:
                a.weights = "FROZEN"

        migrated = run_migration(strategy, pop, subpop=1, external_pool=frozen)

        movers = new_agents(pop, migrated)
        assert len(movers) == 1
        # The migrant carries the frozen agent's weights, proving it was sourced from
        # the snapshot
        assert movers[0].weights == "FROZEN"

    def test_migration_advances_external_pointer_only_on_success(self):
        # The pointer into the ranked external pool advances only after a
        # migration actually happens.
        #
        # Scenario (subpop 0 receives; subpop 1 is the external pool):
        #     open slots, best first:   open0 = 30, open1 = 20
        #     external pool, best first: e0 = 40, e1 = 15, e2 = 10, e3 = 8
        #     open0 (30) < e0 (40)  ->  e0 migrates in; pointer advances to e1
        #     open1 (20) > e1 (15)  ->  open1 keeps its slot; no migration
        #
        # So exactly one migrant results. Had the pointer not advanced past e0, open1
        # would be re-offered e0 (40) and migrate a second copy of it.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2], w=1, s=0, o=2, ln=1
        )
        pop = make_fake_selection_population(
            {0: [50.0, 30.0, 20.0, 5.0], 1: [40.0, 15.0, 10.0, 8.0]}
        )
        e0 = next(a for a in pop if a.subpopulation_id == 1 and a.fitness[-1] == 40.0)
        e0.weights = "E0"
        open1 = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 20.0
        )

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        migrants = new_agents(pop, migrated)
        assert len(migrants) == 1
        assert migrants[0].weights == "E0"
        assert open1 in migrated

    def test_migration_decisions_stop_when_external_pool_is_exhausted(self):
        # Two open slots but a single external candidate: the first slot consumes it,
        # then the second finds the pool exhausted and there is no migration
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        elite = FakeSelectionAgent(100, 0, fitness=10.0)
        open0 = FakeSelectionAgent(101, 0, fitness=1.0)
        open1 = FakeSelectionAgent(102, 0, fitness=1.0)
        external = FakeSelectionAgent(200, 1, fitness=5.0)

        decisions = strategy._migration_decisions(
            subpop=0,
            winners=[elite],
            open_for_migration=[open0, open1],
            external_pool=[external],
        )

        assert [open_agent for open_agent, *_ in decisions] == [open0]

    def test_migration_holds_external_pointer_on_skip(self):
        # A skip (the open agent is already at least as good) must not advance the pointer,
        # so the next open slot is offered the same candidate the skip passed over.
        #
        # Scenario (subpop 0 receives; subpop 1 is the external pool):
        #     open slots, best first:   open0 = 30, open1 = 20
        #     external pool, best first: e0 = 25, e1 = 12, e2 = 10, e3 = 8
        #     open0 (30) > e0 (25)  ->  open0 keeps its slot; pointer stays on e0
        #     open1 (20) < e0 (25)  ->  the same e0 migrates in
        #
        # So exactly one migrant results. Had the skip advanced the pointer, open1 would
        # face e1 (12), beat it, and skip -> zero migrants.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2], w=1, s=0, o=2, ln=1
        )
        pop = make_fake_selection_population(
            {0: [50.0, 30.0, 20.0, 5.0], 1: [25.0, 12.0, 10.0, 8.0]}
        )
        e0 = next(a for a in pop if a.subpopulation_id == 1 and a.fitness[-1] == 25.0)
        e0.weights = "E0"
        open0 = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 30.0
        )

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        migrants = new_agents(pop, migrated)
        assert len(migrants) == 1
        assert migrants[0].weights == "E0"
        assert open0 in migrated


class FakeLLMAgent:
    """Stand-in for an :class:`LLMAlgorithm`."""

    def __init__(
        self,
        index,
        subpopulation_id,
        fitness,
        weights="w",
        lr=1e-3,
        batch_size=64,
        accelerator=None,
        mut="stale-mut",
    ):
        self.index = index
        self.subpopulation_id = subpopulation_id
        self.fitness = [fitness]
        self.weights = weights
        self.lr = lr
        self.batch_size = batch_size
        self.registry = FakeRegistry(["lr", "batch_size"])
        self.registry.hp_config["lr"].value = lr
        self.registry.hp_config["batch_size"].value = batch_size
        self.optimizer = FakeOptimizerWrapper(lr)
        self.accelerator = accelerator
        self.mut = mut
        self.reinit_called = False
        self.mutation_hook_called = False
        self.clean_up_calls = 0

    def clone(self, index=None, wrap=False):
        if self.clean_up_calls:
            msg = f"agent {self.index} was cloned after being freed"
            raise AssertionError(msg)
        new = FakeLLMAgent(
            self.index if index is None else index,
            self.subpopulation_id,
            self.fitness[-1],
            weights=self.weights,
            lr=self.lr,
            batch_size=self.batch_size,
            accelerator=self.accelerator,
            mut=self.mut,  # the real clone copies attributes, incl. the parent's mut
        )
        for name in self.registry.hp_config.names():
            new.registry.hp_config[name].value = self.registry.hp_config[name].value
        return new

    def mutation_hook(self):
        self.mutation_hook_called = True

    def reinit_optimizers(self, optimizer=None):
        self.reinit_called = True
        self.optimizer = FakeOptimizerWrapper(self.lr)

    def clean_up(self):
        self.clean_up_calls += 1


def make_llm_population(subpop_fitnesses, accelerator=None):
    """Build a population of :class:`FakeLLMAgent` with unique indices."""
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for fit in fitnesses:
            population.append(
                FakeLLMAgent(
                    idx, subpop, fit, weights=f"w{idx}", accelerator=accelerator
                )
            )
            idx += 1
    return population


@pytest.fixture
def llm_dispatch(monkeypatch):
    """Route ``select`` through the LLM path by making FakeLLMAgent the LLM type."""
    monkeypatch.setattr(mf_module, "LLMAlgorithm", FakeLLMAgent)


class TestEliteHpValues:
    def test_elite_hp_values_snapshots_every_mutable_hp_by_value(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        elite = FakeSelectionAgent(0, 0, fitness=1.0, lr=0.5)
        elite.batch_size = [32, 64]  # a mutable value, to prove the snapshot is a copy

        values = strategy._elite_hp_values(elite)
        values["batch_size"].append(128)

        assert values == {"lr": 0.5, "batch_size": [32, 64, 128]}
        assert elite.batch_size == [32, 64]

    def test_elite_hp_values_is_empty_without_mutable_hps(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        elite = FakeSelectionAgent(0, 0, fitness=1.0)
        elite.registry = FakeRegistry([])  # an algorithm registering no mutable HPs

        assert strategy._elite_hp_values(elite) == {}

    def test_weights_only_migration_leaves_hps_alone_without_mutable_hps(self):
        # With nothing to reset, the weights-only migrant keeps the external agent's
        # hyperparameters and its optimizer is never rebuilt
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_fake_selection_population(
            {0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]}
        )
        for a in pop:
            a.registry = FakeRegistry([])
            if a.subpopulation_id == 0:  # the faster subpopulation migrants come from
                a.weights = "EXT"
                a.lr = 0.5

        migrated = run_migration(strategy, pop, subpop=1, external_pool=pop)

        movers = new_agents(pop, migrated)
        assert len(movers) == 1
        assert movers[0].weights == "EXT"
        assert movers[0].lr == 0.5
        assert movers[0].reinit_called is False


class TestApplyHpReset:
    def test_apply_hp_reset_sets_hps_syncs_config_and_rebuilds_lr_optimizer(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        agent = FakeLLMAgent(0, 0, 1.0, lr=0.5)

        strategy._apply_hp_reset(agent, {"lr": 0.001, "batch_size": 32})

        assert agent.lr == 0.001
        assert agent.batch_size == 32
        assert agent.registry.hp_config["lr"].value == 0.001
        assert agent.registry.hp_config["batch_size"].value == 32
        assert agent.mutation_hook_called is True
        assert agent.reinit_called is True
        assert agent.optimizer.lr == 0.001

    def test_apply_hp_reset_skips_optimizer_rebuild_when_lr_unchanged(self):
        strategy = make_multi_frequency_selection(n_subpop=2, population_size=8)
        agent = FakeLLMAgent(0, 0, 1.0, lr=0.5)

        strategy._apply_hp_reset(agent, {"batch_size": 32})

        assert agent.batch_size == 32
        assert agent.reinit_called is False


@pytest.mark.usefixtures("llm_dispatch")
class TestSelectLLM:
    def test_select_dispatches_to_llm_path_for_llm_populations(self):
        # The LLM path returns the live elite (not a fresh clone) and scrubs mut on
        # every returned agent. Neither is true of the standard path
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

        elite, new_pop, _indices = strategy.select(pop)

        assert elite in pop
        assert all(a.mut == "None" for a in new_pop)

    def test_select_returns_live_global_elite(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        best = next(a for a in pop if a.fitness[-1] == 8.0)

        elite, new_pop, _indices = strategy.select(pop)

        assert elite is best
        assert elite in new_pop

    def test_select_marks_only_winner_clones_for_mutation(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

        _elite, new_pop, indices = strategy.select(pop)

        assert len(indices) == 1  # one loser slot in the due subpop 0
        clone = next(a for a in new_pop if a.index in indices)
        assert clone.fitness[-1] == 4.0  # cloned from subpop 0's winner

    def test_select_frees_replaced_non_source_agents(self):
        # Subpop 0 is due: its loser (1.0) is cloned over and its open slot
        # (2.0) is migrated over. Both are freed; neither is a clone/migration source
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        loser = next(a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 1.0)
        open_agent = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 2.0
        )

        strategy.select(pop)

        assert loser.clean_up_calls == 1
        assert open_agent.clean_up_calls == 1

    def test_select_frees_a_replaced_agent_only_after_its_last_use_as_a_source(self):
        # Subpop 0's weaker open slot (30.0) is migrated over, yet it is also the agent subpop
        # 1 imports. It therefore cannot be freed until the migration that reads it has run.
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2], w=1, s=0, o=2, ln=1
        )
        strategy.counters = [0, 1]  # a single cycle then fires both subpopulations
        pop = make_llm_population(
            {0: [50.0, 30.0, 20.0, 5.0], 1: [40.0, 15.0, 10.0, 8.0]}
        )
        source = next(a for a in pop if a.fitness[-1] == 30.0)
        source.weights = "SRC30"

        _elite, new_pop, _indices = strategy.select(pop)

        assert any(a.subpopulation_id == 1 and a.weights == "SRC30" for a in new_pop)
        assert source not in new_pop  # its own slot was taken by a migrant
        assert source.clean_up_calls == 1  # and it was freed exactly once, afterwards

    def test_select_does_not_free_surviving_agents(self):
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        winner = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 4.0
        )
        survivor = next(
            a for a in pop if a.subpopulation_id == 0 and a.fitness[-1] == 3.0
        )

        strategy.select(pop)

        assert winner.clean_up_calls == 0
        assert survivor.clean_up_calls == 0
        # The whole not-due subpopulation is untouched
        for a in pop:
            if a.subpopulation_id == 1:
                assert a.clean_up_calls == 0

    def test_select_schedules_subpopulations_at_their_frequency(self):
        strategy = make_multi_frequency_selection(
            n_subpop=3, population_size=12, ratios=[1, 2, 3]
        )
        counters_seen = []

        for _ in range(6):
            pop = make_llm_population(
                {0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]}
            )
            strategy.select(pop)
            counters_seen.append(list(strategy.counters))

        # Subpop 0 (delta 1) resets every cycle; subpop 1 (delta 2) every 2; subpop 2
        # (delta 3) every 3.
        assert counters_seen == [
            [0, 1, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 2],
            [0, 0, 0],
        ]

    def test_select_migrate_full_imports_external_agent_wholesale(self):
        # Subpop 0 due; its open slot is offered subpop 1's best (full clone)
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        ext = next(a for a in pop if a.fitness[-1] == 8.0)
        ext.weights = "EXT8"
        ext.lr = 0.5

        _elite, new_pop, _indices = strategy.select(pop)

        migrant = next(
            a for a in new_pop if a.subpopulation_id == 0 and a.weights == "EXT8"
        )
        assert migrant.lr == 0.5  # full clone keeps the external agent's lr
        assert migrant is not ext

    def test_select_migrate_weights_keeps_external_weights_but_elite_hps(self):
        # Both subpops due. Subpop 1 draws from subpop 0 (weights-only)
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        strategy.counters = [0, 1]
        pop = make_llm_population({0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]})
        for a in pop:
            if a.subpopulation_id == 0:
                a.weights = "FAST"
                a.lr = 0.5
                a.registry.hp_config["lr"].value = 0.5
            else:
                a.lr = 0.001
                a.registry.hp_config["lr"].value = 0.001

        _elite, new_pop, _indices = strategy.select(pop)

        migrant = next(
            a for a in new_pop if a.subpopulation_id == 1 and a.weights == "FAST"
        )
        assert migrant.lr == 0.001
        assert migrant.reinit_called is True
        assert migrant.registry.hp_config["lr"].value == 0.001


class _MultiProcAccelerator:
    """Minimal multi-process accelerator stand-in for the LLM selection path."""

    def __init__(self, is_main_process, num_processes):
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.wait_calls = 0

    def wait_for_everyone(self):
        self.wait_calls += 1


class TestSelectLLMAccelerator:
    def test_main_process_broadcasts_the_plan_to_workers(self, monkeypatch):
        monkeypatch.setattr(mf_module, "LLMAlgorithm", FakeLLMAgent)
        broadcasts = []

        def fake_broadcast(obj, from_process=0):
            broadcasts.append((obj, from_process))
            return obj

        monkeypatch.setattr(mf_module, "broadcast_object_list", fake_broadcast)
        accelerator = _MultiProcAccelerator(is_main_process=True, num_processes=2)
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        pop = make_llm_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]},
            accelerator=accelerator,
        )

        elite, _new_pop, indices = strategy.select(pop)

        assert len(broadcasts) == 1
        payload, from_process = broadcasts[0]
        assert from_process == 0  # decisions originate on the main process
        (plan,) = payload
        assert set(plan) == {"ops", "elite_index", "indices_to_mutate"}
        assert len(plan["ops"]) == len(pop)
        assert plan["elite_index"] == 4  # global best sits at index 4
        assert elite.index == plan["elite_index"]
        assert plan["indices_to_mutate"] == indices

    def test_worker_process_builds_population_from_broadcast_plan(self, monkeypatch):
        # A worker must not advance its own counters/RNG; it consumes the plan the main
        # process broadcast and materialises exactly that generation
        monkeypatch.setattr(mf_module, "LLMAlgorithm", FakeLLMAgent)
        strategy = make_multi_frequency_selection(
            n_subpop=2, population_size=8, ratios=[1, 2]
        )
        accelerator = _MultiProcAccelerator(is_main_process=False, num_processes=2)
        pop = make_llm_population(
            {0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]},
            accelerator=accelerator,
        )

        keep = (MultiFrequencyOp.KEEP,)
        plan = {
            "ops": [
                keep,
                keep,
                keep,
                (MultiFrequencyOp.CLONE, 0, 8, 0),
                keep,
                keep,
                keep,
                keep,
            ],
            "elite_index": 4,
            "indices_to_mutate": [8],
        }
        monkeypatch.setattr(
            mf_module, "broadcast_object_list", lambda obj, from_process=0: [plan]
        )

        elite, new_pop, indices = strategy.select(pop)

        assert strategy.counters == [0, 0]  # worker never advanced its counters
        assert indices == [8]
        assert elite.index == 4
        clone = next(a for a in new_pop if a.index == 8)
        assert clone.fitness[-1] == 4.0  # cloned from the winner
