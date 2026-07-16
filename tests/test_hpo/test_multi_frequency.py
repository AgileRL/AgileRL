"""Unit tests for the multi-frequency selection evolution loop."""

from __future__ import annotations

import numpy as np
import pytest

from agilerl.hpo.multi_frequency import MultiFrequencySelection


class _FakeParam:
    """RLParameter-like stand-in carrying a mutable value."""

    def __init__(self, value=None):
        self.value = value


class FakeHPConfig:
    """Mimics HyperparameterConfig."""

    def __init__(self, names):
        self._params = {name: _FakeParam() for name in names}

    def __iter__(self):
        return iter(self._params)

    def __bool__(self):
        return bool(self._params)

    def names(self):
        return list(self._params)

    def __getitem__(self, key):
        return self._params[key]


class FakeTorchOptimizer:
    """Mimics a torch.optim.Optimizer."""

    def __init__(self, lr):
        self.param_groups = [{"lr": lr}]


class FakeOptimizerWrapper:
    """Mimics a OptimizerWrapper."""

    def __init__(self, lr):
        self.lr = lr
        self.optimizer = FakeTorchOptimizer(lr)


class FakeOptConfig:
    """Mimics OptimizerConfig."""

    def __init__(self, lr="lr", name="optimizer"):
        self.lr = lr
        self.name = name


class FakeRegistry:
    def __init__(self, hp_names):
        self.hp_config = FakeHPConfig(hp_names)
        self.optimizers = [FakeOptConfig(lr="lr", name="optimizer")]


class FakeAgent:
    """Stand-in for an EvolvableAlgorithm."""

    def __init__(
        self, index, subpopulation, fitness, weights="w", lr=1e-3, batch_size=64
    ):
        self.index = index
        self.subpopulation = subpopulation
        self.fitness = [fitness]
        self.weights = weights
        self.lr = lr
        self.batch_size = batch_size
        self.registry = FakeRegistry(["lr", "batch_size"])
        self.optimizer = FakeOptimizerWrapper(lr)
        self.reinit_called = False

    def clone(self, index=None, wrap=False):
        new = FakeAgent(
            self.index if index is None else index,
            self.subpopulation,
            self.fitness[-1],
            weights=self.weights,
            lr=self.lr,
            batch_size=self.batch_size,
        )
        for name in self.registry.hp_config.names():
            new.registry.hp_config[name].value = self.registry.hp_config[name].value
        return new

    def mutation_hook(self):
        self.mutation_hook_called = True

    def reinit_optimizers(self, optimizer=None):
        # Mirror the real reinit_optimizers
        self.reinit_called = True
        self.optimizer = FakeOptimizerWrapper(self.lr)

    def save_checkpoint(self, path):
        with open(path, "w") as fh:
            fh.write("checkpoint")


def make_population(subpop_fitnesses, weights=None):
    """Build a population with unique indices."""
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for j, fit in enumerate(fitnesses):
            w = weights[subpop][j] if weights is not None else f"w{idx}"
            population.append(FakeAgent(idx, subpop, fit, weights=w))
            idx += 1
    return population


def make_strategy(n_subpop=2, n_ind=4, ratios=None, w=1, s=1, o=1, ln=1, seed=42):
    return MultiFrequencySelection(
        n_subpopulations=n_subpop,
        n_individuals_per_subpopulation=n_ind,
        evolution_frequency_ratios=ratios or list(range(1, n_subpop + 1)),
        n_winners=w,
        n_survivors=s,
        n_open_for_migration=o,
        n_losers=ln,
        seed=seed,
    )


def new_agents(before, after):
    """Agents in after that are not one of the objects in before."""
    before_ids = {id(a) for a in before}
    return [a for a in after if id(a) not in before_ids]


def run_migration(strategy, population, subpop, external_pool):
    """Bracket then migrate."""
    winners, _survivors, open_for_migration, _losers = strategy._brackets(
        population, subpop
    )
    return strategy._migrate(
        population, subpop, winners, open_for_migration, external_pool
    )


class TestMultiFrequencySelectionInit:
    def test_valid_construction_sets_derived_attributes(self):
        strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 3])
        assert strategy.pop_size == 8
        assert strategy.deltas == [1, 3]
        assert strategy.bracket_sizes == (1, 1, 1, 1)
        assert strategy.counters == [0, 0]

    def test_defaults_resolve_to_recommended_configuration(self):
        strategy = MultiFrequencySelection()
        assert strategy.n_subpopulations == 2
        assert strategy.n_individuals_per_subpopulation == 8
        assert strategy.pop_size == 16
        assert strategy.bracket_sizes == (2, 0, 2, 4)
        assert strategy.deltas == [1, 5]

    def test_ratios_default_scales_with_subpopulations(self):
        strategy = MultiFrequencySelection(
            n_subpopulations=4, n_individuals_per_subpopulation=8
        )
        assert strategy.deltas == [1, 5, 10, 15]

    def test_empty_ratios_list_resolves_to_default(self):
        strategy = MultiFrequencySelection(
            n_subpopulations=2,
            n_individuals_per_subpopulation=8,
            evolution_frequency_ratios=[],
        )
        assert strategy.deltas == [1, 5]

    def test_losers_default_fills_the_remainder(self):
        strategy = MultiFrequencySelection(
            n_subpopulations=2,
            n_individuals_per_subpopulation=8,
            n_winners=3,
            n_survivors=1,
            n_open_for_migration=1,
        )
        assert strategy.n_losers == 3  # 8 - 3 - 1 - 1

    @pytest.mark.parametrize("n_ind", [1, 2])
    def test_init_rejects_fewer_than_three_individuals(self, n_ind):
        with pytest.raises(
            ValueError, match="n_individuals_per_subpopulation must be >= 3"
        ):
            MultiFrequencySelection(
                n_subpopulations=2, n_individuals_per_subpopulation=n_ind
            )

    def test_init_rejects_zero_open_for_migration(self):
        with pytest.raises(ValueError, match="n_open_for_migration must be >= 1"):
            MultiFrequencySelection(
                n_subpopulations=2,
                n_individuals_per_subpopulation=4,
                n_winners=1,
                n_survivors=1,
                n_open_for_migration=0,
                n_losers=2,
            )

    def test_init_rejects_negative_survivors(self):
        with pytest.raises(ValueError, match="n_survivors must be >= 0"):
            MultiFrequencySelection(
                n_subpopulations=2,
                n_individuals_per_subpopulation=4,
                n_winners=1,
                n_survivors=-1,
                n_open_for_migration=1,
                n_losers=3,
            )

    @pytest.mark.parametrize(
        ("n_survivors", "derived_losers"),
        [(1, 0), (2, -1)],  # n_ind - n_winners - n_survivors - n_open = 4 - 2 - s - 1
    )
    def test_init_rejects_derived_losers_non_positive(
        self, n_survivors, derived_losers
    ):
        with pytest.raises(
            ValueError, match=f"n_losers must be >= 1, got {derived_losers}"
        ):
            MultiFrequencySelection(
                n_subpopulations=2,
                n_individuals_per_subpopulation=4,
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
                "must equal n_individuals",
            ),  # sum 5 != 4
        ],
    )
    def test_init_rejects_invalid_brackets(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2], **kwargs)

    def test_open_for_migration_may_exceed_winners_plus_survivors(self):
        # Migration sources migrants from the frozen pre-evolution snapshot rather than
        # the live population, so a subpopulation may open more slots for migration than
        # it preserves natively
        strategy = make_strategy(
            n_subpop=2, n_ind=4, ratios=[1, 2], w=1, s=0, o=2, ln=1
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
            make_strategy(n_subpop=2, n_ind=4, ratios=ratios)

    def test_init_rejects_negative_bracket(self):
        with pytest.raises(ValueError, match="n_losers must be >= 1, got -1"):
            MultiFrequencySelection(
                n_subpopulations=2,
                n_individuals_per_subpopulation=4,
                evolution_frequency_ratios=[1, 2],
                n_winners=2,
                n_survivors=2,
                n_open_for_migration=1,
                n_losers=-1,
            )

    def test_frozen_subpopulation_rejected(self):
        # A "frozen" subpopulation should raise an error
        with pytest.raises(ValueError, match="n_winners must be >= 1"):
            make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2], w=0, s=4, o=0, ln=0)

    def test_init_rejects_fewer_than_two_subpopulations(self):
        with pytest.raises(ValueError, match="n_subpopulations must be >= 2"):
            MultiFrequencySelection(
                n_subpopulations=1,
                n_individuals_per_subpopulation=4,
                evolution_frequency_ratios=[1],
                n_winners=1,
                n_survivors=1,
                n_open_for_migration=1,
                n_losers=1,
            )


class TestScalarFitness:
    def test_scalar_fitness_reduces_dict_list_array_and_scalar(self):
        assert MultiFrequencySelection._scalar_fitness({"a": 1.0, "b": 3.0}) == 2.0
        assert MultiFrequencySelection._scalar_fitness([1.0, 3.0]) == 2.0
        assert MultiFrequencySelection._scalar_fitness(np.array([2.0, 4.0])) == 3.0
        assert MultiFrequencySelection._scalar_fitness(5.0) == 5.0


class TestSubpopulationAssignment:
    def test_subpopulation_for_index_maps_contiguous_blocks(self):
        fn = MultiFrequencySelection._subpopulation_for_index
        assert [fn(i, 4) for i in range(12)] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_assign_initial_subpopulations_tags_only_untagged_agents(self):
        strategy = make_strategy(n_subpop=2, n_ind=4)
        pop = [FakeAgent(i, None, fitness=0.0) for i in range(8)]
        pop[0] = FakeAgent(0, 1, fitness=0.0)
        pop[5] = FakeAgent(5, 0, fitness=0.0)

        strategy._assign_initial_subpopulations(pop)

        assert [a.subpopulation for a in pop] == [1, 0, 0, 0, 1, 0, 1, 1]

    @pytest.mark.parametrize("size", [6, 10])
    def test_assign_initial_subpopulations_rejects_wrong_population_size(self, size):
        strategy = make_strategy(n_subpop=2, n_ind=4)  # pop_size == 8
        pop = [FakeAgent(i, None, fitness=0.0) for i in range(size)]

        with pytest.raises(ValueError, match=f"{size} agents, expected 8"):
            strategy._assign_initial_subpopulations(pop)

    def test_assign_initial_subpopulations_accepts_correct_population_size(self):
        strategy = make_strategy(n_subpop=2, n_ind=4)  # pop_size == 8
        pop = [FakeAgent(i, None, fitness=0.0) for i in range(8)]

        strategy._assign_initial_subpopulations(pop)

        assert [a.subpopulation for a in pop] == [0, 0, 0, 0, 1, 1, 1, 1]


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
        strategy = make_strategy(n_subpop=2, n_ind=4, w=w, s=s, o=o, ln=ln)
        pop = make_population({0: [2.0, 4.0, 1.0, 3.0], 1: [8.0, 7.0, 6.0, 5.0]})

        winners, survivors, open_, losers = strategy._brackets(pop, subpop=0)

        assert [a.fitness[-1] for a in winners] == expected[0]
        assert [a.fitness[-1] for a in survivors] == expected[1]
        assert [a.fitness[-1] for a in open_] == expected[2]
        assert [a.fitness[-1] for a in losers] == expected[3]
        # Only the studied subpopulation's agents are ever bracketed
        for bracket in (winners, survivors, open_, losers):
            assert all(a.subpopulation == 0 for a in bracket)

    def test_brackets_selects_the_requested_subpopulation(self):
        strategy = make_strategy(n_subpop=2, n_ind=4)
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

        winners, survivors, open_, losers = strategy._brackets(pop, subpop=1)

        assert [a.fitness[-1] for a in winners] == [8.0]
        assert [a.fitness[-1] for a in survivors] == [7.0]
        assert [a.fitness[-1] for a in open_] == [6.0]
        assert [a.fitness[-1] for a in losers] == [5.0]
        for bracket in (winners, survivors, open_, losers):
            assert all(a.subpopulation == 1 for a in bracket)

    def test_brackets_rejects_wrong_member_count(self):
        # A mis-tagged subpop is rejected up front rather than silently mis-sliced
        strategy = make_strategy(n_subpop=2, n_ind=4)
        pop = [FakeAgent(i, 0, fitness=float(i)) for i in range(3)]

        with pytest.raises(
            ValueError, match="Subpopulation 0 has 3 members, expected 4"
        ):
            strategy._brackets(pop, subpop=0)


class TestCloneWinnersOverLosers:
    def test_clone_winners_over_losers_replaces_losers_with_fresh_winner_clones(self):
        strategy = make_strategy(n_subpop=2, n_ind=4)
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        max_index_before = max(a.index for a in pop)
        winners, _s, _o, losers = strategy._brackets(pop, subpop=0)

        new_pop, new_indices = strategy._clone_winners_over_losers(
            pop, winners, losers, subpop=0
        )

        assert len(new_pop) == len(pop)
        assert sum(a.subpopulation == 0 for a in new_pop) == 4
        # The loser (fitness 1.0) was replaced
        assert 1.0 not in [a.fitness[-1] for a in new_pop if a.subpopulation == 0]
        clones = new_agents(pop, new_pop)
        assert len(clones) == 1
        clone = clones[0]
        assert clone.index in new_indices
        assert clone.index > max_index_before
        assert clone.subpopulation == 0
        # The clone inherits its winner-parent's fitness (4.0)
        assert clone.fitness[-1] == 4.0

    def test_clone_winners_over_losers_returns_new_list(self):
        # Cloning must return a new population rather than mutating the one it's handed
        strategy = make_strategy()
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        pop_copy = list(pop)
        winners, _s, _o, losers = strategy._brackets(pop, subpop=0)

        strategy._clone_winners_over_losers(pop, winners, losers, subpop=0)

        assert pop == pop_copy

    def test_clone_winners_over_losers_clones_are_independent_of_parents(self):
        strategy = make_strategy(n_subpop=2, n_ind=4, w=1, s=0, o=1, ln=2)
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        winner = next(a for a in pop if a.subpopulation == 0 and a.fitness[-1] == 4.0)
        winners, _s, _o, losers = strategy._brackets(pop, subpop=0)

        new_pop, _ = strategy._clone_winners_over_losers(pop, winners, losers, subpop=0)

        for clone in new_agents(pop, new_pop):
            assert clone is not winner
            assert clone.index != winner.index
            # Mutating the clone's HP must not touch the parent's
            clone.registry.hp_config["lr"].value = 999.0
            assert winner.registry.hp_config["lr"].value != 999.0

    def test_clone_winners_over_losers_with_zero_survivors_replaces_every_loser(self):
        # Brackets 1/0/1/2: one winner, no survivors, one open, two losers.
        strategy = make_strategy(n_subpop=2, n_ind=4, w=1, s=0, o=1, ln=2)
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
        winners, _s, _o, losers = strategy._brackets(pop, subpop=0)

        new_pop, new_indices = strategy._clone_winners_over_losers(
            pop, winners, losers, subpop=0
        )

        assert sum(a.subpopulation == 0 for a in new_pop) == 4
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
            pop = make_population(**pop_kwargs)
            strategy = make_strategy(n_ind=5, w=2, s=0, o=1, ln=2, seed=7)
            winners, _s, _o, losers = strategy._brackets(pop, subpop=0)
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
                pop = make_population(**pop_kwargs)
                strategy = make_strategy(n_ind=5, w=2, s=0, o=1, ln=2, seed=seed)
                winners, _s, _o, losers = strategy._brackets(pop, subpop=0)
                new_pop, _ = strategy._clone_winners_over_losers(
                    pop, winners, losers, subpop=0
                )
                picks.append(tuple(a.weights for a in new_agents(pop, new_pop)))
            return picks

        assert run(1) != run(2)


class TestSelect:
    def test_select_evolves_each_subpopulation_at_its_frequency(self):
        strategy = make_strategy(n_subpop=3, n_ind=4, ratios=[1, 2, 3])
        fired = []  # (cycle, subpop)
        cycle = {"n": 0}

        def fake_clone(population, winners, losers, subpop):
            fired.append((cycle["n"], subpop))
            return population, []

        strategy._clone_winners_over_losers = fake_clone
        strategy._migrate = (
            lambda population, subpop, winners, open_for_migration, external_pool: (
                population
            )
        )

        for c in range(1, 7):
            cycle["n"] = c
            pop = make_population(
                {0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]}
            )
            strategy.select(pop)

        assert [c for c, s in fired if s == 0] == [1, 2, 3, 4, 5, 6]  # delta=1
        assert [c for c, s in fired if s == 1] == [2, 4, 6]  # delta=2
        assert [c for c, s in fired if s == 2] == [3, 6]  # delta=3

    def test_select_returns_new_population_and_only_clone_indices(self):
        # Subpop 0 (delta 1) is due on cycle 1; subpop 1 (delta 2) is not.
        strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

        new_pop, indices = strategy.select(pop)

        assert len(new_pop) == 8
        assert strategy.counters == [0, 1]
        # Only the due subpopulation introduced fresh agents
        fresh = new_agents(pop, new_pop)
        assert fresh
        assert all(a.subpopulation == 0 for a in fresh)
        # Exactly the winner-clone is marked for mutation, never the migrant
        assert len(indices) == 1
        clone = next(a for a in fresh if a.index in indices)
        assert clone.fitness[-1] == 4.0
        # The not-due subpopulation is left entirely untouched
        original_subpop1 = [a for a in pop if a.subpopulation == 1]
        assert all(a in new_pop for a in original_subpop1)


class TestMigration:
    def test_migration_slow_to_fast_keeps_external_weights_but_elite_hps(self):
        # Studied subpop 1 (delta=2); external subpop 0 (delta=1) -> delta_ext < delta_studied.
        strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
        pop = make_population({0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]})
        for a in pop:
            if a.subpopulation == 0:
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
        strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
        pop = make_population({0: [5.0, 4.0, 3.0, 2.0], 1: [9.0, 8.0, 7.0, 6.0]})
        for a in pop:
            if a.subpopulation == 1:
                a.weights = "EXT"
                a.lr = 0.5

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        movers = new_agents(pop, migrated)
        assert len(movers) == 1
        assert movers[0].weights == "EXT"
        assert movers[0].lr == 0.5

    def test_migration_migrant_is_independent_of_external_parent(self):
        strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
        pop = make_population({0: [5.0, 4.0, 3.0, 2.0], 1: [9.0, 8.0, 7.0, 6.0]})
        external = next(a for a in pop if a.subpopulation == 1 and a.fitness[-1] == 9.0)

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        mover = new_agents(pop, migrated)[0]
        assert mover is not external
        assert mover.index != external.index

    def test_migration_skips_when_open_agent_already_better(self):
        # The open agent of subpop 0 beats every external agent -> no migration
        strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
        pop = make_population({0: [100.0, 99.0, 98.0, 1.0], 1: [9.0, 8.0, 7.0, 6.0]})
        open_agent = next(
            a for a in pop if a.subpopulation == 0 and a.fitness[-1] == 98.0
        )

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        assert open_agent in migrated
        assert not new_agents(pop, migrated)

    def test_migration_sources_migrants_from_external_pool_not_live_population(self):
        strategy = make_strategy(
            n_subpop=2, n_ind=4, ratios=[1, 2], w=1, s=1, o=1, ln=1
        )
        pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [5.0, 4.0, 3.0, 2.0]})
        for a in pop:  # subpop 0 already evolved this cycle -> its members are spent
            if a.subpopulation == 0:
                a.weights = "SPENT"
        frozen = make_population({0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]})
        for a in frozen:
            if a.subpopulation == 0:
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
        strategy = make_strategy(
            n_subpop=2, n_ind=4, ratios=[1, 2], w=1, s=0, o=2, ln=1
        )
        pop = make_population({0: [50.0, 30.0, 20.0, 5.0], 1: [40.0, 15.0, 10.0, 8.0]})
        e0 = next(a for a in pop if a.subpopulation == 1 and a.fitness[-1] == 40.0)
        e0.weights = "E0"
        open1 = next(a for a in pop if a.subpopulation == 0 and a.fitness[-1] == 20.0)

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        migrants = new_agents(pop, migrated)
        assert len(migrants) == 1
        assert migrants[0].weights == "E0"
        assert open1 in migrated

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
        strategy = make_strategy(
            n_subpop=2, n_ind=4, ratios=[1, 2], w=1, s=0, o=2, ln=1
        )
        pop = make_population({0: [50.0, 30.0, 20.0, 5.0], 1: [25.0, 12.0, 10.0, 8.0]})
        e0 = next(a for a in pop if a.subpopulation == 1 and a.fitness[-1] == 25.0)
        e0.weights = "E0"
        open0 = next(a for a in pop if a.subpopulation == 0 and a.fitness[-1] == 30.0)

        migrated = run_migration(strategy, pop, subpop=0, external_pool=pop)

        migrants = new_agents(pop, migrated)
        assert len(migrants) == 1
        assert migrants[0].weights == "E0"
        assert open0 in migrated
