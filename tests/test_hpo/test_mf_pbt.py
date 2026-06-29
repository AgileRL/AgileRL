"""Unit tests for the MF-PBT (Multiple-Frequencies Population-Based Training) regime.

These tests exercise the orchestration logic of :class:`agilerl.hpo.mf_pbt.MFPBT`
(bracket partitioning, evolution, asymmetric migration and the per-subpopulation
evolution-frequency scheduling) against lightweight fake agents that faithfully
implement the small interface MF-PBT relies on (``clone``/``fitness``/``index``/
``subpopulation``/``registry.hp_config``/``get_lr_names``/``reinit_optimizers``).

The real agent/mutation behaviour is covered by ``test_tournament.py`` /
``test_mutation.py`` and by the end-to-end trainer smoke test.
"""

import math

import numpy as np
import pytest

from agilerl.hpo.mf_pbt import MFPBT


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class FakeHPConfig:
    """Mimics ``HyperparameterConfig`` (iterates names, exposes ``names()``)."""

    def __init__(self, names):
        self._names = list(names)

    def __iter__(self):
        return iter(self._names)

    def names(self):
        return list(self._names)


class FakeRegistry:
    def __init__(self, hp_names):
        self.hp_config = FakeHPConfig(hp_names)


class FakeAgent:
    """Faithful stand-in for an ``EvolvableAlgorithm`` for MF-PBT's purposes."""

    def __init__(
        self, index, subpopulation, fitness, weights="w", lr=1e-3, batch_size=64
    ):
        self.index = index
        self.subpopulation = subpopulation
        self.fitness = [fitness]
        self.weights = weights  # marker standing in for network parameters
        self.lr = lr
        self.batch_size = batch_size
        self.registry = FakeRegistry(["lr", "batch_size"])
        self.reinit_called = False
        self.perturbed = False

    def clone(self, index=None, wrap=False):
        new = FakeAgent(
            self.index if index is None else index,
            self.subpopulation,
            self.fitness[-1],
            weights=self.weights,
            lr=self.lr,
            batch_size=self.batch_size,
        )
        return new

    def get_lr_names(self):
        return ["lr"]

    def mutation_hook(self):
        self.mutation_hook_called = True

    def reinit_optimizers(self, optimizer=None):
        self.reinit_called = True

    def save_checkpoint(self, path):
        with open(path, "w") as fh:
            fh.write("checkpoint")


class FakeMutations:
    """Records the ``mutate_elite`` flag seen and marks every perturbed agent."""

    def __init__(self, mutate_elite=False):
        self.mutate_elite = mutate_elite
        self.seen_mutate_elite = None
        self.perturbed_agents = []

    def mutation(self, population, pre_training_mut=False):
        self.seen_mutate_elite = self.mutate_elite
        for agent in population:
            agent.perturbed = True
            self.perturbed_agents.append(agent)
        return population


def make_population(subpop_fitnesses, weights=None):
    """Build a population from ``{subpop: [fitness, ...]}`` with unique indices."""
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for j, fit in enumerate(fitnesses):
            w = weights[subpop][j] if weights is not None else f"w{idx}"
            population.append(FakeAgent(idx, subpop, fit, weights=w))
            idx += 1
    return population


def make_mfpbt(n_subpop=2, n_ind=4, ratios=None, w=1, s=1, o=1, ln=1, seed=42):
    return MFPBT(
        n_subpopulations=n_subpop,
        n_individuals_per_subpopulation=n_ind,
        evolution_frequency_ratios=ratios or list(range(1, n_subpop + 1)),
        n_winners=w,
        n_survivors=s,
        n_open_for_migration=o,
        n_losers=ln,
        rand_seed=seed,
    )


# --------------------------------------------------------------------------- #
# _scalar_fitness
# --------------------------------------------------------------------------- #
def test_scalar_fitness_reduces_dict_list_array_and_scalar():
    assert MFPBT._scalar_fitness({"a": 1.0, "b": 3.0}) == 2.0
    assert MFPBT._scalar_fitness([1.0, 3.0]) == 2.0
    assert MFPBT._scalar_fitness(np.array([2.0, 4.0])) == 3.0
    assert MFPBT._scalar_fitness(5.0) == 5.0


# --------------------------------------------------------------------------- #
# brackets
# --------------------------------------------------------------------------- #
def test_brackets_partition_subpop_by_descending_fitness():
    mf = make_mfpbt()
    pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

    winners, survivors, open_, losers = mf.brackets(pop, subpop=0)

    # All four brackets contain only subpop-0 agents, in descending fitness.
    assert [a.fitness[-1] for a in winners] == [4.0]
    assert [a.fitness[-1] for a in survivors] == [3.0]
    assert [a.fitness[-1] for a in open_] == [2.0]
    assert [a.fitness[-1] for a in losers] == [1.0]
    for bracket in (winners, survivors, open_, losers):
        assert all(a.subpopulation == 0 for a in bracket)


# --------------------------------------------------------------------------- #
# evolution
# --------------------------------------------------------------------------- #
def test_evolution_preserves_sizes_and_perturbs_fresh_clones():
    mf = make_mfpbt(n_subpop=2, n_ind=4)
    pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
    max_index_before = max(a.index for a in pop)
    fm = FakeMutations(mutate_elite=False)

    evolved = mf.evolution(pop, subpop=0, mutation=fm)

    # Total population and the studied subpop both keep their size.
    assert len(evolved) == len(pop)
    assert sum(a.subpopulation == 0 for a in evolved) == 4
    # The loser (fitness 1.0) was removed.
    assert 1.0 not in [a.fitness[-1] for a in evolved if a.subpopulation == 0]
    # Exactly one fresh clone was introduced at -inf with a brand-new index.
    new_clones = [
        a for a in evolved if a.subpopulation == 0 and a.fitness[-1] == -math.inf
    ]
    assert len(new_clones) == 1
    assert new_clones[0].index > max_index_before
    # Every introduced clone was perturbed, with mutate_elite forced True.
    assert all(a.perturbed for a in new_clones)
    assert fm.seen_mutate_elite is True


def test_evolution_does_not_mutate_input_population_list():
    mf = make_mfpbt()
    pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})
    pop_copy = list(pop)

    mf.evolution(pop, subpop=0, mutation=FakeMutations())

    assert pop == pop_copy  # the caller's list object is untouched


def test_evolution_winner_clone_selection_is_reproducible():
    # Two winners with distinguishable markers, two losers -> two random picks.
    weights = {0: ["W_a", "W_b", "x", "y"], 1: ["a", "b", "c", "d"]}
    pop_kwargs = dict(
        subpop_fitnesses={0: [4.0, 3.0, 2.0, 1.0], 1: [8, 7, 6, 5]}, weights=weights
    )

    def run():
        pop = make_population(**pop_kwargs)
        mf = make_mfpbt(w=2, s=0, o=0, ln=2, seed=7)
        evolved = mf.evolution(pop, subpop=0, mutation=FakeMutations(mutate_elite=True))
        return sorted(a.weights for a in evolved if a.fitness[-1] == -math.inf)

    assert run() == run()  # identical winner picks for the same seed


# --------------------------------------------------------------------------- #
# migration (asymmetric, paper Algorithm 2)
# --------------------------------------------------------------------------- #
def test_migration_slow_to_fast_keeps_external_weights_but_elite_hps():
    # studied subpop 1 (delta=2); external subpop 0 (delta=1) -> delta_ext < delta_studied.
    mf = make_mfpbt(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({0: [9.0, 8.0, 7.0, 6.0], 1: [5.0, 4.0, 3.0, 2.0]})
    for a in pop:
        if a.subpopulation == 0:
            a.weights = "EXT"
            a.lr = 0.5
        else:
            a.lr = 0.001  # subpop-1 elite lr

    migrated = mf.migration(pop, subpop=1)

    movers = [
        a for a in migrated if a.subpopulation == 1 and a.fitness[-1] == -math.inf
    ]
    assert len(movers) == 1
    mover = movers[0]
    assert mover.weights == "EXT"  # external network params imported
    assert mover.lr == 0.001  # but HPs reset to the studied subpop's elite
    assert mover.reinit_called is True  # optimizer reinitialised at the elite lr


def test_migration_full_clone_when_external_not_faster():
    # studied subpop 0 (delta=1); external subpop 1 (delta=2) -> delta_ext >= delta_studied.
    mf = make_mfpbt(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({0: [5.0, 4.0, 3.0, 2.0], 1: [9.0, 8.0, 7.0, 6.0]})
    for a in pop:
        if a.subpopulation == 1:
            a.weights = "EXT"
            a.lr = 0.5

    migrated = mf.migration(pop, subpop=0)

    movers = [
        a for a in migrated if a.subpopulation == 0 and a.fitness[-1] == -math.inf
    ]
    assert len(movers) == 1
    assert movers[0].weights == "EXT"
    assert movers[0].lr == 0.5  # full clone keeps the external HPs


def test_migration_skips_when_open_agent_already_better():
    # The open agent of subpop 0 beats every external agent -> no migration.
    mf = make_mfpbt(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({0: [100.0, 99.0, 98.0, 1.0], 1: [9.0, 8.0, 7.0, 6.0]})
    open_agent = next(a for a in pop if a.subpopulation == 0 and a.fitness[-1] == 98.0)

    migrated = mf.migration(pop, subpop=0)

    assert open_agent in migrated  # untouched
    assert not [
        a for a in migrated if a.subpopulation == 0 and a.fitness[-1] == -math.inf
    ]


# --------------------------------------------------------------------------- #
# empty brackets (sizes may be 0)
# --------------------------------------------------------------------------- #
def test_evolution_with_zero_survivors_replaces_every_loser():
    # brackets 1/0/1/2: one winner, no survivors, one open, two losers.
    mf = make_mfpbt(n_subpop=2, n_ind=4, w=1, s=0, o=1, ln=2)
    pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

    evolved = mf.evolution(pop, subpop=0, mutation=FakeMutations())

    assert sum(a.subpopulation == 0 for a in evolved) == 4  # size preserved
    clones = [a for a in evolved if a.subpopulation == 0 and a.fitness[-1] == -math.inf]
    assert len(clones) == 2  # both losers replaced by winner-clones


def test_evolution_with_no_losers_is_a_noop():
    mf = make_mfpbt(n_subpop=2, n_ind=4, w=2, s=2, o=0, ln=0)
    pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

    evolved = mf.evolution(pop, subpop=0, mutation=FakeMutations())

    assert evolved == pop  # nothing removed or added


def test_migration_with_empty_open_bracket_is_a_noop():
    # A frozen subpopulation (no winners/open) must not crash on winners[0].
    mf = make_mfpbt(n_subpop=2, n_ind=4, ratios=[1, 2], w=0, s=4, o=0, ln=0)
    pop = make_population({0: [4.0, 3.0, 2.0, 1.0], 1: [8.0, 7.0, 6.0, 5.0]})

    migrated = mf.migration(pop, subpop=0)

    assert migrated == pop


# --------------------------------------------------------------------------- #
# evolve_population scheduling
# --------------------------------------------------------------------------- #
def test_counter_schedules_subpops_at_their_frequencies():
    mf = make_mfpbt(n_subpop=3, n_ind=4, ratios=[1, 2, 3])
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]})
    fm = FakeMutations()

    fired = []  # (cycle, subpop)
    cycle = {"n": 0}

    def record_evolution(population, subpop, mutation):
        fired.append((cycle["n"], subpop))
        return population

    mf.evolution = record_evolution
    mf.migration = lambda population, subpop: population

    for c in range(1, 7):
        cycle["n"] = c
        mf.evolve_population(pop, mutation=fm)

    assert [c for c, s in fired if s == 0] == [1, 2, 3, 4, 5, 6]  # delta=1
    assert [c for c, s in fired if s == 1] == [2, 4, 6]  # delta=2
    assert [c for c, s in fired if s == 2] == [3, 6]  # delta=3


def test_evolve_population_saves_global_elite(tmp_path):
    mf = make_mfpbt(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
    elite_path = str(tmp_path / "best.pt")

    mf.evolve_population(
        pop, mutation=FakeMutations(), save_elite=True, elite_path=elite_path
    )

    import os

    assert os.path.exists(elite_path)


def test_evolve_population_rejects_accelerator():
    mf = make_mfpbt()
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
    with pytest.raises(NotImplementedError):
        mf.evolve_population(pop, mutation=FakeMutations(), accelerator=object())
