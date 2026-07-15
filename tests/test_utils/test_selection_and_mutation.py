"""Tests for the HPO-strategy dispatch and MF-PBT orchestration in utils."""

from __future__ import annotations

import os

import pytest

import agilerl.utils.utils as utils
from agilerl.hpo.multi_frequency import MultiFrequencyStrategy
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    multi_frequency_selection_and_mutation,
    resolve_selection_strategy,
    run_selection_and_mutation,
)


class FakeAgent:
    def __init__(self, index, subpopulation, fitness):
        self.index = index
        self.subpopulation = subpopulation
        self.fitness = [fitness]

    def save_checkpoint(self, path):
        with open(path, "w") as fh:
            fh.write("checkpoint")


class FakeMutations:
    mutate_elite = False

    def mutation(self, population, pre_training_mut=False, indices=None):
        return population


def make_population(subpop_fitnesses):
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for fit in fitnesses:
            population.append(FakeAgent(idx, subpop, fit))
            idx += 1
    return population


def make_strategy(n_subpop=2, n_ind=4, ratios=None):
    return MultiFrequencyStrategy(
        n_subpopulations=n_subpop,
        n_individuals_per_subpopulation=n_ind,
        evolution_frequency_ratios=ratios or list(range(1, n_subpop + 1)),
        n_winners=1,
        n_survivors=1,
        n_open_for_migration=1,
        n_losers=1,
    )


def test_dispatch_none_returns_population_unchanged():
    pop = [1, 2, 3]
    out = run_selection_and_mutation(
        None, population=pop, mutation=FakeMutations(), env_name="env"
    )
    assert out is pop


def test_dispatch_routes_tournament(monkeypatch):
    sentinel = ["mutated"]
    called = {}

    def fake_tsm(**kwargs):
        called.update(kwargs)
        return sentinel

    monkeypatch.setattr(utils, "tournament_selection_and_mutation", fake_tsm)
    strategy = TournamentSelection(tournament_size=2, elitism=True, population_size=4)

    out = run_selection_and_mutation(
        strategy, population=[1], mutation=FakeMutations(), env_name="env"
    )

    assert out is sentinel
    assert called["tournament"] is strategy


def test_dispatch_routes_mf_pbt(monkeypatch):
    sentinel = ["evolved"]
    called = {}

    def fake_mfsm(**kwargs):
        called.update(kwargs)
        return sentinel

    monkeypatch.setattr(utils, "multi_frequency_selection_and_mutation", fake_mfsm)
    strategy = make_strategy()

    out = run_selection_and_mutation(
        strategy, population=[1], mutation=FakeMutations(), env_name="env"
    )

    assert out is sentinel
    assert called["multi_frequency_strategy"] is strategy


def test_dispatch_mf_pbt_rejects_language_model():
    with pytest.raises(NotImplementedError, match="language_model"):
        run_selection_and_mutation(
            make_strategy(),
            population=[1],
            mutation=FakeMutations(),
            env_name="env",
            language_model=True,
        )


def test_dispatch_routes_duck_typed_tournament(monkeypatch):
    # Backward compatibility: any non-None, non-MF-PBT strategy routes to the
    # tournament path
    sentinel = ["mutated"]
    called = {}

    def fake_tsm(**kwargs):
        called.update(kwargs)
        return sentinel

    monkeypatch.setattr(utils, "tournament_selection_and_mutation", fake_tsm)
    duck = object()

    out = run_selection_and_mutation(
        duck, population=[1], mutation=FakeMutations(), env_name="env"
    )

    assert out is sentinel
    assert called["tournament"] is duck


def _stub_operator_steps(strategy):
    """Stub the per-subpopulation clone/migrate steps.

    Lets the fake agents drive the scheduling/elite logic without needing clone or
    a full registry; select still runs its real counter and bracketing logic.
    """
    strategy._clone_winners_over_losers = lambda population, winners, losers, subpop: (
        population,
        [],
    )
    strategy._migrate = (
        lambda population, subpop, winners, open_for_migration, external_pool: (
            population
        )
    )


def test_orchestration_schedules_subpops_at_their_frequencies():
    strategy = make_strategy(n_subpop=3, n_ind=4, ratios=[1, 2, 3])
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]})
    fm = FakeMutations()

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
        multi_frequency_selection_and_mutation(pop, strategy, mutation=fm)

    assert [c for c, s in fired if s == 0] == [1, 2, 3, 4, 5, 6]  # delta=1
    assert [c for c, s in fired if s == 1] == [2, 4, 6]  # delta=2
    assert [c for c, s in fired if s == 2] == [3, 6]  # delta=3


def test_orchestration_saves_global_elite(tmp_path):
    strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
    elite_path = str(tmp_path / "best.pt")
    # Elite is saved before any subpopulation evolves; stub the operator steps so
    # the fake agents need only ``save_checkpoint``.
    _stub_operator_steps(strategy)

    multi_frequency_selection_and_mutation(
        pop, strategy, mutation=FakeMutations(), save_elite=True, elite_path=elite_path
    )

    assert os.path.exists(elite_path)


def test_orchestration_rejects_accelerator():
    strategy = make_strategy()
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
    with pytest.raises(NotImplementedError, match="Accelerate"):
        multi_frequency_selection_and_mutation(
            pop, strategy, mutation=FakeMutations(), accelerator=object()
        )


def test_orchestration_assigns_missing_subpopulations():
    strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({None: [8, 7, 6, 5, 4, 3, 2, 1]})
    for agent in pop:
        agent.subpopulation = None
    _stub_operator_steps(strategy)

    multi_frequency_selection_and_mutation(pop, strategy, mutation=FakeMutations())

    assert sorted(a.subpopulation for a in pop) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_orchestration_migrates_against_pre_evolution_snapshot():
    strategy = make_strategy(n_subpop=2, n_ind=4, ratios=[1, 2])
    pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
    snapshot = list(pop)

    # Cloning swaps in fresh objects, so sourcing migrants from the live (evolved)
    # population would be visibly different from the pre-evolution snapshot.
    def fake_clone(population, winners, losers, subpop):
        fresh = [
            FakeAgent(100 + i, a.subpopulation, a.fitness[-1])
            for i, a in enumerate(population)
        ]
        return fresh, []

    captured: dict[int, list] = {}

    def fake_migrate(population, subpop, winners, open_for_migration, external_pool):
        captured[subpop] = external_pool
        return population

    strategy._clone_winners_over_losers = fake_clone
    strategy._migrate = fake_migrate
    strategy.counters = [0, 1]  # a single cycle then fires BOTH subpopulations

    multi_frequency_selection_and_mutation(pop, strategy, mutation=FakeMutations())

    # Both subpopulations fired, and each migration saw the identical pre-evolution
    # snapshot -- subpop 1 ran after subpop 0's cloning replaced the live objects.
    assert set(captured) == {0, 1}
    assert captured[0] == snapshot
    assert captured[1] == snapshot


def test_resolve_prefers_new_argument_without_warning(recwarn):
    strategy = make_strategy()
    assert resolve_selection_strategy(strategy, None) is strategy
    assert len(recwarn) == 0


def test_resolve_folds_deprecated_tournament_with_warning():
    tournament = TournamentSelection(tournament_size=2, elitism=True, population_size=4)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        resolved = resolve_selection_strategy(None, tournament)
    assert resolved is tournament


def test_resolve_conflict_prefers_selection_strategy():
    strategy = make_strategy()
    tournament = TournamentSelection(tournament_size=2, elitism=True, population_size=4)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        resolved = resolve_selection_strategy(strategy, tournament)
    assert resolved is strategy
