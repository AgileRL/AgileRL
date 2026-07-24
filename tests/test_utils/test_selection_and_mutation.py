"""Tests for the unified selection-and-mutation entry point."""

from __future__ import annotations

import os

import pytest

from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    resolve_selection_strategy,
    run_selection_and_mutation,
)
from tests.test_hpo.test_multi_frequency import (
    FakeAgent as _OperatorFakeAgent,
)
from tests.test_hpo.test_multi_frequency import (
    make_strategy,
    new_agents,
)


class FakeAgent(_OperatorFakeAgent):
    """The operator's agent stand-in plus the accelerator round-trip bookkeeping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unwrap_calls = 0
        self.wrap_calls = 0
        self.saved: list[str] = []
        self.loaded: list[str] = []

    def save_checkpoint(self, path):
        self.saved.append(path)
        super().save_checkpoint(path)

    def load_checkpoint(self, path):
        self.loaded.append(path)

    def unwrap_models(self):
        self.unwrap_calls += 1

    def wrap_models(self):
        self.wrap_calls += 1


class FakeAccelerator:
    """Minimal stand-in for a HuggingFace accelerator."""

    def __init__(self, is_main_process):
        self.is_main_process = is_main_process
        self.wait_count = 0

    def wait_for_everyone(self):
        self.wait_count += 1


class FakeMutations:
    mutate_elite = False

    def mutation(self, population, pre_training_mut=False, indices=None):
        return population


class FakeStrategy:
    """Minimal selection strategy exposing the unified select contract."""

    def __init__(self, elite, new_population, indices):
        self._result = (elite, new_population, indices)
        self.select_calls: list = []

    def select(self, population):
        self.select_calls.append(population)
        return self._result


class RecordingMutations:
    """Mutation stub that records the indices argument of every call."""

    mutate_elite = False

    def __init__(self, result=None):
        self._result = result
        self.indices_seen: list = []

    def mutation(self, population, pre_training_mut=False, indices=None):
        self.indices_seen.append(indices)
        return self._result if self._result is not None else population


def make_population(subpop_fitnesses):
    """Build a population of the accelerator-aware FakeAgent with unique indices."""
    population = []
    idx = 0
    for subpop, fitnesses in subpop_fitnesses.items():
        for fit in fitnesses:
            population.append(FakeAgent(idx, subpop, fit))
            idx += 1
    return population


class TestRunSelectionAndMutation:
    def test_none_returns_population_unchanged(self):
        pop = [1, 2, 3]
        out = run_selection_and_mutation(
            None, population=pop, mutation=RecordingMutations(), env_name="env"
        )
        assert out is pop

    def test_selects_then_mutates_with_reported_indices(self):
        pop = [object()]
        evolved = [object(), object()]
        strategy = FakeStrategy(elite=None, new_population=evolved, indices=[7])
        mutation = RecordingMutations(result=["mutated"])

        out = run_selection_and_mutation(
            strategy, population=pop, mutation=mutation, env_name="env"
        )

        assert strategy.select_calls == [pop]
        assert mutation.indices_seen == [[7]]
        assert out == ["mutated"]

    def test_tournament_style_indices_none_mutates_whole_population(self):
        strategy = FakeStrategy(elite=None, new_population=[1], indices=None)
        mutation = RecordingMutations()

        run_selection_and_mutation(
            strategy, population=[1], mutation=mutation, env_name="env"
        )

        assert mutation.indices_seen == [None]

    def test_saves_elite_from_select_result(self, tmp_path):
        elite = FakeAgent(0, 0, 5.0)
        strategy = FakeStrategy(elite=elite, new_population=[elite], indices=None)
        elite_path = str(tmp_path / "elite.pt")

        run_selection_and_mutation(
            strategy,
            population=[elite],
            mutation=RecordingMutations(),
            env_name="env",
            save_elite=True,
            elite_path=elite_path,
        )

        assert elite.saved == [elite_path]

    def test_multi_frequency_language_model_dispatches_through_llm_branch(
        self, monkeypatch, tmp_path
    ):
        strategy = make_strategy()
        elite = object()
        monkeypatch.setattr(strategy, "select", lambda pop: (elite, ["evolved"], [3]))
        saved: list = []
        monkeypatch.setattr(
            "agilerl.utils.utils.save_llm_checkpoint",
            lambda agent, path: saved.append((agent, path)),
        )
        mutation = RecordingMutations(result=["mutated"])
        elite_path = str(tmp_path / "elite")

        out = run_selection_and_mutation(
            strategy,
            population=[1],
            mutation=mutation,
            env_name="env",
            language_model=True,
            save_elite=True,
            elite_path=elite_path,
        )

        assert out == ["mutated"]
        assert mutation.indices_seen == [[3]]  # only the winner clones are perturbed
        assert saved == [(elite, elite_path)]

    def test_multi_frequency_language_model_consolidates_under_accelerator(
        self, monkeypatch
    ):
        strategy = make_strategy()
        monkeypatch.setattr(
            strategy, "select", lambda pop: (object(), ["evolved"], [3])
        )
        consolidated: list = []
        monkeypatch.setattr(
            "agilerl.utils.utils.consolidate_mutations",
            lambda pop: consolidated.append(pop),
        )
        accelerator = FakeAccelerator(is_main_process=True)
        mutation = RecordingMutations(result=["mutated"])

        run_selection_and_mutation(
            strategy,
            population=[1],
            mutation=mutation,
            env_name="env",
            language_model=True,
            accelerator=accelerator,
        )

        assert mutation.indices_seen == [[3]]
        assert consolidated == [["mutated"]]  # mutation decisions broadcast to workers


class TestMultiFrequencyOrchestration:
    def test_orchestration_schedules_subpops_at_their_frequencies(self):
        strategy = make_strategy(n_subpop=3, population_size=12, ratios=[1, 2, 3])
        pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5], 2: [12, 11, 10, 9]})
        fired = []  # (cycle, subpop)

        for cycle in range(1, 7):
            new_pop = run_selection_and_mutation(
                strategy, population=pop, mutation=FakeMutations(), env_name="env"
            )
            fired.extend(
                (cycle, subpop)
                for subpop in sorted(
                    {a.subpopulation for a in new_agents(pop, new_pop)}
                )
            )
            pop = new_pop

        assert [c for c, s in fired if s == 0] == [1, 2, 3, 4, 5, 6]  # delta=1
        assert [c for c, s in fired if s == 1] == [2, 4, 6]  # delta=2
        assert [c for c, s in fired if s == 2] == [3, 6]  # delta=3

    def test_orchestration_saves_global_elite(self, tmp_path):
        strategy = make_strategy(n_subpop=2, population_size=8, ratios=[1, 2])
        pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
        elite_path = str(tmp_path / "best.pt")

        run_selection_and_mutation(
            strategy,
            population=pop,
            mutation=FakeMutations(),
            env_name="env",
            save_elite=True,
            elite_path=elite_path,
        )

        assert os.path.exists(elite_path)

    def test_orchestration_accelerator_main_process_evolves_and_saves(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        strategy = make_strategy(n_subpop=2, population_size=8, ratios=[1, 2])
        pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
        accel = FakeAccelerator(is_main_process=True)
        elite_path = str(tmp_path / "best.pt")

        out = run_selection_and_mutation(
            strategy,
            population=pop,
            mutation=FakeMutations(),
            env_name="env",
            accelerator=accel,
            save_elite=True,
            elite_path=elite_path,
        )

        # select() ran on the main process: subpop 0 (delta 1) fired and reset its counter,
        # subpop 1 (delta 2) has not fired yet
        assert strategy.counters == [0, 1]
        assert os.path.exists(elite_path)
        for agent in pop:
            assert agent.unwrap_calls == 1
        for agent in out:
            assert agent.wrap_calls == 1
            assert len(agent.saved) == 1
            assert agent.loaded == []
        assert accel.wait_count >= 3

    def test_orchestration_accelerator_worker_loads_without_evolving(self):
        strategy = make_strategy(n_subpop=2, population_size=8, ratios=[1, 2])
        pop = make_population({0: [4, 3, 2, 1], 1: [8, 7, 6, 5]})
        accel = FakeAccelerator(is_main_process=False)

        out = run_selection_and_mutation(
            strategy,
            population=pop,
            mutation=FakeMutations(),
            env_name="env",
            accelerator=accel,
            algo="DQN",
        )

        assert out is pop
        assert strategy.counters == [0, 0]  # counters untouched -> select() did not run
        for i, agent in enumerate(out):
            assert agent.unwrap_calls == 1
            assert agent.wrap_calls == 1
            assert agent.loaded == [f"models/env/DQN_{i}.pt"]
            assert agent.saved == []
        assert accel.wait_count >= 3

    def test_orchestration_assigns_missing_subpopulations(self):
        strategy = make_strategy(n_subpop=2, population_size=8, ratios=[1, 2])
        pop = make_population({None: [8, 7, 6, 5, 4, 3, 2, 1]})
        for agent in pop:
            agent.subpopulation = None

        run_selection_and_mutation(
            strategy, population=pop, mutation=FakeMutations(), env_name="env"
        )

        assert sorted(a.subpopulation for a in pop) == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_orchestration_migrates_against_pre_evolution_snapshot(self):
        strategy = make_strategy(n_subpop=2, population_size=8, ratios=[1, 2])
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

        def fake_migrate(
            population, subpop, winners, open_for_migration, external_pool
        ):
            captured[subpop] = external_pool
            return population

        strategy._clone_winners_over_losers = fake_clone
        strategy._migrate = fake_migrate
        strategy.counters = [0, 1]  # a single cycle then fires BOTH subpopulations

        run_selection_and_mutation(
            strategy, population=pop, mutation=FakeMutations(), env_name="env"
        )

        # Both subpopulations fired, and each migration saw the identical pre-evolution
        # snapshot -- subpop 1 ran after subpop 0's cloning replaced the live objects.
        assert set(captured) == {0, 1}
        assert captured[0] == snapshot
        assert captured[1] == snapshot


class TestResolveSelectionStrategy:
    def test_resolve_prefers_new_argument_without_warning(self, recwarn):
        strategy = make_strategy()
        assert resolve_selection_strategy(strategy, None) is strategy
        assert len(recwarn) == 0

    def test_resolve_folds_deprecated_tournament_with_warning(self):
        tournament = TournamentSelection(
            tournament_size=2, elitism=True, population_size=4
        )
        with pytest.warns(DeprecationWarning, match="deprecated"):
            resolved = resolve_selection_strategy(None, tournament)
        assert resolved is tournament

    def test_resolve_conflict_prefers_selection_strategy(self):
        strategy = make_strategy()
        tournament = TournamentSelection(
            tournament_size=2, elitism=True, population_size=4
        )
        with pytest.warns(DeprecationWarning, match="deprecated"):
            resolved = resolve_selection_strategy(strategy, tournament)
        assert resolved is strategy
