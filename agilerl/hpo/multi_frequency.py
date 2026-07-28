"""Multiple-Frequencies Population-Based Training (MF-PBT).

Implements the MF-PBT evolution regime of Doulazmi et al.,
*"Multiple-Frequencies Population-Based Training"*, as a drop-in alternative to
AgileRL's tournament-selection + mutation evolution.

A population of population_size agents is split into n_subpopulations
subpopulations of population_size // n_subpopulations agents each, and each
subpopulation evolves at its own frequency delta_i (every delta_i evolution
cycles). All agents still train evo_steps per cycle, so the global evolution
frequency, and therefore the granularity of the discovered hyperparameter
schedules, is preserved, while the slower subpopulations resist the greediness
that makes single-frequency PBT settle into local optima.

Each subpopulation is partitioned by within-subpopulation fitness rank into four
brackets (*winners*, *survivors*, *open-for-migration* and *losers*) that drive
exploitation (clone a winner over a loser, then perturb), preservation (survivors
and open agents are untouched by :meth:`~MultiFrequencySelection.select`) and the
asymmetric cross-frequency *migration* of Algorithm 2 of the paper.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any

import numpy as np
from accelerate.utils import broadcast_object_list

from agilerl.algorithms.core.base import EvolvableAlgorithm, LLMAlgorithm

PopulationT = list[EvolvableAlgorithm]


class MultiFrequencyOp(str, Enum):
    """Operation code tagging a single slot of an MF-PBT generation plan."""

    # The slot is left untouched this cycle
    KEEP = "keep"
    # A loser slot is overwritten by a clone of one of its subpopulation's winners
    CLONE = "clone"
    # An open slot imports a full clone of an external migrant
    MIGRATE_FULL = "migrate_full"
    # An open slot imports an external migrant's networks but resets its mutable
    # hyperparameters to the destination subpopulation's elite
    MIGRATE_WEIGHTS = "migrate_weights"


MigrationDecision = tuple[
    EvolvableAlgorithm, EvolvableAlgorithm, MultiFrequencyOp, EvolvableAlgorithm
]


class MultiFrequencySelection:
    """The multi-frequency selection operator needed in MF-PBT.

    :param population_size: Total number of agents in the population (>= 6, a
        multiple of n_subpopulations, and large enough that
        population_size // n_subpopulations >= 3).
    :type population_size: int
    :param n_subpopulations: Number of subpopulations (>= 2; migration has nothing
        to draw from with a single subpopulation).
    :type n_subpopulations: int
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency ratios
        delta_i (strictly increasing integers, delta_i >= 1; one per
        subpopulation). None or [] -> [1, 5, 10, ...].
    :type evolution_frequency_ratios: list[int] | None
    :param n_winners: Agents in the winners bracket (>= 1; None ->
        round(0.25 * subpopulation_size)).
    :type n_winners: int | None
    :param n_survivors: Agents in the survivors bracket (>= 0; None -> 0).
    :type n_survivors: int | None
    :param n_open_for_migration: Agents in the open-for-migration bracket (>= 1;
        None -> round(0.25 * subpopulation_size)).
    :type n_open_for_migration: int | None
    :param n_losers: Agents in the losers bracket (>= 1; None -> the remainder
        subpopulation_size - n_winners - n_survivors - n_open_for_migration).
    :type n_losers: int | None
    :param seed: Seed for the reproducible winner-clone selection in
        :meth:`_clone_winners_over_losers`, derived from the run's global seed. None
        leaves the RNG unseeded.
    :type seed: int | None
    :raises ValueError: If population_size < 6, population_size is not a multiple of
        n_subpopulations, population_size // n_subpopulations < 3,
        n_subpopulations < 2, n_winners < 1, n_survivors < 0,
        n_open_for_migration < 1, n_losers < 1, the four brackets do not sum to
        population_size // n_subpopulations, or the frequency ratios are not
        n_subpopulations strictly-increasing integers >= 1.
    """

    def __init__(
        self,
        population_size: int,
        n_subpopulations: int = 2,
        evolution_frequency_ratios: list[int] | None = None,
        n_winners: int | None = None,
        n_survivors: int | None = None,
        n_open_for_migration: int | None = None,
        n_losers: int | None = None,
        seed: int | None = None,
    ) -> None:
        (
            self.population_size,
            self.n_subpopulations,
            self.deltas,
            self.n_winners,
            self.n_survivors,
            self.n_open_for_migration,
            self.n_losers,
        ) = self._resolve_and_validate(
            population_size,
            n_subpopulations,
            evolution_frequency_ratios,
            n_winners,
            n_survivors,
            n_open_for_migration,
            n_losers,
        )

        self.subpopulation_size = self.population_size // self.n_subpopulations
        self.bracket_sizes = (
            self.n_winners,
            self.n_survivors,
            self.n_open_for_migration,
            self.n_losers,
        )

        # Per-subpopulation evolution counters (persist across cycles)
        self.counters = [0] * self.n_subpopulations
        # Reproducible RNG for the winner-clone selection
        self.rng = np.random.default_rng(seed)
        self._max_index: int | None = None

    @staticmethod
    def _resolve_and_validate(
        population_size: int,
        n_subpopulations: int,
        evolution_frequency_ratios: list[int] | None,
        n_winners: int | None,
        n_survivors: int | None,
        n_open_for_migration: int | None,
        n_losers: int | None,
    ) -> tuple[int, int, list[int], int, int, int, int]:
        """Resolve the defaults, then hard-check the operator's invariants.

        :param population_size: Total population size (>= 6, a multiple of
            n_subpopulations, with population_size // n_subpopulations >= 3).
        :type population_size: int
        :param n_subpopulations: Number of subpopulations (>= 2).
        :type n_subpopulations: int
        :param evolution_frequency_ratios: Per-subpopulation frequency ratios, or
            None/[] for [1, 5, 10, ...].
        :type evolution_frequency_ratios: list[int] | None
        :param n_winners: Winners bracket size, or None for
            round(0.25 * subpopulation_size).
        :type n_winners: int | None
        :param n_survivors: Survivors bracket size (>= 0; None -> 0).
        :type n_survivors: int | None
        :param n_open_for_migration: Open-for-migration bracket size, or None for
            round(0.25 * subpopulation_size).
        :type n_open_for_migration: int | None
        :param n_losers: Losers bracket size, or None for the remainder
            subpopulation_size - n_winners - n_survivors - n_open_for_migration.
        :type n_losers: int | None
        :return: The resolved (population_size, n_subpopulations,
            evolution_frequency_ratios, n_winners, n_survivors, n_open_for_migration,
            n_losers).
        :rtype: tuple[int, int, list[int], int, int, int, int]
        :raises ValueError: On any violated invariant.
        """
        if population_size < 6:
            msg = (
                "population_size must be >= 6 (the smallest MF-PBT layout is 2 "
                f"subpopulations of 3 agents), got {population_size}."
            )
            raise ValueError(msg)
        if n_subpopulations < 2:
            msg = f"n_subpopulations must be >= 2, got {n_subpopulations}."
            raise ValueError(msg)
        if population_size % n_subpopulations != 0:
            msg = (
                f"population_size ({population_size}) must be divisible by "
                f"n_subpopulations ({n_subpopulations})."
            )
            raise ValueError(msg)

        subpop_size = population_size // n_subpopulations
        if subpop_size < 3:
            msg = (
                "population_size // n_subpopulations must be >= 3 "
                "so each subpopulation can host at least one "
                "winner, one open-for-migration agent and one loser; got "
                f"population_size={population_size}, "
                f"n_subpopulations={n_subpopulations} -> subpopulation_size={subpop_size}."
            )
            raise ValueError(msg)

        # Resolve the None bracket defaults
        if n_winners is None:
            n_winners = round(0.25 * subpop_size)
        if n_survivors is None:
            n_survivors = 0
        if n_open_for_migration is None:
            n_open_for_migration = round(0.25 * subpop_size)
        if n_losers is None:
            n_losers = subpop_size - n_winners - n_survivors - n_open_for_migration

        if n_winners < 1:
            msg = f"n_winners must be >= 1, got {n_winners}."
            raise ValueError(msg)
        if n_survivors < 0:
            msg = f"n_survivors must be >= 0, got {n_survivors}."
            raise ValueError(msg)
        if n_open_for_migration < 1:
            msg = f"n_open_for_migration must be >= 1, got {n_open_for_migration}."
            raise ValueError(msg)
        if n_losers < 1:
            msg = f"n_losers must be >= 1, got {n_losers}."
            raise ValueError(msg)
        bracket_sum = n_winners + n_survivors + n_open_for_migration + n_losers
        if bracket_sum != subpop_size:
            msg = (
                f"n_winners + n_survivors + n_open_for_migration + n_losers "
                f"({bracket_sum}) must equal population_size // n_subpopulations "
                f"({subpop_size})."
            )
            raise ValueError(msg)

        if not evolution_frequency_ratios:
            evolution_frequency_ratios = [1] + [
                5 * i for i in range(1, n_subpopulations)
            ]
        else:
            evolution_frequency_ratios = list(evolution_frequency_ratios)
        if len(evolution_frequency_ratios) != n_subpopulations:
            msg = (
                f"evolution_frequency_ratios must have length n_subpopulations "
                f"({n_subpopulations}), got {len(evolution_frequency_ratios)}."
            )
            raise ValueError(msg)
        if any(r < 1 for r in evolution_frequency_ratios):
            msg = "Each evolution_frequency_ratio must be >= 1."
            raise ValueError(msg)
        if any(
            evolution_frequency_ratios[i] >= evolution_frequency_ratios[i + 1]
            for i in range(len(evolution_frequency_ratios) - 1)
        ):
            msg = "evolution_frequency_ratios must be strictly increasing."
            raise ValueError(msg)

        return (
            population_size,
            n_subpopulations,
            evolution_frequency_ratios,
            n_winners,
            n_survivors,
            n_open_for_migration,
            n_losers,
        )

    @staticmethod
    def _scalar_fitness(fitness: float | np.ndarray | dict[str, float]) -> float:
        """Reduce a vector-valued fitness to a single scalar for ranking.

        :param fitness: The agent's latest fitness value.
        :type fitness: float | numpy.ndarray | dict[str, float]
        :return: The scalar fitness used for ranking.
        :rtype: float
        """
        if isinstance(fitness, dict):
            return float(np.mean(list(fitness.values())))
        if isinstance(fitness, (list, tuple, np.ndarray)):
            return float(np.mean(fitness))
        return float(fitness)

    def _rank(self, agents: PopulationT) -> PopulationT:
        """Return agents sorted by scalar fitness, highest first.

        :param agents: The agents to rank.
        :type agents: list
        :return: A new list ordered by descending scalar fitness.
        :rtype: list
        """
        return sorted(
            agents,
            key=lambda a: self._scalar_fitness(a.fitness[-1]),
            reverse=True,
        )

    def _sync_index(self, population: PopulationT) -> None:
        """Seed the index allocator from the current population's max index.

        :param population: The whole population.
        :type population: list
        """
        current = max(a.index for a in population)
        self._max_index = (
            current if self._max_index is None else max(self._max_index, current)
        )

    def _next_index(self) -> int:
        """Return a fresh, globally-unique agent index.

        :return: The next unused agent index.
        :rtype: int
        """
        self._max_index += 1
        return self._max_index

    @staticmethod
    def _subpopulation_for_position(position: int, subpopulation_size: int) -> int:
        """Map an agent's population slot position to its subpopulation id.

        Both the build-time tagging in
        :func:`agilerl.utils.trainer_utils._assign_subpopulations` and the defensive
        tagging in :meth:`_assign_initial_subpopulations` route through this method.

        :param position: The agent's position in the population list.
        :type position: int
        :param subpopulation_size: Agents per subpopulation.
        :type subpopulation_size: int
        :return: The subpopulation id the agent belongs to.
        :rtype: int
        """
        return position // subpopulation_size

    def _assign_initial_subpopulations(self, population: PopulationT) -> None:
        """Tag any agent lacking a subpopulation.

        Makes the operator robust when driven through the functional trainers with
        a population that was not tagged at build time.

        :param population: The whole population.
        :type population: list
        :raises ValueError: If len(population) does not equal population_size, or the
            agent indices are not globally unique.
        """
        if len(population) != self.population_size:
            msg = (
                f"Population has {len(population)} agents, expected "
                f"{self.population_size} (n_subpopulations * subpopulation_size = "
                f"{self.n_subpopulations} * {self.subpopulation_size})."
            )
            raise ValueError(msg)

        indices = [agent.index for agent in population]
        if len(set(indices)) != len(indices):
            msg = (
                f"MF-PBT requires globally-unique agent indices; got {sorted(indices)}."
            )
            raise ValueError(msg)

        for position, agent in enumerate(population):
            if getattr(agent, "subpopulation_id", None) is None:
                agent.subpopulation_id = self._subpopulation_for_position(
                    position, self.subpopulation_size
                )

    def _bracket_subpopulation(
        self, population: PopulationT, subpop: int
    ) -> tuple[PopulationT, PopulationT, PopulationT, PopulationT]:
        """Partition a subpopulation's members into the four ranked brackets.

        Members are ranked by descending fitness and sliced into
        (winners, survivors, open, losers).

        :param population: The whole population.
        :type population: list
        :param subpop: The subpopulation id to bracket.
        :type subpop: int
        :return: (winners, survivors, open_for_migration, losers).
        :rtype: tuple[list, list, list, list]
        """
        members = self._rank([a for a in population if a.subpopulation_id == subpop])
        if len(members) != self.subpopulation_size:
            msg = (
                f"Subpopulation {subpop} has {len(members)} members, expected "
                f"{self.subpopulation_size}."
            )
            raise ValueError(msg)
        n_w, n_s, n_o, _n_l = self.bracket_sizes
        winners = members[:n_w]
        survivors = members[n_w : n_w + n_s]
        open_for_migration = members[n_w + n_s : n_w + n_s + n_o]
        losers = members[n_w + n_s + n_o :]
        return winners, survivors, open_for_migration, losers

    def select(
        self, population: PopulationT
    ) -> tuple[EvolvableAlgorithm, PopulationT, list[int]]:
        """Select the agents to be migrated and mutated during an MF-PBT evolution cycle.

        :param population: The whole population.
        :type population: list
        :return: (elite, population, indices_to_mutate). The pre-evolution global
            elite, the evolved population with migrants and clones, and the indices
            of the winner clones to be perturbed.
        :rtype: tuple[EvolvableAlgorithm, list, list[int]]
        """
        self._assign_initial_subpopulations(population)
        self._sync_index(population)

        if isinstance(population[0], LLMAlgorithm):
            return self._select_llm_agents(population)
        return self._select_standard_agents(population)

    def _select_standard_agents(
        self, population: PopulationT
    ) -> tuple[EvolvableAlgorithm, PopulationT, list[int]]:
        """Evolve a classic-RL population with in-memory cloning.

        :param population: The whole population.
        :type population: list
        :return: (elite, population, indices_to_mutate).
        :rtype: tuple[EvolvableAlgorithm, list, list[int]]
        """
        elite = max(
            population, key=lambda a: self._scalar_fitness(a.fitness[-1])
        ).clone(wrap=False)

        # The frozen snapshot makes migrations independent of the order in which
        # subpopulations are processed
        frozen = list(population)
        updated = list(population)
        indices_to_mutate: list[int] = []

        for subpop in range(self.n_subpopulations):
            self.counters[subpop] += 1
            if self.counters[subpop] < self.deltas[subpop]:
                continue
            self.counters[subpop] = 0

            winners, _survivors, open_for_migration, losers = (
                self._bracket_subpopulation(updated, subpop)
            )
            updated, clone_indices = self._clone_winners_over_losers(
                updated, winners, losers, subpop
            )
            indices_to_mutate.extend(clone_indices)
            updated = self._migrate(
                updated, subpop, winners, open_for_migration, external_pool=frozen
            )

        return elite, updated, indices_to_mutate

    def _select_llm_agents(
        self, population: PopulationT
    ) -> tuple[EvolvableAlgorithm, PopulationT, list[int]]:
        """Evolve a population of LLM agents.

        :param population: The whole population.
        :type population: list
        :return: (elite, population, indices_to_mutate).
        :rtype: tuple[EvolvableAlgorithm, list, list[int]]
        """
        accelerator = getattr(population[0], "accelerator", None)

        # Only the main process plans the generation, so the operator's mutable
        # state advances on rank 0 alone and deliberately diverges on the workers
        plan: dict[str, Any] | None = None
        if accelerator is None or accelerator.is_main_process:
            plan = self._plan_llm_evolution(population)

        # Broadcast the main process's decisions so all ranks clone identically
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                plan = broadcast_object_list([plan], from_process=0)[0]

        new_population = self._execute_llm_plan(population, plan)

        # Clones inherit their parent's stale mut, so without this reset a
        # survivor/migrant would re-broadcast a mutation it never received
        for agent in new_population:
            agent.mut = "None"

        elite = next(a for a in new_population if a.index == plan["elite_index"])
        return elite, new_population, plan["indices_to_mutate"]

    def _plan_llm_evolution(self, population: PopulationT) -> dict[str, Any]:
        """Decide a whole MF-PBT generation as a serializable, index-based plan.

        :param population: The whole population.
        :type population: list
        :return: A plan {"ops", "elite_index", "indices_to_mutate"}. ops is one
            tuple per population slot (aligned to population order), tagged by a
            leading :class:`MultiFrequencyOp`: (KEEP,),
            (CLONE, src_index, new_index, subpop),
            (MIGRATE_FULL, src_index, new_index, subpop) or
            (MIGRATE_WEIGHTS, src_index, new_index, subpop, hp_values) where
            hp_values are the destination elite's mutable HPs, resolved here so all
            ranks reset to identical values.
        :rtype: dict
        """
        self._sync_index(population)
        elite_index = max(
            population, key=lambda a: self._scalar_fitness(a.fitness[-1])
        ).index

        ops: dict[int, tuple] = {a.index: (MultiFrequencyOp.KEEP,) for a in population}
        indices_to_mutate: list[int] = []

        for subpop in range(self.n_subpopulations):
            self.counters[subpop] += 1
            if self.counters[subpop] < self.deltas[subpop]:
                continue
            self.counters[subpop] = 0

            winners, _survivors, open_for_migration, losers = (
                self._bracket_subpopulation(population, subpop)
            )
            for loser in losers:
                winner = winners[int(self.rng.integers(len(winners)))]
                new_index = self._next_index()
                ops[loser.index] = (
                    MultiFrequencyOp.CLONE,
                    winner.index,
                    new_index,
                    subpop,
                )
                indices_to_mutate.append(new_index)

            for open_agent, ext, kind, elite in self._migration_decisions(
                subpop, winners, open_for_migration, external_pool=population
            ):
                new_index = self._next_index()
                if kind is MultiFrequencyOp.MIGRATE_WEIGHTS:
                    hp_values = {
                        name: copy.deepcopy(getattr(elite, name))
                        for name in elite.registry.hp_config
                    }
                    ops[open_agent.index] = (
                        kind,
                        ext.index,
                        new_index,
                        subpop,
                        hp_values,
                    )
                else:
                    ops[open_agent.index] = (
                        kind,
                        ext.index,
                        new_index,
                        subpop,
                    )

        return {
            "ops": [ops[a.index] for a in population],
            "elite_index": elite_index,
            "indices_to_mutate": indices_to_mutate,
        }

    def _execute_llm_plan(
        self, population: PopulationT, plan: dict[str, Any]
    ) -> PopulationT:
        """Materialise the broadcast plan, cloning collectively and freeing dropped agents.

        To bound GPU memory (LLM agents are multi-GB) it frees an overwritten agent
        as soon as it is no longer needed.

        :param population: The pre-evolution population.
        :type population: list
        :param plan: The plan produced by :meth:`_plan_llm_evolution`.
        :type plan: dict
        :return: The evolved population, aligned to population's slot order.
        :rtype: list
        """
        ops = plan["ops"]
        by_index: dict[int, EvolvableAlgorithm | None] = {
            a.index: a for a in population
        }

        replaced_indices: set[int] = set()
        source_indices: set[int] = set()
        kept_indices: set[int] = set()
        last_use: dict[int, int] = {}
        for i, (agent, op) in enumerate(zip(population, ops, strict=True)):
            if op[0] is MultiFrequencyOp.KEEP:
                kept_indices.add(agent.index)
            else:
                replaced_indices.add(agent.index)
                source_indices.add(op[1])
                last_use[op[1]] = i

        # Free agents whose slot is overwritten and that no operation needs as a source
        for agent in population:
            if agent.index in replaced_indices and agent.index not in source_indices:
                self._clean_up(agent)
                by_index[agent.index] = None

        new_population: PopulationT = []
        for i, (agent, op) in enumerate(zip(population, ops, strict=True)):
            if op[0] is MultiFrequencyOp.KEEP:
                new_population.append(agent)
                continue
            src, new_index, subpop = op[1], op[2], op[3]
            source = by_index[src]
            assert source is not None
            clone = self._collective_clone(source, new_index)
            clone.subpopulation_id = subpop
            if op[0] is MultiFrequencyOp.MIGRATE_WEIGHTS:
                self._apply_hp_reset(clone, op[4])
            new_population.append(clone)
            # A non-kept source is freed after its last use
            if src not in kept_indices and last_use[src] == i:
                self._clean_up(source)
                by_index[src] = None

        return new_population

    @staticmethod
    def _clean_up(agent: EvolvableAlgorithm) -> None:
        """Free an agent, bracketed by its accelerator barriers.

        :param agent: The agent to free.
        :type agent: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        """
        accelerator = getattr(agent, "accelerator", None)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        agent.clean_up()
        if accelerator is not None:
            accelerator.wait_for_everyone()

    @staticmethod
    def _collective_clone(
        source: EvolvableAlgorithm, new_index: int
    ) -> EvolvableAlgorithm:
        """Clone an agent, bracketed by its accelerator barriers.

        :param source: The agent to clone.
        :type source: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param new_index: The clone's globally-unique index.
        :type new_index: int
        :return: The unwrapped clone.
        :rtype: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        """
        accelerator = getattr(source, "accelerator", None)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        clone = source.clone(index=new_index, wrap=False)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        return clone

    def _clone_winners_over_losers(
        self,
        population: PopulationT,
        winners: PopulationT,
        losers: PopulationT,
        subpop: int,
    ) -> tuple[PopulationT, list[int]]:
        """Replace each loser with a clone of a uniformly-random winner.

        :param population: The whole population (not mutated in place).
        :type population: list
        :param winners: The subpopulation's winners bracket to clone from.
        :type winners: list
        :param losers: The subpopulation's losers bracket to replace.
        :type losers: list
        :param subpop: The subpopulation id the clones belong to.
        :type subpop: int
        :return: (new_population, clone_indices). A new population list with the
            losers replaced, and the clones' indices to be perturbed.
        :rtype: tuple[list, list[int]]
        """
        self._sync_index(population)
        clone_for_loser: dict[int, EvolvableAlgorithm] = {}
        clone_indices: list[int] = []
        for loser in losers:
            winner = winners[int(self.rng.integers(len(winners)))]
            clone = winner.clone(index=self._next_index(), wrap=False)
            clone.subpopulation_id = subpop
            clone_for_loser[id(loser)] = clone
            clone_indices.append(clone.index)
        new_population = [clone_for_loser.get(id(a), a) for a in population]
        return new_population, clone_indices

    def _migrate(
        self,
        population: PopulationT,
        subpop: int,
        winners: PopulationT,
        open_for_migration: PopulationT,
        external_pool: PopulationT,
    ) -> PopulationT:
        """Asymmetrically migrate stronger agents into the open-for-migration slots.

        Implements Algorithm 2 of the paper. For each open-for-migration agent (best
        first), it is compared with the next-best agent drawn from the other
        subpopulations of external_pool. If the open agent is at least as good, no
        migration happens (and the external pointer does not advance). Otherwise the
        external agent migrates in: if it comes from a faster-evolving subpopulation
        (smaller delta) only its networks are imported while its mutable
        hyperparameters are reset to the studied subpopulation's elite; otherwise it is
        cloned in full.

        :param population: The live population, used for substitution.
        :type population: list
        :param subpop: The subpopulation id to migrate into.
        :type subpop: int
        :param winners: The subpopulation's winners bracket; winners[0] is the
            elite whose hyperparameters a weights-only migrant adopts.
        :type winners: list
        :param open_for_migration: The subpopulation's open-for-migration bracket
            (best first) whose slots migrants may fill.
        :type open_for_migration: list
        :param external_pool: The pre-evolution population snapshot that migrant
            sources are drawn from.
        :type external_pool: list
        :return: A new population list with migrants substituted in place.
        :rtype: list
        """
        self._sync_index(population)
        replacements: dict[int, EvolvableAlgorithm] = {}
        for open_agent, ext, kind, elite in self._migration_decisions(
            subpop, winners, open_for_migration, external_pool
        ):
            migrant = (
                self._migrate_weights(ext, elite, subpop)
                if kind is MultiFrequencyOp.MIGRATE_WEIGHTS
                else self._migrate_full_clone(ext, subpop)
            )
            replacements[id(open_agent)] = migrant

        return [replacements.get(id(a), a) for a in population]

    def _migration_decisions(
        self,
        subpop: int,
        winners: PopulationT,
        open_for_migration: PopulationT,
        external_pool: PopulationT,
    ) -> list[MigrationDecision]:
        """Decide the asymmetric migrations without cloning.

        :param subpop: The subpopulation id to migrate into.
        :type subpop: int
        :param winners: The destination subpopulation's winners.
        :type winners: list
        :param open_for_migration: The destination's open-for-migration bracket.
        :type open_for_migration: list
        :param external_pool: The pre-evolution snapshot migrant sources are drawn from.
        :type external_pool: list
        :return: A list of (open_agent, external, kind, elite) tuples, one per migration.
        :rtype: list[tuple]
        """
        elite = winners[0]
        # Ranked agents from other subpopulations
        external = self._rank(
            [a for a in external_pool if a.subpopulation_id != subpop]
        )

        decisions: list[MigrationDecision] = []
        external_counter = 0
        # Algorithm 2 of the paper: candidates and open slots are matched greedily
        for open_agent in open_for_migration:
            if external_counter >= len(external):
                break
            ext = external[external_counter]
            if self._scalar_fitness(open_agent.fitness[-1]) >= self._scalar_fitness(
                ext.fitness[-1]
            ):
                continue
            kind = (
                MultiFrequencyOp.MIGRATE_WEIGHTS
                if self.deltas[ext.subpopulation_id] < self.deltas[subpop]
                else MultiFrequencyOp.MIGRATE_FULL
            )
            decisions.append((open_agent, ext, kind, elite))
            external_counter += 1

        return decisions

    def _migrate_full_clone(
        self, external: EvolvableAlgorithm, subpop: int
    ) -> EvolvableAlgorithm:
        """Full clone of an external agent.

        :param external: The external agent migrating in.
        :type external: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param subpop: The destination subpopulation id.
        :type subpop: int
        :return: The migrant agent (independent of external).
        :rtype: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        """
        migrant = external.clone(index=self._next_index(), wrap=False)
        migrant.subpopulation_id = subpop
        return migrant

    def _migrate_weights(
        self, external: EvolvableAlgorithm, elite: EvolvableAlgorithm, subpop: int
    ) -> EvolvableAlgorithm:
        """Clone the external agent's networks but reset mutable HPs to the elite's.

        :param external: The external agent whose networks are imported.
        :type external: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param elite: The studied subpopulation's elite, whose HPs are adopted.
        :type elite: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param subpop: The destination subpopulation id.
        :type subpop: int
        :return: The migrant agent (independent of both parents).
        :rtype: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        """
        migrant = external.clone(index=self._next_index(), wrap=False)
        hp_values = {
            name: copy.deepcopy(getattr(elite, name))
            for name in elite.registry.hp_config
        }
        self._apply_hp_reset(migrant, hp_values)
        migrant.subpopulation_id = subpop
        return migrant

    def _apply_hp_reset(
        self, migrant: EvolvableAlgorithm, hp_values: dict[str, Any]
    ) -> None:
        """Reset a migrant's mutable hyperparameters, rebuilding any LR optimizer.

        :param migrant: The freshly-cloned migrant to reset in place.
        :type migrant: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param hp_values: The destination elite's mutable HP values, keyed by name.
        :type hp_values: dict[str, Any]
        """
        hp_config = migrant.registry.hp_config
        changed = set(hp_values)
        for name, value in hp_values.items():
            setattr(migrant, name, value)
            # Keep the RLParameter value in sync with the plain attribute, so a later
            # mutation of this migrant perturbs the elite's values instead of those
            # of the external agent
            if hp_config and name in hp_config.names():
                hp_config[name].value = value
        # Re-run the registered mutation hooks so HP-derived state is rebuilt for the
        # new values (e.g. PPO sizes its rollout buffer from learn_step via a hook)
        migrant.mutation_hook()
        # Rebuild every optimizer whose learning rate was reset to the elite's
        for opt_config in migrant.registry.optimizers:
            lr_attr = opt_config.lr
            lr_attr_names = lr_attr if isinstance(lr_attr, tuple) else (lr_attr,)
            if any(name in changed for name in lr_attr_names):
                migrant.reinit_optimizers(optimizer=opt_config)
