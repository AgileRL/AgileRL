"""Multiple-Frequencies Population-Based Training (MF-PBT).

Implements the MF-PBT evolution regime of Doulazmi et al.,
*"Multiple-Frequencies Population-Based Training"*, as a drop-in alternative to
AgileRL's tournament-selection + mutation evolution.

A population of n_subpopulations * n_individuals_per_subpopulation agents is
split into subpopulations, each evolving at its own frequency delta_i (every
delta_i evolution cycles). All agents still train evo_steps per cycle, so
the global evolution frequency, and therefore the granularity of the discovered
hyperparameter schedules, is preserved, while the slower subpopulations resist
the greediness that makes single-frequency PBT settle into local optima.

Each subpopulation is partitioned by within-subpopulation fitness rank into four
brackets (*winners*, *survivors*, *open-for-migration* and *losers*) that drive
exploitation (clone a winner over a loser, then perturb), preservation (survivors
and open agents are untouched by :meth:`~MultiFrequencyStrategy.evolution`) and the
asymmetric cross-frequency *migration* of Algorithm 2 of the paper.

The orchestration that schedules subpopulations by frequency and saves the global
elite lives in :func:`agilerl.utils.utils.multi_frequency_selection_and_mutation`;
this module holds only the operator. The single-process path is the only one
supported: an accelerator is rejected by the orchestrator with
:class:`NotImplementedError`.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agilerl.hpo.mutation import Mutations

PopulationType = list[Any]
NEG_INF = float("-inf")


class MultiFrequencyStrategy:
    """The MF-PBT operator.

    :param n_subpopulations: Number of subpopulations (>= 2; None -> 2).
    :type n_subpopulations: int | None
    :param n_individuals_per_subpopulation: Agents in each subpopulation (>= 3;
        None -> 8).
    :type n_individuals_per_subpopulation: int | None
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency ratios
        delta_i (strictly increasing integers, delta_i >= 1; one per
        subpopulation). None or [] -> [1, 5, 10, ...].
    :type evolution_frequency_ratios: list[int] | None
    :param n_winners: Agents in the winners bracket (>= 1; None ->
        round(0.25 * n_individuals_per_subpopulation)).
    :type n_winners: int | None
    :param n_survivors: Agents in the survivors bracket (>= 0; None -> 0).
    :type n_survivors: int | None
    :param n_open_for_migration: Agents in the open-for-migration bracket (>= 1,
        which in turn requires n_subpopulations >= 2; None ->
        round(0.25 * n_individuals_per_subpopulation)).
    :type n_open_for_migration: int | None
    :param n_losers: Agents in the losers bracket (>= 1; None -> the remainder
        n_ind - n_winners - n_survivors - n_open_for_migration).
    :type n_losers: int | None
    :param seed: Seed for the reproducible winner-clone selection in
        :meth:`evolution`, derived from the run's global seed. None leaves the
        RNG unseeded.
    :type seed: int | None
    :raises ValueError: If n_subpopulations < 2 (migration has nothing to draw
        from with a single subpopulation), n_individuals_per_subpopulation < 3,
        n_winners < 1, n_survivors < 0, n_open_for_migration < 1,
        n_losers < 1, the four brackets do not sum to
        n_individuals_per_subpopulation, or the frequency ratios are not
        n_subpopulations strictly-increasing integers >= 1.
    """

    def __init__(
        self,
        n_subpopulations: int | None = None,
        n_individuals_per_subpopulation: int | None = None,
        evolution_frequency_ratios: list[int] | None = None,
        n_winners: int | None = None,
        n_survivors: int | None = None,
        n_open_for_migration: int | None = None,
        n_losers: int | None = None,
        seed: int | None = None,
    ) -> None:
        (
            self.n_subpopulations,
            self.n_individuals_per_subpopulation,
            self.deltas,
            self.n_winners,
            self.n_survivors,
            self.n_open_for_migration,
            self.n_losers,
        ) = self._resolve_and_validate(
            n_subpopulations,
            n_individuals_per_subpopulation,
            evolution_frequency_ratios,
            n_winners,
            n_survivors,
            n_open_for_migration,
            n_losers,
        )

        self.pop_size = self.n_subpopulations * self.n_individuals_per_subpopulation
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
        n_subpopulations: int | None,
        n_individuals_per_subpopulation: int | None,
        evolution_frequency_ratios: list[int] | None,
        n_winners: int | None,
        n_survivors: int | None,
        n_open_for_migration: int | None,
        n_losers: int | None,
    ) -> tuple[int, int, list[int], int, int, int, int]:
        """Resolve the None defaults, then hard-check the operator's invariants.

        :param n_subpopulations: Number of subpopulations, or None for 2.
        :type n_subpopulations: int | None
        :param n_individuals_per_subpopulation: Agents per subpopulation, or None
            for 8.
        :type n_individuals_per_subpopulation: int | None
        :param evolution_frequency_ratios: Per-subpopulation frequency ratios, or
            None/[] for [1, 5, 10, ...].
        :type evolution_frequency_ratios: list[int] | None
        :param n_winners: Winners bracket size, or None for
            round(0.25 * n_individuals_per_subpopulation).
        :type n_winners: int | None
        :param n_survivors: Survivors bracket size, or None for 0.
        :type n_survivors: int | None
        :param n_open_for_migration: Open-for-migration bracket size, or None for
            round(0.25 * n_individuals_per_subpopulation).
        :type n_open_for_migration: int | None
        :param n_losers: Losers bracket size, or None for the remainder
            n_ind - n_winners - n_survivors - n_open_for_migration.
        :type n_losers: int | None
        :return: The resolved (n_subpopulations, n_individuals_per_subpopulation,
            evolution_frequency_ratios, n_winners, n_survivors, n_open_for_migration,
            n_losers).
        :rtype: tuple[int, int, list[int], int, int, int, int]
        :raises ValueError: On any violated invariant.
        """
        # Resolve defaults
        if n_subpopulations is None:
            n_subpopulations = 2
        if n_individuals_per_subpopulation is None:
            n_individuals_per_subpopulation = 8
        if n_survivors is None:
            n_survivors = 0
        n_ind = n_individuals_per_subpopulation
        if n_winners is None:
            n_winners = round(0.25 * n_ind)
        if n_open_for_migration is None:
            n_open_for_migration = round(0.25 * n_ind)
        if n_losers is None:
            n_losers = n_ind - n_winners - n_survivors - n_open_for_migration
        if not evolution_frequency_ratios:
            evolution_frequency_ratios = [1] + [
                5 * i for i in range(1, n_subpopulations)
            ]
        else:
            evolution_frequency_ratios = list(evolution_frequency_ratios)

        # Validate the MF-PBT parameters
        if n_subpopulations < 2:
            msg = f"n_subpopulations must be >= 2, got {n_subpopulations}."
            raise ValueError(msg)
        if n_ind < 3:
            msg = (
                "n_individuals_per_subpopulation must be >= 3 (one winner, one "
                f"open-for-migration and one loser slot are each required), got {n_ind}."
            )
            raise ValueError(msg)
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
        if bracket_sum != n_ind:
            msg = (
                f"n_winners + n_survivors + n_open_for_migration + n_losers "
                f"({bracket_sum}) must equal n_individuals_per_subpopulation ({n_ind})."
            )
            raise ValueError(msg)
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
            n_subpopulations,
            n_individuals_per_subpopulation,
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

    def _rank(self, agents: PopulationType) -> PopulationType:
        """Return agent sorted by scalar fitness, highest first.

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

    def _sync_index(self, population: PopulationType) -> None:
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
    def subpopulation_for_index(
        index: int, n_individuals_per_subpopulation: int
    ) -> int:
        """Map an agent index to its subpopulation id.

        Both the build-time tagging in
        :func:`agilerl.utils.trainer_utils._assign_subpopulations` and the defensive
        tagging in :meth:`_assign_initial_subpopulations` route through this method so
        the layout can only ever change in one place.

        :param index: The agent's population index.
        :type index: int
        :param n_individuals_per_subpopulation: Agents per subpopulation.
        :type n_individuals_per_subpopulation: int
        :return: The subpopulation id the agent belongs to.
        :rtype: int
        """
        return index // n_individuals_per_subpopulation

    def _assign_initial_subpopulations(self, population: PopulationType) -> None:
        """Assign subpopulation to any agent that lacks it.

        Makes the operator robust when driven through the functional trainers with
        a population that was not tagged at build time.

        :param population: The whole population.
        :type population: list
        :raises ValueError: If len(population) does not equal pop_size.
        """
        if len(population) != self.pop_size:
            msg = (
                f"Population has {len(population)} agents, expected {self.pop_size} "
                f"(n_subpopulations * n_individuals_per_subpopulation = "
                f"{self.n_subpopulations} * {self.n_individuals_per_subpopulation})."
            )
            raise ValueError(msg)
        for agent in population:
            if getattr(agent, "subpopulation", None) is None:
                agent.subpopulation = self.subpopulation_for_index(
                    agent.index, self.n_individuals_per_subpopulation
                )

    def brackets(
        self, population: PopulationType, subpop: int
    ) -> tuple[PopulationType, PopulationType, PopulationType, PopulationType]:
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
        members = self._rank([a for a in population if a.subpopulation == subpop])
        if len(members) != self.n_individuals_per_subpopulation:
            msg = (
                f"Subpopulation {subpop} has {len(members)} members, expected "
                f"{self.n_individuals_per_subpopulation}."
            )
            raise ValueError(msg)
        n_w, n_s, n_o, _n_l = self.bracket_sizes
        winners = members[:n_w]
        survivors = members[n_w : n_w + n_s]
        open_for_migration = members[n_w + n_s : n_w + n_s + n_o]
        losers = members[n_w + n_s + n_o :]
        return winners, survivors, open_for_migration, losers

    def evolution(
        self, population: PopulationType, subpop: int, mutation: Mutations
    ) -> PopulationType:
        """Replace a subpopulation's losers with perturbed winner-clones.

        :param population: The whole population.
        :type population: list
        :param subpop: The subpopulation id to evolve.
        :type subpop: int
        :param mutation: The mutation operator used to perturb the winner-clones.
        :type mutation: ~agilerl.hpo.mutation.Mutations
        :return: A new population list (the caller's list is not mutated).
        :rtype: list
        """
        self._sync_index(population)
        winners, _survivors, _open, losers = self.brackets(population, subpop)

        loser_ids = {id(a) for a in losers}
        retained = [a for a in population if id(a) not in loser_ids]

        clones: PopulationType = []
        for _ in losers:
            winner = winners[int(self.rng.integers(len(winners)))]
            clone = winner.clone(index=self._next_index(), wrap=False)
            clone.subpopulation = subpop
            # The fitness of the loser replacement is set to -inf
            clone.fitness = [NEG_INF]
            clones.append(clone)

        # Perturb every introduced clone, forcing mutate_elite so none is skipped
        prev_mutate_elite = mutation.mutate_elite
        mutation.mutate_elite = True
        try:
            clones = mutation.mutation(clones)
        finally:
            mutation.mutate_elite = prev_mutate_elite

        return retained + clones

    def migration(
        self,
        population: PopulationType,
        subpop: int,
        external_pool: PopulationType,
    ) -> PopulationType:
        """Asymmetrically migrate stronger agents into the open-for-migration slots.

        Implements Algorithm 2 of the paper. For each open-for-migration agent (best
        first), it is compared with the next-best agent drawn from the *other*
        subpopulations of external_pool. If the open agent is at least as good, no
        migration happens (and the external pointer does not advance). Otherwise the
        external agent migrates in: if it comes from a faster-evolving subpopulation
        (smaller delta) only its networks/optimizer are imported while its mutable
        hyperparameters are reset to the studied subpopulation's elite (the optimizer
        learning rate is updated in place to the elite value, preserving its Adam
        moments); otherwise it is cloned in full.

        :param population: The live population, used for bracketing and substitution.
        :type population: list
        :param subpop: The subpopulation id to migrate into.
        :type subpop: int
        :param external_pool: The pre-evolution population snapshot that migrant
            sources are drawn from.
        :type external_pool: list
        :return: A new population list with migrants substituted in place.
        :rtype: list
        """
        self._sync_index(population)
        winners, _survivors, open_for_migration, _losers = self.brackets(
            population, subpop
        )
        elite = winners[0]
        # Ranked agents from other subpopulations
        external = self._rank([a for a in external_pool if a.subpopulation != subpop])

        replacements: dict[int, Any] = {}
        external_counter = 0
        for open_agent in open_for_migration:
            if external_counter >= len(external):
                break
            ext = external[external_counter]
            if self._scalar_fitness(open_agent.fitness[-1]) >= self._scalar_fitness(
                ext.fitness[-1]
            ):
                continue
            if self.deltas[ext.subpopulation] < self.deltas[subpop]:
                migrant = self._migrate_weights(ext, elite, subpop)
            else:
                migrant = self._migrate_full_clone(ext, subpop)
            replacements[id(open_agent)] = migrant
            external_counter += 1

        return [replacements.get(id(a), a) for a in population]

    def _migrate_full_clone(self, external: Any, subpop: int) -> Any:
        """Full clone of an external agent.

        :param external: The external agent migrating in.
        :type external: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param subpop: The destination subpopulation id.
        :type subpop: int
        :return: The migrant agent (independent of external).
        :rtype: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        """
        migrant = external.clone(index=self._next_index(), wrap=False)
        migrant.subpopulation = subpop
        # The migrant's fitness is set to -inf
        migrant.fitness = [NEG_INF]
        return migrant

    def _migrate_weights(self, external: Any, elite: Any, subpop: int) -> Any:
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
        hp_config = migrant.registry.hp_config
        changed = []
        for name in elite.registry.hp_config:
            new_value = copy.deepcopy(getattr(elite, name))
            setattr(migrant, name, new_value)
            # Keep the RLParameter value in sync with the plain attribute, so a later
            # mutation of this migrant perturbs the elite's values instead of those
            # of the external agent
            if hp_config and name in hp_config.names():
                hp_config[name].value = new_value
            changed.append(name)
        # Re-run the registered mutation hooks so HP-derived state is rebuilt for the
        # new values (e.g. PPO sizes its rollout buffer from learn_step via a hook)
        migrant.mutation_hook()
        # The networks were cloned whole from external (architecture unchanged),
        # so the optimizers do not need rebuilding. Therefore, the LR is updated and
        # the Adam moments are preserved
        changed_set = set(changed)
        for opt_config in migrant.registry.optimizers:
            lr_attr = opt_config.lr
            lr_attr_names = lr_attr if isinstance(lr_attr, tuple) else (lr_attr,)
            if any(name in changed_set for name in lr_attr_names):
                self._set_optimizer_lr(migrant, opt_config)
        migrant.subpopulation = subpop
        migrant.fitness = [NEG_INF]
        return migrant

    @staticmethod
    def _set_optimizer_lr(agent: Any, opt_config: Any) -> None:
        """Update an optimizer's learning rate in place, preserving its state.

        :param agent: The migrant agent whose optimizer is updated.
        :type agent: ~agilerl.algorithms.core.base.EvolvableAlgorithm
        :param opt_config: The optimizer configuration to update.
        :type opt_config: ~agilerl.algorithms.core.optimizer_wrapper.OptimizerConfig
        """
        if isinstance(opt_config.lr, tuple):
            agent.reinit_optimizers(optimizer=opt_config)
            return

        new_lr = getattr(agent, opt_config.lr)
        wrapper = getattr(agent, opt_config.name)
        wrapper.lr = new_lr
        optimizers = (
            wrapper.optimizer.values()
            if isinstance(wrapper.optimizer, dict)
            else [wrapper.optimizer]
        )
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = new_lr
