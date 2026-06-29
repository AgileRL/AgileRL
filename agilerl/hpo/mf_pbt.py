"""Multiple-Frequencies Population-Based Training (MF-PBT).

Implements the MF-PBT evolution regime of Doulazmi et al.,
*"Multiple-Frequencies Population-Based Training"*, as a drop-in alternative to
AgileRL's tournament-selection + mutation evolution.

A population of ``n_subpopulations * n_individuals_per_subpopulation`` agents is
split into subpopulations, each evolving at its own frequency ``delta_i`` (every
``delta_i`` evolution cycles). All agents still train ``evo_steps`` per cycle, so
the global evolution frequency — and therefore the granularity of the discovered
hyperparameter schedules — is preserved, while the slower subpopulations resist
the greediness that makes single-frequency PBT settle into local optima.

Each subpopulation is partitioned by within-subpopulation fitness rank into four
brackets — *winners*, *survivors*, *open-for-migration* and *losers* — that drive
exploitation (clone a winner over a loser, then perturb), preservation (survivors
and open agents are untouched by ``evolution``) and the asymmetric cross-frequency
*migration* of Algorithm 2 of the paper.

The single-process path is implemented here; an ``accelerator`` is rejected with
:class:`NotImplementedError` (none of the benchmark suites use Accelerate).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agilerl.hpo.mutation import Mutations

PopulationT = list[Any]
NEG_INF = float("-inf")


class MFPBT:
    """The MF-PBT evolution operator.

    A single long-lived instance is built once and reused across evolution cycles
    so that the per-subpopulation frequency counters and the global index
    allocator persist.

    :param n_subpopulations: Number of subpopulations.
    :type n_subpopulations: int
    :param n_individuals_per_subpopulation: Agents in each subpopulation.
    :type n_individuals_per_subpopulation: int
    :param evolution_frequency_ratios: Per-subpopulation evolution-frequency
        ratios ``delta_i`` (strictly increasing integers, ``delta_i >= 1``).
    :type evolution_frequency_ratios: list[int]
    :param n_winners: Agents in the winners bracket.
    :type n_winners: int
    :param n_survivors: Agents in the survivors bracket.
    :type n_survivors: int
    :param n_open_for_migration: Agents in the open-for-migration bracket.
    :type n_open_for_migration: int
    :param n_losers: Agents in the losers bracket.
    :type n_losers: int
    :param rand_seed: Seed for the (reproducible) winner-clone selection.
    :type rand_seed: int
    """

    def __init__(
        self,
        n_subpopulations: int,
        n_individuals_per_subpopulation: int,
        evolution_frequency_ratios: list[int],
        n_winners: int,
        n_survivors: int,
        n_open_for_migration: int,
        n_losers: int,
        rand_seed: int = 42,
    ) -> None:
        self.n_subpopulations = n_subpopulations
        self.n_individuals_per_subpopulation = n_individuals_per_subpopulation
        self.pop_size = n_subpopulations * n_individuals_per_subpopulation
        self.deltas = list(evolution_frequency_ratios)
        self.n_winners = n_winners
        self.n_survivors = n_survivors
        self.n_open_for_migration = n_open_for_migration
        self.n_losers = n_losers
        self.bracket_sizes = (n_winners, n_survivors, n_open_for_migration, n_losers)

        # Per-subpopulation evolution counters (persist across cycles).
        self.counters = [0] * n_subpopulations
        # Reproducible RNG for the winner-clone selection in ``evolution``.
        self.rng = np.random.default_rng(rand_seed)
        # Monotonic high-water mark for fresh agent indices.
        self._max_index: int | None = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _scalar_fitness(fitness: float | np.ndarray | dict[str, float]) -> float:
        """Reduce a possibly vector-valued fitness to a single scalar for ranking.

        Mirrors :meth:`TournamentSelection._scalar_fitness`: multi-agent
        algorithms (e.g. IPPO with ``sum_scores=False``) store per-sub-agent
        fitness values, which we collapse to their mean.
        """
        if isinstance(fitness, dict):
            return float(np.mean(list(fitness.values())))
        if isinstance(fitness, (list, tuple, np.ndarray)):
            return float(np.mean(fitness))
        return float(fitness)

    def _rank(self, agents: PopulationT) -> PopulationT:
        """Return *agents* sorted by scalar fitness, highest first (stable)."""
        return sorted(
            agents,
            key=lambda a: self._scalar_fitness(a.fitness[-1]),
            reverse=True,
        )

    def _sync_index(self, population: PopulationT) -> None:
        """Seed the index allocator from the current population's max index."""
        current = max(a.index for a in population)
        self._max_index = (
            current if self._max_index is None else max(self._max_index, current)
        )

    def _next_index(self) -> int:
        """Return a fresh, globally-unique agent index."""
        self._max_index += 1
        return self._max_index

    def _assign_initial_subpopulations(self, population: PopulationT) -> None:
        """Assign ``subpopulation`` to any agent that lacks it (defensive)."""
        for agent in population:
            if getattr(agent, "subpopulation", None) is None:
                agent.subpopulation = (
                    agent.index // self.n_individuals_per_subpopulation
                )

    # ------------------------------------------------------------------ #
    # Bracketing
    # ------------------------------------------------------------------ #
    def brackets(
        self, population: PopulationT, subpop: int
    ) -> tuple[PopulationT, PopulationT, PopulationT, PopulationT]:
        """Partition a subpopulation's members into the four ranked brackets.

        Members are ranked by descending fitness (equivalently, their order in a
        global ranking) and sliced into ``(winners, survivors, open, losers)``.

        :param population: The whole population.
        :type population: list
        :param subpop: The subpopulation id to bracket.
        :type subpop: int
        :return: ``(winners, survivors, open_for_migration, losers)``.
        :rtype: tuple[list, list, list, list]
        """
        members = self._rank([a for a in population if a.subpopulation == subpop])
        assert len(members) == self.n_individuals_per_subpopulation, (
            f"Subpopulation {subpop} has {len(members)} members, expected "
            f"{self.n_individuals_per_subpopulation}."
        )
        n_w, n_s, n_o, _n_l = self.bracket_sizes
        winners = members[:n_w]
        survivors = members[n_w : n_w + n_s]
        open_for_migration = members[n_w + n_s : n_w + n_s + n_o]
        losers = members[n_w + n_s + n_o :]
        return winners, survivors, open_for_migration, losers

    # ------------------------------------------------------------------ #
    # Evolution (exploit + explore)
    # ------------------------------------------------------------------ #
    def evolution(
        self, population: PopulationT, subpop: int, mutation: Mutations
    ) -> PopulationT:
        """Replace a subpopulation's losers with perturbed winner-clones.

        The loser agents are removed; for each, a randomly (reproducibly) chosen
        winner is cloned, assigned to *subpop* with fitness ``-inf`` and a fresh
        index, and **all** introduced clones are perturbed by *mutation* (the
        ``mutate_elite`` flag is forced on so no clone is skipped). The winners,
        survivors and open-for-migration agents are untouched, and the total
        population and per-subpopulation sizes are preserved.

        :return: A new population list (the caller's list is not mutated).
        :rtype: list
        """
        self._sync_index(population)
        winners, _survivors, _open, losers = self.brackets(population, subpop)

        loser_ids = {id(a) for a in losers}
        retained = [a for a in population if id(a) not in loser_ids]

        clones: PopulationT = []
        for _ in losers:
            winner = winners[int(self.rng.integers(len(winners)))]
            clone = winner.clone(index=self._next_index(), wrap=False)
            clone.subpopulation = subpop
            clone.fitness = [NEG_INF]
            clones.append(clone)

        # Perturb every introduced clone, forcing mutate_elite so none is skipped.
        # Use the returned list (some mutations may swap the agent object).
        if clones:
            prev_mutate_elite = mutation.mutate_elite
            mutation.mutate_elite = True
            try:
                clones = mutation.mutation(clones)
            finally:
                mutation.mutate_elite = prev_mutate_elite

        return retained + clones

    # ------------------------------------------------------------------ #
    # Migration (asymmetric, Algorithm 2)
    # ------------------------------------------------------------------ #
    def migration(self, population: PopulationT, subpop: int) -> PopulationT:
        """Asymmetrically migrate stronger agents into the open-for-migration slots.

        For each open-for-migration agent (best first), it is compared with the
        next-best agent from the *other* subpopulations. If the open agent is at
        least as good, no migration happens (and the external pointer does not
        advance). Otherwise the external agent migrates in: if it comes from a
        *faster*-evolving subpopulation (smaller ``delta``) only its
        networks/optimizer are imported while its mutable hyperparameters are
        reset to the studied subpopulation's elite (the optimizer learning rate is
        updated in place to the elite value, preserving its Adam moments); otherwise
        it is cloned in full.

        :return: A new population list with migrants substituted in place.
        :rtype: list
        """
        self._sync_index(population)
        winners, _survivors, open_for_migration, _losers = self.brackets(
            population, subpop
        )
        if not open_for_migration:  # nothing to migrate into (empty bracket)
            return population
        elite = winners[0]
        external = self._rank([a for a in population if a.subpopulation != subpop])

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
                migrant = self._migrate_reset_hp(ext, elite, subpop)
            else:
                migrant = self._migrate_full_clone(ext, subpop)
            replacements[id(open_agent)] = migrant
            external_counter += 1

        return [replacements.get(id(a), a) for a in population]

    def _migrate_full_clone(self, external: Any, subpop: int) -> Any:
        """Full clone of *external* (weights, optimizer, activations, HPs)."""
        migrant = external.clone(index=self._next_index(), wrap=False)
        migrant.subpopulation = subpop
        migrant.fitness = [NEG_INF]
        return migrant

    def _migrate_reset_hp(self, external: Any, elite: Any, subpop: int) -> Any:
        """Clone *external*'s networks but reset mutable HPs to *elite*'s values."""
        migrant = external.clone(index=self._next_index(), wrap=False)
        hp_config = migrant.registry.hp_config
        changed = []
        for name in elite.registry.hp_config:
            new_value = copy.deepcopy(getattr(elite, name))
            setattr(migrant, name, new_value)
            # Keep the RLParameter value in sync with the plain attribute. clone()
            # deepcopies hp_config (so the migrant carries *external*'s last-mutated
            # value), and rl_hyperparam_mutation only re-derives that value from the
            # attribute when it is None -- so without this a later mutation of this
            # migrant's lineage would perturb from external's stale value, not the
            # reset elite value. Mirrors the crossover operator.
            if hp_config and name in hp_config.names():
                hp_config[name].value = new_value
            changed.append(name)
        # Re-run the registered mutation hooks so HP-derived state is rebuilt for the
        # new values (e.g. PPO sizes its rollout buffer from learn_step via a hook);
        # this mirrors what Mutations.mutation does after a hyperparameter mutation.
        migrant.mutation_hook()
        # The networks were cloned whole from *external* (architecture unchanged),
        # so the optimizers never need rebuilding -- only the step size changes when
        # the elite's learning rate differs. Update it in place to preserve the Adam
        # moment estimates (PBT-faithful exploit), mirroring the crossover operator;
        # reinit_optimizers would instead discard that state.
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

        The networks are unchanged by migration (cloned whole from the external
        agent), so only the step size needs updating; the Adam moment estimates
        carried over by the clone are kept intact. Mirrors the crossover operator's
        ``_set_optimizer_lr`` so the two HPO improvements treat optimizer state
        identically.

        :param agent: The migrant agent whose optimizer is updated.
        :type agent: EvolvableAlgorithm
        :param opt_config: The optimizer configuration to update.
        :type opt_config: OptimizerConfig
        """
        # Tuple LR names only occur for split LLM optimizers, which the benchmark
        # suites (PPO/DQN/IPPO) never use; fall back to a rebuild in that case.
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

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def evolve_population(
        self,
        population: PopulationT,
        *,
        mutation: Mutations,
        env_name: str = "env",
        algo: str | None = None,
        save_elite: bool = False,
        elite_path: str | None = None,
        accelerator: Any | None = None,
    ) -> PopulationT:
        """Run one MF-PBT evolution cycle over the whole population.

        The global elite is saved (if requested) before any evolution; then each
        subpopulation whose frequency counter has reached its ``delta_i`` is
        evolved (exploit/explore) and migrated.

        :return: The evolved population (same length, same per-subpopulation counts).
        :rtype: list
        """
        if accelerator is not None:
            msg = "MF-PBT does not support the Accelerate (multi-process) path."
            raise NotImplementedError(msg)

        self._assign_initial_subpopulations(population)
        self._sync_index(population)

        if save_elite:
            elite_agent = max(
                population, key=lambda a: self._scalar_fitness(a.fitness[-1])
            )
            algo_name = algo or population[0].__class__.__name__
            elite_save_path = (
                elite_path.split(".pt")[0]
                if elite_path is not None
                else f"{env_name}-elite_{algo_name}"
            )
            elite_agent.save_checkpoint(f"{elite_save_path}.pt")

        for i in range(self.n_subpopulations):
            self.counters[i] += 1
            if self.counters[i] < self.deltas[i]:
                continue
            self.counters[i] = 0
            population = self.evolution(population, i, mutation)
            population = self.migration(population, i)

        return population
