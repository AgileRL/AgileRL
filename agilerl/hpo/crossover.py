from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from agilerl.algorithms.core.base import EvolvableAlgorithm
from agilerl.hpo.tournament import TournamentSelection

if TYPE_CHECKING:
    from agilerl.algorithms.core.registry import OptimizerConfig

PopulationT = list[EvolvableAlgorithm]

# Sentinel marking the single, inseparable "gene" bundling an agent's
# architecture, weights and activations together. Unlike the hyperparameters,
# these characteristics are tightly coupled to the learned policy and cannot be
# recombined piece-by-piece without destroying the learned patterns, so they
# always travel together as one chromosome element placed last.
BUNDLE = "__bundle__"


class Crossover:
    """Multi-point recombination operator for evolutionary HPO, an alternative to
    :class:`TournamentSelection <agilerl.hpo.tournament.TournamentSelection>`.

    Calling :func:`Crossover.crossover` on a population returns the elite agent and
    a new, recombined population of the same size. Each agent is encoded as a
    *chromosome*: a (reproducibly shuffled) list of its mutable RL-hyperparameter
    genes followed by a single :data:`BUNDLE` gene holding its
    architecture/weights/activations. Two parents drawn from the top-fitness pool
    are split at ``number_of_crossover_points`` crossover points into
    ``number_of_crossover_points + 1`` sections, and each section is independently
    swapped between the two chromosomes with probability ``swap_prob``. The
    recombined chromosomes are assembled back into two offspring.

    The new evolution procedure is: evaluate fitness → ``crossover`` the population
    → mutate the population (unchanged). Crossover is *not* applied to the single
    pre-training mutation step.

    :param num_parents: Number of top agents (by fitness) that form the
        recombination pool. The elite is always included. The paper uses ~80% of
        the population (e.g. 13 for a population of 16).
    :type num_parents: int
    :param swap_prob: Per-section probability of exchanging a chromosome section
        between the two parents during recombination.
    :type swap_prob: float
    :param elitism: If True, the top ``number_of_elites`` agents (by fitness) are
        cloned unchanged into the next generation.
    :type elitism: bool
    :param population_size: Number of agents in the population.
    :type population_size: int
    :param number_of_elites: Number of highest-fitness agents preserved exactly
        (cloned unchanged) into the next generation when ``elitism`` is True. The
        elite *returned* by :func:`crossover` is always a clone of the single best
        agent regardless of this value. Defaults to 1.
    :type number_of_elites: int
    :param number_of_crossover_points: Number of crossover points used to split
        each chromosome into swappable sections. ``k`` points yield ``k + 1``
        sections, so larger values recombine the parents in finer pieces. Must be
        at least 1; its maximum (validated in :func:`crossover`, where the
        chromosome length is known) is the chromosome length minus one — the value
        that makes every section a single gene (the :data:`BUNDLE` counts as one
        gene). Defaults to 2 (two-point crossover).
    :type number_of_crossover_points: int
    :param rand_seed: Random seed for reproducible recombination.
    :type rand_seed: int
    """

    def __init__(
        self,
        num_parents: int,
        swap_prob: float,
        elitism: bool,
        population_size: int,
        number_of_elites: int = 1,
        number_of_crossover_points: int = 2,
        rand_seed: int = 42,
    ) -> None:
        assert population_size > 0, "Population size must be greater than zero."
        assert num_parents >= 2, "Number of parents must be at least two."
        assert num_parents <= population_size, (
            "Number of parents cannot exceed the population size."
        )
        assert 0.0 <= swap_prob <= 1.0, "Swap probability must be in [0, 1]."
        assert isinstance(elitism, bool), "Elitism must be boolean value True or False."
        assert number_of_elites >= 1, "Number of elites must be at least one."
        assert number_of_elites <= population_size, (
            "Number of elites cannot exceed the population size."
        )
        assert number_of_crossover_points >= 1, (
            "Number of crossover points must be at least one."
        )

        self.num_parents = num_parents
        self.swap_prob = swap_prob
        self.elitism = elitism
        self.population_size = population_size
        self.number_of_elites = number_of_elites
        self.number_of_crossover_points = number_of_crossover_points
        # A single, never-reseeded Generator: determinism comes from constructing
        # it once and consuming the stream across calls (mirrors Mutations'
        # rand_seed, and is deliberately isolated from the global np.random that
        # tournament selection uses).
        self.rng = np.random.default_rng(rand_seed)

    @staticmethod
    def _scalar_fitness(fitness: object) -> float:
        """Reduce a possibly vector-valued fitness to a single scalar for ranking.

        Reuses :func:`TournamentSelection._scalar_fitness` so ranking is identical
        to tournament selection (mean across sub-agents for multi-agent fitness).
        """
        return TournamentSelection._scalar_fitness(fitness)

    def crossover(
        self,
        population: PopulationT,
    ) -> tuple[EvolvableAlgorithm, PopulationT]:
        """Return the elite agent and a new, recombined population of agents.

        :param population: Population of agents.
        :type population: PopulationT
        :return: Elite agent and new population.
        :rtype: tuple[EvolvableAlgorithm, PopulationT]
        """
        # The chromosome is the agent's mutable HP genes plus the single BUNDLE
        # gene; every agent shares the same HP names (same algorithm), so its
        # length is constant across the population. With ``k`` crossover points the
        # chromosome splits into ``k + 1`` sections, so all sections can be a single
        # gene only when ``k + 1 <= len(chromosome)``; beyond that the request is
        # unsatisfiable. Validated here (not in __init__) because the chromosome
        # length is unknown until a population is available.
        chromosome_len = len(population[0].registry.hp_config.names()) + 1  # + BUNDLE
        max_points = chromosome_len - 1
        if self.number_of_crossover_points > max_points:
            msg = (
                f"number_of_crossover_points ({self.number_of_crossover_points}) "
                f"cannot exceed {max_points} (the chromosome length "
                f"{chromosome_len} minus one), the value at which every swappable "
                f"section is a single gene."
            )
            raise ValueError(msg)

        fitnesses = [self._scalar_fitness(indi.fitness[-1]) for indi in population]
        max_id = max(ind.index for ind in population)

        # Rank agents by descending scalar fitness.
        order = list(np.argsort(fitnesses)[::-1])
        elite = population[int(order[0])]

        # Recombination pool: the top-`num_parents` agents (including the elite).
        pool = [population[int(i)] for i in order[: self.num_parents]]

        # A separate clone is handed back for saving so the returned elite is never
        # affected by the subsequent mutation pass (matching tournament selection),
        # even when mutate_elite is True.
        returned_elite = elite.clone(wrap=False)

        new_population: PopulationT = []
        if self.elitism:
            # Carry the top-`number_of_elites` agents over unchanged; each elite's
            # parent is itself. They occupy the front of the population so the
            # subsequent mutation pass can protect exactly them (see
            # Mutations.mutation's num_elites).
            for rank_idx in range(self.number_of_elites):
                parent = population[int(order[rank_idx])]
                elite_clone = parent.clone(wrap=False)
                elite_clone._parent_index = parent.index
                new_population.append(elite_clone)

        # Fill the remaining spots with recombination offspring.
        while len(new_population) < self.population_size:
            remaining = self.population_size - len(new_population)
            if len(pool) >= 2:
                idx_a, idx_b = self.rng.choice(len(pool), size=2, replace=False)
            else:
                idx_a = idx_b = 0
            parent_a, parent_b = pool[int(idx_a)], pool[int(idx_b)]
            offspring = self._recombine(
                parent_a, parent_b, max_id, single=(remaining == 1)
            )
            max_id += len(offspring)
            new_population.extend(offspring)

        return returned_elite, new_population

    def _recombine(
        self,
        parent_a: EvolvableAlgorithm,
        parent_b: EvolvableAlgorithm,
        base_id: int,
        single: bool = False,
    ) -> PopulationT:
        """Apply two-point crossover to two parents, returning their offspring.

        :param parent_a: First parent agent.
        :type parent_a: EvolvableAlgorithm
        :param parent_b: Second parent agent.
        :type parent_b: EvolvableAlgorithm
        :param base_id: Highest index used so far; offspring get ``base_id + 1`` (and
            ``base_id + 2``).
        :type base_id: int
        :param single: If True, only one offspring (chosen reproducibly) is returned
            so the population size stays fixed on an odd final slot.
        :type single: bool
        :return: One or two offspring agents.
        :rtype: PopulationT
        """
        # Build the gene list shared by both parents: shuffled HP names then the
        # inseparable BUNDLE gene, always last.
        hp_names = list(parent_a.registry.hp_config.names())
        self.rng.shuffle(hp_names)
        chromosome = [*hp_names, BUNDLE]
        n = len(chromosome)

        # ``number_of_crossover_points`` cut points split each chromosome into that
        # many sections plus one. (With the default of 2 this reproduces the
        # original two-point, three-section behaviour, including the RNG draw.)
        cuts = sorted(
            int(c)
            for c in self.rng.integers(0, n + 1, size=self.number_of_crossover_points)
        )
        bounds = [0, *cuts, n]
        sections = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

        # For each section, independently decide whether to swap it. Offspring 1
        # defaults to parent_a's genes (swapped sections take parent_b's);
        # offspring 2 is the complement.
        gene_source_1: dict[str, EvolvableAlgorithm] = {}
        gene_source_2: dict[str, EvolvableAlgorithm] = {}
        for (start, end), do_swap in zip(
            sections,
            [self.rng.random() < self.swap_prob for _ in sections],
            strict=True,
        ):
            for gene in chromosome[start:end]:
                gene_source_1[gene] = parent_b if do_swap else parent_a
                gene_source_2[gene] = parent_a if do_swap else parent_b

        if single:
            chosen = gene_source_1 if self.rng.random() < 0.5 else gene_source_2
            return [self._assemble_offspring(chosen, base_id + 1)]

        return [
            self._assemble_offspring(gene_source_1, base_id + 1),
            self._assemble_offspring(gene_source_2, base_id + 2),
        ]

    def _assemble_offspring(
        self,
        gene_source: dict[str, EvolvableAlgorithm],
        new_index: int,
    ) -> EvolvableAlgorithm:
        """Build an offspring agent from a recombined chromosome.

        The offspring is cloned from whichever parent contributed the
        :data:`BUNDLE` gene (so it inherits that parent's architecture, weights and
        activations), then the HP genes that came from the *other* parent are set on
        top, re-initializing any optimizer whose learning rate changed.

        :param gene_source: Mapping from gene name to the parent that contributed it.
        :type gene_source: dict[str, EvolvableAlgorithm]
        :param new_index: Index to assign to the offspring.
        :type new_index: int
        :return: The assembled offspring agent.
        :rtype: EvolvableAlgorithm
        """
        bundle_parent = gene_source[BUNDLE]
        child = bundle_parent.clone(new_index, wrap=False)
        child._parent_index = bundle_parent.index

        hp_config = child.registry.hp_config
        lr_names = child.get_lr_names()
        changed_lrs: set[str] = set()
        hp_from_other: list[str] = []
        for gene, source in gene_source.items():
            if gene == BUNDLE or source is bundle_parent:
                continue
            value = getattr(source, gene)
            setattr(child, gene, value)
            hp_from_other.append(gene)
            # Keep the RLParameter value in sync with the plain attribute so a
            # subsequent mutation that samples this gene mutates from the
            # recombined value, not the bundle parent's stale one.
            if hp_config and gene in hp_config.names():
                hp_config[gene].value = value
            if gene in lr_names:
                changed_lrs.add(gene)

        # Apply any swapped-in learning rate to the existing optimizer in place.
        # Crossover never changes the architecture (the network bundle is inherited
        # whole, with its matching optimizer state, from the bundle parent), so the
        # optimizer never needs rebuilding -- only its step size changes. Updating
        # the LR in place preserves the Adam moment estimates, which is what makes
        # PBT-style exploit (inherit weights + optimizer) / explore (new LR) work.
        # Rebuilding via reinit_optimizers would discard that state, and because
        # crossover re-derives most of the population every generation it would wipe
        # optimizer state across the whole population each generation.
        if changed_lrs:
            for opt_config in child.registry.optimizers:
                lr_attr = opt_config.lr
                lr_attr_names = lr_attr if isinstance(lr_attr, tuple) else (lr_attr,)
                if any(name in changed_lrs for name in lr_attr_names):
                    self._set_optimizer_lr(child, opt_config)

        # Recorded for logging; the immediately-following mutation pass may
        # overwrite these for mutated agents.
        child.mut = "crossover"
        child.mut_details = {
            "category": "crossover",
            "bundle_parent": bundle_parent.index,
            "hp_from_other": hp_from_other,
        }
        return child

    @staticmethod
    def _set_optimizer_lr(
        agent: EvolvableAlgorithm,
        opt_config: OptimizerConfig,
    ) -> None:
        """Update an optimizer's learning rate in place, preserving its state.

        The network the optimizer acts on is unchanged by crossover, so only the
        step size needs updating; the Adam moment estimates carried over from the
        bundle parent are kept intact.

        :param agent: The offspring agent whose optimizer is updated.
        :type agent: EvolvableAlgorithm
        :param opt_config: The optimizer configuration to update.
        :type opt_config: OptimizerConfig
        """
        # Tuple LR names only occur for split LLM optimizers, which crossover never
        # targets; fall back to a rebuild in that (unreachable) case.
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
