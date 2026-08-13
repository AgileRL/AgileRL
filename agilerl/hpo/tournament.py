from __future__ import annotations

import numpy as np
from accelerate.utils import broadcast_object_list

from agilerl.algorithms.core.base import EvolvableAlgorithm, LLMAlgorithm
from agilerl.wrappers.agent import AgentWrapper

PopulationT = list[EvolvableAlgorithm]


def _record_parent_index(
    individual: EvolvableAlgorithm | AgentWrapper, parent_index: int
) -> None:
    """Tag a freshly cloned agent with the population index it was cloned from.

    The tag is written on the *unwrapped* algorithm, never on the wrapper.
    :meth:`AgentWrapper.__setattr__ <agilerl.wrappers.agent.AgentWrapper.__setattr__>`
    forwards an assignment to the agent it wraps only when the agent already carries
    that attribute, which a fresh clone does not -- so assigning to the wrapper would
    leave the tag on the wrapper alone. That is invisible to
    :meth:`Mutations.parameter_mutation <agilerl.hpo.mutation.Mutations.parameter_mutation>`,
    which reads it off the unwrapped agent, and the ReBorn parameter mutation would
    then silently degrade to the Gaussian operator for every wrapped agent. Reads
    through a wrapper still resolve, via ``AgentWrapper.__getattr__``.

    :param individual: The freshly cloned population member, wrapped or not.
    :type individual: EvolvableAlgorithm | AgentWrapper
    :param parent_index: ``index`` of the agent *individual* was cloned from.
    :type parent_index: int
    """
    agent = individual.agent if isinstance(individual, AgentWrapper) else individual
    agent._parent_index = parent_index


class TournamentSelection:
    """The tournament selection class. Calling :func:`TournamentSelection.select() <agilerl.hpo.tournament.TournamentSelection.select>`
    on a population of agents will return a cloned population containing the best performing agent as well as the new generation of agents
    based on their fitness scores.

    :param tournament_size: Tournament selection size
    :type tournament_size: int
    :param elitism: Elitism in tournament selection
    :type elitism: bool
    :param population_size: Number of agents in population
    :type population_size: int
    """

    def __init__(
        self,
        tournament_size: int,
        elitism: bool,
        population_size: int,
    ) -> None:
        assert tournament_size > 0, "Tournament size must be greater than zero."
        assert isinstance(elitism, bool), "Elitism must be boolean value True or False."
        assert population_size > 0, "Population size must be greater than zero."
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population_size = population_size
        self.language_model = None

    @staticmethod
    def _scalar_fitness(fitness: float | np.ndarray | dict[str, float]) -> float:
        """Reduce a possibly vector-valued fitness to a single scalar for ranking.

        When ``sum_scores=False``, multi-agent algorithms store per-sub-agent
        fitness values.  Tournament selection needs a total ordering, so we
        collapse to the mean across sub-agents.

        A diverged agent can carry a non-finite fitness (``NaN`` from training
        instability, ``inf`` from overflow). ``np.argsort`` orders ``NaN`` as
        the *largest* value, so an unguarded NaN fitness would be ranked best
        and selected as the elite (and would win tournaments). We collapse any
        non-finite result to ``-inf`` so the existing selection discards the
        broken agent instead of promoting it.
        """
        if isinstance(fitness, dict):
            value = float(np.mean(list(fitness.values())))
        elif isinstance(fitness, (list, tuple, np.ndarray)):
            value = float(np.mean(fitness))
        else:
            value = float(fitness)
        return value if np.isfinite(value) else float("-inf")

    def _tournament(self, fitness_values: list[float]) -> int:
        """Perform tournament selection given a list of fitness values.

        :param fitness_values: List of fitness values
        :type fitness_values: list[float]
        :return: Index of the selected winner
        :rtype: int
        """
        selection = np.random.randint(0, len(fitness_values), size=self.tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        return selection[np.argmax(selection_values)]

    def _elitism(
        self,
        population: PopulationT,
    ) -> tuple[EvolvableAlgorithm, np.ndarray, int]:
        """Perform elitism selection given a population of agents.

        :param population: Population of agents
        :type population: PopulationT
        :return: Elite member of population, rank array, and max id
        :rtype: tuple[EvolvableAlgorithm, np.ndarray, int]
        """
        last_fitness = [self._scalar_fitness(indi.fitness[-1]) for indi in population]
        rank = np.argsort(last_fitness).argsort()
        max_id = max([ind.index for ind in population])
        model = population[int(np.argsort(rank)[-1])]
        elite = model.clone() if not self.language_model else model.index
        return elite, rank, max_id

    def select(
        self,
        population: PopulationT,
    ) -> tuple[EvolvableAlgorithm, PopulationT]:
        """Select the best agent and new population of agents following tournament selection.

        :param population: Population of agents
        :type population: PopulationT
        :return: Elite agent and new population
        :rtype: tuple[EvolvableAlgorithm, PopulationT]
        """
        if self.language_model is None:
            self.language_model = isinstance(population[0], LLMAlgorithm)

        return (
            self._select_llm_agents(population)
            if self.language_model
            else self._select_standard_agents(population)
        )

    def _select_standard_agents(
        self,
        population: PopulationT,
    ) -> tuple[EvolvableAlgorithm, PopulationT]:
        """Return best agent and new population of agents following tournament selection. Used for
        a population of :class:`RLAlgorithm <agilerl.algorithms.core.RLAlgorithm>` or
        :class:`MultiAgentRLAlgorithm <agilerl.algorithms.core.MultiAgentRLAlgorithm>` agents.

        :param population: Population of agents
        :type population: PopulationT
        :return: Elite agent and new population
        :rtype: tuple[EvolvableAlgorithm, PopulationT]
        """
        elite, rank, max_id = self._elitism(population)
        new_population = []
        if self.elitism:  # keep top agent in population
            elite_clone = elite.clone(wrap=False)
            # The elite is carried over unchanged: its parent is itself.
            _record_parent_index(elite_clone, elite.index)
            new_population.append(elite_clone)
            selection_size = self.population_size - 1
        else:
            selection_size = self.population_size

        # Select parents of next gen using tournament selection
        for _idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(max_id, wrap=False)
            _record_parent_index(new_individual, actor_parent.index)
            new_population.append(new_individual)

        return elite, new_population

    def _select_llm_agents(
        self,
        population: PopulationT,
    ) -> tuple[LLMAlgorithm, PopulationT]:
        """Return best agent and new population of agents following tournament selection. Used for
        a population of :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param population: Population of agents
        :type population: PopulationT
        :return: Elite agent and new population
        :rtype: tuple[LLMAlgorithm, PopulationT]
        """
        accelerator = population[0].accelerator
        new_population_idxs = []
        old_population_idxs = [ind.index for ind in population]
        unwanted_agents = {}

        if accelerator is None or (
            accelerator is not None and accelerator.is_main_process
        ):
            elite_idx, rank, max_id = self._elitism(population)
            if self.elitism:  # keep top agent in population
                new_population_idxs.append((elite_idx, elite_idx, True))
                selection_size = self.population_size - 1
            else:
                elite = population[old_population_idxs.index(elite_idx)]
                selection_size = self.population_size
            # select parents of next gen using tournament selection
            for _ in range(selection_size):
                max_id += 1
                actor_parent_idx = old_population_idxs[self._tournament(rank)]
                new_population_idxs.append(
                    (actor_parent_idx, max_id, False),
                )  # (old_idx_to_clone, new_labelled_idx, is_elite)

            # Isolate any agents that are not in the new population to be deleted
            unwanted_agents = set(old_population_idxs) - {
                idx for idx, *_ in new_population_idxs
            }

        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.num_processes > 1:
                new_population_idxs, old_population_idxs, unwanted_agents = (
                    broadcast_object_list(
                        [new_population_idxs, old_population_idxs, unwanted_agents],
                        from_process=0,
                    )
                )

        # Delete any unwanted agents from memory
        for agent_idx in old_population_idxs:
            if agent_idx in unwanted_agents:
                agent_ref = population[old_population_idxs.index(agent_idx)]
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                agent_ref.clean_up()
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                agent_ref = None

        new_population = []
        index_tracker = {}
        for idx_to_clone, new_idx, is_elite in new_population_idxs:
            if (
                agent_ref := population[old_population_idxs.index(idx_to_clone)]
            ) is not None:
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                actor_parent = agent_ref.clone(new_idx, wrap=False)
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                agent_ref.clean_up()
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                agent_ref = population[old_population_idxs.index(idx_to_clone)] = None
                index_tracker[idx_to_clone] = actor_parent
            else:
                actor_parent = index_tracker[idx_to_clone].clone(new_idx, wrap=False)
            if is_elite:
                elite = actor_parent
            new_population.append(actor_parent)
        return elite, new_population
