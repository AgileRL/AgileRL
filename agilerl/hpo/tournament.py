from typing import List, Tuple

import numpy as np
from accelerate.utils import broadcast_object_list

from agilerl.algorithms.core.base import EvolvableAlgorithm

PopulationType = List[EvolvableAlgorithm]


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
    :param eval_loop: Number of most recent fitness scores to use in evaluation
    :type eval_loop: int
    """

    def __init__(
        self,
        tournament_size: int,
        elitism: bool,
        population_size: int,
        eval_loop: int,
    ) -> None:
        assert tournament_size > 0, "Tournament size must be greater than zero."
        assert isinstance(elitism, bool), "Elitism must be boolean value True or False."
        assert population_size > 0, "Population size must be greater than zero."
        assert eval_loop > 0, "Evo step must be greater than zero."
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population_size = population_size
        self.eval_loop = eval_loop
        self.language_model = None

    def _tournament(self, fitness_values: List[float]) -> int:
        """
        Perform tournament selection given a list of fitness values.

        :param fitness_values: List of fitness values
        :type fitness_values: list[float]
        :return: Index of the selected winner
        :rtype: int
        """
        selection = np.random.randint(0, len(fitness_values), size=self.tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        winner = selection[np.argmax(selection_values)]
        return winner

    def _elitism(
        self, population: PopulationType
    ) -> Tuple[EvolvableAlgorithm, np.ndarray, int]:
        """
        Perform elitism selection given a population of agents.

        :param population: Population of agents
        :type population: PopulationType
        :return: Elite member of population, rank array, and max id
        :rtype: tuple[EvolvableAlgorithm, np.ndarray, int]
        """
        last_fitness = [np.mean(indi.fitness[-self.eval_loop :]) for indi in population]
        rank = np.argsort(last_fitness).argsort()
        max_id = max([ind.index for ind in population])
        model = population[int(np.argsort(rank)[-1])]
        elite = model.clone() if not self.language_model else model.index
        return elite, rank, max_id

    def select(
        self, population: PopulationType
    ) -> Tuple[EvolvableAlgorithm, PopulationType]:
        """
        Select the best agent and new population of agents following tournament selection.

        :param population: Population of agents
        :type population: PopulationType
        :return: Elite agent and new population
        :rtype: tuple[EvolvableAlgorithm, PopulationType]
        """
        if self.language_model is None:
            self.language_model = population[0].algo == "GRPO"

        return (
            self._select_llm_agents(population)
            if self.language_model
            else self._select_standard_agents(population)
        )

    def _select_standard_agents(
        self, population: PopulationType
    ) -> Tuple[EvolvableAlgorithm, PopulationType]:
        """
        Returns best agent and new population of agents following tournament selection. Used for
        a population of :class:`EvolvableAlgorithm <agilerl.algorithms.core.RLAlgorithm>` or
        :class:`MultiAgentRLAlgorithm <agilerl.algorithms.core.MultiAgentRLAlgorithm>` agents.

        :param population: Population of agents
        :type population: PopulationType
        :return: Elite agent and new population
        :rtype: tuple[EvolvableAlgorithm, PopulationType]
        """
        elite, rank, max_id = self._elitism(population)
        new_population = []
        if self.elitism:  # keep top agent in population
            new_population.append(elite.clone(wrap=False))
            selection_size = self.population_size - 1
        else:
            selection_size = self.population_size

        # Select parents of next gen using tournament selection
        for idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(max_id, wrap=False)
            new_population.append(new_individual)

        return elite, new_population

    def _select_llm_agents(
        self, population: PopulationType
    ) -> Tuple[EvolvableAlgorithm, PopulationType]:
        """
        Returns best agent and new population of agents following tournament selection. Used for
        a population of :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param population: Population of agents
        :type population: PopulationType
        :return: Elite agent and new population
        :rtype: tuple[EvolvableAlgorithm, PopulationType]
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
                    (actor_parent_idx, max_id, False)
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
