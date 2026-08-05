# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
from accelerate.utils import broadcast_object_list

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.protocols import EvolvableAlgorithmProtocol

AgentT = TypeVar("AgentT", bound=EvolvableAlgorithmProtocol)


class TournamentSelection:
    """The tournament selection class. Calling :func:`TournamentSelection.select() <agilerl.hpo.tournament.TournamentSelection.select>`
    on a population of agents will return a cloned population containing the best performing agent as well as the new generation of agents
    based on their fitness scores.

    :param tournament_size: Tournament selection size
    :type tournament_size: int
    :param elitism: Elitism in tournament selection. Must be ``True`` for LLM populations.
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
    def _scalar_fitness(fitness: float | npt.NDArray | dict[str, float]) -> float:
        """Reduce a possibly vector-valued fitness to a single scalar for ranking.

        When ``sum_scores=False``, multi-agent algorithms store per-sub-agent
        fitness values.  Tournament selection needs a total ordering, so we
        collapse to the mean across sub-agents.
        """
        if isinstance(fitness, dict):
            return float(np.mean(list(fitness.values())))
        if isinstance(fitness, (list, tuple, np.ndarray)):
            return float(np.mean(fitness))
        return float(fitness)

    def _tournament(self, fitness_values: Sequence[float] | npt.NDArray) -> int:
        """Perform tournament selection given a list of fitness values.

        :param fitness_values: List of fitness values
        :type fitness_values: Sequence[float] | npt.NDArray
        :return: Index of the selected winner
        :rtype: int
        """
        selection = np.random.randint(0, len(fitness_values), size=self.tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        return int(selection[np.argmax(selection_values)])

    def _elitism(
        self,
        population: list[AgentT],
    ) -> tuple[AgentT, npt.NDArray, int]:
        """Perform elitism selection given a population of agents.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Best performing member of the population, rank array, and max id
        :rtype: tuple[AgentT, npt.NDArray, int]
        """
        last_fitness = [self._scalar_fitness(indi.fitness[-1]) for indi in population]
        rank = np.argsort(last_fitness).argsort()
        max_id = max([ind.index for ind in population])
        return population[int(np.argsort(rank)[-1])], rank, max_id

    def select(
        self,
        population: list[AgentT],
    ) -> tuple[AgentT, list[AgentT]]:
        """Select the best agent and new population of agents following tournament selection.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Elite agent and new population
        :rtype: tuple[AgentT, list[AgentT]]
        """
        if self.language_model is None:
            self.language_model = isinstance(population[0], LLMAlgorithm)

        if self.language_model and not self.elitism:
            msg = (
                "TournamentSelection(elitism=False) is not supported for LLM "
                "populations. Construct TournamentSelection with elitism=True."
            )
            raise ValueError(msg)

        return (
            self._select_llm_agents(population)
            if self.language_model
            else self._select_standard_agents(population)
        )

    def _select_standard_agents(
        self,
        population: list[AgentT],
    ) -> tuple[AgentT, list[AgentT]]:
        """Return best agent and new population of agents following tournament selection. Used for
        a population of :class:`RLAlgorithm <agilerl.algorithms.core.RLAlgorithm>` or
        :class:`MultiAgentRLAlgorithm <agilerl.algorithms.core.MultiAgentRLAlgorithm>` agents.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Elite agent and new population
        :rtype: tuple[AgentT, list[AgentT]]
        """
        best_agent, rank, max_id = self._elitism(population)
        elite = best_agent.clone(index=None, wrap=True)
        new_population: list[AgentT] = []
        if self.elitism:  # keep top agent in population
            new_population.append(elite.clone(index=None, wrap=False))
            selection_size = self.population_size - 1
        else:
            selection_size = self.population_size

        # Select parents of next gen using tournament selection
        for _idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(index=max_id, wrap=False)
            new_population.append(new_individual)

        return elite, new_population

    def _select_llm_agents(
        self,
        population: list[AgentT],
    ) -> tuple[AgentT, list[AgentT]]:
        """Return best agent and new population of agents following tournament selection. Used for
        a population of :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Elite agent and new population
        :rtype: tuple[AgentT, list[AgentT]]
        """
        agent_slots: Any = population

        accelerator = population[0].accelerator
        new_population_idxs: list[tuple[int, int]] = []
        old_population_idxs = [ind.index for ind in population]
        unwanted_agents: set[int] = set()

        if accelerator is None or (
            accelerator is not None and accelerator.is_main_process
        ):
            best_agent, rank, max_id = self._elitism(population)
            elite_idx = best_agent.index
            # Elitism is required for LLM populations (enforced in select()), so
            # the elite always heads the (broadcast) selection and is recovered
            # as new_population[0] below.
            new_population_idxs.append((elite_idx, elite_idx))
            selection_size = self.population_size - 1
            # select parents of next gen using tournament selection
            for _ in range(selection_size):
                max_id += 1
                actor_parent_idx = old_population_idxs[self._tournament(rank)]
                new_population_idxs.append(
                    (actor_parent_idx, max_id),
                )  # (old_idx_to_clone, new_labelled_idx)

            # Isolate any agents that are not in the new population to be deleted
            unwanted_agents = set(old_population_idxs) - {
                idx for idx, _ in new_population_idxs
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

        # Delete any unwanted agents from memory. ``agent_slots`` only receives
        # None in the later cloning loop, so no slot is None during this pass.
        for agent_idx in old_population_idxs:
            if agent_idx in unwanted_agents:
                unwanted_ref = agent_slots[old_population_idxs.index(agent_idx)]
                if unwanted_ref.accelerator is not None:
                    unwanted_ref.accelerator.wait_for_everyone()
                unwanted_ref.clean_up()
                if unwanted_ref.accelerator is not None:
                    unwanted_ref.accelerator.wait_for_everyone()

        new_population: list[AgentT] = []
        index_tracker: dict[int, AgentT] = {}
        for idx_to_clone, new_idx in new_population_idxs:
            slot = old_population_idxs.index(idx_to_clone)
            if (agent_ref := agent_slots[slot]) is not None:
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                actor_parent = agent_ref.clone(index=new_idx, wrap=False)
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                agent_ref.clean_up()
                if agent_ref.accelerator is not None:
                    agent_ref.accelerator.wait_for_everyone()
                agent_slots[slot] = None
                index_tracker[idx_to_clone] = actor_parent
            else:
                actor_parent = index_tracker[idx_to_clone].clone(
                    index=new_idx,
                    wrap=False,
                )
            new_population.append(actor_parent)

        elite = new_population[0]
        return elite, new_population
