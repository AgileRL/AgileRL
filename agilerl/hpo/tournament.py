# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.distributed import (
    barrier,
    broadcast_object_list,
    is_main_process,
)
from agilerl.protocols import EvolvableAlgorithmProtocol
from agilerl.utils.population_utils import release_agents, scalar_fitness

AgentT = TypeVar("AgentT", bound=EvolvableAlgorithmProtocol)


def _cleanup_unwanted_llm_slots(
    agent_slots: list[AgentT | None],
    old_population_idxs: list[int],
    unwanted_agents: set[int],
) -> None:
    """Free LLM agents that lost the tournament."""
    for agent_idx in old_population_idxs:
        if agent_idx not in unwanted_agents:
            continue
        unwanted_ref = agent_slots[old_population_idxs.index(agent_idx)]
        if unwanted_ref is None:
            continue
        barrier()
        unwanted_ref.clean_up()
        barrier()


def _clone_llm_population(
    agent_slots: list[AgentT | None],
    old_population_idxs: list[int],
    new_population_idxs: list[tuple[int, int]],
) -> list[AgentT]:
    """Clone selected parents into the next LLM generation."""
    new_population: list[AgentT] = []
    index_tracker: dict[int, AgentT] = {}
    for idx_to_clone, new_idx in new_population_idxs:
        slot = old_population_idxs.index(idx_to_clone)
        agent_ref = agent_slots[slot]
        if agent_ref is not None:
            barrier()
            actor_parent = agent_ref.clone(index=new_idx, wrap=False)
            barrier()
            agent_ref.clean_up()
            barrier()
            agent_slots[slot] = None
            index_tracker[idx_to_clone] = actor_parent
        else:
            actor_parent = index_tracker[idx_to_clone].clone(
                index=new_idx,
                wrap=False,
            )
        new_population.append(actor_parent)
    return new_population


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
        last_fitness = [scalar_fitness(indi.fitness[-1]) for indi in population]
        rank = np.argsort(last_fitness).argsort()
        max_id = max([ind.index for ind in population])
        return population[int(np.argsort(rank)[-1])], rank, max_id

    def select(
        self,
        population: list[AgentT],
    ) -> tuple[AgentT, list[AgentT], list[int] | None]:
        """Select the best agent and new population of agents following tournament selection.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Elite agent, new population, and None (mutate the whole population)
        :rtype: tuple[AgentT, list[AgentT], list[int] | None]
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
    ) -> tuple[AgentT, list[AgentT], None]:
        """Return best agent and new population of agents following tournament selection. Used for
        a population of :class:`SingleAgentAlgorithm <agilerl.algorithms.core.SingleAgentAlgorithm>` or
        :class:`MultiAgentAlgorithm <agilerl.algorithms.core.MultiAgentAlgorithm>` agents.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Elite agent, new population, and None (mutate the whole population)
        :rtype: tuple[AgentT, list[AgentT], None]
        """
        best_agent, rank, max_id = self._elitism(population)
        new_population: list[AgentT] = []
        elite = best_agent.clone(index=None, wrap=False)
        if self.elitism:  # keep top agent in population
            new_population.append(elite)
            selection_size = self.population_size - 1
        else:
            selection_size = self.population_size

        # Select parents of next gen using tournament selection
        for _idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(index=max_id, wrap=False)
            new_population.append(new_individual)

        new_ids = {id(agent) for agent in new_population}
        evicted = [agent for agent in population if id(agent) not in new_ids]
        release_agents(evicted, population[0].accelerator)

        return elite, new_population, None

    def _select_llm_agents(
        self,
        population: list[AgentT],
    ) -> tuple[AgentT, list[AgentT], None]:
        """Return best agent and new population of agents following tournament selection. Used for
        a population of :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param population: Population of agents
        :type population: list[AgentT]
        :return: Elite agent, new population, and None (mutate the whole population)
        :rtype: tuple[AgentT, list[AgentT], None]
        """
        agent_slots: list[AgentT | None] = list(population)

        new_population_idxs: list[tuple[int, int]] = []
        old_population_idxs = [ind.index for ind in population]
        unwanted_agents: set[int] = set()

        if is_main_process():
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

        barrier()
        new_population_idxs, old_population_idxs, unwanted_agents = (
            broadcast_object_list(
                [new_population_idxs, old_population_idxs, unwanted_agents],
                src=0,
            )
        )

        # Delete any unwanted agents from memory. ``agent_slots`` only receives
        # None in the later cloning loop, so no slot is None during this pass.
        _cleanup_unwanted_llm_slots(agent_slots, old_population_idxs, unwanted_agents)
        new_population = _clone_llm_population(
            agent_slots, old_population_idxs, new_population_idxs
        )
        elite = new_population[0]
        return elite, new_population, None
