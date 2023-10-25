import copy

import numpy as np


class TournamentSelection:
    """The tournament selection class.

    :param tournament_size: Tournament selection size
    :type tournament_size: int
    :param elitism: Elitism in tournament selection
    :type elitism: bool
    :param population_size: Number of agents in population
    :type population_size: int
    :param evo_step: Number of most recent fitness scores to use in evaluation
    :type evo_step: int
    """

    def __init__(self, tournament_size, elitism, population_size, evo_step):
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.population_size = population_size
        self.evo_step = evo_step

    def _tournament(self, fitness_values):
        selection = np.random.randint(0, len(fitness_values), size=self.tournament_size)
        selection_values = [fitness_values[i] for i in selection]
        winner = selection[np.argmax(selection_values)]
        return winner

    def _elitism(self, population):
        """Returns elite member of population and its id."""
        last_fitness = [np.mean(indi.fitness[-self.evo_step :]) for indi in population]
        rank = np.argsort(last_fitness).argsort()

        max_id = max([ind.index for ind in population])

        model = population[np.argsort(rank)[-1]]
        elite = copy.deepcopy(model)
        return elite, rank, max_id

    def select(self, population):
        """Returns best agent and new population of agents following tournament selection.

        :param population: Population of agents
        :type population: List[object]
        """
        elite, rank, max_id = self._elitism(population)

        new_population = []
        if self.elitism:  # keep top agent in population
            new_population.append(elite.clone(wrap=False))
            selection_size = self.population_size - 1
        else:
            selection_size = self.population_size

        # select parents of next gen using tournament selection
        for idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(max_id, wrap=False)
            new_population.append(new_individual)

        return elite, new_population
