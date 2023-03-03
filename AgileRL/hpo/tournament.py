import numpy as np
import copy

class TournamentSelection():

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

    def select(self, population):
        last_fitness = [np.mean(indi.fitness[-self.evo_step:]) for indi in population]
        rank = np.argsort(last_fitness).argsort()

        max_id = max([ind.index for ind in population])

        elite = copy.deepcopy([population[np.argsort(rank)[-1]]][0])

        new_population = []
        if self.elitism: # keep top agent in population
            new_population.append(elite.clone())
            selection_size = self.population_size - 1
        else:
            selection_size = self.population_size

        # select parents of next gen using tournament selection
        for idx in range(selection_size):
            max_id += 1
            actor_parent = population[self._tournament(rank)]
            new_individual = actor_parent.clone(max_id)
            new_population.append(new_individual)

        return elite, new_population