Tournament Selection
====================

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If elitism is used, the best agent from a population
is automatically preserved and becomes a member of the next generation. Then, for each tournament, k individuals are randomly chosen, and the agent with the best evaluation
fitness is preserved. This is repeated until the population for the next generation is full.

The class ``TournamentSelection()`` defines the functions required for tournament selection. ``TournamentSelection.select()`` returns the best agent, and the new generation
of agents.

.. code-block:: python

  from agilerl.hpo.tournament import TournamentSelection

  tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                    elitism=True,      # Elitism in tournament selection
                                    population_size=6, # Population size
                                    evo_step=1)        # Evaluate using last N fitness scores


Parameters
------------

.. autoclass:: agilerl.hpo.tournament.TournamentSelection
  :members:
  :inherited-members:
