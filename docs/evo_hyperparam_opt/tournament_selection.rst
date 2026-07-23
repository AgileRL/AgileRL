.. _tournament_selection:

Tournament Selection
====================

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If elitism is used, the best agent from a population
is automatically preserved and becomes a member of the next generation. Then, for each tournament, **k** individuals are randomly chosen, and the agent with the best evaluation
fitness is preserved. This is repeated until the population for the next generation is full.

The class :class:`TournamentSelection <agilerl.hpo.tournament.TournamentSelection>` defines the functions required for tournament selection.
:func:`TournamentSelection.select() <agilerl.hpo.tournament.TournamentSelection.select>` returns the best agent, the new generation of agents, and the
indices of the agents to mutate.

The new generation is then perturbed by the shared :ref:`mutation <mutations>` step to explore the hyperparameter space; under tournament selection every agent in the new generation is eligible for mutation.

.. code-block:: python

    from agilerl.hpo.tournament import TournamentSelection

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=6,  # Population size
    )
