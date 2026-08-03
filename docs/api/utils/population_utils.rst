Population Utils
================

Helpers for aggregating population-level metrics.

:func:`scalar_fitness() <agilerl.utils.population_utils.scalar_fitness>` is the fitness
reduction shared by the selection strategies
(:class:`TournamentSelection <agilerl.hpo.tournament.TournamentSelection>` and
:class:`MultiFrequencySelection <agilerl.hpo.multi_frequency.MultiFrequencySelection>`).
Multi-agent algorithms evaluated with ``sum_scores=False`` record one fitness value per
sub-agent, while ranking a population requires a total order, so those rows are collapsed
to their mean across sub-agents.

.. autofunction:: agilerl.utils.population_utils.scalar_fitness

.. autofunction:: agilerl.utils.population_utils.get_nested_mean

.. autofunction:: agilerl.utils.population_utils.get_values_for_key
