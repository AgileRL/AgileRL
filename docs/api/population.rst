Population
==========

The :class:`~agilerl.population.Population` container manages a collection of
evolutionary agents. It collects per-agent metrics into a
:class:`~agilerl.population.PopulationMetrics` snapshot each evolution step and
formats them into :class:`~agilerl.population.MetricsReport` objects consumed by
:doc:`loggers <logger>`.

Metrics Snapshots
-----------------

.. autoclass:: agilerl.population.PopulationMetrics
   :members:

.. autoclass:: agilerl.population.MetricsReport
   :members:

Population Container
--------------------

.. autoclass:: agilerl.population.Population
   :members:
