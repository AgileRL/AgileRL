Logger
======

The logger hierarchy consumes :class:`~agilerl.population.MetricsReport` snapshots
and writes them to various backends. A training run typically uses several
loggers at once, e.g. ``[StdOutLogger, WandbLogger]``.

Base Class
----------

.. autoclass:: agilerl.logger.Logger
   :members:

Console
-------

.. autoclass:: agilerl.logger.StdOutLogger
   :members:

Weights & Biases
----------------

.. autoclass:: agilerl.logger.WandbLogger
   :members:

CSV
---

.. autoclass:: agilerl.logger.CSVLogger
   :members:

TensorBoard
------------

.. autoclass:: agilerl.logger.TensorboardLogger
   :members:
