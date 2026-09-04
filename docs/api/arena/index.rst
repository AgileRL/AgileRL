Arena
=====

The ``agilerl-arena`` package provides the Python client and CLI for
`Arena <https://arena.agilerl.com>`_, AgileRL's RLOps platform for
cloud-based distributed training with evolutionary hyperparameter optimization.

Package layout
--------------

``agilerl-arena`` is published as its **own PyPI distribution** but exposes
modules through the shared ``agilerl.*`` namespace:

- Core ``agilerl`` ships ``agilerl.models``, ``agilerl.training``, algorithms,
  and other runtime modules.
- ``agilerl-arena`` ships only ``agilerl.arena`` (client, CLI, inference,
  :mod:`agilerl.arena.models`, and :mod:`agilerl.arena.memory`).

After ``pip install agilerl``, import
Arena code as ``agilerl.arena``, not as a top-level ``arena`` or
``agilerl_arena`` module:

.. code-block:: python

   from agilerl.arena import ArenaClient
   from agilerl.arena.models import TrainingManifest

The two wheels merge under one namespace at install time. You do not need to
install both packages to use ``agilerl.arena`` alone, but
:class:`~agilerl.training.trainer.ArenaTrainer` in core ``agilerl`` expects
``agilerl-arena`` to be present.

.. seealso::

   :ref:`arena_training` section for a usage guide with examples.

.. toctree::
   :maxdepth: 1

   client
   models
   stream
   output
   inference
   auth
   exceptions
