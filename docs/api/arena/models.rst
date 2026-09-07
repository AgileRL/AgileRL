Arena manifest models
=====================

The ``agilerl-arena`` package is where the training manifest schema lives:
the Pydantic models under :mod:`agilerl.arena.models` describe every
algorithm, network, and training section, with no dependency on torch. They
are the same classes :class:`~agilerl.training.trainer.LocalTrainer` trains
from; :mod:`agilerl.models` re-exports them and adds only what a local run
needs (see :doc:`../models/index`).

Training manifest
-----------------

.. autoclass:: agilerl.arena.models.TrainingManifest
   :members:

Algorithm registry
------------------

One registry maps algorithm names to their spec classes, and every consumer
-- the local trainer, the manifest, Arena submission -- reads it.

.. autodata:: agilerl.arena.models.MANIFEST_REGISTRY
