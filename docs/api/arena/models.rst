Arena manifest models
=====================

The ``agilerl-arena`` package provides lightweight Pydantic schemas under
:mod:`agilerl.arena.models` for validating and serializing training manifests
before they are sent to Arena. These models are **not** a substitute for
:mod:`agilerl.models` when running :class:`~agilerl.training.trainer.LocalTrainer`.

Training manifest
-----------------

.. autoclass:: agilerl.arena.models.TrainingManifest
   :members:

Algorithm registry
------------------

Arena maintains its own algorithm registry for manifest parsing and
platform-specific eligibility checks.

.. autodata:: agilerl.arena.models.ARENA_REGISTRY
