Training Manifest
=================

:class:`~agilerl.models.manifest.TrainingManifest` *is*
:class:`~agilerl.arena.models.TrainingManifest`. The YAML schema is defined
once in ``agilerl-arena``; the framework does not subclass it.

Building a gym from an environment spec, or a replay buffer from a buffer
spec, is not part of that schema. Those helpers live beside the models, and
in :class:`~agilerl.training.trainer.LocalTrainer`.
:func:`~agilerl.models.manifest.from_trainer_specs` builds a manifest from the
objects a trainer already holds.

.. autoclass:: agilerl.arena.models.TrainingManifest
   :members:

.. autofunction:: agilerl.models.manifest.from_trainer_specs
