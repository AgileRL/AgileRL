Training Manifest
=================

The :class:`TrainingManifest` validates a full YAML/JSON training
configuration and dispatches the algorithm, network, and environment
sections to their concrete spec classes.

.. autoclass:: agilerl.models.manifest.TrainingManifest
   :members:

Arena Manifest
--------------

:class:`ArenaManifest` is the variant used when submitting jobs to Arena. It
inherits all behaviour from :class:`TrainingManifest` but restricts the
``algorithm`` section to algorithms registered with ``arena=True``.

.. autoclass:: agilerl.models.manifest.ArenaManifest
   :show-inheritance:
