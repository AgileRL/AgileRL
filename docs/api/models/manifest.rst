Training Manifest (local training)
==================================

The core :class:`~agilerl.models.manifest.TrainingManifest` validates a full
YAML/JSON training configuration for **local** runs. It dispatches the
algorithm, network, and environment sections to concrete spec classes under
:mod:`agilerl.models` and uses :data:`~agilerl.models.algo.ALGO_REGISTRY`.

For Arena job submission, use the separate manifest model in
:mod:`agilerl.arena.models` instead (see :doc:`../arena/models`).

.. autoclass:: agilerl.models.manifest.TrainingManifest
   :members:
