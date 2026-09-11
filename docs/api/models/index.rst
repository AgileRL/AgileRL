Models
======

Pydantic models that describe a training run. You write these as YAML (a
**manifest**); trainers parse and validate the file before anything starts.
See :ref:`training_manifests` for the file layout.

Every algorithm, network, and training section is defined once in
:mod:`agilerl.arena.models` (the ``agilerl-arena`` package) and re-exported
here. Arena validates submitted jobs against the same schema. Construction a
local run needs -- building a gym from an environment spec, a replay buffer
from a buffer spec -- lives beside those classes as functions.

.. toctree::
   :maxdepth: 1

   manifest
   algorithms
   env
   networks
   training
   hpo
