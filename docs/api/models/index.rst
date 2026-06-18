Models
======

Pydantic specification models used by :doc:`trainers <../train>` and
:doc:`manifests <manifest>` to declaratively configure **local** training runs.

Arena cloud submission uses a separate, lightweight model layer in
:mod:`agilerl.arena.models` (``agilerl-arena`` package).

.. toctree::
   :maxdepth: 1

   manifest
   algo
   env
   networks
   training
   hpo
