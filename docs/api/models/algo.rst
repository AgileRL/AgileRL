Algorithm Specifications
========================

Base classes and the algorithm registry that maps names (e.g. ``"DQN"``)
to their concrete :class:`AlgorithmSpec` subclass.

Registry
--------

.. autoclass:: agilerl.models.algo.AlgorithmRegistry
   :members:

.. autodata:: agilerl.models.algo.ALGO_REGISTRY

Base Specs
----------

.. autoclass:: agilerl.models.algo.AlgorithmSpec
   :members:

.. autoclass:: agilerl.models.algo.RLAlgorithmSpec
   :members:

.. autoclass:: agilerl.models.algo.MultiAgentRLAlgorithmSpec
   :members:

.. autoclass:: agilerl.models.algo.LLMAlgorithmSpec
   :members:

Convenience Types
-----------------

.. autodata:: agilerl.models.JobStatus
.. autoclass:: agilerl.models.ArenaVM
   :members:
.. autoclass:: agilerl.models.ArenaCluster
   :members:
