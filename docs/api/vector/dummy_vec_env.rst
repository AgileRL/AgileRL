Dummy Vectorized Environments
=============================

Lightweight wrappers that expose a single environment through a
``VectorEnv``-like API, adding a leading batch dimension of size 1.

Gymnasium
---------

.. autoclass:: agilerl.vector.dummy_vec_env.DummyVecEnv
   :members:

PettingZoo
----------

.. autoclass:: agilerl.vector.dummy_vec_env.DummyPzVecEnv
   :members:
