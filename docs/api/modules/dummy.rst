.. _dummy:

Dummy Evolvable Wrapper
=======================

Converts a ``torch.nn.Module`` into an evolvable module. This is useful when users wish to load a pre-trained model that is not
an :class:`agilerl.modules.EvolvableModule` and use it in an evolvable AgileRL algorithm, disabling architecture mutations but still
allowing for RL hyperparameter and weight mutations.

Parameters
------------

.. autoclass:: agilerl.modules.dummy.DummyEvolvable
  :members:
