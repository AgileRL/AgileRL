Training & Replay Buffer Specs
==============================

:class:`~agilerl.models.training.TrainingSpec` and
:class:`~agilerl.models.training.ReplayBufferSpec` are the same schema classes
Arena uses. Buffer construction lives beside them as
:func:`~agilerl.models.training.init_buffer` and
:func:`~agilerl.models.training.init_n_step_buffer`.

.. autoclass:: agilerl.arena.models.training.TrainingSpec
   :members:

.. autoclass:: agilerl.arena.models.training.ReplayBufferSpec
   :members:

.. autofunction:: agilerl.models.training.init_buffer

.. autofunction:: agilerl.models.training.init_n_step_buffer
