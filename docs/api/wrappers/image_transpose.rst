Image Transpose
===============

Wrappers that transpose image observations from channels-last ``(H, W, C)``
to channels-first ``(C, H, W)`` format for PyTorch convolutions. Supports
both Gymnasium and PettingZoo environments, including nested
:class:`~gymnasium.spaces.Dict` and :class:`~gymnasium.spaces.Tuple`
observation spaces.

Helpers
-------

.. autofunction:: agilerl.wrappers.image_transpose.is_channels_last

.. autofunction:: agilerl.wrappers.image_transpose.needs_image_transpose

Gymnasium Wrapper
-----------------

.. autoclass:: agilerl.wrappers.image_transpose.ImageTranspose
   :members:

PettingZoo Wrapper
------------------

.. autoclass:: agilerl.wrappers.image_transpose.PettingZooImageTranspose
   :members:
