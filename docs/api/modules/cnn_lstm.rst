.. _cnn_lstm:

Evolvable CNN → LSTM encoder
============================

Use with ``recurrent=True`` on algorithms such as PPO when observations are
3D image ``Box`` spaces (partial observability over pixels). The trainer and
``EvolvableNetwork`` select this encoder automatically; manifests declare
``arch: cnn_lstm`` under ``encoder_config``.

Parameters
----------

.. autoclass:: agilerl.modules.cnn_lstm.EvolvableCnnLstm
  :members:
