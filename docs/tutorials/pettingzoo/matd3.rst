.. _MATD3 tutorial:

AgileRL Tutorial: Implementing MATD3
====================================

This tutorial shows how to train an :ref:`MATD3<matd3>` agent on the `simple speaker listener <https://pettingzoo.farama.org/environments/mpe/simple_speaker_listener/>`_ multi-particle environment.

What is MATD3?
--------------

:ref:`MATD3<matd3>` (Multi-Agent Twin Delayed Deep Deterministic Policy Gradients) extends the :ref:`MADDPG<maddpg>` (Multi-Agent Deep Deterministic Policy Gradients) algorithm to reduce overestimation bias in multi-agent domains through the use of a second set of critic networks and delayed updates of the policy networks. This enables superior performance when compared to MADDPG. For further information on MATD3, check out the :ref:`documentation<matd3>`.

Can I use it?
-------------

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * -
     - Action
     - Observation
   * - Discrete
     - ✔️
     - ✔️
   * - Continuous
     - ✔️
     - ✔️

Code
-----

Train multiple agents using MADDPG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code should run without any issues. The comments are designed to help you understand how to use PettingZoo with AgileRL. If you have any questions, please feel free to ask in the `Discord server <https://discord.com/invite/eB8HyTA2ux>`_.

.. literalinclude:: ../../../tutorials/PettingZoo/agilerl_matd3.py
   :language: python

Watch the trained agents play
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code allows you to load your saved MATD3 algorithm from the previous training block, test the algorithms performance, and then visualise a number of episodes as a gif.

.. literalinclude:: ../../../tutorials/PettingZoo/render_agilerl_matd3.py
   :language: python
