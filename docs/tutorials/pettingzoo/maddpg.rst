.. _MADDPG tutorial:

AgileRL Tutorial: Implementing MADDPG
=====================================

This tutorial shows how to train an :ref:`MADDPG<maddpg>` agent on the `space invaders <https://pettingzoo.farama.org/environments/atari/space_invaders/>`_ atari environment.

What is MADDPG?
---------------

:ref:`MADDPG<maddpg>` (Multi-Agent Deep Deterministic Policy Gradients) extends the :ref:`DDPG<ddpg>` (Deep Deterministic Policy Gradients) algorithm to enable cooperative or competitive training of multiple agents in complex environments, enhancing the stability and convergence of the learning process through decentralized actor and centralized critic architectures. For further information on MADDPG, check out the :ref:`documentation<maddpg>`.

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

.. literalinclude:: ../../../tutorials/PettingZoo/agilerl_maddpg.py
   :language: python

Watch the trained agents play
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code allows you to load your saved MADDPG algorithm from the previous training block, test the algorithms performance, and then visualise a number of episodes as a gif.

.. literalinclude:: ../../../tutorials/PettingZoo/render_agilerl_maddpg.py
   :language: python
