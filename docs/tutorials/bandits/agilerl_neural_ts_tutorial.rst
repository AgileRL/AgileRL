.. _neural_ts_tutorial:

PenDigits with NeuralTS
=======================

In this tutorial, we will be training a NeuralTS agent to solve the PenDigits dataset, converted into
a bandit environment. We will also use evolutionary hyperparameter optimization on a population of agents.

To complete the PenDigits environment, the agent must learn to select the best arm, or action, to take
in a given context, or state.

.. list-table::

    * - .. figure:: ../bandits/NeuralTS-PenDigits-regret.png

          Figure 1: Cumulative regret from training on the PenDigits dataset

      - .. figure:: ../bandits/NeuralTS-PenDigits-reward.png

          Figure 2: Reward from training on the PenDigits dataset

    *  -

       -

NeuralTS (:ref:`Neural Contextual Bandits with Thompson Sampling<neural_ts>`) adapts deep neural networks for both
exploration and exploitation by using a posterior distribution of the reward with a neural network
approximator as its mean, and neural tangent features as its variance.


.. literalinclude:: ../../../tutorials/Bandits/agilerl_neural_ts.py
    :language: python
