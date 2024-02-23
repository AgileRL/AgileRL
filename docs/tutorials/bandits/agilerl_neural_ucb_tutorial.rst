.. _neural_ucb_tutorial:

Iris with NeuralUCB
===================

In this tutorial, we will be training a NeuralUCB agent to solve the Iris dataset, converted into
a bandit environment.

To complete the Iris environment, the agent must learn to select the best arm, or action, to take
in a given context, or state.

.. list-table::

    * - .. figure:: ../bandits/NeuralUCB-IRIS-regret.png

          Figure 1: Cumulative regret from training on the Iris dataset

      - .. figure:: ../bandits/NeuralUCB-IRIS-reward.png

          Figure 2: Reward from training on the Iris dataset

    *  -

       -


NeuralUCB (:ref:`Neural Contextual Bandits with UCB-based Exploration<neural_ucb>`) utilizes the representational capabilities
of deep neural networks and employs a neural network-based random feature mapping to create an upper
confidence bound (UCB) for reward, enabling efficient exploration.

For this tutorial, we will use the labelled Iris dataset from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_.
These datasets can easily be imported and used for training with the Python package ``ucimlrepo``, and to choose from the hundreds of
available datasets it is as simple as changing the ``id`` parameter used by ``fetch_uci_repo``.
We can convert these labelled datasets into a bandit learning environment easily by using the ``agilerl.wrappers.learning.BanditEnv`` class.


.. literalinclude:: ../../../tutorials/Bandits/agilerl_neural_ucb.py
    :language: python
