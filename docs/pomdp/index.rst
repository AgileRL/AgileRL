.. _pomdp:

Partially Observable Markov Decision Processes (POMDPs)
=======================================================

Reinforcement learning problems are often formulated as Markov Decision Processes (MDPs), where the agent has full observability of the environment as it pertains
to the information required to predict optimal actions (i.e. "only the present is required to predict the future"). However, in many real-world applications this
assumption may not hold since some information about the past is needed to make optimal decisions (so the current state only "partially" gives us the information
required to make decisions). Some examples of such problems may be:

    - A robot navigating a maze needs to remember which paths it has already tried, not just its current position
    - A trading agent must track price trends over time, not just the current price point
    - A self-driving car has to remember the recent trajectory of other vehicles to predict where they might go next

These scenarios are examples of scenarios where the agent only receives incomplete or noisy observations of the true environment state. This partial observability
makes the learning task significantly more challenging than fully observable MDPs, since the agent needs to:

    1. Remember important information from past observations
    2. Infer hidden state information from incomplete observations
    3. Deal with uncertainty about the true state of the environment

This is where recurrent neural networks (RNNs) become particularly valuable. Unlike standard feedforward networks, RNNs maintain an internal memory state that can
help the agent identify temporal patterns in the observation sequence and make better decisions with incomplete information.

Recurrent PPO
-------------

AgileRL offers a :class:`PPO <agilerl.algorithms.ppo.PPO>` implementation that supports performing evolutionary hyperparameter optimisation on recurrent neural
networks (RNNs) for partially observable environments. To support these settings, we need to maintain the hidden state of the RNN throughout the rollout collection
process through a special :func:`collect_rollouts_recurrent() <agilerl.rollouts.collect_rollouts_recurrent>` function. This is already integrated in our
:func:`train_on_policy() <agilerl.training.train_on_policy.train_on_policy>` function, and used automatically if we set ``recurrent=True``  and ``use_rollout_buffer=True``
in the :class:`PPO <agilerl.algorithms.ppo.PPO>` constructor.

For an end-to-end example of how to train ``PPO`` on a partially observable environment, please refer to
the :ref:`Partially Observable Pendulum-v1 with Recurrent PPO <agilerl_recurrent_ppo_tutorial>` tutorial.
