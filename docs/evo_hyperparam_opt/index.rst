Evolutionary Hyperparameter Optimization
========================================

Traditionally, hyperparameter optimization (HPO) for reinforcement learning (RL) is particularly difficult when compared to other types of machine learning.
This is for several reasons, including the relative sample inefficiency of RL and its sensitivity to hyperparameters.

AgileRL is initially focused on improving HPO for RL in order to allow faster development with robust training.
Evolutionary algorithms have been shown to allow faster, automatic convergence to optimal hyperparameters than other HPO methods by taking advantage of
shared memory between a population of agents acting in identical environments.

At regular intervals, after learning from shared experiences, a population of agents can be evaluated in an environment. Through tournament selection, the
best agents are selected to survive until the next generation, and their offspring are mutated to further explore the hyperparameter space.
Eventually, the optimal hyperparameters for learning in a given environment can be reached in significantly less steps than are required using other HPO methods.

.. figure:: https://github.com/AgileRL/AgileRL/assets/118982716/27260e5a-80cb-4950-a858-21d1debb5d21
   :align: center

   Our evolutionary approach allows for HPO in a single training run compared to Bayesian methods that require multiple sequential training runs
   to achieve similar, and often inferior, results.
