.. _evo_hyperparam_opt:

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

.. _tournament_selection:

Tournament Selection
--------------------

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If elitism is used, the best agent from a population
is automatically preserved and becomes a member of the next generation. Then, for each tournament, **k** individuals are randomly chosen, and the agent with the best evaluation
fitness is preserved. This is repeated until the population for the next generation is full.

The class :class:`TournamentSelection <agilerl.hpo.tournament.TournamentSelection>` defines the functions required for tournament selection.
:func:`TournamentSelection.select() <agilerl.hpo.tournament.TournamentSelection.select>` returns the best agent, and the new generation of agents.

.. code-block:: python

    from agilerl.hpo.tournament import TournamentSelection

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=6,  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )


.. _mutations:

Mutation
--------

Mutations are periodically applied to our population of agents to explore the hyperparameter space, allowing different hyperparameter combinations to be trialled during training.
If certain hyperparameters prove relatively beneficial to training, then that agent is more likely to be preserved in the next generation, and so those characteristics are more
likely to remain in the population.

The :class:`Mutations <agilerl.hpo.mutation.Mutations>` class is used to mutate agents with pre-set probabilities. The available mutations currently implemented are:

    * **No mutation**: An "identity" mutation, whereby the agent is returned unchanged.
    * **Network architecture mutations**: Involves adding or removing layers or nodes. Trained weights are reused and new weights are initialized randomly.
    * **Network parameters mutation**: Mutating weights with Gaussian noise.
    * **Network activation layer mutation**: Change of activation layer.
    * **RL algorithm mutation**: Mutation of a learning hyperparameter (e.g. learning rate or batch size).

:func:`Mutations.mutation() <agilerl.hpo.mutation.Mutations.mutation>` returns a mutated population.

Tournament selection and mutations are applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

    from agilerl.hpo.mutation import Mutations

    mutations = Mutations(
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

Network Architecture Mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overview
^^^^^^^^

In machine learning it is often difficult to identify the optimal architecture of a neural network and the capacity required for a given task. In RL,
this is particularly challenging due to the large number of transitions required to learn a policy. We address this by introducing a framework for performing
architecture mutations through the :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` abstraction (see :ref:`here <evolvable_networks>`). Specifically,
it allows us to seamlessly track and apply architecture mutations for networks with nested evolvable modules (see below the simplest example of an evolvable
multi-layer perceptron). This is particularly useful for RL algorithms, where we define default architectures suitable for a variety of tasks (i.e. combinations of
observation and action spaces), which require very different network architectures.

.. collapse:: EvolvableMLP

    .. literalinclude:: ../../agilerl/modules/mlp.py
        :language: python

For the above reason, we define the :class:`EvolvableNetwork <agilerl.modules.base.EvolvableNetwork>` base class, which inherits from :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>`.
This abstraction allows us to define common networks used in RL algorithms very simply, since it automatically creates an appropriate encoder for the passed observation space. After,
we just create a head to the the network that processes the encoded observations into an appropriate number of outputs (for e.g. policies or critics).


Single-Agent
^^^^^^^^^^^^

Multi-Agent
^^^^^^^^^^^


.. note::
    AgileRL currently doesn't support architecture mutations for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` objects.



RL Hyperparameter Mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
