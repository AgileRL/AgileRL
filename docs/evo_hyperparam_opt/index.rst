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

:func:`Mutations.mutation(population) <agilerl.hpo.mutation.Mutations.mutation>` returns a mutated population.

Tournament selection and mutations are applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

    from agilerl.hpo.mutation import Mutations

    mutations = Mutations(
        no_mutation=0.4,     # No mutation
        architecture=0.2,    # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,      # Network parameters mutation
        activation=0,        # Activation layer mutation
        rl_hp=0.2,           # RL hyperparameter mutation
        mutation_sd=0.1,     # Mutation strength
        rand_seed=1,         # Random seed
        device=device,
    )

EvolvableAlgorithm API
~~~~~~~~~~~~~~~~~~~~~~

AgileRL algorithms inherit from the :class:`EvolvableAlgorithm <agilerl.algorithms.core.base.EvolvableAlgorithm>` base class, which provides an interface for easily mutating its hyperparameters
and the architecture of its network constituents. A :class:`MutationRegistry <agilerl.algorithms.core.registry.MutationRegistry>` is automatically created upon initialisation that keeps track
of the hyperparameters and evolvable networks registered for mutation. Specifically, algorithms can register mutable attributes in the following ways:

    - Using :func:`EvolvableAlgorithm.register_network_group() <agilerl.algorithms.core.base.EvolvableAlgorithm.register_network_group>` to register a
      :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>` of evolvable networks.

    .. note::
        Any ``EvolvableAlgorithm`` should register at least one ``NetworkGroup`` corresponding to the policy (i.e. the network used to select actions) by setting ``policy=True``.

    - All AgileRL algorithms include a ``hp_config`` argument that can be used to register RL hyperparameters for mutation. Specifically, users should instantiate a
      :class:`HyperparameterConfig <agilerl.algorithms.core.registry.HyperparameterConfig>` dataclass with the :class:`RLParameter <agilerl.algorithms.core.registry.RLParameter>`'s
      you wish to mutate, which should be available as attributes of the algorithm. If we wanted to mutate the learning rate, batch size, and learning step in e.g. ``DQN``:

    .. code-block:: python

        from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

        # Need to use the algorithms attribute names in DQN 'lr', 'batch_size',
        # and 'learn_step' to register the hyperparameters
        hp_config = HyperparameterConfig(
            lr=RLParameter(min=1e-4, max=1e-2, dtype=float),
            batch_size=RLParameter(min=32, max=256, dtype=int),
            learn_step=RLParameter(min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75),
        )

    - The optimizers used in an algorithm are also indirectly mutable since they include mutable parameters such as the learning rate, and optimize evolvable networks. For this reason,
      all optimizers in AgileRL must be wrapped using :class:`OptimizerWrapper <agilerl.algorithms.core.optimizer_wrapper.OptimizerWrapper>`, specifying the ``torch.optim.Optimizer`` to be used
      as well as the attributes containing the mutable networks it must optimize. For example, in ``PPO`` we would wrap the optimizer which updates both the actor and critic networks as follows:

    .. code-block:: python

        from agilerl.algorithms.core.base import EvolvableAlgorithm
        from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
        import torch.optim as optim

        class CustomAlgorithm(EvolvableAlgorithm):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Define the algorithm's attributes / networks
                self.lr = 1e-4
                self.actor = ... # EvolvableModule instance
                self.critic = ... # EvolvableModule instance

                # NOTE: We must pass the attributes containing
                # the mutable networks to the OptimizerWrapper
                self.optimizer = OptimizerWrapper(
                    optim.Adam,
                    networks=[self.actor, self.critic],
                    lr=self.lr
                )

    .. note::
        AgileRL expects ``OptimizerWrapper`` and ``NetworkGroup`` objects to be defined and registered in the ``__init__`` method of an algorithm.

Architecture Mutations
~~~~~~~~~~~~~~~~~~~~~~

.. note::
    AgileRL currently doesn't support architecture mutations for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` objects.


Evolvable Networks Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In machine learning it is often difficult to identify the optimal architecture of a neural network and the capacity required to solve a given problem. In RL,
this is particularly challenging due to the large number of transitions needed to learn a policy. We address this by introducing a framework for performing
architecture mutations through the :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` abstraction (see :ref:`here <evolvable_networks>` for more
details). Specifically, it allows us to seamlessly track and apply architecture mutations in networks with nested evolvable modules. This is particularly useful
in RL algorithms, where we define default configurations suitable for a variety of tasks (i.e. combinations of observation and action spaces), which require
very different architectures.

For the above reason, we define the :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>` base class, which inherits from ``EvolvableModule``.
This abstraction allows us to define common networks used in RL algorithms very simply, since it automatically creates an appropriate encoder for the passed observation space. After,
we just need to create a head to the the network that processes the encoded observations into an appropriate number of outputs for e.g. policies or critics.

It is common for RL algorithms to use multiple networks throughout training (e.g. actors and critics) to mitigate risks intrinsic to the RL learning procedure such as e.g. managing the
trade-off between exploration and exploitation. How we apply architecture mutations in such cases differs slightly in single- and multi-agent settings.

Single-Agent
^^^^^^^^^^^^
Architecture mutations in single-agent settings are straightforward because we can assume that the same base architecture is used in all the networks of an algorithm, allowing us to apply the
same mutation to all the networks (justified by the fact that these usually solve tasks of similar complexity and thus require `roughly` the same capacity). We can do this because
networks in RL typically all process observations into either actions or values. Even though the outputs of e.g. actors and critics differ, they will share the same type of encoder
and head (since the encoder processes the same observations and the head is always an instance of ``EvolvableMLP``) - which means they will share the same mutation methods.

Given this assumption, the procedure to perform an architecture mutation is as follows:

    1. Sample a mutation method for the policy network using :func:`EvolvableModule.sample_mutation_method() <agilerl.modules.base.EvolvableModule.sample_mutation_method>`

    2. Apply the same mutation to the rest of the evaluation networks found in the ``MutationRegistry`` e.g. the critic in ``PPO``.

    3. Reinitialize the networks that share parameters with the evaluation networks but aren't optimized directly during training (e.g. target networks) with the mutated architecture.


Multi-Agent
^^^^^^^^^^^
In :ref:`multi-agent settings <multiagenttraining>`, we can't make the previous assumption and follow the same procedure for various reasons.

- Different agents don't necessarily share the same observation space and thus their policies will have different architectures (i.e. we can't apply a single mutation generally to all agents,
  and probably wouldn't want to do so in the first place since they solve different tasks!). We therefore want to sample a mutation method from the policy of a single agent and apply it
  to the policies of agents that share the same mutation method.

- We often have situations with a combination of both centralized (i.e. process information from all agents) and decentralized (i.e. process information from a single agent) networks. For instance,
  the policies in ``MADDPG`` and ``MATD3`` are decentralized, while the critics are centralized. In these cases, we can't necessarily apply the same mutation to different networks corresponding to the
  same agent. What we can do, however, is try to apply an analogous mutation across the board. For centralized networks in the aforementioned algorithms we employ
  :class:`EvolvableMultiInput <agilerl.modules.multi_input.EvolvableMultiInput>` as an encoder, which allows us to process observations from all agents into a single output. What we do then is look at
  the executed mutations for the policies and try to apply an equivalent mutation to the rest of the evaluation networks..

Summarising the above considerations, the procedure to perform an architecture mutation in multi-agent settings is as follows:

    1. Sample a mutation from the policy of a single sub-agent using :func:`ModuleDict.sample_mutation_method() <agilerl.modules.base.ModuleDict.sample_mutation_method>`

    2. Apply the sampled mutation to other sub-agents that share the same mutation method.

    3. Iterate over the rest of evaluation networks found in the ``MutationRegistry`` and apply an analogous mutation to the mutated agents.

    4. Reinitialize the networks that share parameters with the evaluation networks but aren't optimized directly during training (e.g. target networks) with the mutated architecture.

This has proven to be successful in our experiments, but it is still experimental and we are always open to discussing feedback and suggestions for improvement through our `Discord <https://discord.gg/eB8HyTA2ux>`_.

RL Hyperparameter Mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mutations on algorithm-specific hyperparameters can be configured through the ``hp_config`` argument of the algorithm. This is done by instantiating a
:class:`HyperparameterConfig <agilerl.algorithms.core.registry.HyperparameterConfig>` dataclass with the :class:`RLParameter <agilerl.algorithms.core.registry.RLParameter>`'s
you wish to mutate, which should be available as attributes of the algorithm (will raise an error if not). This configuration is automatically registered with the algorithms
``MutationRegistry`` and used by ``Mutations`` to perform mutations through the :func:`Mutations.rl_hyperparam_mutation() <agilerl.hpo.mutation.Mutations.rl_hyperparam_mutation>`
method. If we wanted to mutate the learning rate, batch size, and learning step in e.g. ``DQN``:

.. code-block:: python

    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

    # Need to use the algorithms attribute names in DQN 'lr', 'batch_size',
    # and 'learn_step' to register the hyperparameters
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2, dtype=float),
        batch_size=RLParameter(min=32, max=256, dtype=int),
        learn_step=RLParameter(min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75),
    )


Network Parameter Mutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
AgileRL allows mutations on the weights of the policy registered through
:func:`EvolvableAlgorithm.register_network_group() <agilerl.algorithms.core.base.EvolvableAlgorithm.register_network_group>`. Specifically, it selects
10% of the weights randomly to mutate (ignoring normalization layers) and applies a Gaussian noise with a standard deviation of ``mutation_sd`` to them. It does so
in three different ways, clamping mutated values to prevent extreme changes:

    - **Normal mutation**: Adds noise with standard deviation proportional to current weight values.

    - **Super mutation**: Adds larger noise for more significant changes.

    - **Reset mutation**: Completely resets weights to new random values.
