.. _mutations:

Mutation
========

Mutations are periodically applied to our population of agents to explore the hyperparameter space, allowing different hyperparameter combinations to be trialled during training.
If certain hyperparameters prove relatively beneficial to training, then that agent is more likely to be preserved in the next generation, and so those characteristics are more
likely to remain in the population.

The :class:`Mutations <agilerl.hpo.mutation.Mutations>` class is used to mutate agents with pre-set probabilities. The available mutations currently implemented are:

    * **No mutation**: An "identity" mutation, whereby the agent is returned unchanged.
    * **Network architecture mutations**: Involves adding or removing layers or nodes. Trained weights are reused, and added capacity is initialized to preserve the network's function where the architecture allows it (see :ref:`function_preserving`), and randomly otherwise.
    * **Network parameters mutation**: Mutating weights with Gaussian noise, preceded by ReGraMa resets of the neurons that have stopped learning.
    * **Network activation layer mutation**: Change of activation layer.
    * **RL algorithm mutation**: Mutation of a learning hyperparameter (e.g. learning rate or batch size).

:func:`Mutations.mutation(population) <agilerl.hpo.mutation.Mutations.mutation>` returns a mutated population.

Mutation is the shared *explore* step of the evolutionary loop: after a selection strategy has reshaped the population, mutation perturbs the nominated agents to trial new hyperparameter and architecture combinations. The selection strategy decides *which* agents are mutated, while ``Mutations`` decides *how*.

Which agents get mutated is carried by the optional ``indices`` argument of :func:`Mutations.mutation() <agilerl.hpo.mutation.Mutations.mutation>`. :ref:`Tournament selection <tournament_selection>` mutates the whole new generation (``indices=None``), whereas :ref:`multi-frequency selection <multi_frequency_selection>` mutates only the clones that replace each subpopulation's losers, whose indices :func:`select() <agilerl.hpo.multi_frequency.MultiFrequencySelection.select>` returns.

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
----------------------

AgileRL algorithms inherit from the :class:`EvolvableAlgorithm <agilerl.algorithms.core.base.EvolvableAlgorithm>` base class, which provides an interface for easily mutating its hyperparameters
and the architecture of its network constituents. A :class:`MutationRegistry <agilerl.algorithms.core.registry.MutationRegistry>` is automatically created upon initialisation that keeps track
of the hyperparameters and evolvable networks registered for mutation. Specifically, algorithms can register mutable attributes in the following ways:

1. Using :func:`EvolvableAlgorithm.register_network_group() <agilerl.algorithms.core.base.EvolvableAlgorithm.register_network_group>` to register a
   :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>` of evolvable networks.

.. note::
    Any ``EvolvableAlgorithm`` should register at least one ``NetworkGroup`` corresponding to the policy (i.e. the network used to select actions) by setting ``policy=True``.

1. All AgileRL algorithms automatically configure sensible default RL hyperparameters for mutation when ``hp_config=None`` (usually the learning rate, batch size, and learning step). The ranges
   are derived dynamically from the algorithm's initial hyperparameter values. If you need to override these defaults, you can pass a custom
   :class:`HyperparameterConfig <agilerl.algorithms.core.registry.HyperparameterConfig>` with the :class:`RLParameter <agilerl.algorithms.core.registry.RLParameter>`'s
   you wish to mutate. For example, to customize the mutation ranges for ``DQN``:

.. code-block:: python

    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

    # Override default mutation ranges for specific hyperparameters
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=32, max=256),
        learn_step=RLParameter(min=1, max=10, grow_factor=1.5, shrink_factor=0.75),
    )

3. The optimizers used in an algorithm are also indirectly mutable since they include mutable parameters such as the learning rate, and optimize evolvable networks. For this reason,
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
----------------------

Evolvable Networks Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In machine learning it is often difficult to identify the optimal architecture of a neural network and the capacity required to solve a given problem. In RL,
this is particularly challenging due to the large number of transitions needed to learn a policy. We address this by introducing a framework for performing
architecture mutations through the :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` abstraction. It allows us to seamlessly track and apply
architecture mutations in networks with nested evolvable modules. This is particularly useful in RL algorithms, where we define default configurations
suitable for a variety of tasks (i.e. combinations of observation and action spaces), which require very different architectures.

For the above reason, we define the :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>` base class, which inherits from ``EvolvableModule``.
This abstraction allows us to define common networks used in RL algorithms very simply, since it automatically creates an appropriate encoder for the passed observation space. After,
we just need to create a head to the the network that processes the encoded observations into an appropriate number of outputs for e.g. policies or critics.

It is common for RL algorithms to use multiple networks throughout training (e.g. actors and critics) to mitigate risks intrinsic to the RL learning procedure such as e.g. managing the
trade-off between exploration and exploitation. How we apply architecture mutations in such cases differs slightly in single- and multi-agent settings.

.. seealso::

   :ref:`evolvable_networks` for a full guide on evolvable modules and architecture mutations.

Single-Agent
~~~~~~~~~~~~~
Architecture mutations in single-agent settings are straightforward because we can assume that the same base architecture is used in all the networks of an algorithm, allowing us to apply the
same mutation to all the networks (justified by the fact that these usually solve tasks of similar complexity and thus require `roughly` the same capacity). We can do this because
networks in RL typically all process observations into either actions or values. Even though the outputs of e.g. actors and critics differ, they will share the same type of encoder
and head (since the encoder processes the same observations and the head is always an instance of ``EvolvableMLP``) - which means they will share the same mutation methods.

Given this assumption, the procedure to perform an architecture mutation is as follows:

    1. Sample a mutation method for the policy network using :func:`EvolvableModule.sample_mutation_method() <agilerl.modules.base.EvolvableModule.sample_mutation_method>`

    2. Apply the same mutation to the rest of the evaluation networks found in the ``MutationRegistry`` e.g. the critic in ``PPO``.

    3. Reinitialize the networks that share parameters with the evaluation networks but aren't optimized directly during training (e.g. target networks) with the mutated architecture.


Multi-Agent
~~~~~~~~~~~
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

.. note::
    AgileRL currently doesn't support architecture mutations for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` objects.

.. _function_preserving:

Function-preserving additions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a mutation adds capacity to a network, we initialise the new units so that the network still
computes what it did before. There are two cases:

  * Widening (``add_node``, ``add_channel``, ``add_latent_node``) leaves the new units' incoming weights
    as the mutation created them and instead fades out the weights that carry them into the next layer.
  * Deepening (``add_layer``) initialises the inserted layer to the identity, so the signal passes
    through it untouched.

An agent therefore comes out of an architecture mutation with the same behaviour, and keeps its
place in the population, giving more chances for the added capacity to train.

When it applies
^^^^^^^^^^^^^^^

There is nothing to configure. We do this for every addition we can and fall back to the original random
initialisation when function preservation cannot be guaranteed. Widening is left alone when:

  * a normalisation layer sits between the widened layer and the layer that reads it. Its statistics are
    pooled over the whole layer, so the new units shift the existing ones however small their
    outgoing weights.
  * the activation mixes units together (``Softmax``, ``LogSoftmax``, ``Softmin`` or ``GumbelSoftmax``).
  * the widened layer sits inside a recurrent core, a multi-input encoder, a residual network or a SimBa
    block.

Deepening also needs an MLP layer whose activation is ReLU or Identity, since the identity initialisation
only holds for idempotent activation functions.

Growing the latent is the exception to the third point, because the surgery happens in the MLP head rather
than the encoder. Therefore, it works for recurrent, residual, SimBa and multi-input encoders alike; only a
normalisation layer or a unit-mixing activation on the encoder's output rules it out.

.. note::
    We fade the outgoing weights to small random values rather than exact zeros, which is what gets
    gradient flowing to the new units immediately, so the output is very slightly perturbed rather than identical.
    With noisy layers the guarantee only holds in evaluation mode: rebuilding a ``NoisyLinear`` resamples
    its noise buffers, which changes what every unit contributes during training, new or not.

RL Hyperparameter Mutations
---------------------------
Mutations on algorithm-specific hyperparameters can be configured through the ``hp_config`` argument of the algorithm. This is done by instantiating a
:class:`HyperparameterConfig <agilerl.algorithms.core.registry.HyperparameterConfig>` dataclass with the :class:`RLParameter <agilerl.algorithms.core.registry.RLParameter>`'s
you wish to mutate, which should be available as attributes of the algorithm (will raise an error if not). This configuration is automatically registered with the algorithms
``MutationRegistry`` and used by ``Mutations`` to perform mutations through the :func:`Mutations.rl_hyperparam_mutation() <agilerl.hpo.mutation.Mutations.rl_hyperparam_mutation>`
method. If we wanted to mutate the learning rate, batch size, and learning step in e.g. ``DQN``:

.. code-block:: python

    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

    # Override default mutation ranges for specific hyperparameters
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=32, max=256),
        learn_step=RLParameter(min=1, max=10, grow_factor=1.5, shrink_factor=0.75),
    )


Network Parameter Mutations
---------------------------
AgileRL allows mutations on the weights of the policy registered through
:func:`EvolvableAlgorithm.register_network_group() <agilerl.algorithms.core.base.EvolvableAlgorithm.register_network_group>`. Specifically, it selects
10% of the weights randomly to mutate (ignoring normalization layers) and applies a Gaussian noise with a standard deviation of ``mutation_sd`` to them, clamping
mutated values to prevent extreme changes. Each selected weight is affected by one of the following:

    - **Normal mutation** (95% of the selected weights): Adds noise with standard deviation proportional to the weight's own current value, scaled by ``mutation_sd``.

    - **Reset mutation** (5% of the selected weights): Completely replaces the weight with a fresh draw from a unit normal, discarding its trained value.

The split is fixed and unconditional — every parameter mutation applies both bands in this proportion.

.. _regrama:

ReGraMa: resetting dormant neurons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As training goes on, deep RL networks steadily lose *plasticity*: a growing fraction of their units stop
receiving any meaningful gradient. The network keeps its nominal size but its *effective* capacity
shrinks, and it becomes progressively worse at fitting anything new. Adding Gaussian noise does not fix
this: noise is applied to randomly chosen weights, with no idea which units have gone quiet.

ReGraMa measures dormancy with the **GraMa** score of Liu et al.,
`"Measure gradients, not activations!" <https://arxiv.org/abs/2505.24061>`_. A neuron's raw score is the
mean absolute gradient of the training loss with respect to its pre-activation, and that raw score is
divided by its own layer's mean. A neuron is *dormant* when its normalised score falls at or below the
threshold. Because the score is normalised within its layer, one setting works across layers and
architectures of very different scales.

Each dormant neuron gets fresh Xavier-uniform incoming weights, a zero bias, and a small, freshly drawn
set of outgoing weights. Its normalisation entry, if it has one, returns to the identity so a decayed gain
cannot immediately re-suppress it. The outgoing weights are deliberately small but non-zero: zeroing
them would leave the revived neuron with exactly zero gradient, so it would be potentially flagged dormant
again in the next evolution, especially if ``evo_steps`` is low.

Every evaluation network is treated this way (actors, critics, and each sub-policy of a multi-agent
algorithm) while target and other shared networks are re-synced from them afterwards. Output layers of
head networks are never reset: those units carry fixed meanings, such as action logits or a state
value, so re-initialising them would throw away the policy itself. Every parameter mutation runs
ReGraMa's resets first, and the Gaussian bands are applied afterwards.

ReGraMa's sensitivity is configured with one manifest field, on the ``mutation`` block:

.. code-block:: yaml

    mutation:
        dormant_threshold: 0.01         # default: 0.01

or, equivalently, in Python:

.. code-block:: python

    from agilerl.hpo.mutation import Mutations

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0,
        rl_hp=0.2,
        dormant_threshold=0.01,
    )

``dormant_threshold`` must be greater than or equal to ``0.0``.

Raising ``dormant_threshold`` resets more units per generation: dormancy is given more importance, but
the forgetting can happen more aggressively, since each reset discards whatever the unit had learned.
Lowering it towards ``0.0`` resets only units whose gradient is exactly zero, which is a reasonable
conservative setting for ReLU networks but degenerate for smooth activations such as Tanh, where gradients
get very small without ever reaching zero.

.. note::
    RNN architectures fall outside what ReGraMa can reset. The hidden units of a recurrent core
    have fused gate non-linearities and no single weight matrix whose rows are one unit's incoming weights,
    so only the layers from the output projection onward are reset.
