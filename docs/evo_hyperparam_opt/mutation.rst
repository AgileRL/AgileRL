.. _mutations:

Mutation
========

Mutations are periodically applied to our population of agents to explore the hyperparameter space, allowing different hyperparameter combinations to be trialled during training.
If certain hyperparameters prove relatively beneficial to training, then that agent is more likely to be preserved in the next generation, and so those characteristics are more
likely to remain in the population.

The :class:`Mutations <agilerl.hpo.mutation.Mutations>` class is used to mutate agents with pre-set probabilities. The available mutations currently implemented are:

    * **No mutation**: An "identity" mutation, whereby the agent is returned unchanged.
    * **Network architecture mutations**: Involves adding or removing layers or nodes. Trained weights are reused, and added capacity is initialized to preserve the network's function where the architecture allows it (see :ref:`function_preserving`), and randomly otherwise.
    * **Network parameters mutation**: Mutating weights with Gaussian noise.
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

When an architecture mutation widens or deepens a network, the trained weights are carried over but the
new capacity is initialised randomly, so the network's output changes the moment it is added. The agent
is then evaluated as a *different* policy from the one that earned its place in the population, its
fitness drops, and selection usually discards it before the extra capacity has been trained into
anything useful. That variance is why architecture mutations are the most disruptive operator in the
loop, and why layer mutations are disabled on the encoder by default.

AgileRL initialises additions to be **function-preserving** wherever the architecture allows it,
following Chen et al., `"Net2Net: Accelerating Learning via Knowledge Transfer"
<https://arxiv.org/abs/1511.05641>`_. Two mechanisms cover the additive mutations:

  * **Widening** (``add_node``, ``add_channel``, ``add_latent_node``) keeps the new units' incoming
    weights and fades the *outgoing* weights that carry them into the next layer. The next layer's
    output is therefore unchanged whatever the activation does, while the new units still receive
    gradient and start contributing as soon as training resumes.
  * **Deepening** (``add_layer``) initialises the inserted layer to the identity, which leaves the
    network unchanged for ReLU and Identity activations.

This makes architecture mutations markedly **less aggressive**. A widened or deepened agent keeps the
fitness it had, so it survives selection on its merits rather than on the accident of its new weights,
and the population can accumulate depth and width over many generations instead of repeatedly
discarding it.

When it applies
^^^^^^^^^^^^^^^

There is nothing to configure: preservation is applied automatically, per mutation, whenever the
architecture supports it. Widening stands down when

  * a **normalisation** layer (``LayerNorm``, ``BatchNorm`` or ``GroupNorm``) sits between the widened
    units and their activation, since it re-scales them using statistics pooled over the whole layer and
    so moves every existing unit however small the new fan-out is;
  * the activation **mixes units together** (``Softmax``, ``LogSoftmax``, ``Softmin`` or
    ``GumbelSoftmax``);
  * the widened layer sits **inside** a **recurrent** core, a **multi-input** encoder, a **residual**
    network or a **SimBa** block, none of which expose a single consuming layer whose columns can be
    faded.

Deepening additionally requires an **MLP** stack whose activation is **ReLU** or **Identity**.

Growing the **latent** is the exception to that last bullet. It is surgery on the head alone -- the
encoder's new output rows are appended at the tail, so every existing latent coordinate survives, and
only the head's new input columns are rewritten. What the encoder is built from therefore does not
matter, and ``add_latent_node`` is preserved for recurrent, residual, SimBa and multi-input encoders
alike; only a normalisation or unit-mixing activation on the encoder's *output* stands it down. The one
case that cannot be preserved is a multi-input encoder asked to widen **its own** latent rather than the
network's, which resizes every sub-encoder's output and so interleaves the new columns through the
fusion layer's input instead of appending them.

Whenever preservation stands down, the mutation simply falls back to AgileRL's original behaviour and the
new capacity is initialised randomly, exactly as before. A warning is emitted once per reason so a
configuration that never gets preservation is never silent about it. **Removals are unchanged**: only
additions are preserved, so the two regimes differ purely in how capacity is added.

.. note::
    The default network configurations set ``layer_norm: true``, which blocks preservation. Set
    ``layer_norm: false`` on the ``encoder_config`` and ``head_config`` blocks to enable it; the shipped
    ``*_func_preserving.yaml`` manifests do exactly that. Setting it on an MLP ``encoder_config`` also
    clears the encoder's output normalisation, which is what ``add_latent_node`` needs.

.. note::
    ``RainbowDQN`` forces normalisation on its encoder and drops ``layer_norm`` from its head
    configuration, so its additions are never preserved.

.. note::
    Widening applies a small symmetry-breaking perturbation to the new outgoing weights, without which
    the new units would share a single gradient and could never differentiate. Preservation is therefore
    near-exact rather than bit-exact. For noisy layers the guarantee is an evaluation-mode one: a
    rebuilt ``NoisyLinear`` resamples its noise buffers, which moves the training-mode output of every
    unit, new or not.

.. note::
    The ReLU-or-Identity requirement is on the activation the network carries **when the layer is
    inserted**. An activation mutation sampled in a later generation is an ordinary unpreserved mutation,
    judged by selection like any other, and it never catches the inserted layer as an identity: mutation
    closes a training cycle, so the layer trains before another mutation can reach it. Running
    ``act_mut`` alongside ``new_layer`` is therefore supported: it only means that depth accumulated
    cheaply under ReLU is composed through whatever activation replaces it, and that later deepenings
    decline while the network sits on a non-ReLU activation.

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
10% of the weights randomly to mutate (ignoring normalization layers) and applies a Gaussian noise with a standard deviation of ``mutation_sd`` to them. It does so
in three different ways, clamping mutated values to prevent extreme changes:

    - **Normal mutation**: Adds noise with standard deviation proportional to current weight values.

    - **Super mutation**: Adds larger noise for more significant changes.

    - **Reset mutation**: Completely resets weights to new random values.
