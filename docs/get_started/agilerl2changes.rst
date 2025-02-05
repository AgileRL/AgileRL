.. _agilerl2changes:

AgileRL 2.0 Release Notes
=========================

This document outlines the new features and updates in AgileRL 2.0. The main focus of this release is to provide a more flexible framework 
for creating custom evolvable network architectures and algorithms to make the most out of automatic evolutionary hyperparameter optimization 
during training. We've also done some heavy refactoring to make the codebase more modular and scalable, with the hope that users find it easier 
to plug-and-play with their arbitrarily complex use-cases.

**Features**:
-------------

- **Support for Dictionary / Tuple Spaces**: We have implemented the :class:`EvolvableMultiInput <agilerl.modules.multi_input.EvolvableMultiInput>` module, which takes in a (single-level) dictionary or tuple space and assigns an :class:`EvolvableCNN <agilerl.modules.cnn.EvolvableCNN>` to each underlying image subspace. Observations from vector / discrete spaces are simply concatenated to the image encodings by default, but users can specify if they want these to be processed by an :class:`EvolvableMLP <agilerl.modules.mlp.EvolvableMLP>` before concatenating.


- **EvolvableModule Class Hierarchy**: A wrapper around ``nn.Module`` that allows us to keep track of the mutation methods in complex networks with nested modules. We use the ``@mutation`` decorator to signal mutation methods and these are registered automatically as such. Such modules should implement a :meth:`recreate_network() <agilerl.modules.base.EvolvableModule.recreate_network>` method that is called automatically after any mutation method is used to modify the network's architecture.

.. note:: 
    Users can now pass in non-evolvable architectures to the algorithms too by wrapping their models with :class:`DummyEvolvable <agilerl.modules.dummy.DummyEvolvable>`. 
    This is useful when you want to use a pre-trained model or a model whose architecture you don't want to mutate, while still enabling random weight and RL hyperparameter mutations. 
    Please refer to :ref:`custom_network_architectures` for more information.

- **EvolvableNetwork Class Hierarchy**: Towards a more general API for algorithm implementation, where complex observation spaces should be inherently supported, networks inheriting from :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>` automatically create an appropriate encoder from a given observation space. Custom networks simply have to specify the head to the network that maps the observation encodings to a number of outputs. As part of this update we implement the following common networks used (by default) in the already implemented algorithms.

    - :class:`QNetwork <agilerl.networks.q_networks.QNetwork>`: State-action value function (used in e.g. DQN).
    - :class:`RainbowQNetwork <agilerl.networks.q_networks.RainbowQNetwork>`: State-action value function that uses a dueling distributional architecture for the network head (used in Rainbow DQN).
    - :class:`ContinuousQNetwork <agilerl.networks.q_networks.ContinuousQNetwork>`: State-action value function for continuous action spaces, which takes the actions as input with the observations.
    - :class:`ValueNetwork <agilerl.networks.value_networks.ValueNetwork>`: Outputs the scalar value of an observation (used in e.g. PPO).
    - :class:`DeterministicActor <agilerl.networks.actors.DeterministicActor>`: Outputs deterministic actions given an action space.
    - :class:`StochasticActor <agilerl.networks.actors.StochasticActor>`: Outputs an appropriate PyTorch distribution over the given action space.

- **EvolvableAlgorithm Class Hierarchy**: We create a class hierarchy for algorithms with a focus on evolutionary hyperparameter optimization. The EvolvableAlgorithm base class implements common methods across any RL algorithm e.g. ``save_checkpoint()``, ``load()``, but also methods pertaining specifically to mutations e.g. ``clone()``. Under-the-hood, it initializes a :class:`MutationRegistry <agilerl.algorithms.core.registry.MutationRegistry>` that users should use to register "network groups". The registry also keeps track of the RL hyperparameters users wish to mutate during training and the optimizers. Users wishing to create custom algorithms should now only need to worry about implementing ``get_action()``, ``learn()``, and (for now) ``test()`` methods.

- **Generalized Mutations**: We have refactored :class:`Mutations <agilerl.hpo.mutation.Mutations>` with the above hierarchies in mind to allow for a generalised mutations framework that works for any combination of evolvable networks in an algorithm. Moreover, we now allow users to pass in any configuration of RL hyperparameters they wish to mutate during training directly to an algorithm inheriting from ``EvolvableAlgorithm``, rather than handling this in ``Mutations``. For an example of how to do this, please refer to the documentation of any of the algorithms implemented in AgileRL, or our tutorials. 

**Breaking Changes**:
----------------------

- We have placed the building blocks of our networks in a dedicated :mod:`agilerl.modules` module, which contains the off-the-shelf evolvable modules that can be used to create custom network architectures (e.g. :class:`EvolvableMLP <agilerl.modules.mlp.EvolvableMLP>`, :class:`EvolvableCNN <agilerl.modules.cnn.EvolvableCNN>`, and :class:`EvolvableMultiInput <agilerl.modules.multi_input.EvolvableMultiInput>`), whereas before these were located in :mod:`agilerl.networks`. In the latter we now keep networks created through the :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>` class hierarchy.

- Pass in ``observation_space`` and ``action_space`` to the algorithms instead of ``state_dim`` and ``action_dim``. This is to support more complex observation spaces, and allow for a simpler generation of default networks in the algorithms by using the ``EvolvableNetwork`` class hierarchy.

- Simplified API in the evolvable modules, mutations, and algorithms. Please refer to the documentation for more information.

- ``net_config`` argument of algorithms should now be passed in with the arguments of the corresponding ``EvolvableNetwork`` class. For example, in :class:`PPO <agilerl.algorithms.ppo.PPO>`, the ``net_config`` argument might include an "encoder_config" key which is different depending on your observation space, and a "head_config" key for the head of the actor (i.e. ``StochasticActor``) and critic (i.e. ``ValueNetwork``). All the networks in an algorithm are initialized with the same architecture by default. If users with to use different architectures, these should be passed as arguments directly to the algorithm.   

Example Network Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    net_config = {
        # For an image observation space we encode observations using EvolvableCNN
        "encoder_config": {
            "channel_size": [32],
            "kernel_size": [3],
            "stride_size": [1],
        }

        # The head is usually an EvolvableMLP by default
        "head_config": {
            "hidden_size": [64, 64],
        }

    }