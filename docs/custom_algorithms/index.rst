Creating Custom Algorithms
==========================

To create a custom algorithm, you must inherit from :class:`RLAlgorithm <agilerl.algorithms.core.base.RLAlgorithm>` for
single-agent algorithms or :class:`MultiAgentAlgorithm <agilerl.algorithms.core.base.MultiAgentAlgorithm>` for multi-agent
algorithms. For an overview of the class hierarchy and the philosophy behind it please refer to :ref:`base_algorithm`. We have implemented
this hierarchy with the idea of making evolutionary hyperparameter optimization as seamless as possible, and have users focus on their
implementation only. The key components in developing a custom AgileRL algorithm are the following:

Network Groups
--------------

Users must specify the "network groups" in their algorithm. A network group is a group of networks that work hand in hand with a common objective,
and is registered through a :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>` object, which contains at least one
**evaluation** network (i.e. a network that is optimized during training e.g. the Q-network in DQN) and, optionally, "shared" networks that share
parameters with the evaluation network in the group but arent optimized during training directly (e.g. the target network in DQN). An RL algorithm
must also contain one :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>` corresponding to the policy (i.e. the network used to
select actions), signalled by the ``policy`` attribute in the group.

Example
~~~~~~~

In PPO, we would need to define two network groups, since there are two different networks that are optimized during training. The first network group
corresponds to the actor network and the second to the critic network. The actor network is responsible for selecting actions, and should therefore be signalled
as the policy through the ``policy`` argument of :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>`. In this case, there are no networks that
share parameters with the actor or the critic so we can bypass the ``shared`` argument. We can register these groups as follows through the ``register_network_group``
method of the algorithm:

.. code-block:: python

    # Register network groups for mutations
    self.register_network_group(
        NetworkGroup(
            eval=self.actor,
            policy=True
        )
    )
    self.register_network_group(
        NetworkGroup(
            eval=self.critic
        )
    )

OptimizerWrapper
----------------

The last thing users should do when creating a custom algorithm is wrap their optimizers in an :class:`OptimizerWrapper <agilerl.algorithms.core.wrappers.OptimizerWrapper>`,
specifying the networks that the optimizer is responsible for optimizing. Since we are mutating network architectures during training, we need to have knowledge of
this in order to reinitiliaze the optimizers correctly when we do so. In the above example, we have a single optimizer that optimizes the parameters of both the actor and critic networks,
so we can wrap it as follows:

.. code-block:: python

    self.optimizer = OptimizerWrapper(
        optim.Adam,
        networks=[self.actor, self.critic],
        lr=self.lr
    )

.. note::
    All of the network groups and optimizers of an algorithm should by convention all be defined in the ``__init__`` method of the algorithm.

Finally, users only need to implement the following methods to train agents with the AgileRL framework:

1. :meth:`learn() <agilerl.algorithms.core.base.EvolvableAlgorithm.learn>`: Responsible for updating the parameters of the networks and the optimizer after collecting
a set of experiences from the environment.

2. :meth:`get_action() <agilerl.algorithms.core.base.EvolvableAlgorithm.get_action>`: Select action/s from a given observation or batch of observations.

3. :meth:`test() <agilerl.algorithms.core.base.EvolvableAlgorithm.test>`: Test the agent in the environment without updating the parameters of the networks.
