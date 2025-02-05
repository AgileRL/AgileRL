.. _custom_network_architectures:

Creating Custom Networks
========================

Creating Evolvable Networks from Scratch
----------------------------------------

In the implemented algorithms, we allow users to pass architecture configurations for common RL networks (made evolvable)
that are available with the framework. We employ a similar philosophy to PyTorch in our way of processing the nested structure
of complex custom architectures to keep track of the available architecture mutation methods in a neural network.

1. :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>`: This is the base class to define a custom module in AgileRL. It is a wrapper around PyTorch's
``nn.Module`` class, allowing users to create complex networks with nested :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>`'s that have different available
architecture mutations. We provide e.g. ``EvolvableMLP``, ``EvolvableCNN``, and ``EvolvableMultiInput`` modules to process
vector, image, and dictionary / tuple observations, respectively, into a desired number of outputs. Modules in AgileRL
each have specific mutation methods (wrapped by a ``@mutation`` decorator to signal their nature) that allow us to dynamically
change their architectures during training. :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` objects should implement a
:meth:`recreate_network() <agilerl.modules.base.EvolvableModule.recreate_network>` method that recreates the network with the new architecture
after a mutation method is applied. This method is called automatically after calling a method wrapped by the ``@mutation`` decorator.

2. :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>`: Abstracting neural networks for RL problems is hard because different observation spaces require different
architectures. To address this, we provide a simple way of defining custom evolvable networks for RL algorithms through the
:class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>` class, which inherits from :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>`.
Under the hood, any network inheriting from :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>`  automatically creates an appropriate encoder from the passed observation space. Custom networks only need to
specify a head that acts as a mapping from the latent space to a number of outputs (e.g. actions). AgileRL provides a variety of
common networks used in RL algorithms:

   -  ``QNetwork``: State-action value function (used in e.g. DQN).
   -  ``RainbowQNetwork``: State-action value function that uses a dueling distributional architecture for the network head (used in Rainbow DQN).
   -  ``ContinuousQNetwork``: State-action value function for continuous action spaces, which takes the actions as input with the observations.
   -  ``ValueNetwork``: Outputs the scalar value of an observation (used in e.g. PPO).
   -  ``DeterministicActor``: Outputs deterministic actions given an action space.
   -  ``StochasticActor``: Outputs an appropriate PyTorch distribution over the given action space.

.. note::
  We impose that the different evolvable networks in an algorithm (e.g. actor and critic in PPO) share the same mutation methods. This
  is done because we apply the same architecture mutations to all of the networks of an individual to reduce variance during training.
  For this reason, we only allow mutation methods in :class:`EvolvableModule <agilerl.networks.base.EvolvableNetwork>` objects to come from the encoder and the head, assuming the same
  modules are used in both. All of the implemented networks in AgileRL follow this structure.

For simple use cases, it might be appropriate to create a network using ``EvolvableMLP`` or ``EvolvableCNN`` directly (depending on your
environments observation space), and passing it in to the desired algorithm as the ``actor_network`` or ``critic_network`` argument.

Please refer to the `RainbowQNetwork <Custom_networks_tutorials>`_ tutorial for an example of how to build a custom network using AgileRL.

3. :class:`DummyEvolvable <agilerl.modules.dummy.DummyEvolvable>`: This is a wrapper that allows users to load a pre-trained model that is not an
:class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` and use it in an evolvable AgileRL algorithm. This disables architecture mutations but still
allows for RL hyperparameter and weight mutations.

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn

    from sgilerl.algorithms import DQN
    from agilerl.modules.dummy import DummyEvolvable

    class BasicNetActorDQN(nn.Module):
      def __init__(self, input_size, hidden_sizes, output_size):
          super().__init__()
          layers = []

          # Add input layer
          layers.append(nn.Linear(input_size, hidden_sizes[0]))
          layers.append(nn.ReLU())  # Activation function

          # Add hidden layers
          for i in range(len(hidden_sizes) - 1):
              layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
              layers.append(nn.ReLU())  # Activation function

          # Add output layer with a sigmoid activation
          layers.append(nn.Linear(hidden_sizes[-1], output_size))

          # Combine all layers into a sequential model
          self.model = nn.Sequential(*layers)

      def forward(self, x):
          return self.model(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_kwargs = {
        "input_size": 4,  # Input size
        "hidden_sizes": [64, 64],  # Hidden layer sizes
        "output_size": 2  # Output size
    }

    actor = DummyEvolvable(BasicNetActor, actor_kwargs, device=device)

    # Use the actor in an algorithm
    observation_space = ...
    action_space = ...
    population = DQN.population(
        size=4,
        observation_space=observation_space,
        action_space=action_space
        actor_network=actor
        )

.. _createcustnet:

Integrating Architecture Mutations Into a Custom PyTorch Network
----------------------------------------------------------------

.. warning::
  The following section pertains to the :class:`MakeEvolvable <agilerl.wrappers.make_evolvable.MakeEvolvable>` wrapper, which will be deprecated in a
  future release. We recommend using the :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` and :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>`
  classes to create custom networks, or wrapping your ``nn.Module`` objects with :class:`DummyEvolvable <agilerl.modules.dummy.DummyEvolvable>`.

For sequential architectures that users have already implemented using PyTorch, it is also possible to add
evolvable functionality through the ``MakeEvolvable`` wrapper. Below is an example of a simple multi-layer
perceptron that can be used by a DQN agent to solve the Lunar Lander environment. The input size is set as
the state dimensions and output size the action dimensions. It's worth noting that, during the model definition,
it is imperative to employ the ``torch.nn`` module to define all layers instead of relying on functions from
``torch.nn.functional`` within the forward() method of the network. This is crucial as the forward hooks implemented
will only be able to detect layers derived from ``nn.Module``.

.. code-block:: python

    import torch.nn as nn
    import torch


    class MLPActor(nn.Module):
        def __init__(self, input_size, output_size):
            super(MLPActor, self).__init__()

            self.linear_layer_1 = nn.Linear(input_size, 64)
            self.linear_layer_2 = nn.Linear(64, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.linear_layer_1(x))
            x = self.linear_layer_2(x)
            return x


To make this network evolvable, simply instantiate an MLP Actor object and then pass it, along with an input tensor into
the ``MakeEvolvable`` wrapper.

.. code-block:: python

    from agilerl.wrappers.make_evolvable import MakeEvolvable

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    actor = MLPActor(observation_space.shape[0], action_space.n)
    evolvable_actor = MakeEvolvable(
                        actor,
                        input_tensor=torch.randn(observation_space.shape[0]),
                        device=device
                      )

When instantiating using ``create_population`` to generate a population of agents with a custom actor,
you need to set ``actor_network`` to ``evolvable_actor``.

.. code-block:: python

    pop = create_population(
            algo="DQN",                                  # Algorithm
            observation_space=observation_space,         # Observation space
            action_space=action_space,                   # Action space
            actor_network=evolvable_actor,               # Custom evolvable actor
            INIT_HP=INIT_HP,                             # Initial hyperparameters
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            device=device
          )

If you are using an algorithm that also uses a single critic (PPO, DDPG), define the critic network and pass it into the
``create_population`` class.

.. code-block:: python

    pop = create_population(
            algo="PPO",                                  # Algorithm
            observation_space=observation_space,         # Observation space
            action_space=action_space,                   # Action space
            actor_network=evolvable_actor,               # Custom evolvable actor
            critic_network=evolvable_critic,             # Custom evolvable critic
            INIT_HP=INIT_HP,                             # Initial hyperparameters
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            device=device
          )

If the single agent algorithm has more than one critic (e.g. TD3), then pass the ``critic_network`` argument a list of two critics.

.. code-block:: python

    pop = create_population(
            algo="TD3",                                           # Algorithm
            observation_space=observation_space,                      # Observation space
            action_space=action_space,                                # Action space
            actor_network=evolvable_actor,                            # Custom evolvable actor
            critic_network=[evolvable_critic_1, evolvable_critic_2],  # Custom evolvable critic
            INIT_HP=INIT_HP,                                          # Initial hyperparameters
            population_size=INIT_HP["POPULATION_SIZE"],               # Population size
            device=device
          )


If you are using a multi-agent algorithm, define ``actor_network`` and ``critic_network`` as lists containing networks for each agent in the
multi-agent environment. The example below outlines how this would work for a two agent environment (asumming you have initialised a multi-agent
environment in the variable ``env``).

.. code-block:: python

    # For MADDPG
    evolvable_actors = [actor_network_1, actor_network_2]
    evolvable_critics = [critic_network_1, critic_network_2]

    # For MATD3, "critics" will be a list of 2 lists as MATD3 uses one more critic than MADDPG
    evolvable_actors = [actor_network_1, actor_network_2]
    evolvable_critics = [[critic_1_network_1, critic_1_network_2],
                         [critic_2_network_1, critic_2_network_2]]

    # Instantiate the populations as follows
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    pop = create_population(
            algo="MADDPG",                                # Algorithm
            observation_space=observation_spaces,         # Observation space
            action_space=action_spaces,                   # Action space
            actor_network=evolvable_actors,               # Custom evolvable actor
            critic_network=evolvable_critics,             # Custom evolvable critic
            INIT_HP=INIT_HP,                              # Initial hyperparameters
            population_size=INIT_HP["POPULATION_SIZE"],   # Population size
            device=device
          )

Finally, if you are using a multi-agent algorithm but need to use CNNs to account for RGB image states, there are a few extra considerations
that need to be taken into account when defining your critic network. In MADDPG and MATD3, each agent consists of an actor and critic and each
critic evaluates the states and actions of all agents that act in the multi-agent system. Unlike with non-RGB environments that require MLPs, we cannot
immediately stack the state and action tensors due to differing dimensions, we must first pass the state tensor through the convolutinal layers,
before flattening the output, combining with the actions tensor, and then passing this combined state-action tensor into the fully-connected layer.
This means that when defining the critic, the ``.forward()`` method must account for two input tensors (states and actions). Below are examples of
how to define actor and critic networks for a two agent system with state tensors of shape (4, 210, 160):

.. code-block:: python

  from agilerl.networks.custom_activation import GumbelSoftmax

  class MultiAgentCNNActor(nn.Module):
    def __init__(self):
    super().__init__()
      self.conv1 = nn.Conv3d(
         in_channels=4, out_channels=16, kernel_size=(1, 3, 3), stride=4
      )
      self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
      )
      # Define the max-pooling layers
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      # Define fully connected layers
      self.fc1 = nn.Linear(15200, 256)
      self.fc2 = nn.Linear(256, 2)

      # Define activation function
      self.relu = nn.ReLU()

      # Define output activation
      self.output_activation = GumbelSoftmax()

    def forward(self, state_tensor):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.output_activation(self.fc2(x))

        return x


  class MultiAgentCNNCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=4, out_channels=16, kernel_size=(2, 3, 3), stride=4
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
        )

        # Define the max-pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define fully connected layers
        self.fc1 = nn.Linear(15208, 256)
        self.fc2 = nn.Linear(256, 2)

        # Define activation function
        self.relu = nn.ReLU()


    def forward(self, state_tensor, action_tensor):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action_tensor], dim=1)

        # Forward pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

To then make these two CNNs evolvable we pass them, along with input tensors into the ``MakeEvolvable`` wrapper.

.. code-block:: python

  actor = MultiAgentCNNActor()
  evolvable_actor = MakeEvolvable(network=actor,
                                  input_tensor=torch.randn(1, 4, 1, 210, 160), # (B, C_in, D, H, W) D = 1 as actors are decentralised
                                  device=device)
  critic = MultiAgentCNNCritic()
  evolvable_critic = MakeEvolvable(network=critic,
                                   input_tensor=torch.randn(1, 4, 2, 210, 160), # (B, C_in, D, H, W)),
                                                                                #  D = 2 as critics are centralised and  so we evaluate both agents
                                   secondary_input_tensor=torch.randn(1,8), # Assuming 2 agents each with action dimensions of 4
                                   device=device)


.. _comparch:

Compatible Architecture
~~~~~~~~~~~~~~~~~~~~~~~

At present, ``MakeEvolvable`` is currently compatible with PyTorch multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs). The
network architecture must also be sequential, that is, the output of one layer serves as the input to the next layer. Outlined below is a comprehensive
table of PyTorch layers that are currently supported by this wrapper:


.. list-table::
   :widths: 25, 50
   :header-rows: 1
   :align: left

   * - **Layer Type**
     - **PyTorch Compatibility**
   * - **Pooling**
     - ``nn.MaxPool2d``, ``nn.MaxPool3d``, ``nn.AvgPool2d``, ``nn.AvgPool3d``
   * - **Activation**
     - ``nn.Tanh``, ``nn.Identity``, ``nn.ReLU``, ``nn.ELU``, ``nn.Softsign``, ``nn.Sigmoid``, ``GumbelSoftmax``,
       ``nn.Softplus``, ``nn.Softmax``, ``nn.LeakyReLU``, ``nn.PReLU``, ``nn.GELU``
   * - **Normalization**
     - ``nn.BatchNorm2d``, ``nn.BatchNorm3d``, ``nn.InstanceNorm2d``, ``nn.InstanceNorm3d``, ``nn.LayerNorm``
   * - **Convolutional**
     - ``nn.Conv2d``, ``nn.Conv3d``
   * - **Linear**
     - ``nn.Linear``

.. _compalgos:

Compatible Algorithms
~~~~~~~~~~~~~~~~~~~~~

The following table highlights which AgileRL algorithms are currently compatible with custom architecture:

.. list-table::
   :widths: 5, 5, 5, 5, 5, 5, 5, 5, 5
   :header-rows: 1

   * - CQL
     - DQN
     - DDPG
     - TD3
     - PPO
     - MADDPG
     - MATD3
     - ILQL
     - Rainbow-DQN
   * - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ✔️
     - ❌
     - ✔️
