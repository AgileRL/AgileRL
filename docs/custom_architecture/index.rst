Using Custom Architecture
=========================

In all other AgileRL tutorials, the networks used by the agents have been consistently defined with a network configuration
dictionary. Alternatively, it is also possible for a user to define their own custom architecture and add evolvable
functionality through the ``MakeEvolvable`` wrapper.

.. _createcustnet:

Creating a Custom Evolvable Network
-----------------------------------

To create a custom evolvable network, firstly you need to define your network class, ensuring correct input and output
dimensions. Below is an example of a simple multi-layer perceptron that can be used by a DQN agent to solve the Lunar
Lander environment. The input size is set as the state dimensions and output size the action dimensions. It's worth noting
that, during the model definition, it is imperative to employ the ``torch.nn`` module to define all layers instead
of relying on functions from ``torch.nn.functional`` within the forward() method of the network. This is crucial as the
forward hooks implemented will only be able to detect layers derived from ``nn.Module``.

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

    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    actor = MLPActor(state_dim[0], action_dim)
    evolvable_actor = MakeEvolvable(actor,
                                    input_tensor=torch.randn(state_dim[0]),
                                    device=device)

There are two further considerations to make when defining custom architecture. The first is when instantiating the
``initialPopulation`` object, you need to set ``net_config`` to ``None`` and ``actor_network`` to ``evolvable_actor``.

.. code-block:: python

    pop = initialPopulation(algo="DQN",  # Algorithm
                            state_dim=state_dim,  # State dimension
                            action_dim=action_dim,  # Action dimension
                            one_hot=one_hot,  # One-hot encoding
                            net_config=None,  # Network configuration set as None
                            actor_network=evolvable_actor, # Custom evolvable actor
                            INIT_HP=INIT_HP,  # Initial hyperparameters
                            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
                            device=device)

If you are using an algorithm that also uses a single critic (PPO, DDPG), define the critic network and pass it into the
``initialPopulation`` class, again setting ``net_config`` to ``None``.

.. code-block:: python

    pop = initialPopulation(algo="DDPG",  # Algorithm
                            state_dim=state_dim,  # State dimension
                            action_dim=action_dim,  # Action dimension
                            one_hot=one_hot,  # One-hot encoding
                            net_config=None,  # Network configuration set as None
                            actor_network=evolvable_actor, # Custom evolvable actor
                            critic_network=evolvable_critic, # Custom evolvable critic
                            INIT_HP=INIT_HP,  # Initial hyperparameters
                            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
                            device=device)

If the single agent algorithm has more than one critic (e.g. TD3), then pass the ``critic_network`` argument a list of two critics.

.. code-block:: python

    pop = initialPopulation(algo="TD3",  # Algorithm
                            state_dim=state_dim,  # State dimension
                            action_dim=action_dim,  # Action dimension
                            one_hot=one_hot,  # One-hot encoding
                            net_config=None,  # Network configuration set as None
                            actor_network=evolvable_actor, # Custom evolvable actor
                            critic_network=[evolvable_critic_1, evolvable_critic_2], # Custom evolvable critic
                            INIT_HP=INIT_HP,  # Initial hyperparameters
                            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
                            device=device)

If you are using a multi-agent algorithm, define ``actor_network`` and ``critic_network`` as lists containing networks for each agent in the
multi-agent environment. The example below outlines how this would work for a two agent environment.

.. code-block:: python

    # For MADDPG
    evolvable_actors = [actor_network_1, actor_network_2]
    evolvable_critics = [critic_network_1, critic_network_2]

    # For MATD3, "critics" will be a list of 2 lists as MATD3 uses one more critic than MADDPG
    evolvable_actors = [actor_network_1, actor_network_2]
    evolvable_critics = [[critic_1_network_1, critic_1_network_2],
                         [critic_2_network_1, critic_2_network_2]]

    # Instantiate the populations as follows
    pop = initialPopulation(algo="MADDPG",  # Algorithm
                            state_dim=state_dim,  # State dimensions
                            action_dim=action_dim,  # Action dimensions
                            one_hot=one_hot,  # One-hot encoding
                            net_config=None,  # Network configuration se t as None
                            actor_network=evolvable_actors, # Custom evolvable actor
                            critic_network=evolvable_critics, # Custom evolvable critic
                            INIT_HP=INIT_HP,  # Initial hyperparameters
                            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
                            device=device)

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


The only other consideration that needs to be made is when instantiating the ``Mutations`` class. The ``arch`` argument should be set
as ``evolvable_actor.arch`` for single agent algorithms or ``evolvable_actors[0].arch`` for multi-agent algorithms.

.. _comparch:

Compatible Architecture
-----------------------

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
---------------------

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
