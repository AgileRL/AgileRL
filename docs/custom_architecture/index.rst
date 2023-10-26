Using Custom Architecture
=====

In all other AgileRL tutorials, the networks used by the agents have been consistently defined with a network configuration
dictionary. Alternatively, it is also possible for a user to define their own custom architecture and add evolvable
functionality through the `MakeEvolvable` wrapper.

.. _createcustnet:
Creating a Custom Evolvable Network
------------

To create a custom evolvable network, firstly you need to define your network class, ensuring correct input and output
dimensions. Below is an example of a simple multi-layer perceptron that can be used by a DQN agent to solve the Lunar
Lander environment. The input size is set as the state dimensions and output size the action dimensions. It's worth noting
that, during the model definition, it is imperative to employ the `torch.nn`` module to define all layers instead
of relying on functions from `torch.nn.functional` within the forward() method of the network. This is crucial as the
forward hooks implemented will only be able to detect layers derived from `nn.Module`.

.. code-block:: python
    import torch.nn as nn
    import torch


    class MLPActor(nn.Module):
        def __init__(self, input_size, output_size):
            super(MLPActor, self).__init__()

            self.linear_layer_1 = nn.Linear(input_size, 32)
            self.linear_layer_2 = nn.Linear(32, output_size)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.relu(self.linear_layer_1(x))
            x = self.softmax(self.linear_layer_2(x))
            return x


To make this network evolvable, simply instantiate an MLP Actor object and then pass it, along with an input tensor into
the `MakeEvolvable` wrapper.

.. code-block:: python
    from agilerl.wrappers.make_evolvable import MakeEvolvable

    state_dim = env.single_observation_space.shape
    action_dim = env.single_action_space.n

    actor = MLPActor(state_dim[0], action_dim)
    evolvable_actor = MakeEvolvable(actor,
                                    input_tensor=torch.randn(state_dim[0]),
                                    device=device)

There are two further considerations to make when defining custom architecture. The first is when instantiating the
`initialPopulation` object, you need to set `net_config` to `None` and `actor_network` to `evolvable_actor`.

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
`initialPopulation` class, again setting `net_config` to `None`.

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

If the single agent algorithm has more than one critic (e.g. TD3), then pass the `critic_network` argument a list of two critics. An example
is shown below.

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

Finally, if you are using a multi-agent algorithm, define `actor_network` and `critic_network` as lists containing networks
for each agent in the multi-agent environment. The example below outlines how this would work for a two agent environment.

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
                            net_config=None,  # Network configuration set as None
                            actor_network=evolvable_actors, # Custom evolvable actor
                            critic_network=evolvable_critics, # Custom evolvable critic
                            INIT_HP=INIT_HP,  # Initial hyperparameters
                            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
                            device=device)

The only other consideration that needs to be made is when instantiating the `Mutations` class. The `arch` argument should be set
as `evolvable_actor.arch` for single agent algorithms or `evolvable_actors[0].arch` for multi-agent algorithms.

.. _comparch:

Compatible Architecture
------------

At present, `MakeEvolvable` is currently compatible with PyTorch multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs). The
network architecture must also be sequential, that is, the output of one layer serves as the input to the next layer. Outlined below is a comprehensive
table of PyTorch layers that are currently supported by this wrapper.


.. list-table::
   :widths: 25, 50
   :header-rows: 1

   * - **Layer Type**
     - **PyTorch Compatibility**
   * - **Pooling**
     - nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool2d, nn.AvgPool3d
   * - **Activation**
     - nn.Tanh, nn.Identity, nn.ReLU, nn.ELU, nn.Softsign, nn.Sigmoid, GumbelSoftmax,
       nn.Softplus, nn.Softmax, nn.LeakyReLU, nn.PReLU, nn.GELU
   * - **Normalization**
     - nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm
   * - **Convolutional**
     - nn.Conv2d, nn.Conv3d
   * - **Linear**
     - nn.Linear

.. _compalgos:

Compatible Algorithms
------------

The following table highlights which AgileRL algorithms are compatible with custom architecture:

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
     - ❌
