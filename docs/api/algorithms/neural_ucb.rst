.. _neural_ucb:

Neural Contextual Bandits with UCB-based Exploration (NeuralUCB)
================================================================

NeuralUCB utilizes the representational capabilities of deep neural networks and employs a neural network-based
random feature mapping to create an upper confidence bound (UCB) for reward, enabling efficient exploration.

This is a contextual multi-armed :ref:`bandit algorithm<bandits>`, meaning it is suited to RL problems with just a single timestep.

* Neural Bandits paper: https://arxiv.org/abs/1911.04462


Example
-------

.. code-block:: python

    from agilerl.algorithms.neural_ucb import NeuralUCB
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.wrappers.learning import BanditEnv

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    # Create environment
    env = BanditEnv(features, targets)
    context_dim = env.context_dim
    action_dim = env.arms

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(memory_size=10000, field_names=field_names)

    bandit = NeuralUCB(state_dim=context_dim, action_dim=action_dim)

    context = env.reset()  # Reset environment at start of episode
    for _ in range(500):
        # Get next action from agent
        action = agent.get_action(context)
        next_context, reward = env.step(action)  # Act in environment

        # Save experience to replay buffer
        memory.save_to_memory(context[action], reward)

        # Learn according to learning frequency
        if len(memory) >= agent.batch_size:
            for _ in range(agent.learn_step):
                experiences = memory.sample(agent.batch_size) # Sample replay buffer
                agent.learn(experiences)    # Learn according to agent's RL algorithm


        context = next_context


Neural Network Configuration
----------------------------

To configure the network architecture, pass a kwargs dict to the NeuralUCB ``net_config`` field. Full arguments can be found in the documentation
of :ref:`EvolvableMLP<evolvable_mlp>` and :ref:`EvolvableCNN<evolvable_cnn>`.
For an MLP, this can be as simple as:
.. code-block:: python

  NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'hidden_size': [32, 32]  # Network hidden size
    }

Or for a CNN:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'cnn',      # Network architecture
        'hidden_size': [128],    # Network hidden size
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
        'normalize': True   # Normalize image from range [0,255] to [0,1]
    }

.. code-block:: python

  agent = NeuralUCB(state_dim=state_dim, action_dim=action_dim, net_config=NET_CONFIG)   # Create NeuralUCB agent

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ucb import NeuralUCB

  agent = NeuralUCB(state_dim=state_dim, action_dim=action_dim)   # Create NeuralUCB agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ucb import NeuralUCB

  checkpoint_path = "path/to/checkpoint"
  agent = NeuralUCB.load(checkpoint_path)

Parameters
----------

.. autoclass:: agilerl.algorithms.neural_ucb_bandit.NeuralUCB
  :members:
  :inherited-members:
