.. _neural_ucb:

Neural Contextual Bandits with UCB-based Exploration (NeuralUCB)
================================================================

`NeuralUCB <https://arxiv.org/abs/1911.04462>`_ utilizes the representational capabilities of deep neural networks and employs a neural network-based
random feature mapping to create an upper confidence bound (UCB) for reward, enabling efficient exploration.

This is a contextual multi-armed :ref:`bandit algorithm<bandits>`, meaning it is suited to RL problems with just a single timestep.


Example
-------

.. code-block:: python

    from tensordict import TensorDict

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

    memory = ReplayBuffer(max_size=10000)

    observation_space = spaces.Box(low=features.values.min(), high=features.values.max())
    action_space = spaces.Discrete(action_dim)
    bandit = NeuralUCB(observation_space, action_space)   # Create NeuralUCB agent

    context = env.reset()  # Reset environment at start of episode
    for _ in range(500):
        # Get next action from agent
        action = agent.get_action(context)
        next_context, reward = env.step(action)  # Act in environment

        # Save experience to replay buffer
        transition = TensorDict({
          "obs": context[action],
          "reward": reward,
          },
          batch_size=[1]
        )
        memory.add(transition)

        # Learn according to learning frequency
        if len(memory) >= agent.batch_size:
            for _ in range(agent.learn_step):
                experiences = memory.sample(agent.batch_size) # Sample replay buffer
                agent.learn(experiences)    # Learn according to agent's RL algorithm


        context = next_context


Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the NeuralUCB ``net_config`` field.
Full arguments can be found in the documentation of :ref:`EvolvableMLP<mlp>`, :ref:`EvolvableCNN<cnn>`, and
:ref:`EvolvableMultiInput<multi_input>`.

For discrete / vector observations:

.. code-block:: python

  NET_CONFIG = {
        "encoder_config": {'hidden_size': [32, 32]},  # Network head hidden size
        "head_config": {'hidden_size': [32]}      # Network head hidden size
    }

For image observations:

.. code-block:: python

  NET_CONFIG = {
      "encoder_config": {
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
      },
      "head_config": {'hidden_size': [32]}  # Network head hidden size
    }

For dictionary / tuple observations containing any combination of image, discrete, and vector observations:

.. code-block:: python

  CNN_CONFIG = {
      "channel_size": [32, 32], # CNN channel size
      "kernel_size": [8, 4],   # CNN kernel size
      "stride_size": [4, 2],   # CNN stride size
  }

  NET_CONFIG = {
      "encoder_config": {
        "latent_dim": 32,
        # Config for nested EvolvableCNN objects
        "cnn_config": CNN_CONFIG,
        # Config for nested EvolvableMLP objects
        "mlp_config": {
            "hidden_size": [32, 32]
        },
        "vector_space_mlp": True # Process vector observations with an MLP
      },
      "head_config": {'hidden_size': [32]}  # Network head hidden size
    }

.. code-block:: python

  agent = NeuralUCB(observation_space, action_space, net_config=NET_CONFIG)   # Create NeuralUCB agent

Evolutionary Hyperparameter Optimization
----------------------------------------

AgileRL allows for efficient hyperparameter optimization during training to provide state-of-the-art results in a fraction of the time.
For more information on how this is done, please refer to the :ref:`Evolutionary Hyperparameter Optimization <evo_hyperparam_opt>` documentation.

Saving and Loading Agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ucb import NeuralUCB

  agent = NeuralUCB(observation_space, action_space)   # Create NeuralUCB agent

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
