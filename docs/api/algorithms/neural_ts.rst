.. _neural_ts:

Neural Contextual Bandits with Thompson Sampling (NeuralTS)
================================================================

NeuralTS adapts deep neural networks for both exploration and exploitation by using a posterior distribution of the reward
with a neural network approximator as its mean, and neural tangent features as its variance.

This is a contextual multi-armed :ref:`bandit algorithm<bandits>`, meaning it is suited to RL problems with just a single timestep.

* Neural Bandits paper: https://arxiv.org/abs/1911.04462


Example
-------

.. code-block:: python

    from agilerl.algorithms.neural_ts import NeuralTS
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

    observation_space = spaces.Box(low=features.values.min(), high=features.values.max())
    action_space = spaces.Discrete(action_dim)
    bandit = NeuralTS(observation_space, action_space)   # Create NeuralTS agent

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

To configure the architecture of the network's encoder / head, pass a kwargs dict to the NeuralTS ``net_config`` field.
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

  NET_CONFIG = {
      "encoder_config": {
        'hidden_size': [32, 32],  # Network head hidden size
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
      },
      "head_config": {'hidden_size': [32]}  # Network head hidden size
    }

.. code-block:: python

  agent = NeuralTS(observation_space, action_space, net_config=NET_CONFIG)   # Create NeuralTS agent

Evolutionary Hyperparameter Optimization
----------------------------------------

AgileRL allows for efficient hyperparameter optimization during training to provide state-of-the-art results in a fraction of the time.
For more information on how this is done, please refer to the :ref:`Evolutionary Hyperparameter Optimization <evo_hyperparam_opt>` documentation.

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ts import NeuralTS

  agent = NeuralTS(observation_space, action_space)   # Create NeuralTS agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ts import NeuralTS

  checkpoint_path = "path/to/checkpoint"
  agent = NeuralTS.load(checkpoint_path)

Parameters
----------

.. autoclass:: agilerl.algorithms.neural_ts_bandit.NeuralTS
  :members:
  :inherited-members:
