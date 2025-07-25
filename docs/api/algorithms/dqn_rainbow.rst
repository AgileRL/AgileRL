.. _dqn_rainbow:

Rainbow DQN
===========

`Rainbow DQN <https://arxiv.org/abs/1710.02298>`_ is an extension of :ref:`DQN <dqn>` that integrates multiple improvements and techniques to achieve state-of-the-art performance.
These improvements include:

  * **Double DQN (DDQN)**: Addresses the overestimation bias of Q-values by using two networks to decouple the selection and evaluation of the action in the Q-learning target.
  * **Prioritized Experience Replay**: Instead of uniformly sampling from the replay buffer, it samples more important transitions more frequently based on the magnitude of their temporal difference (TD) error.
  * **Dueling Networks**: Splits the Q-network into two separate streams — one for estimating the state value function and another for estimating the advantages for each action. They are then combined to produce Q-values.
  * **Multi-step Learning (n-step returns)**: Instead of using just the immediate reward for learning, it uses multi-step returns which consider a sequence of future rewards.
  * **Distributional RL**: Instead of estimating the expected value of the cumulative future reward, it predicts the entire distribution of the cumulative future reward.
  * **Noisy Nets**: Adds noise directly to the weights of the network, providing a way to explore the environment without the need for epsilon-greedy exploration.
  * **Categorical DQN (C51)**: A specific form of distributional RL where the continuous range of possible cumulative future rewards is discretized into a fixed set of categories.

Compatible Action Spaces
------------------------

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * - ``Discrete``
     - ``Box``
     - ``MultiDiscrete``
     - ``MultiBinary``
   * - ✔️
     - ❌
     - ❌
     - ❌

Example
------------

.. code-block:: python

  import gymnasium as gym
  from agilerl.utils.utils import make_vect_envs
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.dqn_rainbow import RainbowDQN

  # Create environment and Experience Replay Buffer
  num_envs = 8
  env = make_vect_envs('LunarLander-v3', num_envs=num_envs)
  observation_space = env.single_observation_space
  action_space = env.single_action_space

  memory = ReplayBuffer(max_size=10000)

  # Create agent
  agent = RainbowDQN(observation_space, action_space)
  agent.set_training_mode(True)

  obs, info = env.reset()  # Reset environment at start of episode
  while True:
      action = agent.get_action(obs, epsilon)    # Get next action from agent
      next_obs, reward, done, _, _ = env.step(action)   # Act in environment

      # Save experience to replay buffer
      transition = Transition(
          obs=obs,
          action=action,
          reward=reward,
          next_obs=next_obs,
          done=done,
          batch_size=[num_envs]
      )
      transition = transition.to_tensordict()
      memory.add(transition)

      # Learn according to learning frequency
      if len(memory) >= agent.batch_size:
          for _ in range(num_envs // agent.learn_step):
              experiences = memory.sample(agent.batch_size) # Sample replay buffer
              agent.learn(experiences)    # Learn according to agent's RL algorithm


Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the RainbowDQN ``net_config`` field.
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

  # Create agent
  agent = RainbowDQN(
    observation_space=observation_space,
    action_space=action_space,
    net_config=NET_CONFIG
    )

Evolutionary Hyperparameter Optimization
----------------------------------------

AgileRL allows for efficient hyperparameter optimization during training to provide state-of-the-art results in a fraction of the time.
For more information on how this is done, please refer to the :ref:`Evolutionary Hyperparameter Optimization <evo_hyperparam_opt>` documentation.

Saving and Loading Agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.dqn_rainbow import RainbowDQN

  agent = RainbowDQN(observation_space, action_space)   # Create Rainbow DQN agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.dqn_rainbow import RainbowDQN

  checkpoint_path = "path/to/checkpoint"
  agent = RainbowDQN.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.dqn_rainbow.RainbowDQN
  :members:
  :inherited-members:
