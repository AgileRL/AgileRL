.. _dqn:

Deep Q-Learning (DQN)
=====================

DQN is an extension of Q-learning that makes use of a replay buffer and target network to improve learning stability.

* DQN paper: https://arxiv.org/abs/1312.5602

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
  from agilerl.algorithms.dqn import DQN

  # Create environment and Experience Replay Buffer
  num_envs = 8
  env = make_vect_envs('LunarLander-v3', num_envs=num_envs)
  observation_space = env.single_observation_space
  action_space = env.single_action_space

  memory = ReplayBuffer(max_size=10000)

  # Create DQN agent
  agent = DQN(observation_space, action_space)
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
      memory.add(transition)

      # Learn according to learning frequency
      if len(memory) >= agent.batch_size:
          for _ in range(num_envs // agent.learn_step):
              experiences = memory.sample(agent.batch_size) # Sample replay buffer
              agent.learn(experiences)    # Learn according to agent's RL algorithm

Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the DQN ``net_config`` field.
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

  # Create DQN agent
  agent = DQN(
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

  from agilerl.algorithms.dqn import DQN

  agent = DQN(observation_space, action_space)   # Create DQN agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.dqn import DQN

  checkpoint_path = "path/to/checkpoint"
  agent = DQN.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.dqn.DQN
  :members:
  :inherited-members:
