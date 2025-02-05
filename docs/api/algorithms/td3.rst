.. _td3:

Twin Delayed Deep Deterministic Policy Gradient (TD3)
=====================================================

TD3 is an extension of DDPG that addresses overestimation bias by introducing an extra
critic network, delayed actor network updates, and action noise regularization.

* TD3 paper: https://arxiv.org/abs/1802.09477

Can I use it?
-------------

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * -
     - Action
     - Observation
   * - Discrete
     - ❌
     - ✔️
   * - Continuous
     - ✔️
     - ✔️

Example
------------

.. code-block:: python

  import gymnasium as gym
  from agilerl.utils.algo_utils import obs_channels_to_first
  from agilerl.utils.utils import make_vect_envs, observation_space_channels_first
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.td3 import TD3

  # Create environment and Experience Replay Buffer
  num_envs = 1
  env = make_vect_envs('LunarLanderContinuous-v2', num_envs=num_envs)
  observation_space = env.observation_space
  action_space = env.action_space

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      observation_space = observation_space_channels_first(observation_space)

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(memory_size=10000, field_names=field_names)

  agent = TD3(observation_space, action_space)   # Create TD3 agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      if channels_last:
          state = obs_channels_to_first(state)
      action = agent.get_action(state, training=True)    # Get next action from agent
      next_state, reward, done, _, _ = env.step(action)   # Act in environment

      # Save experience to replay buffer
      if channels_last:
          memory.save_to_memory_vect_envs(state, action, reward, obs_channels_to_first(next_state), done)
      else:
          memory.save_to_memory_vect_envs(state, action, reward, next_state, done)

      # Learn according to learning frequency
      if len(memory) >= agent.batch_size:
          experiences = memory.sample(agent.batch_size) # Sample replay buffer
          agent.learn(experiences)    # Learn according to agent's RL algorithm

Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the TD3 ``net_config`` field.
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

  agent = TD3(observation_space, action_space, net_config=NET_CONFIG)   # Create TD3 agent

Evolutionary Hyperparameter Optimization
----------------------------------------

AgileRL allows for efficient hyperparameter optimization during training to provide state-of-the-art results in a fraction of the time.
For more information on how this is done, please refer to the :ref:`Evolutionary Hyperparameter Optimization <evo_hyperparam_opt>` documentation.

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.td3 import TD3

  agent = TD3(observation_space, action_space)   # Create TD3 agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.td3 import TD3

  checkpoint_path = "path/to/checkpoint"
  agent = TD3.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.td3.TD3
  :members:
  :inherited-members:
