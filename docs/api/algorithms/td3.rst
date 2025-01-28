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
  from agilerl.utils.utils import make_vect_envs
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.td3 import TD3

  # Create environment and Experience Replay Buffer
  num_envs = 1
  env = make_vect_envs('LunarLanderContinuous-v2', num_envs=num_envs)

  max_action = float(env.single_action_space.high[0])

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      state_dim = (state_dim[2], state_dim[0], state_dim[1])

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(memory_size=10000, field_names=field_names)

  agent = TD3(observation_space=observation_space, action_space=action_space,  max_action=max_action)   # Create TD3 agent

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

To configure the network architecture, pass a kwargs dict to the TD3 ``net_config`` field. Full arguments can be found in the documentation
of :ref:`EvolvableMLP<mlp>` and :ref:`EvolvableCNN<cnn>`.
For an MLP, this can be as simple as:

.. code-block:: python

  NET_CONFIG = {
        'hidden_size': [32, 32]  # Network hidden size
    }

Or for a CNN:

.. code-block:: python

  NET_CONFIG = {
        'hidden_size': [128],    # Network hidden size
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
        'normalize': True   # Normalize image from range [0,255] to [0,1]
    }

.. code-block:: python

  agent = TD3(state_dim=state_dim, action_dim=action_dim, one_hot=False, net_config=NET_CONFIG)   # Create TD3 agent

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.td3 import TD3

  agent = TD3(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create TD3 agent

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
