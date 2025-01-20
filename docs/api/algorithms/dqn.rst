.. _dqn:

Deep Q-Learning (DQN)
=====================

DQN is an extension of Q-learning that makes use of a replay buffer and target network to improve learning stability.

* DQN paper: https://arxiv.org/abs/1312.5602

Can I use it?
--------------

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * -
     - Action
     - Observation
   * - Discrete
     - ✔️
     - ✔️
   * - Continuous
     - ❌
     - ✔️

Example
------------

.. code-block:: python

  import gymnasium as gym
  from agilerl.utils.utils import make_vect_envs
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.dqn import DQN

  # Create environment and Experience Replay Buffer
  num_envs = 8
  env = make_vect_envs('LunarLander-v2', num_envs=num_envs

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      state_dim = (state_dim[2], state_dim[0], state_dim[1])

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(memory_size=10000, field_names=field_names)

  agent = DQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create DQN agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      if channels_last:
          state = obs_channels_to_first(state)
      action = agent.get_action(state, epsilon)    # Get next action from agent
      next_state, reward, done, _, _ = env.step(action)   # Act in environment

      # Save experience to replay buffer
      if channels_last:
          memory.save_to_memory_vect_envs(state, action, reward, obs_channels_to_first(next_state), done)
      else:
          memory.save_to_memory_vect_envs(state, action, reward, next_state, done)

      # Learn according to learning frequency
      if len(memory) >= agent.batch_size:
          for _ in range(num_envs // agent.learn_step):
              experiences = memory.sample(agent.batch_size) # Sample replay buffer
              agent.learn(experiences)    # Learn according to agent's RL algorithm

Neural Network Configuration
----------------------------

To configure the network architecture, pass a kwargs dict to the DQN ``net_config`` field. Full arguments can be found in the documentation
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

  agent = DQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot, net_config=NET_CONFIG)   # Create DQN agent

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.dqn import DQN

  agent = DQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create DQN agent

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
