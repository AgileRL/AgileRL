.. _cql:

Conservative Q-Learning (CQL)
=============================

CQL is an extension of Q-learning that addresses the typical overestimation of values induced by the distributional shift between
the dataset and the learned policy in offline RL algorithms. A conservative Q-function is learned, such that the expected value of a
policy under this Q-function lower-bounds its true value

* CQL paper: https://arxiv.org/abs/2006.04779

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
     - ✔️
     - ✔️

So far, we have implemented CQN - CQL applied to DQN, which cannot be used on continuous action spaces. We will soon be
adding other CQL extensions of algorithms for offline RL.

Example
-------

.. code-block:: python

  import gymnasium as gym
  import h5py
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.cqn import CQN

  # Create environment and Experience Replay Buffer, and load dataset
  env = gym.make('CartPole-v1')
  try:
      state_dim = env.observation_space.n       # Discrete observation space
      one_hot = True                            # Requires one-hot encoding
  except Exception:
      state_dim = env.observation_space.shape   # Continuous observation space
      one_hot = False                           # Does not require one-hot encoding
  try:
      action_dim = env.action_space.n           # Discrete action space
  except Exception:
      action_dim = env.action_space.shape[0]    # Continuous action space

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      state_dim = (state_dim[2], state_dim[0], state_dim[1])

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(memory_size=10000, field_names=field_names)
  dataset = h5py.File('data/cartpole/cartpole_random_v1.1.0.h5', 'r')  # Load dataset

  # Save transitions to replay buffer
  dataset_length = dataset['rewards'].shape[0]
  for i in range(dataset_length-1):
      state = dataset['observations'][i]
      next_state = dataset['observations'][i+1]
      if channels_last:
          state = np.moveaxis(state, [-1], [-3])
          next_state = np.moveaxis(next_state, [-1], [-3])
      action = dataset['actions'][i]
      reward = dataset['rewards'][i]
      done = bool(dataset['terminals'][i])
      memory.save2memory(state, action, reward, next_state, done)

  agent = CQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create DQN agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      experiences = memory.sample(agent.batch_size)   # Sample replay buffer
      # Learn according to agent's RL algorithm
      agent.learn(experiences)

Neural Network Configuration
----------------------------

To configure the network architecture, pass a dict to the CQN ``net_config`` field. Full arguments can be found in the documentation
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

  agent = CQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot, net_config=NET_CONFIG)   # Create CQN agent


Saving and loading agents
-------------------------

To save an agent, use the ``saveCheckpoint`` method:

.. code-block:: python

  from agilerl.algorithms.cqn import CQN

  agent = CQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create CQN agent

  checkpoint_path = "path/to/checkpoint"
  agent.saveCheckpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.cqn import CQN

  checkpoint_path = "path/to/checkpoint"
  agent = CQN.load(checkpoint_path)

Parameters
----------

.. autoclass:: agilerl.algorithms.cqn.CQN
  :members:
  :inherited-members:
