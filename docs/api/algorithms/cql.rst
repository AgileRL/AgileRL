Conservative Q-Learning (CQL)
=====================

CQL is an extension of Q-learning that adresses the typical overestimation of values induced by the distributional shift between 
the dataset and the learned policy in offline RL algorithms. A conservative Q-function is learned, such that the expected value of a
policy under this Q-function lower-bounds its true value

* CQL paper: https://arxiv.org/abs/2006.04779

Can I use it?
------------

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
------------

.. code-block:: python

  import gymnasium as gym
  import h5py
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.cqn import CQN

  # Create environment and Experience Replay Buffer, and load dataset
  env = gym.make('CartPole-v1')
  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(action_dim=4, memory_size=10000, field_names=field_names)
  dataset = h5py.File('data/cartpole/cartpole_random_v1.1.0.h5', 'r')  # Load dataset

  # Save transitions to replay buffer
  dataset_length = dataset['rewards'].shape[0]
  for i in range(dataset_length-1):
      state = dataset['observations'][i]
      next_state = dataset['observations'][i+1]
      if swap_channels:
          state = np.moveaxis(state, [3], [1])
          next_state = np.moveaxis(next_state, [3], [1])
      action = dataset['actions'][i]
      reward = dataset['rewards'][i]
      done = bool(dataset['terminals'][i])
      memory.save2memory(state, action, reward, next_state, done)

  agent = CQN(states_dim=4, action_dim=2, one_hot=False)   # Create DQN agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      experiences = memory.sample(agent.batch_size)   # Sample replay buffer
      # Learn according to agent's RL algorithm
      agent.learn(experiences)

To configure the network architecture, pass a dict to the CQN ``net_config`` field. For an MLP, this can be as simple as:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32]  # Network hidden size
    }

Or for a CNN:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'cnn',      # Network architecture
        'h_size': [128],    # Network hidden size
        'c_size': [32, 32], # CNN channel size
        'k_size': [8, 4],   # CNN kernel size
        's_size': [4, 2]    # CNN stride size
    }

.. code-block:: python

  agent = CQN(states_dim=4, action_dim=2, one_hot=False, net_config=NET_CONFIG)   # Create DQN agent  

Parameters
------------

.. autoclass:: agilerl.algorithms.cqn.CQN
  :members:
  :inherited-members: