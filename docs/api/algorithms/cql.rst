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
  from agilerl.utils.algo_utils import obs_channels_to_first

  # Create environment and Experience Replay Buffer, and load dataset
  env = gym.make('CartPole-v1')
  observation_space = env.observation_space
  action_space = env.action_space

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(memory_size=10000, field_names=field_names)
  dataset = h5py.File('data/cartpole/cartpole_random_v1.1.0.h5', 'r')  # Load dataset

  # Save transitions to replay buffer
  dataset_length = dataset['rewards'].shape[0]
  for i in range(dataset_length-1):
      state = dataset['observations'][i]
      next_state = dataset['observations'][i+1]
      if channels_last:
          state = obs_channels_to_first(state)
          next_state = obs_channels_to_first(next_state)

      action = dataset['actions'][i]
      reward = dataset['rewards'][i]
      done = bool(dataset['terminals'][i])
      memory.save_to_memory(state, action, reward, next_state, done)

  agent = CQN(observation_space=observation_space, action_space=action_space)   # Create DQN agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      experiences = memory.sample(agent.batch_size)   # Sample replay buffer
      # Learn according to agent's RL algorithm
      agent.learn(experiences)

Neural Network Configuration
----------------------------

To configure the network architecture, pass a dict to the CQN ``net_config`` field. Full arguments can be found in the documentation
of :ref:`EvolvableMLP<mlp>` and :ref:`EvolvableCNN<cnn>`.
For a basic vector observation space, this can be as simple as:

.. code-block:: python

  NET_CONFIG = {
      'encoder_config': {
        'hidden_size': [32]  # Network hidden size
      }
      'head_config': {
        'hidden_size': [32]
    }

Or for an image space:

.. code-block:: python

  NET_CONFIG = {
      'encoder_config': {
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
      }
      'head_config': {
        'hidden_size': [128] # Network head hidden size
    }

.. code-block:: python

  # Create CQN agent
  agent = CQN(
    observation_space=observation_space,
    action_space=action_space,
    net_config=NET_CONFIG
    normalize_images=True
    )


Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.cqn import CQN

  # Create CQN agent
  agent = CQN(observation_space=observation_space, action_space=action_space)

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

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
