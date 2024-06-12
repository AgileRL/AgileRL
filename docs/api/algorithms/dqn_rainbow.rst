.. _dqn_rainbow:

Rainbow DQN
===========

Rainbow DQN is an extension of DQN that integrates multiple improvements and techniques to achieve state-of-the-art performance.
These improvements include:

* Double DQN (DDQN): Addresses the overestimation bias of Q-values by using two networks to decouple the selection and evaluation of the action in the Q-learning target.
* Prioritized Experience Replay: Instead of uniformly sampling from the replay buffer, it samples more important transitions more frequently based on the magnitude of their temporal difference (TD) error.
* Dueling Networks: Splits the Q-network into two separate streams — one for estimating the state value function and another for estimating the advantages for each action. They are then combined to produce Q-values.
* Multi-step Learning (n-step returns): Instead of using just the immediate reward for learning, it uses multi-step returns which consider a sequence of future rewards.
* Distributional RL: Instead of estimating the expected value of the cumulative future reward, it predicts the entire distribution of the cumulative future reward.
* Noisy Nets: Adds noise directly to the weights of the network, providing a way to explore the environment without the need for epsilon-greedy exploration.
* Categorical DQN (C51): A specific form of distributional RL where the continuous range of possible cumulative future rewards is discretized into a fixed set of categories.

Rainbow DQN paper: https://arxiv.org/abs/1710.02298

Can I use it?
-------------

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
  from agilerl.utils.utils import makeVectEnvs
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.dqn_rainbow import RainbowDQN

  # Create environment and Experience Replay Buffer
  env = makeVectEnvs('LunarLander-v2', num_envs=1)
  try:
      state_dim = env.single_observation_space.n          # Discrete observation space
      one_hot = True                                      # Requires one-hot encoding
  except:
      state_dim = env.single_observation_space.shape      # Continuous observation space
      one_hot = False                                     # Does not require one-hot encoding
  try:
      action_dim = env.single_action_space.n              # Discrete action space
  except:
      action_dim = env.single_action_space.shape[0]       # Continuous action space

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      state_dim = (state_dim[2], state_dim[0], state_dim[1])

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(memory_size=10000, field_names=field_names)

  agent = RainbowDQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      if channels_last:
          state = np.moveaxis(state, [3], [1])
      action = agent.getAction(state, epsilon)    # Get next action from agent
      next_state, reward, done, _, _ = env.step(action)   # Act in environment

      # Save experience to replay buffer
      if channels_last:
          memory.save2memoryVectEnvs(state, action, reward, np.moveaxis(next_state, [3], [1]), done)
      else:
          memory.save2memoryVectEnvs(state, action, reward, next_state, done)

      # Learn according to learning frequency
      if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
          experiences = memory.sample(agent.batch_size) # Sample replay buffer
          agent.learn(experiences)    # Learn according to agent's RL algorithm

To configure the network architecture, pass a dict to the DQN ``net_config`` field. For an MLP, this can be as simple as:

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

  agent = RainbowDQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot, net_config=NET_CONFIG)   # Create agent

Saving and loading agents
-------------------------

To save an agent, use the ``saveCheckpoint`` method:

.. code-block:: python

  from agilerl.algorithms.dqn_rainbow import RainbowDQN

  agent = RainbowDQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create Rainbow DQN agent

  checkpoint_path = "path/to/checkpoint"
  agent.saveCheckpoint(checkpoint_path)

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
