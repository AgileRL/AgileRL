Proximal Policy Optimization (PPO)
=========================================

PPO is a policy gradient method that uses a clipped objective to constrain policy updates.
It aims to combine the stability of Trust Region Policy Optimization (TRPO) with the simplicity
and scalability of vanilla policy gradients, effectively maintaining a balance between exploration
and exploitation. PPO is an on-policy algorithm.

* PPO paper: https://arxiv.org/abs/1707.06347v2

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

Example
------------

.. code-block:: python

  import gymnasium as gym
  from agilerl.utils import makeVectEnvs
  from agilerl.algorithms.ppo import PPO

  # Create environment
  env = makeVectEnvs('LunarLanderContinuous-v2', num_envs=1)
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

  agent = PPO(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create PPO agent

  num_episodes = 10  # Number of episodes
  max_steps = 100  # Max steps per episode

  for episode in range(num_episodes):
      for step in range(max_steps):
          if channels_last:
              state = np.moveaxis(state, [3], [1])
          # Get next action from agent
          action, log_prob, _, value = agent.getAction(state)
          next_state, reward, done, trunc, _ = env.step(action)  # Act in environment

          states.append(state)
          actions.append(action)
          log_probs.append(log_prob)
          rewards.append(reward)
          dones.append(done)
          values.append(value)

          state = next_state

      experiences = (
          states,
          actions,
          log_probs,
          rewards,
          dones,
          values,
          next_state,
      )
      # Learn according to agent's RL algorithm
      agent.learn(experiences)

To configure the network architecture, pass a dict to the PPO ``net_config`` field. For an MLP, this can be as simple as:

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
        's_size': [4, 2],   # CNN stride size
        'normalize': True   # Normalize image from range [0,255] to [0,1]
    }

.. code-block:: python

  agent = DDPG(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot, net_config=NET_CONFIG)   # Create DQN agent

Parameters
------------

.. autoclass:: agilerl.algorithms.ppo.PPO
  :members:
  :inherited-members:
