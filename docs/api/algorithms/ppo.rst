.. _ppo:

Proximal Policy Optimization (PPO)
=========================================

PPO is a policy gradient method that uses a clipped objective to constrain policy updates.
It aims to combine the stability of Trust Region Policy Optimization (TRPO) with the simplicity
and scalability of vanilla policy gradients, effectively maintaining a balance between exploration
and exploitation. PPO is an on-policy algorithm.

* PPO paper: https://arxiv.org/abs/1707.06347v2

Can I use it?
-------------

Action Space
^^^^^^^^^^^^

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * - ``Discrete``
     - ``Box``
     - ``MultiDiscrete``
     - ``MultiBinary``
   * - ✔️
     - ✔️
     - ✔️
     - ✔️

Example
------------

.. code-block:: python

  import gymnasium as gym
  from agilerl.utils.algo_utils import obs_channels_to_first
  from agilerl.utils.utils import make_vect_envs, observation_space_channels_first
  from agilerl.algorithms.ppo import PPO

  # Create environment
  num_envs = 1
  env = make_vect_envs('LunarLanderContinuous-v2', num_envs=num_envs)
  observation_space = env.observation_space
  action_space = env.action_space

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      observation_space = observation_space_channels_first(observation_space)

  agent = PPO(observation_space, action_space)   # Create PPO agent

  while True:

      states = []
      actions = []
      log_probs = []
      rewards = []
      dones = []
      values = []

      done = np.zeros(num_envs)

      for step in range(agent.learn_step):
          if channels_last:
              state = obs_channels_to_first(state)

          # Get next action from agent
          action, log_prob, _, value = agent.get_action(state)
          next_state, reward, term, trunc, _ = env.step(action)  # Act in environment
          next_done = np.logical_or(term, trunc).astype(np.int8)

          states.append(state)
          actions.append(action)
          log_probs.append(log_prob)
          rewards.append(reward)
          dones.append(done)
          values.append(value)

          state = next_state
          done = next_done

      experiences = (
          states,
          actions,
          log_probs,
          rewards,
          dones,
          values,
          next_state,
          next_done,
      )
      # Learn according to agent's RL algorithm
      agent.learn(experiences)

Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the PPO ``net_config`` field.
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

  # Create PPO agent
  agent = PPO(
    observation_space=observation_space,
    action_space=action_space,
    net_config=NET_CONFIG
    )

Evolutionary Hyperparameter Optimization
----------------------------------------

AgileRL allows for efficient hyperparameter optimization during training to provide state-of-the-art results in a fraction of the time.
For more information on how this is done, please refer to the :ref:`Evolutionary Hyperparameter Optimization <evo_hyperparam_opt>` documentation.

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.ppo import PPO

  agent = PPO(observation_space, action_space)   # Create PPO agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.ppo import PPO

  checkpoint_path = "path/to/checkpoint"
  agent = PPO.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.ppo.PPO
  :members:
  :inherited-members:
