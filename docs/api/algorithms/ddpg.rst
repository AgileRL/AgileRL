.. _ddpg:

Deep Deterministic Policy Gradient (DDPG)
=========================================

DDPG is an extension of DQN to work in continuous action spaces by introducing an actor
network that outputs continuous actions.

* DDPG paper: https://arxiv.org/abs/1509.02971

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
  from agilerl.utils.utils import make_vect_envs, observation_space_channels_to_first
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.algorithms.ddpg import DDPG

  # Create environment and Experience Replay Buffer
  num_envs = 1
  env = make_vect_envs('LunarLanderContinuous-v2', num_envs=num_envs)
  observation_space = env.observation_space
  action_space = env.action_space

  channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

  if channels_last:
      observation_space = observation_space_channels_to_first(observation_space)

  memory = ReplayBuffer(max_size=10000)

  agent = DDPG(observation_space, action_space)   # Create DDPG agent

  state = env.reset()[0]  # Reset environment at start of episode
  while True:
      if channels_last:
          state = obs_channels_to_first(state)

      action = agent.get_action(state, training=True)    # Get next action from agent
      next_state, reward, done, _, _ = env.step(action)   # Act in environment

      # Save experience to replay buffer
      next_state = obs_channels_to_first(next_state) if channels_last else next_state
      transition = Transition(
          obs=state,
          action=action,
          reward=reward,
          next_obs=next_state,
          done=done,
          batch_size=[num_envs]
      )
      transition = transition.to_tensordict()
      memory.add(transition)

      # Learn according to learning frequency
      if len(memory) >= agent.batch_size:
          experiences = memory.sample(agent.batch_size) # Sample replay buffer
          agent.learn(experiences)    # Learn according to agent's RL algorithm

Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the DDPG ``net_config`` field.
Full arguments can be found in the documentation of :ref:`EvolvableMLP<mlp>`, :ref:`EvolvableCNN<cnn>`, and
:ref:`EvolvableMultiInput<multi_input>`.

For discrete / vector observations:

.. code-block:: python

  NET_CONFIG = {
        "encoder_config": {
            'hidden_size': [32, 32] # Network encoder hidden size
        },
        "head_config": {
            'hidden_size': [32] # Network head hidden size
        }
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

  # Create DDPG agent
  agent = DDPG(
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

  from agilerl.algorithms.ddpg import DDPG

  agent = DDPG(observation_space, action_space)   # Create DDPG agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.ddpg import DDPG

  checkpoint_path = "path/to/checkpoint"
  agent = DDPG.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.ddpg.DDPG
  :members:
  :inherited-members:
