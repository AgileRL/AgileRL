.. _ppo:

Proximal Policy Optimization (PPO)
=========================================

`PPO <https://arxiv.org/abs/1707.06347v2>`_ is an on-policy policy gradient algorithm that uses a clipped objective to constrain policy updates.
It aims to combine the stability of `Trust Region Policy Optimization (TRPO) <https://arxiv.org/abs/1502.05477>`_ with the simplicity
and scalability of vanilla policy gradients, effectively maintaining a balance between exploration and exploitation.

AgileRL offers support for recurrent policies in PPO to solve Partially Observable Markov Decision Processes (POMDPs). For more information, please
refer to the :ref:`Partially Observable Markov Decision Processes (POMDPs) <pomdp>` documentation, or our tutorial on solving ``Pendulum-v1`` with masked
angular velocity observations :ref:`here <agilerl_recurrent_ppo_tutorial>`.

Compatible Action Spaces
------------------------

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

LunarLanderContinuous-v3 Example
---------------------------------

.. code-block:: python

  import numpy as np
  from gymnasium import spaces
  from tqdm import tqdm

  from agilerl.utils.utils import make_vect_envs
  from agilerl.algorithms.ppo import PPO

  # Create environment
  num_envs = 16
  max_steps = 100000
  env = make_vect_envs('LunarLanderContinuous-v3', num_envs=num_envs)
  observation_space = env.single_observation_space
  action_space = env.single_action_space

  # Create PPO agent
  agent = PPO(
      observation_space,
      action_space,
      lr=1e-3,
      batch_size=128,
      learn_step=2048
  )

  pbar = tqdm(total=max_steps)
  while True:
      observations = []
      actions = []
      log_probs = []
      rewards = []
      dones = []
      values = []

      done = np.zeros(num_envs)
      obs, info = env.reset()
      agent.set_training_mode(True)
      for _ in range(-(agent.learn_step // -num_envs)):
          # Get next action from agent
          action, log_prob, _, value = agent.get_action(obs)

          # Clip to action space
          if isinstance(agent.action_space, spaces.Box):
              if agent.actor.squash_output:
                  clipped_action = agent.actor.scale_action(action)
              else:
                  clipped_action = np.clip(action, agent.action_space.low, agent.action_space.high)
          else:
              clipped_action = action

          next_obs, reward, term, trunc, _ = env.step(clipped_action)  # Act in environment
          next_done = np.logical_or(term, trunc).astype(np.int8)

          observations.append(obs)
          actions.append(action)
          log_probs.append(log_prob)
          rewards.append(reward)
          dones.append(done)
          values.append(value)

          obs = next_obs
          done = next_done

      experiences = (
          observations,
          actions,
          log_probs,
          rewards,
          dones,
          values,
          next_obs,
          next_done,
      )
      agent.learn(experiences)    # Learn according to agent's RL algorithm

      pbar.update(agent.learn_step)
      pbar.set_description(f"Score: {np.mean(np.sum(rewards, axis=0))}")

Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the PPO ``net_config`` field.
Full arguments can be found in the documentation of :ref:`EvolvableMLP<mlp>`, :ref:`EvolvableCNN<cnn>`,
:ref:`EvolvableMultiInput<multi_input>`, and :ref:`EvolvableLSTM<lstm>`.

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

For recurrent observations:

.. code-block:: python

  NET_CONFIG = {
    "encoder_config": {
      "hidden_state_size": 64,
      "num_layers": 1,
      "max_seq_len": 512,
    },
    "head_config": {
      "hidden_size": [64],
    }
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

Saving and Loading Agents
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
