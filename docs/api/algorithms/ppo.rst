.. _ppo:

Proximal Policy Optimization (PPO)
=========================================

`PPO <https://arxiv.org/abs/1707.06347v2>`__ is an on-policy policy gradient algorithm that uses a clipped objective to constrain policy updates.
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
  from tqdm import tqdm

  from agilerl.algorithms.ppo import PPO
  from agilerl.rollouts.on_policy import collect_rollouts
  from agilerl.utils.utils import make_vect_envs

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
  total_steps = 0
  while total_steps < max_steps:
      agent.set_training_mode(True)

      # Collect rollouts and save in the agent's rollout buffer
      episode_scores = collect_rollouts(agent, env)

      agent.learn()    # Learn from rollout buffer

      total_steps += agent.learn_step
      agent.steps += agent.learn_step
      pbar.update(agent.learn_step)
      if episode_scores:
          pbar.set_description(f"Score: {np.mean(episode_scores)}")

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
