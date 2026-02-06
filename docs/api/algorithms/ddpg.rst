.. _ddpg:

Deep Deterministic Policy Gradient (DDPG)
=========================================

`DDPG <https://arxiv.org/abs/1509.02971>`_ is an extension of :ref:`DQN <dqn>` to work in continuous action spaces by introducing an actor
network that outputs continuous actions.

Compatible Action Spaces
------------------------

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * - ``Discrete``
     - ``Box``
     - ``MultiDiscrete``
     - ``MultiBinary``
   * - ❌
     - ✔️
     - ❌
     - ❌

Example
------------

.. code-block:: python

  import torch
  from agilerl.utils.utils import make_vect_envs
  from agilerl.components.replay_buffer import ReplayBuffer
  from agilerl.components.data import Transition
  from agilerl.algorithms.ddpg import DDPG
  from agilerl.networks.actors import DeterministicActor

  # Create environment and Experience Replay Buffer
  num_envs = 1
  env = make_vect_envs('LunarLanderContinuous-v3', num_envs=num_envs)
  observation_space = env.single_observation_space
  action_space = env.single_action_space

  memory = ReplayBuffer(max_size=10000)

  # Create DDPG agent
  agent = DDPG(observation_space, action_space)
  agent.set_training_mode(True)

  obs, info = env.reset()  # Reset environment at start of episode
  while True:
      action = agent.get_action(obs, training=True)  # Get next action from agent (raw + noise)
      # Rescale action from network output bounds to env action space
      action = DeterministicActor.rescale_action(
          action=torch.from_numpy(action),
          low=agent.action_low,
          high=agent.action_high,
          output_activation=agent.actor.output_activation,
      ).numpy()
      next_obs, reward, done, _, _ = env.step(action)   # Act in environment

      # Save experience to replay buffer
      transition = Transition(
          obs=obs,
          action=action,
          reward=reward,
          next_obs=next_obs,
          done=done,
          batch_size=[num_envs]
      )
      transition = transition.to_tensordict()
      memory.add(transition)

      # Learn according to learning frequency
      if len(memory) >= agent.batch_size:
          experiences = memory.sample(agent.batch_size) # Sample replay buffer
          agent.learn(experiences)    # Learn according to agent's RL algorithm

.. note::
  In the loop above, actions are rescaled after :meth:`get_action` using the static method
  :meth:`DeterministicActor.rescale_action <agilerl.networks.actors.DeterministicActor.rescale_action>`.
  This maps the actor's output (in the range implied by its output activation, e.g. ``[-1, 1]`` for
  Tanh) to the environment's action space ``[action_low, action_high]``. When using
  ``get_action(..., training=True)`` for exploration, the agent returns actions in the activation
  range; rescaling to the env space is required before :meth:`env.step`. When not in training mode,
  the agent applies this rescaling internally.

Custom actor networks
---------------------

DDPG allows actor networks that are not :ref:`DeterministicActor <actors>`. If you use a custom actor,
it must define an attribute **output_activation** (a string) set to one of the allowed output
activations: ``"Tanh"``, ``"Softsign"``, ``"Sigmoid"``, ``"Softmax"``, or ``"GumbelSoftmax"``. This
is used by :meth:`DeterministicActor.rescale_action <agilerl.networks.actors.DeterministicActor.rescale_action>`
to map network outputs to the environment action space correctly.

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

Saving and Loading Agents
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
