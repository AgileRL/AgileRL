.. _matd3:

Multi-Agent Twin-Delayed Deep Deterministic Policy Gradient (MATD3)
===================================================================

MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradients) extends the MADDPG algorithm to reduce overestimation bias
in multi-agent domains through the use of a second set of critic networks and delayed updates of the policy networks. This
enables superior performance when compared to MADDPG.

* MATD3 paper: https://arxiv.org/abs/1910.01465

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
     - ✔️
     - ✔️

Gumbel-Softmax
--------------

The Gumbel-Softmax activation function is a differentiable approximation that enables gradient-based optimization through
continuous relaxation of discrete action spaces in multi-agent reinforcement learning, allowing agents to learn and improve
decision-making in complex environments with discrete choices. If you would like to customise the mlp output activation function,
you can define it within the network configuration using the key "output_activation". User definition for the output activation is however,
unnecessary, as the algorithm will select the appropriate function given the environments action space.

Agent Masking
-------------

If you need to take actions from agents at different timesteps, you can use agent masking to only retrieve new actions for certain agents whilst
providing 'environment defined actions' for other agents, which act as a nominal action for such "masked" agents to take. These nominal actions
should be returned as part of the ``info`` dictionary. Following the PettingZoo API we recommend the ``info`` dictionary to be keyed by the
agents, with ``env_defined_actions`` defined as follows:

.. code-block:: python

    info = {'speaker_0': {'env_defined_actions':  None},
            'listener_0': {'env_defined_actions': np.array([0,0,0,0,0])}

For agents that you wish not to be masked, the ``env_defined_actions`` should be set to ``None``. If your environment has discrete action spaces
then provide 'env_defined_actions' as a numpy array with a single value. For example, an action space of type ``Discrete(5)`` may have an
``env_defined_action`` of ``np.array([4])``. For an environment with continuous actions spaces (e.g. ``Box(0, 1, (5,))``) then the shape of the
array should be the size of the action space (``np.array([0.5, 0.5, 0.5, 0.5, 0.5])``). Agent masking is handled automatically by the AgileRL
multi-agent training function, but can be implemented in a custom loop as follows:

.. code-block:: python

    env_defined_actions = {agent: info[agent]["env_defined_actions"] for agent in env.agents}
    state, info = env.reset()  # or: next_state, reward, done, truncation, info = env.step(action)
    cont_actions, discrete_action = agent.get_action(state, env_defined_actions=env_defined_actions)
    if agent.discrete_actions:
        action = discrete_action
    else:
        action = cont_actions

Example
------------

.. code-block:: python

    import numpy as np
    import torch
    from pettingzoo.mpe import simple_speaker_listener_v4
    from tqdm import trange

    from agilerl.utils.algo_utils import obs_channels_to_first
    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 8
    env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
    env.reset()

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    channels_last = False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    n_agents = env.num_agents
    agent_ids = [agent_id for agent_id in env.agents]
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=field_names,
        agent_ids=agent_ids,
        device=device,
    )

    agent = MATD3(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        vect_noise_dim=num_envs,
        device=device,
    )

    # Define training loop parameters
    max_steps = 100000  # Max steps
    total_steps = 0

    while agent.steps[-1] < max_steps:
        state, info  = env.reset() # Reset environment at start of episode
        scores = np.zeros(num_envs)
        completed_episode_scores = []
        if channels_last:
            state = {agent_id: obs_channels_to_first(s) for agent_id, s in state.items()}

        for _ in range(1000):
            # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                states=state,
                training=True,
                infos=info,
            )
            if agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Act in environment
            next_state, reward, termination, truncation, info = env.step(action)

            scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
            total_steps += num_envs
            steps += num_envs

            # Save experiences to replay buffer
            if channels_last:
                next_state = {
                    agent_id: obs_channels_to_first(ns)
                    for agent_id, ns in next_state.items()
                }
            memory.save_to_memory(state, cont_actions, reward, next_state, done, is_vectorised=True)

            # Learn according to learning frequency
            if len(memory) >= agent.batch_size:
                for _ in range(num_envs // agent.learn_step):
                    experiences = memory.sample(agent.batch_size) # Sample replay buffer
                    agent.learn(experiences) # Learn according to agent's RL algorithm

            # Update the state
            state = next_state

            # Calculate scores and reset noise for finished episodes
            reset_noise_indices = []
            term_array = np.array(list(termination.values())).transpose()
            trunc_array = np.array(list(truncation.values())).transpose()
            for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                if np.any(d) or np.any(t):
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0
                    reset_noise_indices.append(idx)
            agent.reset_action_noise(reset_noise_indices)

        agent.steps[-1] += steps

Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the MATD3 ``net_config`` field.
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

  NET_CONFIG = {
      "encoder_config": {
        'hidden_size': [32, 32],  # Network head hidden size
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
      },
      "head_config": {'hidden_size': [32]}  # Network head hidden size
    }
.. code-block:: python

  # Create MATD3 agent
  agent = MATD3(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config=NET_CONFIG
    )

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.matd3 import MATD3

  # Create MATD3 agent
  agent = MATD3(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config=NET_CONFIG
    )

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.matd3 import MATD3

  checkpoint_path = "path/to/checkpoint"
  agent = MATD3.load(checkpoint_path)


Parameters
------------

.. autoclass:: agilerl.algorithms.matd3.MATD3
  :members:
  :inherited-members:
