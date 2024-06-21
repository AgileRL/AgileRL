.. _maddpg:

Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
=======================================================

MADDPG (Multi-Agent Deep Deterministic Policy Gradients) extends the DDPG (Deep Deterministic Policy Gradients)
algorithm to enable cooperative or competitive training of multiple agents in complex environments, enhancing the
stability and convergence of the learning process through decentralized actor and centralized critic architectures.

* MADDPG paper: https://arxiv.org/pdf/1706.02275.pdf

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

If you need to take actions from agents at different timesteps, you can use agent masking to only retrieve new actions for certain agents. This
can be defined by your environment, and should be returned in 'info' as a dictionary. Info must contain two dictionaries - one named 'agent_mask',
which contains a boolean value for whether an action should be returned for each agent, and another named 'env_defined_actions', which contains
the actions for each agent that a new action is not generated for. This is handled automatically by the AgileRL multi-agent training function, but
can be implemented in a custom loop as follows:

.. code-block:: python

    info = {'agent_mask': {'speaker_0': True, 'listener_0': False},
            'env_defined_actions': {'speaker_0': None, 'listener_0': np.array([0,0,0,0,0])}}

.. code-block:: python

    state, info = env.reset()  # or: next_state, reward, done, truncation, info = env.step(action)
    cont_actions, discrete_action = agent.get_action(state, epsilon, info['agent_mask'], info['env_defined_actions'])
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

    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 8
    env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
    env = PettingZooVectorizationParallelWrapper(env, n_envs=num_envs)
    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]

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

    agent = MADDPG(
        state_dims=state_dim,
        action_dims=action_dim,
        one_hot=one_hot,
        n_agents=n_agents,
        agent_ids=agent_ids,
        max_action=max_action,
        min_action=min_action,
        vect_noise_dim=num_envs,
        discrete_actions=discrete_actions,
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
            state = {agent_id: np.moveaxis(s, [-1], [-3]) for agent_id, s in state.items()}

        for _ in range(1000):
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                states=state,
                training=True,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
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
                    agent_id: np.moveaxis(ns, [-1], [-3])
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

To configure the network architecture, pass a kwargs dict to the MADDPG ``net_config`` field. Full arguments can be found in the documentation
of :ref:`EvolvableMLP<evolvable_mlp>` and :ref:`EvolvableCNN<evolvable_cnn>`.
For an MLP, this can be as simple as:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'hidden_size': [32, 32]  # Network hidden size
    }

Or for a CNN:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'cnn',      # Network architecture
        'hidden_size': [32,32],    # Network hidden size
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [3, 3],   # CNN kernel size
        'stride_size': [2, 2],   # CNN stride size
        'normalize': True   # Normalize image from range [0,255] to [0,1]
    }

.. code-block:: python

  agent = MADDPG(state_dims=state_dim,
                 action_dims=action_dim,
                 one_hot=one_hot,
                 n_agents=n_agents,
                 agent_ids=agent_ids,
                 max_action=max_action,
                 min_action=min_action,
                 discrete_actions=discrete_actions,
                 net_config=NET_CONFIG)   # Create MADDPG agent

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms.maddpg import MADDPG

  agent = MADDPG(state_dims=state_dim,
                 action_dims=action_dim,
                 one_hot=one_hot,
                 n_agents=n_agents,
                 agent_ids=agent_ids,
                 max_action=max_action,
                 min_action=min_action,
                 discrete_actions=discrete_actions)   # Create MADDPG agent

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.maddpg import MADDPG

  checkpoint_path = "path/to/checkpoint"
  agent = MADDPG.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.maddpg.MADDPG
  :members:
  :inherited-members:
