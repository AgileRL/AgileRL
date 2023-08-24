Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
=========================================

MADDPG (Multi-Agent Deep Deterministic Policy Gradients) extends the DDPG (Deep Deterministic Policy Gradients) 
algorithm to enable cooperative or competitive training of multiple agents in complex environments, enhancing the 
stability and convergence of the learning process through decentralized actor and centralized critic architectures.

* MADDPG paper: https://arxiv.org/pdf/1706.02275.pdf

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

    import torch
    from pettingzoo.mpe import simple_speaker_listener_v4
    from agilerl.algorithms.maddpg import MADDPG 
    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
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
    memory = MultiAgentReplayBuffer(memory_size=1_000_000,
                                    field_names=field_names,
                                    agent_ids=agent_ids,
                                    device=device)

    agent = MADDPG(state_dims=state_dim,
                    action_dims=action_dim,
                    one_hot=one_hot,
                    n_agents=n_agents,
                    agent_ids=agent_ids,
                    max_action=max_action,
                    min_action=min_action,
                    discrete_actions=discrete_actions,
                    device=device)

    episodes = 1000
    max_steps = 25 # For atari environments it is recommended to use a value of 500
    epsilon = 1.0
    eps_end = 0.1
    eps_decay = 0.995

    for ep in range(episodes):
        state, _  = env.reset() # Reset environment at start of episode
        agent_reward = {agent_id: 0 for agent_id in env.agents}
        if channels_last:
                state = {agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1]) for agent_id, s in state.items()}

        for _ in range(max_steps):
            action = agent.getAction(state, epsilon) # Get next action from agent
            next_state, reward, done, _, _ = env.step(action) # Act in environment

            # Save experiences to replay buffer
            if channels_last:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                    next_state = {agent_id: np.moveaxis(ns, [2], [0]) for agent_id, ns in next_state.items()}
            memory.save2memory(state, action, reward, next_state, done)
                
            for agent_id, r in reward.items():
                    agent_reward[agent_id] += r 

            # Learn according to learning frequency
            if (memory.counter % agent.learn_step == 0) and (len(
                    memory) >= agent.batch_size):
                experiences = memory.sample(agent.batch_size) # Sample replay buffer
                agent.learn(experiences) # Learn according to agent's RL algorithm

            # Update the state 
            if channels_last:
                next_state = {agent_id: np.expand_dims(ns,0) for agent_id, ns in next_state.items()}
            state = next_state

        # Save the total episode reward
        score = sum(agent_reward.values())
        agent.scores.append(score)

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

.. code-block:: python

  NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32]  # Network hidden size
    }

Or for a CNN:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'cnn',      # Network architecture
        'h_size': [32,32],    # Network hidden size
        'c_size': [3, 32], # CNN channel size
        'k_size': [(1,3,3), (1,3,3)],   # CNN kernel size
        's_size': [2, 2],   # CNN stride size
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

Parameters
------------

.. autoclass:: agilerl.algorithms.maddpg.MADDPG
  :members:
  :inherited-members:


