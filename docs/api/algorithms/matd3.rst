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
    cont_actions, discrete_action = agent.getAction(state, epsilon, info['agent_mask'], info['env_defined_actions'])
    if agent.discrete_actions:
        action = discrete_action
    else:
        action = cont_actions

Example
------------

.. code-block:: python

    import torch
    from pettingzoo.mpe import simple_speaker_listener_v4
    from agilerl.algorithms.matd3 import MATD3
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

    agent = MATD3(state_dims=state_dim,
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
        state, info  = env.reset() # Reset environment at start of episode
        agent_reward = {agent_id: 0 for agent_id in env.agents}
        if channels_last:
            state = {agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1]) for agent_id, s in state.items()}

        for _ in range(max_steps):
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = agent.getAction(
                state, epsilon, agent_mask, env_defined_actions
            )
            if agent.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            next_state, reward, termination, truncation, info = env.step(
                action
            )  # Act in environment

            # Save experiences to replay buffer
            if channels_last:
                state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                next_state = {agent_id: np.moveaxis(ns, [2], [0]) for agent_id, ns in next_state.items()}
            memory.save2memory(state, cont_actions, reward, next_state, done)

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

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

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

  agent = MATD3(state_dims=state_dim,
                action_dims=action_dim,
                one_hot=one_hot,
                n_agents=n_agents,
                agent_ids=agent_ids,
                max_action=max_action,
                min_action=min_action,
                discrete_actions=discrete_actions,
                net_config=NET_CONFIG)   # Create MATD3 agent

Parameters
------------

.. autoclass:: agilerl.algorithms.matd3.MATD3
  :members:
  :inherited-members:
