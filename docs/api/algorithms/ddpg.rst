Deep Deterministic Policy Gradient (DDPG)
=========================================

DDPG is an extension of DQN to work in continuous action spaces by introducing an actor 
network that outputs continuous actions.

* DDPG paper: https://arxiv.org/abs/1509.02971

Can I use it?
------------

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
    from agilerl.utils import makeVectEnvs
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.algorithms.ddpg import DDPG

    # Create environment and Experience Replay Buffer
    env = makeVectEnvs('LunarLanderContinuous-v2', num_envs=1)
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(n_actions=2, memory_size=10000, field_names=field_names)

    agent = DDPG(n_states=8, n_actions=2)   # Create DQN agent

    state = env.reset()[0]  # Reset environment at start of episode
    while True:
        action = agent.getAction(state, epsilon)    # Get next action from agent
        next_state, reward, done, _, _ = env.step(action)   # Act in environment

        # Save experience to replay buffer
        memory.save2memoryVectEnvs(state, action, reward, next_state, done)

        # Learn according to learning frequency
        if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
            experiences = memory.sample(agent.batch_size) # Sample replay buffer
            agent.learn(experiences)    # Learn according to agent's RL algorithm


Parameters
------------

.. autoclass:: agilerl.algorithms.ddpg.DDPG
  :members:
  :inherited-members:
