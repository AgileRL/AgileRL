.. _neural_ts:

Neural Contextual Bandits with Thompson Sampling (NeuralTS)
================================================================

NeuralTS adapts deep neural networks for both exploration and exploitation by using a posterior distribution of the reward
with a neural network approximator as its mean, and neural tangent features as its variance.

This is a contextual multi-armed bandit algorithm, meaning it is suited to RL problems with just a single timestep.

* Neural Bandits paper: https://arxiv.org/abs/1911.04462


Example
-------

.. code-block:: python

    from agilerl.algorithms.neural_ts import NeuralTS
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.wrappers.learning import BanditEnv

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    # Create environment
    env = BanditEnv(features, targets)
    context_dim = env.context_dim
    action_dim = env.arms

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim, memory_size=10000, field_names=field_names)

    bandit = NeuralTS(state_dim=context_dim, action_dim=action_dim)

    context = env.reset()  # Reset environment at start of episode
    for _ in range(500):
        # Get next action from agent
        action = agent.getAction(context)
        next_context, reward = env.step(action)  # Act in environment

        # Save experience to replay buffer
        memory.save2memory(context[action], reward)

        # Learn according to learning frequency
        if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
            experiences = memory.sample(agent.batch_size) # Sample replay buffer
            agent.learn(experiences)    # Learn according to agent's RL algorithm

        context = next_context


To configure the network architecture, pass a dict to the NeuralTS ``net_config`` field. For an MLP, this can be as simple as:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32]  # Network hidden size
    }

Or for a CNN:

.. code-block:: python

  NET_CONFIG = {
        'arch': 'cnn',      # Network architecture
        'h_size': [128],    # Network hidden size
        'c_size': [32, 32], # CNN channel size
        'k_size': [8, 4],   # CNN kernel size
        's_size': [4, 2],   # CNN stride size
        'normalize': True   # Normalize image from range [0,255] to [0,1]
    }

.. code-block:: python

  agent = NeuralTS(state_dim=state_dim, action_dim=action_dim, net_config=NET_CONFIG)   # Create NeuralTS agent

Saving and loading agents
-------------------------

To save an agent, use the ``saveCheckpoint`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ts import NeuralTS

  agent = NeuralTS(state_dim=state_dim, action_dim=action_dim)   # Create NeuralTS agent

  checkpoint_path = "path/to/checkpoint"
  agent.saveCheckpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms.neural_ts import NeuralTS

  checkpoint_path = "path/to/checkpoint"
  agent = NeuralTS.load(checkpoint_path)

Parameters
----------

.. autoclass:: agilerl.algorithms.neural_ts.NeuralTS
  :members:
  :inherited-members:
