Get started
=====

.. _install:

Install AgileRL
------------

To use AgileRL, first download the source code and install requirements.

Install as a package with pip: 

.. code-block:: bash

   pip install agilerl

Or install in development mode: (Recommended due to nascent nature of this library)

.. code-block:: bash

   git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
   pip install -r requirements.txt


Quickstart: Training an online RL agent
----------------

Before starting training, there are some meta-hyperparameters and settings that must be set.
These are defined in ``INIT_HP``, for general parameters, ``MUTATION_PARAMS``, which define the evolutionary 
probabilities, and ``NET_CONFIG``, which defines the network architecture. For example:

.. code-block:: python

    INIT_HP = {
        'ENV_NAME': 'LunarLander-v2',   # Gym environment name
        'ALGO': 'DQN',                  # Algorithm
        'DOUBLE': True,                 # Use double Q-learning
        'CHANNELS_LAST': False,         # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'BATCH_SIZE': 256,              # Batch size
        'LR': 1e-3,                     # Learning rate
        'EPISODES': 2000,               # Max no. episodes
        'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                  # Discount factor
        'MEMORY_SIZE': 10000,           # Max memory buffer size
        'LEARN_STEP': 1,                # Learning frequency
        'TAU': 1e-3,                    # For soft update of target parameters
        'TOURN_SIZE': 2,                # Tournament size
        'ELITISM': True,                # Elitism in tournament selection
        'POP_SIZE': 6,                  # Population size
        'EVO_EPOCHS': 20,               # Evolution frequency
        'POLICY_FREQ': 2,               # Policy network update frequency
        'WANDB': True                   # Log with Weights and Biases
    }

.. code-block:: python

    MUTATION_PARAMS = {
        # Relative probabilities
        'NO_MUT': 0.4,                              # No mutation
        'ARCH_MUT': 0.2,                            # Architecture mutation
        'NEW_LAYER': 0.2,                           # New layer mutation
        'PARAMS_MUT': 0.2,                          # Network parameters mutation
        'ACT_MUT': 0,                               # Activation layer mutation
        'RL_HP_MUT': 0.2,                           # Learning HP mutation
        'RL_HP_SELECTION': ['lr', 'batch_size'],    # Learning HPs to choose from
        'MUT_SD': 0.1,                              # Mutation strength
        'RAND_SEED': 1,                             # Random seed
    }

.. code-block:: python

    NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32], # Actor hidden size
    }

First, use ``utils.utils.initialPopulation()`` to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.

.. code-block:: python

    from agilerl.utils.utils import makeVectEnvs, initialPopulation
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = makeVectEnvs(env_name=INIT_HP['ENV_NAME'], num_envs=16)
    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except Exception:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n             # Discrete action space
    except Exception:
        action_dim = env.single_action_space.shape[0]      # Continuous action space

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    agent_pop = initialPopulation(algo=INIT_HP['ALGO'],     # Algorithm
                                  state_dim=state_dim,      # State dimension
                                  action_dim=action_dim,    # Action dimension
                                  one_hot=one_hot,          # One-hot encoding
                                  net_config=NET_CONFIG,    # Network configuration
                                  INIT_HP=INIT_HP,          # Initial hyperparameters
                                  population_size=6,        # Population size
                                  device=torch.device("cuda"))

Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.hpo.mutation import Mutations
    import torch

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim,                # Number of agent actions
                          memory_size=INIT_HP['MEMORY_SIZE'],   # Max replay buffer size
                          field_names=field_names,              # Field names to store in memory
                          device=torch.device("cuda"))

    tournament = TournamentSelection(tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
                                     elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
                                     population_size=INIT_HP['POP_SIZE'],   # Population size
                                     evo_step=INIT_HP['EVO_EPOCHS'])        # Evaluate using last N fitness scores

    mutations = Mutations(algo=INIT_HP['ALGO'],                                 # Algorithm
                          no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
                          architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
                          new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
                          parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
                          activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
                          rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
                          rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],   # Learning HPs to choose from
                          mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
                          arch=NET_CONFIG['arch'],                              # Network architecture
                          rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
                          device=torch.device("cuda"))

The easiest training loop implementation is to use our ``training.train()`` function. It requires the agent have functions ``getAction()`` and ``learn()``.

.. code-block:: python

    from agilerl.training.train import train

    trained_pop, pop_fitnesses = train(env=env,                                 # Gym-style environment
                                       env_name=INIT_HP['ENV_NAME'],            # Environment name
                                       algo=INIT_HP['ALGO'],                    # Algorithm
                                       pop=agent_pop,                           # Population of agents
                                       memory=memory,                           # Replay buffer
                                       swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
                                       n_episodes=INIT_HP['EPISODES'],          # Max number of training episodes
                                       evo_epochs=INIT_HP['EVO_EPOCHS'],        # Evolution frequency
                                       evo_loop=1,                              # Number of evaluation episodes per agent
                                       target=INIT_HP['TARGET_SCORE'],          # Target score for early stopping
                                       tournament=tournament,                   # Tournament selection object
                                       mutation=mutations,                      # Mutations object
                                       wb=INIT_HP['WANDB'])                     # Weights and Biases tracking

Quickstart: Training an offline RL agent
----------------

Like with online RL, above, there are some meta-hyperparameters and settings that must be set before starting training. These are defined in ``INIT_HP``, for general parameters, and ``MUTATION_PARAMS``, which define the evolutionary probabilities, and ``NET_CONFIG``, which defines the network architecture. For example:

.. code-block:: python
    
    INIT_HP = {
        'ENV_NAME': 'CartPole-v1',      # Gym environment name
        'DATASET': 'data/cartpole/cartpole_random_v1.1.0.h5', # Offline RL dataset
        'ALGO': 'CQN',                  # Algorithm
        'DOUBLE': True,                 # Use double Q-learning
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False,
        'BATCH_SIZE': 256,              # Batch size
        'LR': 1e-3,                     # Learning rate
        'EPISODES': 2000,               # Max no. episodes
        'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                  # Discount factor
        'MEMORY_SIZE': 10000,           # Max memory buffer size
        'LEARN_STEP': 1,                # Learning frequency
        'TAU': 1e-3,                    # For soft update of target parameters
        'TOURN_SIZE': 2,                # Tournament size
        'ELITISM': True,                # Elitism in tournament selection
        'POP_SIZE': 6,                  # Population size
        'EVO_EPOCHS': 20,               # Evolution frequency
        'POLICY_FREQ': 2,               # Policy network update frequency
        'WANDB': True                   # Log with Weights and Biases
    }

.. code-block:: python

    MUTATION_PARAMS = {
        # Relative probabilities
        'NO_MUT': 0.4,                              # No mutation
        'ARCH_MUT': 0.2,                            # Architecture mutation
        'NEW_LAYER': 0.2,                           # New layer mutation
        'PARAMS_MUT': 0.2,                          # Network parameters mutation
        'ACT_MUT': 0,                               # Activation layer mutation
        'RL_HP_MUT': 0.2,                           # Learning HP mutation
        'RL_HP_SELECTION': ['lr', 'batch_size'],    # Learning HPs to choose from
        'MUT_SD': 0.1,                              # Mutation strength
        'RAND_SEED': 1,                             # Random seed
    }

.. code-block:: python

    NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32], # Actor hidden size
    }

First, use ``utils.utils.initialPopulation`` to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.

.. code-block:: python

    from agilerl.utils.utils import makeVectEnvs, initialPopulation
    import torch
    import h5py
    import gymnasium as gym

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = makeVectEnvs(INIT_HP['ENV_NAME'], num_envs=1)
    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except Exception:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n             # Discrete action space
    except Exception:
        action_dim = env.single_action_space.shape[0]      # Continuous action space

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    dataset = h5py.File(INIT_HP['DATASET'], 'r')

    agent_pop = initialPopulation(algo=INIT_HP['ALGO'],                 # Algorithm
                                  state_dim=state_dim,                  # State dimension
                                  action_dim=action_dim,                # Action dimension
                                  one_hot=one_hot,                      # One-hot encoding
                                  net_config=NET_CONFIG,                # Network configuration
                                  INIT_HP=INIT_HP,                      # Initial hyperparameters
                                  population_size=INIT_HP['POP_SIZE'],  # Population size
                                  device=torch.device("cuda"))

Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.hpo.mutation import Mutations
    import torch

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim,                # Number of agent actions
                          memory_size=INIT_HP['MEMORY_SIZE'],   # Max replay buffer size
                          field_names=field_names,              # Field names to store in memory
                          device=torch.device("cuda"))

    tournament = TournamentSelection(tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
                                     elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
                                     population_size=INIT_HP['POP_SIZE'],   # Population size
                                     evo_step=INIT_HP['EVO_EPOCHS'])        # Evaluate using last N fitness scores

    mutations = Mutations(algo=INIT_HP['ALGO'],                                 # Algorithm
                          no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
                          architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
                          new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
                          parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
                          activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
                          rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
                          rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],   # Learning HPs to choose from
                          mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
                          arch=NET_CONFIG['arch'],                              # Network architecture
                          rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
                          device=torch.device("cuda"))

The easiest training loop implementation is to use our ``training.train_offline.train()`` function. It requires the ``agent`` have functions ``getAction()`` and ``learn().``

.. code-block:: python

    from agilerl.training.train_offline import train

    trained_pop, pop_fitnesses = train(env=env,                                 # Gym-style environment
                                       env_name=INIT_HP['ENV_NAME'],            # Environment name
                                       dataset=dataset,                         # Offline dataset
                                       algo=INIT_HP['ALGO'],                    # Algorithm
                                       pop=agent_pop,                           # Population of agents
                                       memory=memory,                           # Replay buffer
                                       swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
                                       n_episodes=INIT_HP['EPISODES'],          # Max number of training episodes
                                       evo_epochs=INIT_HP['EVO_EPOCHS'],        # Evolution frequency
                                       evo_loop=1,                              # Number of evaluation episodes per agent
                                       target=INIT_HP['TARGET_SCORE'],          # Target score for early stopping
                                       tournament=tournament,                   # Tournament selection object
                                       mutation=mutations,                      # Mutations object
                                       wb=INIT_HP['WANDB'])                     # Weights and Biases tracking
