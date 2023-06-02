Offline Training
=====

Offline reinforcement learning learns exclusively from static datasets of previously collected interactions, making it feasible to extract policies from 
large and diverse training datasets. Effective offline RL algorithms have a much wider range of applications than online RL, being particularly appealing 
for real-world applications, such as education, healthcare, and robotics. ref: https://arxiv.org/abs/2203.01387

AgileRL's offline RL training framework enables you to leverage evolutionary HPO for faster training on your own datasets, without the need for a simulator.

.. _evoHPO:

Evolutionary Hyperparameter Optimization
------------

Traditionally, hyperparameter optimization (HPO) for reinforcement learning (RL) is particularly difficult when compared to other types of machine learning.  
This is for several reasons, including the relative sample inefficiency of RL and its sensitivity to hyperparameters.

AgileRL is initially focused on improving HPO for RL in order to allow faster development with robust training. 
Evolutionary algorithms have been shown to allow faster, automatic convergence to optimal hyperparameters than other HPO methods by taking advantage of 
shared memory between a population of agents acting in identical environments.

At regular intervals, after learning from shared experiences, a population of agents can be evaluated in an environment. Through tournament selection, the 
best agents are selected to survive until the next generation, and their offspring are mutated to further explore the hyperparameter space.
Eventually, the optimal hyperparameters for learning in a given environment can be reached in significantly less steps than are required using other HPO methods.


.. _initpop:
Population Creation
------------

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to 
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters 
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below.

.. code-block:: python

    from agilerl.utils.utils import initialPopulation
    import gymnasium as gym
    import h5py
    import torch

    NET_CONFIG = {
        'arch': 'mlp',          # Network architecture
        'h_size': [32, 32],     # Actor hidden size
    }

    INIT_HP = {
        'ENV_NAME': 'CartPole-v1',      # Gym environment name
        'DATASET': 'data/cartpole/cartpole_random_v1.1.0.h5', # Offline RL dataset
        'DOUBLE': True,                 # Use double Q-learning
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False,
        'BATCH_SIZE': 256,              # Batch size
        'LR': 1e-3,                     # Learning rate
        'EPISODES': 2000,               # Max no. episodes
        'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                  # Discount factor
        'LEARN_STEP': 1,                # Learning frequency
        'TAU': 1e-3,                    # For soft update of target parameters
        'TOURN_SIZE': 2,                # Tournament size
        'ELITISM': True,                # Elitism in tournament selection
        'POP_SIZE': 6,                  # Population size
        'EVO_EPOCHS': 20,               # Evolution frequency
        'POLICY_FREQ': 2,               # Policy network update frequency
        'WANDB': True                   # Log with Weights and Biases
        }

    env = gym.make(INIT_HP['ENV_NAME'])
    try:
        state_dim = env.observation_space.n       # Discrete observation space
        one_hot = True                            # Requires one-hot encoding
    except Exception:
        state_dim = env.observation_space.shape   # Continuous observation space
        one_hot = False                           # Does not require one-hot encoding
    try:
        action_dim = env.action_space.n           # Discrete action space
    except Exception:
        action_dim = env.action_space.shape[0]    # Continuous action space

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


.. _memory:

Experience Replay
------------

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed 
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve 
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself. 

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``. 
During training it can be added to using the ``ReplayBuffer.save2memory()`` function, or ``ReplayBuffer.save2memoryVectEnvs()`` for vectorized environments (recommended). 
To sample from the replay buffer, call ``ReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer
    import torch

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim,                # Number of agent actions
                          memory_size=INIT_HP['MEMORY_SIZE'],   # Max replay buffer size
                          field_names=field_names,              # Field names to store in memory
                          device=torch.device("cuda"))



.. _tournament:

Tournament Selection
------------

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If elitism is used, the best agent from a population 
is automatically preserved and becomes a member of the next generation. Then, for each tournament, k individuals are randomly chosen, and the agent with the best evaluation 
fitness is preserved. This is repeated until the population for the next generation is full.

The class ``TournamentSelection()`` defines the functions required for tournament selection. ``TournamentSelection.select()`` returns the best agent, and the new generation 
of agents.

.. code-block:: python

    from agilerl.hpo.tournament import TournamentSelection

    tournament = TournamentSelection(tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
                                     elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
                                     population_size=INIT_HP['POP_SIZE'],   # Population size
                                     evo_step=INIT_HP['EVO_EPOCHS'])        # Evaluate using last N fitness scores


.. _mutate:

Mutation
------------

Mutation is periodically used to explore the hyperparameter space, allowing different hyperparameter combinations to be trialled during training. If certain hyperparameters 
prove relatively beneficial to training, then that agent is more likely to be preserved in the next generation, and so those characteristics are more likely to remain in the 
population.

The ``Mutations()`` class is used to mutate agents with pre-set probabilities. The available mutations currently implemented are:
    * No mutation
    * Network architecture mutation - adding layers or nodes. Trained weights are reused and new weights are initialized randomly.
    * Network parameters mutation - mutating weights with Gaussian noise.
    * Network activation layer mutation - change of activation layer.
    * RL algorithm mutation - mutation of learning hyperparameter, such as learning rate or batch size.

``Mutations.mutation()`` returns a mutated population.

Tournament selection and mutation should be applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

    from agilerl.hpo.mutation import Mutations
    import torch

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


.. _trainloop:

Training Loop
------------

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment, it is 
easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train_offline import train
    import gymnasium as gym
    import torch

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


Alternatively, use a custom training loop. Combining all of the above:

.. code-block:: python

    from agilerl.utils.utils import initialPopulation
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.hpo.mutation import Mutations
    import gymnasium as gym
    import h5py
    import numpy as np
    import torch

    NET_CONFIG = {
                    'arch': 'mlp',       # Network architecture
                    'h_size': [32, 32],  # Actor hidden size
                }

    INIT_HP = {
                'DOUBLE': True,         # Use double Q-learning
                'BATCH_SIZE': 128,      # Batch size
                'LR': 1e-3,             # Learning rate
                'GAMMA': 0.99,          # Discount factor
                'LEARN_STEP': 1,        # Learning frequency
                'TAU': 1e-3,            # For soft update of target network parameters
                'CHANNELS_LAST': False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
            }

    env = gym.make('CartPole-v1')   # Create environment
    dataset = h5py.File('data/cartpole/cartpole_random_v1.1.0.h5', 'r')  # Load dataset

    try:
        state_dim = env.observation_space.n       # Discrete observation space
        one_hot = True                            # Requires one-hot encoding
    except Exception:
        state_dim = env.observation_space.shape   # Continuous observation space
        one_hot = False                           # Does not require one-hot encoding
    try:
        action_dim = env.action_space.n           # Discrete action space
    except Exception:
        action_dim = env.action_space.shape[0]    # Continuous action space

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    pop = initialPopulation(algo='CQN',             # Algorithm
                            state_dim=state_dim,    # State dimension
                            action_dim=action_dim,  # Action dimension
                            one_hot=one_hot,        # One-hot encoding
                            net_config=NET_CONFIG,  # Network configuration
                            INIT_HP=INIT_HP,        # Initial hyperparameters
                            population_size=6,      # Population size
                            device=torch.device("cuda"))

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim,    # Number of agent actions
                          memory_size=10000,        # Max replay buffer size
                          field_names=field_names,  # Field names to store in memory
                          device=torch.device("cuda"))

    tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=6, # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores

    mutations = Mutations(algo='CQN',                           # Algorithm
                          no_mutation=0.4,                      # No mutation
                          architecture=0.2,                     # Architecture mutation
                          new_layer_prob=0.2,                   # New layer mutation
                          parameters=0.2,                       # Network parameters mutation
                          activation=0,                         # Activation layer mutation
                          rl_hp=0.2,                            # Learning HP mutation
                          rl_hp_selection=['lr', 'batch_size'], # Learning HPs to choose from
                          mutation_sd=0.1,                      # Mutation strength
                          arch=NET_CONFIG['arch'],              # Network architecture
                          rand_seed=1,                          # Random seed
                          device=torch.device("cuda"))

    max_episodes = 1000 # Max training episodes
    max_steps = 500     # Max steps per episode

    evo_epochs = 5      # Evolution frequency
    evo_loop = 1        # Number of evaluation episodes

    # Save transitions to replay buffer
    dataset_length = dataset['rewards'].shape[0]
    for i in range(dataset_length-1):
        state = dataset['observations'][i]
        next_state = dataset['observations'][i+1]
        if INIT_HP['CHANNELS_LAST']:
            state = np.moveaxis(state, [3], [1])
            next_state = np.moveaxis(next_state, [3], [1])
        action = dataset['actions'][i]
        reward = dataset['rewards'][i]
        done = bool(dataset['terminals'][i])
        memory.save2memory(state, action, reward, next_state, done)

    # TRAINING LOOP
    for idx_epi in range(max_episodes):
        for agent in pop:   # Loop through population
            for idx_step in range(max_steps):
                experiences = memory.sample(agent.batch_size)   # Sample replay buffer
                # Learn according to agent's RL algorithm
                agent.learn(experiences)

        # Now evolve population if necessary
        if (idx_epi+1) % evo_epochs == 0:
            
            # Evaluate population
            fitnesses = [agent.test(env, swap_channels=INIT_HP['CHANNELS_LAST'], max_steps=max_steps, loop=evo_loop) for agent in pop]

            print(f'Episode {idx_epi+1}/{max_episodes}')
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)