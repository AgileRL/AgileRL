Online Training
=====

In online reinforcement learning, an agent is able to gather data by directly interacting with its environment. It can then use this experience to learn from and 
update its policy. To enable our agent to interact in this way, the agent needs to act either in the real world, or in a simulation.

AgileRL's online training framework enables agents to learn in environments, using the standard Gym interface, 10x faster than SOTA by using our 
Evolutionary Hyperparameter Optimization algorithm.

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

    from agilerl.utils.utils import makeVectEnvs, initialPopulation
    import torch

    NET_CONFIG = {
        'arch': 'mlp',          # Network architecture
        'h_size': [32, 32],     # Actor hidden size
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

    env = makeVectEnvs('LunarLander-v2', num_envs=16)   # Create environment
    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n              # Discrete action space
    except:
        action_dim = env.single_action_space.shape[0]       # Continuous action space

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    agent_pop = initialPopulation(algo='DQN',               # Algorithm
                                  state_dim=state_dim,      # State dimension
                                  action_dim=action_dim,    # Action dimension
                                  one_hot=one_hot,          # One-hot encoding
                                  net_config=NET_CONFIG,    # Network configuration
                                  INIT_HP=INIT_HP,          # Initial hyperparameters
                                  population_size=6,        # Population size
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
    memory = ReplayBuffer(action_dim=action_dim,    # Number of agent actions
                          memory_size=10000,        # Max replay buffer size
                          field_names=field_names,  # Field names to store in memory
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

    tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=6, # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores


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

    mutations = Mutations(algo='DQN',                           # Algorithm
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


.. _trainloop:

Training Loop
------------

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment, it is 
easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train import train
    import gymnasium as gym
    import torch

    trained_pop, pop_fitnesses = train(env=env,                     # Gym-style environment
                                       env_name='LunarLander-v2',   # Environment name
                                       algo='DQN',                  # Algorithm
                                       pop=agent_pop,               # Population of agents
                                       memory=memory,               # Replay buffer
                                       swap_channels=False,         # Swap image channel from last to first
                                       n_episodes=1000,             # Max number of training episodes
                                       evo_epochs=20,               # Evolution frequency
                                       evo_loop=1,                  # Number of evaluation episodes per agent
                                       target=200.,                 # Target score for early stopping
                                       tournament=tournament,       # Tournament selection object
                                       mutation=mutations,          # Mutations object
                                       wb=False,                    # Weights and Biases tracking
                                       device=torch.device("cuda"))


Alternatively, use a custom training loop. Combining all of the above:

.. code-block:: python

    from agilerl.utils.utils import makeVectEnvs, initialPopulation
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.hpo.mutation import Mutations
    import gymnasium as gym
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

    pop = initialPopulation(algo='DQN',             # Algorithm
                            state_dim=(8,),         # State dimension
                            action_dim=4,           # Action dimension
                            one_hot=False,          # One-hot encoding
                            net_config=NET_CONFIG,  # Network configuration
                            INIT_HP=INIT_HP,        # Initial hyperparameters
                            population_size=6,      # Population size
                            device=torch.device("cuda"))

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=4,             # Number of agent actions
                          memory_size=10000,        # Max replay buffer size
                          field_names=field_names,  # Field names to store in memory
                          device=torch.device("cuda"))

    tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=6, # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores

    mutations = Mutations(algo='DQN',                           # Algorithm
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

    # Exploration params
    eps_start = 1.0     # Max exploration
    eps_end = 0.1       # Min exploration
    eps_decay = 0.995   # Decay per episode
    epsilon = eps_start

    evo_epochs = 5      # Evolution frequency
    evo_loop = 1        # Number of evaluation episodes

    env = makeVectEnvs('LunarLander-v2', num_envs=16)   # Create environment

    # TRAINING LOOP
    for idx_epi in range(max_episodes):
        for agent in pop:   # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0
            for idx_step in range(max_steps):
                action = agent.getAction(state, epsilon)    # Get next action from agent
                next_state, reward, done, _, _ = env.step(action)   # Act in environment
                
                # Save experience to replay buffer
                memory.save2memoryVectEnvs(state, action, reward, next_state, done)

                # Learn according to learning frequency
                if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
                    experiences = memory.sample(agent.batch_size) # Sample replay buffer
                    agent.learn(experiences)    # Learn according to agent's RL algorithm
                
                state = next_state
                score += reward

        epsilon = max(eps_end, epsilon*eps_decay) # Update epsilon for exploration

        # Now evolve population if necessary
        if (idx_epi+1) % evo_epochs == 0:
            
            # Evaluate population
            fitnesses = [agent.test(env, swap_channels=False, max_steps=max_steps, loop=evo_loop) for agent in pop]

            print(f'Episode {idx_epi+1}/{max_episodes}')
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
