Usage
=====

.. _install:

Installation
------------

To use AgileRL, first download the source code and install requirements:

.. code-block:: console

   (.venv) $ git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
   (.venv) $ pip install -r requirements.txt

Training an RL agent
----------------

Before starting training, there are some meta-hyperparameters and settings that must be set.
These are defined in ``INIT_HP``, for general parameters, and ``MUTATION_PARAMS``, which define the evolutionary probabilities. For example:

.. code-block:: python

    INIT_HP = {
        'ENV_NAME': 'LunarLander-v2',   # Gym environment name
        'ALGO': 'DQN',                  # Algorithm
        'HIDDEN_SIZE': [64,64],         # Actor network hidden size
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

First, use ``utils.initialPopulation()`` to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(INIT_HP['ENV_NAME'], render_mode='rgb_array')
    num_states = env.observation_space.shape[0]
    try:
        num_actions = env.action_space.n
    except:
        num_actions = env.action_space.shape[0]

    agent_pop = initialPopulation(INIT_HP['ALGO'],
    num_states,
    num_actions,
    INIT_HP,
    INIT_HP['POP_SIZE'],
    device=device)

Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.

.. code-block:: python

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(num_actions, INIT_HP['MEMORY_SIZE'], field_names=field_names, device=device)

    tournament = TournamentSelection(INIT_HP['TOURN_SIZE'],
        INIT_HP['ELITISM'],
        INIT_HP['POP_SIZE'],
        INIT_HP['EVO_EPOCHS'])
        
    mutations = Mutations(no_mutation=MUTATION_PARAMS['NO_MUT'], 
        architecture=MUTATION_PARAMS['ARCH_MUT'], 
        new_layer_prob=MUTATION_PARAMS['NEW_LAYER'], 
        parameters=MUTATION_PARAMS['PARAMS_MUT'], 
        activation=MUTATION_PARAMS['ACT_MUT'], 
        rl_hp=MUTATION_PARAMS['RL_HP_MUT'], 
        rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'], 
        mutation_sd=MUTATION_PARAMS['MUT_SD'], 
        rand_seed=MUTATION_PARAMS['RAND_SEED'],
        device=device)

The easiest training loop implementation is to use our ``training.train()`` function. It requires the agent have functions ``getAction()`` and ``learn()``.

.. code-block:: python

    trained_pop, pop_fitnesses = train(env,
        INIT_HP['ENV_NAME'],
        INIT_HP['ALGO'],
        agent_pop,
        memory=memory,
        n_episodes=INIT_HP['EPISODES'],
        evo_epochs=INIT_HP['EVO_EPOCHS'],
        evo_loop=1,
        target=INIT_HP['TARGET_SCORE'],
        chkpt=INIT_HP['SAVE_CHKPT'],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP['WANDB'],
        device=device)
