.. _multiagenttraining:
Multi-Agent Training
=====

In multi-agent reinforcement learning, multiple agents are trained to act in the same environment in both
co-operative and competitive scenarios. With AgileRL, agents can be trained to act in multi-agent environments
using our implementation of the MADDPG or MATD3 algorithms alongside our Evolutionary Hyperparameter
Optimisation algorithm.

.. _initpop:
Population Creation
------------

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below.

.. code-block:: python

    from agilerl.utils.utils import initialPopulation
    from pettingzoo.mpe import simple_speaker_listener_v4
    import torch

    NET_CONFIG = {
        'arch': 'mlp',          # Network architecture
        'h_size': [32, 32],     # Actor hidden size
    }

    INIT_HP = {
        'ALGO': 'MADDPG',                  # Algorithm
        'BATCH_SIZE': 1024,             # Batch size
        'LR': 0.01,                     # Learning rate
        'EPISODES': 10_000,             # Max no. episodes
        'GAMMA': 0.95,                  # Discount factor
        'MEMORY_SIZE': 1_000_000,       # Max memory buffer size
        'LEARN_STEP': 5,                # Learning frequency
        'TAU': 0.01,                    # For soft update of target parameters
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
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
        INIT_HP['DISCRETE_ACTIONS'] = True
        INIT_HP['MAX_ACTION'] = None
        INIT_HP['MIN_ACTION'] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP['DISCRETE_ACTIONS'] = False
        INIT_HP['MAX_ACTION'] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP['MIN_ACTION'] = [env.action_space(agent).low for agent in env.agents]

    if INIT_HP['CHANNELS_LAST']:
        state_dim = [(state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim]

    INIT_HP['N_AGENTS'] = env.num_agents
    INIT_HP['AGENT_IDS'] = [agent_id for agent_id in env.agents]

    agent_pop = initialPopulation(algo=INIT_HP['ALGO'],
                                  state_dim=state_dim,
                                  action_dim=action_dim,
                                  one_hot=one_hot,
                                  net_config=NET_CONFIG,
                                  INIT_HP=INIT_HP,
                                  population_size=6,
                                  device=device)

.. _memory:

Experience Replay
------------

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``MultiAgentReplayBuffer()`` for
multi-agent environments. During training it can be added to using the ``MultiAgentReplayBuffer.save2memory()`` function and sampled using the  ``MultiAgentReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    import torch

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(memory_size=1_000_000,        # Max replay buffer size
                                    field_names=field_names,  # Field names to store in memory
                                    agent_ids=INIT_HP['AGENT_IDS'],
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

    mutations = Mutations(algo=INIT_HP['ALGO'],                 # Algorithm
                          no_mutation=0.4,                      # No mutation
                          architecture=0.2,                     # Architecture mutation
                          new_layer_prob=0.2,                   # New layer mutation
                          parameters=0.2,                       # Network parameters mutation
                          activation=0,                         # Activation layer mutation
                          rl_hp=0.2,                            # Learning HP mutation
                          rl_hp_selection=['lr', 'batch_size'], # Learning HPs to choose from
                          mutation_sd=0.1,                      # Mutation strength
                          agent_ids=INIT_HP['AGENT_IDS'],
                          arch=NET_CONFIG['arch'],              # Network architecture
                          rand_seed=1,                          # Random seed
                          device=torch.device("cuda"))

.. _trainloop:

Training Loop
------------

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment (e.g. pettingzoo
for multi-agent environments) it is easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train_multi_agent import train_multi_agent
    import gymnasium as gym
    import torch

    trained_pop, pop_fitnesses = train_multi_agent(env=env,                              # Pettingzoo-style environment
                                                env_name='simple_speaker_listener_v4',   # Environment name
                                                algo=INIT_HP['ALGO'],                    # Algorithm
                                                pop=agent_pop,                           # Population of agents
                                                memory=memory,                           # Replay buffer
                                                INIT_HP=INIT_HP,                         # IINIT_HP dictionary
                                                MUT_P=MUTATION_PARAMS,                   # MUTATION_PARAMS dictionary
                                                net_config=NET_CONFIG,                   # Network configuration
                                                swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
                                                n_episodes=1000,                         # Max number of training episodes
                                                evo_epochs=20,                           # Evolution frequency
                                                evo_loop=1,                              # Number of evaluation episodes per agent
                                                max_steps=900,                           # Max steps to take in the environment
                                                target=200.,                             # Target score for early stopping
                                                tournament=tournament,                   # Tournament selection object
                                                mutation=mutations,                      # Mutations object
                                                wb=INIT_HP["WANDB"])                     # Weights and Biases tracking


Alternatively, use a custom training loop. Combining all of the above:

.. code-block:: python

    from agilerl.utils.utils import initialPopulation
    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.hpo.mutation import Mutations
    from pettingzoo.mpe import simple_speaker_listener_v4
    import numpy as np
    import torch

    NET_CONFIG = {
        'arch': 'mlp',          # Network architecture
        'h_size': [32, 32],     # Actor hidden size
    }

    INIT_HP = {
        'ALGO': 'MADDPG',                  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'BATCH_SIZE': 1024,             # Batch size
        'LR': 0.01,                     # Learning rate
        'EPISODES': 10_000,             # Max no. episodes
        'GAMMA': 0.95,                  # Discount factor
        'MEMORY_SIZE': 1_000_000,       # Max memory buffer size
        'LEARN_STEP': 5,                # Learning frequency
        'TAU': 0.01,                    # For soft update of target parameters
        'CHANNELS_LAST': False          # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
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
        INIT_HP['DISCRETE_ACTIONS'] = True
        INIT_HP['MAX_ACTION'] = None
        INIT_HP['MIN_ACTION'] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP['DISCRETE_ACTIONS'] = False
        INIT_HP['MAX_ACTION'] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP['MIN_ACTION'] = [env.action_space(agent).low for agent in env.agents]

    if INIT_HP['CHANNELS_LAST']:
        state_dim = [(state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim]

    INIT_HP['N_AGENTS'] = env.num_agents
    INIT_HP['AGENT_IDS'] = [agent_id for agent_id in env.agents]

    agent_pop = initialPopulation(algo=INIT_HP['ALGO'],
                                  state_dim=state_dim,
                                  action_dim=action_dim,
                                  one_hot=one_hot,
                                  net_config=NET_CONFIG,
                                  INIT_HP=INIT_HP,
                                  population_size=6,
                                  device=device)

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(memory_size=1_000_000,        # Max replay buffer size
                                    field_names=field_names,  # Field names to store in memory
                                    agent_ids=INIT_HP['AGENT_IDS'],
                                    device=torch.device("cuda"))

    tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=6, # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores

    mutations = Mutations(algo=INIT_HP['ALGO'],                           # Algorithm
                          no_mutation=0.4,                      # No mutation
                          architecture=0.2,                     # Architecture mutation
                          new_layer_prob=0.2,                   # New layer mutation
                          parameters=0.2,                       # Network parameters mutation
                          activation=0,                         # Activation layer mutation
                          rl_hp=0.2,                            # Learning HP mutation
                          rl_hp_selection=['lr', 'batch_size'], # Learning HPs to choose from
                          mutation_sd=0.1,                      # Mutation strength
                          agent_ids=INIT_HP['AGENT_IDS'],
                          arch=NET_CONFIG['arch'],              # Network architecture
                          rand_seed=1,                          # Random seed
                          device=torch.device("cuda"))

    max_episodes = 10_000 # Max training episodes
    max_steps = 25        # Max steps per episode

    # Exploration params
    eps_start = 1.0     # Max exploration
    eps_end = 0.1       # Min exploration
    eps_decay = 0.995   # Decay per episode
    epsilon = eps_start

    evo_epochs = 5      # Evolution frequency
    evo_loop = 1        # Number of evaluation episodes

    # TRAINING LOOP
    for idx_epi in range(max_episodes):
        if accelerator is not None:
            accelerator.wait_for_everyone()
        for agent in pop:   # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            agent_reward = {agent_id: 0 for agent_id in env.agents}

            while env.agents:
                total_steps += 1
                if swap_channels:
                    state = np.moveaxis(state, [3], [1])
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)   # Act in environment

                # Save experience to replay buffer
                if swap_channels:
                    memory.save2memory(
                        state, action, reward, np.moveaxis(next_state, [3], [1]), done)
                else:
                    memory.save2memory(
                        state, action, reward, next_state, done)

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                #Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (len(
                        memory) >= agent.batch_size):
                    # Sample replay buffer
                    experiences = sampler.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

                state = next_state

            score = sum(agent_reward.values())
            agent.scores.append(score)

            agent.steps[-1] += total_steps

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve population if necessary
        if (idx_epi+1) % evo_epochs == 0:

            # Evaluate population
            fitnesses = [agent.test(env, swap_channels=False, max_steps=max_steps, loop=evo_loop) for agent in pop]

            # Update step counter
            for agent in pop:
                agent.steps.append(agent.steps[-1])

            print(f'Episode {idx_epi+1}/{max_episodes}')
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

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
