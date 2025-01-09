.. _multiagenttraining:

Multi-Agent Training
====================

In multi-agent reinforcement learning, multiple agents are trained to act in the same environment in both
co-operative and competitive scenarios. With AgileRL, agents can be trained to act in multi-agent environments
using our implementation of the MADDPG or MATD3 algorithms alongside our Evolutionary Hyperparameter
Optimisation algorithm.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`MADDPG<maddpg>`
     - :ref:`Space Invaders <MADDPG tutorial>`
   * - :ref:`MATD3<matd3>`
     - :ref:`Simple Speaker Listener <MATD3 tutorial>`


.. _initpop_ma:

Population Creation
-------------------

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below. At present, evolutionary
hyper-parameter tuning is only compatible with **cooperative** multi-agent environments.

.. code-block:: python

    from agilerl.utils.utils import create_population
    from pettingzoo.mpe import simple_speaker_listener_v4
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 32,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 100,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 8
    # Define the simple speaker listener environment as a parallel environment
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.single_observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.single_observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.single_action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.single_action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.single_action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.single_action_space(agent).low for agent in env.agents]

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = create_population(
        "MADDPG",
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

.. _memory:

Experience Replay
-----------------

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``MultiAgentReplayBuffer()`` for
multi-agent environments. During training it can be added to using the ``MultiAgentReplayBuffer.save_to_memory()`` function and sampled using the  ``MultiAgentReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

.. _tournament:

Tournament Selection
--------------------

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If elitism is used, the best agent from a population
is automatically preserved and becomes a member of the next generation. Then, for each tournament, k individuals are randomly chosen, and the agent with the best evaluation
fitness is preserved. This is repeated until the population for the next generation is full.

The class ``TournamentSelection()`` defines the functions required for tournament selection. ``TournamentSelection.select()`` returns the best agent, and the new generation
of agents.

.. code-block:: python

    from agilerl.hpo.tournament import TournamentSelection

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )
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

    mutations = Mutations(
        algo="MADDPG",
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        agent_ids=INIT_HP["AGENT_IDS"],
        rand_seed=1,
        device=device,
    )

.. _trainloop:

Training Loop
-------------

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment (e.g. pettingzoo
for multi-agent environments) it is easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train_multi_agent import train_multi_agent
    import gymnasium as gym
    import torch

    trained_pop, pop_fitnesses = train_multi_agent(
        env=env,  # Pettingzoo-style environment
        env_name='simple_speaker_listener_v4',  # Environment name
        algo="MADDPG",  # Algorithm
        pop=agent_pop,  # Population of agents
        memory=memory,  # Replay buffer
        INIT_HP=INIT_HP,  # IINIT_HP dictionary
        MUT_P=MUTATION_PARAMS,  # MUTATION_PARAMS dictionary
        net_config=NET_CONFIG,  # Network configuration
        swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
        max_steps=2000000,  # Max number of training steps
        evo_steps=10000,  # Evolution frequency
        eval_steps=None,  # Number of steps in evaluation episode
        eval_loop=1,  # Number of evaluation episodes
        learning_delay=1000,  # Steps before starting learning
        target=200.,  # Target score for early stopping
        tournament=tournament,  # Tournament selection object
        mutation=mutations,  # Mutations object
        wb=False,  # Weights and Biases tracking
    )


Alternatively, use a custom training loop. Combining all of the above:

.. code-block:: python

    import numpy as np
    import torch
    from pettingzoo.mpe import simple_speaker_listener_v4
    from tqdm import trange

    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.utils.utils import create_population
    from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 32,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 100,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 8
    # Define the simple speaker listener environment as a parallel environment
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
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
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = create_population(
        "MADDPG",
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        algo="MADDPG",
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        agent_ids=INIT_HP["AGENT_IDS"],
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_steps = 1000000  # Max steps
    learning_delay = 0  # Steps before starting learning

    evo_steps = 10000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            if INIT_HP["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(s, [-1], [-3])
                    for agent_id, s in state.items()
                }

            for idx_step in range(evo_steps // num_envs):

                # Get next action from agent
                cont_actions, discrete_action = agent.get_action(
                    states=state,
                    training=True,
                    infos=info
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

                # Image processing if necessary for the environment
                if INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }

                # Save experiences to replay buffer
                memory.save_to_memory(
                    state,
                    cont_actions,
                    reward,
                    next_state,
                    termination,
                    is_vectorised=True,
                )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)
                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

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

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
