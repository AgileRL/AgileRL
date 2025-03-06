Offline Training
================

Offline reinforcement learning learns exclusively from static datasets of previously collected interactions, making it feasible to extract policies from
large and diverse training datasets. Effective offline RL algorithms have a much wider range of applications than online RL, being particularly appealing
for real-world applications, such as education, healthcare, and robotics. ref: https://arxiv.org/abs/2203.01387

AgileRL's offline RL training framework enables you to leverage evolutionary HPO for faster training on your own datasets, without the need for a simulator.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`CQL<cql>`
     - --
   * - :ref:`ILQL<ilql>`
     - --

.. _initpop_offline:

Population Creation
-------------------

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below.

.. code-block:: python

    from agilerl.utils.utils import create_population, make_vect_envs, observation_space_channels_to_first
    import gymnasium as gym
    import h5py
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INIT_HP = {
        "DOUBLE": True,  # Use double Q-learning
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target network parameters
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 1
    env = make_vect_envs("CartPole-v1", num_envs=num_envs)  # Create environment
    dataset = h5py.File("data/cartpole/cartpole_random_v1.1.0.h5", "r")  # Load dataset

    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP['CHANNELS_LAST']:
        observation_space = observation_space_channels_to_first(observation_space)

    # RL hyperparameter configuration for mutations
    hp_config = HyperparameterConfig(
        lr = RLParameter(min=1e-4, max=1e-2),
        batch_size = RLParameter(min=8, max=64, dtype=int),
        learn_step = RLParameter(
            min=1, max=120, dtype=int, grow_factor=1.5, shrink_factor=0.75
            )
    )

    pop = create_population(
        algo="CQN",  # Algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        hp_config=hp_config,  # RL hyperparameters configuration
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized envs
        device=device,
    )

.. _memory_offline:

Experience Replay
-----------------

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``.
During training it can be added to using the ``ReplayBuffer.save_to_memory()`` function, or ``ReplayBuffer.save_to_memory_vect_envs()`` for vectorized environments (recommended).
To sample from the replay buffer, call ``ReplayBuffer.sample()``.

We must fill the replay buffer with our offline data so that we can sample and learn.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    print("Filling replay buffer with dataset...")
    # Save transitions to replay buffer
    dataset_length = dataset["rewards"].shape[0]
    for i in trange(dataset_length - 1):
        state = dataset["observations"][i]
        next_state = dataset["observations"][i + 1]
        if INIT_HP["CHANNELS_LAST"]:
            state = obs_channels_to_first(state)
            next_state = obs_channels_to_first(next_state)
        action = dataset["actions"][i]
        reward = dataset["rewards"][i]
        done = bool(dataset["terminals"][i])
        # Save experience to replay buffer
        memory.save_to_memory(state, action, reward, next_state, done)


.. _trainloop_offline:

Training Loop
-------------

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment, it is
easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train_offline import train_offline

    trained_pop, pop_fitnesses = train_offline(
        env=env,  # Gym-style environment
        env_name="CartPole-v1",  # Environment name
        dataset=dataset,  # Offline dataset
        pop=pop,  # Population of agents
        memory=memory,  # Replay buffer
        swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
        max_steps=500000,  # Max number of training steps
        evo_steps=10000,  # Evolution frequency
        eval_steps=None,  # Evaluation steps
        eval_loop=1,  # Number of evaluation episodes per agent
        target=200.,  # Target score for early stopping
        tournament=tournament,  # Tournament selection object
        mutation=mutations,  # Mutations object
        wb=True,  # Weights and Biases tracking
    )


Alternatively, use a custom training loop. Combining all of the above:

.. code-block:: python

    import h5py
    import numpy as np
    import torch
    from tqdm import trange

    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.utils.utils import (
        create_population,
        make_vect_envs,
        observation_space_channels_to_first
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "head_config": {"hidden_size": [32, 32]}  # Actor head hidden size
    }

    INIT_HP = {
        "DOUBLE": True,  # Use double Q-learning
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target network parameters
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 1
    env = make_vect_envs("CartPole-v1", num_envs=num_envs)  # Create environment
    dataset = h5py.File("data/cartpole/cartpole_random_v1.1.0.h5", "r")  # Load dataset

    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP['CHANNELS_LAST']:
        observation_space = observation_space_channels_to_first(observation_space)

    pop = create_population(
        algo="CQN",  # Algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized envs
        device=device,
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    print("Filling replay buffer with dataset...")
    # Save transitions to replay buffer
    dataset_length = dataset["rewards"].shape[0]
    for i in trange(dataset_length - 1):
        state = dataset["observations"][i]
        next_state = dataset["observations"][i + 1]
        if INIT_HP["CHANNELS_LAST"]:
            state = obs_channels_to_first(state)
            next_state = obs_channels_to_first(next_state)
        action = dataset["actions"][i]
        reward = dataset["rewards"][i]
        done = bool(dataset["terminals"][i])
        # Save experience to replay buffer
        memory.save_to_memory(state, action, reward, next_state, done)

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 200000  # Max steps

    evo_steps = 10000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        for agent in pop:  # Loop through population
            for idx_step in range(max_steps):
                experiences = memory.sample(agent.batch_size)  # Sample replay buffer
                agent.learn(experiences)  # Learn according to agent's RL algorithm
            total_steps += max_steps
            agent.steps[-1] += max_steps

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

        print(f"--- Global Steps {total_steps} ---")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
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
