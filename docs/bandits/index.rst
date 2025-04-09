.. _bandits:

Contextual Multi-Armed Bandits
==============================

Bandit algorithms solve a special case of reinforcement learning, when there is only one timestep.
Contextual multi-armed bandits are a framework for decision-making where an algorithm chooses between multiple
options (arms) to maximize its reward, with each choice informed by the current context or situation. The
algorithm learns over time which arm is likely to yield the best outcome based on the context, improving its
decisions through a balance of exploring new options and exploiting known rewarding options. This approach is
widely used in areas such as personalized recommendations, adaptive content delivery, and optimal strategy selection.

In this framework, the "context" refers to any relevant information available at the time of making a decision,
which could include user profiles, environmental conditions, or historical interactions. The algorithm uses this
information to predict the potential reward of each action within the specific context, aiming to choose the action
that maximizes expected rewards. Over time, as it accumulates more data from its choices and their outcomes, it
refines its predictions and strategy. This adaptive learning process allows for more personalized and efficient
decision-making, as the algorithm becomes better at identifying which actions are most beneficial under different circumstances.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`NeuralUCB<neural_ucb>`
     - :ref:`Iris Dataset<neural_ucb_tutorial>`
   * - :ref:`NeuralTS<neural_ts>`
     - :ref:`PenDigits Dataset<neural_ts_tutorial>`


Population Creation and Environment Setup
-----------------------------------------

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below.

To demonstrate our bandit algorithms, we will use a labelled dataset from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_. These can easily
be imported and used for training with the Python package ``ucimlrepo``, and to choose from the hundreds of available datasets it is as simple as changing the
``id`` parameter used by ``fetch_uci_repo``.
We can convert these labelled datasets into a bandit learning environment easily by using the ``agilerl.wrappers.learning.BanditEnv`` class.

.. code-block:: python

    from agilerl.utils.utils import create_population
    from agilerl.wrappers.learning import BanditEnv
    import torch
    from ucimlrepo import fetch_ucirepo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "encoder_config": {"hidden_size": [128]},  # Encoder hidden size
    }

    INIT_HP = {
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.000625,  # Loss regularization factor
        "LEARN_STEP": 2,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POP_SIZE": 4,  # Population size
    }

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr = RLParameter(min=6.25e-5, max=1e-2),
        batch_size = RLParameter(min=8, max=512, dtype=int),
        learn_step = RLParameter(min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75)
    )

    obs_space = spaces.Box(low=features.values.min(), high=features.values.max())
    action_space = spaces.Discrete(action_dim)
    pop = create_population(
        algo="NeuralUCB",  # Algorithm
        observation_space=obs_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        hp_config=hp_config,  # Hyperparameter configuration
        population_size=INIT_HP["POP_SIZE"],  # Population size
        device=device,
    )

Experience Replay
-----------------

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``.
During training we use the ``ReplayBuffer.add()`` function to add experiences to the buffer as ``TensorDict`` objects. Specifically, we wrap transitions through the
``Transition`` tensorclass that wraps the ``obs``, ``action``, ``reward``, ``next_obs``, and ``done`` fields as ``torch.Tensor`` objects. To sample from the replay
buffer, call ``ReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer

    memory = ReplayBuffer(
        max_size=10000,  # Max replay buffer size
        device=device,
    )


Bandit Training Loop
-----------------------

The easiest way to train a population of bandits is to use our training function:

.. code-block:: python

    from agilerl.training.train_bandits import train_bandits

    trained_pop, pop_fitnesses = train_bandits(
        env,  # Bandit environment
        INIT_HP["ENV_NAME"],  # Environment name
        "NeuralUCB",  # Algorithm
        agent_pop,  # Population of agents
        memory=memory,  # Experience replay buffer
        INIT_HP=INIT_HP,  # Initial hyperparameters
        MUT_P=MUTATION_PARAMS,  # Mutation parameters
        swap_channels=INIT_HP["CHANNELS_LAST"],  # Swap image channel from last to first
        max_steps=10000,  # Max number of training steps
        episode_steps=500,  # Steps in episode
        evo_steps=500,  # Evolution frequency
        eval_steps=500,  # Number of steps in evaluation episode,
        eval_loop=1,  # Number of evaluation episodes
        target=INIT_HP["TARGET_SCORE"],  # Target score for early stopping
        tournament=tournament,  # Tournament selection object
        mutation=mutations,  # Mutations object
        wb=INIT_HP["WANDB"],  # Weights and Biases tracking
    )

Alternatively, use a custom bandit training loop:

.. code-block:: python

    from datetime import datetime

    import numpy as np
    import torch
    from tensordict import TensorDict
    from tqdm import trange
    from ucimlrepo import fetch_ucirepo

    import wandb
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.utils.utils import create_population
    from agilerl.wrappers.learning import BanditEnv


    if __name__ == "__main__":
    print("===== AgileRL Bandit Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "hidden_size": [128],  # Actor hidden size
    }

    INIT_HP = {
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.000625,  # Loss regularization factor
        "LEARN_STEP": 2,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POP_SIZE": 4,  # Population size
    }

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    obs_space = spaces.Box(low=features.values.min(), high=features.values.max())
    action_space = spaces.Discrete(action_dim)
    pop = create_population(
        algo="NeuralUCB",  # Algorithm
        observation_space=obs_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        device=device,
    )

    memory = ReplayBuffer(max_size=10000, device=device)

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )
    mutations = Mutations(
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.5,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0.2,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 10000  # Max steps per episode
    episode_steps = 500  # Steps in episode
    evo_steps = 500  # Evolution frequency
    eval_steps = 500  # Evaluation steps per episode
    eval_loop = 1  # Number of evaluation episodes

    print("Training...")

    wandb.init(
        # set the wandb project where this run will be logged
        project="AgileRL-Bandits",
        name="NeuralUCB-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")),
        # track hyperparameters and run metadata
        config=INIT_HP,
    )

    total_steps = 0
    evo_count = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            score = 0
            losses = []
            context = env.reset()  # Reset environment at start of episode
            for idx_step in range(episode_steps):
                if INIT_HP["CHANNELS_LAST"]:
                    context = obs_channels_to_first(context)
                # Get next action from agent
                action = agent.get_action(context)
                next_context, reward = env.step(action)  # Act in environment

                transition = TensorDict(
                    {
                        "obs": context[action],
                        "reward": reward,
                    },
                ).float()
                transition.batch_size = [1]
                # Save experience to replay buffer
                memory.add(transition)

                # Learn according to learning frequency
                if len(memory) >= agent.batch_size:
                    for _ in range(agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        experiences = memory.sample(agent.batch_size)
                        loss = agent.learn(experiences)
                        losses.append(loss)

                context = next_context
                score += reward
                agent.regret.append(agent.regret[-1] + 1 - reward)

            agent.scores.append(score)
            pop_episode_scores.append(score)
            agent.steps[-1] += episode_steps
            total_steps += episode_steps
            pbar.update(episode_steps // len(pop))

            wandb_dict = {
                "global_step": total_steps,
                "train/loss": np.mean(losses),
                "train/score": score,
                "train/mean_regret": np.mean([agent.regret[-1] for agent in pop]),
            }
            wandb.log(wandb_dict)

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

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Regret: {[agent.regret[-1] for agent in pop]}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        if pop[0].steps[-1] // evo_steps > evo_count:
            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            evo_count += 1

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
