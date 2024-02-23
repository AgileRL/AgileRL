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

    from agilerl.utils.utils import initialPopulation
    from agilerl.wrappers.learning import BanditEnv
    import torch
    from ucimlrepo import fetch_ucirepo

    NET_CONFIG = {
        'arch': 'mlp',  # Network architecture
        'h_size': [128],  # Actor hidden size
    }

    INIT_HP = {
        "POPULATION_SIZE": 4,  # Population size
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.000625,  # Loss regularization factor
        "LEARN_STEP": 1,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
    }

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    pop = initialPopulation(
        algo="NeuralUCB",  # Algorithm
        state_dim=context_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=None,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )

Experience Replay
-----------------

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``.
During training it can be added to using the ``ReplayBuffer.save2memory()`` method. To sample from the replay buffer, call ``ReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer
    import torch

    field_names = ["context", "reward"]
    memory = ReplayBuffer(
        action_dim=action_dim,  # Number of agent actions
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

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
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,  # Evaluate using last N fitness scores
    )

Mutation
--------

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

    mutations = Mutations(
        algo="NeuralUCB",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.5,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0.2,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=1,  # Random seed
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
        n_episodes=INIT_HP["EPISODES"],  # Max number of training episodes
        evo_epochs=INIT_HP["EVO_EPOCHS"],  # Evolution frequency
        evo_loop=1,  # Number of evaluation episodes per agent
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
    import wandb
    from tqdm import trange
    from ucimlrepo import fetch_ucirepo

    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.utils.utils import initialPopulation
    from agilerl.wrappers.learning import BanditEnv


    if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        NET_CONFIG = {
            "arch": "mlp",  # Network architecture
            "h_size": [128],  # Actor hidden size
        }

        INIT_HP = {
            "POPULATION_SIZE": 4,  # Population size
            "BATCH_SIZE": 64,  # Batch size
            "LR": 1e-3,  # Learning rate
            "GAMMA": 1.0,  # Scaling factor
            "LAMBDA": 1.0,  # Regularization factor
            "REG": 0.000625,  # Loss regularization factor
            "LEARN_STEP": 1,  # Learning frequency
            # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
            "CHANNELS_LAST": False,
        }

        # Fetch data  https://archive.ics.uci.edu/
        iris = fetch_ucirepo(id=53)
        features = iris.data.features
        targets = iris.data.targets

        env = BanditEnv(features, targets)  # Create environment
        context_dim = env.context_dim
        action_dim = env.arms

        pop = initialPopulation(
            algo="NeuralUCB",  # Algorithm
            state_dim=context_dim,  # State dimension
            action_dim=action_dim,  # Action dimension
            one_hot=None,  # One-hot encoding
            net_config=NET_CONFIG,  # Network configuration
            INIT_HP=INIT_HP,  # Initial hyperparameters
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            device=device,
        )

        field_names = ["context", "reward"]
        memory = ReplayBuffer(
            action_dim=action_dim,  # Number of agent actions
            memory_size=10000,  # Max replay buffer size
            field_names=field_names,  # Field names to store in memory
            device=device,
        )

        tournament = TournamentSelection(
            tournament_size=2,  # Tournament selection size
            elitism=True,  # Elitism in tournament selection
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            evo_step=1,
        )  # Evaluate using last N fitness scores

        mutations = Mutations(
            algo="NeuralUCB",  # Algorithm
            no_mutation=0.4,  # No mutation
            architecture=0.2,  # Architecture mutation
            new_layer_prob=0.5,  # New layer mutation
            parameters=0.2,  # Network parameters mutation
            activation=0.2,  # Activation layer mutation
            rl_hp=0.2,  # Learning HP mutation
            rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
            mutation_sd=0.1,  # Mutation strength
            arch=NET_CONFIG["arch"],  # Network architecture
            rand_seed=1,  # Random seed
            device=device,
        )

        max_episodes = 50  # Max training episodes
        max_steps = 100  # Max steps per episode

        evo_epochs = 2  # Evolution frequency
        evo_loop = 1  # Number of evaluation episodes

        print("Training...")

        regret = [[0] for _ in pop]

        wandb.init(
            # set the wandb project where this run will be logged
            project="AgileRL-Bandits",
            name="NeuralUCB-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")),
            # track hyperparameters and run metadata
            config=INIT_HP,
        )

        total_steps = 0

        # TRAINING LOOP
        for idx_epi in trange(max_episodes):
            for i, agent in enumerate(pop):  # Loop through population
                score = 0
                losses = []
                context = env.reset()  # Reset environment at start of episode
                for idx_step in range(max_steps):
                    # Get next action from agent
                    action = agent.getAction(context)
                    next_context, reward = env.step(action)  # Act in environment

                    # Save experience to replay buffer
                    memory.save2memory(context[action], reward)

                    # Learn according to learning frequency
                    if (
                        memory.counter % agent.learn_step == 0
                        and len(memory) >= agent.batch_size
                    ):
                        for _ in range(2):
                            experiences = memory.sample(
                                agent.batch_size
                            )  # Sample replay buffer
                            # Learn according to agent's RL algorithm
                            loss = agent.learn(experiences)
                            losses.append(loss)

                    context = next_context
                    score += reward
                    regret[i].append(regret[i][-1] + 1 - reward)

                total_steps += max_steps

                wandb_dict = {
                    "global_step": total_steps,
                    "train/loss": np.mean(losses),
                    "train/score": score,
                    "train/regret": regret[0][-1],
                }
                wandb.log(wandb_dict)

            # Now evolve population if necessary
            if (idx_epi + 1) % evo_epochs == 0:
                # Evaluate population
                fitnesses = [
                    agent.test(
                        env,
                        swap_channels=INIT_HP["CHANNELS_LAST"],
                        max_steps=max_steps,
                        loop=evo_loop,
                    )
                    for agent in pop
                ]

                print(f"Episode {idx_epi+1}/{max_episodes}")
                print(f"Regret: {[regret[i][-1] for i in range(len(pop))]}")

                # Tournament selection and population mutation
                elite, pop = tournament.select(pop)
                pop = mutations.mutation(pop)
