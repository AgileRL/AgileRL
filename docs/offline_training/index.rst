Offline Training
================

Offline reinforcement learning learns exclusively from static datasets of previously collected interactions, making it feasible to extract policies from
large and diverse training datasets. Effective offline RL algorithms have a much wider range of applications than online RL, being particularly appealing
for real-world applications, such as education, healthcare, and robotics. (`A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems <https://arxiv.org/abs/2203.01387>`_)

AgileRL's offline RL training framework enables you to leverage evolutionary HPO for faster training on your own datasets, without the need for a simulator.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`CQL <cql>`
     - --
   * - :ref:`ILQL <ilql>`
     - --

Training with LocalTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to train an offline RL agent is with a YAML manifest and the
:class:`~agilerl.training.trainer.LocalTrainer`. This handles population
creation, dataset loading, replay buffers, evolutionary HPO, and the training
loop automatically.

Below is an example manifest for training CQN on the CartPole-v1 environment (Minari dataset).

.. collapse:: cqn.yaml

   .. code-block:: yaml

      algorithm:
        name: CQN
        batch_size: 256
        lr: 0.001
        learn_step: 1
        gamma: 0.99
        tau: 0.001
        double: true

      environment:
        name: CartPole-v1
        num_envs: 16
        minari_dataset_id: cartpole/random-v0

      training:
        max_steps: 50_000
        target_score: 200.0
        pop_size: 4
        evo_steps: 5_000
        learning_delay: 1000

      network:
        latent_dim: 64
        encoder_config:
          hidden_size: [64]
        head_config:
          hidden_size: [64]

      replay_buffer:
        max_size: 100_000

      mutation:
        probabilities:
          no_mut: 0.4
          arch_mut: 0.2
          new_layer: 0.2
          params_mut: 0.2
          act_mut: 0.2
          rl_hp_mut: 0.2
        rl_hp_selection:
          lr:   { min: 0.0001, max: 0.01 }
          batch_size: { min: 8, max: 1024 }
        mutation_sd: 0.1
        rand_seed: 42

      selection_strategy:
        tournament_size: 2
        elitism: true

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from agilerl import LocalTrainer

         trainer = LocalTrainer.from_manifest("cqn.yaml")
         population, fitnesses = trainer.train()

   .. tab-item:: CLI

      .. code-block:: bash

         python -m agilerl.train cqn.yaml

.. seealso::

   :ref:`trainers` for full manifest reference and additional options.


Customised Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _initpop_offline:

Population Creation and Environment Setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below. The referenced
CartPole-v1 dataset can be found in the `AgileRL repository <https://github.com/AgileRL/AgileRL/blob/main/data/cartpole/>`_.

.. collapse:: Population Creation and Environment Setup
    :open:

    .. code-block:: python

        import gymnasium as gym
        import h5py
        import torch

        from agilerl.algorithms import CQN
        from agilerl.utils.utils import make_vect_envs

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create environment and load offline dataset
        num_envs = 1
        env = make_vect_envs("CartPole-v1", num_envs=num_envs)  # Create environment
        dataset = h5py.File("data/cartpole/cartpole_random_v1.1.0.h5", "r")  # Load dataset

        observation_space = env.single_observation_space
        action_space = env.single_action_space

        # Configure network architecture
        net_config = {
            "encoder_config": {"hidden_size": [32, 32]},  # Encoder hidden size
            "head_config": {"hidden_size": [32]},  # Head hidden size
        }

        # Algorithm hyperparameters
        init_hp = {
            "double": True,
            "batch_size": 128,
            "lr": 1e-3,
            "gamma": 0.99,
            "learn_step": 1,
            "tau": 1e-3,
        }

        # Initialize population
        pop = CQN.population(
            size=4,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            device=device,
            **init_hp,
        )

.. _memory_offline:

Experience Replay
^^^^^^^^^^^^^^^^^

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``.
During training we use the ``ReplayBuffer.add()`` function to add experiences to the buffer as ``TensorDict`` objects. Specifically, we wrap transitions through the
``Transition`` tensorclass that wraps the ``obs``, ``action``, ``reward``, ``next_obs``, and ``done`` fields as ``torch.Tensor`` objects. To sample from the replay
buffer, call ``ReplayBuffer.sample()``.

We must fill the replay buffer with our offline data so that we can sample and learn.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.components.data import Transition

    memory = ReplayBuffer(
        max_size=10000,  # Max replay buffer size
        device=device,
    )

    # Save transitions to replay buffer
    dataset_length = dataset["rewards"].shape[0]
    for i in trange(dataset_length - 1):
        state = dataset["observations"][i]
        next_obs = dataset["observations"][i + 1]
        action = dataset["actions"][i]
        reward = dataset["rewards"][i]
        done = bool(dataset["terminals"][i])

        transition = Transition(
            obs=state,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
        )
        transition = transition.unsqueeze(0) # Add vectorized dimension
        transition.batch_size = [1]

        # Save experience to replay buffer
        memory.add(transition.to_tensordict())


Evolutionary HPO
^^^^^^^^^^^^^^^^^

Tournament selection is used to select the agents from a population which will make up the next generation of agents.
Mutation is periodically used to explore the hyperparameter space.

.. code-block:: python

    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=4,  # Population size
    )

    mutations = Mutations(
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,  # Random seed
        device=device,
    )

.. seealso::

   :ref:`evo_hyperparam_opt` for details on how evolutionary HPO works.

.. _trainloop_offline:

Training Loop
^^^^^^^^^^^^^

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
        init_hp=init_hp,  # Algorithm hyperparameters
        max_steps=500000,  # Max number of training steps
        evo_steps=10000,  # Evolution frequency
        eval_steps=None,  # Evaluation steps
        eval_loop=1,  # Number of evaluation episodes per agent
        target=200.,  # Target score for early stopping
        selection_strategy=tournament,  # Selection strategy object
        mutation=mutations,  # Mutations object
        wb=True,  # Weights and Biases tracking
    )


Alternatively, use a custom training loop. Combining all of the above:

.. collapse:: Custom Training Loop

    .. code-block:: python

        import h5py
        import numpy as np
        import torch
        from tqdm import trange

        from agilerl.algorithms import CQN
        from agilerl.components.replay_buffer import ReplayBuffer
        from agilerl.hpo.mutation import Mutations
        from agilerl.hpo.tournament import TournamentSelection
        from agilerl.utils.utils import make_vect_envs, default_progress_bar

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create vectorized environment
        num_envs = 1
        env = make_vect_envs("CartPole-v1", num_envs=num_envs)  # Create environment
        dataset = h5py.File("data/cartpole/cartpole_random_v1.1.0.h5", "r")  # Load dataset
        observation_space = env.single_observation_space
        action_space = env.single_action_space

        # Configure network architecture
        net_config = {
            "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},  # Encoder config
            "head_config": {"hidden_size": [32]},  # Head hidden size
        }

        # Algorithm hyperparameters
        init_hp = {
            "double": True,
            "batch_size": 128,
            "lr": 1e-3,
            "gamma": 0.99,
            "learn_step": 1,
            "tau": 1e-3,
        }

        # Initialize population
        population_size = 4
        pop = CQN.population(
            size=population_size,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            device=device,
            **init_hp,
        )

        memory = ReplayBuffer(
            max_size=10000,  # Max replay buffer size
            device=device,
        )

        print("Filling replay buffer with dataset...")
        # Save transitions to replay buffer
        dataset_length = dataset["rewards"].shape[0]
        for i in trange(dataset_length - 1):
            obs = dataset["observations"][i]
            next_obs = dataset["observations"][i + 1]
            action = dataset["actions"][i]
            reward = dataset["rewards"][i]
            done = bool(dataset["terminals"][i])

            # Save experience to replay buffer
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
            transition = transition.unsqueeze(0) # Add vectorized dimension
            transition.batch_size = [1]

            memory.add(transition.to_tensordict())

        # Tournament and mutations for Evo-HPO
        tournament = TournamentSelection(
            tournament_size=2,  # Tournament selection size
            elitism=True,  # Elitism in tournament selection
            population_size=population_size,  # Population size
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
        pbar = default_progress_bar(max_steps)
        while np.less([agent.steps for agent in pop], max_steps).all():
            for agent in pop:  # Loop through population
                for idx_step in range(evo_steps):
                    experiences = memory.sample(agent.batch_size)  # Sample replay buffer
                    agent.learn(experiences)  # Learn according to agent's RL algorithm

                total_steps += evo_steps
                agent.steps += evo_steps
                pbar.update(evo_steps // len(pop))

            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    max_steps=eval_steps,
                    loop=eval_loop,
                )
                for agent in pop
            ]

            pbar.write(
                f"--- Global Steps {total_steps} ---\n"
                f"Steps: {[agent.steps for agent in pop]}\n"
                f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}\n'
                f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop, _ = tournament.select(pop)
            pop = mutations.mutation(pop)

        pbar.close()
        env.close()
