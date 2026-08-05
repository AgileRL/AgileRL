.. _off_policy:

Off-Policy Training
===================

In online reinforcement learning, an agent is able to gather data by directly interacting with its environment. It can then use this experience to learn from and
update its policy. To enable our agent to interact in this way, the agent needs to act either in the real world, or in a simulation.

AgileRL's online training framework enables agents to learn in environments, using the standard Gym interface, 10x faster than SOTA by using our
Evolutionary Hyperparameter Optimization algorithm.

Off-policy reinforcement learning involves decoupling the learning policy from the data collection policy. Algorithms like Q-learning and DDPG enable learning
from experiences collected by a different, possibly exploratory policy, allowing for greater flexibility in exploration and improved sample efficiency. By learning
from a diverse set of experiences, off-policy methods can leverage past data more effectively, separating the exploration strategy from the learning strategy and
enabling the agent to learn optimal policies even from suboptimal or random exploration policies. This independence between data collection and learning policies
often results in higher potential for reuse of previously gathered experiences and facilitates more efficient learning.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`DQN <dqn>`
     - :ref:`Curriculum learning with self-play <DQN tutorial>`
   * - :ref:`Rainbow DQN <dqn_rainbow>`
     - :ref:`Cartpole <rainbow_tutorial>`
   * - :ref:`DDPG <ddpg>`
     - --
   * - :ref:`TD3 <td3>`
     - :ref:`Lunar Lander <td3_tutorial>`


.. _off_policy_trainer:

Training with LocalTrainer
~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to train a fully-customised off-policy agent is through a YAML manifest and the
:class:`~agilerl.training.trainer.LocalTrainer`. This handles population
creation, replay buffers, evolutionary HPO, and the training loop automatically.

Here is an example manifest to train DQN on LunarLander-v3:

.. collapse:: dqn.yaml

  .. code-block:: yaml

    algorithm:
      name: DQN
      batch_size: 128
      lr: 6.3e-4
      learn_step: 4
      gamma: 0.99
      tau: 0.001

    environment:
      name: LunarLander-v3
      num_envs: 16

    training:
      max_steps: 1_000_000
      target_score: 200.0
      pop_size: 4
      evo_steps: 10_000

    network:
      latent_dim: 128
      encoder_config:
        hidden_size: [128]
      head_config:
        hidden_size: [128]

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
        lr:
          min: 0.0000625
          max: 0.01
        batch_size:
          min: 8
          max: 512
      mutation_sd: 0.1
      rand_seed: 42

    tournament_selection:
      tournament_size: 2
      elitism: true

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from agilerl import LocalTrainer

         trainer = LocalTrainer.from_manifest("dqn.yaml")
         population, fitnesses = trainer.train()

   .. tab-item:: CLI

      .. code-block:: bash

         python -m agilerl.train dqn.yaml

.. seealso::

   :ref:`trainers` for full manifest reference and additional options.


Customised Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _initpop_off_policy:

Population Creation
^^^^^^^^^^^^^^^^^^^

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below.

.. collapse:: Create a Population of DQN Agents
  :open:

  .. code-block:: python

    import torch

    from agilerl.algorithms import DQN
    from agilerl.utils.utils import make_vect_envs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize vectorized environments
    num_envs = 16
    env = make_vect_envs("LunarLander-v3", num_envs=num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # Configure network architecture
    net_config = {
        "encoder_config": {
            "hidden_size": [32, 32]
        },
        "head_config": {
            "hidden_size": [32, 32]
        }
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
    pop = DQN.population(
        size=population_size,
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config,
        device=device,
        **init_hp,
    )


.. _memory_off_policy:

Experience Replay
^^^^^^^^^^^^^^^^^

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

Evolutionary HPO
^^^^^^^^^^^^^^^^^

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If
elitism is used, the best agent from a population is automatically preserved and becomes a member of the next generation.
Then, for each tournament, *k* individuals are randomly chosen, and the agent with the best evaluation fitness is preserved.
This is repeated until the population for the next generation is full.

Mutation is periodically used to explore the hyperparameter space, allowing different hyperparameter combinations to be
trialled during training. If certain hyperparameters prove relatively beneficial to training, then that agent is more
likely to be preserved in the next generation, and so those characteristics are more likely to remain in the population.

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

.. _trainloop_off_policy:

Training Loop
^^^^^^^^^^^^^

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment, you
can use our off-the-shelf training function, which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train_off_policy import train_off_policy

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,  # Gym-style environment
        env_name="LunarLander-v3",  # Environment name
        algo="DQN",  # Algorithm
        pop=pop,  # Population of agents
        memory=memory,  # Replay buffer
        init_hp=init_hp,  # Algorithm hyperparameters
        max_steps=200000,  # Max number of training steps
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

.. collapse:: Custom Training Loop

    .. code-block:: python

        from agilerl.algorithms import DQN
        from agilerl.components.replay_buffer import ReplayBuffer
        from agilerl.components.data import Transition
        from agilerl.hpo.mutation import Mutations
        from agilerl.hpo.tournament import TournamentSelection
        from agilerl.utils.utils import make_vect_envs, default_progress_bar
        import numpy as np
        import torch
        from tqdm import trange

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize vectorized environments
        num_envs = 16
        env = make_vect_envs("LunarLander-v3", num_envs=num_envs)
        observation_space = env.single_observation_space
        action_space = env.single_action_space

        # Configure network architecture
        net_config = {
            "encoder_config": {
                "hidden_size": [32, 32]
            },
            "head_config": {
                "hidden_size": [32, 32]
            }
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
        pop = DQN.population(
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

        # Evo-HPO
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
            mutation_sd=0.1,  # Mutation strength
            rand_seed=1,  # Random seed
            device=device,
        )

        # Training parameters
        max_steps = 200000  # Max steps
        learning_delay = 1000  # Steps before starting learning
        eps_start = 1.0  # Max exploration
        eps_end = 0.1  # Min exploration
        eps_decay = 0.995  # Decay per episode
        epsilon = eps_start
        evo_steps = 10000  # Evolution frequency
        eval_steps = None  # Evaluation steps per episode - go until done
        eval_loop = 1  # Number of evaluation episodes
        total_steps = 0

        # TRAINING LOOP
        pbar = default_progress_bar(max_steps)
        while np.less([agent.steps for agent in pop], max_steps).all():
            pop_episode_scores = []
            for agent in pop:  # Loop through population
                agent.set_training_mode(True)

                obs, info = env.reset()  # Reset environment at start of episode
                scores = np.zeros(num_envs)
                completed_episode_scores = []
                steps = 0
                epsilon = eps_start

                for idx_step in range(evo_steps // num_envs):
                    action = agent.get_action(obs, epsilon)  # Get next action from agent
                    epsilon = max(
                        eps_end, epsilon * eps_decay
                    )  # Decay epsilon for exploration

                    # Act in environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    scores += np.array(reward)
                    steps += num_envs
                    total_steps += num_envs

                    # Collect scores for completed episodes
                    for idx, (d, t) in enumerate(zip(terminated, truncated)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0

                    # Wrap transition as TensorDict
                    transition = Transition(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        done=terminated,
                        batch_size=[num_envs]
                    )
                    transition = transition.to_tensordict()

                    # Save experience to replay buffer
                    memory.add(transition)

                    # Learn according to learning frequency
                    if memory.size > learning_delay and len(memory) >= agent.batch_size:
                        for _ in range(num_envs // agent.learn_step):
                            experiences = memory.sample(
                                agent.batch_size
                            )  # Sample replay buffer
                            agent.learn(
                                experiences
                            )  # Learn according to agent's RL algorithm

                    obs = next_obs

                pbar.update(evo_steps // len(pop))
                agent.steps += steps
                pop_episode_scores.append(completed_episode_scores)

            # Reset epsilon start to latest decayed value for next round of population training
            eps_start = epsilon

            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
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

            pbar.write(
                f"--- Global steps {total_steps} --- \n"
                f"Steps: {[agent.steps for agent in pop]} \n"
                f"Scores: {mean_scores} \n"
                f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]} \n'
                f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}',
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

        pbar.close()
        env.close()

.. tutorial::

   :ref:`rainbow_tutorial`
      Rainbow DQN on CartPole-v1.

   :ref:`td3_tutorial`
      TD3 on LunarLander-v3.

   :ref:`DQN tutorial`
      Curriculum learning and self-play with DQN.
