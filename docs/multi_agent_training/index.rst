.. _multiagenttraining:

Multi-Agent Training
====================

In multi-agent reinforcement learning, multiple agents are trained to act in the same environment in both
co-operative and competitive scenarios. With AgileRL, agents can be trained to act in multi-agent environments
using our implementation of several multi-agent algorithms alongside Evolutionary Hyperparameter Optimisation.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`MADDPG<maddpg>`
     - :ref:`Space Invaders <MADDPG tutorial>`
   * - :ref:`MATD3<matd3>`
     - :ref:`Simple Speaker Listener <MATD3 tutorial>`
   * - :ref:`IPPO<ippo>`
     -

Formulation
-----------
AgileRL builds on the `PettingZoo <https://pettingzoo.farama.org/>`_ framework for multi-agent environments. In this framework, each agent is identified by a unique ID, and the environment
is defined by a set of agents. Multi-agent algorithms in AgileRL have an ``agent_ids`` argument which should be passed in from the possible agents in the environment, alongside the lists of
``observation_spaces`` and ``action_spaces``, whereby the space at index ``i`` is the observation/action space for the agent with ID ``agent_ids[i]``.

Agent Definitions
~~~~~~~~~~~~~~~~~
In AgileRL we also follow the convention that agent IDs should be formatted by their homogeneity as ``<group_id>_<agent_idx>``. For example, if we have a multi-agent setting with agents
``[bob_0, bob_1, fred_0, fred_1]``, the assumption is that the agents with the same prefix (or ``group_id``) as separated by ``_`` are homogeneous (i.e. have the same observation space and are
interchangeable). This allows us to automatically create centralized policies where suitable (please refer to :ref:`IPPO <ippo>` for more details).

Vectorised Environments
~~~~~~~~~~~~~~~~~~~~~~~
We implement our own wrapper to vectorise multi-agent environments through the :class:`AsyncPettingZooVecEnv <agilerl.vector.pz_async_vec_env.AsyncPettingZooVecEnv>` class, which
contains a shared memory buffer. In order to create a vectorised environment, users can also make use of the :func:`make_multi_agent_vect_envs() <agilerl.utils.utils.make_multi_agent_vect_envs>`
function.

.. code-block:: python

    from pettingzoo.mpe import simple_speaker_listener_v4

    from agilerl.utils.utils import make_multi_agent_vect_envs

    # Define the environment
    def make_env():
        return simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    # Vectorise the environment
    env = make_multi_agent_vect_envs(make_env, num_envs=8)

.. _multi_agent_networks:

Configuring Network Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Network architectures in multi-agent settings are configured in the same way as single-agent settings through the ``net_config`` argument of an algorithm. The main
difference lies in the ability to pass this in as a nested dictionary including the configurations for individual agents or groups of agents that are homogeneous.
In other words, instead of passing in ``net_config`` as the arguments to an individual ``EvolvableNetwork``, users can choose to pass the configurations to the networks
of different agents / agent groups in an algorithm.

If we have a setting with the following possible agents with their respective observation and action spaces:

.. collapse:: Environment definition
    :open:

    .. code-block:: python

        from gymnasium.spaces import Box, Discrete

        agent_ids = ["bob_0", "bob_1", "fred_0", "fred_1"]
        observation_spaces = [
            Box(low=-1, high=1, shape=(16,)), # bob_0
            Box(low=-1, high=1, shape=(16,)), # bob_1
            Box(low=-1, high=1, shape=(32,)), # fred_0
            Box(low=-1, high=1, shape=(32,)), # fred_1
        ]
        action_spaces = [
            Discrete(2), # bob_0
            Discrete(2), # bob_1
            Discrete(2), # fred_0
            Discrete(2), # fred_1
        ]

We could specify the architecture for individual agents as follows in a yaml file:

.. collapse:: Configuring architectures for individual agents

    .. code-block:: yaml

        bob_0:
            latent_dim: 32
            encoder_config:
                hidden_size: [32]
                activation: ReLU
            head_config:
                hidden_size: [32]
        bob_1:
            latent_dim: 32
            encoder_config:
                hidden_size: [64, 64]
                activation: ReLU
            head_config:
                hidden_size: [32]
        fred_0:
            latent_dim: 32
            encoder_config:
                hidden_size: [64, 64]
                activation: ReLU
            head_config:
                hidden_size: [32]
        fred_1:
            latent_dim: 32
            encoder_config:
                hidden_size: [64, 64]
                activation: ReLU
            head_config:
                hidden_size: [32]

Alternatively, we could specify the architectures for homogeneous agents as a group:

.. collapse:: Configuring architectures for homogeneous agents

    .. code-block:: yaml

        bob:
            latent_dim: 32
            encoder_config:
                hidden_size: [32]
                activation: ReLU
            head_config:
                hidden_size: [32]
        fred:
            latent_dim: 32
            encoder_config:
                hidden_size: [64, 64]
                activation: ReLU
            head_config:
                hidden_size: [32]

In simple situations where all agents can use the same architecture (i.e. require the same encoder type to process observations), we can also pass a single-level
``net_config`` like in single-agent settings. In the above example, since all observations can be processed using an ``EvolvableMLP`` network, we could pass the
following which would assign the same network architecture to all agents:

.. collapse:: Configuring a single network architecture for all agents

    .. code-block:: yaml

        latent_dim: 32
        encoder_config:
            hidden_size: [32]
            activation: ReLU
        head_config:
            hidden_size: [32]

Parameter Sharing
~~~~~~~~~~~~~~~~~
It is common in multi-agent settings to require centralized policies for groups of homogeneous agents during training for scalability, since the number of trainable parameters
can increase significantly with the number of agents. In this manner, we obtain a more sample efficient training process. Currently, AgileRL only includes the
:class:`IPPO <agilerl.algorithms.ippo.IPPO>` algorithm which supports this. In such cases, we restrict users to pass in network configurations to the groups directly. For the
setting described above, we could only use the latter configuration.

Asynchronous Agents
~~~~~~~~~~~~~~~~~~~
We often encounter settings where agents don't act simultaneously, but rather do so asynchronously in turns or with different frequencies. AgileRL follows
the convention that such environments only return observations for agents that should act in the following timestep. To handle these scenarios, we've implemented the
:class:`AsyncAgentsWrapper <agilerl.wrappers.agent.AsyncAgentsWrapper>` class, which automatically processes observations and actions to be compatible with
``AsyncPettingZooVecEnv``.

.. warning::
    The :class:`AsyncAgentsWrapper <agilerl.wrappers.agents.AsyncAgentsWrapper>` class is currently only compatible with the
    :class:`IPPO <agilerl.algorithms.ippo.IPPO>` algorithm.

.. _initpop_ma:

Evolutionary Hyperparameter Optimisation
----------------------------------------

To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but learn individually, allowing us to
determine the efficacy of certain hyperparameters. Individual agents which learn best are more likely to survive until the next generation, and so their hyperparameters
are more likely to remain present in the population. The sequence of evolution (tournament selection followed by mutation) is detailed further below. At present, evolutionary
hyper-parameter tuning is only compatible with **cooperative** multi-agent environments.

.. _multi_off_policy:

Off-Policy Training
-------------------

Similarly to single-agent settings, off-policy learning in multi-agent settings involves learning a target policy from data generated by a behaviour policy. AgileRL
currently includes implementations of :class:`MADDPG <agilerl.algorithms.maddpg.MADDPG>` and :class:`MATD3 <agilerl.algorithms.matd3.MATD3>`.

Creating a Population of Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the snippet below, we show an example of how to create a population of MADDPG agents for the simple speaker listener environment.

.. collapse:: Create a population of MADDPG agents

    .. code-block:: python

        from agilerl.utils.utils import create_population
        from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
        from pettingzoo.mpe import simple_speaker_listener_v4
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the network configuration
        NET_CONFIG = {
            "speaker_0": {
                "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},
                "head_config": {"hidden_size": [32]},
            },
            "listener_0": {
                "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},
                "head_config": {"hidden_size": [32]},
            },
        }

        # Define the initial hyperparameters
        INIT_HP = {
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
        env = AsyncPettingZooVecEnv(
            [
                lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
                for _ in range(num_envs)
            ]
        )
        env.reset()

        # Configure the multi-agent algo input arguments
        observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
        action_spaces = [env.single_action_space(agent) for agent in env.agents]

        # Append number of agents and agent IDs to the initial hyperparameter dictionary
        INIT_HP["AGENT_IDS"] = env.agents

        # Mutation config for RL hyperparameters
        hp_config = HyperparameterConfig(
            lr_actor = RLParameter(min=1e-4, max=1e-2),
            lr_critic = RLParameter(min=1e-4, max=1e-2),
            batch_size = RLParameter(min=8, max=512, dtype=int),
            learn_step = RLParameter(
                min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
                )
        )

        # Create a population ready for evolutionary hyper-parameter optimisation
        pop = create_population(
            "MADDPG",
            observation_spaces,
            action_spaces,
            NET_CONFIG,
            INIT_HP,
            hp_config,
            population_size=INIT_HP["POP_SIZE"],
            num_envs=num_envs,
            device=device,
        )

.. _memory:

Experience Replay
~~~~~~~~~~~~~~~~~

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

.. _trainloop:

Training Loop
~~~~~~~~~~~~~

Now it is time to insert the evolutionary HPO components into our training loop. If you are using a Gym-style environment (e.g. pettingzoo
for multi-agent environments) it is easiest to use :func:`train_multi_agent_off_policy() <agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy>`,
which returns a population of trained agents and logged training metrics.

.. code-block:: python

    from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy

    trained_pop, pop_fitnesses = train_multi_agent_off_policy(
        env=env,  # Pettingzoo-style environment
        env_name='simple_speaker_listener_v4',  # Environment name
        algo="MADDPG",  # Algorithm
        pop=pop,  # Population of agents
        memory=memory,  # Replay buffer
        INIT_HP=INIT_HP,  # IINIT_HP dictionary
        net_config=NET_CONFIG,  # Network configuration
        max_steps=2000000,  # Max number of training steps
        evo_steps=10000,  # Evolution frequency
        eval_steps=None,  # Number of steps in evaluation episode
        eval_loop=1,  # Number of evaluation episodes
        learning_delay=1000,  # Steps before starting learning
        target=-30.0,  # Target score for early stopping
        tournament=tournament,  # Tournament selection object
        mutation=mutations,  # Mutations object
        wb=False,  # Weights and Biases tracking
    )


Alternatively, use a custom training loop. Combining all of the above:

.. collapse:: Custom training loop

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
            "speaker_0": {
                "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},
                "head_config": {"hidden_size": [32]},
            },
            "listener_0": {
                "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},
                "head_config": {"hidden_size": [32]},
            },
        }

        # Define the initial hyperparameters
        INIT_HP = {
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
        env = AsyncPettingZooVecEnv(
            [
                lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
                for _ in range(num_envs)
            ]
        )
        env.reset()

        # Configure the multi-agent algo input arguments
        observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
        action_spaces = [env.single_action_space(agent) for agent in env.agents]

        # Append number of agents and agent IDs to the initial hyperparameter dictionary
        INIT_HP["AGENT_IDS"] = env.agents

        # Create a population ready for evolutionary hyper-parameter optimisation
        pop = create_population(
            "MADDPG",
            observation_spaces,
            action_spaces,
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
            no_mutation=0.2,  # Probability of no mutation
            architecture=0.2,  # Probability of architecture mutation
            new_layer_prob=0.2,  # Probability of new layer mutation
            parameters=0.2,  # Probability of parameter mutation
            activation=0,  # Probability of activation function mutation
            rl_hp=0.2,  # Probability of RL hyperparameter mutation
            mutation_sd=0.1,  # Mutation strength
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
                agent.set_training_mode(True)

                obs, info = env.reset()  # Reset environment at start of episode
                scores = np.zeros(num_envs)
                completed_episode_scores = []
                steps = 0
                for idx_step in range(evo_steps // num_envs):
                    # Get next action from agent
                    action, raw_action = agent.get_action(
                        obs=obs,
                        infos=info
                    )

                    # Act in environment
                    next_obs, reward, termination, truncation, info = env.step(action)
                    total_steps += num_envs
                    steps += num_envs

                    agent_rewards = np.array(list(reward.values())).transpose()
                    agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
                    score_increment = (
                        (
                            np.sum(agent_rewards, axis=-1)[:, np.newaxis]
                            if is_vectorised
                            else np.sum(agent_rewards, axis=-1)
                        )
                        if sum_scores
                        else agent_rewards
                    )

                    # Save experiences to replay buffer
                    memory.save_to_memory(
                        obs,
                        raw_action,
                        reward,
                        next_obs,
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

                    obs = next_obs

                    # Find which agents are "done" - i.e. terminated or truncated
                    dones = {}
                    for agent_id in agent.agent_ids:
                        terminated = termination.get(agent_id, True)
                        truncated = truncation.get(agent_id, False)

                        # Replace NaNs with True (indicate killed agent)
                        terminated = np.where(
                            np.isnan(terminated), True, terminated
                        ).astype(bool)
                        truncated = np.where(np.isnan(truncated), False, truncated).astype(
                            bool
                        )

                        dones[agent_id] = terminated | truncated

                    # Calculate scores and reset noise for finished episodes
                    reset_noise_indices = []
                    for idx, agent_dones in enumerate(zip(*dones.values())):
                        if all(agent_dones):
                            completed_score = (
                                float(scores[idx]) if sum_scores else list(scores[idx])
                            )
                            completed_episode_scores.append(completed_score)
                            agent.scores.append(completed_score)
                            scores[idx].fill(0)
                            reset_noise_indices.append(idx)

                    agent.reset_action_noise(reset_noise_indices)

                pbar.update(evo_steps // len(pop))

                agent.steps[-1] += steps
                pop_episode_scores.append(completed_episode_scores)

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

On-Policy Training
------------------
Similarly to off-policy training, we've adapted our single-agent on-policy training loop for multi-agent settings in :file:`train_multi_agent_on_policy.py`. Currently, only
:class:`Independent Proximal Policy Optimisation (IPPO) <agilerl.algorithms.ippo.IPPO>` has been implemented to be used with this training function, but we are looking to add
more algorithms in the future!

Create a Population of Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the snippet below, we show an example of how to create a population of IPPO agents for the simple speaker listener environment.

.. collapse:: Create a population of IPPO agents

    .. code-block:: python

        from pettingzoo.mpe import simple_speaker_listener_v4
        import torch

        from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
        from agilerl.utils.utils import create_population
        from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the network configuration
        NET_CONFIG = {
            "speaker_0": {
                "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},
                "head_config": {"hidden_size": [32]},
            },
            "listener_0": {
                "encoder_config": {"hidden_size": [32, 32], "activation": "ReLU"},
                "head_config": {"hidden_size": [32]},
            },
        }

        # Define the simple speaker listener environment as a parallel environment
        num_envs = 8
        env = AsyncPettingZooVecEnv(
            [
                lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
                for _ in range(num_envs)
            ]
        )
        env.reset()

        # Configure the multi-agent algo input arguments
        observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
        action_spaces = [env.single_action_space(agent) for agent in env.agents]

        # Append number of agents and agent IDs to the initial hyperparameter dictionary
        INIT_HP["AGENT_IDS"] = env.agents

        # Mutation config for RL hyperparameters
        hp_config = HyperparameterConfig(
            lr = RLParameter(min=1e-4, max=1e-2),
            batch_size = RLParameter(min=8, max=1024, dtype=int),
            learn_step = RLParameter(
                min=256, max=8192, dtype=int, grow_factor=1.5, shrink_factor=0.75
                )
        )

        # Create a population ready for evolutionary hyper-parameter optimisation
        population_size = 4
        pop = create_population(
            "IPPO",
            observation_spaces,
            action_spaces,
            NET_CONFIG,
            INIT_HP,
            hp_config,
            population_size=population_size,
            num_envs=num_envs,
            device=device,
        )

Training Loop
~~~~~~~~~~~~~

Similarly to the off-policy alternative, the simplest way to train multi-agent on-policy algorithms is through our training function
:func:`train_multi_agent_on_policy() <agilerl.training.train_multi_agent_on_policy.train_multi_agent_on_policy>`, which returns a population of trained agents and logged training metrics.

.. collapse:: Training loop
    :open:

    .. code-block:: python

        from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy

        trained_pop, pop_fitnesses = train_multi_agent_on_policy(
            env,
            env_name='simple_speaker_listener_v4',  # Environment name
            algo="IPPO",  # Algorithm
            pop=pop,  # Population of agents
            sum_scores=True,
            INIT_HP=INIT_HP,
            MUT_P=MUTATION_PARAMS,
            max_steps=1000000,  # Max number of training steps
            evo_steps=10000,  # Evolution frequency
            eval_steps=None,  # Number of steps in evaluation episode
            eval_loop=1,  # Number of evaluation episodes
            target=-30.0,  # Target score for early stopping
            tournament=tournament,  # Tournament selection object
            mutation=mutations,  # Mutations object
            wb=False,  # Weights and Biases tracking
            accelerator=accelerator,
        )
