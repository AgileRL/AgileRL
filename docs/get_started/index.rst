Getting Started
---------------

.. raw:: html

   <h3 id="install-agilerl">Install AgileRL</h3>

To use AgileRL, first download the source code and install requirements.

Install as a package with pip:

.. code-block:: bash

   pip install agilerl

Or install in development mode:

.. code-block:: bash

   git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
   pip install -e .

To install the ``nightly`` version of AgileRL with the latest features, use:

.. code-block:: bash

   pip install git+https://github.com/AgileRL/AgileRL.git@nightly

.. raw:: html

   <h3 id="algorithms">Algorithms</h3>

.. raw:: html

   <style>
    /* CSS styles for tiles with rounded corners, centered titles, and always displayed algorithm list */

    /* Style for the container */

   @media (max-width: 750px) {
      .tiles_2 {
         display: grid;
         grid-template-columns: 100%; /* 2 columns */
         grid-auto-rows: 0% 50% 50% 0%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 0px; /*48px;*/
         margin-bottom: 58px;
         width: 100%;
         align-content: center;
         height: auto;
         min-height: 185px;
      }

      .tiles_3 {
         display: grid;
         grid-template-columns: 100%; /* 3 columns */
         grid-auto-rows: 33%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 72px;
         width: 100%;
         align-content: start;
         height: auto;
         min-height: 185px;
      }
   }

   @media (min-width: 750px) {
      .tiles_2 {
         display: grid;
         grid-template-columns: 16.5% 33% 33% 16.5%; /* 2 columns */
         grid-auto-rows: 100%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 0px; /*48px;*/
         margin-bottom: 58px;
         width: 100%;
         align-content: center;
         height: auto;
         min-height: 185px;
      }
      .tiles_3 {
         display: grid;
         grid-template-columns: 33% 33% 33%; /* 3 columns */
         grid-auto-rows: 100%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 25px;/*58px;*/
         width: 100%;
         align-content: start;
         height: auto;
         min-height: 185px;
      }
   }

    /* Style for each tile */
    .tile {
        padding: 0px 20px 20px; ; /* Fixed padding */
        transition: background-color 0.3s ease; /* Smooth transition */
        text-decoration: none;
        width: auto; /* Fixed width */
        height: auto; /* Fixed height */
        overflow: hidden; /* Hide overflow content */
        display: flex; /* Use flexbox for content alignment */
        flex-direction: column; /* Align content vertically */
        /*justify-content: center; /* Center content vertically */
        /*align-items: flex-start;*/
        background-color: transparent; /* Dark grey background */
        border-radius: 7px; /* Rounded corners */
        position: relative; /* Relative positioning for algorithm list */
        box-shadow: 0 4px 8px rgba(0, 150, 150, 0.5);
    }

    .column {
    flex: 1; /* Equal flex distribution */
    width: 50%; /* 50% width for each column */
    display: flex;
    flex-direction: column;
    /* Additional styles */
   }

    /* Lighter background color on hover */
    .tile:hover {
        background-color: #48b8b8; /* Lighter grey on hover */
        color: white;
    }

    /* Title styles */
    .tile h2 {
        margin-bottom: 8px; /* Adjust the margin */
        font-size: 24px; /* Adjust the font size */
        text-align: center; /* Center title text */
    }

   .tile p {
         margin-top: 12px;
         margin-bottom: 8px; /* Adjust the margin */
         font-size: 16px; /* Adjust the font size */
         text-align: left;
         word-wrap: break-word;
      }


    /* Learn more link styles */
    .tile a {
        display: block;
        margin-top: 8px; /* Adjust the margin */
        text-decoration: none;
        /*color: white; /* Link color */
        font-size: 14px; /* Adjust the font size */
        text-align: center; /* Center link text */
    }

    .tile a:hover {
        color: white; /* Link color on hover */
    }
   </style>

   <div class="tiles_3 article">
      <a href="../on_policy/index.html" class="tile on-policy article">
         <h2>On-policy</h2>
         <p>
               Algorithms: PPO
         </p>
      </a>
      <a href="../off_policy/index.html" class="tile off-policy">
         <h2> Off-policy</h2>
            <p>
                  Algorithms: DQN, Rainbow DQN, TD3, DDPG
                  <!-- Add more algorithms as needed -->
            </p>
      </a>
      <a href="../offline_training/index.html" class="tile online">
         <h2>Offline</h2>
         <p>
               Algorithms: CQL, ILQL
               <!-- Add more algorithms as needed -->
         </p>
      </a>
   </div>
   <div class="tiles_2 article">
      <div></div>
      <a href="../multi_agent_training/index.html" class="tile multi-agent">
         <h2>Multi Agent</h2>
         <p>
               Algorithms: MADDPG, MATD3, IPPO
               <!-- Add more algorithms as needed -->
         </p>
      </a>
      <a href="../bandits/index.html" class="tile bandit">
         <h2>Contextual Bandits</h2>
         <p>
               Algorithms: NeuralUCB, NeuralTS
               <!-- Add more algorithms as needed -->
         </p>
      </a>
   </div>

.. raw:: html

   <h3 id="tutorials">Tutorials</h3>

We are constantly updating our tutorials to showcase the latest features of AgileRL and how users can leverage our evolutionary HPO to achieve 10x
faster hyperparameter optimization. Please see the available tutorials below.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Tutorial Type
     - Description
     - Tutorials
   * - `Single-agent tasks <../tutorials/gymnasium/index.html>`_
     - Guides for training both on and off-policy agents to beat a variety of Gymnasium environments.
     - `PPO - Acrobot <../tutorials/gymnasium/agilerl_ppo_tutorial.html>`_ |br|
       `TD3 - Lunar Lander <../tutorials/gymnasium/agilerl_td3_tutorial.html>`_ |br|
       `Rainbow DQN - CartPole <../tutorials/gymnasium/agilerl_rainbow_dqn_tutorial.html>`_
       `Recurrent PPO - Masked Pendulum <../tutorials/gymnasium/agilerl_recurrent_ppo_tutorial.html>`_
   * - `Multi-agent tasks <../tutorials/pettingzoo/index.html>`_
     - Use of PettingZoo environments such as training DQN to play Connect Four with curriculum learning and self-play, and for multi-agent tasks in MPE environments.
     - `DQN - Connect Four <../tutorials/pettingzoo/dqn.html>`_ |br|
       `MADDPG - Space Invaders <../tutorials/pettingzoo/maddpg.html>`_ |br|
       `MATD3 - Speaker Listener <../tutorials/pettingzoo/matd3.html>`_
   * - `Hierarchical curriculum learning <../tutorials/skills/index.html>`_
     - Shows how to teach agents Skills and combine them to achieve an end goal.
     - `PPO - Lunar Lander <../tutorials/skills/index.html>`_
   * - `Contextual multi-arm bandits <../tutorials/bandits/index.html>`_
     - Learn to make the correct decision in environments that only have one timestep.
     - `NeuralUCB - Iris Dataset <../tutorials/bandits/agilerl_neural_ucb_tutorial.html>`_ |br|
       `NeuralTS - PenDigits <../tutorials/bandits/agilerl_neural_ts_tutorial.html>`_
   * - `Custom Modules & Networks <../tutorials/custom_networks/index.html>`_
     - Learn how to create custom evolvable modules and networks for RL algorithms.
     - `Dueling Distributional Q Network <../tutorials/custom_networks/agilerl_rainbow_tutorial.html>`_ |br|
       `EvolvableSimBa <../tutorials/custom_networks/agilerl_simba_tutorial.html>`_
   * - `LLM Finetuning <../tutorials/llm_finetuning/index.html>`_
     - Learn how to finetune an LLM using AgileRL.
     - `GRPO <../tutorials/llm_finetuning/index.html>`_

.. |br| raw:: html

   <br>

.. raw:: html

   <h3 id="train-an-agent">Train an Agent</h3>

Train an agent to beat a Gym environment.

Before starting training, there are some meta-hyperparameters and settings that must be set. These are defined in ``INIT_HP``, for general
parameters, and ``MUTATION_PARAMS``, which define the evolutionary probabilities, and ``NET_CONFIG``, which defines the network architecture. For example:

.. collapse:: Algorithm Hyperparameters

   .. code-block:: python

      INIT_HP = {
          'ENV_NAME': 'LunarLander-v3',   # Gym environment name
          'ALGO': 'DQN',                  # Algorithm
          'DOUBLE': True,                 # Use double Q-learning
          'CHANNELS_LAST': False,         # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
          'BATCH_SIZE': 256,              # Batch size
          'LR': 1e-3,                     # Learning rate
          'MAX_STEPS': 1_000_000,         # Max no. steps
          'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
          'GAMMA': 0.99,                  # Discount factor
          'MEMORY_SIZE': 10000,           # Max memory buffer size
          'LEARN_STEP': 1,                # Learning frequency
          'TAU': 1e-3,                    # For soft update of target parameters
          'TOURN_SIZE': 2,                # Tournament size
          'ELITISM': True,                # Elitism in tournament selection
          'POP_SIZE': 6,                  # Population size
          'EVO_STEPS': 10_000,            # Evolution frequency
          'EVAL_STEPS': None,             # Evaluation steps
          'EVAL_LOOP': 1,                 # Evaluation episodes
          'LEARNING_DELAY': 1000,         # Steps before starting learning
          'WANDB': True,                  # Log with Weights and Biases
      }

.. collapse:: Mutation Hyperparameters

   .. code-block:: python

      MUTATION_PARAMS = {
          # Relative probabilities
          'NO_MUT': 0.4,                              # No mutation
          'ARCH_MUT': 0.2,                            # Architecture mutation
          'NEW_LAYER': 0.2,                           # New layer mutation
          'PARAMS_MUT': 0.2,                          # Network parameters mutation
          'ACT_MUT': 0,                               # Activation layer mutation
          'RL_HP_MUT': 0.2,                           # Learning HP mutation
          'MUT_SD': 0.1,                              # Mutation strength
          'RAND_SEED': 1,                             # Random seed
      }

.. collapse:: Network Configuration

   .. code-block:: python

      NET_CONFIG = {
          'latent_dim': 16
          'encoder_config': {
            'hidden_size': [32]     # Observation encoder configuration
          }
          'head_config': {
            'hidden_size': [32]     # Network head configuration
          }

      }

.. raw:: html

   <br>
   <h3>Creating a Population of Agents</h3>

First, use ``utils.utils.create_population`` to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.

.. collapse:: Population Creation Example
   :open:

   .. code-block:: python

      import torch
      from agilerl.utils.utils import (
          make_vect_envs,
          create_population,
          observation_space_channels_to_first
      )

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      num_envs = 16
      env = make_vect_envs(env_name=INIT_HP['ENV_NAME'], num_envs=num_envs)

      observation_space = env.single_observation_space
      action_space = env.single_action_space
      if INIT_HP['CHANNELS_LAST']:
          observation_space = observation_space_channels_to_first(observation_space)

      agent_pop = create_population(
          algo=INIT_HP['ALGO'],                 # Algorithm
          observation_space=observation_space,  # Observation space
          action_space=action_space,            # Action space
          net_config=NET_CONFIG,                # Network configuration
          INIT_HP=INIT_HP,                      # Initial hyperparameters
          population_size=INIT_HP['POP_SIZE'],  # Population size
          num_envs=num_envs,                    # Number of vectorized environments
          device=device
      )

.. raw:: html

   <h3>Initializing Evolutionary HPO</h3>

Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.

.. collapse:: Mutations and Tournament Selection Example
   :open:

   .. code-block:: python

      from agilerl.components.replay_buffer import ReplayBuffer
      from agilerl.hpo.tournament import TournamentSelection
      from agilerl.hpo.mutation import Mutations

      memory = ReplayBuffer(
          max_size=INIT_HP['MEMORY_SIZE'],   # Max replay buffer size
          device=device,
      )

      tournament = TournamentSelection(
          tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
          elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
          population_size=INIT_HP['POP_SIZE'],   # Population size
          eval_loop=INIT_HP['EVAL_LOOP'],        # Evaluate using last N fitness scores
      )

      mutations = Mutations(
          no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
          architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
          new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
          parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
          activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
          rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
          mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
          rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
          device=device,
      )

.. raw:: html

   <h3>Train a Population of Agents</h3>

The easiest training loop implementation is to use our :func:`train_off_policy() <agilerl.training.train_off_policy.train_off_policy>` function.
It requires the ``agent`` have methods ``get_action()`` and ``learn()``.

.. collapse:: Training Example
   :open:

   .. code-block:: python

      from agilerl.training.train_off_policy import train_off_policy

      trained_pop, pop_fitnesses = train_off_policy(
          env=env,                                   # Gym-style environment
          env_name=INIT_HP['ENV_NAME'],              # Environment name
          algo=INIT_HP['ALGO'],                      # Algorithm
          pop=agent_pop,                             # Population of agents
          memory=memory,                             # Replay buffer
          swap_channels=INIT_HP['CHANNELS_LAST'],    # Swap image channel from last to first
          max_steps=INIT_HP["MAX_STEPS"],            # Max number of training steps
          evo_steps=INIT_HP['EVO_STEPS'],            # Evolution frequency
          eval_steps=INIT_HP["EVAL_STEPS"],          # Number of steps in evaluation episode
          eval_loop=INIT_HP["EVAL_LOOP"],            # Number of evaluation episodes
          learning_delay=INIT_HP['LEARNING_DELAY'],  # Steps before starting learning
          target=INIT_HP['TARGET_SCORE'],            # Target score for early stopping
          tournament=tournament,                     # Tournament selection object
          mutation=mutations,                        # Mutations object
          wb=INIT_HP['WANDB'],                       # Weights and Biases tracking
      )
