Getting Started
---------------

.. raw:: html

   <h3 id="install-agilerl">Installation</h3>

Install as a package with pip:

.. code-block:: bash

   pip install agilerl

Or install in development mode:

.. code-block:: bash

   git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
   pip install -e .

AgileRL ships optional dependency groups that you can install as needed:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Installation
     - Description
   * - ``agilerl[box2d]``
     - Box2D physics engine for Gymnasium environments.
   * - ``agilerl[arena]``
     - Installs ``agilerl-arena`` (Python SDK and CLI).
       Use for `Arena <https://arena.agilerl.com>`_ environment validation,
       cloud training, and deployment.
   * - ``pip install agilerl-arena``
     - Arena SDK & CLI for remote training only.
   * - ``agilerl[llm]``
     - LLM reinforcement fine-tuning.
   * - ``agilerl[all]``
     - Cover all functionalities of AgileRL.

In development mode, quote the extras - for example:

.. code-block:: bash

   pip install -e ".[all]"

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
         grid-template-columns: 100%; /* 1 column */
         grid-auto-rows: auto; /* auto rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 0px;
         margin-bottom: 58px;
         width: 100%;
         align-content: center;
         height: auto;
         min-height: 185px;
      }

      .tiles_3 {
         display: grid;
         grid-template-columns: 100%; /* 1 column */
         grid-auto-rows: auto; /* auto rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 25px;
         width: 100%;
         align-content: start;
         height: auto;
         min-height: 185px;
      }
   }

   @media (min-width: 750px) {
      .tiles_2 {
         display: grid;
         grid-template-columns: 33% 33% 33%; /* 3 columns */
         grid-auto-rows: 100%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 25px;
         width: 100%;
         align-content: start;
         height: auto;
         min-height: 185px;
      }
      .tiles_3 {
         display: grid;
         grid-template-columns: 33% 33% 33%; /* 3 columns */
         grid-auto-rows: 100%; /* 2 rows */
         gap: 25px; /* Adjust the gap between tiles */
         margin-top: 48px;
         margin-bottom: 58px;
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
        color: #468082;
    }

   .tile p {
         margin-top: 12px;
         margin-bottom: 8px; /* Adjust the margin */
         font-size: 16px; /* Adjust the font size */
         text-align: left;
         word-wrap: break-word;
         color: #468082;
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
      <a href="../multi_agent_training/index.html" class="tile multi-agent">
         <h2>Multi-agent</h2>
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
      <a href="../llm_finetuning/index.html" class="tile llm-finetuning">
         <h2>LLM Finetuning</h2>
         <p>
               Algorithms: DPO, GRPO, GSPO, CISPO, LLMPPO, LLMREINFORCE
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
   * - `Training on Arena <../tutorials/arena_training/index.html>`_
     - Upload and validate custom environments, submit training jobs on managed cloud infrastructure, and deploy trained agents for inference.
     - `PPO - Merge Custom Environment <../tutorials/arena_training/ppo_custom_env.html>`_
   * - `LLM Finetuning <../tutorials/llm_finetuning/index.html>`_
     - Learn how to finetune an LLM using AgileRL.
     - `GRPO <../tutorials/llm_finetuning/grpo_finetuning.html>`_ |br|
       `GRPO with Evo HPO <../tutorials/llm_finetuning/grpo_hpo.html>`_ |br|
       `Multi-turn GRPO/PPO <../tutorials/llm_finetuning/env_grpo_ppo.html>`_ |br|
       `SFT + DPO <../tutorials/llm_finetuning/sft_dpo_finetuning.html>`_

.. |br| raw:: html

   <br>

.. raw:: html

   <h3 id="quick-start">Quick Start</h3>

**Training a Single Agent without Evolutionary HPO:**

The simplest way to train an RL agent with AgileRL is through the
:class:`~agilerl.training.trainer.LocalTrainer`. Here is an example of training a DQN agent on the LunarLander-v3 environment:

.. code-block:: python

   from agilerl.training.trainer import LocalTrainer

   trainer = LocalTrainer(algorithm="DQN", environment="LunarLander-v3")
   population, fitnesses = trainer.train()

.. note::

   With no other arguments provided, ``LocalTrainer`` defaults to 1,000,000 steps with a
   single agent and the algorithm's default hyperparameters. No evolutionary
   HPO is applied.

**Training a Population with Evolutionary HPO:**

To unlock AgileRL's evolutionary hyperparameter optimization, train a population
of agents whose hyperparameters will evolve and mutate towards their optimal
values:

.. code-block:: python

   from agilerl import LocalTrainer
   from agilerl.models import TrainingSpec

   trainer = LocalTrainer(
       algorithm="DQN",
       environment="LunarLander-v3",
       training=TrainingSpec(pop_size=4), # Train four agents synchronously
       hpo=True, # Enable evolutionary HPO using default settings
   )
   population, fitnesses = trainer.train()

This trains a population of four DQN agents that share experiences but learn individually. Every 10,000 steps
(default value for ``evo_steps`` in ``TrainingSpec``), tournament selection identifies the best
performers and mutations are applied to explore the hyperparameter space.

.. seealso::

   :ref:`evo_hyperparam_opt` for details on how evolutionary HPO works.

Or via a YAML manifest:

.. collapse:: dqn.yaml

  .. code-block:: yaml

    algorithm:
        name: DQN
        batch_size: 128
        lr: 6.3e-4
        learn_step: 4
        gamma: 0.99
        tau: 0.001
        double: false
        cudagraphs: false

    environment:
        name: LunarLander-v3
        num_envs: 16

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
            learn_step:
                min: 1
                max: 10
        mutation_sd: 0.1
        rand_seed: 42

    network:
        latent_dim: 128
        encoder_config:
            hidden_size:
                - 128
        head_config:
            hidden_size:
                - 128

    replay_buffer:
        max_size: 100_000

    selection_strategy:
        tournament_size: 2
        elitism: true

    training:
        max_steps: 1_000_000
        target_score: 200.0
        pop_size: 4
        evo_steps: 10_000
        eval_steps:
        eval_loop: 1
        learning_delay: 0
        eps_start: 1.0
        eps_end: 0.1
        eps_decay: 0.99

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from agilerl import LocalTrainer

         trainer = LocalTrainer.from_manifest("dqn.yaml")
         population, fitnesses = trainer.train()

   .. tab-item:: CLI

      .. code-block:: bash

         python -m agilerl.train dqn.yaml


Every aspect of the training pipeline is customisable: from modifying
hyperparameters and mutation strategies in our off-the-shelf tools, to
implementing your own :ref:`evolvable algorithms <custom_algorithms>`,
:ref:`network architectures <evolvable_networks>`, and
:ref:`training loops <off_policy>`.

.. seealso::

   - :ref:`trainers` for full manifest reference, Pydantic model construction,
     and all ``LocalTrainer`` options.
   - :ref:`off_policy` for a detailed walkthrough of customising the training
     pipeline (replay buffers, exploration, evaluation, and evolution).
