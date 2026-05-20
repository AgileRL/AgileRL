.. _trainers:

Trainers
========

AgileRL provides a **Trainer** abstraction that encapsulates the full
evolutionary training pipeline — environment creation, population management,
mutation, tournament selection, and the training loop — behind a single,
declarative interface. Instead of stitching these components together manually,
you describe *what* to train and the trainer handles the
*how*.

Two concrete trainers are available:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Trainer
     - Description
   * - :ref:`LocalTrainer <local_trainer>`
     - Run evolutionary RL training on your local machine (CPU or GPU).
   * - :ref:`ArenaTrainer <arena_trainer>`
     - Run evolutionary RL training jobs on `Arena <https://arena.agilerl.com>`_, AgileRL's managed RLOps platform for cloud-scale distributed training.


.. _training_manifests:

Manifest Formulation
--------------------

A training manifest is a YAML (or JSON) file that fully describes an AgileRL training
run. Every manifest is validated against the :class:`~agilerl.models.manifest.TrainingManifest`
Pydantic model, ensuring correctness and completeness of the training configuration. It contains up to six top-level sections:

.. list-table::
   :widths: 20 75
   :header-rows: 1

   * - Section
     - Description
   * - ``algorithm``
     - Algorithm configuration. Users must provide a ``name`` field corresponding to the name of the algorithm class.
   * - ``environment``
     - Environment to train on. If only ``name`` is provided, a Gymnasium / PettingZoo environment is assumed. Users can train on custom environments by providing an entrypoint, a path to the environment directory, and a config with arguments to pass to the environment constructor. Alternatively, we can simply pass a custom environment factory function.
   * - ``training``
     - Training configuration that determines the number of steps to train for, the number of steps taken before an evolution takes place (or simply the frequency with which to report metrics for non-evolutionary settings), and the number of individuals to train, as well as other training-specific hyperparameters.
   * - ``mutation``
     - Mutation configuration that determines the probability of each mutation type, as well as the hyperparameter ranges and scaling factors for the algorithm-specific hyperparameters.
   * - ``tournament_selection``
     - Tournament selection configuration that determines the size of the tournament, as well as whether to use elitism.
   * - ``replay_buffer``
     - Replay buffer configuration that determines the maximum size of the replay buffer, and the type of buffer to use. Only applicable to off-policy algorithms.
   * - ``network``
     - Network architecture specification (i.e. the arguments of the ``EvolvableNetwork`` corresponding to the chosen algorithm). This is passed as the ``net_config`` argument of most algorithms (except LLM algorithms).

.. note::

    Example manifests for every supported algorithm can be found in the `AgileRL repository <https://github.com/AgileRL/AgileRL/tree/main/configs/training>`_.

.. _local_trainer:

Training Locally with LocalTrainer
----------------------------------

:class:`~agilerl.training.trainer.LocalTrainer` is the simplest way to run
training on your own hardware. It resolves the manifest into concrete objects
(vectorized environments, agent population, replay buffer, mutations, and
tournament selection) and delegates to the algorithm-specific training loops.

Example Usage
~~~~~~~~~~~~~

The simplest way to train with the AgileRL framework is instantiate a ``LocalTrainer`` by specifying
a supported algorithm and a registered Gymnasium/PettingZoo environment. This is mostly useful for quick experiments
and benchmarking.

**Minimal example:**

.. code-block:: python

  from agilerl import LocalTrainer

  # Specify algorithm and Gymnasium/PettingZoo environment and use
  # default parameters for training.
  trainer = LocalTrainer(algorithm="PPO", environment="LunarLanderContinuous-v3")
  population, fitnesses = trainer.train()

**From a manifest file:**

Specifying their training configuration from a manifest file also allows users to use the ``agilerl/train.py`` CLI entry point,
which wraps the ``LocalTrainer`` and provides a convenient way to train a population of agents.

Below is a minimal off-policy manifest to train DQN on LunarLander-v3.

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
      arch: mlp
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

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         import torch
         from agilerl import LocalTrainer

         # Instantiate the trainer from a manifest file.
         device = "cuda" if torch.cuda.is_available() else "cpu"
         trainer = LocalTrainer.from_manifest(
            manifest="dqn.yaml",
            device=device
         )

         # Train the population of agents.
         population, fitnesses = trainer.train(
            wb=True, # Enable Weights & Biases logging
            verbose=True # Print verbose output
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         python -m agilerl.train dqn.yaml --wb --verbose

.. seealso::

   Example manifests for every supported algorithm can be found in the `AgileRL repository <https://github.com/AgileRL/AgileRL/tree/main/configs/training>`_.

**From Pydantic Models:**

Trainers can also be instantiated explicitly from the Pydantic models used under-the-hood to validate a training
configuration automatically. In the example below we show a more advanced configuration for training DQN on LunarLander-v3
applying evolutionary HPO with custom mutation probabilities.

.. code-block:: python

   import torch
   from agilerl import LocalTrainer
   from agilerl.models import (
    TrainingSpec,
    MutationSpec,
    TournamentSelectionSpec,
    MutationProbabilities,
   )

   # Custom training configuration.
   training_spec = TrainingSpec(
       max_steps=5_000_000,
       evo_steps=10_000,
       population_size=6, # Train six agents simultaneously
       target_score=200.0, # Target score to achieve for LunarLander-v3
   )

   # Customise Evo-HPO configuration.
   mutation_spec = MutationSpec(
       probabilities=MutationProbabilities(
        no_mut=0.5,
        arch_mut=0.3,
        rl_hp_mut=0.2
       ),
   )
   tournament_spec = TournamentSelectionSpec(tournament_size=2, elitism=True)

   # Instantiate the trainer from the custom training configuration.
   device = "cuda" if torch.cuda.is_available() else "cpu"
   trainer = LocalTrainer(
       algorithm="DQN",
       environment="LunarLander-v3",
       training=training_spec,
       mutation=mutation_spec,
       tournament=tournament_spec,
       device=device,
   )

   # Train the population of agents.
   population, fitnesses = trainer.train(
      wb=True, # Enable Weights & Biases logging
      verbose=True # Print verbose output
   )


How ``LocalTrainer.train()`` works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling ``LocalTrainer.train()`` assembles keyword arguments from the stored specs
and dispatches to the algorithm-specific training function (e.g.
``train_off_policy``, ``train_on_policy``, ``train_multi_agent_off_policy``).
The return value is always a tuple of ``(population, fitness_history)``.

.. seealso::

   :ref:`API documentation <trainers_api>` for the full method signature, including options to
   customise monitoring and checkpointing.


.. _arena_trainer:

Training on Managed Cloud Infrastructure with ArenaTrainer
----------------------------------------------------------

:class:`~agilerl.training.trainer.ArenaTrainer` submits the same
manifest-based configuration to `Arena <https://arena.agilerl.com>`_, AgileRL's
managed RLOps platform. The trainer validates the specified training configuration against the
:class:`~agilerl.models.manifest.TrainingManifest`, then uses an
:class:`~agilerl.arena.client.ArenaClient` to submit the job for training on a remote cluster.

.. note::

  `Sign up to Arena <https://arena.agilerl.com>`_ for free now and get **110 free training credits (~20 hours)** to get started!

Pre-requisites
~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

Make sure you have the extra dependencies for Arena installed, available via:

.. code-block:: bash

  pip install agilerl[arena]


Authentication
^^^^^^^^^^^^^^
In order to authenticate with Arena, users must have a registered account. Then, there are a few ways to authenticate before submitting a training job:

1. Set the ``ARENA_API_KEY`` environment variable:

   .. code-block:: bash

      export ARENA_API_KEY="arena_pat..."

   .. note::
      Personal access tokens can be found in the Arena account profile, under *Profile management* -> *CLI API Key*.


2. Use ``arena login`` CLI command:

   .. code-block:: bash

      arena login

3. Use ``ArenaClient`` SDK class:

   .. code-block:: python

      from agilerl.arena import ArenaClient

      # Option 1: Set the API key explicitly
      client = ArenaClient(api_key="arena_pat...")

      # Option 2: OAuth device login
      client = ArenaClient()
      client.login() # Opens a browser for interactive authentication

.. seealso::

   :ref:`ArenaClient <arena_client>` for more information on the full functionality of the client library.

Submitting a Training Job to Arena
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the Arena client is authenticated, users can submit a training job to Arena by instantiating an ``ArenaTrainer`` and calling ``train()``,
or through the **Arena CLI** ``arena experiments submit`` command.

Here is an example using the same ``dqn.yaml`` manifest file as in the :ref:`local_trainer` section.

.. tab-set::

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         from agilerl import ArenaTrainer

         # Instantiate the trainer from a manifest file.
         trainer = ArenaTrainer.from_manifest(manifest="dqn.yaml")

         # Train on Arena.
         trainer.train(
          resource_id="arena-medium",
          num_nodes=2,
          project="my-project",
          experiment_name="lunar-lander-dqn",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit -m dqn.yaml

.. seealso::

   :ref:`tutorial_arena_end_to_end` for a complete walkthrough of validating, training, and deploying
   using a custom environment.
