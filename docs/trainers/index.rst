.. _trainers:

Trainers
========

AgileRL provides a **Trainer** abstraction that encapsulates the full
evolutionary training pipeline (environment creation, population management,
mutation, tournament selection, and the training loop) behind a single,
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
run. Before training starts, the file is parsed and validated by
:class:`~agilerl.models.manifest.TrainingManifest` (through Pydantic). Invalid manifests fail
fast with field-level errors rather than failing mid-training.

**Required top-level fields:** ``algorithm`` and ``environment``.

**Optional top-level fields:** ``training``, ``network``, ``mutation``,
``selection_strategy``, and ``replay_buffer``. When omitted, each section falls back
to the defaults of its Pydantic spec (e.g. :class:`~agilerl.models.training.TrainingSpec`
uses ``max_steps=1_000_000``, ``pop_size=1``).

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Section
     - Role
   * - ``algorithm``
     - Algorithm class name (``name``) plus algorithm-specific hyperparameters (learning rate, ``gamma``, etc.).
   * - ``environment``
     - Where and how environments are created. Interpretation depends on the algorithm type (Gymnasium, PettingZoo, offline, bandit, or LLM).
   * - ``training``
     - Population training loop: step budget, evolution interval, population size, exploration schedule, checkpoints, etc.
   * - ``network``
     - Evolvable network architecture for RL algorithms, or pretrained model / LoRA settings for LLM finetuning. Becomes the algorithm's ``net_config`` when supported.
   * - ``mutation``
     - Evolutionary HPO: mutation probabilities, RL hyperparameter ranges, noise scale, random seed.
   * - ``selection_strategy``
     - Selection strategy to be used during HPO: tournament selection or multi-frequency selection. Also accepted as ``tournament_selection``.
   * - ``replay_buffer``
     - Off-policy experience storage (size, n-step, prioritized replay). Not used for pure on-policy runs.

.. note::

   Example manifests for every supported algorithm live under
   `configs/training <https://github.com/AgileRL/AgileRL/tree/main/configs/training>`_ in the repository.
   Validate a file using the :meth:`~agilerl.models.manifest.TrainingManifest.get_validated` method.

How validation works
~~~~~~~~~~~~~~~~~~~~

1. ``algorithm``
^^^^^^^^^^^^^^^^
Dispatches to the appropriate Pydantic model for algorithm argument validation based on the ``name`` field
(e.g. ``"DQN"`` → :class:`~agilerl.models.algorithms.dqn.DQNSpec`). Names are **case-sensitive** and must
match the registered class name exactly.

.. code-block:: yaml

   algorithm:
     name: DQN   # Required: must match a registered algorithm name
     # ... other fields depend on the algorithm
     lr: 6.3e-4
     batch_size: 128

.. _manifest_models_split:

Local training vs Arena manifest models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Local validation uses :class:`~agilerl.models.manifest.TrainingManifest` and
:data:`~agilerl.models.algo.ALGO_REGISTRY` to dispatch manifest fields to the appropriate Pydantic model.
Arena submission uses the
**separate** models shipped in the ``agilerl-arena`` package (:class:`~agilerl.arena.models.TrainingManifest` and
:data:`~agilerl.arena.models.ARENA_REGISTRY`, containing the algorithms eligible for Arena training). Local training
models can instantiate environments and tie into local training loops; Arena models are lightweight schemas for remote
validation and serialization.


1. ``environment``
^^^^^^^^^^^^^^^^^^

Validated differently for local and Arena training. Locally, we validate against an appropriate model depending on the chosen
algorithm since each training paradigm (e.g. single-agent, multi-agent, LLM, etc.) has different environment requirements. When
training on Arena, we check that the environment has been registered and validated before submission.

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Algorithm type
     - Field Overview
   * - Single-agent (Gymnasium)
     - ``name`` (e.g. ``LunarLander-v3``), ``num_envs``. Allow custom environments through: ``entrypoint``, ``path``, ``config``, ``wrappers``, ``sync``.
   * - Multi-agent (PettingZoo)
     - ``name`` as a module path (e.g. ``pettingzoo.mpe.simple_speaker_listener_v4``), ``num_envs``. Allow custom environments as in single-agent training.
   * - Offline
     - ``name`` for evaluation env **plus** exactly one of ``minari_dataset_id`` or ``dataset_path`` for the offline dataset.
   * - Bandit
     - ``name`` plus either ``features`` and ``targets`` (paths or in-memory tables) **or** a custom ``entrypoint``.
   * - LLM Fine-tuning
     -  ``dataset``, reward/column/template fields per type. ``env_type`` is usually inferred from the algorithm if omitted.

3. ``training``
^^^^^^^^^^^^^^^

Defines the training loop arguments and evolutionary schedule. Sensible defaults are set and provided by :class:`~agilerl.models.training.TrainingSpec`.

.. code-block:: yaml

   training:
     max_steps: 1_000_000      # Total environment steps (required, >= 1)
     evo_steps: 10_000         # Steps between mutations when HPO is enabled
     pop_size: 4               # Population size

4. ``network``
^^^^^^^^^^^^^^

**Standard RL:**

For standard RL algorithms, this section is validated against :class:`~agilerl.models.networks.NetworkSpec` and is passed to the algorithm as ``net_config``.

.. code-block:: yaml

   network:
     latent_dim: 128
     encoder_config:
       hidden_size: [128]
     head_config:
       hidden_size: [128]

.. note::

  Make sure the fields you set under ``encoder_config`` match the initialisation arguments of the
  encoder used for that observation space. See :ref:`evolvable_networks` for more detail.

**LLM Fine-tuning:**

For LLM algorithms, the ``network`` section is validated against
:class:`~agilerl.models.networks.FinetuningNetworkSpec`. ``pretrained_model_name_or_path`` can be
any model available on Hugging Face or locally.

.. code-block:: yaml

   network:
     pretrained_model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
     max_context_length: 512
     lora_config:
       lora_r: 16
       lora_alpha: 64
       target_modules: [q_proj, k_proj, v_proj, o_proj]
       lora_dropout: 0.05
       task_type: CAUSAL_LM

1. ``mutation``
^^^^^^^^^^^^^^^

Configures the mutation probabilities and the hyperparameters of the selected algorithm
to mutate during training.

.. code-block:: yaml

   mutation:
     probabilities:
       no_mut: 0.4
       arch_mut: 0.2
       new_layer: 0.2
       params_mut: 0.2
       act_mut: 0.0
       rl_hp_mut: 0.2
     rl_hp_selection:
       lr:
         min: 1.0e-4
         max: 1.0e-2
       batch_size:
         min: 8
         max: 512
     mutation_sd: 0.1
     rand_seed: 42

.. note::
   Keys under ``rl_hp_selection`` must name hyperparameters that the algorithm class
   exposes as attributes and that you want to mutate (e.g. ``lr`` for DQN, ``lr_actor`` /
   ``lr_critic`` for MADDPG). Each entry is an :class:`~agilerl.models.hpo.RLHyperparameter`
   range (``min``, ``max``, optional ``grow_factor`` / ``shrink_factor``).

6. ``selection_strategy``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   selection_strategy:
     tournament_size: 2   # Must be >= 2
     elitism: true

7. ``replay_buffer``
^^^^^^^^^^^^^^^^^^^^^

Used by off-policy single- and multi-agent algorithms. When omitted, a default replay buffer with size 100,000 is used.
N-step and prioritized replay are supported only for single-agent algorithms.

.. code-block:: yaml

   replay_buffer:
     max_size: 100_000


.. _local_trainer:

Training Locally with LocalTrainer
----------------------------------

:class:`~agilerl.training.trainer.LocalTrainer` is the simplest way to run
training on your own hardware. It instantiates the components necessary for training (algorithm, environment, etc.) from
a manifest-like specification (either a YAML file or the Pydantic models directly) and delegates to the algorithm-specific
training loops.

The simplest way to train with ``LocalTrainer`` is to specify a supported algorithm and a registered Gymnasium/PettingZoo
environment. This is mostly useful for quick experiments and benchmarking.

**Minimal example:**

.. code-block:: python

  from agilerl import LocalTrainer

  # Specify algorithm and environment with default parameters for training.
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

    selection_strategy:
      tournament_size: 2
      elitism: true

.. tab-set::

   .. tab-item:: Python
      :sync: python

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
            verbose=True # Print logs
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
   selection_spec = TournamentSelectionSpec(tournament_size=2, elitism=True)

   # Instantiate the trainer from the custom training configuration.
   device = "cuda" if torch.cuda.is_available() else "cpu"
   trainer = LocalTrainer(
       algorithm="DQN",
       environment="LunarLander-v3",
       training=training_spec,
       mutation=mutation_spec,
       selection_strategy=selection_spec,
       device=device,
   )

   # Train the population of agents.
   population, fitnesses = trainer.train(
      wb=True, # Enable Weights & Biases logging
      verbose=True # Print verbose output
   )


.. _arena_trainer:

Training on Managed Cloud Infrastructure
----------------------------------------

:class:`~agilerl.training.trainer.ArenaTrainer` submits the same
manifest-based configuration to `Arena <https://arena.agilerl.com>`_, AgileRL's
managed RLOps platform. The trainer validates the specified training configuration against the
:class:`~agilerl.arena.models.manifest.TrainingManifest`, then uses an
:class:`~agilerl.arena.client.ArenaClient` to submit the job for training on a remote cluster.

.. tip::

  `Sign up to Arena <https://arena.agilerl.com>`_ for free now and get **110 free training credits (~20 hours)** to get started!

Pre-requisites
~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

Make sure you have `agilerl-arena` installed, either directly or through the AgileRL extra:

.. code-block:: bash

  pip install agilerl-arena
  # or
  pip install "agilerl[arena]"


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

3. Use the ``ArenaClient`` class in Python:

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

   .. tab-item:: Python
      :sync: python

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

         arena experiments submit dqn.yaml \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project my-project \
             --experiment-name lunar-lander-dqn

.. seealso::

   :ref:`tutorial_arena_end_to_end` for a complete walkthrough of validating, training, and deploying
   using a custom environment.
