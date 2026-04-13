.. _trainers:

Trainers
========

AgileRL provides a **Trainer** abstraction that encapsulates the full
evolutionary training pipeline — environment creation, population management,
mutation, tournament selection, and the training loop — behind a single,
declarative interface. Instead of stitching these components together manually,
you describe *what* to train in a YAML manifest and the trainer handles the
*how*.

Two concrete trainers are available:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Trainer
     - Description
   * - :ref:`LocalTrainer <local_trainer>`
     - Runs evolutionary RL training on your local machine (CPU or GPU).
   * - :ref:`ArenaTrainer <arena_trainer>`
     - Run evolutionary RL training jobs on `Arena <https://arena.agilerl.com>`_, AgileRL's managed RLOps platform for cloud-scale distributed training.

Both share the same ``from_manifest`` factory, making it trivial to switch
between local experimentation and cloud-scale runs.

.. _training_manifests:

Training Manifests
------------------

A training manifest is a YAML (or JSON) file that fully describes a training
run. Every manifest is validated against the :class:`~agilerl.models.manifest.TrainingManifest`
Pydantic model and contains up to six top-level sections:

.. list-table::
   :widths: 20 75
   :header-rows: 1

   * - Section
     - Description
   * - ``algorithm``
     - Algorithm configuration. Users must provide a ``name`` field corresponding to the name of the algorithm class.
   * - ``environment``
     - Environment to train on. If only ``name`` is provided, a Gymnasium / PettingZoo environment is assumed. Users can provide a custom environment by providing an entrypoint, path, config, and wrappers. Users can also provide a custom environment by providing a custom environment factory function.
   * - ``training``
     - Training configuration. Users must provide a ``max_steps`` field corresponding to the total number of steps to train for, as well as ``evo_steps`` corresponding to the number of steps taken before an evolution takes place (or simply the frequency with which to report metrics for non-evolutionary settings), and ``pop_size`` corresponding to the number of individuals to train.
   * - ``mutation``
     - Mutation configuration. Users must provide a ``probabilities`` field corresponding to the probability of each mutation type, as well as ``rl_hp_selection`` corresponding to the hyperparameter ranges and scaling factors for the algorithm-specific hyperparameters.
   * - ``tournament_selection``
     - Tournament selection configuration. Users must provide a ``tournament_size`` field corresponding to the size of the tournament, as well as an ``elitism`` flag corresponding to whether to use elitism.
   * - ``replay_buffer``
     - Replay buffer configuration. Users must provide a ``max_size`` field corresponding to the maximum size of the replay buffer, and are able to choose between different buffer types depending on the chosen algorithm.
   * - ``network``
     - Network architecture specification (i.e. the arguments of the ``EvolvableNetwork`` corresponding to the chosen algorithm). This is passed as the ``net_config`` argument of most algorithms (except LLM algorithms).

Below is a minimal off-policy manifest training DQN on LunarLander:

.. collapse:: DQN Manifest LunarLander-v3

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
        lr:   { min: 0.0000625, max: 0.01 }
        batch_size: { min: 8, max: 512 }
      mutation_sd: 0.1
      rand_seed: 42

    tournament_selection:
      tournament_size: 2
      elitism: true

.. note::

    Users can find example manifests for every supported algorithm in the repository
    under ``configs/training/``.

.. _local_trainer:

LocalTrainer
------------

:class:`~agilerl.training.trainer.LocalTrainer` is the simplest way to run
training on your own hardware. It resolves the manifest into concrete objects
(vectorized environments, agent population, replay buffer, mutations, and
tournament selection) and delegates to the algorithm-specific training loops.

**From a manifest file (recommended):**

.. code-block:: python

   import torch
   from agilerl.training.trainer import LocalTrainer

   # Instantiate the trainer from a manifest file.
   trainer = LocalTrainer.from_manifest(
      manifest="configs/training/dqn/dqn.yaml",
      device="cuda" if torch.cuda.is_available() else "cpu"
  )

   population, fitnesses = trainer.train(wb=True, verbose=True)

**From Pydantic Models:**

Users can also choose to instantiate trainers explicitly from the Pydantic models used under-the-hood to validate a training
configuration automatically.

.. code-block:: python

   from agilerl.training.trainer import LocalTrainer
   from agilerl.models.training import TrainingSpec, ReplayBufferSpec
   from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec

   trainer = LocalTrainer(
       algorithm="DQN",
       environment="LunarLander-v3",
       training=TrainingSpec(
           max_steps=500_000,
           evo_steps=10_000,
           population_size=4,
           target_score=200.0,
       ),
       mutation=MutationSpec(
           probabilities={"no_mut": 0.4, "arch_mut": 0.2, "rl_hp_mut": 0.2},
       ),
       tournament=TournamentSelectionSpec(tournament_size=2, elitism=True),
       replay_buffer=ReplayBufferSpec(max_size=100_000),
       device="cuda",
   )

   # Train the population of agents.
   population, fitnesses = trainer.train()

**Minimal example:**

.. code-block:: python

  training_spec = TrainingSpec(
      max_steps=1_000_000,
      evo_steps=10_000, # Number of steps between metric reports in the absence of evo-HPO
      population_size=4
  )

  trainer = LocalTrainer(
      algorithm="PPO",
      environment="LunarLanderContinuous-v3",
      training=training_spec
  )

  trainer.train()

How ``train()`` works
~~~~~~~~~~~~~~~~~~~~~

Calling ``trainer.train()`` assembles keyword arguments from the stored specs
and passes them to the algorithm's training function (e.g.
``train_off_policy``, ``train_on_policy``, ``train_multi_agent_off_policy``).
The return value is always a tuple of ``(population, fitness_history)``.

``train()`` accepts several optional keyword arguments for logging - please refer to the :ref:`API documentation <trainers_api>` for more details.

CLI entry point
~~~~~~~~~~~~~~~

The ``agilerl/train.py`` script wraps ``LocalTrainer`` for command-line use:

.. code-block:: bash

   python -m agilerl.train -m configs/training/ppo/ppo.yaml --device cuda --wb

.. _arena_trainer:

ArenaTrainer
------------

:class:`~agilerl.training.trainer.ArenaTrainer` submits the same
manifest-based configuration to `Arena <https://arena.agilerl.com>`_, AgileRL's
managed RLOps platform. The trainer serializes its state to a
:class:`~agilerl.models.manifest.TrainingManifest`, then uses an
:class:`~agilerl.arena.client.ArenaClient` to submit the job for remote execution.

Pre-requisites
~~~~~~~~~~~~~~

A pre-requisite for using ArenaTrainer is to have an Arena account and API key. You can get your API key by logging in to Arena and clicking on the "API Keys" tab in the left sidebar.
You can then set the ``ARENA_API_KEY`` environment variable to your API key.

.. code-block:: bash

  export ARENA_API_KEY="your-arena-api-key"

Alternatively, you can pass the API key to the ``ArenaTrainer`` constructor.

.. code-block:: python

  from agilerl.training.trainer import ArenaTrainer
  from agilerl.arena.client import ArenaClient

  client = ArenaClient(api_key="your-arena-api-key")

  trainer = ArenaTrainer.from_manifest(
      manifest="configs/training/dqn/dqn.yaml",
      client=client,
  )

You must also have the extra dependencies for Arena installed, available via:

.. code-block:: bash

  pip install agilerl[arena]


**From a manifest file (recommended):**

.. code-block:: python

   from agilerl.training.trainer import ArenaTrainer
   from agilerl.models.training import TrainingSpec

   trainer = ArenaTrainer.from_manifest(
       manifest="configs/training/dqn/dqn.yaml",
   )

   # Submit to Arena (non-blocking)
   response = trainer.train()

   # Or stream logs until completion
   result = trainer.train(stream=True)

**Programmatic construction:**

.. code-block:: python

   from agilerl.training.trainer import ArenaTrainer
   from agilerl.models.env import ArenaEnvSpec
   from agilerl.models.training import TrainingSpec

   trainer = ArenaTrainer(
       algorithm="DQN",
       environment=ArenaEnvSpec(name="Your-Arena-Env", num_envs=16),
       training=TrainingSpec(max_steps=500_000, evo_steps=10_000, population_size=4),
       api_key="your-arena-api-key",
   )

   response = trainer.train()
