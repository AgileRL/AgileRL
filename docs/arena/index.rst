.. _arena_client:

Arena Client
============

Arena is an **RLOps platform** that streamlines the reinforcement learning
development process and accelerates iteration through distributed training and
evolutionary hyperparameter optimization. It provides managed cloud
infrastructure purpose-built for RL workloads, so researchers and engineers can
focus on algorithms and environments rather than cluster management.

The Arena client — available as both a **CLI** (``arena``) and a **Python SDK**
(:class:`~agilerl.arena.client.ArenaClient`) — allows registered users to do
everything from their own development environment: validate custom environments,
submit training experiments, monitor progress, and deploy trained agents for
inference.

.. tip::

  `Sign up to Arena <https://arena.agilerl.com>`_ for free now and get **110 free training credits (~20 hours)** to get started!

Installation
------------

To use the Arena client, you need to install the extra dependencies for Arena, available via:

.. code-block:: bash

   pip install agilerl[arena]

Authentication
--------------

Authentication is resolved automatically in priority order:

1. **api_key argument** passed directly to the client or CLI.
2. **ARENA_API_KEY environment variable** — if set, no login is needed.
3. **Stored OAuth credentials** from ``~/.arena/credentials.json`` (persisted
   after a successful ``arena login``).
4. **Interactive device login** — opens a browser for OAuth authorization.

The simplest approach for scripting and CI is to set the environment variable:

.. code-block:: bash

   export ARENA_API_KEY="arena_pat_..."

.. note::
   Personal access tokens can be found in the Arena account profile, under *Profile management* -> *CLI API Key*.

Once set, all CLI commands and SDK calls authenticate automatically without
requiring ``arena login``.

For interactive use, ``arena login`` opens a browser-based OAuth flow and
persists credentials locally so you only need to log in once per machine:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # One-time interactive login (persists to ~/.arena/credentials.json)
         arena login

         # Or skip login entirely with an env var or flag
         export ARENA_API_KEY="arena_pat_..."

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena.client import ArenaClient

         # Option 1: Set ARENA_API_KEY env var — no login needed
         client = ArenaClient()

         # Option 2: Pass the key explicitly
         client = ArenaClient(api_key="arena_pat_...")

         # Option 3: Interactive device login (one-time)
         client = ArenaClient()
         client.login()  # opens browser, persists credentials


.. _arena_environments:

Custom Environments
-------------------

Before launching a training run a custom environment or dataset, Arena requires these to be **validated**
— ensuring that the environment is importable, has the correct interface (e.g. Gymnasium or
PettingZoo), and can be stepped without errors. After validation,
environments are automatically **profiled** to determine their compute and
memory footprint. This information is then used to estimate a ceiling on the resources
available for training, depending on the selected cluster tier.

Users can register two types of environments on Arena:

- **Custom Gymnasium/PettingZoo** — upload a source directory or
  ``.tar.gz`` archive containing your environment code.
- **LLM Datasets** — upload a dataset file or reference a HuggingFace
  dataset ID to create a language-model fine-tuning environment.

Uploading and Validating
~~~~~~~~~~~~~~~~~~~~~~~~

In order to create and validate a custom Gymnasium/PettingZoo environment from scratch, users need to provide a source
file or directory containing the environment code. Multi-agent environments must be signaled through the ``multi_agent`` flag.
Additionally, an entrypoint can be provided to the environment class to use, which is useful when multiple entrypoints exist in the same path.
Please refer to the :meth:`ArenaClient.validate_environment <agilerl.arena.client.ArenaClient.validate_environment>` method documentation for more details.

Once your environment / dataset has been validated successfully, you will be able to view it in the **Environments / Datasets** section in Arena.

RL Environments
^^^^^^^^^^^^^^^

The following commands assume you have a valid environment in the ``my_env/`` directory.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Upload and validate in one step
         arena env validate --source my_env

         # With an explicit entrypoint (when multiple exist in the same path)
         arena env validate --source my_env --entrypoint my_module:MyEnvClass

         # Validate an already-registered environment
         arena env validate my-env --version v1

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena import ArenaClient

         client = ArenaClient()

         # Upload, create, and validate in one step
         result = client.validate_environment(source="./my_env/")

         # With an explicit entrypoint (assuming my_module.py exists in source and contains the MyEnvClass class)
         result = client.validate_environment(
             source="./my_env/",
             entrypoint="my_module:MyEnvClass",
         )

         # Validate an already-registered environment
         result = client.validate_environment(name="my-env", version="v1")

LLM Datasets
^^^^^^^^^^^^^

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Upload and validate in one step
         arena dataset validate --source ./my_dataset/ --version v1

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena import ArenaClient

         client = ArenaClient()
         result = client.validate_dataset(source="./my_dataset/")

Additional Tools
^^^^^^^^^^^^^^^^

Here is a list of additional methods provided by the :class:`~agilerl.arena.client.ArenaClient` to help you navigate the custom environments workflow in Arena.
You can find the analogous CLI commands by running ``arena env --help``.

- :meth:`~agilerl.arena.client.ArenaClient.list_environments` — List all registered environments and their versions.
- :meth:`~agilerl.arena.client.ArenaClient.list_environment_entrypoints` — List available entrypoints for a specific environment version.
- :meth:`~agilerl.arena.client.ArenaClient.environment_exists` — Check whether an environment (and optionally a version) is registered.
- :meth:`~agilerl.arena.client.ArenaClient.profile_environment` — Profile a validated environment to determine its resource requirements.
- :meth:`~agilerl.arena.client.ArenaClient.delete_environment` — Delete one or all versions of a registered environment.

.. _arena_projects:

Project Management
------------------

Projects are the top-level organisational unit in Arena. Every experiment belongs to a
project, and you must specify a project when submitting training jobs.

Some useful command to manage your projects:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena import ArenaClient

         client = ArenaClient()

         # List all projects
         projects = client.list_projects()

         # Create a new project
         client.create_project(name="CartPole-HPO", description="HPO on CartPole")

         # Delete a project
         client.delete_project("CartPole-HPO")

         # Show the current default
         client.get_default_project()

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # List all projects
         arena projects list

         # Create a new project
         arena projects create CartPole-HPO --description "HPO on CartPole"

         # Delete a project
         arena projects delete CartPole-HPO --yes

         # Show the current default
         arena projects get-default

.. tip::
   You can set a default project to work on by doing the following. This will be stored in the ``~/.arena/config.json`` file.

   .. code-block:: bash

      arena projects set-default <project-name>

.. _arena_training:

Submitting Experiments
----------------------

Once your environment or dataset has been validated, you can submit training
jobs for it to Arena. AgileRL has its own managed cloud infrastructure with
automatic checkpointing, metric logging, and real-time monitoring accessible
directly from the Arena dashboard. Training runs in a distributed manner across multiple nodes
(this can be customized by the user) to achieve lightning-fast training for arbitrarily large
populations of agents.

Available Clusters on Arena
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We currently provide a wide variety of compute resources on Arena, including GPU and CPU clusters.
These might change over time, so it is recommended to use the :meth:`ArenaClient.list_resources <agilerl.arena.client.ArenaClient.list_resources>`
method to get the latest list of available resources before setting off an experiment.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena import ArenaClient

         client = ArenaClient()
         resources = client.list_resources()

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena resources list

Example output from CLI command:

.. list-table::
   :header-rows: 1
   :widths: 25 22 6 10 18

   * - Resource ID
     - GPUs
     - CPUs
     - RAM (GB)
     - Credits/node-hour
   * - arena-medium
     - 1x nvidia-l4
     - 15
     - 55.0
     - 2.41
   * - arena-large
     - 1x nvidia-l4
     - 30
     - 110.0
     - 3.65
   * - arena-large-4gpu
     - 4x nvidia-l4
     - 47
     - 165.0
     - 8.41
   * - arena-large-compute-96
     - —
     - 95
     - 180.0
     - 8.57
   * - arena-a100-40gb-2gpu
     - 2x nvidia-tesla-a100
     - 23
     - 156.0
     - 15.74
   * - arena-extra-large
     - 8x nvidia-l4
     - 94
     - 370.0
     - 16.82
   * - arena-large-compute-192
     - —
     - 191
     - 370.0
     - 17.15
   * - arena-a100-40gb-4gpu
     - 4x nvidia-tesla-a100
     - 47
     - 312.0
     - 31.47

Job Submission
~~~~~~~~~~~~~~
A training configuration is defined via a **manifest** (YAML or JSON file) describing
the algorithm, environment, training parameters, and evolutionary HPO settings. The formulation of the manifest is described in the :ref:`training_manifests` section,
and is mostly analogous for both the :class:`~agilerl.training.trainer.LocalTrainer` and training jobs in Arena.

Here is an example manifest for training DQN on LunarLander-v3:

.. collapse:: dqn.yaml

  .. code-block:: yaml

    algorithm:
      name: DQN

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
        activation: ReLU
      head_config:
        hidden_size: [128]
        activation: ReLU

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
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Submit an experiment
         arena experiments submit \
             --manifest dqn.yaml \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project my-project \
             --experiment-name lunar-lander-dqn

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena import ArenaClient

         client = ArenaClient()
         result = client.submit_experiment(
             manifest="dqn.yaml",
             resource_id="arena-medium",
             num_nodes=2,
             project="my-project",
             experiment_name="lunar-lander-dqn",
         )

You can also submit experiments via the :class:`~agilerl.training.trainer.ArenaTrainer`
class, which provides a higher-level interface:

.. code-block:: python

   from agilerl.training.trainer import ArenaTrainer

   # Don't need to provide a client if we have already authenticated through `arena login`
   trainer = ArenaTrainer.from_manifest(
       manifest="dqn.yaml",
   )
   result = trainer.train()

.. seealso::

   :ref:`training_manifests` section for an overview of the training manifest and its options.

   :ref:`trainers` section for more information on the ``ArenaTrainer`` class and its usage.

Additional Tools
^^^^^^^^^^^^^^^^

Here is a list of additional methods provided by the :class:`~agilerl.arena.client.ArenaClient` for managing experiments.
You can find the analogous CLI commands by running ``arena experiments --help``.

- :meth:`~agilerl.arena.client.ArenaClient.list_experiments` — List all experiments in a project.
- :meth:`~agilerl.arena.client.ArenaClient.list_checkpoints` — List saved checkpoints for an experiment.
- :meth:`~agilerl.arena.client.ArenaClient.resume_experiment` — Resume a stopped experiment with a new step budget.
- :meth:`~agilerl.arena.client.ArenaClient.stop_experiment` — Stop a running experiment.
- :meth:`~agilerl.arena.client.ArenaClient.download_experiment_metrics` — Download training metrics as CSV to a local file.
- :meth:`~agilerl.arena.client.ArenaClient.deploy_agent` — Deploy a trained checkpoint to an inference endpoint.


.. _arena_deployment:

Agent Deployment
----------------

Once training is complete, you can deploy a trained agent to an Arena inference
endpoint and interact with it in real time. Deployed agents expose an HTTP API
that accepts observations and returns actions, making it easy to integrate
trained RL policies into applications.

Deploying an Agent
~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Deploy the best checkpoint from an experiment
         arena deploy lunar-lander-dqn

         # Deploy a specific checkpoint
         arena deploy lunar-lander-dqn --checkpoint step_500000

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena.client import ArenaClient

         with ArenaClient() as client:
             client.deploy_agent(
                 experiment_name="lunar-lander-dqn",
                 checkpoint="step_500000",  # optional, defaults to best
             )

Interacting with a Deployed Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once deployed, use the :class:`~agilerl.arena.inference.Agent` class to query
the endpoint:

.. code-block:: python

   import numpy as np
   from agilerl.arena.inference import Agent

   agent = Agent(
       "https://<deployment-id>.inference.agilerl.com",
       api_key="your-arena-api-key",
   )

   obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
   hidden_state = None

   # Get an action from the deployed agent
   status, action, hidden_state = agent.get_action(
       obs, hidden_state=hidden_state,
   )

   print(f"Action: {action}")

The ``Agent`` class handles serialization of NumPy arrays, supports batched
inference, dict/tuple observation spaces, and recurrent hidden states.

.. code-block:: python

   # Batched inference
   obs_batch = np.random.randn(8, 4)  # batch of 8 observations
   status, actions, hidden_state = agent.get_action(
       obs_batch, batched=True, hidden_state=hidden_state,
   )

   # Dict observation space
   obs_dict = {
       "image": np.random.randn(64, 64, 3),
       "velocity": np.array([1.0, 0.5]),
   }
   status, action, hidden_state = agent.get_action(obs_dict)

.. tutorial::

   :ref:`tutorial_arena_end_to_end`
      Complete walkthrough of validating, training, and deploying a custom environment on Arena.
