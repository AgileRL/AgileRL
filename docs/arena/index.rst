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

Installation
------------

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

         # All subsequent commands use the stored credentials
         arena env list

         # Or skip login entirely with an env var or flag
         export ARENA_API_KEY="arena_pat_..."
         arena env list

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         from agilerl.arena.client import ArenaClient

         # Option 1: Set ARENA_API_KEY env var — no login needed
         client = ArenaClient()
         client.list_environments()  # works immediately

         # Option 2: Pass the key explicitly
         client = ArenaClient(api_key="arena_pat_...")

         # Option 3: Interactive device login (one-time)
         client = ArenaClient()
         client.login()  # opens browser, persists credentials


.. _arena_environments:

Validating Custom Environments
------------------------------

Before launching a training run, Arena requires environments to be **validated**
— ensuring that the environment is importable, has the correct Gymnasium or
PettingZoo interface, and can be stepped without errors. After validation,
environments are automatically **profiled** to determine their compute and
memory footprint. This information is then used to estimate a ceiling on the resources
available for training, depending on the selected cluster tier.

Users can register environments in two ways:

- **Custom Gym/PettingZoo environments** — upload a source directory or
  ``.tar.gz`` archive containing your environment code.
- **LLM environments** — upload a dataset file or reference a HuggingFace
  dataset ID to create a language-model fine-tuning environment.

Uploading and Validating
~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Upload, create, and validate in one step
         arena env validate --source ./my_env/ --version v1

         # With an explicit entrypoint (when multiple exist)
         arena env validate --source ./my_env/ --version v1 \
             --entrypoint my_module:MyEnvClass

         # Validate an already-registered environment
         arena env validate my-env --version v1

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         from agilerl.arena.client import ArenaClient

         with ArenaClient() as client:
             # Upload, create, and validate in one step
             result = client.validate_environment(
                 source="./my_env/",
                 version="v1",
             )

             # With an explicit entrypoint
             result = client.validate_environment(
                 source="./my_env/",
                 version="v1",
                 entrypoint="my_module:MyEnvClass",
             )

             # Validate an already-registered environment
             result = client.validate_environment(name="my-env", version="v1")

Listing Environments and Entrypoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # List all registered environments
         arena env list

         # List available entrypoints for a specific environment
         arena env list-entrypoints my-env --version v1

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         with ArenaClient() as client:
             environments = client.list_environments()
             entrypoints = client.list_environment_entrypoints(
                 name="my-env", version="v1"
             )

Profiling
~~~~~~~~~

After validation, you can explicitly request a resource profile:

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena env profile my-env --version v1

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         with ArenaClient() as client:
             profile = client.profile_environment(name="my-env", version="v1")


.. _arena_training:

Submitting Experiments
----------------------

Once your environment is validated and profiled, you can submit training
experiments to Arena. Experiments run on managed cloud infrastructure with
automatic checkpointing, metric logging, and real-time monitoring accessible
directly from the Arena dashboard.

A training configuration is defined via a **manifest** — a YAML file describing
the algorithm, environment, training parameters, and HPO settings.

.. seealso::

   :ref:`training_manifests` section for full manifest schema and options.

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Submit an experiment
         arena experiment submit \
             --manifest configs/training/dqn/dqn.yaml \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project my-project \
             --experiment-name lunar-lander-dqn

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         from agilerl.arena.client import ArenaClient

         with ArenaClient() as client:
             result = client.submit_experiment(
                 manifest="configs/training/dqn/dqn.yaml",
                 resource_id="arena-medium",
                 num_nodes=2,
                 project="my-project",
                 experiment_name="lunar-lander-dqn",
             )

You can also submit experiments via the :class:`~agilerl.training.trainer.ArenaTrainer`
class, which provides a higher-level interface:

.. code-block:: python

   from agilerl.training.trainer import ArenaTrainer

   trainer = ArenaTrainer.from_manifest(
       manifest="configs/training/dqn/dqn.yaml",
   )
   result = trainer.train()

Managing Experiments
~~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # List experiments in a project
         arena experiment list --project my-project

         # List checkpoints for an experiment
         arena experiment checkpoints lunar-lander-dqn

         # Resume a stopped experiment
         arena experiment resume lunar-lander-dqn --max-steps 1000000

         # Stop a running experiment
         arena experiment stop lunar-lander-dqn

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         with ArenaClient() as client:
             # List experiments in a project
             experiments = client.list_experiments(project="my-project")

             # List checkpoints
             checkpoints = client.list_checkpoints(
                 experiment_name="lunar-lander-dqn"
             )

             # Resume a stopped experiment
             client.resume_experiment(
                 experiment_name="lunar-lander-dqn",
                 max_steps=1_000_000,
             )

             # Stop a running experiment
             client.stop_experiment("lunar-lander-dqn")

Downloading Metrics
~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: interface

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Download all metrics as CSV
         arena experiment metrics lunar-lander-dqn -o metrics.csv

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         with ArenaClient() as client:
             payload, content_type, _ = client.download_experiment_metrics(
                 experiment_name="lunar-lander-dqn",
             )
             with open("metrics.csv", "wb") as f:
                 f.write(payload)


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

   .. tab-item:: SDK
      :sync: sdk

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
