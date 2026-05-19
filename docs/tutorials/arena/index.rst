.. _tutorial_arena_end_to_end:

Training PPO on a Custom Environment in Arena
==============================================

This tutorial walks through the full Arena workflow: validating a custom environment,
submitting a training job, monitoring progress, and deploying the trained agent — all
using a **BinPacking2D** environment as our example.

Prerequisites
-------------

Install Arena dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Arena requires additional packages (``httpx``, ``rich``, etc.) that are not included in the
base AgileRL installation. Install them with:

.. code-block:: bash

   pip install agilerl[arena]

Authenticate with Arena
~~~~~~~~~~~~~~~~~~~~~~~

All Arena operations require authentication. You can authenticate in one of two ways:

1. **API Key** — set the ``ARENA_API_KEY`` environment variable:

   .. code-block:: bash

      export ARENA_API_KEY="arena_pat..."

   .. note::

      Personal access tokens can be found in the Arena dashboard under
      *Profile Management* → *CLI API Key*.

2. **Device login** (interactive) — run the login command, which opens a browser for OAuth authentication:

   .. tab-set::
      :sync-group: interface

      .. tab-item:: SDK
         :sync: sdk

         .. code-block:: python

            from agilerl.arena import ArenaClient

            client = ArenaClient()
            client.login()

      .. tab-item:: CLI
         :sync: cli

         .. code-block:: bash

            arena login

   Credentials are persisted locally so you only need to authenticate once per machine.

The Environment
---------------

Our agent must place randomly generated 2D packages into a 10×10 bin, maximising space
usage while respecting height and support constraints. The environment follows the
standard Gymnasium interface with a discrete action space (position × orientation × package).

.. collapse:: bin_packing_env.py

   .. literalinclude:: /_static/examples/bin_packing_env.py
      :language: python


Step 1: Validate the Environment
---------------------------------

Before training, we need to register and validate our environment on Arena. Validation
ensures the environment is importable, has the correct interface, and can be stepped
without errors. Since there are multiple available entrypoints in the same file, we need to provide the
specific class we want to validate for training to avoid ambiguity.

.. tab-set::
   :sync-group: interface

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         from agilerl.arena import ArenaClient

         client = ArenaClient()

         result = client.validate_environment(
             source="./bin_packing_env.py",
             entrypoint="bin_packing_env:BinPacking2DEnv",
             name="bin-packing-env"
         )
         print(result)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena env validate \
             --source ./custom-gym-env/ \
             --entrypoint bin_packing_env:BinPacking2DEnv \
             --name bin-packing-env

After validation succeeds, the environment is automatically profiled to determine its
resource requirements. You will be able to view it in the **Environments** section of the Arena
dashboard.

Step 2: Submit a Training Job
------------------------------

With the environment validated, we can submit a training experiment. We define the
training configuration in a YAML manifest. Note how in the ``environment`` section we reference the
environment by its name as seen on Arena. If no version is specified, the latest one is used.

.. collapse:: bin_packing_ppo.yaml

   .. code-block:: yaml

      algorithm:
        name: PPO
        lr: 0.0003
        gamma: 0.99
        batch_size: 64
        learn_step: 2048
        gae_lambda: 0.95
        clip_coef: 0.2
        ent_coef: 0.01
        vf_coef: 0.5
        update_epochs: 4

      environment:
        name: bin-packing-env
        version: v1

      training:
        max_steps: 5_000_000
        evo_steps: 10_000
        pop_size: 6

      mutation:
        no_mutation: 0.4
        architecture: 0.2
        new_layer_prob: 0.5
        parameters: 0.2
        activation: 0.1
        rl_hp: 0.1

      tournament:
        tournament_size: 2
        elitism: true

.. tab-set::
   :sync-group: interface

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         result = client.submit_experiment(
             manifest="bin_packing_ppo.yaml",
             resource_id="arena-medium",
             num_nodes=2,
             project="BinPacking Tutorial",
             experiment_name="bin-packing-ppo-v1",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit \
             --manifest bin_packing_ppo.yaml \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project 'BinPacking Tutorial' \
             --experiment-name bin-packing-ppo-v1

.. note::

   Since we specified two nodes and a population size of 6, Arena will train 3 agents on each of the nodes in parallel.

Step 3: Monitor Training
-------------------------

You can monitor training progress directly from the Arena dashboard, or download
metrics programmatically:

.. tab-set::
   :sync-group: interface

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         # Download training metrics
         client.download_experiment_metrics(
             experiment_name="bin-packing-ppo-v1",
             output_path="metrics.csv",
         )

         # List available checkpoints
         checkpoints = client.list_checkpoints(
             experiment_name="bin-packing-ppo-v1"
         )
         print(checkpoints)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Download metrics
         arena experiments metrics bin-packing-ppo-v1 --output-file metrics.csv

         # List checkpoints
         arena experiments checkpoints bin-packing-ppo-v1

Step 4: Deploy the Trained Agent
---------------------------------

Once training is complete, deploy the best checkpoint to an inference endpoint:

.. tab-set::
   :sync-group: interface

   .. tab-item:: SDK
      :sync: sdk

      .. code-block:: python

         # Deploy the best checkpoint
         client.deploy_agent(experiment_name="bin-packing-ppo-v1")

         # Or deploy a specific checkpoint
         client.deploy_agent(
             experiment_name="bin-packing-ppo-v1",
             checkpoint="step_500000",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Deploy the best checkpoint
         arena experiments deploy bin-packing-ppo-v1

         # Deploy a specific checkpoint
         arena experiments deploy bin-packing-ppo-v1 --checkpoint step_500000

Step 5: Interact with the Deployed Agent
-----------------------------------------

After deployment, you can send observations and receive actions through the inference
API:

.. code-block:: python

   from agilerl.arena import Agent

   agent = Agent(experiment_name="bin-packing-ppo-v1")

   # Get an action from the deployed model
   observation = env.reset()[0]
   action = agent.get_action(observation)
   print(f"Agent chose action: {action}")

.. seealso::

   :ref:`arena_client` for the full reference on all Arena client methods.
