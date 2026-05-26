.. _tutorial_arena_end_to_end:

Training PPO on a Custom Environment in Arena
==============================================

This tutorial walks through the full Arena workflow: validating a custom environment,
submitting a training job, monitoring progress, and deploying the trained agent — all
using a **BinPacking2D** environment as our example.

Prerequisites
-------------

Installation
~~~~~~~~~~~~

Arena requires additional packages (``httpx``, ``rich``, etc.) that are not included in the
base AgileRL installation. Install them with:

.. code-block:: bash

   pip install agilerl[arena]

Authentication
~~~~~~~~~~~~~~

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

      .. tab-item:: Python
         :sync: python

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
~~~~~~~~~~~~~~~

Our agent must place randomly generated 2D packages into a 10×10 bin, maximising space
usage while respecting height and support constraints. The environment follows the
standard Gymnasium interface with a discrete action space (position × orientation × package).

.. collapse:: bin_packing_env.py

   .. literalinclude:: /_static/examples/bin_packing_env.py
      :language: python


Validate the Environment
~~~~~~~~~~~~~~~~~~~~~~~~

Before training, we need to register and validate our environment on Arena. Validation
ensures the environment is importable, has the correct interface, and can be stepped
without errors.

An entrypoint corresponds to the specific class we want to validate for training. In order to be
identified as a prospect for validation, a class must inherit from either of the following:

- ``gymnasium.Env``: Single-agent environments.
- ``pettingzoo.ParallelEnv``: Multi-agent environments.
- ``gem.env.GemEnv``: Multi-turn LLM environments.
- ``alpyne.sim.AnylogicSim``: Anylogic simulation environment.

All available entrypoints in the specified environment source are automatically identified before validation.
If there are multiple available entrypoints in the same file, we need to provide the
one we want to validate against to avoid ambiguity through the ``entrypoint`` parameter. Since there two entrypoints
``BinPacking2DEnv`` and ``BinPacking2DEnvCNN`` in the ``bin_packing_env.py`` file, we choose to validate the ``BinPacking2DEnv``
class for this example.

If no version is specified when creating an environment from scratch, ``v1`` is used by default.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.validate_environment(
             source="bin_packing_env.py",
             entrypoint="bin_packing_env:BinPacking2DEnv",
             name="bin-packing-env"
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena env validate \
             --source ./bin_packing_env.py \
             --entrypoint bin_packing_env:BinPacking2DEnv \
             --name bin-packing-env

.. note::

   The ``entrypoint`` parameter is optional and can be omitted if there is only one entrypoint in the environment source.

After validation succeeds, the environment is automatically profiled to determine its
resource requirements. You will be able to view it in the **Environments** section of the Arena
dashboard, along with all of the data gathered for it during validation.

Creating a Project
~~~~~~~~~~~~~~~~~~

Before submitting a training job, we need to create a project to submit it to (if you have not already done so).

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         client.create_project(name="BinPacking Tutorial")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena projects create "BinPacking Tutorial"

.. tip::

   You can set a default project to work on by using the ``arena projects set-default <project-name>`` command.

Submit a Training Job
~~~~~~~~~~~~~~~~~~~~~

With the environment validated, we can now submit a training job for it. For this example, we will
train a ``PPO`` agent since the action space is discrete. We define the training configuration in a YAML manifest
(``bin_packing_ppo.yaml``). Note how in the ``environment`` section we reference the validated environment by its
name as seen on Arena. If no version is specified, the latest one is used.

.. collapse:: bin_packing_ppo.yaml

   .. code-block:: yaml

      algorithm:
      name: PPO
      lr: 0.0003
      gamma: 0.99
      batch_size: 128
      learn_step: 4096
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
      pop_size: 8

      mutation:
      probabilities:
         no_mut: 0.4
         arch_mut: 0.2
         new_layer: 0.5
         params_mut: 0.2
         act_mut: 0.1
         rl_hp_mut: 0.1
      rl_hp_selection:
         lr:
            min: 0.0001
            max: 0.01
         batch_size:
            min: 8
            max: 1024
         learn_step:
            min: 256
            max: 8192
         ent_coef:
            min: 0.001
            max: 0.1
         update_epochs:
            min: 1
            max: 10

      network:
      latent_dim: 256
      max_latent_dim: 512
      arch: mlp
      encoder_config:
         hidden_size:
            - 256
         activation: ReLU
         min_mlp_nodes: 64
         max_mlp_nodes: 500
      head_config:
         hidden_size:
            - 256
         activation: ReLU
         min_hidden_layers: 1
         max_hidden_layers: 3
         min_mlp_nodes: 64
         max_mlp_nodes: 500

      tournament:
      tournament_size: 2
      elitism: true


For this example, we will train on the ``arena-medium`` resource, which has 1x nvidia-l4 GPU, 15x CPUs, and 55GB of RAM
(costing around 2.41 credits/node-hour on Arena), and using 2 nodes for quicker results. Since we are training a population
size of 6, Arena will train 3 agents on each of the nodes in parallel.

.. tip::

   You can view all of the available resources to train on by running the CLI command ``arena resources list``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

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

.. warning::

   The training cost scales linearly with the number of nodes. In this case, the training cost will be
   2.41 credits/node-hour * 2 nodes = 4.82 credits / hour.

Monitor Training
~~~~~~~~~~~~~~~~

You can monitor training progress directly from the Arena dashboard, or download
metrics programmatically:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

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

Deploy the Trained Agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, deploy the best checkpoint to an inference endpoint:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

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

Interact with the Deployed Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
