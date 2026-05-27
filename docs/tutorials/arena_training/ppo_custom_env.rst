.. _tutorial_arena_end_to_end:

Training PPO on a Custom Environment in Arena
==============================================

This tutorial walks through the full Arena workflow: validating a custom environment,
submitting a training job, monitoring progress, and deploying the trained agent — all
using a **BipedalWalker** environment as our example.

Prerequisites
-------------

Installation
~~~~~~~~~~~~

Arena requires additional packages (``httpx``, ``rich``, etc.) that are not included in the
base AgileRL installation. Install them with:

.. code-block:: bash

   pip install agilerl[arena]

This environment also requires Box2D (for example, ``pip install "gymnasium[box2d]"`` on Linux/macOS).

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

Our agent controls a simple 4-joint walker robot on uneven terrain. The goal is to move
forward as far as possible without falling. The environment follows the standard Gymnasium
interface with a continuous action space (motor torques for both hips and knees).

The environment source is taken from
`Gymnasium's BipedalWalker <https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/bipedal_walker.py>`_.

.. collapse:: bipedal_walker.py

   .. literalinclude:: /_static/examples/bipedal_walker.py
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
one we want to validate against to avoid ambiguity through the ``entrypoint`` parameter.
The ``bipedal_walker.py`` file defines a single Gymnasium environment class, ``BipedalWalker``,
so the ``entrypoint`` parameter is optional for this example.

If no version is specified when creating an environment from scratch, ``v1`` is used by default.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.validate_environment(
             source="bipedal_walker.py",
             name="bipedal-walker-env"
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena env validate \
             --source ./bipedal_walker.py \
             --name bipedal-walker-env

After validation succeeds, the environment is automatically profiled to determine its
resource usage. You will be able to view it in the **Environments** section of the Arena
dashboard, along with all of the data gathered for it during validation.

Creating a Project
~~~~~~~~~~~~~~~~~~

Before submitting a training job, we need to create a project to submit it to (if you have not already done so).

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         client.create_project(name="Bipedal Walker Tutorial")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena projects create "Bipedal Walker Tutorial"

.. tip::

   You can set a default project to work on by using the ``arena projects set-default <project-name>`` CLI command.

Submit a Training Job
~~~~~~~~~~~~~~~~~~~~~

With the environment validated, we can now submit a training job for it. For this example, we will
train a ``PPO`` agent on this continuous-control task. We define the training configuration in a YAML manifest
(``bipedal_walker_ppo.yaml``). Note how in the ``environment`` section we reference the validated environment by its
name as seen on Arena. If no version is specified, the latest one is used.

.. collapse:: bipedal_walker_ppo.yaml

   .. literalinclude:: /_static/examples/bipedal_walker_ppo.yaml
      :language: yaml


For this example, we will train on the ``arena-medium`` resource, which has 1x nvidia-l4 GPU, 15x CPUs, and 55GB of RAM
(costing around 2.41 credits/node-hour on Arena), and using 2 nodes for quicker results. Since we are training a population
size of 8, Arena will train 4 agents on each of the nodes in parallel.

.. tip::

   You can view all of the available resources to train on by running the CLI command ``arena resources list``.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.submit_experiment(
             manifest="bipedal_walker_ppo.yaml",
             resource_id="arena-medium",
             num_nodes=2,
             project="Bipedal Walker Tutorial",
             experiment_name="bipedal-walker-ppo-v1",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit \
             --manifest bipedal_walker_ppo.yaml \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project 'Bipedal Walker Tutorial' \
             --experiment-name bipedal-walker-ppo-v1

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
             experiment_name="bipedal-walker-ppo-v1",
             output_path="metrics.csv",
         )

         # List available checkpoints
         checkpoints = client.list_checkpoints(
             experiment_name="bipedal-walker-ppo-v1"
         )
         print(checkpoints)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Download metrics
         arena experiments metrics bipedal-walker-ppo-v1 --output-file metrics.csv

         # List checkpoints
         arena experiments checkpoints bipedal-walker-ppo-v1

Deploy the Trained Agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, deploy the best checkpoint to an inference endpoint:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Deploy the best checkpoint
         client.deploy_agent(experiment_name="bipedal-walker-ppo-v1")

         # Or deploy a specific checkpoint
         client.deploy_agent(
             experiment_name="bipedal-walker-ppo-v1",
             checkpoint="step_500000",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Deploy the best checkpoint
         arena agent deploy bipedal-walker-ppo-v1

         # Deploy a specific checkpoint
         arena agent deploy bipedal-walker-ppo-v1 --checkpoint step_500000

Interact with the Deployed Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After deployment, you can send observations and receive actions through the inference
API:

.. code-block:: python

   from agilerl.arena import Agent
   from bipedal_walker import BipedalWalker

   env = BipedalWalker()
   agent = Agent(experiment_name="bipedal-walker-ppo-v1")

   # Get an action from the deployed model
   observation, _ = env.reset()
   action = agent.get_action(observation)
   print(f"Agent chose action: {action}")

.. seealso::

   :ref:`arena_client` for the full reference on all Arena client methods.
