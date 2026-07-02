.. _tutorial_arena_end_to_end:

Training PPO on a Custom Environment in Arena
==============================================

This tutorial walks through the full Arena workflow: validating a custom environment,
submitting a training job, monitoring progress, and deploying the trained agent. We use a
**Merge** environment, an air traffic arrival manager, as our running example.

Prerequisites
-------------

Installation
~~~~~~~~~~~~

Arena is provided by the separate ``agilerl-arena`` distribution (``agilerl.arena.*``
in the shared namespace). It pulls in lightweight client dependencies (``httpx``,
``rich``, etc.) rather than core training stacks. Install it directly, or through
the AgileRL extra:

.. code-block:: bash

   pip install agilerl-arena
   # or
   pip install "agilerl[arena]"

Authentication
~~~~~~~~~~~~~~

All Arena operations require authentication. You can authenticate in one of two ways:

1. **API Key**: set the ``ARENA_API_KEY`` environment variable:

   .. code-block:: bash

      export ARENA_API_KEY="arena_pat..."

   .. note::

      Personal access tokens can be found in the Arena dashboard under
      *Profile Management* → *CLI API Key*.

2. **Device login** (interactive): run the login command, which opens a browser for OAuth authentication:

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

Our agent is an air traffic controller. It guides a single aircraft (the ownship) into a
stream of incoming aircraft so that it reaches the final approach fix before the runway,
while keeping a safe separation from the other aircraft and staying on course. The
environment is built on the `BlueSky <https://github.com/TUDelft-CNS-ATM/bluesky>`_ air
traffic simulator and follows the standard Gymnasium interface.

The observation is a dictionary describing the ownship (its drift from the target heading,
airspeed, distance to the next waypoint) and the relative position, velocity and track of
the nearest aircraft. The action is continuous, with shape ``(2,)``: a heading change and a
speed change. PPO handles continuous actions with a Gaussian policy, which makes it a good
fit for this task.

The environment source is taken from
`bluesky-gym <https://github.com/TUDelft-CNS-ATM/bluesky-gym>`_.

.. collapse:: merge_env.py

   .. literalinclude:: /_static/examples/merge-env/merge_env.py
      :language: python


Environment Directory Layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We keep the environment in its own directory. Alongside the code, we include a
``requirements.txt`` with the packages the environment needs, and an ``env_config.yaml``
with the keyword arguments passed to the environment's constructor:

.. code-block:: text

   merge-env/
   ├── merge_env.py
   ├── requirements.txt
   └── env_config.yaml

When you point Arena at a directory, it packages the whole folder and picks these files up
automatically. ``requirements.txt`` is installed on Arena before your environment runs, and
``env_config.yaml`` is applied when the environment is created:

.. literalinclude:: /_static/examples/merge-env/requirements.txt
   :caption: requirements.txt

.. literalinclude:: /_static/examples/merge-env/env_config.yaml
   :caption: env_config.yaml
   :language: yaml

If these files live elsewhere, you can point to them directly with the ``--requirements``
and ``--env-config`` options (or the matching arguments in Python).

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
The ``merge_env.py`` file defines a single Gymnasium environment class, ``MergeEnv``,
so the ``entrypoint`` parameter is optional for this example.

If no version is specified when creating an environment from scratch, ``v1`` is used by default.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.validate_environment(
             source="merge-env/",
             name="merge-env",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena env validate \
             --source merge-env/ \
             --name merge-env

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

         client.create_project(name="Merge Tutorial")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena projects create "Merge Tutorial"

.. tip::

   You can set a default project to work on by using the ``arena projects set-default <project-name>`` CLI command.

Submit a Training Job
~~~~~~~~~~~~~~~~~~~~~

With the environment validated, we can now submit a training job for it. For this example, we will
train a ``PPO`` agent on this task. We define the training configuration in a YAML manifest
(``merge_ppo.yaml``). Note how in the ``environment`` section we reference the validated environment by its
name as seen on Arena. If no version is specified, the latest one is used.

.. collapse:: merge_ppo.yaml

   .. literalinclude:: /_static/examples/merge_ppo.yaml
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
             manifest="merge_ppo.yaml",
             resource_id="arena-medium",
             num_nodes=2,
             project="Merge Tutorial",
             experiment_name="merge-ppo-v1",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena experiments submit merge_ppo.yaml \
             --resource-id arena-medium \
             --num-nodes 2 \
             --project 'Merge Tutorial' \
             --experiment-name merge-ppo-v1

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
             experiment_name="merge-ppo-v1",
             output_path="metrics.csv",
         )

         # List available checkpoints
         checkpoints = client.list_checkpoints(
             experiment_name="merge-ppo-v1"
         )
         print(checkpoints)

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Download metrics
         arena experiments metrics merge-ppo-v1 --output-file metrics.csv

         # List checkpoints
         arena experiments checkpoints merge-ppo-v1

Deploy the Trained Agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Once training is complete, deploy the best checkpoint to an inference endpoint:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Deploy the best checkpoint
         client.deploy_agent(experiment_name="merge-ppo-v1")

         # Or deploy a specific checkpoint
         client.deploy_agent(
             experiment_name="merge-ppo-v1",
             checkpoint="step_500000",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Deploy the best checkpoint
         arena agent deploy merge-ppo-v1

         # Deploy a specific checkpoint
         arena agent deploy merge-ppo-v1 --checkpoint step_500000

Interact with the Deployed Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After deployment, open the agent by name and send it observations to receive actions. The
deployment name matches the experiment you deployed; you can also list your deployments with
``arena agent list``.

.. code-block:: python

   from agilerl.arena import ArenaClient
   from merge_env import MergeEnv

   client = ArenaClient()
   agent = client.open_inference_agent("merge-ppo-v1")

   # Get an action from the deployed model
   env = MergeEnv()
   observation, _ = env.reset()
   action, _ = agent.get_action(observation)
   print(f"Agent chose action: {action}")

.. seealso::

   :ref:`arena_client` for the full reference on all Arena client methods.
