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
   pip install agilerl

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
the nearest aircraft. The action is continuous, with shape ``(2,)``: - a heading change and a speed change.
PPO handles continuous actions with a Gaussian policy, which makes it a good fit for this task.

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

When submitting an environment for validation, ``agilerl-arena`` automatically packages the whole folder.
``requirements.txt`` is installed on the validation environment before the checks are ran, and
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
ensures the environment is importable, has the correct interface, and can be stepped reliably
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
so passing ``entrypoint`` is unnecessary.

If no version is specified when creating an environment from scratch, ``v1`` is used by default.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.validate_environment(
             name="merge-env",
             source="merge-env/",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena env validate merge-env --source merge-env/

Validation uploads the environment, installs its requirements, and runs a series of interface
checks. For the environment as shipped, some of these checks fail:

.. container:: scrollable-output

   .. code-block:: text

      INFO     No version specified, defaulting to v1.
      INFO     Uploading environment 'merge-env:v1' (13.4 KB)
      INFO     Installing requirements…
      INFO       Resolving dependencies…
      INFO       Installing 23 package(s)
      INFO       Downloading kiwisolver
      INFO       Downloading pillow
      INFO       Downloading pandas
      INFO       Downloading fonttools
      INFO       Downloading matplotlib
      INFO       Downloading scipy
      INFO       Downloading pygame
      INFO       Downloading openap
      INFO       Downloading bluesky-navdata
      INFO       Installed bluesky-gym 0.2.0
      INFO       Installed bluesky-navdata 1.0.0
      INFO       Installed bluesky-simulator 1.1.1
      INFO       Installed cloudpickle 3.1.2
      INFO       Installed contourpy 1.3.3
      INFO       Installed cycler 0.12.1
      INFO       Installed fonttools 4.63.0
      INFO       Installed kiwisolver 1.5.0
      INFO       Installed matplotlib 3.11.0
      INFO       Installed msgpack 1.2.1
      INFO       Installed openap 2.6.0
      INFO       Installed packaging 26.2
      INFO       Installed pandas 3.0.3
      INFO       Installed pillow 12.3.0
      INFO       Installed pygame 2.6.1
      INFO       Installed pyparsing 3.3.2
      INFO       Installed python-dateutil 2.9.0.post0
      INFO       Installed pyyaml 6.0.3
      INFO       Installed pyzmq 27.1.0
      INFO       Installed scipy 1.18.0
      INFO       Installed six 1.17.0
      INFO       Installed stable-baselines3 2.9.0
      INFO       Installed zmq 0.0.0
      INFO     Installed 23 package(s)
      INFO     Identifying available entrypoints
      INFO     Environment uploaded successfully
      INFO     Running validation checks for 'merge_env:MergeEnv'
      ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
      ┃ Check                         ┃ Result ┃ Details                                                                     ┃
      ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
      │ Environment class path        │ PASS   │                                                                             │
      │ Class exists in path          │ PASS   │                                                                             │
      │ Preliminary environment       │ PASS   │                                                                             │
      │ Action space                  │ PASS   │                                                                             │
      │ Action space limits           │ PASS   │                                                                             │
      │ Observation space             │ PASS   │                                                                             │
      │ Observation space limits      │ PASS   │                                                                             │
      │ Seed deprecation              │ PASS   │                                                                             │
      │ Reset return info deprecation │ PASS   │                                                                             │
      │ Reset return type             │ FAIL   │ dtype error in key faf_reached of observation: The observation dtype does   │
      │                               │        │ not match the dtype defined in the observation space. Returned observation  │
      │                               │        │ has dtype int64, expected float64.                                          │
      │ Reset seed                    │ FAIL   │ dtype error in key faf_reached of observation: The observation dtype does   │
      │                               │        │ not match the dtype defined in the observation space. Returned observation  │
      │                               │        │ has dtype int64, expected float64.                                          │
      │ Reset options                 │ PASS   │                                                                             │
      │ Reset                         │ PASS   │                                                                             │
      │ Step                          │ FAIL   │ The obs returned by the `step()` method was expecting observation numpy     │
      │                               │        │ array dtype to be float64, actual type: int64                               │
      │ Episode lifecycle             │ PASS   │                                                                             │
      └───────────────────────────────┴────────┴─────────────────────────────────────────────────────────────────────────────┘
      INFO     Validation checks did not pass. Please review the errors above and re-validate the environment.

The failing checks all point to the same issue: the ``faf_reached`` key of the observation. The
observation space declares it as ``float64``:

.. code-block:: python

   "faf_reached": spaces.Box(0, 1, shape=(1,), dtype=np.float64),

but the environment builds that key with ``np.array([self.wpt_reach])``, and since
``self.wpt_reach`` is an integer this returns an ``int64`` array. The fix is to declare the space
with the same integer dtype the environment actually returns:

.. code-block:: python

   "faf_reached": spaces.Box(0, 1, shape=(1,), dtype=np.int64),

Re-run the validation command with the corrected environment and all checks now pass (you will need to
give it a new version v2).

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

For this example, we will train a ``PPO`` agent on this task. We define the training configuration in a YAML manifest
(``merge_ppo.yaml``). Note how in the ``environment`` section we reference the validated environment by its
name as seen on Arena. If no version is specified, the latest one is used.

.. collapse:: merge_ppo.yaml

   .. literalinclude:: /_static/examples/merge_ppo.yaml
      :language: yaml

Notice that the ``network`` section doesn't set an ``arch``. Arena infers the encoder
architecture from the environment's observation space, so a ``Dict`` observation like ours
gets a multi-input encoder automatically. Set ``simba: true`` in the network config, or
``recurrent: true`` in the algorithm config, to opt into a SimBa or recurrent encoder instead.

For this example, we will train on the ``arena-medium`` resource, which has 1x nvidia-l4 GPU, 15x CPUs, and 55GB of RAM
(costing around 2.41 credits/node-hour on Arena), and using 2 nodes for quicker results. Since we are training a population
size of 4, Arena will train 2 agents on each of the nodes in parallel.

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
deployment name matches the experiment you deployed by default. You can list available deployments
with ``arena agent list``.

.. code-block:: python

   from agilerl.arena import ArenaClient
   from merge_env import MergeEnv

   # Initialize the environment
   env = MergeEnv()

   # Initialize the client and the deployed agent
   with ArenaClient() as client:
      with client.open_inference_agent("merge-ppo-v1") as agent:
         observation, _ = env.reset()
         action, _ = agent.get_action(observation)
         print(f"Action: {action}")

.. seealso::

   :ref:`arena_client` for the full reference on all Arena client methods.
