.. _arena_client:

Arena Client
============

Arena is an **RLOps platform** that streamlines the reinforcement learning
development process and accelerates iteration through distributed training and
evolutionary hyperparameter optimization. It provides managed cloud
infrastructure purpose-built for RL workloads, so researchers and engineers can
focus on algorithms and environments rather than cluster management.

The Arena client, available as both a **CLI** (``arena``) and a **Python SDK**
(:class:`~agilerl.arena.client.ArenaClient`), allows registered users to do
everything from their own development environment: validate custom environments,
submit training experiments, monitor progress, and deploy trained agents for
inference.

.. tip::

  `Sign up to Arena <https://arena.agilerl.com>`_ for free now and get **110 free training credits (~20 hours)** to get started!

Installation
------------

To use the Arena client, install the standalone package directly with lightweight dependencies, or install AgileRL, which depends on it:

.. code-block:: bash

   pip install agilerl-arena
   # or
   pip install agilerl

Authentication
--------------

Authentication is resolved automatically in priority order:

1. **Organisation key** plus **external user id** (constructor arguments
   ``org_key`` / ``external_user_id``, or ``ARENA_ORG_KEY`` /
   ``ARENA_EXTERNAL_USER_ID``). Sends ``Authorization: Bearer <org key>``
   and the header ``X-External-User-Id``. For partner servers.
2. **api_key argument** passed directly to the client or CLI.
3. **ARENA_API_KEY environment variable**: if set, no login is needed.
4. **Stored OAuth credentials** from ``~/.arena/credentials.json`` (persisted
   after a successful ``arena login``).
5. **Interactive device login**: opens a browser for OAuth authorization.

The simplest approach for scripting and CI is to set the environment variable:

.. code-block:: bash

   export ARENA_API_KEY="arena_pat_..."

.. note::
   Personal access tokens can be found in the Arena account profile, under *Profile management* -> *CLI API Key*.
   Organisation keys (``arena_org_…``) are minted by platform operators for partner
   integrations and must be sent with ``X-External-User-Id``.

Once set, all CLI commands and SDK calls authenticate automatically without
requiring ``arena login``.

For interactive use, ``arena login`` opens a browser-based OAuth flow and
persists credentials locally so you only need to log in once per machine:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         from agilerl.arena.client import ArenaClient

         # Option 1: Set ARENA_API_KEY env var, no login needed
         client = ArenaClient()

         # Option 2: Pass the key explicitly
         client = ArenaClient(api_key="arena_pat_...")

         # Option 3: Partner organisation key (requires an external user id)
         client = ArenaClient(
             org_key="arena_org_...",
             external_user_id="partner-user-id",
         )

         # Option 4: Interactive device login (one-time)
         client = ArenaClient()
         client.login()  # opens browser, persists credentials

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # One-time interactive login (persists to ~/.arena/credentials.json)
         arena login

         # Or skip login entirely with an env var or flag
         export ARENA_API_KEY="arena_pat_..."
         # Partner servers:
         export ARENA_ORG_KEY="arena_org_..."
         export ARENA_EXTERNAL_USER_ID="partner-user-id"


.. _arena_environments:

Custom Environments
-------------------

Before launching a training run on a custom environment or dataset, Arena requires these to be **validated**,
ensuring that the environment is importable, follows the GEM API, and can be stepped without errors.

Users can register two types of environments on Arena:

- **Custom GEM environments**: upload a source directory, file, or
  ``.tar.gz`` archive containing your multi-turn environment code.
- **LLM Datasets**: upload a dataset file or reference a HuggingFace
  dataset ID to create a language-model fine-tuning environment.

Uploading and Validating
~~~~~~~~~~~~~~~~~~~~~~~~

In order to create and validate a custom GEM environment from scratch, users need to provide a source
file or directory containing the environment code. Additionally, an entrypoint can be provided to the environment class to use, which is useful when multiple entrypoints exist in the same path.
Please refer to the :meth:`ArenaClient.validate_environment <agilerl.arena.client.ArenaClient.validate_environment>` method documentation for more details.

When you point ``source`` at a directory, Arena packages the whole folder and uploads it. A
``requirements.txt`` in the folder is installed before your environment runs, and an
``env_config.yaml`` is used to set the arguments passed to your environment's constructor. You
can also pass these files explicitly with ``requirements`` and ``env_config`` (``--requirements``
and ``--env-config`` on the CLI).

Once your environment / dataset has been validated successfully, you will be able to view it in the **Environments / Datasets** section in Arena.

GEM Environments
^^^^^^^^^^^^^^^^

The following commands use a ``my-gem-env/`` directory containing a GEM environment.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Upload, create, and validate in one step
         result = client.validate_environment(
             source="my-gem-env/",
             name="my-gem-env",
         )

         # Re-validate an already-registered environment
         result = client.validate_environment(name="my-gem-env", version="v1")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Upload and validate in one step
         arena env validate my-gem-env \
             --source my-gem-env/

         # Re-validate an already-registered environment
         arena env validate my-gem-env --version v1

LLM Datasets
^^^^^^^^^^^^^

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.create_dataset(
             name="my-dataset",
             category="reasoning",
             column_mapping={"question": "prompt", "answer": "completion"},
             file="./my_dataset/data.csv",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Upload a local CSV and create the dataset
         arena datasets create my-dataset \
             --category reasoning \
             --column-mapping '{"question": "prompt", "answer": "completion"}' \
             --file ./my_dataset/data.csv

Recorded model calls (agent traces) go in a ``traces`` dataset: OpenAI chat
messages in prompt-completion rows, checked locally before upload and trained
by SFT or on-policy distillation with no column mapping. See :doc:`traces`.

.. code-block:: bash

   arena datasets create helpdesk-traces --category traces --file ./traces.jsonl

Additional Tools
^^^^^^^^^^^^^^^^

Here is a list of additional methods provided by the :class:`~agilerl.arena.client.ArenaClient` to help you navigate the custom environments workflow in Arena.
You can find the analogous CLI commands by running ``arena env --help``.

- :meth:`~agilerl.arena.client.ArenaClient.list_environments`: List all registered environments and their versions.
- :meth:`~agilerl.arena.client.ArenaClient.list_environment_entrypoints`: List available entrypoints for a specific environment version.
- :meth:`~agilerl.arena.client.ArenaClient.environment_exists`: Check whether an environment (and optionally a version) is registered.
- :meth:`~agilerl.arena.client.ArenaClient.delete_environment`: Delete one or all versions of a registered environment.

.. _arena_models:

HuggingFace Models
------------------

Arena keeps a catalog of HuggingFace models you can train. List the catalog, or
fetch LoRA target modules and context length for one model.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         models = client.list_models()
         for row in models:
             print(row["model_name"], row["num_params"], row["max_context_length"])

         info = client.get_model_info("ibm-granite/granite-3.3-2b-instruct")
         print(info["modules"], info["max_context_length"])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena models list
         arena models info ibm-granite/granite-3.3-2b-instruct

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

.. _arena_secrets:

Organization Secrets
--------------------

Secrets hold API keys your training jobs need, such as an LLM judge's provider key.
In a manifest, ``judge.api_key_env`` names one of these secrets, and Arena passes it
to the job as an environment variable of that name. Any org member can list, set and
delete secrets. Arena never shows any part of a value, only each secret's name, when
it was created and last updated, and who last updated it.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         import os

         client.list_secrets()
         client.set_secret(name="OPENAI_API_KEY", value=os.environ["OPENAI_API_KEY"])
         client.delete_secret("OPENAI_API_KEY")

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena secrets list

         # Hidden prompt; or pipe the value on stdin, or use --from-env VAR
         arena secrets set OPENAI_API_KEY
         arena secrets set OPENAI_API_KEY --from-env OPENAI_API_KEY

         arena secrets delete OPENAI_API_KEY --yes

The CLI never takes the value as an argument, so it stays out of your shell history.

.. _arena_training:

Submitting Experiments
----------------------

Once your environment or dataset has been validated, you can submit training
jobs for it to Arena. AgileRL has its own managed cloud infrastructure with
automatic checkpointing, metric logging, and real-time monitoring accessible
directly from the Arena dashboard. Training runs in a distributed manner across multiple nodes
(this can be customized by the user) to achieve lightning-fast training for arbitrarily large
populations of agents.

Available GPU Types on Arena
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arena trains on several GPU types. These might change over time, so it is recommended to use the
:meth:`ArenaClient.list_resources <agilerl.arena.client.ArenaClient.list_resources>` method to get
the latest list before setting off an experiment. Pass a GPU type as ``gpu_type`` (``--gpu-type`` on
the CLI) and a GPU count as ``num_gpus`` (``--num-gpus``) when you submit.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         resources = client.list_resources()

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena resources list

Example output from CLI command. Arena cloud L4 and A100 GPUs are not available for LLM training,
so they are not listed:

.. list-table::
   :header-rows: 1
   :widths: 20 20 10

   * - GPU type
     - GPUs per node
     - Dev only
   * - gpu-b200-sxm-a
     - 1, 8
     -
   * - gpu-rtx6000-a
     - 1, 8
     -
   * - h100-80
     - 8
     -

Job Submission
~~~~~~~~~~~~~~
A training configuration is defined via a **manifest** (YAML or JSON file) describing
the algorithm, environment, training parameters, and evolutionary HPO settings. The formulation of the manifest is described in the :ref:`training_manifests` section,
and is mostly analogous for both the :class:`~agilerl.training.trainer.LocalTrainer` and training jobs in Arena.

For an LLM manifest, check the GPU before you submit. ``arena memory estimate``
prints the training and generation bars against the card; ``arena memory solve``
finds the longest context the run fits, given the rest of the manifest.
See :ref:`llm_memory_estimate`.

Here is an example manifest for fine-tuning ``Qwen/Qwen3.5-0.8B`` with GRPO on GSM8K:

.. collapse:: gsm8k_grpo.yaml

   .. literalinclude:: /_static/examples/gsm8k_grpo.yaml
      :language: yaml

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         result = client.submit_experiment(
             manifest="gsm8k_grpo.yaml",
             reward_file="reward.py",
             project="my-project",
             experiment_name="gsm8k-grpo",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Submit an experiment
         arena experiments submit gsm8k_grpo.yaml \
             --reward-file reward.py \
             --project my-project \
             --experiment-name gsm8k-grpo

You can also submit experiments via the :class:`~agilerl.training.trainer.ArenaTrainer`
class, which provides a higher-level interface:

.. code-block:: python

   from agilerl.training.trainer import ArenaTrainer

   # Don't need to provide a client if we have already authenticated through `arena login`
   trainer = ArenaTrainer.from_manifest(
       manifest="my_manifest.yaml",
   )
   result = trainer.train()

.. seealso::

   - :ref:`training_manifests` section for an overview of the training manifest and its options.

   - :ref:`trainers` section for more information on the ``ArenaTrainer`` class and its usage.

   - :ref:`llm_distributed` for sizing multi-GPU LLM runs (``fsdp``, ``training_gpus_per_agent``).

Additional Tools
^^^^^^^^^^^^^^^^

Here is a list of additional methods provided by the :class:`~agilerl.arena.client.ArenaClient` for managing experiments.
You can find the analogous CLI commands by running ``arena experiments --help``.

- :meth:`~agilerl.arena.client.ArenaClient.list_experiments`: List all experiments in a project.
- :meth:`~agilerl.arena.client.ArenaClient.list_checkpoints`: List saved checkpoints for an experiment.
- :meth:`~agilerl.arena.client.ArenaClient.resume_experiment`: Resume a stopped experiment with a new step budget.
- :meth:`~agilerl.arena.client.ArenaClient.stop_experiment`: Stop a running experiment.
- :meth:`~agilerl.arena.client.ArenaClient.download_experiment_metrics`: Download training metrics as CSV to a local file.
- :meth:`~agilerl.arena.client.ArenaClient.deploy_agent`: Deploy a trained checkpoint to an inference endpoint.


.. _arena_deployment:

Agent Deployment
----------------

Once training is complete, you can deploy a checkpoint from an experiment to an Arena inference
endpoint and interact with it in real time. Deployed agents expose an HTTP API
that accepts prompts and returns completions, making the integration of
fine-tuned models into applications seamless.

To deploy an agent, you can use the :meth:`~agilerl.arena.client.ArenaClient.deploy_agent` method by specifying the experiment
name and optionally the checkpoint you wish to deploy. If the latter is not provided, the checkpoint with the largest fitness value
will be deployed by default.

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         # Deploy an agent from an experiment
         client.deploy_agent(
            experiment_name="<experiment-name>",
            checkpoint="<checkpoint-name>",  # optional, defaults to best fitness
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         # Deploy the best checkpoint from an experiment
         arena agent deploy gsm8k-grpo

         # Deploy a specific checkpoint
         arena agent deploy gsm8k-grpo --checkpoint step_500

GPUs
~~~~

Pick the GPUs an agent is served on with ``num_gpus`` and
``gpu_type`` (a short GPU name such as ``h100-80``, ``b200`` or ``l4``):

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         client.deploy_agent(
            experiment_name="<experiment-name>",
            num_gpus=2,
            gpu_type="h100-80",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena agent deploy my-chat-model --num-gpus 2 --gpu-type h100-80

Without either option (``--num-gpus`` / ``--gpu-type`` on the CLI), a redeploy
keeps the agent's resources. A new deployment runs on the cluster the agent was
trained on if that is your own cluster, else on the cheapest node type that can
serve the model. If Arena can't size the model, it uses its default node type
and rejects ``num_gpus`` and ``gpu_type``.

Memory Scope
~~~~~~~~~~~~

An LLM deployment keeps chat sessions either per user or across your whole
organisation. Choose which with ``memory_scope`` when you first deploy:

.. tab-set::
   :sync-group: interface

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         client.deploy_agent(
            experiment_name="<experiment-name>",
            memory_scope="organization",
         )

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: bash

         arena agent deploy my-chat-model --memory-scope organization

A new deployment uses ``user`` when you leave it out, so each caller only ever
sees their own conversations. The scope is fixed once the deployment exists.
Redeploying without the option keeps whatever is stored rather than resetting it
to ``user``, and it cannot be changed afterwards.

Deployments show their scope in ``arena agent list``.

Interacting with a Deployed Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each deployment runs one agent type. The simplest way to interact with one is to open it by
name with :meth:`~agilerl.arena.client.ArenaClient.open_inference_agent`, which returns an
:class:`~agilerl.arena.inference.Agent` you can make requests to:

.. code-block:: python

   from agilerl.arena import ArenaClient

   with ArenaClient() as client:
       with client.open_inference_agent("<deployment-name>") as agent:
           print(agent.metadata.agent.algo, agent.metadata.agent.llm)

Call the matching agent method; the server returns HTTP 400 if the route does
not match the deployment:

- **Supervised (SFT)**: :meth:`~agilerl.arena.inference.Agent.predict`
- **LLM**: :meth:`~agilerl.arena.inference.Agent.generate` or
  :meth:`~agilerl.arena.inference.Agent.generate_stream`

.. tip::

   For LLM deployments, set an active agent with ``arena agent run <deployment-name>`` and you
   can drop the deployment name from later CLI commands. ``arena agent generate --prompt "..."``
   then makes inference requests to the active agent.

Deployment URLs are cached in ``~/.arena/inference.json`` so ordinary commands do
not have to look them up. Redeploying moves a deployment to a new URL, which the
client notices on its own: a 404 from the cached URL refetches it once and
retries, so the stale entry repairs itself. ``--refresh`` forces the lookup up
front if you would rather not rely on that.

Chat Sessions
^^^^^^^^^^^^^

LLM deployments can keep chat history. Pass ``session_id`` to continue an earlier
conversation, and list what is stored with
:meth:`~agilerl.arena.inference.Agent.list_sessions`:

.. code-block:: python

   with ArenaClient() as client:
       with client.open_inference_agent("<deployment-name>") as agent:
           for session in agent.list_sessions():
               print(session.session_id, session.created_at, session.last_updated)

           results = agent.generate("What did I just ask?", session_id="<session-id>")

Sessions come back most recently updated first. Leave ``session_id`` out and the
request does not carry one at all, so the deployment starts a new conversation.
Nothing on this side invents a session id for you.

The same is available from the CLI with ``arena agent sessions list``,
``arena agent sessions get <session-id>``, and ``--session-id`` on
``arena agent generate``.

Sessions from the CLI
"""""""""""""""""""""

``arena agent generate`` keeps a conversation going on its own. The first prompt
starts a session, and later prompts continue it:

.. code-block:: bash

   arena agent generate --prompt "My name is Sam."
   arena agent generate --prompt "What is my name?"

Every prompt runs in a session, because the deployment starts one either way. The
CLI picks the id when it starts a conversation and logs which one it chose. The
id is remembered per deployment, so switching active agents never carries a
session across to one where it means nothing.

Two options change which conversation a prompt runs in:

.. code-block:: bash

   # Start a fresh conversation and continue that one from now on
   arena agent generate --prompt "New topic" --new-session

   # Use a different session for this prompt only, leaving the current one alone
   arena agent generate --prompt "Back to that" --session-id 8ab31d77

To pick up an older conversation, resume it once:

.. code-block:: bash

   arena agent sessions resume

With no ``--session-id``, this lists your sessions and lets you choose one with
the arrow keys. Press Enter to pick it, or Escape to cancel. Pass
``--session-id`` to skip the picker, which is what you need in a script, where
there is no terminal to draw it on:

.. code-block:: bash

   arena agent sessions resume --session-id 1f4c9e02

``arena agent sessions list`` marks the current session in its first column.

When you are done, clear it so the next prompt starts a new conversation. Only
the local pointer is cleared. The session stays on the deployment, so
``arena agent sessions resume`` can pick it up again later:

.. code-block:: bash

   arena agent sessions clear

To remove a conversation for good, delete it. This asks for confirmation, takes
``--yes`` to skip the prompt, and cannot be undone: the messages go from the
deployment, not just from this machine.

.. code-block:: bash

   arena agent sessions delete 8ab31d77

Deleting the session you are currently in clears the local pointer too, so the
next prompt starts a new conversation rather than writing to an id that is gone.

Inference requests carry your own Arena credential, so a deployment that keeps
memory per user knows which conversations are yours. Run ``arena login``, or set
``ARENA_API_KEY`` to a personal access token from Profile then CLI API key.
Without either, the deployment returns a 403 asking you to log in.

.. tutorial::

   :ref:`tutorial_arena_grpo_gsm8k`
      Complete walkthrough of fine-tuning an LLM with GRPO on the GSM8K dataset in Arena.

In-account Snowflake
--------------------

Train on a Snowflake table and serve the model without the data leaving the
account. See :doc:`snowflake`.

.. toctree::
   :maxdepth: 1

   traces
   snowflake
