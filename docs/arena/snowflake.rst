Train in a Snowflake account
============================

``arena snowflake`` trains and serves LLMs inside your own Snowflake account,
on Snowpark Container Services. Training data, checkpoints, metrics, and the
served model stay in the account.

Two people are involved:

- An **account admin** runs ``arena snowflake setup`` once. It creates the
  Snowflake objects, copies the trainer image into the account, and checks
  the GPU driver.
- A **data scientist** runs ``arena snowflake join`` once, then submits
  training runs, publishes the result to the Model Registry, and calls it.

Data scientists need a Snowflake login and the ``arena_trainer`` role. They do
not need an Arena login.

What runs where
---------------

One ``arena snowflake submit`` starts three job services in the account:

.. list-table::
   :header-rows: 1

   * - Service
     - Compute pool
     - What it does
   * - Broker
     - ``arena_pulsar_pool`` (default ``CPU_X64_L``)
     - Runs the Pulsar message broker for the run, with a 1000 GiB block
       volume.
   * - Head
     - ``arena_head_pool`` (default ``CPU_X64_SL``)
     - Reads the source table, runs the trainer driver, uploads checkpoints,
       loads metrics, and talks to Arena. It has no GPU.
   * - Workers
     - ``arena_gpu_<family>``, one pool per GPU family
     - One worker per GPU node. Each worker uses the whole node.

The head reads your table or query, writes it to parquet in the run's stage
directory, and starts training. Checkpoints go to the stage
``arena.training.runs`` and metrics go to the table
``arena.training.arena_metrics``. When training ends, the head drops the
broker, and the head and workers exit, so the run stops using credits.

CPU pools have ``MIN_NODES = 1`` and ``MAX_NODES = 4``. GPU pools have
``MIN_NODES = 1``. Every pool suspends after 5 minutes idle.

What leaves the account
-----------------------

Containers in Snowpark Container Services have no egress until an external
access integration allows it. Setup creates two:

.. list-table::
   :header-rows: 1

   * - Integration
     - Hosts (port 443)
     - Used by
   * - ``arena_hf_eai``
     - ``huggingface.co``, ``cas-server.xethub.hf.co``,
       ``cas-server.xethub-eu.hf.co``, ``transfer.xethub.hf.co``,
       ``transfer.xethub-eu.hf.co``, ``us.aws.cdn.hf.co``,
       ``us.gcp.cdn.hf.co``, ``cdn-lfs.hf.co``, ``cdn-lfs-us-1.hf.co``,
       ``cdn-lfs-eu-1.hf.co``
     - Head and workers, to download the base model from Hugging Face.
   * - ``arena_api_eai``
     - The Arena API host only. Default ``arena.agilerl.com``.
     - Head only, for the licence check and usage records below.

The broker has no egress. ``arena snowflake grant --host`` adds a third
integration, ``arena_env_eai``, for an RL environment outside the account.

The head sends Arena exactly this, with ``Authorization: Bearer <org key>``
and ``X-External-User-Id: <external user id>``:

1. ``GET /api/cli/v1/capabilities`` when the head starts. Training stops
   unless Arena says Snowflake training is on for the organisation.
2. ``POST /api/cli/v1/snowflake/runs`` once every node has joined:

   .. code-block:: json

      {
        "runId": "arena-20261111093000-a1b2c3",
        "snowflakeAccount": "MYORG-MYACCOUNT",
        "gpuCount": 4,
        "gpuFamily": "GPU_NV_M",
        "startedAt": "2026-11-11T09:31:12.408000+00:00"
      }

3. ``POST /api/cli/v1/snowflake/runs/<runId>/heartbeat`` with body ``{}``
   every 5 minutes while the trainer runs.
4. ``POST /api/cli/v1/snowflake/runs/<runId>/complete`` when the run ends:

   .. code-block:: json

      {
        "runId": "arena-20261111093000-a1b2c3",
        "gpuCount": 4,
        "gpuFamily": "GPU_NV_M",
        "startedAt": "2026-11-11T09:31:12.408000+00:00",
        "endedAt": "2026-11-11T12:02:45.117000+00:00"
      }

Nothing else is sent: no table rows, prompts, completions, rewards, metric
values, checkpoints, or weights.

Arena answers the open and every heartbeat with a signed run token that lasts
30 minutes. The trainer refuses to start without a valid token and checks it
again at least once a minute. If Arena cannot be reached, the heartbeat keeps
retrying. If Arena refuses to renew (Snowflake training was turned off, or the
run was already closed), training stops when the current token expires. No
token, no training.

The admin's machine also talks to three hosts during setup: the Snowflake
account, the Arena API (to check the org key), and ``public.ecr.aws`` (the
trainer image). The account itself never pulls from a public registry.

Requirements
------------

- A Snowflake account with Snowpark Container Services. Trial accounts cannot
  create external access integrations, so setup stops on them with Snowflake
  error 509009. Add a payment method first.
- A user with ``ACCOUNTADMIN`` to run setup. Setup creates a role, a service
  user, a warehouse, a resource monitor, compute pools, external access
  integrations, and a budget, and grants ``BIND SERVICE ENDPOINT`` on the
  account. Resource monitors and account-level grants need ``ACCOUNTADMIN``,
  so setup checks for that role and stops without it. ``pools``, ``grant``,
  ``invite``, and ``teardown`` need it too.
- A GPU family in the account's region. See `GPUs`_. Families vary by cloud,
  and availability varies by region. Setup lists what the region has when the
  family you ask for is missing.
- Docker running on the admin's machine, with about 30 GB of free disk. The
  trainer image is about 13 GB compressed. Setup pulls it and pushes it into
  the account, so the push takes as long as your upload speed needs for
  13 GB.
- Python 3.10 to 3.13 on every machine that runs ``arena``.
- The Snowflake extra:

  .. code-block:: bash

     pip install 'agilerl-arena[snowflake]'

  ``arena --help`` works without it. The Snowflake commands load it when they
  run.

Get Arena credentials
---------------------

Setup asks for two values from AgileRL:

- An **org key**, which starts with ``arena_org_``. A key scoped to
  ``snowflake`` is enough.
- The **external user id** that goes with it.

Ask AgileRL for both, and ask them to turn on Snowflake training for your
organisation. Setup checks the key and that flag with Arena before it creates
anything. A wrong key or a missing flag stops setup with nothing created.

Setup stores the values as Snowflake secrets ``arena.training.arena_org_key``,
``arena.training.arena_external_user_id``, and ``arena.training.arena_base_url``.
Only the head reads them.

For an Arena host other than ``arena.agilerl.com``, pass it before
``snowflake``: ``arena --base-url https://arena.example.com snowflake setup``.
Setup allows that one host in ``arena_api_eai``.

Admin: set up the account
-------------------------

.. code-block:: bash

   arena snowflake setup

Setup signs in with Snowflake's browser OAuth. The first time, it asks for the
account identifier and your Snowflake login name, saves them as
``[arena-admin]`` in ``~/.snowflake/connections.toml`` (no password), and
opens a browser. Sign in as a user that has ``ACCOUNTADMIN``.

For SSO through ``externalbrowser``, a key pair, or a shell with no browser,
add a connection to ``connections.toml`` and pass ``--connection NAME``.
Password connections are rejected.

Setup then asks for:

- the Arena org key and external user id;
- the warehouse credit cap per month (default 100);
- the compute pool credit budget per month (default 500);
- an email for budget alerts (Enter to skip).

It prints everything it will create and asks once. Nothing is created,
pulled, or written before you answer yes. After that it runs the SQL, writes
the ``[arena]`` connection, pushes the trainer image, and runs ``nvidia-smi``
and the head's Arena licence check in that image on the GPU pool. The licence
check calls Arena from inside Snowflake, so the Arena URL must be reachable
from the public internet. The account is ready when it prints ``The account is
ready.``

When the account has no active event table, the summary warns that ``logs``
cannot read finished jobs and prints the fix:

.. code-block:: sql

   ALTER ACCOUNT SET EVENT_TABLE = SNOWFLAKE.TELEMETRY.EVENTS;

To read the exact SQL first, without connecting to anything:

.. code-block:: bash

   arena snowflake setup --print-sql

It prints the statements for a fresh account, with placeholders for the
secrets and the public key. It assumes ``GPU_NV_S`` unless you pass
``--gpu-family``.

Setup is safe to run again. A re-run keeps the stored Arena secrets, credit
caps, and pools unless you pass new values.

.. list-table::
   :header-rows: 1

   * - Flag
     - Meaning
   * - ``--connection``
     - Admin connection. Unset uses ``[arena-admin]``, or signs in with a
       browser and creates it.
   * - ``--org-key`` / ``--external-user-id``
     - Arena credentials. Unset reads ``ARENA_ORG_KEY`` /
       ``ARENA_EXTERNAL_USER_ID``, then prompts.
   * - ``--gpu-family``
     - GPU pool family. Repeat for each one. Default: the families already
       set up, else the smallest GPU family in the region.
   * - ``--max-gpus``
     - GPUs each GPU pool can hold. Default: the saved cap, else two nodes.
   * - ``--broker-family`` / ``--head-family``
     - CPU families for the broker and head pools. Defaults ``CPU_X64_L`` and
       ``CPU_X64_SL``.
   * - ``--warehouse-credits``
     - Warehouse credit cap per month. Default 100.
   * - ``--gpu-credits``
     - Credit budget per month for all Arena pools. Default 500.
   * - ``--notify-email``
     - Address for budget alerts. It has to be verified in Snowflake.
   * - ``--image-tag``
     - Trainer image version, such as ``32.6.0``. Unset uses the version this
       ``agilerl-arena`` release was built with.
   * - ``--key`` / ``--rotate-key``
     - Use an existing PKCS#8 key for ``arena_job_user``, or replace the key
       on that user with the local one.
   * - ``--print-sql``
     - Print the statements and exit. Creates nothing.
   * - ``--yes``
     - Skip the prompts. Needs ``--connection``. The first run also needs the
       Arena credentials.

What setup creates
~~~~~~~~~~~~~~~~~~

Object names are fixed.

.. list-table::
   :header-rows: 1

   * - Object
     - Name
   * - Role for data scientists and jobs
     - ``arena_trainer``
   * - Service user (key pair, used by ``check``)
     - ``arena_job_user``
   * - Warehouse (X-Small, suspends after 60 s)
     - ``arena_wh``
   * - Resource monitor (notify at 80%, suspend the warehouse at 100%)
     - ``arena_monitor``
   * - Compute pools
     - ``arena_pulsar_pool``, ``arena_head_pool``, ``arena_gpu_<family>``
   * - Budget for the pools (notification only)
     - ``arena.training.arena_gpu_budget``
   * - Database and schema
     - ``arena.training``
   * - Image repository (``arena_trainer`` gets READ)
     - ``arena.training.images``
   * - Runs stage
     - ``arena.training.runs``
   * - Metrics table
     - ``arena.training.arena_metrics``
   * - Arena secrets
     - ``arena_org_key``, ``arena_external_user_id``, ``arena_base_url``
   * - External access integrations
     - ``arena_hf_eai``, ``arena_api_eai``

Resource monitors do not cover compute pools. The pools stop by suspending
when idle; the budget only sends email.

Setup also writes ``@arena.training.runs/_arena/setup.json``: the pools, GPU
caps, integrations, and trainer image. Every ``arena snowflake`` command reads
it, so admin changes reach data scientists on their next command.

On the admin's machine, setup writes ``[arena]`` (user ``arena_job_user``,
key ``arena_job_user.p8`` beside ``connections.toml``, mode ``0600``).

Check the account
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake check

``check`` signs in as ``[arena]``, confirms every object is there, records the
trainer image in ``setup.json``, and runs the GPU driver check again. The
driver must be 580.65.06 or newer. It prints the same event table warning as
setup.

Invite data scientists
~~~~~~~~~~~~~~~~~~~~~~

Create the person's Snowflake user in Snowsight first, then:

.. code-block:: bash

   arena snowflake invite JANE

``invite`` grants ``arena_trainer`` to that user and prints the ``join``
command to send them. It does not create users.

Let jobs read a table or reach a host
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake grant --table ANALYTICS.PUBLIC.SFT_EXAMPLES
   arena snowflake grant --host env.example.com

``--table`` grants ``arena_trainer`` USAGE on the database and schema and
SELECT on the table. ``--host`` lets jobs reach that host on 443 through
``arena_env_eai`` and adds the integration to ``setup.json``, so every later
run gets it. Repeat either flag for more.

Compute pools and GPU caps
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake pools
   arena snowflake pools set GPU_NV_M --max-gpus 16

``pools`` lists each pool, its family, its cap, and the nodes in use.
``pools set`` sets a family's GPU cap. The pool's ``MAX_NODES`` becomes the
node count that holds that many GPUs. A family with no pool yet gets one.
Snowflake limits nodes per family (see `GPUs`_); above that, ask your
Snowflake account team.

The cap has to hold the largest ``--gpus`` anyone submits. A publish that
runs while training is on the pool needs a spare node.

Tear down
~~~~~~~~~

.. code-block:: bash

   arena snowflake teardown

``teardown`` lists what it will drop and asks once. It stops and drops the
Arena compute pools and their services, the three integrations, the
``arena`` database (secrets, stage, checkpoints, metrics, image repository,
and published models in ``arena.training``), ``arena_job_user``,
``arena_trainer``, ``arena_wh``, and ``arena_monitor``. It then removes
``[arena]``, ``[arena-admin]``, and the key setup created.
A database named ``arena`` that setup did not create is left alone.
``--yes`` skips the question.

Data scientist: train a model
-----------------------------

Join the account
~~~~~~~~~~~~~~~~

You need a Snowflake user with ``arena_trainer`` (the admin runs ``invite``)
and either a key pair registered on your user or a programmatic access token.

.. code-block:: bash

   pip install 'agilerl-arena[snowflake]'
   arena snowflake join --account MYORG-MYACCOUNT --user JANE

``join`` asks how you sign in (``token`` or ``key``), writes ``[arena]`` in
``~/.snowflake/connections.toml`` with role ``arena_trainer``, warehouse
``arena_wh``, and schema ``arena.training``, then checks that you can use the
role and read ``setup.json``. ``--token`` or ``--key PATH`` skip the question.
``--yes`` replaces an existing ``[arena]`` without asking. Passwords are
rejected.

You run ``join`` once. Every command below reads ``setup.json`` from the
account, so an image upgrade, ``pools set``, or ``grant --host`` takes effect
without joining again. The commands use ``[arena]`` when ``connections.toml``
has it, else your default Snowflake connection, unless you pass
``--connection``.

A sample table
~~~~~~~~~~~~~~

Any table works. This one is enough to try SFT. The admin, or anyone who owns
a database, creates it and grants it to Arena jobs:

.. code-block:: sql

   CREATE TABLE analytics.public.sft_examples (prompt STRING, response STRING);
   INSERT INTO analytics.public.sft_examples VALUES
     ('What is the capital of France?', 'Paris.'),
     ('Translate "good morning" to Spanish.', 'Buenos días.');

.. code-block:: bash

   arena snowflake grant --table analytics.public.sft_examples

Snowflake stores unquoted names in upper case. The table above has columns
``PROMPT`` and ``RESPONSE``, and the manifest must use those exact names. A
column created as ``"prompt"`` (quoted) keeps its lower case. When a mapped
column is missing, the job fails before training and lists the real column
names.

SFT manifest
~~~~~~~~~~~~

.. code-block:: yaml

   algorithm:
     name: SFT
   environment:
     env_type: dataset
     objective: sft
     columns:
       prompt: PROMPT
       response: RESPONSE
   network:
     pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
     max_context_length: 2048
   training:
     max_steps: 200

``columns`` maps the names the trainer expects (left) to your columns
(right). The job reads only the mapped columns. The rows come from
``--source-table``, so the manifest has no ``dataset``.

Snowflake submit exports merged weights by default, which ``publish`` needs.
Submit refuses ``training.checkpoint_export`` with ``format: adapter`` or
``trigger: on_demand``, since neither writes ``merged/`` during training.

.. code-block:: bash

   arena snowflake submit sft.yaml --gpus 1 \
     --source-table analytics.public.sft_examples

GRPO manifest
~~~~~~~~~~~~~

GRPO over table rows needs a prompt template in the manifest and
``--reward-file`` on submit. With a table
``analytics.public.math_questions`` that has ``QUESTION`` and ``ANSWER``
columns (granted with ``grant --table``):

.. code-block:: yaml

   algorithm:
     name: GRPO
     batch_size: 8
     group_size: 4
   environment:
     env_type: rollout
     columns:
       question: QUESTION
       answer: ANSWER
     prompt_template:
       system_0: Think step by step, then give the final number inside <answer></answer>.
       user_1: "{question}"
     max_reward: 1.0
   network:
     pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
     max_context_length: 2048
   training:
     max_steps: 200

``reward.py`` defines ``reward_fn``, called with each completion and the row's
``answer`` and ``question``:

.. code-block:: python

   import re


   def reward_fn(completion: str, answer: str, question: str) -> float:
       match = re.search(r"<answer>\s*(-?[\d.]+)\s*</answer>", completion)
       return 1.0 if match and match.group(1) == str(answer).strip() else 0.0

.. code-block:: bash

   arena snowflake submit grpo.yaml --gpus 4 \
     --source-table analytics.public.math_questions \
     --reward-file reward.py

RL against an environment
~~~~~~~~~~~~~~~~~~~~~~~~~

A rollout manifest with ``env_url`` gets its prompts from the environment, so
it needs no ``--source-table``. The job has to reach the URL. A Snowpark
Container Services service in the same account needs nothing extra. A host
outside the account needs ``arena snowflake grant --host`` from the admin, or
``--eai`` with an integration that allows it. ``env_image`` is not supported.

Submit
~~~~~~

``submit`` prints a run id such as ``arena-20261111093000-a1b2c3`` and the
three service names (``ARENA_RUN_20261111093000_A1B2C3_BROKER``, ``_HEAD``,
and ``_WORKERS``).

.. list-table::
   :header-rows: 1

   * - Flag
     - Meaning
   * - ``--gpus``
     - GPUs for the run. Required. Workers are this count divided by the
       family's GPUs per node.
   * - ``--gpu-family``
     - GPU family. Unset: of the families whose GPUs per node divide
       ``--gpus`` and that the memory estimator says can hold the manifest,
       the one with the least GPU memory, then the fewest nodes.
   * - ``--source-table``
     - ``DATABASE.SCHEMA.TABLE`` the job reads. Must be fully qualified. The
       manifest needs no ``dataset``.
   * - ``--source-query``
     - A query instead of a table. Pass at most one of the two.
   * - ``--reward-file``
     - Python reward file. Required for GRPO and other RL over table rows.
       The manifest needs no ``rubric_file_path``.
   * - ``--resume``
     - Start from the newest finished checkpoint of an earlier run.
   * - ``--checkpoint-steps``
     - Training steps between checkpoints. Unset uses ``max_steps // 10``.
   * - ``--eai``
     - Extra external access integration. Repeat for each one. Added to the
       integrations in ``setup.json``.
   * - ``--image``, ``--warehouse``, ``--runs-stage``, ``--metrics-table``
     - Override ``setup.json``. Rarely needed.
   * - ``--connection``
     - Connection name. Default ``arena`` when ``connections.toml`` has it.

If no enabled family divides ``--gpus``, submit lists the counts that work.
If the workers need more nodes than the pool's ``MAX_NODES``, submit stops
before it starts anything and prints the ``pools set`` command an admin can
run.

Status, logs, and cancel
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake status RUN_ID
   arena snowflake logs RUN_ID
   arena snowflake logs RUN_ID --role worker --instance 1
   arena snowflake cancel RUN_ID

``status`` prints the state of the broker, head, and workers, the containers
of each, and one overall state: ``running``, ``succeeded``, ``failed``,
``cancelling``, ``cancelled``, or ``gone``.

``logs`` prints the ``main`` container log of one role (``broker``, ``head``,
or ``worker``; default ``head``) and instance (default 0). Logs of a finished
job come from the account's event table, up to the last 1000 lines from the
past 30 days. A dropped service has no logs to read. The head drops the
broker when the run ends, and ``cancel`` drops all three services.

``cancel`` drops the three services. As the head stops, it tries to record
the end of the run with Arena. If that does not land, Arena closes the run at
its last heartbeat (see `Billing`_).

Resume a run
~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake submit grpo.yaml --gpus 4 \
     --source-table analytics.public.math_questions \
     --reward-file reward.py \
     --resume arena-20261111093000-a1b2c3

The new run gets its own run id and starts from the newest checkpoint of the
old run that finished uploading. Snowflake's ``COPY FILES`` copies the
checkpoint, and it takes files up to 5 GB each. Submit stops when a file of
that checkpoint is larger.

Metrics
~~~~~~~

.. code-block:: sql

   SELECT seq, time, metrics
   FROM arena.training.arena_metrics
   WHERE run_id = 'arena-20261111093000-a1b2c3'
   ORDER BY seq;

``metrics`` is a ``VARIANT`` with one key per metric. Rows are keyed on
``(RUN_ID, SEQ)``, so loading the same file twice adds nothing. The Arena web
app has no live charts for these runs.

Run directory
~~~~~~~~~~~~~

Each run has a directory on the stage:

.. code-block:: text

   @arena.training.runs/<run id>/manifest.yaml
   @arena.training.runs/<run id>/job.json
   @arena.training.runs/<run id>/reward.py
   @arena.training.runs/<run id>/data/shard-<n>.parquet
   @arena.training.runs/<run id>/checkpoint_<step>/
   @arena.training.runs/<run id>/checkpoint_<step>/merged/
   @arena.training.runs/<run id>/metrics.jsonl
   @arena.training.runs/<run id>/.done

``reward.py`` is there only with ``--reward-file``, and ``data/`` only when
the run reads a table or query. ``merged/`` is uploaded once the trainer
finishes writing it, with its ``.complete`` file last. ``.done`` holds the
trainer's exit code.

Maintenance windows
-------------------

Snowflake maintains Snowpark Container Services nodes on weekends: Saturday
20:00 to Sunday 08:00 and Sunday 20:00 to Monday 08:00, in the region's local
time. Some accounts also have a daily window from 23:00. Snowflake can cancel
a job during a window and does not restart it. Start long runs so they finish
before a window, or resume a cancelled run with ``submit --resume``.

Billing
-------

Arena bills GPU hours: the number of GPUs in the run times the time from the
first GPU worker container start to the end of the run. If the end of a run
never reaches Arena, Arena closes it at its last heartbeat.

Snowflake bills your account for the compute pools, the warehouse, and
storage as usual. Arena does not see or resell that.

Publish and serve
-----------------

.. code-block:: bash

   arena snowflake publish RUN_ID --model-name support_bot --serve

``publish`` starts a job in the account that loads the newest complete
``merged/`` checkpoint of the run, logs it to the Model Registry in
``arena.training``, and with ``--serve`` starts a vLLM inference service with
a public endpoint. The weights never leave Snowflake. The command waits for
the job, which can take 20 minutes or more while the GPU service starts, then
prints the model, version, service, and endpoint.

.. list-table::
   :header-rows: 1

   * - Flag
     - Meaning
   * - ``--model-name``
     - Model Registry name. Required.
   * - ``--serve`` / ``--no-serve``
     - Start a vLLM service after logging the model. Default ``--no-serve``.
   * - ``--version``
     - Registry version. Default: the checkpoint directory, such as
       ``checkpoint_200``.
   * - ``--step``
     - Checkpoint step. Default: the newest with a complete ``merged/``.
   * - ``--service-name``
     - Service name. Default ``<model-name>_<version>``.
   * - ``--gpu-count``
     - GPUs the service uses. Above 1, also sets vLLM tensor parallelism.
       Default 1.
   * - ``--max-model-len``
     - vLLM ``--max-model-len``. Unset uses vLLM's default.
   * - ``--compute-pool``
     - Pool for the publish job and the service. Default: the account's GPU
       pool when there is one GPU family, else the run's pool.
   * - ``--database`` / ``--schema``
     - Where the model and service go. Pass both. Default
       ``arena.training``.
   * - ``--runs-stage``, ``--image``, ``--warehouse``
     - Override ``setup.json``. Rarely needed.
   * - ``--connection``
     - Connection name. Default ``arena`` when ``connections.toml`` has it.

The trainer writes ``merged/`` when the ``checkpoint_export`` trigger fires,
by default at the final save. A run stopped before then has no ``merged/``
and cannot be published, and ``--step`` takes only a checkpoint with a
complete ``merged/``. ``publish`` stops before it starts anything when the
version or the service already exists. The job needs host memory of about the weights' size plus
4 GiB on one node (bf16 is 2 bytes per parameter, so about 12 GiB for a 4B
model) and holds it while the service starts, so the pool needs a second node
unless one node fits both.

Get the endpoint
~~~~~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake endpoint support_bot

``endpoint`` prints the host name of the model's service. Pass ``--version``
when more than one version is served. Until ingress is ready, it prints
Snowflake's provisioning message instead.

Call the model
~~~~~~~~~~~~~~

The endpoint speaks the OpenAI chat API. Authenticate with a Snowflake
programmatic access token (PAT):

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(
       base_url="https://<endpoint>/v1",
       api_key="unused",
       default_headers={"Authorization": 'Snowflake Token="<PAT>"'},
   )
   reply = client.chat.completions.create(
       model="unused",
       messages=[{"role": "user", "content": "Hello"}],
   )
   print(reply.choices[0].message.content)

The PAT's user needs a role that can use the endpoint. ``arena_trainer`` owns
the service, so a PAT restricted to it works:

.. code-block:: sql

   ALTER USER jane ADD PROGRAMMATIC ACCESS TOKEN support_bot_client
     ROLE_RESTRICTION = 'ARENA_TRAINER';

To let another role call it:

.. code-block:: sql

   GRANT SERVICE ROLE arena.training.support_bot_checkpoint_200!ALL_ENDPOINTS_USAGE
     TO ROLE app_role;

By default Snowflake accepts a PAT only from a user that has a network
policy, and the policy has to allow the address you call from. See
`Snowflake's PAT docs
<https://docs.snowflake.com/en/user-guide/programmatic-access-tokens>`_.

A request with a missing or rejected token, or from a role without access,
returns 404, not 401 or 403. Check the token, its role, and the network
policy before you suspect the URL.

The service starts with ``min_instances = 0``. After about 30 minutes without
requests it suspends, and the next request waits while it starts again.

Stop serving
~~~~~~~~~~~~

.. code-block:: bash

   arena snowflake unpublish support_bot
   arena snowflake unpublish support_bot --version checkpoint_200 --drop-version

``unpublish`` drops the version's inference service. ``--drop-version`` also
drops the version, and drops the model when it was the only version.

Upgrade
-------

Install the new ``agilerl-arena[snowflake]`` and run setup again, then check:

.. code-block:: bash

   pip install -U 'agilerl-arena[snowflake]'
   arena snowflake setup
   arena snowflake check

Setup pushes the trainer image that matches the installed release and saves
its path in ``setup.json``. ``--image-tag 32.6.0`` picks a release instead.
The next submit uses the new image; data scientists do not join again. Runs
that are already going keep their image.

GPUs
----

Arena sizes these families. Snowflake offers different families on each
cloud, and not every region has every family. The default node limit is
Snowflake's; "on request" means ask your Snowflake account team.

.. list-table::
   :header-rows: 1

   * - Cloud
     - Family
     - GPUs per node
     - GPU memory
     - Default node limit
   * - AWS
     - ``GPU_NV_S``
     - 1 A10G
     - 24 GB
     - 150
   * - AWS
     - ``GPU_NV_M``
     - 4 A10G
     - 24 GB
     - 10
   * - AWS
     - ``GPU_NV_L``
     - 8 A100
     - 40 GB
     - 10
   * - AWS
     - ``GPU_L40S_G1_8``, ``GPU_L40S_G1_16``
     - 1 L40S
     - 48 GB
     - 5
   * - AWS
     - ``GPU_L40S_G1_48`` / ``GPU_L40S_G1_192``
     - 4 / 8 L40S
     - 48 GB
     - on request
   * - AWS
     - ``GPU_R6K_G1_8``, ``GPU_R6K_G1_16``, ``GPU_R6K_G1_32``
     - 1 RTX PRO 6000
     - 96 GB
     - 5
   * - AWS
     - ``GPU_R6K_G1_48`` / ``GPU_R6K_G1_96`` / ``GPU_R6K_G1_192``
     - 2 / 4 / 8 RTX PRO 6000
     - 96 GB
     - on request
   * - Azure
     - ``GPU_NV_XS``
     - 1 T4
     - 16 GB
     - 10
   * - Azure
     - ``GPU_NV_SM``
     - 1 A10
     - 24 GB
     - 10
   * - Azure
     - ``GPU_NV_2M``
     - 2 A10
     - 24 GB
     - 5
   * - Azure
     - ``GPU_NV_3M`` / ``GPU_NV_SL``
     - 2 / 4 A100
     - 80 GB
     - on request
   * - Google Cloud
     - ``GPU_GCP_NV_L4_1_24G`` / ``GPU_GCP_NV_L4_4_24G``
     - 1 / 4 L4
     - 24 GB
     - 10
   * - Google Cloud
     - ``GPU_A100_G1_12`` / ``GPU_A100_G1_48`` / ``GPU_GCP_NV_A100_8_40G``
     - 1 / 4 / 8 A100
     - 40 GB
     - on request

Submit runs the manifest through the memory estimator for each family and
stops when the GPUs cannot hold it. ``arena memory estimate`` runs the same
check on your machine.

Common errors
-------------

``Ask AgileRL for a Snowflake org key (it starts with arena_org_).``
   Setup got a value that is not an Arena org key. See `Get Arena
   credentials`_.

``This account has no Arena setup at @arena.training.runs/_arena/setup.json.``
   ``submit``, ``join``, ``check``, ``pools``, or ``grant --host`` found no
   ``setup.json``.
   An account admin runs ``arena snowflake setup``.

``This account has no setup.json. Pass --runs-stage.``
   ``publish`` found no ``setup.json``. Pass the stage that holds the run.

Limits
------

- The broker's block volume uses a Snowflake preview feature for job
  services. An account can hold at most 100 block volumes.
- Logs of a finished job come from the account's active event table. Without
  one, or once the service is dropped, they cannot be read. Setup does not set
  one; ``ALTER ACCOUNT SET EVENT_TABLE = SNOWFLAKE.TELEMETRY.EVENTS`` does.
- Ray uses ports 61001 to 61999 in the job. Submit rejects a manifest whose
  population size and rollout engines would need trainer ports in that range.
- A ``COPY FILES`` copy for ``--resume`` takes files up to 5 GB each.
