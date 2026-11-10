Train in a Snowflake account
============================

``arena snowflake submit`` starts a job service on a GPU compute pool in the
customer's account. The service runs the AgileRL trainer image. It reads a
table or query there, trains, copies checkpoints to a stage, and loads metrics
into a table. The table, the checkpoints, the metric values, and the model
stay in the account. RL prompts and completions do not: see below.

Install the extra that talks to Snowflake:

.. code-block:: bash

   pip install 'agilerl-arena[snowflake]'

``arena --help`` works without that extra. The Snowflake commands import it
when they run.

What leaves the account
-----------------------

Checkpoints, metric values, and the model stay in Snowflake. SFT and DPO
training data stays in Snowflake too.

SFT, DPO, and RL all use AgileRL's Pulsar broker, named with ``--pulsar-url``.
Metric names travel to it. For RL, the rollout messages carry the prompts and
the model's completions, so **RL prompts and completions leave the account in
plaintext**. They travel over plain ``pulsar://``, with no
authentication. That is acceptable for a pilot on a development account. It is
not acceptable for production customer data. There is no default broker URL.

The Arena UI does not show live metrics for these runs. Read ``ARENA_METRICS``
instead.

An RL manifest with ``env_type: rollout`` and an ``env_url`` is env-backed: the
environment serves the prompts, so submit needs no ``--source-table`` or
``--source-query``. A source passed anyway is not exported, and submit warns.
SFT, DPO, and dataset-backed RL manifests need one. The environment has to be
reachable by URL from the job. An ``env_url`` host outside the account needs
an external access integration. An SPCS service in the same account needs
none. An ``env_image`` pod is not started.

GPU hosts need an NVIDIA driver of 580 or newer.

Setup
-----

Run :download:`snowflake_setup.sql` as ``ACCOUNTADMIN``. It creates a role, an
X-Small warehouse that suspends after 60 seconds, a small ``GPU_NV_S`` compute
pool, an image repository, a runs stage, a JSON file format, the
``ARENA_METRICS`` table, egress for Hugging Face and for the Pulsar broker, a
resource monitor for the warehouse, a budget for the compute pool, and a
key-pair service user. The role gets ``CREATE SERVICE`` on the schema for job
services, plus ``CREATE MODEL`` on the schema and ``BIND SERVICE ENDPOINT`` on
the account for publishing. Resource monitors do not cover compute pools, and the
budget only notifies, so ``AUTO_SUSPEND_SECS`` is what stops an idle pool.
``MAX_NODES`` has to be at least the largest ``--nodes`` you submit. Grant
the trainer role ``SELECT`` on the source table; the script has those grants
commented out.

Build the trainer image, push it to the image repository, and pass that image
path to ``--image``. The path looks like
``/<database>/<schema>/<repository>/snowflake-job:<tag>``.

Submit
------

The connection comes from ``~/.snowflake/connections.toml``. Password auth is
rejected. Use a key pair (``SNOWFLAKE_JWT``) or a programmatic access token.
Sessions opened from that file set the Snowflake application name to
``AgileRL_Arena``.

Inside the job, the container reads the OAuth token Snowflake mounts at
``/snowflake/session/token`` and the ``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_HOST``,
``SNOWFLAKE_DATABASE``, and ``SNOWFLAKE_SCHEMA`` variables. Queries use the
warehouse passed as ``--warehouse``. The token is read again for each session.

.. code-block:: bash

   arena snowflake submit manifest.yaml \
     --source-table arena.training.prompts \
     --compute-pool arena_gpu_pool \
     --image /arena/training/images/snowflake-job:latest \
     --warehouse arena_wh \
     --runs-stage arena.training.runs \
     --metrics-table arena.training.arena_metrics \
     --pulsar-url pulsar://broker.example:6650 \
     --pulsar-topic-prefix non-persistent://customer/arena \
     --eai arena_hf_eai \
     --eai arena_pulsar_eai \
     --connection arena

.. list-table::
   :header-rows: 1

   * - Flag
     - Meaning
   * - ``--source-table`` / ``--source-query``
     - Table or query the job exports. Pass one, except for an env-backed
       rollout, which needs neither.
   * - ``--compute-pool``
     - GPU compute pool. Its instance family sets the GPUs and memory.
   * - ``--image``
     - Trainer image path in the account.
   * - ``--warehouse``
     - Warehouse for the queries the job runs.
   * - ``--runs-stage``
     - Stage that receives the run directory.
   * - ``--metrics-table``
     - Table that receives metric rows.
   * - ``--pulsar-url``
     - Pulsar broker for this run. No default.
   * - ``--pulsar-topic-prefix``
     - Topic prefix for this run, for example
       ``non-persistent://customer/arena``.
   * - ``--nodes``
     - Job instances, one pool node each. Default 1.
   * - ``--eai``
     - External access integration. Repeat for each one.
   * - ``--checkpoint-steps``
     - Training steps between checkpoints. Unset, the trainer checkpoints
       every ``max_steps // 10`` steps.
   * - ``--reward-file``
     - Python reward function. Required for dataset-backed RL manifests.
   * - ``--connection``
     - Connection name in ``~/.snowflake/connections.toml``. Unset uses the
       connection named ``default``.

The job writes the named columns from the manifest's ``columns`` mapping to
parquet and does not rename them. Add an ``--eai`` for the RL environment host
when ``env_url`` is outside the account.

Column names in the ``columns`` mapping must match Snowflake's names exactly.
Snowflake stores unquoted names in upper case, so a column created as
``prompt`` is mapped as ``PROMPT``. A column created as ``"prompt"`` keeps its
lower case and is mapped as ``prompt``. When a mapped column is missing, the
job fails before training and lists the source's real column names.

A dataset-backed RL manifest needs ``--reward-file reward.py``. The file is
copied to the run directory and passed to the trainer as its reward function.

The command prints a run id (``arena-<UTC yyyymmddHHMMSS>-<6 hex>``) and the
service name (``ARENA_<yyyymmddHHMMSS>_<HEX>``).

``--nodes N`` starts N instances, one pool node each. Every instance starts
Ray. Instance 0 is the head: it exports the source, if any, PUTs the shards to
the run's ``data/`` directory on the stage, which every instance mounts, and
runs the trainer. Every instance uploads the checkpoints written on it. When the
trainer exits, the head uploads its last files, PUTs ``.done`` with the
trainer's exit code, and stops Ray. Once the head's Ray port closes, workers
read ``.done`` from the stage, upload their last files, and exit with that
code. A worker that loses the head without a ``.done`` exits with an error.
The job service then ends and stops consuming credits.

``arena snowflake status SERVICE`` prints one row per container from
``SHOW SERVICE CONTAINERS``, then the job state from the service status:
``state: done`` once it is ``DONE``, which Snowflake sets after every instance
exits 0, ``state: failed`` for ``FAILED`` or ``INTERNAL_ERROR``, and
``state: running`` otherwise. ``arena snowflake logs SERVICE`` prints the
``main`` container log. ``--instance`` selects another instance. The default
is 0. Both take ``--connection``.

Set ``training.checkpoint_export.format`` to ``merged``. The runs stage then
holds:

.. code-block:: text

   @<runs stage>/<run id>/manifest.yaml
   @<runs stage>/<run id>/job.json
   @<runs stage>/<run id>/reward.py
   @<runs stage>/<run id>/data/shard-<n>.parquet
   @<runs stage>/<run id>/checkpoint_<step>/
   @<runs stage>/<run id>/checkpoint_<step>/merged/
   @<runs stage>/<run id>/metrics.jsonl
   @<runs stage>/<run id>/.done

``reward.py`` is there only when ``--reward-file`` is passed. ``data/`` holds
the exported source and is absent for an env-backed rollout. ``merged/`` is uploaded once the trainer writes its
``.complete`` file, and ``.complete`` is uploaded after every other file in
the directory. ``.done`` holds the trainer's exit code. An upload that hits a
network or connector error is retried, and fails the job only at the final
upload. Other upload errors fail the job at once.

Metrics
-------

.. code-block:: sql

   SELECT run_id, seq, time, metrics
   FROM arena.training.arena_metrics
   ORDER BY run_id, seq;

Loading the same metrics file again inserts nothing. Rows match on
``(RUN_ID, SEQ)``.

Publish
-------

``arena snowflake publish`` logs a run's merged checkpoint to the Model
Registry and can serve it on vLLM. The model work runs in the account: the
command starts a job service on the trainer image, and that job reads the
checkpoint through the run directory mount. The weights stay in Snowflake, and
the machine running the command needs only the ``snowflake`` extra.

.. code-block:: bash

   arena snowflake publish "$RUN_ID" \
     --runs-stage arena.training.runs \
     --model-name arena_policy \
     --serve \
     --gpu-count 1 \
     --max-model-len 8192 \
     --connection arena

.. list-table::
   :header-rows: 1

   * - Flag
     - Meaning
   * - ``RUN_ID``
     - Run id printed by ``arena snowflake submit``.
   * - ``--runs-stage``
     - Stage that holds the run directory. Required.
   * - ``--model-name``
     - Model Registry model name. Required.
   * - ``--version``
     - Registry version. Default: the checkpoint directory name, for example
       ``checkpoint_10``.
   * - ``--step``
     - Checkpoint step to publish. Unset, the highest ``checkpoint_<step>/``
       whose ``merged/.complete`` is on the stage.
   * - ``--serve`` / ``--no-serve``
     - Start a vLLM service after logging the model. Default ``--no-serve``.
   * - ``--service-name``
     - Service name with ``--serve``. Default ``<model-name>_<version>``.
   * - ``--compute-pool``
     - Pool for the publish job and the service. Default: the run's pool.
   * - ``--image``
     - Job image path. Default: the run's image.
   * - ``--warehouse``
     - Warehouse for the job's queries. Default: the run's warehouse.
   * - ``--gpu-count``
     - GPUs the service requests. Above 1, also sets vLLM
       ``--tensor-parallel-size``. Default 1.
   * - ``--max-model-len``
     - vLLM ``--max-model-len`` for the service. Unset uses vLLM's default.
   * - ``--database`` / ``--schema``
     - Database and schema for the model and the service. Pass both. Unset,
       the session's current database and schema.
   * - ``--connection``
     - Connection name in ``~/.snowflake/connections.toml``. Unset uses the
       connection named ``default``.

Train with ``training.checkpoint_export.format: merged``. Serving a LoRA
adapter without that merged export is not supported. Names are unquoted
Snowflake identifiers. The command fails before it starts anything when the
model already has that version, or when ``--serve`` is set and the service
exists.

The command starts a job service named ``ARENA_PUBLISH_<yyyymmddHHMMSS>_<HEX>``
and writes its request to ``publish-<job name>.json`` in the run directory, so
publishes of one run can run at the same time. The job:

1. loads ``merged/`` with ``transformers`` in its saved dtype;
2. logs it for Snowpark Container Services with the OpenAI chat signature, the
   run id and checkpoint as the version comment, and metrics from
   ``metrics.jsonl`` as the version metrics;
3. with ``--serve``, creates a vLLM service with a public endpoint on the same
   pool.

The version metrics come from the last row whose ``agent_<n>/steps`` equals the
published checkpoint's step. Null values are left out. A checkpoint with no
matching row is logged without metrics.

The job needs host memory of about the merged weights' size (bf16 is 2 bytes
per parameter, so about 8 GiB for a 4B model) plus 4 GiB. It reserves that
on one node while the service starts, so give the pool a second node
(``MAX_NODES = 2``) unless one node holds both. Pick a GPU that holds the
weights plus the KV cache for ``--max-model-len``. The role needs
``CREATE MODEL`` on the schema and ``BIND SERVICE ENDPOINT`` on the account;
:download:`snowflake_setup.sql` grants both (see `Setup`_).

The service starts with ``min_instances = 0``. It suspends when idle and
starts again on the next request, which waits for the service to come up.

The command prints the job name, then the model, version, service, and
endpoint. When the job fails, the command prints the job's log.
``arena snowflake logs ARENA_PUBLISH_...`` shows it again. The endpoint is a
bare host, or Snowflake's provisioning message until ingress is ready. Call
it with a programmatic access token:

.. code-block:: python

   from openai import OpenAI

   client = OpenAI(
       base_url="https://<endpoint>/v1",
       api_key="unused",
       default_headers={
           "Authorization": 'Snowflake Token="<programmatic access token>"'
       },
   )
   client.chat.completions.create(
       model="unused",
       messages=[{"role": "user", "content": "Hello"}],
   )

GPUs
----

The pool's instance family sets the GPU count and the node memory below.
The container requests that memory minus 0.5 GiB for the stage volume sidecar.
Families this client knows:

.. list-table::
   :header-rows: 1

   * - Instance family
     - GPUs
     - Host memory
   * - ``GPU_NV_S``
     - 1 A10G
     - 28 GiB
   * - ``GPU_NV_M``
     - 4 A10G
     - 178 GiB
   * - ``GPU_NV_L``
     - 8 A100
     - 1112 GiB
   * - ``GPU_L40S_G1_8``
     - 1 L40S
     - 58 GiB
   * - ``GPU_R6K_G1_8``
     - 1 RTX PRO 6000
     - 58 GiB

``arena memory estimate`` knows these cards. The RTX PRO 6000 name is the
product name; ``torch.cuda.get_device_name()`` on Snowflake is not confirmed
yet, so per-device overhead stays on the catalogue default. Sizes below are a
rough LoRA guide, not a measured fit. Estimate the manifest before you submit.

.. list-table::
   :header-rows: 1

   * - GPU
     - Memory
     - Rough LoRA size
   * - A10G
     - 24 GB
     - about 0.8B to 2B
   * - A100
     - 40 GB
     - about 4B
   * - L40S
     - 48 GB
     - about 4B at a short context
   * - RTX PRO 6000 Blackwell Server Edition
     - 96 GB
     - about 9B to 27B at a short context

The service spec opens ports for the manifest's population size and rollout
engines per member. Unset or ``auto`` engines count every GPU in the job.
Ray uses ports 61001 to 61999. Submit rejects a manifest whose trainer ports
would reach them.

.. literalinclude:: snowflake_setup.sql
   :language: sql
