Train in a Snowflake account
============================

``arena snowflake submit`` starts a Snowflake ML Job on a GPU compute pool in
the customer's account. The job reads a table or query there, trains, copies
checkpoints to a stage, and loads metrics into a table. The table, the
checkpoints, the metric values, and the model stay in the account.

Install the extra that talks to Snowflake:

.. code-block:: bash

   pip install 'agilerl-arena[snowflake]'

``arena --help`` works without that extra. The Snowflake commands import it
when they run.

What leaves the account
-----------------------

Training data, checkpoints, metric values, and the model stay in Snowflake.

SFT, DPO, and RL all use a Pulsar broker the user names with ``--pulsar-url``.
Rollout messages and metric names travel to that broker over plain
``pulsar://``, with no authentication. That is acceptable for a pilot on a
development account. It is not acceptable for production customer data. There
is no default broker URL.

The Arena UI does not show live metrics for these runs. Read ``ARENA_METRICS``
instead. An RL environment has to be reachable by URL from the job, through an
external access integration. An ``env_image`` pod is not started: Snowflake ML
Jobs do not run Kubernetes pods.

Setup
-----

Run :download:`snowflake_setup.sql` as ``ACCOUNTADMIN``. It creates a role, an
X-Small warehouse that suspends after 60 seconds, a small ``GPU_NV_S`` compute
pool, an image repository, payload and runs stages, a JSON file format, the
``ARENA_METRICS`` table, egress for Hugging Face and for the Pulsar broker, a
resource monitor, and a key-pair service user.

Then build the GPU runtime image, push it to the image repository, and
register it. The image path below is the repository path from
``SHOW IMAGE REPOSITORIES``.

.. code-block:: sql

   CREATE CUSTOM RUNTIME ENVIRONMENT arena_runtime
     IMAGE_PATH = '/arena/training/images/arena-snowflake:latest'
     BASE_IMAGE_TYPE = GPU;

Submit
------

The connection comes from ``~/.snowflake/connections.toml``. Password auth is
rejected. Use a key pair (``SNOWFLAKE_JWT``) or a programmatic access token.
Sessions opened from that file set the Snowflake application name to
``AgileRL_Arena``. A job that is already running keeps the session it logged
in with. The application name is a login field and cannot be changed after
login.

.. code-block:: bash

   arena snowflake submit manifest.yaml \
     --source-table arena.training.prompts \
     --compute-pool arena_gpu_pool \
     --runtime-environment cre@arena_runtime \
     --payload-stage arena.training.payload \
     --runs-stage arena.training.runs \
     --metrics-table arena.training.arena_metrics \
     --pulsar-url pulsar://broker.example:6650 \
     --pulsar-topic-prefix non-persistent://customer/arena \
     --eai arena_hf_eai \
     --eai arena_pulsar_eai \
     --connection arena

``--source-table`` and ``--source-query`` are mutually exclusive. The job
writes the named columns from the manifest's ``columns`` mapping to parquet
and does not rename them. Pass ``--eai`` once per external access integration.
Add one for the RL environment host when the manifest sets ``env_url``.

The command prints a run id (``arena-<UTC yyyymmddHHMMSS>-<6 hex>``) and the
Snowflake ML Job id. ``arena snowflake status JOB_ID`` and
``arena snowflake logs JOB_ID`` read that job.

Set ``training.checkpoint_export.format`` to ``merged``. The runs stage then
holds:

.. code-block:: text

   @<runs stage>/<run id>/manifest.yaml
   @<runs stage>/<run id>/checkpoint_<step>/
   @<runs stage>/<run id>/checkpoint_<step>/merged/
   @<runs stage>/<run id>/metrics.jsonl

``merged/`` is uploaded after its ``.complete`` file exists.

Metrics
-------

.. code-block:: sql

   SELECT run_id, seq, time, metrics
   FROM arena.training.arena_metrics
   ORDER BY run_id, seq;

Loading the same metrics file again inserts nothing. Rows match on
``(RUN_ID, SEQ)``.

GPUs
----

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

.. literalinclude:: snowflake_setup.sql
   :language: sql
