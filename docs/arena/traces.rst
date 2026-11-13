.. _arena_traces_datasets:

Traces datasets
===============

A ``traces`` dataset holds recorded model calls. The same dataset trains
**SFT** and **on-policy distillation** (OPD, including self-distillation) with
no re-upload and no column mapping. Deployed-model traces from continual
learning and hand-made teacher traces use the same shape.

Standard
--------

- **Messages and tools** follow the OpenAI Chat Completions schema, which
  ``POST /v1/chat/completions`` (vLLM), customer harnesses and the inference
  trace recorder already use.
- **Rows** follow Hugging Face TRL's conversational *prompt-completion* layout:
  ``prompt`` (the messages the model saw) plus ``completion`` (the message it
  produced), with ``tools`` beside them.
- OpenEnv defines environment-server actions and observations, not a trace
  format. Record an OpenEnv rollout in this format by logging each model call
  it makes.

Row (one model call)
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 10 42

   * - Column
     - Type
     - Required
     - Meaning
   * - ``prompt``
     - list of messages
     - yes
     - The conversation the model was given, in order, including system
       messages and tool results. It does not end with an assistant message.
   * - ``completion``
     - list of exactly one assistant message
     - yes
     - The reference turn: what the teacher or deployed model replied.
   * - ``tools``
     - list of tool definitions, or null
     - no
     - The tool schemas available on this call.
   * - ``answer``
     - string, or null
     - no
     - A short gold final answer for a rubric. OPD with the default zero
       reward ignores it.
   * - ``call_id``
     - string
     - no
     - Unique row id (ULID from the recorder), for deduplication and score joins.
   * - ``session_id``
     - string, or null
     - no
     - Conversation key. Rows sharing one, ordered by ``segment_index`` then
       ``received_at``, form a conversation.
   * - ``segment_index``
     - int, or null
     - no
     - Segment within a session. A rewritten history starts a new segment.
   * - ``model_version``
     - string, or null
     - no
     - Hash of base model, adapter and chat template that produced ``completion``.
   * - ``checkpoint_id``
     - string, or null
     - no
     - Arena checkpoint or adapter that produced ``completion``.
   * - ``base_model``
     - string, or null
     - no
     - Hub id of that model.
   * - ``received_at``
     - timestamp (UTC), or null
     - no
     - When the call happened, for windowing.
   * - ``scores``
     - list of ``{name, value, source, label, comment}``, or null
     - no
     - Named scores (user feedback, judge id plus rubric hash). ``value`` is a
       number or null; ``label`` and ``comment`` are optional free text.
       Filter only.
   * - ``finish_reason``
     - string, or null
     - no
     - ``stop``, ``length`` or ``tool_calls``. Readers may drop ``length``
       rows (truncated completions).
   * - ``metadata``
     - map of string to string, or null
     - no
     - Free-form. Ignored by training.

Unknown extra columns are allowed and ignored. A hand upload needs only
``prompt`` and ``completion``, plus ``tools`` when there are tools.

**Encoding.** ``prompt``, ``completion`` and ``tools`` hold native JSON in
JSONL. In parquet they are JSON text in a string column, because message
objects vary in keys and a fixed Arrow struct breaks across files. Readers and
the validator accept either. Snapshot ``calls.parquet`` files always use JSON
text, and store ``scores`` as a native Arrow list of structs.

**Duplicate prompts are valid.** A request with ``n > 1`` gives several rows
with the same ``prompt`` and different ``completion``. SFT and OPD use each row.

A conversation may instead be one row with a ``messages`` column (TRL
conversational form), as the continual-learning loop's SFT export writes it.
SFT trains every assistant turn of such a row. A dataset with both forms is
read as ``prompt`` / ``completion``, and its ``messages`` column is ignored. A rollout needs
``prompt`` / ``completion`` rows.

Messages
^^^^^^^^

Exactly the OpenAI Chat Completions message objects:

- ``{"role": "system" | "developer" | "user", "content": <string | list of content parts>}``
- ``{"role": "assistant", "content": <string | null>, "reasoning_content": <string>, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "<JSON string>"}}]}``
- ``{"role": "tool", "tool_call_id": "...", "content": <string>}``

Reasoning is not in the OpenAI schema. Writers use ``reasoning_content``
(vLLM, DeepSeek, and the key chat templates render). Readers also accept
``reasoning`` and ``thinking``. ``function.arguments`` stays a JSON *string*
encoding an object, as OpenAI sends it. A tool result answers a call made
earlier in the same row, and with ``tools`` set every call names one of them.

Tool definitions
^^^^^^^^^^^^^^^^

``{"type": "function", "function": {"name": "...", "description": "...", "parameters": <JSON Schema>}}``

Example
^^^^^^^

.. code-block:: json

   {
     "prompt": [
       {"role": "system", "content": "You are the IT helpdesk agent."},
       {"role": "user", "content": "I can't sign in, MFA keeps failing."},
       {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup_user", "arguments": "{\"email\": \"a@b.com\"}"}}]},
       {"role": "tool", "tool_call_id": "c1", "content": "{\"mfa\": \"totp\", \"locked\": true}"}
     ],
     "completion": [
       {"role": "assistant", "reasoning_content": "Account is locked after failed TOTP attempts...", "content": "Your account was locked after too many attempts. I've sent an unlock link."}
     ],
     "tools": [{"type": "function", "function": {"name": "lookup_user", "description": "Find a user", "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}}],
     "call_id": "01JD...", "session_id": "s-42", "segment_index": 0, "model_version": "m-7f3a"
   }

How each algorithm reads a row
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Algorithm
     - Reads
     - Notes
   * - SFT
     - ``prompt`` + ``completion`` (+ ``tools``), rendered with the student's
       chat template
     - Loss on the completion only; assistant turns inside ``prompt`` are
       context. ``messages`` rows train every assistant turn.
   * - OPD, separate teacher (``teacher.model``)
     - ``prompt`` (+ ``tools``) as the single-turn rollout prompt
     - The student samples one assistant turn and the teacher scores it.
       ``completion`` is unused.
   * - OPD, self-distillation (default teacher)
     - ``prompt`` (+ ``tools``) as the rollout prompt; ``completion`` as the
       teacher's reference
     - ``{reference}`` in ``teacher.template`` is the completion rendered as
       the student would write it: reasoning, content and tool calls.
   * - Reward
     - Zero without a rubric, so the OPD update is pure distillation
     - A ``rubric_file_path`` replaces it, and is required for any other
       rollout algorithm.

Every row is one step deep: a multi-turn tool conversation becomes one row per
assistant turn, each with the history so far as ``prompt``. That gives OPD over
agent traces without a replay environment, so ``max_turns`` stays 1.

Not supported yet: preference pairs (DPO uses the ``preference`` dataset type),
per-turn rewards, token ids and log-probs, and GRPO on traces without a rubric.

Manifest
--------

Set ``environment.dataset_format: traces``. A traces environment takes no
``columns`` and no ``prompt_template``. A rollout takes no ``objective``; a
dataset environment takes ``objective: sft``.

On-policy distillation:

.. code-block:: yaml

   algorithm:
     name: OPD
     teacher: {}            # self-distillation: the reference is each row's completion
   environment:
     env_type: rollout
     dataset_format: traces
     dataset_path: s3://bucket/traces.parquet   # or dataset / hf_dataset_id
     # rubric_file_path: optional; without it every turn scores 0

``teacher.context_key`` defaults to ``reference_trace`` for self-distillation
and is filled from ``completion``. A ``teacher.model`` reads no reference
unless ``context_key`` is set.

SFT:

.. code-block:: yaml

   algorithm:
     name: SFT
   environment:
     env_type: dataset
     objective: sft
     dataset_format: traces
     dataset_path: s3://bucket/traces.parquet

Full examples: ``configs/training/opd/opd_traces.yaml`` and
``configs/training/sft/sft_traces.yaml`` in ``agilerl-ray``.

Uploading
---------

.. code-block:: bash

   arena datasets create helpdesk-traces --category traces --file ./traces.jsonl

The file is JSONL (``.jsonl``, ``.ndjson`` or ``.json`` holding one object per
line), a parquet file or a folder of parquet shards. The CLI (and
:meth:`~agilerl.arena.client.ArenaClient.create_dataset`) checks every row
against this page before uploading and lists bad rows by number, counting from
1. Parquet checks need ``pyarrow``. Validate without uploading with
:func:`agilerl.arena.models.traces.validate_traces_file`.

Storage
-------

A ``traces`` dataset is backed by **either**:

1. an **uploaded file** in this shape, or
2. a **snapshot**: a frozen export from the continual-learning trace query
   service (``manifest.json`` + ``calls.parquet``) at an artifact-store URI,
   which may be the customer's own bucket. Arena stores only the snapshot id,
   URI, row count, model versions present and creator.

Snapshots never change, and their lineage (filter, parent snapshot) is kept in
``manifest.json``. A growing dataset is a series of snapshots, and a run names
one. The column and snapshot manifest field names are defined once, in
``agilerl.arena.traces_format``, which imports only the standard library.
:func:`~agilerl.arena.traces_format.read_manifest` needs only the keys the
platform reads (``REQUIRED_MANIFEST_FIELDS``), ignores unknown keys at any level,
and type-checks the rest.
