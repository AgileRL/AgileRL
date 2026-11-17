.. _arena_traces_datasets:

Traces datasets
===============

A ``traces`` dataset holds recorded model sessions, one segment per row. The same dataset trains
**SFT** and **on-policy distillation** (OPD, including self-distillation) with
no re-upload and no column mapping. Deployed-model traces from continual
learning and hand-made teacher traces use the same shape.

Standard
--------

- **Messages and tools** follow the OpenAI Chat Completions schema, which
  ``POST /v1/chat/completions`` (vLLM), customer harnesses and the inference
  trace recorder already use.
- **Rows** follow Hugging Face TRL's conversational ``messages`` layout: one
  row is one stretch of a session, with ``tools`` beside it.
- OpenEnv defines environment-server actions and observations, not a trace
  format. Record an OpenEnv rollout in this format by logging each model call
  it makes.

Row (one segment)
-----------------

A row is one append-only stretch of a session: every model call in it only
appended to the history. A rewritten history (compaction, a dropped tool
output, a changed system prompt) or a ``model_version`` change starts a new
row. A 50-turn segment is stored once instead of about 50 times; training
expands it in memory.

.. list-table::
   :header-rows: 1
   :widths: 18 30 10 42

   * - Column
     - Type
     - Required
     - Meaning
   * - ``messages``
     - list of messages
     - yes
     - The segment in order. It has at least one assistant message and does
       not start with one. Each assistant message's tool calls are answered by
       ``tool`` messages before the next assistant message; the final one's
       may be unanswered. Messages after the last assistant message are
       ignored by training.
   * - ``tools``
     - list of tool definitions, or null
     - no
     - The tool schemas available in the segment.
   * - ``call_ids``
     - list of strings or nulls, or null
     - no
     - One per assistant message, in order. A null entry marks that message as
       context, not a training target: the harness inserted it, or the
       previous segment already trained it. At least one entry is non-null.
       Without ``call_ids`` every assistant message is a target.
   * - ``finish_reasons``
     - list of strings, or null
     - no
     - One per assistant message: ``stop``, ``length`` or ``tool_calls``. A
       ``length`` turn (truncated) is not a training target.
   * - ``answer``
     - string, or null
     - no
     - A short gold final answer for a rubric. OPD with the default zero
       reward ignores it.
   * - ``session_id``
     - string, or null
     - no
     - Conversation key. Rows sharing one, ordered by ``segment_index``, form a
       session.
   * - ``segment_index``
     - int, or null
     - no
     - Segment within a session.
   * - ``model_version``
     - string, or null
     - no
     - Hash of base model, adapter and chat template that produced the segment.
   * - ``checkpoint_id``
     - string, or null
     - no
     - Arena checkpoint or adapter that produced the segment.
   * - ``base_model``
     - string, or null
     - no
     - Hub id of that model.
   * - ``received_at``
     - timestamp (UTC), or null
     - no
     - When the segment's first call happened, for windowing.
   * - ``scores``
     - list of ``{name, value, source, label, comment, turn}``, or null
     - no
     - Named scores (user feedback, judge id plus rubric hash). ``value`` is a
       number or null; ``label`` and ``comment`` are optional free text;
       ``turn`` (0-based index into the assistant messages) scores one turn,
       and unset scores the whole segment. Filter only.
   * - ``metadata``
     - map of string to string, or null
     - no
     - Free-form. Ignored by training.

Unknown extra columns are allowed and ignored. A hand upload needs only
``messages``, plus ``tools`` when there are tools.

**Encoding.** ``messages`` and ``tools`` hold native JSON in JSONL. In parquet
they are JSON text in a string column, because message objects vary in keys
and a fixed Arrow struct breaks across files. Readers and the validator accept
either. Snapshot ``segments.parquet`` files always use JSON text, and store
``scores`` as a native Arrow list of structs.

**Duplicate rows are valid.**

Importing prompt/completion pairs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An upload may instead hold one model call per row: ``prompt`` (a list of
messages, or a string as one user message) and ``completion`` (one assistant
message, a list of messages, or a string as one assistant message), with
optional ``call_id`` and ``finish_reason``. A file with both ``prompt`` and
``completion`` is read as pairs, and a ``messages`` column beside them (a raw
recorder dump's request) is ignored. Arena converts each pair to the segment
``messages = prompt + completion`` on import. Only the completion trains: its
assistant messages get the row's ``call_id`` (``pair-<row index>`` when unset)
and the prompt's get null. Rows that are prefixes of a longer row are merged
into it, so a per-call dump becomes the real conversation and no turn trains
twice. The trainer reads only segments: a pair reaching it is rejected.

Messages
^^^^^^^^

Exactly the OpenAI Chat Completions message objects:

- ``{"role": "system" | "developer" | "user", "content": <string | list of content parts>}``
- ``{"role": "assistant", "content": <string | null>, "reasoning_content": <string>, "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "<JSON string>"}}]}``
- ``{"role": "tool", "tool_call_id": "...", "content": <string | null>}``, null
  only after the last assistant message

Reasoning is not in the OpenAI schema. Writers use ``reasoning_content``
(vLLM, DeepSeek, and the key chat templates render). Readers also accept
``reasoning`` and ``thinking``. ``function.arguments`` stays a JSON *string*
encoding an object, as OpenAI sends it. A tool result answers a call made
earlier in the same row, and with ``tools`` set every call names one of them.
Two assistant messages in a row are allowed only when the later one is context
(a null ``call_ids`` entry, such as a message the harness inserted): training
gives each target the messages before it as its prompt, which must not end
with an assistant turn.

Tool definitions
^^^^^^^^^^^^^^^^

``{"type": "function", "function": {"name": "...", "description": "...", "parameters": <JSON Schema>}}``

Example
^^^^^^^

.. code-block:: json

   {
     "messages": [
       {"role": "system", "content": "You are the IT helpdesk agent."},
       {"role": "user", "content": "I can't sign in, MFA keeps failing."},
       {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "lookup_user", "arguments": "{\"email\": \"a@b.com\"}"}}]},
       {"role": "tool", "tool_call_id": "c1", "content": "{\"mfa\": \"totp\", \"locked\": true}"},
       {"role": "assistant", "reasoning_content": "Account is locked after failed TOTP attempts...", "content": "Your account was locked after too many attempts. I've sent an unlock link."}
     ],
     "tools": [{"type": "function", "function": {"name": "lookup_user", "description": "Find a user", "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}}}],
     "call_ids": ["01JC...", "01JD..."], "finish_reasons": ["tool_calls", "stop"],
     "session_id": "s-42", "segment_index": 0, "model_version": "m-7f3a"
   }

How each algorithm reads a row
------------------------------

Each target assistant message of a row (see ``call_ids`` and
``finish_reasons``) trains exactly once. Assistant message *k* is the turn
``prompt = messages[:k]``, ``completion = [messages[k]]``, built in memory.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Algorithm
     - Reads
     - Notes
   * - SFT
     - The whole segment (+ ``tools``), rendered once with the student's chat
       template
     - Loss on every target assistant turn; other messages are context. A
       template that renders earlier turns differently once later ones follow
       (gpt-oss drops their analysis) trains each target turn after its own
       prompt instead, with a warning.
   * - OPD, separate teacher (``teacher.model``)
     - Each target turn's prompt (+ ``tools``) as a single-turn rollout prompt
     - The student samples one assistant turn and the teacher scores it. The
       recorded turn is unused.
   * - OPD, self-distillation (default teacher)
     - Each target turn's prompt (+ ``tools``) as the rollout prompt; the turn
       itself as the teacher's reference
     - ``{reference}`` in ``teacher.template`` is the turn rendered as the
       student would write it: reasoning, content and tool calls.
   * - Reward
     - Zero without a rubric, so the OPD update is pure distillation
     - A ``rubric_file_path`` replaces it, and is required for any other
       rollout algorithm.

Every rollout prompt is one step deep: each target turn of a segment is a
prompt with the history so far. That gives OPD over agent traces without a
replay environment, so ``max_turns`` stays 1.

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
     teacher: {}            # self-distillation: the reference is each target turn
   environment:
     env_type: rollout
     dataset_format: traces
     dataset_path: s3://bucket/traces.parquet   # or dataset / hf_dataset_id
     # rubric_file_path: optional; without it every turn scores 0

``teacher.context_key`` defaults to ``reference_trace`` for self-distillation
and is filled from each target turn. A ``teacher.model`` reads no reference
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
line), a parquet file or a folder of parquet shards, in segment rows or
prompt/completion pairs. The CLI (and
:meth:`~agilerl.arena.client.ArenaClient.create_dataset`) checks every row
against this page, a pair as the segment it becomes, before uploading the
file as it is and lists bad rows by number, counting from
1. Parquet checks need ``pyarrow``. Validate without uploading with
:func:`agilerl.arena.models.traces.validate_traces_file`.

Storage
-------

A ``traces`` dataset is backed by **either**:

1. an **uploaded file** in this shape, or
2. a **snapshot**: a frozen export from the continual-learning trace query
   service (``manifest.json`` + ``segments.parquet``) at an artifact-store
   URI, which may be the customer's own bucket. The manifest's ``row_count``
   counts segments and ``turn_count`` assistant messages. Arena stores only the snapshot id, URI, row count, model versions
   present and creator.

Snapshots never change, and their lineage (filter, parent snapshot) is kept in
``manifest.json``. A growing dataset is a series of snapshots, and a run names
one. The column and snapshot manifest field names are defined once, in
``agilerl.arena.traces_format``, which imports only the standard library.
:func:`~agilerl.arena.traces_format.read_manifest` needs only the keys the
platform reads (``REQUIRED_MANIFEST_FIELDS``), ignores unknown keys at any level,
and type-checks the rest.
