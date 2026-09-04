.. _llm_environments:

Environments (OpenEnv)
======================

Every LLM-training environment in AgileRL is reached the same way: **text in, text
out**, through the `OpenEnv <https://github.com/meta-pytorch/OpenEnv>`_ API
(installed with the ``[llm]`` extra). A ``RolloutHarness`` drives the env in the
training process, or over a URL.

Two types
---------

``env_type`` is required. It is not inferred from the algorithm.

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - ``env_type``
     - What it means
   * - ``rollout``
     - The model generates and the environment scores it. A single-turn scored
       task is ``max_turns: 1``, not a third type.
   * - ``dataset``
     - Supervised, no generation. Requires ``objective: sft`` or
       ``objective: preference``.

The text interface
------------------

An environment is any object with two methods:

.. code-block:: python

   def reset(self, seed: int | None = None) -> tuple[str, dict]:
       # returns (prompt_text, info)
       ...

   def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
       # returns (observation_text, reward, terminated, truncated, info)
       ...

``reset`` returns the first prompt; ``step`` receives the model's decoded text and
returns the next observation, a reward, and whether the episode has ended. Optional
extras: a ``tools`` attribute, and a ``dataset_size`` property with extra ``reset``
kwargs (covered below).

Each concurrent rollout gets its own env instance, created once and reused for the
whole run. Over a URL, **one server fronts the whole batch**: each rollout opens its
own WebSocket session and the server builds a fresh env for that session.

Writing one in Python
---------------------

Pass the env to :meth:`RolloutHarness.local <agilerl.llm_envs.RolloutHarness.local>`.
It runs **in the training process, with no HTTP**:

.. code-block:: python

   from agilerl.llm_envs import RolloutHarness


   class GuessEnv:
       """Guess-the-number: the model proposes a number; the env scores it."""

       def __init__(self, target: int = 7, max_turns: int = 3) -> None:
           self.target, self.max_turns, self._turn = target, max_turns, 0

       def reset(self, seed: int | None = None) -> tuple[str, dict]:
           self._turn = 0
           return "Guess a number between 1 and 10.", {}

       def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
           self._turn += 1
           guess = int("".join(c for c in action if c.isdigit()) or 0)
           correct = guess == self.target
           done = correct or self._turn >= self.max_turns
           feedback = "Correct!" if correct else ("Too high." if guess > self.target else "Too low.")
           return feedback, float(correct), done, False, {}


   env = RolloutHarness.local(GuessEnv(), tokenizer, max_turns=3)

``max_turns`` bounds the episode; the env can still finish earlier by returning
``terminated=True``. For a batch, wrap the same call so each slot gets its own
instance:

.. code-block:: python

   env_factory = lambda: RolloutHarness.local(GuessEnv(), tokenizer, max_turns=3)

Naming one from a manifest
--------------------------

A rollout names **exactly one source**. That is what decides where it runs.

.. list-table::
   :widths: 28 36 36
   :header-rows: 1

   * - The manifest names
     - The environment runs
     - Started by
   * - ``dataset`` plus a reward / rubric file
     - in the training process
     - the job
   * - ``entrypoint``
     - in the training process
     - the job
   * - ``env_url``
     - wherever it already is
     - someone else, beforehand

Dataset rows plus a reward file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model answers each labelled row once; a Python file scores it. This is still
``env_type: rollout`` (the model generates). Full example:
``configs/training/llm_finetuning/grpo.yaml``.

.. code-block:: yaml

   environment:
       env_type: rollout
       dataset: Jiayi-Pan/Countdown-Tasks-3to4
       columns:
           nums: question
           target: answer
       rubric_file_path: reward.py
       rubric_name: RUBRIC
       prompt_template:
           system_0: You are a helpful assistant.
           user_1: Using each number in {question}, make an equation that equals {answer}.
       max_reward: 2.0
       train_test_split: 0.8

``dataset`` is a HuggingFace id or a Parquet path. ``rubric_file_path`` also
accepts the alias ``reward_file_path``. The reward file sits next to the
manifest (see ``docs/_static/examples/gsm8k-grpo/reward.py``).

A Python entrypoint
~~~~~~~~~~~~~~~~~~~

``entrypoint`` is ``module:attr`` or ``path/to/file.py:attr``, and ``env_config``
is its keyword arguments — so a library factory works unchanged:

.. code-block:: yaml

   environment:
       env_type: rollout
       entrypoint: gem:make
       env_config:
           env_id: game:GuessTheNumber-v0-easy
       max_turns: 50
       max_reward: 1.0

The same ``GuessEnv`` from above, as a file next to the run:

.. code-block:: yaml

   environment:
       env_type: rollout
       entrypoint: guess_env.py:GuessEnv
       env_config:
           target: 7
       max_turns: 3

That env runs **in the training process**. Leaving ``max_turns`` out is allowed
here: it is probed off one throwaway env and cached. A remote env cannot be
probed, which is why ``env_url`` requires it.

Put a system prompt in ``env_config``; ``RolloutHarness`` renders it as a leading
``system`` message:

.. code-block:: yaml

   env_config:
       env_id: game:GuessTheNumber-v0-easy
       system_prompt: You are playing a guessing game. Reply with one number.

An already-running server
~~~~~~~~~~~~~~~~~~~~~~~~~

If the environment is already an OpenEnv server, point at its URL. Nothing is
started locally. ``max_turns`` is required. To stand the server up from an
entrypoint, use :func:`~agilerl.llm_envs.openenv_server.resolve_env` (same
``entrypoint`` / ``env_config`` as a local run):

.. code-block:: python

   from agilerl.llm_envs.openenv_server import resolve_env

   url, server = resolve_env(
       "gem:make",
       env_config={"env_id": "code:Taco8k", "sandbox_type": "bwrap"},
       port=8000,
       max_concurrent_envs=16,
   )

.. code-block:: yaml

   environment:
       env_type: rollout
       env_url: http://127.0.0.1:8000
       max_turns: 1
       request_timeout_s: 600

Each rollout opens its own ``/ws`` session. The server's ``max_concurrent_envs``
must cover ``batch_size * group_size`` sessions **plus one** for the evaluation
env, which is built lazily at the first evaluation. ``request_timeout_s`` bounds
each message at 300 seconds by default (``0`` disables the bound).

Walkthrough: :ref:`remote_env_server_tutorial` — host a coding env on one
machine and train on a GPU box.

Supervised (SFT / DPO)
~~~~~~~~~~~~~~~~~~~~~~~~~~

No generation. ``objective`` picks the columns and the loss:

.. code-block:: yaml

   environment:
       env_type: dataset
       objective: sft
       dataset: HumanLLMs/Human-Like-DPO-Dataset
       response_column: chosen
       train_test_split: 0.9

For DPO, ``objective: preference``.

.. _env_packages:

Declaring what the environment needs
------------------------------------

If the entrypoint's package is not installed, add ``env_packages``:

.. code-block:: yaml

   environment:
       env_type: rollout
       entrypoint: gem:make
       env_config:
           env_id: game:GuessTheNumber-v0-easy
       env_packages:
           uv: [gem-llm==1.0.0]
       max_turns: 50

.. warning::

   ``env_packages`` **installs into the interpreter training is running in**.
   The install runs only when the entrypoint cannot already be imported. There
   is no prompt and no dry-run. Install the package yourself and leave the
   field out if you would rather manage it.

   Both ``uv:`` and ``pip:`` are installed **with uv**, so ``uv`` has to be on
   ``PATH``. Package names only; anything starting with ``-`` is rejected.

   If the packages cannot resolve alongside the trainer's own dependencies, the
   install fails rather than half-upgrading the environment. Host it with
   :func:`~agilerl.llm_envs.openenv_server.resolve_env` and point ``env_url``
   at the URL it returns.

Manifest fields
---------------

These are the names this package reads:

.. list-table::
   :widths: 28 14 58
   :header-rows: 1

   * - Field
     - Default
     - Meaning
   * - ``env_type``
     - *required*
     - ``rollout`` or ``dataset``.
   * - ``dataset``
     - ``None``
     - HuggingFace id or Parquet path. Alias: ``name``.
   * - ``columns``
     - ``None``
     - Rename source columns (e.g. ``nums: question``).
   * - ``rubric_file_path``
     - ``None``
     - Reward / rubric file for a dataset-backed rollout. Alias:
       ``reward_file_path``.
   * - ``rubric_name``
     - ``None``
     - Symbol in that file. Alias: ``reward_fn_name``.
   * - ``prompt_template``
     - ``None``
     - Chat-template pieces rendered into the prompt on reset.
   * - ``chat_template_kwargs``
     - ``{}``
     - Extra kwargs for ``apply_chat_template``.
   * - ``train_test_split``
     - ``0.9``
     - Fraction of the dataset used for training.
   * - ``max_reward``
     - ``None``
     - Maximum achievable reward, used for accuracy reporting.
   * - ``entrypoint``
     - ``None``
     - ``module:attr`` or ``path/to/file.py:attr`` returning a text env.
   * - ``env_config``
     - ``None``
     - Keyword arguments for the entrypoint. ``system_prompt`` is set on the
       env after construction, not passed into the constructor.
   * - ``env_packages``
     - ``None``
     - ``{uv: [...]}`` or ``{pip: [...]}`` to install before import.
   * - ``max_turns``
     - probed
     - Turn budget (``ge=1``). Required with ``env_url``; probed for
       ``entrypoint``.
   * - ``env_url``
     - ``None``
     - URL of an already-running OpenEnv server.
   * - ``mcp_tool``
     - ``None``
     - MCP tool name; only applies with ``env_url``.
   * - ``action_field``
     - ``message``
     - Field the env puts the model's text in (``message``, ``code``, …).
   * - ``observation_field``
     - ``None``
     - Field the observation's text lives in.
   * - ``observation_processor``
     - ``None``
     - ``module:fn`` that renders a payload to prompt text.
   * - ``request_timeout_s``
     - ``300``
     - Per-message bound on a ``/ws`` session. ``0`` disables it.
   * - ``objective``
     - ``None``
     - ``sft`` or ``preference``; required when ``env_type: dataset``.
   * - ``response_column``
     - ``response``
     - SFT completion column.

Evaluation and grouped rollouts
-------------------------------

A dataset-backed env exposes ``dataset_size`` and accepts two extra kwargs on
``reset``:

.. code-block:: python

   def reset(self, seed=None, *, row_index: int | None = None,
             evaluation: bool | None = None) -> tuple[str, dict]:
       # row_index selects a row; evaluation=True serves the held-out split
       ...

The collector draws one row per batch item and reuses it across that item's
GRPO group, so grouped-advantage compares like with like. At evaluation it
passes ``evaluation=True`` and opens one extra session (the ``+ 1`` in
``max_concurrent_envs``). The env must be reproducible from
``(seed, row_index)``.

Advantage granularity
---------------------

On GRPO (and CISPO / GSPO), ``algorithm.advantage_granularity`` is
``auto | trajectory | turn``, default ``auto``. ``trajectory`` is one
group-relative scalar per completion; ``turn`` normalises each turn's reward
within the group; ``auto`` picks ``turn`` when the batch has per-turn rewards
and more than one turn. ``token`` is not a GRPO value. The old
``action_granularity`` spelling is no longer accepted; rename it in existing
manifests.

LLMPPO and LLMREINFORCE take ``advantage_granularity`` over
``{turn, token, auto}``.

Observations that are not plain text
------------------------------------

Third-party OpenEnv environments do not agree on where their observation's text
lives — a coding env returns ``{"stdout": "...", "stderr": "", "exit_code": 0}``,
BrowserGym ``{"text": "..."}``, OpenSpiel a numeric ``{"info_state": [...]}``.
The harness renders the payload via an **observation processor**:

* ``observation_field`` names the field, when a lookup is enough.
* ``observation_processor`` is a ``module:fn`` / ``path/to/file.py:fn`` that
  turns the payload into prompt text.

.. code-block:: yaml

   environment:
       env_type: rollout
       env_url: https://my-openspiel-env.example.com
       max_turns: 12
       observation_processor: render_board.py:render

.. code-block:: python

   # render_board.py
   def render(payload: dict) -> str:
       return "Board state: " + ", ".join(str(v) for v in payload["info_state"])

Keep the processor a pure function of the payload; it runs on the collector's
I/O threads.

.. note::

   ``observation_field`` and ``observation_processor`` are read by local
   ``agilerl`` training. On `Arena <https://arena.agilerl.com>`_, have the
   environment server render text itself.

Tools
-----

An environment advertises tool schemas with a ``tools`` attribute (a list of
JSON schemas). AgileRL renders them into the chat template. There is no
separate tools config.

For an MCP tool instead of the plain text interface, pass
``mcp_tool="<tool_name>"`` on ``RolloutHarness`` or in the manifest (with
``env_url``).

Lifecycle
---------

* The environment is built once per rollout and reused for the whole run.
* ``close()`` (if the env has one) is called **exactly once**, on teardown —
  not per step.
* Closing a ``RolloutHarness`` ends its WebSocket session or closes the
  in-process env.

Lower-level pieces
------------------

* :class:`InProcessEnvClient <agilerl.llm_envs.openenv.InProcessEnvClient>` —
  drives a local env's ``reset`` / ``step`` directly. What
  :meth:`RolloutHarness.local` uses.
* :class:`RemoteEnvClient <agilerl.llm_envs.openenv.RemoteEnvClient>` — one
  ``/ws`` session against a server, which backs each session with its own env.
* :func:`resolve_env <agilerl.llm_envs.openenv_server.resolve_env>` — start a
  server from the same ``entrypoint`` / ``env_config`` a local manifest would
  use. Returns ``(url, server)``. ``max_concurrent_envs`` gives each session
  its own env instance.

.. code-block:: python

   from agilerl.llm_envs.openenv import RemoteEnvClient
   from agilerl.llm_envs.openenv_server import resolve_env

   url, server = resolve_env(
       "guess_env.py:GuessEnv",
       env_config={"target": 7},
       max_concurrent_envs=8,
   )
   client = RemoteEnvClient(url)
   payload, _ = client.reset()
   payload, reward, terminated, truncated, info = client.step("5")
   client.close()
   server.stop()

:meth:`RolloutHarness.from_spec <agilerl.llm_envs.RolloutHarness.from_spec>`
picks the trainer-side transport: a URL → WebSocket client; a
``package.module:EnvClass`` entrypoint → in-process. Use ``resolve_env`` when
you want that entrypoint hosted as a URL instead.

Training on Arena
-----------------

This package's manifest names a dataset, an ``entrypoint``, or an ``env_url``.
On `Arena <https://arena.agilerl.com>`_, a container or an external URL is an
environment version you pick in the UI, not a field this spec validates.
Unknown extra keys on ``environment`` are ignored.

.. _llm_env_migration:

Migrating from the pre-OpenEnv API
----------------------------------

Retired keys are **rejected rather than coerced**, so a stale manifest fails at
parse instead of quietly training something else.

Manifest keys:

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - Was
     - Is now
   * - ``env_type: multiturn``
     - ``env_type: rollout``
   * - ``env_type: reasoning``
     - ``env_type: rollout`` with ``max_turns: 1``
   * - ``env_type: sft``
     - ``env_type: dataset`` with ``objective: sft``
   * - ``env_type: preference``
     - ``env_type: dataset`` with ``objective: preference``
   * - ``action_granularity``
     - ``advantage_granularity`` — for GRPO ``auto | trajectory | turn``,
       default ``auto`` (``token`` is not a GRPO value); for ``LLMPPO`` and
       ``LLMREINFORCE`` ``turn | token | auto``. The old spelling is
       rejected, not aliased.

Python API:

.. list-table::
   :widths: 44 56
   :header-rows: 1

   * - Was
     - Is now
   * - ``HuggingFaceGym``, ``ReasoningGym``, ``SyncMultiTurnVecEnv``
     - :class:`~agilerl.llm_envs.RolloutHarness`, built with
       :meth:`~agilerl.llm_envs.RolloutHarness.local`, a ``url``, or
       :meth:`~agilerl.llm_envs.RolloutHarness.from_spec`
   * - ``SFTGym``, ``PreferenceGym``
     - :class:`~agilerl.llm_envs.DatasetEnv` with ``objective="sft"`` or
       ``"preference"``
   * - ``PromptDatasetEnv``
     - ``QADatasetEnv`` (:mod:`agilerl.llm_envs.qa_dataset`)
   * - ``IterablePromptBatchGym``
     - :class:`~agilerl.llm_envs.RolloutCollector` plus
       :class:`~agilerl.llm_envs.TaskAssigner`
   * - ``TokenObservationWrapper``
     - ``observation_field`` / ``observation_processor`` on the spec, or
       :func:`~agilerl.llm_envs.process_observation`
   * - ``finetune_llm_multiturn``, ``finetune_llm_reasoning``
     - :func:`~agilerl.training.llm.train_llm_rollout`
   * - ``finetune_llm_sft``, ``finetune_llm_preference``
     - :func:`~agilerl.training.llm.train_llm_dataset`

There is no compatibility shim; the old names are gone.

Rollout data: the trajectory tensor on the wire is ``token_ids``, not
``completion_ids``. A custom environment no longer subclasses anything —
``reset`` and ``step``, in text, are the whole interface.

See :doc:`the API reference </api/wrappers/llm_envs>` for the full signatures.
