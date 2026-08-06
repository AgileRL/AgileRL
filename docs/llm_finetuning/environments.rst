.. _llm_environments:

Environments (OpenEnv)
======================

Every LLM-training environment in AgileRL is reached the same way — **text in, text
out** — through the `OpenEnv <https://github.com/meta-pytorch/OpenEnv>`_ contract
(installed with the ``[llm]`` extra). A ``RolloutHarness`` drives your env through a
**backend** over one of two transports: **in-process** (a local Python env — no server,
no sockets) or **remote** (an OpenEnv server reached by URL). The trainer drives both
identically, so a local env and a remote one look the same to it.

The text contract
-----------------

An environment is any object with two methods:

.. code-block:: python

   def reset(self, seed: int | None = None) -> tuple[str, dict]:
       # returns (prompt_text, info)
       ...

   def step(self, action: str) -> tuple[str, float, bool, bool, dict]:
       # returns (observation_text, reward, terminated, truncated, info)
       ...

``reset`` returns the first prompt; ``step`` receives the model's decoded text and
returns the next observation, a reward, and whether the episode has ended. That is the
whole contract — there are no per-environment serializers to write. (Optional extras: a
``tools`` attribute and a ``dataset_size`` property, both covered below.)

One environment per rollout
---------------------------

AgileRL gives **each concurrent rollout its own environment instance**, created once and
reused for the whole run — never rebuilt per step. You don't manage that by hand; the
batched rollout collector builds one per rollout and tears them all down at the end.

This keeps the model simple and isolated: one misbehaving environment cannot corrupt
another's state, and a stateful environment (a counter, a game, a sandbox) stays alive
across all the turns of an episode. In-process, that instance is just the env object
itself. Over HTTP, **one server fronts the whole batch**: each rollout opens its own
WebSocket session and the server builds a fresh env instance for that session, so
isolation comes from sessions, not from extra servers.

Running a local environment
---------------------------

Pass the env to :meth:`RolloutHarness.local <agilerl.llm_envs.RolloutHarness.local>`. It runs
**in the training process, with no HTTP** — the ``RolloutHarness`` drives the env's
``reset`` / ``step`` directly (``tokenizer`` is a HuggingFace tokenizer, as in the
tutorials):

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

``max_turns`` bounds the episode, but the environment itself decides when to finish by
returning ``terminated=True`` (here, on a correct guess). Single-turn *reasoning* is
just ``max_turns=1``: the model answers once and ``step`` scores it.

For a **batch** — the ``env_factory`` handed to ``train_llm_rollout`` or
``RolloutCollector`` — wrap the same call in a zero-argument factory so each rollout slot
gets its own in-process env instance:

.. code-block:: python

   env_factory = lambda: RolloutHarness.local(GuessEnv(), tokenizer, max_turns=3)

Connecting to a remote environment
----------------------------------

If the environment is already running as an OpenEnv server (a container, a hosted
Space), pass its URL straight to ``RolloutHarness`` — nothing is started locally:

.. code-block:: python

   env = RolloutHarness("https://my-env.example.com", tokenizer, max_turns=4)

Each rollout opens its own ``/ws`` WebSocket session against that URL and the server
builds a fresh env per session, so the server's ``max_concurrent_envs`` must cover
``batch_size * group_size`` rollout sessions **plus one** for the lazily built
evaluation env. In a run manifest, ``request_timeout_s`` bounds each message at 300
seconds by default (``0`` disables the bound); when constructing a ``RolloutHarness``
directly, pass ``timeout_s``.

Either way the trainer drives ``env`` identically.
:meth:`RolloutHarness.from_spec <agilerl.llm_envs.RolloutHarness.from_spec>` picks for you from a
spec string: a URL → a WebSocket session client; a ``"package.module:EnvClass"``
entrypoint → loaded and run **in-process**.

Observations that are not plain text
------------------------------------

Third-party OpenEnv environments do not agree on where their observation's text lives
— a coding env returns ``{"stdout": "hi\n", "stderr": "", "exit_code": 0}``, BrowserGym
``{"text": "..."}``, OpenSpiel and Atari numeric payloads like ``{"info_state": [...]}``
with no text at all. The backend hands the payload over as received, and the
``RolloutHarness`` renders it to the prompt via its **observation processor**. Two knobs
cover the spectrum:

* ``observation_field`` names the field the text lives in, when a field lookup is
  enough. Unset, the default processor
  (:func:`process_observation <agilerl.llm_envs.process_observation>`) reads the
  specified shapes — our ``TextObservation``, an MCP tool result — and otherwise the
  single non-bookkeeping text field the payload carries, warning so you can pin it down.
* ``observation_processor`` replaces the default with your own ``payload -> str``
  callable, for composite renderings or payloads with no text fields at all. In a
  manifest it is a ``module:fn`` / ``path/to/file.py:fn`` entrypoint — upload the file
  alongside the run exactly like a rubric file:

.. code-block:: yaml

   environment:
       env_type: rollout
       env_url: https://my-openspiel-env.example.com
       max_turns: 12
       observation_processor: render_board.py:render

.. code-block:: python

   # render_board.py — receives the observation payload, returns the prompt text.
   def render(payload: dict) -> str:
       return "Board state: " + ", ".join(str(v) for v in payload["info_state"])

Both knobs apply to ``entrypoint`` and ``env_url`` environments alike (a pip-installed
hub env run in-process needs them just as much as a hosted one), and the processor runs
on the collector's concurrent I/O threads, so keep it a pure function of the payload.

Tools
-----

An environment advertises tool schemas by exposing a ``tools`` attribute (a list of
JSON tool schemas). AgileRL reads them through the backend and renders them into the chat
template, so the model sees the tools it may call. Tools are **environment-driven** —
there is no separate configuration step; whatever the environment advertises is what
the model gets.

For an external environment that exposes a tool over MCP rather than the plain text
contract, pass ``mcp_tool="<tool_name>"`` to ``RolloutHarness``; the model's text is sent as
a tool call and the result rendered back to text.

Datasets, evaluation, and grouped rollouts
------------------------------------------

A dataset-backed environment is just a regular environment that exposes a
``dataset_size`` property and accepts two extra keyword arguments on ``reset``:

.. code-block:: python

   def reset(self, seed=None, *, row_index: int | None = None,
             evaluation: bool | None = None) -> tuple[str, dict]:
       # row_index selects a specific row; evaluation serves the held-out split
       ...

These let the batched collector pin every rollout in a GRPO **group** to the same
prompt, so grouped-advantage normalization compares like with like: it draws one row
per batch item and reuses it across that item's group, reshuffling each epoch. Because
``row_index`` and ``evaluation`` are passed straight to ``reset``, this works the same
whether the environment is in-process or remote — provided the environment is
reproducible from ``(seed, row_index)``.

Lifecycle
---------

* The environment is built once per rollout and reused for the whole run.
* Your environment's ``close()`` (if it has one) is called **exactly once**, on teardown
  — not per step — so resources (subprocesses, connections, sandboxes) live for the whole
  episode and are released cleanly at the end. (Over HTTP, the hosting server closes a
  session's env when that session ends, never per request.)
* Closing a ``RolloutHarness`` releases its backend — ending its WebSocket session or closing
  the in-process env.

Lower-level pieces
------------------

``RolloutHarness`` builds on a few small pieces you can use directly if you need to:

* :class:`InProcessEnvClient <agilerl.llm_envs.openenv.InProcessEnvClient>` — the in-process backend:
  drives a local env's ``reset`` / ``step`` directly (no server, no socket). This is what
  :meth:`RolloutHarness.local` uses.
* :class:`RemoteEnvClient <agilerl.llm_envs.openenv.RemoteEnvClient>` — the WebSocket
  backend: a synchronous client holding one ``/ws`` session against a server, which backs
  each session with its own env instance. (An async caller, e.g. a Ray actor, gets its
  concurrency from the actor boundary, not the client.)
* :class:`OpenEnvServer <agilerl.llm_envs.openenv.OpenEnvServer>` — the building block for hosting
  a local env *as a URL* (``start`` / ``stop``, or as a context manager) and reading its
  ``base_url``; a container, a Ray actor, or a script stands one up. Pass ``env`` for one
  shared env, or ``make_env`` with ``max_concurrent_envs`` for a fresh env per session.

.. code-block:: python

   from agilerl.llm_envs.openenv import RemoteEnvClient, OpenEnvServer

   with OpenEnvServer(GuessEnv()) as server:
       client = RemoteEnvClient(server.base_url)
       payload, _ = client.reset()          # the observation as received
       payload, reward, terminated, truncated, info = client.step("5")
       client.close()

See :doc:`the API reference </api/wrappers/llm_envs>` for the full signatures.
