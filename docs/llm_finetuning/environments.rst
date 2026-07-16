.. _llm_environments:

Environments (OpenEnv)
======================

Every LLM-training environment in AgileRL is reached the same way — **text in, text
out** — through the `OpenEnv <https://github.com/meta-pytorch/OpenEnv>`_ contract
(installed with the ``[llm]`` extra). A ``RolloutEnv`` drives your env through a
**backend**: either **in-process** (a local Python env, no HTTP) or over a **URL** (a
remote env hosted as an HTTP server). The trainer drives both identically, so a local
env and a remote one look the same to it.

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
across all the turns of an episode. When the env runs over HTTP that instance is one
server hosting one env; in-process it is just the env object itself.

Running a local environment
---------------------------

Pass the env to :meth:`RolloutEnv.local <agilerl.llm_envs.RolloutEnv.local>`. It runs
**in the training process, with no HTTP** — the ``RolloutEnv`` drives the env's
``reset`` / ``step`` directly (``tokenizer`` is a HuggingFace tokenizer, as in the
tutorials):

.. code-block:: python

   from agilerl.llm_envs import RolloutEnv


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


   env = RolloutEnv.local(GuessEnv(), tokenizer, max_turns=3)

``max_turns`` bounds the episode, but the environment itself decides when to finish by
returning ``terminated=True`` (here, on a correct guess). Single-turn *reasoning* is
just ``max_turns=1``: the model answers once and ``step`` scores it.

To **host** the env as an HTTP server instead — to expose it to other clients, or run it
out-of-process — use :meth:`RolloutEnv.serving <agilerl.llm_envs.RolloutEnv.serving>`
(``RolloutEnv.serving(lambda: GuessEnv(), tokenizer, …)``), which starts an in-process
uvicorn server on an ephemeral port and drives it over HTTP.

Connecting to a remote environment
----------------------------------

If the environment is already running as an OpenEnv server (a container, a hosted
Space), pass its URL straight to ``RolloutEnv`` — nothing is started locally:

.. code-block:: python

   env = RolloutEnv("https://my-env.example.com", tokenizer, max_turns=4)

Either way the trainer drives ``env`` identically.
:meth:`RolloutEnv.from_spec <agilerl.llm_envs.RolloutEnv.from_spec>` picks for you from a
spec string: a URL → an HTTP client; a ``"package.module:EnvClass"`` entrypoint → loaded
and run **in-process** (or ``serve=True`` to host it over HTTP).

Tools
-----

An environment advertises tool schemas by exposing a ``tools`` attribute (a list of
JSON tool schemas). AgileRL reads them through the backend and renders them into the chat
template, so the model sees the tools it may call. Tools are **environment-driven** —
there is no separate configuration step; whatever the environment advertises is what
the model gets.

For an external environment that exposes a tool over MCP rather than the plain text
contract, pass ``mcp_tool="<tool_name>"`` to ``RolloutEnv``; the model's text is sent as
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
  episode and are released cleanly at the end. (Over HTTP, the hosting server closes it
  once on stop, never per request.)
* Closing a ``RolloutEnv`` releases its backend — stopping the server it owns, or closing
  the in-process env.

Lower-level pieces
------------------

``RolloutEnv`` builds on a few small pieces you can use directly if you need to:

* :class:`LocalEnvClient <agilerl.llm_envs.LocalEnvClient>` — the in-process backend:
  drives a local env's ``reset`` / ``step`` directly (no server, no socket). This is what
  :meth:`RolloutEnv.local` uses.
* :class:`OpenEnvClient <agilerl.llm_envs.OpenEnvClient>` — the HTTP backend: a synchronous
  client that drives a server's ``reset`` / ``step`` over a URL. (An async caller, e.g. a
  Ray actor, gets its concurrency from the actor boundary, not the client.)
* :class:`OpenEnvServer <agilerl.llm_envs.OpenEnvServer>` — host a local env in-process
  (``start`` / ``stop``, or as a context manager) and read its ``base_url``. This is what
  :meth:`RolloutEnv.serving` uses.

.. code-block:: python

   from agilerl.llm_envs import OpenEnvClient, OpenEnvServer

   with OpenEnvServer(GuessEnv()) as server:
       client = OpenEnvClient(server.base_url)
       prompt, _ = client.reset()
       obs, reward, terminated, truncated, info = client.step("5")

See :doc:`the API reference </api/wrappers/llm_envs>` for the full signatures.
