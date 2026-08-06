LLM environments
================

Every LLM-training environment is reached through the `OpenEnv
<https://github.com/meta-pytorch/OpenEnv>`_ text contract (the ``[llm]`` extra): whatever
backs an env — a prompt dataset, plain Python functions, an imported gem / AxonRL env, or
a sandboxed VM — a :class:`~agilerl.llm_envs.RolloutHarness` drives it through a **backend**,
in one of two transports: **in-process** (a
:class:`~agilerl.llm_envs.openenv.InProcessEnvClient` — no server, no sockets) or **remote** (an
OpenEnv server reached over a URL by an
:class:`~agilerl.llm_envs.openenv.RemoteEnvClient`). :class:`~agilerl.llm_envs.openenv.OpenEnvServer`
is the building block for hosting a local env *as a URL* — a container, a Ray actor, or a
script stands one up and then points a :class:`~agilerl.llm_envs.RolloutHarness` at its
address. A server wraps each env in an
:class:`~agilerl.llm_envs.openenv.OpenEnvWrapper`, and a standard text contract —
:class:`~agilerl.llm_envs.openenv.TextAction` (``message``) and
:class:`~agilerl.llm_envs.openenv.TextObservation` (``prompt``) — carries the model's text both
ways, so there are no per-env codecs. See :ref:`llm_environments` for a guide.

* :class:`~agilerl.llm_envs.RolloutHarness` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and drives the env through
  its backend (``reset`` / ``step``). Build it with
  :meth:`~agilerl.llm_envs.RolloutHarness.local` (a local env, in-process), a ``url`` (a
  hosted OpenEnv server reached over a WebSocket session), or
  :meth:`~agilerl.llm_envs.RolloutHarness.from_spec` (resolve a URL or a ``module:Class``
  entrypoint). Single-turn reasoning is simply ``max_turns=1``.
* :class:`~agilerl.llm_envs.RolloutCollector` — runs independent groups of ``RolloutHarness``
  rollouts over a batch; a shared :class:`~agilerl.llm_envs.TaskAssigner` hands each
  group one common task — a dataset row (drawn in per-epoch shuffled order) or a
  reset seed.
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised fine-tuning and
  preference-optimization regimes.
* :class:`~agilerl.llm_envs.openenv.InProcessEnvClient` (in-process) and
  :class:`~agilerl.llm_envs.openenv.RemoteEnvClient` (one WebSocket session against a hosted
  server) are the backends a ``RolloutHarness`` drives;
  :class:`~agilerl.llm_envs.openenv_server.OpenEnvServer` hosts a local env over HTTP — one shared
  env, or a fresh env per session via ``make_env`` / ``max_concurrent_envs``;
  :func:`~agilerl.llm_envs.openenv_server.resolve_env` resolves a spec to a ``(url, server)``
  (hosting an entrypoint), and :func:`~agilerl.llm_envs.env_specs.spec_to_factory` resolves a
  non-URL spec to an env factory for the in-process path.

:func:`~agilerl.utils.llm_utils.apply_chat_template` lives in
:mod:`agilerl.utils.llm_utils` and is re-exported here for convenience.

**Declaring what an env needs.** A manifest's ``entrypoint`` names a callable that
returns a text env — your own class, or an env library's factory, as in
``entrypoint: gem:make`` with ``env_config: {env_id: game:Sudoku-v0-easy}``. Add
``env_packages`` (``{"uv": [...]}`` or ``{"pip": [...]}``) to declare what has to be
installed for that import to work, and the first env build installs it here if it is
missing. If those requirements cannot resolve alongside the trainer's own, the env
needs a virtualenv — and so a process — of its own: that is what the same field means
to an orchestrator, which installs them onto a dedicated env host and drives the env
over ``/ws``, so one manifest covers both.

To drive a *real external* OpenEnv server (e.g. an env on the HuggingFace Hub), point
:class:`~agilerl.llm_envs.RolloutHarness` at its URL; each rollout opens its own ``/ws``
session via :class:`~agilerl.llm_envs.openenv.RemoteEnvClient` and the server backs it with
a fresh env instance, so the server's ``max_concurrent_envs`` must cover ``batch_size *
group_size`` sessions plus one for the lazily built eval env. If that server exposes an
MCP tool, pass ``mcp_tool=`` to :class:`~agilerl.llm_envs.RolloutHarness` /
:class:`~agilerl.llm_envs.openenv.RemoteEnvClient` to make the tool available during
rollouts. In a run manifest, ``request_timeout_s`` bounds each message at 300 seconds by
default; ``0`` disables the bound.

**One environment per rollout.** Each concurrent rollout gets its own env instance,
created once and reused for the whole run. For batched training,
:class:`~agilerl.llm_envs.RolloutCollector` calls the ``env_factory`` ``batch_size *
group_size`` times — once per slot — so the count is determined by the batch, at the
training layer. A local env factory (``lambda: RolloutHarness.local(make_env(), tok)``) runs
each in-process. To run against a hosted service instead, stand up an
:class:`~agilerl.llm_envs.openenv_server.OpenEnvServer` (with ``make_env`` /
``max_concurrent_envs`` so it backs each session with a fresh env instance) and point a factory
at its URL
(``lambda: RolloutHarness(server.base_url, tok, max_turns=...)``); the server's
``max_concurrent_envs`` must cover ``batch_size * group_size`` sessions plus one for the
lazily built eval env.

.. autoclass:: agilerl.llm_envs.RolloutHarness
.. autoclass:: agilerl.llm_envs.RolloutCollector
.. autoclass:: agilerl.llm_envs.TaskAssigner
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.openenv_server.OpenEnvWrapper
.. autoclass:: agilerl.llm_envs.openenv_server.TextAction
.. autoclass:: agilerl.llm_envs.openenv_server.TextObservation
.. autoclass:: agilerl.llm_envs.openenv_server.OpenEnvServer
.. autoclass:: agilerl.llm_envs.openenv.RemoteEnvClient
.. autoclass:: agilerl.llm_envs.openenv.InProcessEnvClient
.. autofunction:: agilerl.llm_envs.openenv_server.resolve_env
.. autofunction:: agilerl.llm_envs.env_specs.spec_to_factory
