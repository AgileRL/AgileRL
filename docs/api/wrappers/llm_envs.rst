LLM environments
================

Every LLM-training environment is reached through the `OpenEnv
<https://github.com/meta-pytorch/OpenEnv>`_ text contract (the ``[llm]`` extra): whatever
backs an env — a prompt dataset, plain Python functions, an imported gem / AxonRL env, or
a sandboxed VM — a :class:`~agilerl.llm_envs.RolloutEnv` drives it through a **backend**,
in one of two transports: **in-process** (a
:class:`~agilerl.llm_envs.openenv.LocalEnvClient` — no server, no sockets) or **remote** (an
OpenEnv server reached over a URL by an
:class:`~agilerl.llm_envs.openenv.OpenEnvSessionClient`). :class:`~agilerl.llm_envs.openenv.OpenEnvServer`
is the building block for hosting a local env *as a URL* — a container, a Ray actor, or a
script stands one up and then points a :class:`~agilerl.llm_envs.RolloutEnv` at its
address. A server wraps each env in an
:class:`~agilerl.llm_envs.openenv.OpenEnvWrapper`, and a standard text contract —
:class:`~agilerl.llm_envs.openenv.TextAction` (``message``) and
:class:`~agilerl.llm_envs.openenv.TextObservation` (``prompt``) — carries the model's text both
ways, so there are no per-env codecs. See :ref:`llm_environments` for a guide.

* :class:`~agilerl.llm_envs.RolloutEnv` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and drives the env through
  its backend (``reset`` / ``step``). Build it with
  :meth:`~agilerl.llm_envs.RolloutEnv.local` (a local env, in-process), a ``url`` (a
  hosted OpenEnv server reached over a WebSocket session), or
  :meth:`~agilerl.llm_envs.RolloutEnv.from_spec` (resolve a URL or a ``module:Class``
  entrypoint). Single-turn reasoning is simply ``max_turns=1``.
* :class:`~agilerl.llm_envs.BatchRolloutEnv` — runs independent groups of ``RolloutEnv``
  rollouts over a batch; a shared :class:`~agilerl.llm_envs.TaskAssigner` hands each
  group one common task — a dataset row (drawn in per-epoch shuffled order) or a
  reset seed.
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised fine-tuning and
  preference-optimization regimes.
* :class:`~agilerl.llm_envs.openenv.LocalEnvClient` (in-process) and
  :class:`~agilerl.llm_envs.openenv.OpenEnvSessionClient` (one WebSocket session against a hosted
  server) are the backends a ``RolloutEnv`` drives;
  :class:`~agilerl.llm_envs.openenv_server.OpenEnvServer` hosts a local env over HTTP — one shared
  env, or a fresh env per session via ``make_env`` / ``max_concurrent_envs``;
  :func:`~agilerl.llm_envs.openenv_server.resolve_env` resolves a spec to a ``(url, server)``
  (hosting an entrypoint), and :func:`~agilerl.llm_envs.spec_resolvers.spec_to_factory` resolves a
  non-URL spec to an env factory for the in-process path.

:func:`~agilerl.utils.llm_utils.apply_chat_template` lives in
:mod:`agilerl.utils.llm_utils` and is re-exported here for convenience.

To drive a *real external* OpenEnv server (e.g. an env on the HuggingFace Hub), point
:class:`~agilerl.llm_envs.RolloutEnv` at its URL; each rollout opens its own ``/ws``
session via :class:`~agilerl.llm_envs.openenv.OpenEnvSessionClient` and the server backs it with
a fresh env instance, so the server's ``max_concurrent_envs`` must cover ``batch_size *
group_size`` sessions plus one for the lazily built eval env. If that server exposes an
MCP tool, pass ``mcp_tool=`` to :class:`~agilerl.llm_envs.RolloutEnv` /
:class:`~agilerl.llm_envs.openenv.OpenEnvSessionClient` to make the tool available during
rollouts. In a run manifest, ``request_timeout_s`` bounds each message at 300 seconds by
default; ``0`` disables the bound.

**One environment per rollout.** Each concurrent rollout gets its own env instance,
created once and reused for the whole run. For batched training,
:class:`~agilerl.llm_envs.BatchRolloutEnv` calls the ``env_factory`` ``batch_size *
group_size`` times — once per slot — so the count is determined by the batch, at the
training layer. A local env factory (``lambda: RolloutEnv.local(make_env(), tok)``) runs
each in-process. To run against a hosted service instead, stand up an
:class:`~agilerl.llm_envs.openenv_server.OpenEnvServer` (with ``make_env`` /
``max_concurrent_envs`` so it backs each session with a fresh env instance) and point a factory
at its URL
(``lambda: RolloutEnv(server.base_url, tok, max_turns=...)``); the server's
``max_concurrent_envs`` must cover ``batch_size * group_size`` sessions plus one for the
lazily built eval env.

.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autoclass:: agilerl.llm_envs.BatchRolloutEnv
.. autoclass:: agilerl.llm_envs.TaskAssigner
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.openenv_server.OpenEnvWrapper
.. autoclass:: agilerl.llm_envs.openenv_server.TextAction
.. autoclass:: agilerl.llm_envs.openenv_server.TextObservation
.. autoclass:: agilerl.llm_envs.openenv_server.OpenEnvServer
.. autoclass:: agilerl.llm_envs.openenv.OpenEnvSessionClient
.. autoclass:: agilerl.llm_envs.openenv.LocalEnvClient
.. autofunction:: agilerl.llm_envs.openenv_server.resolve_env
.. autofunction:: agilerl.llm_envs.spec_resolvers.spec_to_factory
