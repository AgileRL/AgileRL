LLM environments
================

Every LLM-training environment is reached through the `OpenEnv
<https://github.com/meta-pytorch/OpenEnv>`_ text contract (the ``[llm]`` extra): whatever
backs an env — a prompt dataset, plain Python functions, an imported gem / AxonRL env, or
a sandboxed VM — a :class:`~agilerl.llm_envs.RolloutEnv` drives it through a **backend**,
in one of three deployment shapes: **in-process** (a
:class:`~agilerl.llm_envs.LocalEnvClient` — no server, no sockets), **served locally** (a
:class:`~agilerl.llm_envs.ServedEnvFactory` hosting one shared in-process
:class:`~agilerl.llm_envs.OpenEnvServer` — the local rehearsal of production), or
**remote** (an already-hosted OpenEnv server reached over a URL by an
:class:`~agilerl.llm_envs.OpenEnvSessionClient`). A server wraps each env in an
:class:`~agilerl.llm_envs.OpenEnvWrapper`, and a standard text contract —
:class:`~agilerl.llm_envs.TextAction` (``message``) and
:class:`~agilerl.llm_envs.TextObservation` (``prompt``) — carries the model's text both
ways, so there are no per-env codecs. See :ref:`llm_environments` for a guide.

* :class:`~agilerl.llm_envs.RolloutEnv` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and drives the env through
  its backend (``reset`` / ``step``). Build it with
  :meth:`~agilerl.llm_envs.RolloutEnv.local` (a local env, in-process), a ``url`` (HTTP),
  :meth:`~agilerl.llm_envs.RolloutEnv.serving` (host one standalone env over HTTP), or
  :meth:`~agilerl.llm_envs.RolloutEnv.from_spec` (resolve a URL or a ``module:Class``
  entrypoint). Single-turn reasoning is simply ``max_turns=1``.
* :class:`~agilerl.llm_envs.BatchRolloutEnv` — runs independent groups of ``RolloutEnv``
  rollouts over a batch, sharing a :class:`~agilerl.llm_envs.BatchPointer` dataset cursor
  (per-epoch shuffle + GRPO group pinning).
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised fine-tuning and
  preference-optimization regimes.
* :class:`~agilerl.llm_envs.LocalEnvClient` (in-process),
  :class:`~agilerl.llm_envs.OpenEnvSessionClient` (one WebSocket session against a
  server) and :class:`~agilerl.llm_envs.ServedEnvClient` (a private server + session,
  owned as one backend) are the backends a ``RolloutEnv`` drives;
  :class:`~agilerl.llm_envs.OpenEnvServer` hosts a local env over HTTP — one shared env,
  or a fresh env per session via ``make_env`` / ``max_concurrent_envs``;
  :func:`~agilerl.llm_envs.resolve_env` resolves a spec to a ``(url, server)`` (hosting an
  entrypoint), and :func:`~agilerl.llm_envs.load_env` just builds an entrypoint env (no
  hosting) for the in-process path.

:func:`~agilerl.utils.llm_utils.apply_chat_template` lives in
:mod:`agilerl.utils.llm_utils` and is re-exported here for convenience.

To drive a *real external* OpenEnv server (e.g. an env on the HuggingFace Hub), point
:class:`~agilerl.llm_envs.RolloutEnv` at its URL; each rollout opens its own ``/ws``
session via :class:`~agilerl.llm_envs.OpenEnvSessionClient` and the server backs it with
a fresh env instance, so the server's ``max_concurrent_envs`` must cover ``batch_size *
group_size`` sessions plus one for the lazily built eval env. If that server exposes an
MCP tool, pass ``mcp_tool=`` to :class:`~agilerl.llm_envs.RolloutEnv` /
:class:`~agilerl.llm_envs.OpenEnvSessionClient` to make the tool available during
rollouts. In a run manifest, ``request_timeout_s`` bounds each message at 300 seconds by
default; ``0`` disables the bound.

**One environment per rollout.** Each concurrent rollout gets its own env instance,
created once and reused for the whole run. For batched training,
:class:`~agilerl.llm_envs.BatchRolloutEnv` calls the ``env_factory`` ``batch_size *
group_size`` times — once per slot — so the count is determined by the batch, at the
training layer. A local env factory (``lambda: RolloutEnv.local(make_env(), tok)``) runs
each in-process; a :class:`~agilerl.llm_envs.ServedEnvFactory` hosts one shared server
(one URL) and gives each slot its own WebSocket session backed by a fresh env instance —
plus one extra session for the lazily built eval env — stopping the server when the last
env it built closes. :meth:`~agilerl.llm_envs.RolloutEnv.serving`, which owns a private
server per env, suits one standalone served env, not a batch.

.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autoclass:: agilerl.llm_envs.BatchRolloutEnv
.. autoclass:: agilerl.llm_envs.BatchPointer
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.OpenEnvWrapper
.. autoclass:: agilerl.llm_envs.TextAction
.. autoclass:: agilerl.llm_envs.TextObservation
.. autoclass:: agilerl.llm_envs.OpenEnvServer
.. autoclass:: agilerl.llm_envs.OpenEnvSessionClient
.. autoclass:: agilerl.llm_envs.ServedEnvClient
.. autoclass:: agilerl.llm_envs.ServedEnvFactory
.. autoclass:: agilerl.llm_envs.LocalEnvClient
.. autofunction:: agilerl.llm_envs.resolve_env
.. autofunction:: agilerl.llm_envs.load_env
