LLM environments
================

Every LLM-training environment is reached through the `OpenEnv
<https://github.com/meta-pytorch/OpenEnv>`_ text contract (the ``[llm]`` extra): whatever
backs an env — a prompt dataset, plain Python functions, an imported gem / AxonRL env, or
a sandboxed VM — a :class:`~agilerl.llm_envs.RolloutEnv` drives it through a **backend**.
The backend is a :class:`~agilerl.llm_envs.LocalEnvClient` when the env runs in-process
(no HTTP), or an :class:`~agilerl.llm_envs.OpenEnvClient` when it is reached over a URL (a
remote Space, or a local env hosted by :class:`~agilerl.llm_envs.OpenEnvServer`, which
wraps it in an :class:`~agilerl.llm_envs.OpenEnvWrapper`). A standard text contract —
:class:`~agilerl.llm_envs.TextAction` (``message``) and
:class:`~agilerl.llm_envs.TextObservation` (``prompt``) — carries the model's text both
ways, so there are no per-env codecs. See :ref:`llm_environments` for a guide.

* :class:`~agilerl.llm_envs.RolloutEnv` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and drives the env through
  its backend (``reset`` / ``step``). Build it with
  :meth:`~agilerl.llm_envs.RolloutEnv.local` (a local env, in-process), a ``url`` (HTTP),
  :meth:`~agilerl.llm_envs.RolloutEnv.serving` (host a local env over HTTP), or
  :meth:`~agilerl.llm_envs.RolloutEnv.from_spec` (resolve a URL or a ``module:Class``
  entrypoint). Single-turn reasoning is simply ``max_turns=1``.
* :class:`~agilerl.llm_envs.BatchRolloutEnv` — runs independent groups of ``RolloutEnv``
  rollouts over a batch, sharing a :class:`~agilerl.llm_envs.BatchPointer` dataset cursor
  (per-epoch shuffle + GRPO group pinning).
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised fine-tuning and
  preference-optimization regimes.
* :class:`~agilerl.llm_envs.LocalEnvClient` (in-process) and
  :class:`~agilerl.llm_envs.OpenEnvClient` (HTTP) are the two backends a ``RolloutEnv``
  drives; :class:`~agilerl.llm_envs.OpenEnvServer` hosts a local env over HTTP;
  :func:`~agilerl.llm_envs.resolve_env` resolves a spec to a ``(url, server)`` (hosting an
  entrypoint), and :func:`~agilerl.llm_envs.load_env` just builds an entrypoint env (no
  hosting) for the in-process path.

:func:`~agilerl.utils.llm_utils.apply_chat_template` lives in
:mod:`agilerl.utils.llm_utils` and is re-exported here for convenience.

To drive a *real external* OpenEnv server (e.g. an env on the HuggingFace Hub), point
:class:`~agilerl.llm_envs.RolloutEnv` at its URL; the env is reached over HTTP by the
:class:`~agilerl.llm_envs.OpenEnvClient` just like a locally hosted one. If that server
exposes an MCP tool, pass ``mcp_tool=`` to :class:`~agilerl.llm_envs.RolloutEnv` /
:class:`~agilerl.llm_envs.OpenEnvClient` to make the tool available during rollouts.

**One environment per rollout.** Each concurrent rollout gets its own env instance,
created once and reused for the whole run. For batched training,
:class:`~agilerl.llm_envs.BatchRolloutEnv` calls the ``env_factory`` ``batch_size *
group_size`` times — once per slot — so the count is determined by the batch, at the
training layer. A local env factory (``lambda: RolloutEnv.local(make_env(), tok)``) runs
each in-process; a served factory (:meth:`~agilerl.llm_envs.RolloutEnv.serving`) gives
each its own HTTP server (one OS thread + port), closed on ``close``.

.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autoclass:: agilerl.llm_envs.BatchRolloutEnv
.. autoclass:: agilerl.llm_envs.BatchPointer
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.OpenEnvWrapper
.. autoclass:: agilerl.llm_envs.TextAction
.. autoclass:: agilerl.llm_envs.TextObservation
.. autoclass:: agilerl.llm_envs.OpenEnvServer
.. autoclass:: agilerl.llm_envs.OpenEnvClient
.. autoclass:: agilerl.llm_envs.LocalEnvClient
.. autofunction:: agilerl.llm_envs.resolve_env
.. autofunction:: agilerl.llm_envs.load_env
