LLM environments
================

Every LLM-training environment is hosted and reached through `OpenEnv
<https://github.com/meta-pytorch/OpenEnv>`_ (the ``[llm]`` extra): whatever backs an
env — a prompt dataset, plain Python functions, an imported gem / AxonRL env, or a
sandboxed VM — it is wrapped in a :class:`~agilerl.llm_envs.OpenEnvWrapper` (an
OpenEnv ``Environment``), served by OpenEnv's ``HTTPEnvServer`` via
:class:`~agilerl.llm_envs.OpenEnvServer` / :func:`~agilerl.llm_envs.serve`, and a
:class:`~agilerl.llm_envs.RolloutEnv` drives it from a URL over HTTP with an
:class:`~agilerl.llm_envs.OpenEnvClient`. A standard text contract —
:class:`~agilerl.llm_envs.TextAction` (``message``) and
:class:`~agilerl.llm_envs.TextObservation` (``prompt``) — carries the model's text
both ways, so there are no per-env codecs.

* :class:`~agilerl.llm_envs.RolloutEnv` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and talks to the env
  over the OpenEnv API (``reset`` / ``step``). Constructed with a ``url``, with
  :meth:`~agilerl.llm_envs.RolloutEnv.serving` to host a fresh env from a 0-arg
  factory on its own server, or with
  :meth:`~agilerl.llm_envs.RolloutEnv.from_dataset` for a prompt dataset scored by a
  ``reward_fn`` (the canonical reasoning task), which it likewise hosts on its own
  server.
* :class:`~agilerl.llm_envs.BatchRolloutEnv` — runs independent groups of
  ``RolloutEnv`` rollouts over a batch (the dataset shuffle + GRPO group pinning).
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised
  fine-tuning and preference-optimization regimes.
* :class:`~agilerl.llm_envs.OpenEnvServer` / :func:`~agilerl.llm_envs.serve` host a
  local env over HTTP; :class:`~agilerl.llm_envs.OpenEnvClient` is the client a
  ``RolloutEnv`` drives; and :func:`~agilerl.llm_envs.resolve_env` turns an env spec
  (a URL, a registered name, or a ``module:Class`` / ``path.py:Class`` entrypoint)
  into a URL, hosting it locally when it is not already a URL.

``apply_chat_template`` is also re-exported from :mod:`agilerl.utils.llm_utils`.

To drive a *real external* OpenEnv server (e.g. an env on the HuggingFace Hub), point
:class:`~agilerl.llm_envs.RolloutEnv` at its URL; the env is reached over HTTP by the
:class:`~agilerl.llm_envs.OpenEnvClient` just like a locally hosted one. If that
server exposes an MCP tool, pass ``mcp_tool=`` to
:class:`~agilerl.llm_envs.RolloutEnv` / :class:`~agilerl.llm_envs.OpenEnvClient` to
make the tool available during rollouts.

**Concurrency — one server per rollout.** A served env handles one episode at a
time, so a single shared URL is correct only for one in-flight rollout
(``batch_size = 1``, or a GRPO group that shares a prompt). For batched training,
host one server per rollout: pass :meth:`~agilerl.llm_envs.RolloutEnv.serving` (or
:meth:`~agilerl.llm_envs.RolloutEnv.from_dataset`, which always hosts its own
server) as the ``env_factory``. A :class:`~agilerl.llm_envs.BatchRolloutEnv` calls
that factory ``batch_size * group_size`` times, so it spins up that many isolated
server instances (one OS thread + port each) and stops them all on ``close`` — the
count is determined by the batch, at the training layer.

.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autoclass:: agilerl.llm_envs.BatchRolloutEnv
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.OpenEnvWrapper
.. autoclass:: agilerl.llm_envs.TextAction
.. autoclass:: agilerl.llm_envs.TextObservation
.. autoclass:: agilerl.llm_envs.OpenEnvServer
.. autoclass:: agilerl.llm_envs.OpenEnvClient
.. autofunction:: agilerl.llm_envs.serve
.. autofunction:: agilerl.llm_envs.resolve_env
.. autofunction:: agilerl.llm_envs.apply_chat_template
