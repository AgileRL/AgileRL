LLM environments
================

Every LLM-training environment is hosted and reached through `OpenEnv
<https://github.com/meta-pytorch/OpenEnv>`_ (the ``[llm]`` extra): whatever backs an
env — a prompt dataset, plain Python functions, an imported gem / AxonRL env, or a
sandboxed VM — it is wrapped in a :class:`~agilerl.llm_envs.GymEnvironment` (an
OpenEnv ``Environment``), served by OpenEnv's ``HTTPEnvServer`` via
:class:`~agilerl.llm_envs.OpenEnvServer` / :func:`~agilerl.llm_envs.serve` (or driven
in-process by the socket-free :func:`~agilerl.llm_envs.local_transport`), and a
:class:`~agilerl.llm_envs.RolloutEnv` drives it from a URL over the OpenEnv wire. A
standard text contract — :class:`~agilerl.llm_envs.TextAction` (``message``) and
:class:`~agilerl.llm_envs.TextObservation` (``prompt``) — carries the model's text
both ways, so there are no per-env codecs.

* :class:`~agilerl.llm_envs.RolloutEnv` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and talks to the env
  over the OpenEnv API (``reset`` / ``step``). Constructed with a ``url``, or with
  :meth:`~agilerl.llm_envs.RolloutEnv.from_dataset` for an in-process prompt dataset
  scored by a ``reward_fn`` (the canonical reasoning task).
* :class:`~agilerl.llm_envs.BatchRolloutEnv` — runs independent groups of
  ``RolloutEnv`` rollouts over a batch (the dataset shuffle + GRPO group pinning).
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised
  fine-tuning and preference-optimization regimes.
* :class:`~agilerl.llm_envs.OpenEnvServer` / :func:`~agilerl.llm_envs.serve` host a
  local env over HTTP; :class:`~agilerl.llm_envs.OpenEnvClient` is the client a
  ``RolloutEnv`` drives; :func:`~agilerl.llm_envs.local_transport` is the
  socket-free in-process transport; and :func:`~agilerl.llm_envs.resolve_env` turns
  an env spec (a URL, a registered name, or a ``module:Class`` / ``path.py:Class``
  entrypoint) into a URL, hosting it locally when it is not already a URL.

``apply_chat_template`` is also re-exported from :mod:`agilerl.utils.llm_utils`.

For driving a *real external* OpenEnv server (e.g. an env on the HuggingFace Hub),
:class:`~agilerl.llm_envs.OpenEnvHTTPEnv` bridges that env's typed schema to the text
contract so it can be served / driven like any local env.

**Concurrency — one server per rollout.** A served env handles one episode at a
time, so a single shared URL is correct only for one in-flight rollout
(``batch_size = 1``, or a GRPO group that shares a prompt). For batched training,
host one server per rollout: pass :meth:`~agilerl.llm_envs.RolloutEnv.serving` (or
:meth:`~agilerl.llm_envs.RolloutEnv.from_dataset` with ``serve=True``) as the
``env_factory``. A :class:`~agilerl.llm_envs.BatchRolloutEnv` calls that factory
``batch_size * group_size`` times, so it spins up that many isolated server
instances (one OS thread + port each) and stops them all on ``close`` — the count is
determined by the batch, at the training layer. An in-process env sidesteps this
entirely: :func:`~agilerl.llm_envs.local_transport` (what ``from_dataset`` uses by
default) already gives each rollout its own env, no socket.

.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autoclass:: agilerl.llm_envs.BatchRolloutEnv
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.GymEnvironment
.. autoclass:: agilerl.llm_envs.TextAction
.. autoclass:: agilerl.llm_envs.TextObservation
.. autoclass:: agilerl.llm_envs.OpenEnvServer
.. autoclass:: agilerl.llm_envs.OpenEnvClient
.. autoclass:: agilerl.llm_envs.OpenEnvHTTPEnv
.. autofunction:: agilerl.llm_envs.serve
.. autofunction:: agilerl.llm_envs.local_transport
.. autofunction:: agilerl.llm_envs.resolve_env
.. autofunction:: agilerl.llm_envs.apply_chat_template
