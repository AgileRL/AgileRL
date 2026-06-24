LLM environments
================

Every LLM-training environment is reached the same way: over a small text-in /
text-out HTTP protocol (the OpenEnv interface). Whatever backs an env — a prompt
dataset, plain Python functions, an imported gem / AxonRL env, or a sandboxed VM —
it is wrapped by an :class:`~agilerl.llm_envs.OpenEnvServer` (or driven in-process
by the socket-free :func:`~agilerl.llm_envs.local_transport`) and a
:class:`~agilerl.llm_envs.RolloutEnv` drives it from a URL.

* :class:`~agilerl.llm_envs.RolloutEnv` — the token-level rollout env: it owns
  tokenisation, the multi-turn loop and the provenance mask, and talks to the env
  over the OpenEnv API (``reset`` / ``step``). Constructed with a ``url`` (or with
  :meth:`~agilerl.llm_envs.RolloutEnv.from_dataset` for the reasoning case).
* :class:`~agilerl.llm_envs.BatchRolloutEnv` — runs independent groups of
  ``RolloutEnv`` rollouts over a batch (the dataset shuffle + GRPO group pinning).
* :class:`~agilerl.llm_envs.ReasoningEnv` — a dataset of prompts scored by a
  ``reward_fn`` (the canonical reasoning task); a plain local env, served like any
  other.
* :class:`~agilerl.llm_envs.DatasetEnv` — the teacher-forced supervised
  fine-tuning and preference-optimization regimes.
* :class:`~agilerl.llm_envs.OpenEnvServer` / :func:`~agilerl.llm_envs.serve` host a
  local env over HTTP; :class:`~agilerl.llm_envs.OpenEnvClient` is the client a
  ``RolloutEnv`` drives; :func:`~agilerl.llm_envs.local_transport` is the
  socket-free in-process transport; and :func:`~agilerl.llm_envs.resolve_env` turns
  an env spec (a URL, a registered name, or a ``module:Class`` / ``path.py:Class``
  entrypoint) into a URL, hosting it locally when it is not already a URL.

``apply_chat_template`` is also re-exported from :mod:`agilerl.utils.llm_utils`.

.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autoclass:: agilerl.llm_envs.BatchRolloutEnv
.. autoclass:: agilerl.llm_envs.ReasoningEnv
.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.OpenEnvServer
.. autoclass:: agilerl.llm_envs.OpenEnvClient
.. autofunction:: agilerl.llm_envs.serve
.. autofunction:: agilerl.llm_envs.local_transport
.. autofunction:: agilerl.llm_envs.resolve_env
.. autofunction:: agilerl.llm_envs.apply_chat_template
