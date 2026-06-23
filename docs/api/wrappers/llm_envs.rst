LLM environments
================

Gymnasium-style environments for supervised fine-tuning, preference optimization,
and reasoning RL. :class:`~agilerl.llm_envs.DatasetEnv` covers the teacher-forced
SFT (``kind="sft"``) and preference (``kind="preference"``) regimes;
:class:`~agilerl.llm_envs.RolloutEnv` covers generation. ``apply_chat_template``
is also re-exported from :mod:`agilerl.utils.llm_utils` for backwards compatibility.

.. autoclass:: agilerl.llm_envs.DatasetEnv
.. autoclass:: agilerl.llm_envs.RolloutEnv
.. autofunction:: agilerl.llm_envs.apply_chat_template
