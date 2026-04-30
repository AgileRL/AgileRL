LLM environments
================

Gymnasium-style environments for supervised fine-tuning, preference optimization,
and reasoning RL. These types are also re-exported from
:mod:`agilerl.utils.llm_utils` for backwards compatibility.

.. autoclass:: agilerl.llm_envs.HuggingFaceGym
.. autoclass:: agilerl.llm_envs.SFTGym
.. autoclass:: agilerl.llm_envs.PreferenceGym
.. autoclass:: agilerl.llm_envs.ReasoningGym
.. autofunction:: agilerl.llm_envs.apply_chat_template
