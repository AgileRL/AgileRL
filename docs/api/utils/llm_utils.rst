LLM Utils
=========

Helpers for LLM training (FSDP gathers, model loading, eval sampling, chat
templating).  For Gymnasium-style LLM datasets, see :mod:`agilerl.llm_envs`.

.. autofunction:: agilerl.utils.llm_utils.apply_chat_template
.. autofunction:: agilerl.utils.llm_utils.get_lora_params
.. autofunction:: agilerl.utils.llm_utils.create_model_from_name_or_path
.. autofunction:: agilerl.utils.llm_utils.sample_eval_prompts
.. autofunction:: agilerl.utils.llm_utils.compare_responses
