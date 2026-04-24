LLM Utils
=========

Helpers for LLM training (DeepSpeed ZeRO, model loading, eval sampling).  For
Gymnasium-style LLM datasets, see :mod:`agilerl.llm_envs` (also
re-exported here for backwards compatibility).

.. autofunction:: agilerl.utils.llm_utils.gather_if_zero3
.. autofunction:: agilerl.utils.llm_utils.get_state_dict
.. autofunction:: agilerl.utils.llm_utils.create_model_from_name_or_path
.. autofunction:: agilerl.utils.llm_utils.sample_eval_prompts
.. autofunction:: agilerl.utils.llm_utils.compare_responses
