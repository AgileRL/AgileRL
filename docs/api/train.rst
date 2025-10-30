Training
=========

If you are using a Gym-style environment, it is easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. autofunction:: agilerl.training.train_off_policy.train_off_policy

.. autofunction:: agilerl.training.train_on_policy.train_on_policy

If you are training on static, offline data, you can use our offline RL training function.

.. autofunction:: agilerl.training.train_offline.train_offline

The multi agent training function handles Pettingzoo-style environments and multi-agent algorithms.

.. autofunction:: agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy

Finally, if you are training a LLM, you can use our LLM training functions. We have one for preference-based reinforcement learning (``finetune_llm_preference``) which should be used
with DPO, and one for reinforcement learning with verifiable rewards (``finetune_llm_reasoning``) which should be used with GRPO.

.. autofunction:: agilerl.training.train_llm.finetune_llm_reasoning

.. autofunction:: agilerl.training.train_llm.finetune_llm_preference
