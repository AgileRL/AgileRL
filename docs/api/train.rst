Training
=========

If you are using a Gym-style environment, it is easiest to use our training function, which returns a population of trained agents and logged training metrics.

.. autofunction:: agilerl.training.train_off_policy.train_off_policy

.. autofunction:: agilerl.training.train_on_policy.train_on_policy

If you are training on static, offline data, you can use our offline RL training function.

.. autofunction:: agilerl.training.train_offline.train_offline

The multi agent training function handles Pettingzoo-style environments and multi-agent algorithms.

.. autofunction:: agilerl.training.train_multi_agent_off_policy.train_multi_agent_off_policy

Finally, if you are training a LLM, you can use our LLM training functions. ``train_llm_rollout`` runs online RL over rollout (generate-and-score) environments and should be
used with GRPO, PPO or REINFORCE; it drives multi-turn rollouts, and single-turn reasoning is the ``max_turns=1`` case. ``train_llm_dataset`` runs offline, teacher-forced
training over a ``DatasetEnv`` dataloader; the algorithm of the population selects the regime, with DPO for pairwise preference data and SFT for supervised fine-tuning on static data.

.. autofunction:: agilerl.training.train_llm.train_llm_rollout

.. _train_llm_dataset:

.. autofunction:: agilerl.training.train_llm.train_llm_dataset
