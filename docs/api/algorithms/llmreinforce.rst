.. _llmreinforce:

LLM REINFORCE
=============

LLM REINFORCE provides a policy-gradient finetuning algorithm for causal
language models using REINFORCE-style updates. It supports tokenized
trajectory rollouts and can be used in reasoning and multi-turn training
setups.

Example
-------

.. code-block:: python

  import torch
  from transformers import AutoTokenizer
  from agilerl.algorithms import LLMReinforce

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
  agent = LLMReinforce(
      actor_network="Qwen/Qwen2.5-0.5B",
      pad_token_id=tokenizer.eos_token_id,
      pad_token=tokenizer.eos_token,
      device="cuda" if torch.cuda.is_available() else "cpu",
      batch_size=8,
      max_output_tokens=128,
      max_model_len=1024,
  )

Training
--------

Typical training entry points are ``finetune_llm_reasoning`` and
``finetune_llm_multiturn`` in ``agilerl.training.train_llm``.

Parameters
----------

.. autoclass:: agilerl.algorithms.reinforce_llm.REINFORCE
  :members:
  :inherited-members:
