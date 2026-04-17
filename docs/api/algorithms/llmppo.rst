.. _llmppo:

LLM Proximal Policy Optimization (LLM PPO)
==========================================

LLM PPO adapts proximal policy optimization to language-model finetuning.
It uses a policy head and value head on top of a causal LM, computes
token-level trajectories, and optimizes a clipped surrogate objective with
KL regularization and entropy terms.

Example
-------

.. code-block:: python

  import torch
  from transformers import AutoTokenizer
  from agilerl.algorithms import LLMPPO

  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
  agent = LLMPPO(
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

.. autoclass:: agilerl.algorithms.ppo_llm.PPO
  :members:
  :inherited-members:
