.. _gspo:

Grouped Sequence Policy Optimization (GSPO)
===========================================

`GSPO <https://arxiv.org/abs/2507.18071>`_ (Group Sequence Policy Optimization)
proposes sequence-level importance-ratio optimization for LLM RL. Compared with
token-level clipping, this reduces per-token noise and can improve long-run
training stability on long responses.

In AgileRL, GSPO can be used for single-turn reasoning tasks or multi-turn agentic finetuning. In the multi-turn case,
rollouts are still treated as a bandit problem, with environment generated tokens masked and reward signal calculated
from cumulative episode reward.

Example
-------

.. code-block:: python

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from agilerl.algorithms import GSPO

  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

  agent = GSPO(
      actor_network=model,
      pad_token_id=tokenizer.eos_token_id,
      pad_token=tokenizer.eos_token,
      device="cuda" if torch.cuda.is_available() else "cpu",
      batch_size=8,
      group_size=8,
  )

Training and Usage
------------------

``GSPO`` is designed as a drop-in replacement for ``GRPO`` in AgileRL training
entry points such as ``finetune_llm_reasoning`` and
``finetune_llm_multiturn``.

Saving and Loading Agents
-------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.utils.utils import save_llm_checkpoint

  save_llm_checkpoint(agent, "path/to/checkpoint")

Parameters
----------

.. autoclass:: agilerl.algorithms.gspo.GSPO
  :members:
  :inherited-members:
