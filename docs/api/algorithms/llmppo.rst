.. _llmppo:

LLM Proximal Policy Optimization (LLM PPO)
==========================================

`PPO <https://arxiv.org/abs/1707.06347>`_ (Proximal Policy Optimization)
is a policy-gradient method that keeps updates inside a clipped trust region.
``LLMPPO`` adapts this idea to causal language models and is designed for both
single-turn and multi-turn fine-tuning.

In AgileRL, the implementation is turn-aware:

* **Turn-level credit assignment:** each generated turn is treated as one RL
  action, with discounting across turns.
* **Actor-critic optimization:** policy and value adapters are updated jointly,
  with clipped policy/value losses plus entropy regularization.
* **Single-turn and multi-turn parity:** single-turn prompting is treated as
  the special case where all action tokens belong to turn ``0``.

This algorithm can therefore be used in multi-turn agentic finetuning or single-turn reasoning tasks.

Example
-------

.. code-block:: python

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from agilerl.algorithms import LLMPPO

  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-0.5B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

  agent = LLMPPO(
      actor_network=model,
      pad_token_id=tokenizer.eos_token_id,
      pad_token=tokenizer.eos_token,
      device="cuda" if torch.cuda.is_available() else "cpu",
      batch_size=8,
      update_epochs=1,
      clip_coef=0.2,
      max_output_tokens=128,
      max_model_len=1024,
  )

Training
--------

Typical training entry points are ``finetune_llm_reasoning`` and
``finetune_llm_multiturn`` in ``agilerl.training.train_llm``.

Saving and Loading Agents
-------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.utils.utils import save_llm_checkpoint

  save_llm_checkpoint(agent, "path/to/checkpoint")

As with other AgileRL LLM algorithms, loading is done with Hugging Face
``from_pretrained`` APIs for the base model and adapter.

Parameters
----------

.. autoclass:: agilerl.algorithms.ppo_llm.PPO
  :members:
  :inherited-members:
