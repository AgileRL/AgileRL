.. _llmreinforce:

LLM REINFORCE
=============

`REINFORCE <https://doi.org/10.1007/BF00992696>`_ is the classic
score-function policy-gradient method. ``LLMReinforce`` brings this approach to
causal language model finetuning with turn-aware trajectories.

In AgileRL, the algorithm uses Return Batch Normalization (ReBN) to improve
stability in practice:

* **Turn-level Monte Carlo returns:** discounted returns are computed across
  turns for each sampled trajectory.
* **Batch-normalized returns (ReBN):** turn returns are z-scored across valid
  ``(sample, turn)`` pairs before being broadcast to token-level advantages.
* **Value-head-free training:** unlike PPO-style actor-critic updates, this
  path optimizes the policy directly from normalized returns.

Example
-------

.. code-block:: python

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from agilerl.algorithms import LLMReinforce

  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-0.5B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

  agent = LLMReinforce(
      actor_network=model,
      pad_token_id=tokenizer.eos_token_id,
      pad_token=tokenizer.eos_token,
      device="cuda" if torch.cuda.is_available() else "cpu",
      batch_size=8,
      update_epochs=1,
      gamma=0.99,
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

Loading follows the standard Hugging Face ``from_pretrained`` flow for the base
model and any finetuned adapter.

Parameters
----------

.. autoclass:: agilerl.algorithms.reinforce_llm.REINFORCE
  :members:
  :inherited-members:
