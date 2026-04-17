.. _cispo:

Clipped Importance Sampling Policy Optimization (CISPO)
=======================================================

``CISPO`` (Clipped Importance Sampling Policy Optimization) is a
:class:`GRPO <agilerl.algorithms.grpo.GRPO>` specialization that clips
importance weights directly and uses them to scale a log-prob objective.

In practice this keeps all tokens in the gradient path while still bounding the
magnitude of policy updates through clipped IS weights. The variant is used in
modern LLM RL tooling and research codebases as an alternative to PPO-style
surrogate clipping.

In AgileRL, ``CISPO``:

* inherits the full ``GRPO`` rollout/training stack,
* fixes ``loss_type="cispo"`` at construction time,
* keeps the same constructor arguments as ``GRPO`` except ``loss_type``.

Example
-------

.. code-block:: python

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from agilerl.algorithms import CISPO

  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

  agent = CISPO(
      actor_network=model,
      pad_token_id=tokenizer.eos_token_id,
      pad_token=tokenizer.eos_token,
      device="cuda" if torch.cuda.is_available() else "cpu",
      batch_size=8,
      group_size=8,
  )

Training and Usage
------------------

Use ``CISPO`` anywhere you would use ``GRPO`` in AgileRL training loops, such
as ``finetune_llm_reasoning`` or ``finetune_llm_multiturn``.

Saving and Loading Agents
-------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.utils.utils import save_llm_checkpoint

  save_llm_checkpoint(agent, "path/to/checkpoint")

Parameters
----------

.. autoclass:: agilerl.algorithms.cispo.CISPO
  :members:
  :inherited-members:
