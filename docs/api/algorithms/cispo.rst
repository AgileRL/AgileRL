.. _cispo:

Clipped Importance Sampling Policy Optimization (CISPO)
=======================================================

`CISPO <https://arxiv.org/abs/2506.13585>`__ (Clipped Importance Sampling Policy Optimization) is a
:class:`GRPO <agilerl.algorithms.grpo.GRPO>` specialization that clips
importance weights directly and uses them to scale a log-prob objective.

CISPO uses the same group-based advantage calculation as GRPO, however, the objective function
is closer to that of REINFORCE, multiplying the log-probability term of the function by a scaled
importance ratio. A stop gradient is applied to the importance ratio, meaning the ratio is treated as
a constant that scales each token's contribution to the overall policy gradient.

In AgileRL, CISPO can be used for single-turn reasoning tasks or multi-turn agentic finetuning. In the multi-turn case,
rollouts are still treated as a bandit problem, with environment generated tokens masked and reward signal calculated
from cumulative episode reward.

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
