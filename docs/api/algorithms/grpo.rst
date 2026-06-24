.. _grpo:

Group Relative Policy Optimization (GRPO)
=========================================

`GRPO <https://arxiv.org/pdf/2402.03300>`__ (Group Relative Policy Optimization) is an elegant simplification of :ref:`PPO<ppo>` (Proximal Policy Optimization)
that makes reinforcement learning more computationally efficient, especially for large language models.

The two key innovations are:

* **Eliminating the critic network:** Instead of training a separate value function to estimate expected rewards (which requires additional compute and memory), GRPO normalizes rewards across a batch of samples. It calculates advantage by subtracting the mean reward from each sample's reward and dividing by the standard deviation.
* **Group-based evaluation:** GRPO generates multiple outputs using the same policy, evaluates them as a group, and then updates the model. This approach reduces variance in the training signal by smoothing out the randomness inherent in probabilistic environments.

These changes are particularly valuable for LLM training because they reduce computational overhead by removing the
need for a separate critic model, provide more stable gradient updates in environments with sparse or noisy rewards,
and they simplify implementation while maintaining or improving performance.

In AgileRL, GRPO can be used for single-turn reasoning tasks or multi-turn agentic finetuning. In the multi-turn case,
rollouts are still treated as a bandit problem, with environment generated tokens masked and reward signal calculated
from cumulative episode reward.

The objective is selected via the ``loss_type`` argument, which accepts ``"grpo"`` (the default token-level PPO-style
clipped surrogate), ``"gspo"`` (sequence-level importance ratio, see :ref:`GSPO<gspo>`) and ``"cispo"`` (clamped
importance-weighted log-prob objective, see :ref:`CISPO<cispo>`). The :class:`~agilerl.algorithms.cispo.CISPO` and
:class:`~agilerl.algorithms.gspo.GSPO` classes are thin subclasses that pin ``loss_type`` to the matching variant.

Variance Reduction
------------------

GRPO replaces PPO's learned value head with **group-relative normalization**:
for each prompt, ``group_size`` rollouts are drawn and their returns are
z-scored within the group to form the advantage. The upside is that there is
no critic to train, fit or tune, which is attractive for LLM scale; the
downside is that the baseline degenerates when the group's returns collapse
(e.g. all rollouts succeed or all fail), and the quality of the variance
reduction is tied to the group size. Compare with the
:ref:`learned value baseline used by LLM PPO<llmppo>` and
:ref:`Return Batch Normalization (ReBN) used by LLM REINFORCE<llmreinforce>`.


Example
-------

For more details on how to set up GRPO and use it for training, check out the :ref:`tutorial<grpo_tutorial>`.

.. code-block:: python

  import torch
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from agilerl.algorithms import GRPO

  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

  agent = GRPO(
    actor_network=model,
    pad_token_id=tokenizer.eos_token_id,
    pad_token=tokenizer.eos_token,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=8,
    group_size=8,
  )

Saving and Loading Agents
-------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.utils.utils import save_llm_checkpoint

  checkpoint_path = "path/to/checkpoint"
  save_llm_checkpoint(agent, checkpoint_path)


To load a trained model, you must use the HuggingFace `.from_pretrained` method, AgileRL is
compatible with HuggingFace and Peft models:

.. code-block:: python

  from transformers import AutoModelForCausalLM, AutoTokenizer
  from peft import PeftModel
  import torch

  base_model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B",
      torch_dtype=torch.bfloat16,
      device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
  model = PeftModel.from_pretrained(base_model, "path/to/model/directory")

Parameters
------------

.. autoclass:: agilerl.algorithms.grpo.GRPO
  :members:
  :inherited-members:
