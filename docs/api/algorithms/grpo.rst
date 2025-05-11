.. _grpo:

Group Relative Policy Optimization (GRPO)
=========================================

GRPO (Group Relative Policy Optimization) GRPO is an elegant simplification of :ref:`PPO<ppo>` (Proximal Policy Optimization)
that makes reinforcement learning more computationally efficient, especially for large language models.

The two key innovations are:

* **Eliminating the critic network:** Instead of training a separate value function to estimate expected rewards (which requires additional compute and memory), GRPO normalizes rewards across a batch of samples. It calculates advantage by subtracting the mean reward from each sample's reward and dividing by the standard deviation.
* **Group-based evaluation:** GRPO generates multiple outputs using the same policy, evaluates them as a group, and then updates the model. This approach reduces variance in the training signal by smoothing out the randomness inherent in probabilistic environments.

These changes are particularly valuable for LLM training because they reduce computational overhead by removing the
need for a separate critic model, provide more stable gradient updates in environments with sparse or noisy rewards,
and they simplify implementation while maintaining or improving performance.

* GRPO paper: https://arxiv.org/pdf/2402.03300


Example
-------

For more details on how to set up GRPO and use it for training, check out the :ref:`tutorial<grpo_tutorial>`.

.. code-block:: python

  from agilerl.algorithms import GRPO
  from agilerl.utils.llm_utils import HuggingFaceGym

  model = create_model(...)
  tokenizer = create_tokenizer(...)
  env = HuggingFaceGym(...)

  agent = GRPO(
    env.observation_space,
    env.action_space,
    actor_network=model,
    pad_token_id=tokenizer.eos_token_id,
    device="cuda:0",
    batch_size=8,
    group_size=8,
    reduce_memory_peak=True,
  )

Saving and loading agents
-------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.algorithms.grpo import GRPO
  from agilerl.utils.utils import save_llm_checkpoint

  agent = GRPO(
    env.observation_space,
    env.action_space,
    actor_network=model,
    pad_token_id=tokenizer.eos_token_id,
  )

  checkpoint_path = "path/to/checkpoint"
  save_llm_checkpoint(agent, checkpoint_path)


To load a saved agent, you must use the HuggingFace `.from_pretrained` method, AgileRL is
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
