.. _dpo:

Direct Preference Optimization (DPO)
================================

`DPO <https://arxiv.org/pdf/2305.18290>`_ (Direct Preference Optimization) is an elegant simplification of :ref:`RLHF<_rlhf>` (Reinforcement Learning from Human Feedback)
that makes preference learning more computationally efficient, especially for large language models.

The two key innovations are:

* **Eliminating the reward model:** Instead of training a separate reward model to score outputs (which requires additional compute and memory), DPO directly optimizes the policy using preference data. It reparameterizes the reward function implicitly through the policy itself, deriving a closed-form solution for the optimal policy.
* **Preference-based optimization:** DPO t reats the preference learning problem as a classification task over pairs of responses. It maximizes the likelihood that preferred responses are ranked higher than rejected ones under the current policy, relative to a reference policy. This approach eliminates the need for sampling and reward model queries during training.

These changes are particularly valuable for LLM training because they reduce computational overhead by removing the
need for a separate reward model and RL training loop, provide more stable training dynamics by avoiding the complexities
of reinforcement learning, and they simplify implementation while achieving comparable or better performance than traditional RLHF.

Example
-------

.. code-block:: python

  from agilerl.algorithms import DPO
  from agilerl.utils.llm_utils import PreferenceGym
  from accelerate import Accelerator
  from datasets import load_dataset
  from peft import get_peft_model
  from transformers import AutoModelForCausalLM, AutoTokenizer
  import torch

  # Instantiate the model and the associated tokenizer
  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-3B",
      torch_dtype=torch.bfloat16,
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

  # Instantiate an accelerator object for distributed training
  accelerator = Accelerator()

  # Load the dataset into a PreferenceGym environment
  raw_dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train").shuffle(seed=42)
  train_test_split = raw_dataset.train_test_split(test_size=0.1)
  train_dataset = train_test_split["train"]
  test_dataset = train_test_split["test"]
  env = PreferenceGym(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    tokenizer=tokenizer,
    data_batch_size_per_gpu=16,
    accelerator=accelerator,
  )

  # Instantiate the agent
  agent = DPO(
    env.observation_space,
    env.action_space,
    actor_network=model,
    pad_token_id=tokenizer.eos_token_id,
    pad_token=tokenizer.eos_token,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32,
    lr=0.000005,
    beta=0.001,
    update_epochs=1,
    seed=42,
    reduce_memory_peak=True,
    accelerator=accelerator,
  )

Training a DPO agent
--------------------

To train a DPO agent on a single preference gym environment, use the :ref:`finetune_llm_preference<finetune_llm_preference>` function:

.. code-block:: python

  from agilerl.training.train_llm import finetune_llm_preference

  finetune_llm_preference(
    [agent],
    env,
    num_epochs=1,
    checkpoint_steps=250,
    accelerator=accelerator,
  )

Saving and Loading Agents
-------------------------

To save an agent, use the :ref:`save_llm_checkpoint<save_llm_checkpoint>` function:

.. code-block:: python

  from agilerl.utils.utils import save_llm_checkpoint

  save_llm_checkpoint(agent, "path/to/checkpoint")


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

.. autoclass:: agilerl.algorithms.dpo.DPO
  :members:
  :inherited-members:
