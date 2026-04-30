.. _llmreinforce:

LLM REINFORCE
=============

`REINFORCE <https://doi.org/10.1007/BF00992696>`_ is the classic
score-function policy-gradient method. ``LLMREINFORCE`` brings this approach to
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
  from agilerl.algorithms import LLMREINFORCE

  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-0.5B",
      torch_dtype=torch.bfloat16,
      device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

  agent = LLMREINFORCE(
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

.. code-block:: python

  from datasets import Dataset
  from agilerl.training.train_llm import (
      finetune_llm_multiturn,
      finetune_llm_reasoning,
  )
  from agilerl.llm_envs import ReasoningGym, TokenObservationWrapper

  # Tiny mock reasoning dataset
  train_ds = Dataset.from_dict(
      {
          "question": ["2+2?", "Capital of France?"],
          "answer": ["4", "Paris"],
      }
  )
  test_ds = Dataset.from_dict(
      {
          "question": ["3+3?"],
          "answer": ["6"],
      }
  )

  def reward_fn(completion: str, answer: str, question: str) -> float:
      del question
      return float(answer.lower() in completion.lower())

  reasoning_env = ReasoningGym(
      train_dataset=train_ds,
      test_dataset=test_ds,
      tokenizer=tokenizer,
      reward_fn=reward_fn,
      conversation_template=[{"role": "user", "content": "Q: {question}\nA:"}],
      data_batch_size_per_gpu=2,
  )

  # 1) Single-turn / reasoning datasets (ReasoningGym)
  trained_pop = finetune_llm_reasoning(
      pop=[agent],
      env=reasoning_env,
      max_steps=2000,
      evaluation_interval=50,
  )

  # 2) Multi-turn text environments (factory + wrapper)
  class ToyMultiTurnEnv:
      def reset(self, seed=None):
          del seed
          return "Start: What is 2+2?", {}

      def step(self, action: str):
          reward = 1.0 if "4" in action else 0.0
          return "Done.", reward, True, False, {"correct": bool(reward)}

  def env_factory():
      return TokenObservationWrapper(
          env=ToyMultiTurnEnv(),
          tokenizer=tokenizer,
          max_turns=4,
          pad_id=tokenizer.eos_token_id,
          max_model_len=1024,
          max_output_tokens=128,
      )

  trained_pop = finetune_llm_multiturn(
      pop=[agent],
      max_turns=4,
      env_factory=env_factory,
      max_steps=2000,
      evaluation_interval=50,
  )

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
