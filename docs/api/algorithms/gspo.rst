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

.. code-block:: python

  from datasets import Dataset
  from agilerl.llm_envs import ReasoningGym, TokenObservationWrapper
  from agilerl.training.train_llm import (
      finetune_llm_multiturn,
      finetune_llm_reasoning,
  )

  # 1) Single-turn / reasoning datasets (ReasoningGym)
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

Parameters
----------

.. autoclass:: agilerl.algorithms.gspo.GSPO
  :members:
  :inherited-members:
