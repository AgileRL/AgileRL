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

Variance Reduction
------------------

LLM PPO reduces the variance of its policy gradient with a **learned value
baseline**: a value head is trained alongside the policy to predict expected
return, and the advantage is computed as return minus the value estimate
(GAE-style when discounting across turns). Compared with the group-relative
normalization used by :ref:`GRPO<grpo>` and the Return Batch Normalization
(ReBN) used by :ref:`LLM REINFORCE<llmreinforce>`, this is the most
expressive variance reducer (state-conditioned, no group requirement) but
also the most expensive: an extra adapter must be trained, the baseline is
biased while the value head is catching up, and value-fit pathologies are an
extra failure mode to debug.

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

The typical training entry point is ``train_llm_rollout`` in
``agilerl.training.train_llm``. Single-turn reasoning is the ``max_turns=1``
case of the same function.

.. code-block:: python

  from agilerl.training.train_llm import train_llm_rollout
  from agilerl.llm_envs import RolloutEnv, TokenObservationWrapper

  def reward_fn(completion: str, answer: str, question: str) -> float:
      del question
      return float(answer.lower() in completion.lower())

  # 1) Single-turn / reasoning datasets (RolloutEnv with max_turns=1)
  def env_factory(evaluation_mode: bool = False):
      raw_env = RolloutEnv(
          max_turns=1,
          questions=["2+2?", "Capital of France?"],
          answers=["4", "Paris"],
          reward_fn=reward_fn,
          prompt_builder=lambda question: f"Q: {question}\nA:",
          test_questions=["3+3?"],
          test_answers=["6"],
      )
      raw_env.evaluation_mode = evaluation_mode
      return TokenObservationWrapper(
          raw_env,
          tokenizer=tokenizer,
          max_turns=1,
          pad_id=tokenizer.eos_token_id,
          apply_chat_template=True,
          max_model_len=1024,
      )

  trained_pop = train_llm_rollout(
      pop=[agent],
      max_turns=1,
      env_factory=env_factory,
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

  trained_pop = train_llm_rollout(
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

As with other AgileRL LLM algorithms, loading is done with Hugging Face
``from_pretrained`` APIs for the base model and adapter.

Parameters
----------

.. autoclass:: agilerl.algorithms.ppo_llm.PPO
  :members:
  :inherited-members:
