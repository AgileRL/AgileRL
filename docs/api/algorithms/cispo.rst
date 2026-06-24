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
as ``train_llm_rollout``. Single-turn reasoning is the ``max_turns=1`` case
of the same function.

.. code-block:: python

  from agilerl.llm_envs import RolloutEnv, local_transport
  from agilerl.training.train_llm import train_llm_rollout

  def reward_fn(completion: str, answer: str, question: str) -> float:
      del question
      return float(answer.lower() in completion.lower())

  # 1) Single-turn / reasoning datasets (a prompt dataset driven by RolloutEnv, max_turns=1)
  def env_factory(evaluation_mode: bool = False):
      env = RolloutEnv.from_dataset(
          questions=["2+2?", "Capital of France?"],
          answers=["4", "Paris"],
          reward_fn=reward_fn,
          tokenizer=tokenizer,
          prompt_builder=lambda question: f"Q: {question}\nA:",
          test_questions=["3+3?"],
          test_answers=["6"],
          max_turns=1,
          pad_id=tokenizer.eos_token_id,
          apply_chat_template=True,
          max_model_len=1024,
      )
      env.evaluation_mode = evaluation_mode
      return env

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
      return RolloutEnv(
          None,
          tokenizer=tokenizer,
          max_turns=4,
          transport=local_transport(ToyMultiTurnEnv()),
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

Parameters
----------

.. autoclass:: agilerl.algorithms.cispo.CISPO
  :members:
  :inherited-members:
