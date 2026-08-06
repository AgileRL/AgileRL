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

  from agilerl.llm_envs import RolloutHarness
  from agilerl.training.llm import train_llm_rollout

  def reward_fn(completion: str, answer: str, question: str) -> float:
      del question
      return float(answer.lower() in completion.lower())

  # 1) Single-turn / reasoning datasets (a prompt dataset is just an env, max_turns=1)
  class PromptDataset:
      """Single-turn dataset env: serve a question on reset, score it on step."""

      def __init__(self, questions, answers, reward_fn, prompt_builder,
                   test_questions=None, test_answers=None):
          self.questions, self.answers = questions, answers
          self.test_questions, self.test_answers = test_questions, test_answers
          self.reward_fn, self.prompt_builder = reward_fn, prompt_builder
          self._cursor, self._split = 0, ""

      @property
      def dataset_size(self) -> int:
          return len(self.questions)

      def reset(self, seed=None, *, row_index=None, evaluation=None):
          if evaluation and self.test_questions is not None:
              qs, ans, split = self.test_questions, self.test_answers, "eval"
          else:
              qs, ans, split = self.questions, self.answers, "train"
          if row_index is None:
              if split != self._split:
                  self._cursor, self._split = 0, split
              row_index, self._cursor = self._cursor, self._cursor + 1
          self._q, self._a = qs[row_index % len(qs)], ans[row_index % len(ans)]
          return self.prompt_builder(self._q), {}

      def step(self, action):
          return "", float(self.reward_fn(action, self._a, self._q)), True, False, {}

  env_factory = lambda: RolloutHarness.local(
      PromptDataset(
          questions=["2+2?", "Capital of France?"],
          answers=["4", "Paris"],
          reward_fn=reward_fn,
          prompt_builder=lambda question: f"Q: {question}\nA:",
          test_questions=["3+3?"],
          test_answers=["6"],
      ),
      tokenizer,
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

  # 2) Multi-turn text environments (each rollout drives its own in-process env)
  class ToyRolloutEnv:
      def reset(self, seed=None):
          del seed
          return "Start: What is 2+2?", {}

      def step(self, action: str):
          reward = 1.0 if "4" in action else 0.0
          return "Done.", reward, True, False, {"correct": bool(reward)}

  env_factory = lambda: RolloutHarness.local(
      ToyRolloutEnv(),
      tokenizer,
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

Parameters
----------

.. autoclass:: agilerl.algorithms.cispo.CISPO
  :members:
  :inherited-members:
