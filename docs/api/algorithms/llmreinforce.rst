.. _llmreinforce:

LLM REINFORCE
=============

`REINFORCE <https://doi.org/10.1007/BF00992696>`_ is the classic
score-function policy-gradient method. ``LLMREINFORCE`` brings this approach to
causal language model finetuning with turn-aware trajectories.

In AgileRL, the algorithm uses Return Batch Normalization (ReBN), as
popularized by the `GEM paper <https://arxiv.org/abs/2510.01051>`_, to
improve stability in practice:

* **Turn-level Monte Carlo returns:** discounted returns are computed across
  turns for each sampled trajectory.
* **Batch-normalized returns (ReBN):** turn returns are z-scored across valid
  ``(sample, turn)`` pairs before being broadcast to token-level advantages.
* **Value-head-free training:** unlike PPO-style actor-critic updates, this
  path optimizes the policy directly from normalized returns.

Variance Reduction
------------------

LLM policy-gradient algorithms differ mostly in *how* they reduce the variance
of the Monte Carlo return signal. Three families show up in this codebase:

* **Learned value baseline (:ref:`LLM PPO<llmppo>`)**: subtract a learned
  state-value estimate to form an advantage. Strong asymptotic variance
  reduction, but spends parameters and compute on a value head and is
  sensitive to value-function staleness.
* **Group-relative normalization (:ref:`GRPO<grpo>` and variants)**: sample
  a group of ``G`` rollouts per prompt and z-score their returns within the
  group. No critic to train; effective when rewards are sparse and rollouts
  are cheap, but the baseline degenerates as the group's returns collapse and
  it ties variance reduction to having a large group size.
* **Return Batch Normalization (this algorithm)**: z-score returns across
  every valid ``(sample, turn)`` pair in the batch. No critic and no group
  requirement, and it remains well-defined under arbitrary discount factors
  and per-step dense rewards (where group-relative normalization is
  awkward). The trade-off is that the baseline is global to the batch, so it
  reduces less variance than a state-conditioned critic on tasks where
  reward depends sharply on the prompt.

.. note::

   ReBN itself is a specific application of the long-standing "advantage
   normalization" trick (z-scoring the policy-gradient signal across a
   batch) that has been standard in PPO implementations since
   `OpenAI Baselines <https://github.com/openai/baselines>`_ and was
   systematically studied by
   `Engstrom et al. 2020 <https://arxiv.org/abs/2005.12729>`_ and
   `Andrychowicz et al. 2021 <https://arxiv.org/abs/2006.05990>`_. The
   `GEM paper <https://arxiv.org/abs/2510.01051>`_ names the specific
   variant that operates on per-transition Monte Carlo returns across the
   whole batch (rather than on GAE advantages or within a per-prompt group).

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

The typical training entry point is ``train_llm_rollout`` in
``agilerl.training.llm``. Single-turn reasoning is the ``max_turns=1``
case of the same function.

.. code-block:: python

  from agilerl.training.llm import train_llm_rollout
  from agilerl.llm_envs import RolloutHarness

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

Loading follows the standard Hugging Face ``from_pretrained`` flow for the base
model and any finetuned adapter.

Parameters
----------

.. autoclass:: agilerl.algorithms.reinforce_llm.REINFORCE
  :members:
  :inherited-members:
