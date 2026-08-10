.. _llm_finetuning:

LLM Fine-Tuning
===============

Reinforcement learning (RL) has emerged as a powerful technique for improving reasoning capabilities in Large Language Models.
Reinforcement learning has been used in LLM post-training for several years now, initially through techniques like RLHF (Reinforcement Learning from Human Feedback)
which leverages human preferences to guide LLM responses, RLAIF (Reinforcement Learning from AI Feedback) which leverages AI feedback to guide LLM responses, and
more recently through RLVR (Reinforcement Learning with Verifiable Rewards) a technique that uses ground truth answers to score LLM responses and leads to the
development of reasoning capabilities. Models like DeepSeek-R1 and OpenAI's o1 exemplify this approach, demonstrating how RL can be used to develop LLMs with superior
reasoning abilities without relying on traditional supervised fine-tuning. Through training with reinforcement learning, models
develop *agency* and can be described as **agents**.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`GRPO<grpo>`
     - :ref:`LLM reasoning with GRPO<grpo_tutorial>`
   * - :ref:`GRPO<grpo>` (with evolutionary HPO)
     - :ref:`LLM Finetuning with HPO<llm_finetuning_hpo>`
   * - :ref:`CISPO<cispo>` and :ref:`GSPO<gspo>`
     - GRPO specializations with alternative loss objectives (see the :ref:`GRPO tutorial<grpo_tutorial>`).
   * - :ref:`LLM PPO<llmppo>`, :ref:`LLM REINFORCE<llmreinforce>` and :ref:`GRPO<grpo>`
     - :ref:`Multi-turn finetuning with LLMPPO, LLMREINFORCE, and GRPO<multiturn_grpo_ppo_tutorial>`
   * - :ref:`SFT<sft>` and :ref:`DPO<dpo>`
     - :ref:`LLM fine-tuning with SFT and DPO<sft_dpo_finetuning>`

.. _rl_for_reasoning:
.. _rlhf:

Reinforcement Learning for Reasoning
------------------------------------

The standard approach to creating instruction-following LLMs has traditionally relied on Supervised Fine-Tuning,
where models are trained on high-quality human-generated examples. However, this method has limitations when it comes to complex
reasoning tasks. What makes reinforcement learning particularly effective for enhancing reasoning is that it:

  #. **Rewards the process, not just the outcome:** By designing reward mechanisms that value step-by-step thinking and self-correction
  #. **Allows for exploration:** Models can try different reasoning approaches and learn which ones lead to better outcomes
  #. **Enables self-improvement cycles:** Creating a virtuous loop where better reasoning leads to better rewards

What makes this approach powerful is that the model discovers effective reasoning strategies on its own. It might learn to:

* Break complex problems into manageable steps
* Double-check calculations along the way
* Backtrack when it encounters contradictions
* Generate structural outlines before diving into details
* Verify final answers by working backward

These are called *emergent behaviours*.

The agent receives no explicit instructions on which specific reasoning techniques to employ. It learns through trial and error which approaches
tend to produce correct answers. This allows the emergence of sophisticated reasoning patterns that weren't necessarily anticipated
by the model's creators, similar to how `AlphaGo <https://deepmind.google/research/projects/alphago/>`_ discovered novel chess strategies through self-play.

This example demonstrates how to use the GRPO algorithm to fine-tune a LLM on a reasoning task.

.. collapse:: Example

  .. literalinclude:: ../../tutorials/llm_finetuning/grpo_reasoning.py
      :language: python

.. toctree::
   :hidden:

   batch_sizing
   fused_logprobs
   quantization
   llm_checkpoints

.. seealso::

   :doc:`batch_sizing`
      Rollout, mini- and micro-batch sizes, and how often the optimizer steps.

   :doc:`fused_logprobs`
      Fused linear log-probability computation for memory-efficient training.

   :doc:`llm_checkpoints`
      Saving and loading LLM checkpoints during fine-tuning.

.. tutorial::

   :ref:`grpo_tutorial`
      GRPO for LLM reasoning tasks.

   :ref:`sft_dpo_finetuning`
      Supervised fine-tuning and DPO.

   :ref:`llm_finetuning_hpo`
      Evolutionary HPO for GRPO fine-tuning.

   :ref:`multiturn_grpo_ppo_tutorial`
      Multi-turn GRPO and PPO for LLMs.

.. note::

   Colocated rollouts (where the trainer and vLLM share a single GPU and hand it
   back and forth using vLLM's native sleep/wake) are covered for users under
   "Colocated rollout (native vLLM sleep/wake)" in the :doc:`quantization` topic.
   A deeper, code-level walkthrough of the same mechanism lives in
   ``docs/llm_finetuning/vllm_sleep_handoff.md`` (a developer note kept in the
   repository, not built into this documentation site).
