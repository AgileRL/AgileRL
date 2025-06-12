.. _llm_finetuning:

LLM Finetuning
==============

Reinforcement learning (RL) has emerged as a powerful technique for improving reasoning capabilities in Large Language Models.
Models like DeepSeek-R1 and OpenAI's o1 exemplify this approach, demonstrating how RL can be used to develop LLMs with superior
reasoning abilities without relying on traditional supervised fine-tuning. Through training with reinforcement learning, models
develop *agency* and can be described as **agents**.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - **Algorithms**
     - **Tutorials**
   * - :ref:`GRPO<grpo>`
     - :ref:`LLM reasoning with GRPO<grpo_tutorial>`


.. _rl_for_reasoning:

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

This example demonstrates how to use the GRPO algorithm to finetune a LLM on a reasoning task.

.. collapse:: Example

  .. literalinclude:: ../../tutorials/LLM_Finetuning/grpo_reasoning.py
      :language: python
