.. _arena_reward_inputs:

Prompts and reward inputs
=========================

This page covers a rollout over dataset rows: ``env_type: rollout`` with a
``dataset`` and a ``prompt_template``. It says how Arena turns each row into the
prompt the model sees, and exactly which strings your reward function gets
back. Local training with :class:`~agilerl.training.trainer.LocalTrainer` uses
the same code, so a reward that works locally works on Arena.

How ``prompt_template`` becomes the prompt
------------------------------------------

For each row, Arena:

1. **Turns each key into one chat message.** The role is the text before the
   first ``_`` in the key: ``system_0`` is a system message, ``user_1`` a user
   message, ``assistant_2`` an assistant message.
2. **Orders the messages by the number at the end of each key.** Arena stores
   the manifest as JSON and does not keep key order, so number every key
   (``system_0``, ``user_1``, ...). Keys without a number are used in the order
   they arrive, which on Arena is alphabetical. Numbering only some keys, or
   repeating a number, is an error.
3. **Fills** ``{question}`` **and** ``{answer}`` from the row, after
   ``environment.columns`` has mapped your dataset's columns onto those names.
   No other placeholder is filled: any other brace, such as ``{"n": 1}`` or
   ``{other_column}``, stays in the text as written. Putting ``{answer}`` in the
   prompt shows the model the answer.
4. **Applies the model's chat template** with
   ``tokenizer.apply_chat_template(messages, add_generation_prompt=True, **chat_template_kwargs)``.
   ``chat_template_kwargs`` comes from the ``environment`` section of the manifest.

The chat template decides the rest:

- **System prompt.** There is one only if you add a ``system_*`` key, or the
  model's template adds its own when you don't. ``Qwen/Qwen2.5-0.5B-Instruct``
  adds ``You are Qwen, created by Alibaba Cloud. You are a helpful
  assistant.``; Qwen3.5 adds none.
- **Generation prompt and thinking.** The template writes the opening of the
  assistant turn. For Qwen3.5 that includes the thinking block, and the default
  depends on the size: ``Qwen/Qwen3.5-0.8B`` and ``Qwen/Qwen3.5-2B`` close it at
  once (``<think>\n\n</think>\n\n``, thinking off), while ``Qwen/Qwen3.5-4B``
  and ``Qwen/Qwen3.5-9B`` leave it open (``<think>\n``, thinking on). Set
  ``chat_template_kwargs: {enable_thinking: true}`` or ``false`` to choose.
- **Assistant prefill.** When the last message is an assistant message with
  text, the model continues that text instead of starting a new turn. The
  prefill is part of the prompt, not of the completion. Qwen3.5's template puts
  an empty thinking block before any assistant message, so a ``<think>`` prefill
  does not turn thinking on there; use ``enable_thinking: true``.

The rendered string is the prompt. Nothing else is added.

What the reward function receives
---------------------------------

Arena calls your reward function once per completion, with arguments by name:

.. code-block:: python

   def reward_fn(completion: str, answer, question) -> float: ...

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Argument
     - Value
   * - ``completion``
     - Everything the model generated in its turn, decoded with special
       tokens kept: reasoning, its end marker (``</think>``) and the answer.
       End-of-turn tokens at the end (``<|im_end|>``) are dropped and
       whitespace is stripped from both ends. It starts where the prompt ends,
       so text the prompt already holds, such as Qwen3.5's opening ``<think>\n``
       or an assistant prefill, is not in it.
   * - ``answer``
     - The row's answer column, unchanged (for GSM8K, the full worked answer
       ending ``#### 5``).
   * - ``question``
     - The row's question column, unchanged. Not the rendered prompt.

``include_reasoning`` does not change these: it decides what an env receives,
and what a ``Rubric`` class reads as ``TurnAction.message``. A ``Rubric`` class
reads the same whole turn as ``TurnAction.completion`` (see
:class:`~agilerl.llm_envs.rubrics.TurnAction`).

A completion counts as solved when its reward reaches the manifest's
``max_reward``.

Examples
--------

All three use this template and a row with ``question: What is 2 + 3?`` and
``answer: '#### 5'``:

.. code-block:: yaml

   prompt_template:
       system_0: Solve the problem. Put the final number in <answer></answer> tags.
       user_1: '{question}'

**Qwen3.5 with thinking on** (``Qwen/Qwen3.5-0.8B``,
``chat_template_kwargs: {enable_thinking: true}``):

.. code-block:: python

   prompt = (
       "<|im_start|>system\nSolve the problem. Put the final number in <answer></answer> tags.<|im_end|>\n"
       "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n"
       "<|im_start|>assistant\n<think>\n"
   )
   # The model generates "2 plus 3 is 5.\n</think>\n\n<answer>5</answer><|im_end|>"
   completion = "2 plus 3 is 5.\n</think>\n\n<answer>5</answer>"

**Qwen3.5 with thinking off** (``Qwen/Qwen3.5-0.8B`` with no
``chat_template_kwargs``, the same as ``enable_thinking: false``). The model may
still reason, but in plain text after the closed block:

.. code-block:: python

   prompt = (
       "<|im_start|>system\nSolve the problem. Put the final number in <answer></answer> tags.<|im_end|>\n"
       "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n"
       "<|im_start|>assistant\n<think>\n\n</think>\n\n"
   )
   # The model generates "2 plus 3 is 5.\n<answer>5</answer><|im_end|>"
   completion = "2 plus 3 is 5.\n<answer>5</answer>"

**A model with no thinking mode** (``Qwen/Qwen2.5-0.5B-Instruct``), taught
``<think>`` tags by its system prompt:

.. code-block:: python

   prompt = (
       "<|im_start|>system\nSolve the problem. Put the final number in <answer></answer> tags.<|im_end|>\n"
       "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n"
       "<|im_start|>assistant\n"
   )
   # The model generates "<think>2 plus 3 is 5.</think>\n<answer>5</answer><|im_end|>"
   completion = "<think>2 plus 3 is 5.</think>\n<answer>5</answer>"

Writing a reward that fits
--------------------------

- Check the answer after the reasoning ends, so a draft answer inside the
  reasoning does not count: take the text after the last ``</think>``, or the
  last ``<answer>`` match.
- Check the format against ``completion``, and match what your model and
  template produce. With Qwen3.5 thinking on, the opening ``<think>`` is in the
  prompt and ``</think>`` is followed by a blank line, so a pattern that needs
  ``<think>`` at the start of ``completion``, or exactly one newline after
  ``</think>``, never matches.
- Ask the model for any tags you check for. A reward that needs
  ``<answer>`` tags scores 0 when the prompt never asks for them.

Check a template before you submit
----------------------------------

Render the prompt locally with the same functions Arena uses:

.. code-block:: python

   from transformers import AutoTokenizer

   from agilerl.utils.env_utils import make_conversation_template
   from agilerl.utils.llm_utils import render_chat_template

   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
   messages = make_conversation_template(
       {"system_0": "Solve the problem.", "user_1": "{question}"}
   )
   print(
       render_chat_template(
           messages,
           tokenizer,
           chat_template_kwargs={"enable_thinking": True},
           question="What is 2 + 3?",
           answer="#### 5",
       )
   )

Then call your reward function with completions shaped like the examples above.
