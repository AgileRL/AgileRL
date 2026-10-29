.. _fused_logprobs:

Memory-efficient log-probabilities
==================================

Reinforcement-learning algorithms for LLMs are driven by **log-probabilities**:
at every position in a sequence, how likely did the model think the token it
actually produced was? Comparing those numbers between the current policy and an
older or reference policy is what tells the algorithm which way to adjust the
weights.

Computing log-probabilities the obvious way is surprisingly memory-hungry.
AgileRL does it in a way that keeps the GPU memory cost low, automatically, with
nothing to switch on. This page explains why the naive way is expensive, what
AgileRL does instead, and the few settings you can change.

A few terms
-----------

* **Token**: a small chunk of text (roughly a word-piece) that the model reads
  and generates one at a time.
* **Vocabulary** (``V``): the full set of tokens the model can choose from.
  Modern models have large vocabularies, often well over 100,000 tokens.
* **Hidden state**: the model's internal vector for a position, produced by the
  transformer just before the final output layer.
* **lm_head**: the final output layer, a single large matrix that turns one
  hidden state into one **logit** (a raw, unnormalised score) for every token in
  the vocabulary.
* **Log-probability** (log-prob): the logits turned into probabilities and then
  log-transformed. An RL algorithm only needs the log-prob of the *one* token
  that was actually chosen at each position.

The problem: scoring the whole vocabulary is huge
-------------------------------------------------

To get a log-probability the model has to score every possible next token. Done
naively, that produces a **logits tensor** of shape
``(batch, sequence length, vocabulary)``, often written ``(B, T, V)``, holding
one number for every token at every position.

With a large vocabulary that single temporary tensor can dominate GPU memory. For
a batch of 8, sequences of 2048 tokens, and a 152k-token vocabulary it is about
**5 GB on its own**, on top of the model itself. Yet the algorithm throws almost
all of it away: it keeps only the log-prob of the token that was actually chosen
at each position, a result of shape ``(B, T)``, smaller by a factor of the
vocabulary size (here over 100,000×).

How AgileRL avoids it
---------------------

AgileRL never builds the full ``(B, T, V)`` tensor. Instead it:

#. Stops the model's forward pass at the hidden states (it temporarily replaces
   ``lm_head`` with a pass-through, so the model hands back hidden states instead
   of logits), and then
#. Computes the per-token log-probs itself with a **chunked matrix multiply**:
   it multiplies a few positions at a time by the ``lm_head`` weight, so only a
   thin ``(chunk_rows, V)`` slice of scores ever exists at once.

This is always on (there is no flag to enable or disable it) and it keeps
memory bounded in both directions:

* **Rollout (no gradients).** The "old-policy" and reference log-probs are
  computed this way.
* **Training (with gradients).** Computing gradients would normally require
  keeping the logits around for the backward pass. Instead AgileRL recomputes
  each chunk's logits on demand during the backward pass (a standard trick
  called *gradient checkpointing*), so the memory spike stays bounded while
  training too.

Because ``lm_head`` is used directly as a weight in this matrix multiply, it is
kept out of the LoRA adapters and (when you quantize the model) left
unquantized, so the manual multiply stays numerically exact.

Speed
-----

The per-chunk matrix multiply and its ``log_softmax`` (the step that normalises
logits into log-probs) are compiled with ``torch.compile`` into fast GPU kernels
on the first CUDA call. When compilation isn't available (on CPU or Apple MPS,
without Triton, or on an unsupported backend), AgileRL automatically falls back
to a plain ("eager") PyTorch path. To force the eager path explicitly:

.. code-block:: python

    from agilerl.algorithms.core.llm_ops import fused_logprobs

    fused_logprobs._FUSED_LOGPROB_COMPILE_STATE["disabled"] = True

How many positions go in each chunk (``chunk_rows``) is chosen automatically from
the vocabulary size, so that one ``(chunk_rows, V)`` slab stays near a fixed
memory budget (256 MiB of fp32 logits): models with bigger vocabularies get fewer rows
per chunk, smaller ones more
(:func:`~agilerl.arena.memory.resolve.chunk.resolve_chunk_rows`).

The fused Liger loss
--------------------

When the ``liger-kernel`` package is installed, AgileRL fuses the **policy/KL
part of the loss** (including the backward pass through ``lm_head``) into
`Liger Kernel <https://github.com/linkedin/Liger-Kernel>`_ Triton primitives
wherever the kernel can express the objective. There is no switch: the rule in
:func:`~agilerl.arena.memory.resolve.liger.liger_loss_blocker` decides, and
the memory estimator prices the path that runs. The update falls back to the
standard path for KL advantage shaping, a final-logit softcap the kernel
cannot apply, and (for GRPO, CISPO and GSPO) vLLM sampling supports.

How the kernel chunks depends on the **importance-sampling level**:

* **Token level** (GRPO, CISPO, PPO, REINFORCE). Each token is scored
  independently, so the kernel chunks by token: every chunk materializes only
  ``(token_chunk, V)`` logits, sized by ``chunk_rows``.
* **Turn and trajectory level** (PPO, REINFORCE; GRPO and GSPO at trajectory
  level). These pool the tokens of a turn or completion, so a token chunk would
  see only part of the pooled unit. PPO and REINFORCE chunk one whole sequence
  at a time and hold its ``(sequence length, V)`` fp32 logits per chunk; GRPO's
  trajectory kernel tiles each sequence as token level does. Updates that weight
  tokens by vLLM sampling log-probs (the vLLM correction, GRPO's off-policy
  masks) run the standard path above token level, as do CISPO, turn-level GRPO
  and per-turn GRPO advantages at trajectory level.

The standard path is always fused-linear and memory-bounded.

Precision
---------

The log-prob reductions (the ``matmul``, ``gather`` and ``logsumexp`` steps)
always run in fp32 before casting back to the model's dtype. Because the working
set is only one ``(chunk_rows, V)`` slice, fp32 is cheap here and keeps
importance-sampling ratios free of bf16 quantisation error. The Liger
gradient-time kernels use their own internal math for the loss backward.

Usage
-----

.. code-block:: python

    from agilerl.algorithms import GRPO, CISPO, GSPO, LLMPPO, LLMREINFORCE

    # Memory-bounded fused log-probs are always on. The fused Liger loss runs
    # whenever liger-kernel is installed and the objective allows it.
    agent = GRPO(...)
    agent.liger_loss  # True when the update runs the fused Liger loss

Example: what changes in memory?
--------------------------------

The table below isolates the **vocabulary projection** (the step described above)
for the same batch, sequence length, and vocabulary. It compares storing the full
logits once against chunking so that only a thin slice of scores exists at a time.
The numbers are order-of-magnitude; a real run adds the rest of the model on top.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Setting
     - Largest temporary tensor
     - Rough size (bf16) for ``B=8``, ``T=2048``, ``V≈152k``
   * - Full ``lm_head`` logits (not used by AgileRL)
     - Logits ``(8, 2048, V)``
     - ~5 GB for that tensor alone
   * - Chunked fused log-probs (always on)
     - One chunk of logits ``(chunk_rows, V)`` at a time
     - ~0.3 GB peak for that slice (≈10–50× smaller, depending on chunk size)
