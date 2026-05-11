.. _fused_logprobs:

Fused linear log-prob optimizations
=================================

When you train an LLM with reinforcement learning, the model still has to turn
each hidden vector into a score for every vocabulary token. That intermediate
usually has shape ``(batch, sequence length, vocab size)``. For large vocabs (typically >100k)
that tensor alone can dominate GPU memory, even though many algorithms only
need the log-probability of the **token that was actually chosen** at each
position—a much smaller ``(batch, sequence length)`` result.

AgileRL offers two **optional** speed/memory switches. Both default to
``False``. They are meant to implement the same training objective as the
standard code, up to normal floating-point differences.

.. list-table::
   :header-rows: 1
   :widths: 28 38 34

   * - Flag
     - What it does
     - When it runs
   * - ``use_fused_linear_logprobs``
     - Chunked ``lm_head`` + log-softmax + gather: never stores the full
       ``(..., vocab)`` logits tensor at once.
     - Rollout-side work only (e.g. "old" and reference log-probs) when
       gradients are off—no impact on how the policy loss backprops.
   * - ``use_liger_loss``
     - Fused chunking for the **policy / KL part of the loss**, including
       backward through ``lm_head``, using `Liger Kernel
       <https://github.com/linkedin/Liger-Kernel>`_ primitives under the hood.
     - While the loss is being differentiated (PPO, REINFORCE, GRPO, CISPO,
       GSPO family).

``use_fused_linear_logprobs`` is pure AgileRL code and does **not** require
``liger-kernel``. ``use_liger_loss`` **does** require ``liger-kernel``; if it is
missing you get a warning and the flag is turned off. If you use ``use_liger_loss``
with LoRA, ``lm_head`` is excluded from LoRA adapters (with a warning) because
the fused kernel expects a single full head weight matrix.

``cast_logprobs_to_fp32`` (on ``LLMAlgorithm``, default ``True``) controls
whether the **chunked log-prob reductions** in the standard and
``use_fused_linear_logprobs`` paths run the numerically sensitive ``logsumexp`` /
gather steps in fp32, then cast back. That stays closer to a textbook
``log_softmax`` and keeps rollout vs. trainer log-probs consistent. Setting it
to ``False`` can save a bit of work but may shift log-probs enough to change
importance-sampling ratios on large vocabs in low precision (e.g. bf16). The
Liger gradient-time kernels use their own fused math and **ignore** this flag.

Usage
-----

.. code-block:: python

    from agilerl.algorithms import GRPO, CISPO, GSPO, LLMPPO, LLMREINFORCE

    agent = GRPO(..., use_fused_linear_logprobs=True)   # no-grad rollout path
    agent = LLMPPO(..., use_liger_loss=True)             # gradient-time policy loss

    agent = GRPO(
        ...,
        use_liger_loss=True,
        use_fused_linear_logprobs=True,
    )

Tiny batches (only a few hundred tokens total) may not see much benefit from
chunking and can even be slightly slower; very large sequences may still run out
of memory for non-vocabulary reasons (attention, backbone activations).

Example: what changes in memory?
--------------------------------

Illustrative peak **workspace** for the vocabulary projection only: same batch,
sequence, and vocab, comparing storing full logits once versus fusing/chunking so
only a thin slice of vocab scores exists at a time. Numbers are
order-of-magnitude; real runs add the rest of the model on top.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Setting
     - Dominant temporary tensor
     - Rough size (bf16) for ``B=8``, ``T=2048``, ``V≈152k``
   * - Standard ``lm_head``
     - Logits ``(8, 2048, V)``
     - ~5 GB for that tensor alone
   * - ``use_fused_linear_logprobs`` (chunked)
     - One chunk of logits ``(chunk_rows, V)`` at a time (e.g. chunk_rows ≈ 1024)
     - ~0.3 GB peak for that slice (≈10–50× smaller slice, depending on chunk size)
