.. _fused_logprobs:

Fused linear log-prob optimizations
=================================

When you train an LLM with reinforcement learning, the model still has to turn
each hidden vector into a score for every vocabulary token. That intermediate
usually has shape ``(batch, sequence length, vocab size)``. For large vocabs (typically >100k)
that tensor alone can dominate GPU memory, even though many algorithms only
need the log-probability of the **token that was actually chosen** at each
position—a much smaller ``(batch, sequence length)`` result.

AgileRL offers two **optional** speed/memory switches. They are meant to
implement the same training objective as the standard code, up to normal
floating-point differences.

* ``use_liger_loss`` defaults to ``None``, which resolves to ``True`` when
  ``liger-kernel`` is importable and ``False`` otherwise. Pass ``False``
  explicitly to force the standard path even when Liger is installed.
* ``use_fused_linear_logprobs`` defaults to ``None``, which resolves to the
  same value as the resolved ``use_liger_loss``. Without Liger the
  gradient-time path already materializes the full ``(B, T, V)`` logits, so
  fusing the no-grad rollout alone wouldn't lower overall peak — and would
  force the model to expose a discoverable ``lm_head``. Pass ``True``
  explicitly to fuse only the rollout side without enabling Liger.

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
``liger-kernel``. ``use_liger_loss`` **does** require ``liger-kernel``; if you
explicitly pass ``True`` without it installed you get a warning and the flag
is turned off. Leaving it at the default ``None`` simply opts out of the
fused loss when Liger is missing — no warning. If you use ``use_liger_loss``
with LoRA, ``lm_head`` is excluded from LoRA adapters (with a warning) because
the fused kernel expects a single full head weight matrix.

``cast_logprobs_to_fp32`` (on ``LLMAlgorithm``) controls whether the
**chunked log-prob reductions** in the standard and
``use_fused_linear_logprobs`` paths run the numerically sensitive
``logsumexp`` / gather steps in fp32, then cast back. It defaults to ``None``,
which resolves to the same value as the resolved ``use_liger_loss``: Liger
users get fp32 (consistent with Liger's own gradient-time math and cheap
because the fused rollout workspace is small), while non-Liger users get
bf16 to avoid a ~10 GB fp32 chunk workspace landing on top of the full
``(B, T, V)`` bf16 logits on the unfused grad path. Set ``True`` explicitly
if you want fp32 precision and have the memory headroom; set ``False`` to
force bf16 even with Liger. Note that the Liger gradient-time kernels use
their own fused math and **ignore** this flag for the loss backward — it
only governs the rollout-side log-prob reductions.

Usage
-----

.. code-block:: python

    from agilerl.algorithms import GRPO, CISPO, GSPO, LLMPPO, LLMREINFORCE

    # Both fused paths are on by default when liger-kernel is installed.
    agent = GRPO(...)

    # Force the standard PyTorch paths even if Liger is available.
    agent = GRPO(
        ...,
        use_liger_loss=False,
        use_fused_linear_logprobs=False,
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
