.. _fused_logprobs:

Fused linear log-prob optimizations
=================================

When you train an LLM with reinforcement learning, the model still has to turn
each hidden vector into a score for every vocabulary token. That intermediate
usually has shape ``(batch, sequence length, vocab size)``. For large vocabs
(typically >100k) that tensor alone can dominate GPU memory, even though most
RL algorithms only need the log-probability of the **token that was actually
chosen** at each position — a much smaller ``(batch, sequence length)`` result.

AgileRL never materializes the full ``(B, T, V)`` logits tensor for log-prob
computation. Instead it identity-patches ``lm_head`` so the model forward
returns hidden states, then computes per-token log-probs with a **chunked
matmul** over the ``lm_head`` weight, bounding the transient logits workspace
to ``(chunk_rows, V)``. This is unconditional — there is no flag to turn it on
or off — and it applies to both:

* the no-grad rollout side (old-policy and reference log-probs), and
* the gradient-time policy log-probs: the chunked matmul is routed through a
  gradient-checkpointed autograd Function that recomputes each chunk's logits
  in the backward pass, so the spike stays bounded in **both** directions.

Because ``lm_head`` is needed as a standalone weight for this matmul, it is
excluded from LoRA adapters and (when quantizing) kept unquantized so the
manual matmul stays exact.

Speed
-----

The per-chunk matmul + ``log_softmax`` reduction is compiled with
``torch.compile`` (Triton kernels) on the first CUDA call, with an automatic
eager fallback when compilation is unavailable (CPU/MPS, no Triton, or an
unsupported backend). Set ``AGILERL_DISABLE_FUSED_COMPILE=1`` to force the
eager path.

The chunk size (``chunk_rows`` of the flattened ``(B*T)`` workspace) is
**vocab-aware**: it is sized so one fp32 ``(chunk_rows, V)`` slab stays near a
fixed byte budget (large-vocab models get fewer rows per chunk, small-vocab
models more). Override it with ``AGILERL_FUSED_LOGPROBS_CHUNK_ROWS``.

The Liger loss
--------------

``use_liger_loss`` (default ``False``) is the one remaining loss switch. It
fuses the **policy / KL part of the loss**, including the backward through
``lm_head``, into `Liger Kernel <https://github.com/linkedin/Liger-Kernel>`_
Triton primitives — a single fused pass that is faster than the standard path
when it applies. It requires ``liger-kernel``; passing ``True`` without it
installed warns and falls back to ``False``.

Memory behavior of the Liger path depends on the importance-sampling level:

* **Token-level** GRPO / CISPO / PPO / REINFORCE: the hidden states are
  token-flattened to ``(B*T, 1, H)`` so the fused kernel chunks **tokens** —
  each chunk materializes only ``(token_chunk, vocab)`` logits. Bounded.
  The ``liger_token_chunk_size`` constructor argument (default 2048) sets the
  tokens-per-chunk.
* **Turn- and sequence-level** (e.g. GSPO): pooling couples a turn/sequence's
  tokens, so a token chunk would only see part of the pooled unit — the
  flatten trick cannot apply. The fused kernel processes one whole sequence
  per chunk and materializes ``(seq_len, vocab)`` per trajectory, which is
  **not** memory-bounded at long context. AgileRL warns and, where it can,
  routes these to the standard path; set ``use_liger_loss=False`` for bounded
  memory (the standard path is always fused-linear/bounded).

So the rule of thumb: keep ``use_liger_loss=True`` for token-level objectives
(fastest, bounded); use ``use_liger_loss=False`` for turn-/sequence-level
objectives at long context.

Precision
---------

``cast_logprobs_to_fp32`` (on ``LLMAlgorithm``, default ``True``) controls
whether the chunked log-prob reductions (``gather`` / ``logsumexp``) run in
fp32 before casting back to the input dtype. Because the workspace is only a
single ``(chunk_rows, V)`` slab, fp32 is cheap. The Liger gradient-time
kernels use their own fused math and **ignore** this flag for the loss
backward — it only governs the log-prob reductions.

Usage
-----

.. code-block:: python

    from agilerl.algorithms import GRPO, CISPO, GSPO, LLMPPO, LLMREINFORCE

    # Bounded fused log-probs are always on. Liger is opt-in for the loss.
    agent = GRPO(...)                    # standard (always-bounded) loss path
    agent = GRPO(..., use_liger_loss=True)  # fused Liger loss (token-level bounded)

Example: what changes in memory?
--------------------------------

Illustrative peak **workspace** for the vocabulary projection only: same batch,
sequence, and vocab, comparing storing full logits once versus chunking so only
a thin slice of vocab scores exists at a time. Numbers are order-of-magnitude;
real runs add the rest of the model on top.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Setting
     - Dominant temporary tensor
     - Rough size (bf16) for ``B=8``, ``T=2048``, ``V≈152k``
   * - Full ``lm_head`` logits (not used by AgileRL)
     - Logits ``(8, 2048, V)``
     - ~5 GB for that tensor alone
   * - Chunked fused log-probs (always on)
     - One chunk of logits ``(chunk_rows, V)`` at a time
     - ~0.3 GB peak for that slice (≈10–50× smaller, depending on chunk size)
