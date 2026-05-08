.. _fused_logprobs:

Fused Linear Log-Prob Optimizations
===================================

LLM RL training spends most of its memory budget on a single intermediate
tensor: the per-token vocabulary logits of shape ``(B, T, V)``. For
Qwen2.5-3B (``V = 152k``) at ``B = 8, T = 2048`` in bf16 that is ~5 GB on its
own; PPO doubles it because the actor and critic each run their own forward
through the body, and autograd may need to keep activations across the whole
chain for backward. The full vocabulary tensor is rarely what we actually
need — we only need the log-probability of the *chosen* token at each
position, a tensor of shape ``(B, T)``.

AgileRL ships two opt-in paths that compute that ``(B, T)`` tensor without
ever materializing the full ``(B, T, V)`` intermediate.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - flag
     - what it fuses
     - where it applies
   * - ``use_fused_linear_logprobs``
     - ``lm_head`` matmul + log-softmax + gather, chunked
     - no-grad rollout-side log-prob computation (old-policy + reference)
   * - ``use_liger_loss``
     - ``lm_head`` matmul + log-softmax + gather + policy clip + KL +
       reduction, chunked, fused forward + backward
     - gradient-time policy + KL loss (PPO, LLMREINFORCE, GRPO, CISPO,
       GSPO)

Both default to ``False`` — turning them on is a memory/speed knob and does
not change the loss function or the gradient direction.

Why "fused linear log-prob"?
----------------------------

* **Linear** refers to ``lm_head``, which is a :class:`torch.nn.Linear`
  projecting hidden states ``(B, T, H)`` to vocabulary logits ``(B, T, V)``.
* **Log-prob** refers to the per-token log-probability of the chosen
  action token — what PPO/GRPO/REINFORCE actually consume.
* **Fused** means combining the matmul, the ``log_softmax``, and the
  ``gather`` into a single chunked operation so the ``(B, T, V)``
  intermediate never exists in memory all at once.

The unfused pipeline materializes the full vocab tensor:

.. code-block:: text

    hidden (B, T, H)
        │  lm_head matmul
        ▼
    logits (B, T, V)              ← peak memory lives here
        │  log_softmax
        ▼
    log_probs (B, T, V)
        │  gather chosen tokens
        ▼
    per-token logprob (B, T)      ← all we actually need

The fused pipeline tiles over rows of the flattened ``(B*T, H)`` input and
processes them ``_chunk_rows`` at a time:

.. code-block:: text

    for chunk of (rows, H) in (B*T, H):
        chunk_logits  = chunk @ lm_head.weight.T          # (rows, V)
        log_softmax → gather chosen token                 # (rows,)
        write into output, free the chunk

Peak workspace drops from ``(B, T, V)`` to ``(_chunk_rows, V)`` — typically
30–50× smaller. The result is byte-identical (or within bf16 quantisation
noise when ``cast_to_fp32=False``) to the unfused path.

No-grad path: ``use_fused_linear_logprobs``
-------------------------------------------

This applies to the *rollout-side* log-prob computation that runs once per
:meth:`~agilerl.algorithms.grpo.GRPO.learn` step under
:func:`torch.inference_mode` to produce ``old_log_probs`` and
``reference_log_probs``. The kernel is a plain Python function — no
gradients ever flow through it, so we just chunk the matmul, gather, and
free.

Hooking it into the model's forward uses a small monkey-patch trick. We
swap ``lm_head`` for :class:`torch.nn.Identity` for the duration of the
no-grad call, so ``output.logits`` is now the post-final-norm hidden state.
We feed that hidden state, plus the saved-aside ``lm_head.weight``, into
the chunked kernel. The Identity-patch is restored on context-manager exit
(including on exception).

Usage:

.. code-block:: python

    from agilerl.algorithms import GRPO

    agent = GRPO(
        ...,
        use_fused_linear_logprobs=True,   # default False
    )

The flag is gated on ``not torch.is_grad_enabled()`` — gradient-time call
sites (``_loss``) keep the unfused path automatically. Default-off so
existing behaviour is byte-identical.

Gradient-time path: ``use_liger_loss``
--------------------------------------

This is the harder case. The gradient-time policy loss needs gradients to
flow back through the matmul into the LoRA adapters and ``lm_head``.
"Chunk the forward and free intermediates" doesn't work — autograd would
still record a graph that needs the full ``(B, T, V)`` logits for
backward.

The trick is to absorb the entire
``{matmul → log_softmax → gather → clipped policy ratio → KL → reduce}``
chain into a single :class:`torch.autograd.Function` whose forward returns
just the **scalar loss** and whose backward returns gradients computed
chunk-by-chunk inside the forward. Because the user only sees the scalar
loss, autograd has no reason to retain the ``(B, T, V)`` intermediate.
Internally:

1. The forward iterates over chunks of the flattened ``(B*T, H)`` hidden
   state.
2. For each chunk, :func:`torch.func.grad_and_value` computes the loss
   contribution **and** its gradient with respect to the chunk's hidden
   state and the shared ``lm_head.weight`` in one pass.
3. Gradients are accumulated into pre-allocated ``grad_input`` and
   ``grad_weight`` buffers; the chunk's ``(chunk, V)`` intermediates are
   freed.
4. The forward returns the scalar loss; the backward just hands back the
   accumulated gradients.

So instead of the standard graph

.. code-block:: text

    hidden ── matmul ── logits ── log_softmax ── gather ── clip ── max ── mean ── loss
                            ▲
                  must save all of this for backward

you get

.. code-block:: text

    hidden ────────[fused chunked Function]────────  loss (scalar)
           chunks {matmul, log_softmax, gather, clip, max, KL, reduce, gradient}
           per-chunk, accumulating grads into output buffers, freeing each chunk

The autograd graph as far as the rest of the program is concerned is just
``hidden → loss``. No ``(B, T, V)`` lives anywhere across the
forward+backward boundary. This is why the savings on the gradient-time
path are larger than on the no-grad path — it eliminates the autograd-
saved tensors as well as the forward intermediate.

This builds on top of the
`Liger Kernel <https://github.com/linkedin/Liger-Kernel>`_ project's
:class:`LigerFusedLinearPPOBase`, which provides the chunked
forward+backward scaffolding (the ``torch.func.grad_and_value`` machinery,
the chunk loop, the gradient accumulation buffers). The actual loss math
lives in a per-algorithm subclass.

Per-algorithm dispatch
~~~~~~~~~~~~~~~~~~~~~~

Different RL algorithms feed advantages into the policy gradient with
different shapes. There are three categories:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - advantage shape
     - algorithms
     - Liger function
   * - ``(B,)`` per-trajectory scalar
     - GRPO, CISPO, GSPO
     - ``LigerFusedLinearGRPOFunction`` (Liger)
   * - ``(B, T)`` per-token (GAE / per-token ReBN)
     - PPO, REINFORCE
     - ``LigerFusedLinearLLMPPOFunction`` (this PR)
   * - ``(B, max_turns)`` per-turn
     - turn-PPO with ``turn_level_clip=True``
     - ``LigerFusedLinearLLMPPOFunction``, turn-mode branch

A note on naming: Liger's :class:`LigerFusedLinearGRPOFunction` exposes
an ``importance_sampling_level`` knob that takes ``"token"`` or
``"sequence"``. This is **not** about advantage shape — it controls how
the importance-sampling *ratio* is computed (per-token
``exp(log_pi_t − log_pi_old_t)`` vs per-sequence
``exp(mean_t(log_pi_t − log_pi_old_t))``, which is GSPO). Either way the
advantage is still ``(B,)`` and broadcast across all tokens of the
trajectory. PPO with GAE genuinely needs *different* advantages at
different token positions, which is why ``LigerFusedLinearGRPOFunction``
cannot be reused — its ``advantages.unsqueeze(1)`` is hardcoded to
broadcast a per-trajectory scalar.

PPO and REINFORCE specifics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **LLMPPO** runs two body forwards per minibatch step (mirroring
  :meth:`_fused_forward`): an actor pass under
  ``select_adapter("actor")`` whose ``lm_head`` input is captured via a
  forward-pre-hook and fed into the fused Function; a critic pass under
  ``select_adapter("critic")`` whose value head produces ``(B, T)``
  values that are reduced into the (unfused) clipped value loss outside
  the fusion. Token + turn granularity are both supported. Turn mode
  scatter-adds token log-ratios into ``(B, max_turns)`` per-turn
  log-ratios and applies clipping at the turn level.
* **LLMREINFORCE** has no value head. KL is logged as a metric only
  because REINFORCE folds the KL penalty into the advantage upstream
  (see :meth:`_compute_rebn_advantages`), so the gradient-time loss runs
  with ``beta=0`` — the fused Function returns a pure clipped policy
  gradient. REINFORCE's ``"turn"`` granularity affects only how the
  upstream advantage is computed (per-turn ReBN broadcast back to
  per-token); the loss itself is always per-token.

GRPO, CISPO, GSPO specifics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

These three reuse Liger's stock :class:`LigerFusedLinearGRPOFunction`
(``(B,)`` scalar advantages) and select between them by setting
``loss_type``:

* ``loss_type="grpo"`` → Liger ``loss_type="grpo"``,
  ``importance_sampling_level="token"``.
* ``loss_type="cispo"`` → Liger ``loss_type="cispo"``,
  ``importance_sampling_level="token"``. CISPO clamps importance
  weights against an *absolute* upper bound, so internally
  ``epsilon_high = clip_coef_max`` rather than the offset
  ``clip_coef_max - 1.0`` used by GRPO/GSPO.
* ``loss_type="gspo"`` → Liger ``loss_type="grpo"``,
  ``importance_sampling_level="sequence"``. GSPO is GRPO-style
  clipping applied at the sequence level (geometric-mean ratio across
  the trajectory's tokens).

Usage
-----

.. code-block:: python

    from agilerl.algorithms import GRPO, CISPO, GSPO, LLMPPO, LLMREINFORCE

    # GRPO / CISPO / GSPO — fused gradient-time loss
    agent = GRPO(..., use_liger_loss=True)
    agent = CISPO(..., use_liger_loss=True)
    agent = GSPO(..., use_liger_loss=True)

    # PPO / REINFORCE — fused gradient-time loss
    agent = LLMPPO(..., use_liger_loss=True)
    agent = LLMREINFORCE(..., use_liger_loss=True)

    # Either flag can be combined with the no-grad fused log-prob path:
    agent = GRPO(
        ...,
        use_liger_loss=True,
        use_fused_linear_logprobs=True,
    )

Both flags require ``liger-kernel`` to be installed. With
``use_liger_loss=True`` and ``liger-kernel`` missing, a ``UserWarning``
is emitted at construction and the flag is forced back to ``False``;
training proceeds on the unfused path.

When ``use_liger_loss=True`` is set with LoRA, ``lm_head`` is excluded
from the LoRA target modules (a one-time warning is emitted). The fused
kernel needs the full ``lm_head`` weight tensor and the LoRA decomposition
on the head would force re-materializing the full output during backward.

Memory savings on A100-40GB / Qwen2.5-3B
----------------------------------------

End-to-end gradient-step peak with ``use_liger_loss=True``, gradient
checkpointing on, ``B=2``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - algorithm
     - T=4096 unfused
     - T=4096 liger
     - T=8192 unfused
     - T=8192 liger
   * - GRPO
     - 30.27 GB
     - 15.85 GB
     - OOM
     - 25.20 GB
   * - CISPO
     - 30.28 GB
     - 15.85 GB
     - OOM
     - 25.20 GB
   * - GSPO
     - 30.28 GB
     - 15.85 GB
     - OOM
     - 25.20 GB
   * - PPO
     - OOM
     - 15.89 GB
     - OOM
     - 25.27 GB
   * - REINFORCE
     - 30.28 GB
     - 15.88 GB
     - OOM
     - 25.26 GB

PPO is consistently the heaviest because of the doubled-forward (actor
+ critic adapter on the same body); it OOMs unfused at ``T=4096`` where
the GRPO family and REINFORCE still fit. With Liger every algorithm
lands at the same ~16 GB / ~25 GB regardless of family — the dominant
remaining memory is body activations, which is shared. At ``T=16384``
all five algorithms OOM with Liger; the body activations + attention
scores are the next ceiling.

Isolated no-grad pass (``use_fused_linear_logprobs=True`` only) on the
same hardware:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - shape (B, T, V)
     - unfused peak
     - liger peak
     - savings
   * - (8, 512, 151643)
     - 8.23 GB
     - 6.52 GB
     - −1.71 GB (−21%)
   * - (8, 2048, 151643)
     - 15.18 GB
     - 7.18 GB
     - −8.00 GB (−53%)
   * - (32, 512, 151643)
     - 15.18 GB
     - 7.18 GB
     - −8.00 GB (−53%)

When *not* to enable
--------------------

* **`liger-kernel` not installed** — ``use_liger_loss=True`` will warn
  and fall back; ``use_fused_linear_logprobs=True`` works without it.
* **Tiny shapes** — at ``B*T < chunk_rows`` (default 1024) the no-grad
  path's per-chunk fp32 promotion can use slightly more memory than the
  unfused bf16 path. The wall-clock cost of the chunk loop also doesn't
  pay for itself below ~B*T = 2k tokens.
* **PPO with ``turn_level_clip=False``** — turn mode in this
  configuration uses token-level policy and per-turn value loss; both
  flags work but the savings ratio is the same as token mode.
