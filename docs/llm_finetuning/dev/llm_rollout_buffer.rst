LLM rollout buffer
==================

.. note::

   **Developer / implementation note.** Internal detail of
   ``agilerl/components/llm_rollout_buffer.py``. Synchronous LLM training uses it
   automatically; there is nothing for users to configure.

``LLMRolloutBuffer`` is a bounded FIFO of rollout groups. It holds trajectory
tensors **by reference** — nothing is copied on the way in or on the way out —
and pads into rectangular batches once, at collate time.

Layout
------

The unit of storage and eviction is the **prompt group** (all ``group_size``
completions of one prompt). Groups are added and evicted whole, so the
group-divisibility the algorithms rely on holds by construction.

Three types make up the module:

- ``Trajectory`` — one episode: ``completion_ids`` ``(1, T)``, ``action_masks``
  and ``turn_ids`` ``(1, T - 1)``, per-turn ``rewards``, and optional
  ``sampling_logps``. Validated on construction, so a mask or turn-id tensor
  that is not exactly one shorter than the token ids fails immediately rather
  than misaligning a loss much later.
- ``RolloutGroup`` — exactly ``group_size`` trajectories.
- ``LLMExperienceBatch`` — the collated result: ragged ``completion_ids`` and
  ``action_masks`` that ``learn()`` pads itself, plus pre-stacked ``rewards``
  and ``turn_ids`` rectangles and ``completion_lengths``.

Once ``memory_size`` groups are held, admitting another evicts the oldest. If an
incoming group has a different ``group_size`` than the last one admitted — which
evolutionary HPO can cause mid-run — the buffer is cleared first, so a batch is
never assembled from two different group layouts.

API
---

Synchronous training uses two methods:

- ``add_group(group)`` — returns the number of stale groups dropped
- ``pop_all()`` — drain everything into one ``LLMExperienceBatch``

``pop_groups(n)`` consumes the oldest ``n`` groups instead, returning ``None``
when fewer are held. ``agilerl/rollouts/on_policy.py::buffer_llm_rollouts``
stages a collected rollout through a buffer sized to the batch and drains it;
the multi-turn trainer calls it and feeds ``batch.experiences()`` to ``learn()``.

Why the tensors are not packed
------------------------------

Packing rollouts into pre-allocated flat lanes addressed by per-row
``(offset, length)`` suits a buffer that holds many batches across many steps,
where the allocation is amortised over the buffer's lifetime. The synchronous
trainer fills and drains within a single step, so there is nothing to amortise:
copying each tensor into a lane on admission and rebuilding an equivalent torch
tensor on drain measures ~5x the cost of passing the tensors through, and
allocates more transient memory rather than less.

Holding references costs nothing and still gives the group-atomic guarantees and
the shape validation. The tensors ``collect_rollouts_llm`` produces are already
CPU tensors of exactly the dtype and shape ``learn()`` wants.

One consequence worth knowing: the batch aliases the environment's own tensors,
so it is valid only until that environment is reset. The synchronous loop drains
and learns before the next reset; a consumer that outlives a reset must copy.
