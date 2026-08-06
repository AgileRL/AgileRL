LLM rollout records
===================

.. note::

   **Developer / implementation note.** Internal detail of
   ``agilerl/components/llm_rollout_data.py``. Synchronous LLM training uses it
   automatically; there is nothing for users to configure.

The module holds the record types for collected rollouts and the collation that
turns them into a learn-ready batch. Trajectory tensors are held **by
reference** — nothing is copied on the way in or on the way out — and padding
into rectangular batches happens once, in ``collate_rollout_groups``.

Layout
------

The unit of collation is the **prompt group** (all ``group_size`` completions of
one prompt). Groups are collated whole, so the group-divisibility the algorithms
rely on holds by construction.

Three types make up the module:

- ``Trajectory`` — one episode: ``token_ids`` ``(1, T)``, ``action_masks``
  and ``turn_ids`` ``(1, T - 1)``, per-turn ``rewards``, and optional
  ``sampling_logps``. Validated on construction, so a mask or turn-id tensor
  that is not exactly one shorter than the token ids fails immediately rather
  than misaligning a loss much later.
- ``RolloutGroup`` — exactly ``group_size`` trajectories.
- ``LLMExperienceBatch`` — the collated result: ragged ``token_ids`` and
  ``action_masks`` that ``learn()`` pads itself, plus pre-stacked ``rewards``
  and ``turn_ids`` rectangles, ``token_lengths``, and the per-row
  ``sampling_logps``.

API
---

``collate_rollout_groups(groups)`` flattens a list of groups into one
``LLMExperienceBatch``. ``agilerl/rollouts/on_policy.py::collate_llm_rollouts``
wraps a collected rollout's parallel tensor lists into groups and collates them;
the rollout trainer calls it and feeds ``batch.experiences()`` to ``learn()``.

Why the tensors are not packed
------------------------------

Packing rollouts into pre-allocated flat lanes addressed by per-row
``(offset, length)`` suits a buffer that holds many batches across many steps,
where the allocation is amortised over the buffer's lifetime. The synchronous
trainer collates and learns within a single step, so there is nothing to
amortise: copying each tensor into a lane on admission and rebuilding an
equivalent torch tensor on drain measures ~5x the cost of passing the tensors
through, and allocates more transient memory rather than less.

Holding references costs nothing and still gives the group-atomic guarantees and
the shape validation. The tensors ``collect_rollouts_llm`` produces are already
CPU tensors of exactly the dtype and shape ``learn()`` wants.

One consequence worth knowing: the batch aliases the environment's own tensors,
so it is valid only until that environment is reset. The synchronous loop
collates and learns before the next reset; a consumer that outlives a reset must
copy.
