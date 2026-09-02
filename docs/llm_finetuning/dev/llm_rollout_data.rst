LLM rollout records
===================

.. note::

   **Developer note.** Internal to ``agilerl/components/llm_rollout_data.py``.
   Synchronous LLM training uses it automatically; there is nothing to configure.

Rollouts land as tensors. This module names them, checks their shapes, and
stacks a list of prompt-groups into one batch for ``learn()``. The tensors are
held **by reference** — nothing is copied in or out.

Types
-----

Collation is by **prompt group**: all ``group_size`` completions of one prompt
stay together, so group-relative algorithms never see a split group.

* ``Trajectory`` — one episode. ``token_ids`` is ``(1, T)``;
  ``action_masks`` and ``turn_ids`` are ``(1, T - 1)``; plus per-turn
  ``rewards`` and optional ``sampling_logps``. Construction fails if a mask
  or turn-id tensor is not exactly one shorter than the token ids.
* ``RolloutGroup`` — exactly ``group_size`` trajectories.
* ``LLMExperienceBatch`` — the collated result. ``token_ids`` and
  ``action_masks`` stay ragged (``learn()`` pads them); ``rewards`` and
  ``turn_ids`` are already rectangles.

``collate_rollout_groups(groups)`` flattens a list of groups into one batch.
The trainer wraps collected tensors into groups, collates, and passes
``batch.experiences()`` to ``learn()``.

The batch aliases the environment's tensors, so it is valid only until that
environment is reset. The training loop collates and learns before the next
reset. Copy if you need to keep a batch longer than that.
