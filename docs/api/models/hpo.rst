HPO Specifications
==================

A training manifest selects its evolution regime through the single
``selection_strategy`` block, a discriminated union keyed on ``strategy``:
``tournament`` (the default) validates the block as :class:`TournamentSelectionSpec`, and
``multi_frequency`` validates it as :class:`MultiFrequencySelectionSpec`. The two regimes
are therefore mutually exclusive by construction, and a block that omits ``strategy``
is treated as tournament selection so existing configs are unchanged.

The block was previously named ``tournament_selection``; that spelling is still
accepted as an alias, so manifests written against earlier versions keep validating.
Serialized output still uses ``tournament_selection``.

Both regimes take their population size from the mandatory ``training.pop_size`` field.

In Python, the trainers and
:func:`~agilerl.models.manifest.from_trainer_specs`
take this block through a single ``selection_strategy`` argument. The former
spellings (``tournament`` on the trainers and ``tournament_selection`` on
``from_trainer_specs``) are still accepted with a ``DeprecationWarning``.

.. autoclass:: agilerl.models.hpo.MutationProbabilities
   :members:

.. autoclass:: agilerl.models.hpo.MutationSpec
   :members:

.. autoclass:: agilerl.models.hpo.TournamentSelectionSpec
   :members:

.. autoclass:: agilerl.models.hpo.MultiFrequencySelectionSpec
   :members:
