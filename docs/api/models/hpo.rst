HPO Specifications
==================

A training manifest selects its evolution regime through the single
``tournament_selection`` block, a discriminated union keyed on ``selection_strategy``:
``tournament`` (the default) validates the block as :class:`TournamentSelectionSpec`, and
``multi_frequency`` validates it as :class:`MultiFrequencySelectionSpec`. The two regimes
are therefore mutually exclusive by construction, and a block that omits
``selection_strategy`` is treated as tournament selection so existing configs are
unchanged.

Both regimes take their population size from the mandatory ``training.pop_size`` field.

.. autoclass:: agilerl.models.hpo.MutationProbabilities
   :members:

.. autoclass:: agilerl.models.hpo.MutationSpec
   :members:

.. autoclass:: agilerl.models.hpo.TournamentSelectionSpec
   :members:

.. autoclass:: agilerl.models.hpo.MultiFrequencySelectionSpec
   :members:
