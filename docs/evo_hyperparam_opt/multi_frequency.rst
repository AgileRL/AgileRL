.. _multi_frequency_strategy:

Multiple-Frequencies Population-Based Training (MF-PBT)
=======================================================

MF-PBT (`Doulazmi et al. <https://arxiv.org/abs/2506.03225>`_) is an alternative
evolution regime that **replaces tournament selection**. Rather than updating the whole
population together, MF-PBT splits it into ``n_subpopulations`` subpopulations that
each evolve at their own frequency: subpopulation ``i`` only evolves every
``evolution_frequency_ratios[i]`` evaluation cycles. Keeping several evolution
frequencies alive at once counteracts the greediness of single-frequency PBT, where a
short evolution horizon collapses the population onto whichever schedule looks best in
the near term.

Each evolution step ranks a subpopulation by fitness into four brackets: **winners**,
**survivors**, **open-for-migration** and **losers**. Every loser is replaced by a
perturbed clone of a winner, and the winners/survivors are kept unchanged. **Migration**
then fills the open-for-migration slots with stronger agents from other subpopulations:
a migrant from a faster subpopulation contributes only its trained weights while adopting
the local elite's hyperparameters, whereas a migrant from a slower-or-equal subpopulation
is cloned in full.

**When to use it.** MF-PBT shines with **16 or more agents** (as recommended by the
paper), where there is room for several subpopulations at different frequencies; its
slower subpopulations make it more robust to premature convergence than tournament
selection. The trade-offs are that it needs a larger population to be worthwhile, exposes
more configuration parameters, and is **single-process only** (it cannot
be used with an :class:`~accelerate.Accelerator`). For small populations or the LLM
finetuning algorithms, prefer :ref:`tournament_selection`.

MF-PBT and tournament selection share the single ``tournament_selection`` manifest block,
discriminated by a ``selection_strategy`` field that defaults to ``tournament``. To use
MF-PBT, set ``selection_strategy: multi_frequency`` and supply the subpopulation layout in
the same block (the two regimes are mutually exclusive, so only one is ever configured).
``pop_size`` is derived as ``n_subpopulations * n_individuals_per_subpopulation`` and
should be omitted. The bracket sizes and frequency ratios have sensible defaults (leave
them out to accept them):

.. code-block:: yaml

    # `selection_strategy` defaults to `tournament`; set it to `multi_frequency` for MF-PBT.
    # Omit `training.pop_size` — it is derived from the subpopulation layout.
    tournament_selection:
      selection_strategy: multi_frequency
      n_subpopulations: 2                 # >= 2
      n_individuals_per_subpopulation: 8  # >= 3  (pop_size = 2 * 8 = 16)
      n_winners: 2                        # >= 1
      n_survivors: 0                      # >= 0
      n_open_for_migration: 2             # >= 1
      n_losers: 4                         # >= 1
      evolution_frequency_ratios: [1, 5]  # strictly increasing ints >= 1, one per subpop

The bracket sizes must sum to ``n_individuals_per_subpopulation``. The
four bracket defaults are ``round(0.25 * n)`` winners and open-for-migration agents, ``0``
survivors and the remainder as losers; the frequency-ratio default is ``[1, 5, 10, …]``.
The recommended configuration is the one shown above: **2 subpopulations of 8 agents**,
split into **2 winners / 0 survivors / 2 open-for-migration / 4 losers** with frequency
ratios **[1, 5]**.

MF-PBT is implemented by :class:`MultiFrequencyStrategy <agilerl.hpo.multi_frequency.MultiFrequencyStrategy>`
and dispatched from the trainers via
:func:`run_selection_and_mutation <agilerl.utils.utils.run_selection_and_mutation>`,
which routes to either tournament selection or MF-PBT by the strategy's type.
