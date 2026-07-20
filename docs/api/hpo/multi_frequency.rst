Multi-Frequency Selection
=========================

**Multi-frequency selection** followed by :ref:`mutation <mutations>` is what makes
**Multiple-Frequencies Population-Based Training** (MF-PBT,
`Doulazmi et al. <https://arxiv.org/abs/2506.03225>`_) possible: the selection operator
reshapes the population and nominates the agents to perturb, and the shared mutation step
then perturbs them.

Multi-frequency selection replaces tournament selection. The population is split into
``n_subpopulations`` subpopulations that each evolve at their own frequency — a slow
subpopulation only exploits/explores every few evaluation cycles. Preserving several
evolution frequencies keeps the greedy short-horizon behaviour of single-frequency PBT
from collapsing the whole population onto one hyperparameter schedule.

Within each subpopulation, agents are ranked by fitness into four brackets —
**winners**, **survivors**, **open-for-migration** and **losers**. Each evolution step
keeps the winners and survivors unchanged and replaces every loser with a clone of a
winner; migration then fills the open-for-migration slots with stronger agents from other
subpopulations, importing their weights but resetting hyperparameters to the local elite
when the migrant comes from a faster subpopulation. On the other hand, if the agent comes
from a slower subpopulation, a full clone is migrated.

Selection returns the indices of those winner-clones, which are finally perturbed by the
shared :func:`Mutations.mutation() <agilerl.hpo.mutation.Mutations.mutation>` step — called
with its ``indices`` argument so only the clones are mutated (see :ref:`mutations`).

The class :class:`MultiFrequencySelection <agilerl.hpo.multi_frequency.MultiFrequencySelection>`
implements the multi-frequency selection operator needed in MF-PBT; its
:func:`select() <agilerl.hpo.multi_frequency.MultiFrequencySelection.select>` returns the
global elite, the evolved population, and the winner-clone indices to mutate. The per-cycle
scheduling and elite saving are driven from the trainers via
:func:`run_selection_and_mutation <agilerl.utils.utils.run_selection_and_mutation>`, the single
entry point shared with tournament selection.
MF-PBT supports an ``accelerator`` but, at the moment,
does not support the LLM RL algorithms.

.. code-block:: python

  from agilerl.hpo.multi_frequency import MultiFrequencySelection

  multi_frequency_selection = MultiFrequencySelection(
      n_subpopulations=2,                  # Two subpopulations...
      n_individuals_per_subpopulation=8,   # ...of eight agents each (pop_size = 16)
      evolution_frequency_ratios=[1, 5],   # One fast, one 5x slower subpopulation
      n_winners=2,                         # Bracket sizes must sum to
      n_survivors=0,                       # n_individuals_per_subpopulation (= 8)
      n_open_for_migration=2,
      n_losers=4,
      seed=42,                             # Derived from the run's global seed
  )

**When to use it.** MF-PBT is most effective with **16 or more agents** (as recommended
by the paper), where there is room for several subpopulations at different frequencies;
its slower subpopulations make it more robust to premature convergence than tournament
selection. The trade-offs are that it needs a larger population to be worthwhile and
exposes more configuration parameters. For small populations or the LLM finetuning
algorithms, prefer :ref:`tournament_selection`.

Configuring from a manifest
---------------------------

In a training manifest, multi-frequency and tournament selection share the single
``tournament_selection`` block, discriminated by a ``selection_strategy`` field that
defaults to ``tournament``. Set ``selection_strategy: multi_frequency`` and provide the
subpopulation layout in the same block; ``training.pop_size`` is derived from it and must
be omitted:

.. code-block:: yaml

    tournament_selection:
      selection_strategy: multi_frequency
      n_subpopulations: 2                 # >= 2
      n_individuals_per_subpopulation: 8  # >= 3   (pop_size = 2 * 8 = 16, derived)
      evolution_frequency_ratios: [1, 5]  # strictly increasing ints >= 1, one per subpop
      n_winners: 2                        # >= 1
      n_survivors: 0                      # >= 0
      n_open_for_migration: 2             # >= 1
      n_losers: 4                         # >= 1

``n_subpopulations`` defaults to ``2`` and ``n_individuals_per_subpopulation`` to ``8``. The
bracket sizes must sum to ``n_individuals_per_subpopulation``; when omitted they default to
``round(0.25 * n)`` winners and open-for-migration agents, ``0`` survivors and the remainder
as losers, and ``evolution_frequency_ratios`` defaults to ``[1, 5, 10, …]``. The recommended
configuration is the one shown above — **2 subpopulations of 8 agents**, split into
**2 winners / 0 survivors / 2 open-for-migration / 4 losers** with frequency ratios
**[1, 5]**.


Parameters
----------

.. autoclass:: agilerl.hpo.multi_frequency.MultiFrequencySelection
  :members:
  :inherited-members:
