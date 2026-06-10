"""Comparison tooling for HPO benchmarks.

A thin, read-only layer on top of the benchmarking harness
(``hpo_improvements/benchmarking``). Given the results folders of two
finished benchmarks — a *studied* one and a *baseline* one — that used the **same
RL algorithm**, it restricts attention to the ``(environment, seed)`` pairs the
two share and quantifies how much the studied benchmark improves on the baseline:

* the **probability of improvement** ``P(studied > baseline)`` over final best
  normalized fitness, with stratified-bootstrap confidence intervals (rliable,
  Agarwal et al. 2021), and
* the **IQM of the per-pair normalized-fitness difference**
  ``f_studied - f_baseline`` as a function of per-agent environment interactions
  (``global_steps / pop_size``), with stratified-bootstrap confidence bands.

Run it from this folder::

    python compare.py --studied <studied_results_folder> \
        --baseline <baseline_results_folder>

Both arguments are the names of folders under
``hpo_improvements/benchmarking/results`` (absolute paths also work). Any
omitted argument is prompted for interactively.
"""
