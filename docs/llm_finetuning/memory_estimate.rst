.. _llm_memory_estimate:

GPU memory estimate
===================

``arena memory estimate`` sizes an LLM training manifest against a GPU before
you submit it. It reads the training manifest, the GPU name, and the
checkpoint ``config.json``. There is no profiling step and no weight download.

Exit 0 if both phases fit, 3 if either is over budget.

.. code-block:: bash

   pip install 'agilerl-arena[hub]'

   arena memory estimate manifest.yaml --gpu "NVIDIA L4"

``python -m agilerl.arena.memory`` is the same CLI if the ``arena`` script is
not on ``PATH``. Pass ``--config path/to/config.json`` to stay offline.

The two phase bars (training and generation) never peak at once. Resource
selection sizes against the larger of the two. The model is safety-biased:
it prefers to over-predict.

See the package README at ``agilerl/arena/memory/README.md`` for the
formulas and the validated error band.
