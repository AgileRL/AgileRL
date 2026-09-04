.. _llm_memory_estimate:

GPU memory estimate
===================

``arena memory`` sizes an LLM run against a GPU before you submit it. It reads
the training manifest (or a serving model id), the GPU name, and the
checkpoint ``config.json``. There is no profiling step and no weight download.

Two commands:

* ``arena memory estimate`` — peak VRAM for a training manifest. Exit 0 if
  both phases fit, 3 if either is over budget.
* ``arena memory solve KNOB`` — largest value of one knob that still fits,
  with everything else held fixed. Use this to pick a context length, a
  concurrent-sequence cap, or a micro-batch.

.. code-block:: bash

   pip install 'agilerl-arena[hub]'

   arena memory estimate manifest.yaml --gpu "NVIDIA L4"

   arena memory solve max_model_len --inference --gpu "NVIDIA L4" \
       --model Qwen/Qwen2.5-7B-Instruct

``--inference`` is a dedicated serving GPU (utilization 0.9, 8 sequences, no
trainer residual). ``--max-num-seqs 1`` is the longest single-request context.
Without ``--inference``, pass a training manifest and the solve is that run.

Invertible knobs: ``max_model_len``, ``max_num_seqs``,
``micro_batch_size_per_gpu``.

``python -m agilerl.arena.memory`` is the same CLI if the ``arena`` script is
not on ``PATH``. Pass ``--config path/to/config.json`` to stay offline.

The two phase bars (training and generation) never peak at once. Resource
selection sizes against the larger of the two. The model is safety-biased:
it prefers to over-predict.

See the package README at ``agilerl/arena/memory/README.md`` for the
formulas, the validated error band, and how a new architecture is checked.
