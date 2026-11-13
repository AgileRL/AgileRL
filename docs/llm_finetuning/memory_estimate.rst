.. _llm_memory_estimate:

GPU memory estimate
===================

``arena memory`` sizes an LLM training run against a GPU before you submit it.
It reads the training manifest, the GPU name, and the checkpoint
``config.json``.

Two commands:

* ``arena memory estimate`` — peak VRAM for a training manifest. Exit 0 if
  both phases fit, 3 if either is over budget, 2 on a usage error. Without ``--gpu`` or
  ``--device-gb`` it picks the cheapest Arena resource tier (by credits per
  node-hour) whose node holds the manifest's GPU layout and fits both phases.
  This needs ``arena login``.
* ``arena memory solve max_model_len`` — the longest context the training run
  still fits on ``--gpu`` or ``--device-gb``, with the rest of the manifest
  held fixed. It uses the same underprediction buffer as ``estimate``. The
  search stops at the checkpoint's RoPE limit, or at ``--hi`` if lower.

.. code-block:: bash

   pip install 'agilerl-arena[hub]'

   arena memory estimate manifest.yaml
   arena memory estimate manifest.yaml --gpu "NVIDIA L4"

   arena memory solve max_model_len manifest.yaml --gpu "NVIDIA L4"

Models in the Arena catalog need no download. For any other model, the ``hub``
extra fetches ``config.json`` from Hugging Face; pass
``--config path/to/config.json`` to stay offline.

The run picks ``vllm_config.tensor_parallel_size`` from the node's GPU count
when the manifest leaves it unset. It splits an unset ``max_num_seqs`` across
the engines ``rollout_engines_per_agent: auto`` starts. For either, pass
``--num-gpus`` with ``--gpu`` or ``--device-gb``; a resource tier brings its
own count. ``solve`` picks them again at each candidate value.

The two phase bars (training and generation) never peak at once. Resource
selection sizes against the larger of the two. The estimate errs high.

A deployment sizes one vLLM engine per node at the widest tensor parallel size,
up to the node's GPU count, that the model's attention and Mamba heads split
over. A node of 6 GPUs serving a model with 32 attention heads runs at width 4.
The caller sets the sequence count, ``gpu_memory_utilization`` and
``max_model_len``.

See the package README at ``agilerl/arena/memory/README.md`` for the
formulas.
