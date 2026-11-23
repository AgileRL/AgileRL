.. _llm_memory_estimate:

GPU memory estimate
===================

``arena memory estimate`` sizes an LLM training run against a GPU before you
submit it. It sends the training manifest and the GPU to Arena, which runs the
estimate, so it needs ``arena login`` (or an API key). It sizes models in the
Arena catalog.

Exit 0 if both phases fit, 3 if either is over budget, 2 on a usage error.
Pass ``--gpu`` with a GPU type from ``arena resources list`` (for example
``--gpu h100-80``), or ``--device-gb``; without either it exits 2. ``--json``
prints the full estimate.

.. code-block:: bash

   arena memory estimate manifest.yaml --gpu h100-80
   arena memory estimate manifest.yaml --gpu "NVIDIA L4"

The run picks ``vllm_config.tensor_parallel_size`` from the node's GPU count
when the manifest leaves it unset. It splits an unset ``max_num_seqs`` across
the engines ``rollout_engines_per_agent: auto`` starts. For either, pass
``--num-gpus`` with ``--gpu`` or ``--device-gb``.

The two phase bars (training and generation) never peak at once. Resource
selection sizes against the larger of the two. The estimate errs high.

A deployment sizes one vLLM engine per node at the widest tensor parallel size,
up to the node's GPU count, that the model's attention and Mamba heads split
over. A node of 6 GPUs serving a model with 32 attention heads runs at width 4.
The caller sets the sequence count, ``gpu_memory_utilization`` and
``max_model_len``.
