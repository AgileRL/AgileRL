.. _llm_distributed:

Multi-GPU LLM training
======================

LLM training in AgileRL is **torch-native**. There is no Accelerate or DeepSpeed
on this path. Launch with ``torchrun``; ``torch.distributed`` is the process
group. Classic RL still uses Accelerate — see :ref:`distributed_training`.

Data parallel (no sharding)
---------------------------

``torchrun`` on an unchanged script is data parallel. Do not pass
``fsdp_config``. Each rank loads a full copy of the actor; LoRA gradients are
averaged across ranks at the optimizer step.

.. code-block:: python

    agent = GRPO(
        model_name="Qwen/Qwen2.5-3B",
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_vllm=True,
        # no fsdp_config — data parallel
    )

.. code-block:: bash

    torchrun --nproc_per_node=N path/to/training_script

Use this when the model already fits on one GPU and you want more throughput
from extra devices.

FSDP2 (sharded)
---------------

Pass an :class:`~agilerl.distributed.FSDPConfig` when the actor does not fit,
or you want the memory headroom. Parameters are sharded with PyTorch
``fully_shard`` (transformer blocks, embeddings, untied ``lm_head``, then the
root). Each rank stores a slice.

``FSDPConfig`` requires a process group. Construct it only under ``torchrun``,
not in a single-process script.

.. code-block:: python

    from agilerl.algorithms import GRPO
    from agilerl.distributed import FSDPConfig

    fsdp_config = FSDPConfig(
        reshard_after_forward=True,
        cpu_offload=False,
        optim_cpu_offload=True,
        defer_grad_sync=True,
    )

    agent = GRPO(
        model_name="Qwen/Qwen2.5-3B",
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        use_vllm=True,
        fsdp_config=fsdp_config,
    )

The same object works on any LLM algorithm (GRPO, CISPO, GSPO, LLMPPO,
LLMREINFORCE, SFT, DPO) and on ``.population()``:

.. code-block:: python

    pop = GRPO.population(
        size=4,
        model_name="Qwen/Qwen2.5-3B",
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        fsdp_config=fsdp_config,
        **init_hp,
    )

Same launch command as data parallel:

.. code-block:: bash

    torchrun --nproc_per_node=N path/to/training_script

Or a training manifest:

.. code-block:: bash

    torchrun --nproc_per_node=N -m agilerl.train path/to/manifest.yaml --device cuda

``FSDPConfig`` fields
---------------------

Construct with :class:`~agilerl.distributed.FSDPConfig`. Defaults are the
usual production settings for LoRA / QLoRA.

.. list-table::
   :widths: 28 14 58
   :header-rows: 1

   * - Field
     - Default
     - What it does
   * - ``reshard_after_forward``
     - ``True``
     - Free the all-gathered parameters after each module's forward so the
       next module can reuse that memory. ``False`` keeps them gathered
       (faster, more VRAM).
   * - ``optim_cpu_offload``
     - ``True``
     - Keep parameters and gradients on GPU; move Adam ``m``/``v`` to CPU
       except during ``step()``. Mutually exclusive with ``cpu_offload``.
   * - ``cpu_offload``
     - ``False``
     - Offload sharded parameters and gradients to CPU
       (``CPUOffloadPolicy``). Requires ``use_vllm=True`` — HuggingFace
       generate assumes weights stay on the compute device.
   * - ``defer_grad_sync``
     - ``True``
     - Skip reduce-scatter until the last micro-batch of an optimizer step
       (``set_requires_gradient_sync``). Saves communication; holds
       unsharded grads in between. ``False`` reduce-scatters every backward.

``cpu_offload`` and ``optim_cpu_offload`` cannot both be ``True``.

With ``create_population``, set ``FSDP`` in ``INIT_HP`` — ``True`` for
defaults, or a dict of fields:

.. code-block:: python

    INIT_HP = {
        "ALGO": "GRPO",
        "FSDP": True,
        # or: "FSDP": {"cpu_offload": False, "defer_grad_sync": True},
        ...
    }

Checkpoints under FSDP2 use the same on-disk layout as a single GPU
(gathered tensors in ``attributes.pt``). See :ref:`llm_checkpoints`.
QLoRA + FSDP2 is covered in :ref:`llm_quantization`. Mini-batch vs
micro-batch (and why ranks must step together) is :ref:`llm_batch_sizing`.
