.. _llm_distributed:

Multi-GPU LLM training
======================

LLM training in AgileRL is **torch-native**. Launch with ``torchrun``;
``torch.distributed`` is the process group. Classic RL still uses Accelerate —
see :ref:`distributed_training`.

Data parallel (no sharding)
---------------------------

``torchrun`` on an unchanged script is data parallel (DDP). Do not pass
``fsdp_config``. Each rank loads a full copy of the actor; LoRA gradients are
averaged across ranks at the optimizer step.

.. code-block:: python

    agent = GRPO(
        model_name="Qwen/Qwen2.5-3B",
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        vllm_config=VLLMConfig(),
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
    from agilerl.utils.algo_utils import VLLMConfig

    fsdp_config = FSDPConfig()

    agent = GRPO(
        model_name="Qwen/Qwen2.5-3B",
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        vllm_config=VLLMConfig(),
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

On Arena
--------

Arena launches the ranks for you; there is no ``torchrun``. Size the run in the
manifest and pick a multi-GPU resource from ``arena resources list``:

.. code-block:: yaml

    algorithm:
      name: GRPO
      # fsdp is optional: the run trains DDP when it fits one GPU and is no
      # slower, else FSDP. Set fsdp.ep or fsdp.tp to shard experts or layers.

    training:
      rollout_batch_size: 8        # multiple of training_gpus_per_agent
      training_gpus_per_agent: 4   # data-parallel trainer ranks per population member
      rollout_engines_per_agent: 2 # vLLM engines per member, one GPU each

.. code-block:: bash

    arena experiments submit grpo.yaml \
        --resource-id arena-a100-40gb-4gpu \
        --num-nodes 2 \
        --project my-project

See :ref:`arena_training` for authentication and job management.

``FSDPConfig`` fields
---------------------

Construct with :class:`~agilerl.distributed.FSDPConfig`. Defaults are the
usual production settings for LoRA / QLoRA.

``reshard_after_forward``, ``optim_cpu_offload``, ``routed_expert_chunk_mib``,
``activation_offload`` and ``cpu_offload`` are memory-vs-speed settings.
Leave them unset: the algorithm picks the fastest combination the memory
estimate fits on the device. With ``ep=1`` and ``tp=1`` it trains the
faster plan that fits: DDP when the whole model, adapters and optimizer
state fit one GPU with an equal-or-cheaper plan, else FSDP. FSDP can win
while DDP fits, for example by checkpointing every 4th block where DDP
checkpoints every block. The start-up log names the mode (``DDP`` or
``FSDP``). Sharded, it gives up speed in this order: reshard after forward,
a smaller expert chunk, a smaller micro-batch, the optimizer to CPU,
activation offload, then full CPU offload (colocated vLLM only). On Ray the
driver also picks ``checkpoint_every_n_blocks`` with the
activation-checkpointing choice. A training manifest rejects all of them.

.. list-table::
   :widths: 28 14 58
   :header-rows: 1

   * - Field
     - Default
     - What it does
   * - ``reshard_after_forward``
     - picked
     - Free the all-gathered parameters after each module's forward so the
       next module can reuse that memory. ``False`` keeps them gathered
       (faster, more VRAM).
   * - ``optim_cpu_offload``
     - picked
     - Keep parameters and gradients on GPU; move Adam ``m``/``v`` to CPU
       except during ``step()``. Mutually exclusive with ``cpu_offload``.
   * - ``routed_expert_chunk_mib``
     - picked
     - Byte budget (MiB) of each routed-expert chunk in MoE LoRA, one of
       64/128/256/512. Larger chunks run faster and raise peak memory.
   * - ``activation_offload``
     - picked
     - Move saved activations to host memory during the training forward.
       Spends host RAM and PCIe time.
   * - ``checkpoint_every_n_blocks``
     - picked
     - Checkpoint every n-th transformer block when activation checkpointing
       is on; the other blocks keep their activations.
   * - ``cpu_offload``
     - picked
     - Offload sharded parameters and gradients to CPU
       (``CPUOffloadPolicy``). Picked only when nothing else fits. Requires
       colocated vLLM (``vllm_config``); HuggingFace generate and async
       rollout engines assume weights stay on the compute device.

``cpu_offload`` and ``optim_cpu_offload`` cannot both be ``True``.

Each transformer block is one FSDP unit, and the next unit is all-gathered
while the current one runs. Parameters under 100,000 elements stay
replicated. Gradients are reduce-scattered only on the last micro-batch of
an optimizer step, so unsharded grads are held in between.

With ``create_population``, set ``FSDP`` in ``INIT_HP`` — ``True`` for
defaults, or a dict of fields:

.. code-block:: python

    INIT_HP = {
        "ALGO": "GRPO",
        "FSDP": True,
        # or: "FSDP": {"shard_group_size": 8},
        ...
    }

Checkpoints under FSDP2 use the same on-disk layout as a single GPU
(gathered tensors in ``attributes.pt``). See :ref:`llm_checkpoints`.
QLoRA + FSDP2 is covered in :ref:`llm_quantization`. Mini-batch vs
micro-batch (and why ranks must step together) is :ref:`llm_batch_sizing`.
