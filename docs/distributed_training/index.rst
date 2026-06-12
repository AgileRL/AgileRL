Distributed Training
====================

AgileRL's classic RL algorithms (on-policy, off-policy, offline, multi-agent and
bandit) run on a single device. Multi-GPU training is a feature of the LLM
fine-tuning stack (:ref:`GRPO<grpo>`, PPO, SFT, DPO), built directly on native
``torch.distributed`` — there is no wrapper object or external launcher
dependency.

Launching with torchrun
-----------------------

Launch an LLM training script across multiple GPUs with ``torchrun``:

.. code-block:: bash

  torchrun --nproc_per_node 4 train_script.py

``torchrun`` sets the standard rendezvous environment variables
(``RANK``/``LOCAL_RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``), and
the LLM algorithm constructors call :func:`agilerl.utils.distributed.init_distributed`
automatically, so no changes to the training script are needed: the same script
runs on a single device or across many GPUs. An orchestration layer such as Ray
can equally set these variables (or initialise the process group itself) and
the library will pick it up.

By default, training is data-parallel: each rank holds a full model replica and
gradients are averaged across ranks at the gradient-accumulation boundary.

Sharding with FSDP2
-------------------

For models too large for a single GPU, shard the actor with PyTorch FSDP2 by
passing an :class:`~agilerl.utils.distributed.FSDPConfig` to the algorithm
constructor:

.. code-block:: python

  from agilerl.algorithms import GRPO
  from agilerl.utils.distributed import FSDPConfig

  agent = GRPO(
      ...,
      gradient_accumulation_steps=4,
      fsdp_config=FSDPConfig(
          reshard_after_forward=True,  # ZeRO-3-like memory profile
          cpu_offload=False,
      ),
  )

See the :ref:`LLM fine-tuning docs<llm_finetuning>` and the
:ref:`GRPO tutorial<grpo_tutorial>` for end-to-end distributed training
examples, and ``agilerl.utils.distributed`` for the full set of helpers.
