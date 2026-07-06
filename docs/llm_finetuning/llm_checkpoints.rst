.. _llm_checkpoints:

Saving and Loading LLM Checkpoints
==================================

LLM checkpoints in AgileRL can persist just LoRA adapters, the full model, and
optionally the optimizer/LR-scheduler state. The same on-disk format is
written for plain (single-process) training and distributed training
(data-parallel or FSDP2).
The defaults are ``lora_only=True`` and ``save_optimizer=True``.

Checkpoint layout on disk
-------------------------

A typical checkpoint directory written by :meth:`save_checkpoint` looks like:

.. code-block:: text

    checkpoint_dir/
    ├── attributes.pt              # algorithm hyperparameters; may also
    │                              # contain the actor state_dict and/or
    │                              # optimizer state depending on flags
    ├── actor/
    │   ├── adapter_model.safetensors
    │   └── adapter_config.json
    ├── reference/                 # only if use_separate_reference_adapter=True
    │   ├── adapter_model.safetensors
    │   └── adapter_config.json
    └── critic/                    # only for algorithms with a value head
        ├── adapter_model.safetensors
        └── adapter_config.json

Which adapter subdirectories appear depends on the algorithm:

* **SFT**: ``actor`` only.
* **DPO, GRPO**: ``actor`` + ``reference``.
* **PPO-LLM** (with value head): ``actor`` + ``reference`` + ``critic``.

Saving
------

.. code-block:: python

    agent.save_checkpoint(
        path,
        lora_only=True,        # default: adapters only, no base weights
        save_optimizer=True,   # default: persist optimizer + LR scheduler
    )

The four combinations (identical for single-device, data-parallel and FSDP2 runs):

+---------------+--------------------+---------------------------------------------------+
| ``lora_only`` | ``save_optimizer`` | Produces                                          |
+===============+====================+===================================================+
| ``True``      | ``True``           | Adapter dirs on disk; optimizer state inside      |
|               |                    | ``attributes.pt``.                                |
+---------------+--------------------+---------------------------------------------------+
| ``True``      | ``False``          | Adapter dirs only. No optimizer state.            |
+---------------+--------------------+---------------------------------------------------+
| ``False``     | ``True``           | Full actor ``state_dict`` + optimizer state, both |
|               |                    | inside ``attributes.pt``.                         |
+---------------+--------------------+---------------------------------------------------+
| ``False``     | ``False``          | Full actor ``state_dict`` inside ``attributes.pt``|
|               |                    | (no optimizer state).                             |
+---------------+--------------------+---------------------------------------------------+

Under FSDP2, sharded parameters and optimizer state are gathered to full
tensors before writing, so checkpoints are rank-count independent and can be
loaded on a different number of GPUs.

Common scenarios:

.. code-block:: python

    # Periodic snapshot during training (adapters + optimizer, so training
    # can resume where it left off):
    agent.save_checkpoint(path)

    # Release a deployable artefact (adapters only, no training state):
    agent.save_checkpoint(path, save_optimizer=False)

    # Persist the full model, base weights included, not just the adapters
    # (e.g. for hand-off to a consumer that can't re-download the base):
    agent.save_checkpoint(path, lora_only=False, save_optimizer=False)

Loading
-------

.. code-block:: python

    agent.load_checkpoint(
        path,
        load_optimizer=True,   # default: restore optimizer + LR scheduler
    )

``save_optimizer`` and ``load_optimizer`` are independent flags: you can
load a checkpoint that contains optimizer state while passing
``load_optimizer=False`` to keep the live optimizer, or load a
weights-only checkpoint with ``load_optimizer=True`` (in which case a
``UserWarning`` is emitted and the existing optimizer is kept as-is).

:meth:`load_checkpoint` expects the live algorithm to already be configured
against the same base model. It restores adapter weights on top of that base
and, by default, copies the just-loaded ``actor`` adapter onto ``reference``
so that SFT → DPO → GRPO pipelines work out of the box: the actor trained
in stage *N* becomes the reference for stage *N+1*.

The checkpoint's LoRA config must match the live algorithm's (rank,
target modules, etc.); a mismatch raises ``ValueError``. Re-create the
agent with the checkpoint's LoRA config to load it.

Common scenarios:

.. code-block:: python

    # Resume training:
    agent.load_checkpoint(path)

    # Inference / evaluation with a checkpoint that may or may not contain
    # optimizer state, which we don't need:
    agent.load_checkpoint(path, load_optimizer=False)

Distributed training
--------------------

The save/load paths are uniform across distributed modes (data-parallel or
FSDP2): adapter directories plus ``attributes.pt``, with optimizer state
embedded when ``save_optimizer=True``. Under FSDP2, parameters and optimizer
state are gathered to full tensors via ``torch.distributed.checkpoint`` before
writing; note that loading adapter weights into an already-sharded model is
not yet supported — load checkpoints before sharding, or run unsharded
(single process or data-parallel).

Multi-process correctness (only the main process writes ``attributes.pt``,
followed by a barrier) is handled internally — you call
:meth:`save_checkpoint` / :meth:`load_checkpoint` the same way whether
you're on one GPU or many.
