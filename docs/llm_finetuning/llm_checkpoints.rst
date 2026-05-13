.. _llm_checkpoints:

Saving and Loading LLM Checkpoints
==================================

LLM checkpoints in AgileRL can persist just LoRA adapters, the full model, and
optionally the optimizer/LR-scheduler state — with separate code paths for
plain (single-process) training and distributed training via
`DeepSpeed <https://www.deepspeed.ai/>`_ + `Accelerate
<https://huggingface.co/docs/accelerate/index>`_. The defaults are
``lora_only=True`` and ``save_optimizer=True``.

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
    ├── critic/                    # only for algorithms with a value head
    │   ├── adapter_model.safetensors
    │   └── adapter_config.json
    └── save_checkpoint/           # DeepSpeed sharded checkpoint; only when
                                   # training with an Accelerator

Which adapter subdirectories appear depends on the algorithm:

* **SFT** — ``actor`` only.
* **DPO, GRPO** — ``actor`` + ``reference``.
* **PPO-LLM** (with value head) — ``actor`` + ``reference`` + ``critic``.

Saving
------

.. code-block:: python

    agent.save_checkpoint(
        path,
        lora_only=True,        # default — adapters only, no base weights
        save_optimizer=True,   # default — persist optimizer + LR scheduler
    )

The four combinations on the non-distributed path:

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
|               |                    | — no optimizer state.                             |
+---------------+--------------------+---------------------------------------------------+

On the DeepSpeed path, ``save_optimizer=True`` writes a sharded checkpoint
into ``<path>/save_checkpoint/`` via the engine instead of bundling optimizer
state into ``attributes.pt``. ``lora_only=True`` still writes adapter
directories. The ``lora_only=False, save_optimizer=False`` cell gathers ZeRO-3
shards and injects the full ``state_dict`` into ``attributes.pt``.

Common scenarios:

.. code-block:: python

    # Periodic snapshot during training (adapters + optimizer, so training
    # can resume where it left off):
    agent.save_checkpoint(path)

    # Release a deployable artefact — adapters only, no training state:
    agent.save_checkpoint(path, save_optimizer=False)

    # Persist the full merged model (e.g. for hand-off to a non-AgileRL
    # consumer that doesn't understand PEFT):
    agent.save_checkpoint(path, lora_only=False, save_optimizer=False)

Loading
-------

.. code-block:: python

    agent.load_checkpoint(
        path,
        load_optimizer=True,   # default — restore optimizer + LR scheduler
    )

``save_optimizer`` and ``load_optimizer`` are independent flags — you can
load a checkpoint that contains optimizer state while passing
``load_optimizer=False`` to keep the live optimizer, or load a
weights-only checkpoint with ``load_optimizer=True`` (in which case a
``UserWarning`` is emitted and the existing optimizer is kept as-is).

:meth:`load_checkpoint` expects the live algorithm to already be configured
against the same base model. It restores adapter weights on top of that base
and, by default, copies the just-loaded ``actor`` adapter onto ``reference``
so that SFT → DPO → GRPO pipelines work out of the box — the actor trained
in stage *N* becomes the reference for stage *N+1*.

LoRA config mismatch between the checkpoint and the live algorithm
(e.g. after a rank mutation) is handled non-destructively: rank is merged as
``max(current, checkpoint)`` with weights padded into the larger shape;
``target_modules`` / ``modules_to_save`` are unioned. See
:meth:`load_checkpoint` for details.

Common scenarios:

.. code-block:: python

    # Resume training:
    agent.load_checkpoint(path)

    # Inference / evaluation with a checkpoint that may or may not contain
    # optimizer state — we don't need it:
    agent.load_checkpoint(path, load_optimizer=False)

DeepSpeed and Accelerate
------------------------

When an :class:`~accelerate.Accelerator` with a ``DeepSpeedPlugin`` is
attached, the save/load paths differ as follows:

* ``save_optimizer=True`` delegates to the DeepSpeed engine's own sharded
  checkpoint format, written to ``<path>/save_checkpoint/``. The matching
  load path reads the same directory.
* ``save_optimizer=False`` falls back to the PEFT / torch-save path, which
  produces the same adapter directories / ``attributes.pt`` as plain training.
* ZeRO-3 sharded parameters are gathered via the appropriate gather context
  before being written, so the on-disk layout is identical regardless of
  ZeRO stage.

Multi-process correctness (only the main process writes ``attributes.pt``,
followed by ``accelerator.wait_for_everyone()``) is handled internally — you
call :meth:`save_checkpoint` / :meth:`load_checkpoint` the same way whether
you're on one GPU or many.
