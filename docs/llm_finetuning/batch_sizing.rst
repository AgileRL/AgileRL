.. _llm_batch_sizing:

Batch sizes and optimizer steps
===============================

Three separate numbers control one LLM training update, and conflating them is
the easiest way to get a run that trains far more slowly than it should. This
page explains what each one means, how they combine, and what AgileRL does when
you leave them unset.

A few terms
-----------

* **Trajectory**: one completion produced during rollout — a single episode, or
  a single sampled answer to a prompt. It is the unit all three batch sizes
  count.
* **Rollout batch**: every trajectory collected before the policy is updated.
  For the GRPO family this is ``batch_size * group_size``, since each prompt is
  answered ``group_size`` times.
* **Backward pass**: one forward-and-backward through the model, producing
  gradients. Its size is what determines peak GPU memory.
* **Optimizer step**: the moment the optimizer (Adam) actually changes the
  weights. Gradients from one or more backward passes are summed first.
* **Gradient accumulation**: running several backward passes and adding their
  gradients together before taking a single optimizer step. Lets you train at a
  large effective batch size without holding it in memory at once.

The three sizes
---------------

.. list-table::
   :widths: 26 20 54
   :header-rows: 1

   * - Size
     - Argument
     - What it controls
   * - Rollout batch
     - ``batch_size`` x ``group_size``
     - How much experience is collected before the policy is updated.
   * - Mini-batch
     - ``mini_batch_size``
     - How many trajectories one **optimizer step** covers, per rank.
   * - Micro-batch
     - ``micro_batch_size_per_gpu``
     - How many trajectories go through one **backward pass**, per rank.

They are related by a single rule, which AgileRL applies to the DeepSpeed engine
for you:

.. code-block:: text

    gradient_accumulation_steps = mini_batch_size / micro_batch_size_per_gpu

``mini_batch_size`` must be a whole multiple of ``micro_batch_size_per_gpu``, and
the trajectories a rank holds must divide evenly into mini-batches. AgileRL
validates both and raises rather than silently rounding.

The two settings are independent:
``micro_batch_size_per_gpu`` is a **memory** setting, and
``mini_batch_size`` is a **learning-cadence** setting. Lowering the
micro-batch to fit a longer sequence into memory does not change how often the
optimizer steps.

Defaults
--------

Leaving ``mini_batch_size`` unset is fine, and what it resolves to depends on the
algorithm:

* **RL rollout algorithms** (:ref:`GRPO<grpo>`, :ref:`GSPO<gspo>`,
  :ref:`CISPO<cispo>`, :ref:`LLM PPO<llmppo>`, :ref:`LLM REINFORCE<llmreinforce>`)
  default to ``micro_batch_size_per_gpu`` — one optimizer step per backward pass.
* **SFT and DPO** default to the rank's whole batch — one optimizer step per
  batch, the usual supervised cadence.

Why the cadence matters
-----------------------

It is tempting to treat gradient accumulation as free: the same trajectories are
seen either way, so surely the model learns the same amount? It does not. Adam's
progress is dominated by the *number of steps it takes*, not by the magnitude of
any single gradient. Averaging ten micro-batches into one step moves the weights
roughly as far as one step, not ten.

Two runs over identical data can therefore differ by an order of magnitude in how
much the policy actually moves per update, purely from this setting. If a run
looks like it is barely learning while the loss and rewards look reasonable,
check the optimizer step count before anything else.

A worked example
----------------

Take ``batch_size: 2``, ``group_size: 5``, on two data-parallel trainer ranks,
with ``micro_batch_size_per_gpu: 1``:

* The rollout batch is ``2 * 5 = 10`` trajectories.
* Split across 2 ranks, each rank holds **5 trajectories**.
* With the micro-batch at 1, each rank runs **5 backward passes**.

Now the cadence is yours to choose:

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - ``mini_batch_size``
     - Accumulation
     - Optimizer steps per rank, per update
   * - unset (defaults to 1)
     - 1
     - 5 — one per backward pass
   * - 5
     - 5
     - 1 — the whole rank batch in a single step
   * - 3
     - --
     - rejected: 5 trajectories do not divide into mini-batches of 3

Setting it
----------

In Python, pass it to the constructor alongside the micro-batch:

.. code-block:: python

    from agilerl.algorithms import GRPO

    agent = GRPO(
        pad_token_id=tokenizer.pad_token_id,
        pad_token=tokenizer.pad_token,
        model_name="Qwen/Qwen2.5-3B",
        batch_size=2,
        group_size=5,
        micro_batch_size_per_gpu=1,  # memory
        mini_batch_size=5,           # cadence: one step per rank per update
        accelerator=accelerator,
    )

Or in the ``algorithm`` section of a training manifest:

.. code-block:: yaml

    algorithm:
        name: GRPO
        batch_size: 2
        group_size: 5
        micro_batch_size_per_gpu: 1
        mini_batch_size: 5

.. note::

    Under data parallelism every rank derives the same accumulation width and so
    takes the same number of optimizer steps per update. This is required, not
    incidental: ranks that stepped at different times would desynchronise the
    ZeRO collectives.

To pick ``micro_batch_size_per_gpu`` from a GPU's free memory, invert that
knob with :ref:`arena memory solve<llm_memory_estimate>` instead of guessing.
