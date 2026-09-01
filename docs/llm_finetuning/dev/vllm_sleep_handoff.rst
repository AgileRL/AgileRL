Colocated vLLM sleep/wake
=========================

.. note::

   **Developer note.** How colocated rollouts work in
   ``agilerl/algorithms/core/base.py``. Not needed to *use* AgileRL; for
   config, see :doc:`../quantization`.

Trainer and vLLM share one GPU by taking turns. Each keeps its own base.
Only the LoRA adapter is copied each step. vLLM's ``sleep(level=1)`` parks
its weights (and KV cache) in host RAM and frees the GPU; ``wake_up()``
puts them back at the same addresses, including a bitsandbytes 4-bit base.

At most one full base sits on the GPU at a time.

Init
----

bitsandbytes quantizes on the GPU during ``from_pretrained``, even with
``device_map="cpu"``. Doing that *after* vLLM has set up CUDA segfaults.
Under ``sleep_mode`` with a fresh bnb trainer, AgileRL builds the trainer
first, offloads it (``_offload_trainer_to_cpu_for_colocated_vllm``), then
constructs ``vllm.LLM(...)`` with sleep mode on. A dense or cloned trainer
starts vLLM first.

Each step
---------

**Rollout** (``_prepare_vllm_for_generation``):

1. If ``use_memory_efficient_params``, move the trainer base to CPU so both
   bases are never on the GPU together.
2. ``wake_up()`` — vLLM restores its base from host RAM.
3. ``_move_lora_to_vllm`` — sync the latest adapter (``add_lora``). The
   base does not move.
4. Generate.

**Training** (``_prepare_vllm_for_training``):

1. ``sleep(level=1)`` — vLLM parks its base and frees the GPU.
2. ``_memory_efficient_params`` brings the trainer base onto the GPU for
   the forward/backward, then parks it again. Off under DeepSpeed ZeRO-3,
   where params are already sharded.

``torch.cuda.memory_allocated()`` does not track vLLM's allocator. Use
``nvidia-smi`` to see memory free on sleep.
