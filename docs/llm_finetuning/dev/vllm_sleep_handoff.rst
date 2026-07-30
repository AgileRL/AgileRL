Colocated vLLM sleep/wake for QLoRA RL
======================================

.. note::

   **Developer / implementation note.** This is a code-level walkthrough of how
   colocated rollouts work internally (with references to ``base.py``); it is
   not needed to *use* AgileRL. For the user-facing guide on when and how to turn
   quantization and colocated rollouts on, see :doc:`../quantization`.

AgileRL can train an LLM and serve its own rollouts from a **single GPU**
("colocated"): an HF/PEFT trainer (optionally bitsandbytes-quantized for QLoRA)
and a vLLM engine share one device. Both cannot sit on the GPU at full size at
once, so the GPU is handed back and forth between the rollout phase and the
training step. The mechanism is **vLLM's native sleep/wake**, and a fix in recent
vLLM is what makes this work for *quantized* (bitsandbytes) bases, which is the
feature that unlocks colocated QLoRA.

The enabling vLLM feature
-------------------------

vLLM ≥ 0.22 (AgileRL pins ``vllm>=0.23``) implements ``sleep(level=1)`` as a
lossless host-RAM round-trip: on ``sleep(level=1)`` it copies the engine's
weights to pinned CPU memory and frees their GPU pages (and the KV cache); on
``wake_up()`` it copies them back to the same device addresses. Crucially this
now restores a **bitsandbytes 4-bit** base bit-for-bit, not only a dense one.

Earlier vLLM could not do this for bnb: ``level=1`` did not usefully reclaim the
GPU and ``level=2`` re-quantized the bnb base into garbage on wake, so colocated
QLoRA was not possible. Both are fixed in ≥ 0.22 (verified here by writing known
sentinel values into the base weights and confirming they come back unchanged
after a sleep/wake cycle, plus an end-to-end colocated CISPO run on vLLM 0.22.1
and 0.23.0).

That single capability is the whole design: **each side keeps its own base** (no
shared tensors, no manual weight copying), and AgileRL just cycles vLLM's base
on and off the GPU around each phase.

What the code does
------------------

All in ``agilerl/algorithms/core/base.py``.

**1. Init: CUDA-safe ordering.** A bitsandbytes trainer quantizes on the GPU
during ``from_pretrained`` even with ``device_map="cpu"``. Running those bnb
kernels *after* vLLM has initialised its own CUDA / cumem state segfaults. So
under ``sleep_mode`` with a fresh bnb trainer, AgileRL builds the **trainer
first**, offloads it to CPU (``_offload_trainer_to_cpu_for_colocated_vllm``) to
free the GPU, then constructs ``vllm.LLM(...)``. (A dense or cloned trainer, or
``sleep_mode`` off, takes the simpler vLLM-first order.) vLLM is created with
sleep mode enabled.

**2. Rollout: ``_prepare_vllm_for_generation``.**

- If ``use_memory_efficient_params``, move the trainer's own base to CPU first so
  the two bases never coexist on the GPU.
- ``self.llm.wake_up()``: vLLM restores its base from host RAM onto the GPU.
- Sync the freshly-trained LoRA adapter into vLLM via ``_move_lora_to_vllm``
  (vLLM-native ``add_lora``). Only the adapter moves per step; the base is
  untouched.
- Generate.

**3. Training step: ``_prepare_vllm_for_training``.**

- ``self.llm.sleep(level=1)``: vLLM backs its base up to host RAM and frees the
  GPU (base pages + KV cache), handing the device to the trainer.
- The ``_memory_efficient_params`` context manager brings the trainer's own base
  onto the GPU just for the forward/backward and parks it back on CPU afterwards
  (disabled under DeepSpeed ZeRO-3, where params are already sharded).

Net effect: at most one full base is resident on the GPU at any instant, so a
QLoRA trainer plus a quantized vLLM rollout fit on a single 40 GB A100.

Notes
-----

- Only LoRA adapters are synced trainer to vLLM each step; base weights never
  move between the two engines.
- ``torch.cuda.memory_allocated()`` is unreliable under vLLM's cumem allocator
  (the CUDA virtual-memory allocator vLLM uses to free/restore pages on
  sleep/wake); use ``nvidia-smi`` to observe what actually frees on sleep.
- See :doc:`../quantization` for the full quantization guide and configuration
  options.
