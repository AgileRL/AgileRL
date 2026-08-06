.. _llm_quantization:

Quantization
============

Quantization stores a model's **weights** (its learned parameters) at lower
numerical precision than the usual 16-bit ``bfloat16``/``float16``, typically as
8-bit integers or 4-bit floats. Fewer bits per weight means the model takes up
far less GPU memory, at the cost of a small amount of rounding error.

AgileRL uses
`bitsandbytes <https://github.com/bitsandbytes-foundation/bitsandbytes>`_ to
quantize the Hugging Face **trainer** (the model that actually learns), and pairs
it with a matching bitsandbytes-quantized `vLLM <https://docs.vllm.ai/>`_
**rollout engine** (the fast inference engine that generates responses during
training), so both sides run from the same low-precision base weights.

.. note:: A few terms used throughout this page

   * **Base model / base weights**: the pretrained model you start from. In
     AgileRL it stays frozen; only small adapters are trained on top of it.
   * **LoRA adapter**: a small set of trainable weights added alongside the
     frozen base. Training updates only the adapter, which is what makes
     fine-tuning cheap in memory.
   * **QLoRA**: LoRA on top of a *quantized* base. The base is compressed and
     frozen, while the adapter is trained at full precision.
   * **Rollout**: the generation phase of RL, where the model produces the
     responses that are then scored and learned from. AgileRL uses vLLM for this.
   * **Colocated**: the trainer and the vLLM rollout engine share a single GPU,
     rather than running on separate devices.
   * **KV cache**: scratch memory vLLM keeps per token while generating. It can
     be sizeable and competes for room with everything else on the GPU.

Why quantize?
-------------

* **Fits larger models on smaller GPUs.** A 4-bit base model is roughly a
  quarter of the BF16 footprint, so a model that would otherwise need an
  80 GB card can train on far less memory.
* **Leaves headroom for everything else.** LLM RL training has to hold the
  base weights, LoRA adapters, optimizer state, gradients, activations and
  (when vLLM is colocated) a KV cache, all on the same device. Shrinking the
  base frees room for a larger batch, longer context, or the vLLM KV pool.
* **Negligible quality cost with QLoRA.** Because only the LoRA adapters are
  trained (in BF16) while the quantized base stays frozen, 4-bit NF4
  quantization (NF4 is a 4-bit number format designed for neural-network
  weights) of the base recovers full-fine-tuning quality in practice. This is
  the `QLoRA <https://arxiv.org/abs/2305.14314>`_ recipe.
* **Faster rollout.** Quantized weights mean less memory bandwidth per
  forward pass, which speeds up generation on the vLLM side.

The cost is some quantization error on the frozen base. The defaults below
are chosen to keep that cost minimal.

How quantization fits the training loop
----------------------------------------

AgileRL decouples the **trainer** (a Hugging Face model that holds the
gradients and optimizer state) from the **rollout engine** (vLLM, used to
generate completions). They are quantized independently:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Component
     - Trainer (Hugging Face)
     - Rollout (vLLM)
   * - Weight quantization
     - bitsandbytes: ``int8`` or ``nf4`` (4-bit)
     - ``bitsandbytes`` (AgileRL-validated); other vLLM methods (AWQ, GPTQ,
       ...) forward verbatim via ``VLLMConfig.quantization`` but are not
       validated here
   * - What is synced
     - Trains the BF16 LoRA adapters
     - Receives **only** the LoRA adapters each step; base weights are never
       re-uploaded

When you train with QLoRA and a colocated vLLM rollout, the trainer and the
rollout engine each hold their own copy of the (quantized) base on the same
GPU and take turns using it (see :ref:`colocated_native_sleep` below). Every
training step only the small LoRA adapter weights are exported from the trainer
and loaded into vLLM (see :class:`~agilerl.utils.algo_utils.VLLMConfig` with
``enable_lora=True``, the default). The quantized base weights stay frozen in
4-bit; they are never dequantized, merged, or re-uploaded. This keeps the
per-step sync cheap.

.. _colocated_native_sleep:

Colocated rollout (native vLLM sleep/wake)
------------------------------------------

When vLLM is colocated with the trainer on the same GPU, the two share the GPU
*in time* rather than sharing weight storage. vLLM loads (and, with
``quantization="bitsandbytes"``, quantizes) its own base; the trainer loads its
own base too. Across each rollout↔train cycle at most one base is resident on
the GPU at a time:

* during **rollout**, the trainer's base is offloaded to CPU
  (``use_memory_efficient_params``, on by default for a colocated trainer) so
  the vLLM engine owns the GPU;
* between rollout and **training**, vLLM is put to sleep with its native
  ``sleep(level=1)`` call. That backs its base up to host (CPU) RAM and frees the
  KV cache; the trainer's base is then moved back onto the GPU for the forward
  and backward passes (the training step). ``wake_up()`` restores vLLM's base
  before the next rollout.

This relies on vLLM ``>= 0.22``, whose ``sleep(level=1)`` moves the base out to
host RAM and back without any loss of precision, for **both** dense (BF16/FP16)
and bitsandbytes 4-bit weights. (Earlier vLLM versions could not restore a 4-bit
base correctly, so this colocated path requires a recent vLLM; AgileRL pins
``vllm>=0.23``.) It happens automatically for colocated rollouts; there is
nothing to configure.

Two practicalities:

* **CUDA-safe init order.** bitsandbytes quantizes on the GPU during
  ``from_pretrained``; starting vLLM first can leave the CUDA allocator in a
  state where the trainer's subsequent device copies segfault. So for a fresh
  quantized trainer under ``sleep_mode`` the trainer is built first, offloaded
  to CPU, then vLLM starts.
* **Text-only rollouts for multimodal bases.** RL rollouts are text-only, so a
  multimodal base's unused vision/audio towers can be freed from vLLM's GPU
  memory with ``VLLMConfig(strip_multimodal_towers=True)`` (or a list of
  attribute names for non-standard layouts). Checkpoints are unaffected, since
  only the LoRA adapter is saved.

Quantizing the trainer (bitsandbytes + QLoRA)
---------------------------------------------

Pass a :class:`~transformers.BitsAndBytesConfig` as ``quantization_config``
to any LLM algorithm (:class:`~agilerl.algorithms.GRPO`,
:class:`~agilerl.algorithms.LLMPPO`, :class:`~agilerl.algorithms.SFT`,
:class:`~agilerl.algorithms.DPO`, :class:`~agilerl.algorithms.LLMREINFORCE`).
The base model is quantized as it is loaded; AgileRL then runs PEFT's k-bit
preprocessing (the standard step that readies a quantized model for LoRA
training) and attaches the trainable LoRA adapters on top (i.e. QLoRA).

.. code-block:: python

    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    from agilerl.algorithms import GRPO

    # 4-bit NF4: the QLoRA recipe.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )

    agent = GRPO(
        ...,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        lora_config=lora_config,
        quantization_config=quantization_config,
    )

Two presets cover the common cases. Rather than build the
``BitsAndBytesConfig`` by hand you can call
:func:`~agilerl.utils.llm_utils.build_bnb_quantization_config`, which also
accepts the spec declaratively from YAML / ``INIT_HP``:

.. list-table::
   :header-rows: 1
   :widths: 16 24 60

   * - Spec
     - Footprint
     - Notes
   * - ``"int8"``
     - ~1 byte/param
     - LLM.int8() 8-bit weights. Most conservative: smallest quality
       impact, larger than 4-bit.
   * - ``"nf4"``
     - ~0.56 byte/param
     - 4-bit NF4 with BF16 compute, BF16 quant storage and double
       quantization. The QLoRA recipe; ZeRO-3 compatible. Also accepted as
       ``"4bit"``, ``"4-bit"``, ``"bnb-4bit"``, or ``"bnb_4bit"``.
   * - ``None`` / ``"none"``
     - 2 bytes/param
     - No quantization (BF16 baseline).

.. code-block:: python

    from agilerl.utils.llm_utils import build_bnb_quantization_config

    quantization_config = build_bnb_quantization_config("nf4")  # or "int8"

A ``dict`` is also accepted and forwarded verbatim as
``BitsAndBytesConfig(**spec)`` for full control.

.. note::

   AgileRL currently fine-tunes **adapters only**: a ``lora_config`` is
   always used. Quantizing the base with bitsandbytes therefore means QLoRA:
   a frozen quantized base plus trainable BF16 LoRA adapters. The base is
   never updated, so its quantization error does not accumulate during
   training.

Works with DeepSpeed
~~~~~~~~~~~~~~~~~~~~~

The ``nf4`` preset sets ``bnb_4bit_quant_storage=torch.bfloat16`` specifically
so the quantized parameters present a floating-point storage dtype that
`DeepSpeed <https://www.deepspeed.ai/>`_ and
`Accelerate <https://huggingface.co/docs/accelerate/index>`_ can shard. QLoRA
training therefore composes with ZeRO (DeepSpeed's memory-saving strategy for
splitting model and optimizer state across GPUs, including its most aggressive
ZeRO-3 tier) and gradient checkpointing; no extra configuration is required
beyond the usual
:class:`~accelerate.Accelerator` / ``DeepSpeedPlugin`` setup. See
:ref:`distributed_training` for the distributed setup.

When the trainer is quantized and a colocated vLLM rollout runs in sleep
mode, AgileRL loads the bitsandbytes-quantized trainer **before** starting
vLLM (then offloads it to CPU during vLLM init). bitsandbytes runs its GPU
quantization kernels during ``from_pretrained``, and doing that after vLLM
has initialised the CUDA allocator can crash. The ordering is handled
internally; you don't need to do anything.

Quantizing the vLLM rollout
---------------------------

The rollout engine is configured through
:class:`~agilerl.utils.algo_utils.VLLMConfig`. The AgileRL-validated path is
``quantization="bitsandbytes"``, which pairs naturally with an ``nf4`` trainer
for colocated QLoRA rollouts:

.. code-block:: python

    from agilerl.algorithms import GRPO
    from agilerl.utils.algo_utils import VLLMConfig

    vllm_config = VLLMConfig(
        gpu_memory_utilization=0.3,
        quantization="bitsandbytes",   # AgileRL-validated path
        dtype="bfloat16",
        enable_lora=True,              # sync adapters only (default)
        max_lora_rank=16,              # >= trainer lora_config.r
    )

    agent = GRPO(..., use_vllm=True, vllm_config=vllm_config)

``VLLMConfig.quantization`` is forwarded verbatim to
``vllm.LLM(quantization=...)``. Any other vLLM-supported backend (e.g. ``"awq"``,
``"gptq"``, ``"fp8"``, ``"compressed-tensors"``) is accepted but is not validated
by AgileRL. Those
paths typically need a pre-quantized checkpoint, set separately via
``vllm_model_name_or_path``:

.. code-block:: python

    vllm_config = VLLMConfig(
        quantization="awq",
        vllm_model_name_or_path="my-org/Qwen2.5-7B-AWQ",
        enable_lora=True,
    )

``vllm_model_name_or_path`` lets the rollout load a different checkpoint from
the trainer (for example a bitsandbytes NF4 base on the trainer side and an
AWQ export for vLLM) while the LoRA adapters trained on the former are still
synced into the latter each step.

KV-cache quantization
---------------------

Storing the rollout KV cache in 8-bit (FP8) instead of 16-bit roughly halves its
memory, leaving more room for longer contexts or a bigger batch.
:class:`~agilerl.utils.algo_utils.VLLMConfig` exposes a bare ``kv_cache_dtype``
passthrough (forwarded verbatim to ``vllm.LLM(kv_cache_dtype=...)``) for users on
FP8-capable hardware (compute capability 8.9+, i.e. NVIDIA's Ada Lovelace,
Hopper, or Blackwell generations; "compute capability" is NVIDIA's version number
for a GPU's feature set). AgileRL does not validate any value here; vLLM emits its
own hardware errors / warnings.

On **A100 (Ampere, SM 8.0)** leave ``kv_cache_dtype`` unset; the FP8 paths
require Triton's ``fp8e4nv`` dtype, which is not implemented on Ampere and
the kernels fail to compile with errors such as ``fp8e4nv not supported``.

.. note::

   KV-cache quantization only affects the **rollout** KV pool; the trainer
   forward/backward pass does not use a persistent KV cache (with gradient
   checkpointing, ``use_cache=False`` is forced; the K/V tensors live
   transiently in the activation graph, where ``activation_offload`` is the
   relevant memory lever).

Putting it together
--------------------

A typical memory-constrained QLoRA + colocated-vLLM setup on A100:

.. code-block:: python

    import torch
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    from agilerl.algorithms import GRPO
    from agilerl.utils.algo_utils import VLLMConfig

    # Trainer: 4-bit NF4 base + BF16 LoRA adapters (QLoRA).
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear",
        task_type="CAUSAL_LM", lora_dropout=0.05,
    )

    # Rollout: bnb-quantized vLLM engine, adapters synced each step.
    vllm_config = VLLMConfig(
        gpu_memory_utilization=0.3,
        quantization="bitsandbytes",
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
    )

    agent = GRPO(
        ...,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        lora_config=lora_config,
        quantization_config=quantization_config,
        use_vllm=True,
        vllm_config=vllm_config,
    )

Here the base weights are quantized once on each side and frozen; only the
BF16 LoRA adapters are trained and synced into vLLM every step.
