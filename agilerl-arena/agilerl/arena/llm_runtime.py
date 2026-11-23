# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Trainer runtime choices the public framework needs without the memory estimator.

The training-memory search stays in ``agilerl_memory``. These helpers are the
values the trainer, LoRA kernels, and vLLM config read while a model is built.
"""

from __future__ import annotations

from typing import Literal

MiB = 1024**2
# Mean row length as a share of the row budget (0.64 countdown, 0.56 Sudoku),
# from Qwen3-4B LoRA on 8x A100-80.
ROW_FILL = 0.6
# Widest [rows, features] activation of one routed-expert row chunk, one fp32
# expert-parallel combine chunk, and one grouped LoRA GEMM output when no FSDP
# config sets one. Each chunk adds a fixed set of small kernel launches.
DEFAULT_ROUTED_CHUNK_BYTES = 64 * MiB
# Compute capability where the fused combined Mamba kernel returns NaN.
UNFUSED_MAMBA_SCAN_CAPABILITY = (8, 9)

# Sequences a colocated engine decodes when ``vllm_config`` sets none.
COLOCATED_MAX_NUM_SEQS = 16
# Share of the card an omitted ``gpu_memory_utilization`` reserves when
# generation shares the trainer GPU.
OMITTED_GPU_MEMORY_UTILIZATION = 0.3

# Ranks vLLM will accept as ``max_lora_rank``.
VLLM_ACCEPTED_MAX_LORA_RANKS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 320, 512)
# Rank an estimate sizes when it names no adapter.
OMITTED_MAX_LORA_RANK = 16

# Model types whose trainer attention backend is fixed whatever the device.
FAMILY_ATTN_IMPLEMENTATIONS: dict[str, str] = {
    "nemotron_h": "flash_attention_2",
    "nemotron_h_omni": "flash_attention_2",
    "gemma3": "flex_attention",
    "gemma3_text": "flex_attention",
    "gemma4": "flex_attention",
    "gemma4_text": "flex_attention",
    "gpt_oss": "flex_attention",
    "mimo_v2": "flex_attention",
    "deepseek_v4": "flex_attention",
    # Hugging Face's GLM-5 Next classes support no other backend.
    "glm5_next": "sdpa",
    "glm5_next_text": "sdpa",
}
# Backends the context-parallel (Ulysses) attention patches.
CONTEXT_PARALLEL_ATTN_IMPLEMENTATIONS = frozenset(
    {"flash_attention_2", "flex_attention"}
)
# Model types whose own context-parallel plan gathers the keys for each
# sequence shard, so context parallel runs on their pinned backend.
SEQUENCE_SHARD_ATTENTION_MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})
# Model types checked to train correctly with flash_attention_2 on CUDA and
# sdpa elsewhere. Every catalog model type is here or in the table above.
KERNEL_RULE_MODEL_TYPES = frozenset(
    {
        "granitemoe",
        "granitemoehybrid",
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3_5_moe",
    }
)

# Attention backends that build a padding-free (packed) forward. A dense backend
# builds an O(N^2) block-diagonal mask over the packed row instead, so packing
# under one is silently a padded forward.
PACKING_ATTN_IMPLEMENTATIONS = frozenset({"flash_attention_2", "flex_attention"})
# Model types whose recurrent mixers reset their scan and conv state at every
# packed document boundary.
BOUNDARY_RESET_MODEL_TYPES = frozenset(
    {
        "granitemoehybrid",
        "nemotron_h",
        "nemotron_h_omni",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    }
)
# Model types whose decoder builds attention masks without ``position_ids``. A
# flex packed forward isolates documents only through that mask.
POSITION_BLIND_MASK_MODEL_TYPES = frozenset({"gpt_oss", "granitemoehybrid"})

# Bounds on fused-logprob chunk rows. Floor: ``lm_head`` is re-read once per
# chunk. Ceiling: the fp32 logit tile grows linearly with rows.
FUSED_CHUNK_ROWS_MIN = 128
FUSED_CHUNK_ROWS_MAX = 4096
FUSED_LOGIT_TILE_BYTES = 256 * MiB

ExpertGemmBackend = Literal["grouped_mm", "triton"]

# Triton's half-precision ``tl.dot`` compiles from sm80.
TRITON_MIN_SM_MAJOR = 8
TRITON_DTYPES = frozenset({"bfloat16", "float16"})
# SM majors where ``torch.nn.functional.grouped_mm`` runs one fused kernel.
GROUPED_GEMM_SM_MAJORS = frozenset({9, 10})
# SM majors where the Triton kernels beat the fused ``grouped_mm``.
TRITON_OVER_GROUPED_GEMM_SM_MAJORS = frozenset({9})
GROUPED_GEMM_DTYPE = "bfloat16"
# ``torch.nn.functional.grouped_mm`` needs every inner-dim row stride on a 16-byte boundary.
GROUPED_GEMM_ALIGNMENT_BYTES = 16


def resolve_mini_batch_size(configured: int | None, per_rank: int) -> int:
    """Per-rank trajectories one optimizer step covers.

    :param configured: Pinned mini-batch, or ``None`` when it is omitted.
    :param per_rank: Trajectories that rank holds.
    :return: The pinned mini-batch, otherwise ``per_rank``.
    """
    if configured is not None:
        return configured
    return per_rank


def resolve_chunk_rows(vocab_size: int) -> int:
    """Fused-logprob chunk rows for a 256 MiB fp32 logit tile.

    :param vocab_size: ``lm_head`` width.
    :return: The tile size clamped to 128..4096.
    """
    rows = FUSED_LOGIT_TILE_BYTES // max(1, vocab_size * 4)
    return max(FUSED_CHUNK_ROWS_MIN, min(FUSED_CHUNK_ROWS_MAX, rows))


def resolve_attn_implementation(model_type: str | None, cuda: bool) -> str:
    """Attention backend the trainer loads the language model with.

    :param model_type: Hugging Face ``model_type`` of the checkpoint.
    :param cuda: The trainer runs on a CUDA device.
    :return: The family's backend, else ``flash_attention_2`` on CUDA, else
        ``sdpa``.
    """
    family = FAMILY_ATTN_IMPLEMENTATIONS.get(model_type or "")
    if family is not None:
        return family
    if cuda:
        return "flash_attention_2"
    return "sdpa"


def context_parallel_blocker(
    attn_implementation: str, cp: int, model_type: str | None = None
) -> str | None:
    """Why context parallel cannot run on this model and attention backend, or ``None`` when it can.

    :param attn_implementation: The language model's attention backend.
    :param cp: Context-parallel degree.
    :param model_type: Hugging Face ``model_type`` of the language model, or None.
    :return: A short reason, or ``None`` when ``cp`` is 1 or the model or
        backend has a context-parallel path.
    """
    if (
        cp == 1
        or attn_implementation in CONTEXT_PARALLEL_ATTN_IMPLEMENTATIONS
        or model_type in SEQUENCE_SHARD_ATTENTION_MODEL_TYPES
    ):
        return None
    return (
        f"fsdp.cp={cp} needs flash_attention_2 or flex_attention attention; "
        f"this model trains with {attn_implementation}"
    )


def leaks_recurrent_state(model_type: str | None, has_recurrent_layers: bool) -> bool:
    """Whether packing would carry recurrent state across documents of this model.

    :param model_type: Hugging Face ``model_type`` of the checkpoint.
    :param has_recurrent_layers: The decoder has Mamba or linear-attention layers.
    :return: ``True`` when it has recurrent layers that do not reset at boundaries.
    """
    return has_recurrent_layers and model_type not in BOUNDARY_RESET_MODEL_TYPES


def sequence_packing_blocker(
    supports_sequence_packing: bool,
    attn_implementation: str | None,
    row_copies: int,
    recurrent_state_leaks: bool,
    vision_language: bool,
    model_type: str | None,
) -> str | None:
    """Why the gradient forward cannot pack its sequences, or ``None`` when it packs.

    :param supports_sequence_packing: The algorithm builds a packed gradient forward.
    :param attn_implementation: The language model's attention backend.
    :param row_copies: Times the packed row is stacked, one per routed
        adapter. LLMPPO's fused actor and critic pass stacks 2.
    :param recurrent_state_leaks: A recurrent mixer would carry state across
        packed documents, as :func:`leaks_recurrent_state`.
    :param vision_language: The model has a vision tower.
    :param model_type: Hugging Face ``model_type`` of the language model.
    :return: A short reason, or ``None`` when packing applies.
    """
    if not supports_sequence_packing:
        return "this algorithm's gradient forward is padded"
    if attn_implementation not in PACKING_ATTN_IMPLEMENTATIONS:
        return (
            f"attn_implementation {attn_implementation!r} has no varlen or "
            "block-sparse path"
        )
    if (
        attn_implementation == "flex_attention"
        and model_type in POSITION_BLIND_MASK_MODEL_TYPES
    ):
        return (
            f"{model_type} builds its flex attention masks without position_ids, "
            "so packed documents would attend to each other"
        )
    # Text forwards hand FlashAttention-2 offsets spanning every stacked row.
    # A vision-language forward leaves its text layers to derive them from
    # position_ids, which FlashAttention-2 does for a single row only.
    if (
        attn_implementation == "flash_attention_2"
        and row_copies > 1
        and vision_language
    ):
        return (
            "flash_attention_2 derives a vision-language forward's packed "
            f"offsets from one row; this forward stacks {row_copies}"
        )
    if recurrent_state_leaks:
        return "recurrent mixers do not reset at packed document boundaries"
    return None


def resolve_max_lora_rank(adapter_rank: int) -> int:
    """vLLM ``max_lora_rank`` for an engine serving one adapter.

    :param adapter_rank: LoRA rank the engine serves.
    :return: Smallest accepted rank that covers the adapter rank.
    :raises ValueError: When the adapter rank is above every accepted rank.
    """
    largest = VLLM_ACCEPTED_MAX_LORA_RANKS[-1]
    if adapter_rank > largest:
        msg = (
            f"LoRA adapter rank {adapter_rank} needs a vLLM max_lora_rank above "
            f"the largest value vLLM accepts ({largest})"
        )
        raise ValueError(msg)
    return next(
        value for value in VLLM_ACCEPTED_MAX_LORA_RANKS if value >= adapter_rank
    )


def mamba_scan_is_unfused(compute_capability: tuple[int, int]) -> bool:
    """Whether Mamba layers train with HF's unfused path on this device.

    :param compute_capability: CUDA ``(major, minor)``.
    :return: ``True`` on compute capability 8.9 (Ada, e.g. L4), where the
        fused SSM kernels return NaN.
    """
    return compute_capability == UNFUSED_MAMBA_SCAN_CAPABILITY


def grouped_gemm_supported(compute_capability: tuple[int, int], dtype: str) -> bool:
    """Whether the fused grouped GEMM serves this device and dtype.

    :param compute_capability: CUDA ``(major, minor)``.
    :param dtype: Activation dtype name, e.g. ``bfloat16``.
    :return: ``True`` on SM 9.x or 10.x in bf16.
    """
    return (
        dtype == GROUPED_GEMM_DTYPE and compute_capability[0] in GROUPED_GEMM_SM_MAJORS
    )


def expert_gemm_backend(
    compute_capability: tuple[int, int], dtype: str, triton_installed: bool
) -> ExpertGemmBackend:
    """Kernel the per-expert GEMMs over expert-sorted rows run on a CUDA device.

    :param compute_capability: CUDA ``(major, minor)``.
    :param dtype: Activation dtype name, e.g. ``bfloat16``.
    :param triton_installed: Triton is importable.
    :return: ``triton`` on sm80+ in bf16 or fp16, except ``grouped_mm`` where
        :func:`grouped_gemm_supported` passes outside
        :data:`TRITON_OVER_GROUPED_GEMM_SM_MAJORS`. Without Triton,
        ``grouped_mm`` wherever it is supported.
    :raises RuntimeError: If neither kernel serves the device and dtype.
    """
    triton_runs = (
        triton_installed
        and compute_capability[0] >= TRITON_MIN_SM_MAJOR
        and dtype in TRITON_DTYPES
    )
    if grouped_gemm_supported(compute_capability, dtype) and not (
        triton_runs and compute_capability[0] in TRITON_OVER_GROUPED_GEMM_SM_MAJORS
    ):
        return "grouped_mm"
    if triton_runs:
        return "triton"
    major, minor = compute_capability
    msg = (
        f"MoE expert GEMMs have no kernel on SM {major}.{minor} in {dtype}"
        f"{'' if triton_installed else ' without Triton'}: they need SM 9.x or "
        "10.x in bfloat16, or sm80+ in bfloat16 or float16 with Triton installed"
    )
    raise RuntimeError(msg)


def padded_expert_lora_rank(rank: int, element_size: int) -> int:
    """Smallest expert LoRA rank at or above ``rank`` the grouped GEMM accepts.

    :param rank: Configured LoRA rank.
    :param element_size: Bytes per element of the LoRA factors (2 for bf16).
    :return: ``rank`` rounded up to a multiple of ``16 // element_size``.
    """
    multiple = GROUPED_GEMM_ALIGNMENT_BYTES // element_size
    return -(-rank // multiple) * multiple


def resolve_moe_lora_recompute(checkpointed: bool) -> bool:
    """Whether one routed-experts LoRA block recomputes its up-projection in backward.

    :param checkpointed: The block holding the experts is activation-checkpointed,
        so its backward already reruns the expert forward.
    :return: ``not checkpointed``.
    """
    return not checkpointed
