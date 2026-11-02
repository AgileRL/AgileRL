# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""KV cache and recurrent-state pages of the vLLM rollout engine.

Every function is a pure function of the specs in
:mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

from agilerl.arena.memory.specs import DTYPE_BYTES, ModelArch, WeightDtype


def kv_heads_per_rank(n_kv_heads: int, tensor_parallel: int) -> int:
    """KV heads stored on one vLLM rank.

    vLLM divides the heads by the tensor-parallel size. Fewer heads than
    ranks are replicated, so each rank keeps at least one.
    """
    return max(1, n_kv_heads // tensor_parallel)


def kv_storing_layers(arch: ModelArch) -> int:
    """Attention layers that keep their own K/V."""
    return arch.attention_layers - min(arch.n_kv_shared_layers, arch.attention_layers)


def kv_layer_head_dims(arch: ModelArch) -> tuple[float, int, int]:
    """(windowed layer fraction, sliding head dim, full-attention head dim).

    Full-attention layers may store a wider head. With no sliding window
    every storing layer uses the full-attention width.
    """
    window_dim = arch.head_dim
    full_dim = arch.global_head_dim or arch.head_dim
    if arch.sliding_window is None:
        return 0.0, window_dim, full_dim
    return arch.sliding_window_layer_fraction, window_dim, full_dim


def kv_cache_bytes_per_token(arch: ModelArch, weight_dtype: WeightDtype) -> float:
    """K + V bytes per cached token position across all layers.

    Layers that share an earlier layer's KV store nothing of their own.
    Hybrid Mamba layers contribute recurrent state only, constant in
    context length and counted by :func:`mamba_state_bytes`.

    Storing layers are charged at their own head width: sliding layers at
    :attr:`ModelArch.head_dim`, full-attention layers at
    :attr:`ModelArch.global_head_dim` when that is wider.
    """
    kv_bytes = DTYPE_BYTES[weight_dtype]
    windowed, window_dim, full_dim = kv_layer_head_dims(arch)
    mean_head_dim = windowed * window_dim + (1.0 - windowed) * full_dim
    return 2 * kv_storing_layers(arch) * arch.n_kv_heads * mean_head_dim * kv_bytes


# vLLM's default KV block size.
KV_BLOCK_SIZE_DEFAULT = 16
# Scheduler steps in flight under vLLM's async scheduling, each up to
# max_num_batched_tokens; sliding-window layers keep their blocks meanwhile.
VLLM_ASYNC_CONCURRENT_BATCHES = 2
# max_num_batched_tokens a rollout engine starts vLLM with when neither the
# manifest nor the model family sets it.
ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS = 8192
# vLLM's max_num_batched_tokens when the engine config is built with no usage context.
VLLM_ENGINE_MAX_NUM_BATCHED_TOKENS = 2048
# Alignment the hybrid KV block size is rounded up to.
MAMBA_KERNEL_BLOCK_ALIGNMENT = 16
# Recurrent-state blocks resident per sequence. vLLM prefix caching
# resolves every hybrid to align mode, which caps residency at two.
MAMBA_STATE_BLOCKS_PER_SEQ = 2
# Align-mode prefill checkpoint block a Mamba-2 mixer adds per sequence.
MAMBA2_PREFILL_CHECKPOINT_BLOCKS = 1


def mamba_page_bytes(arch: ModelArch, state_dtype: WeightDtype = "bf16") -> int:
    """One recurrent layer's state: the conv window plus the SSM state.

    The conv window uses ``state_dtype``; the SSM state uses
    :attr:`ModelArch.mamba_ssm_state_dtype`.
    """
    if not arch.is_hybrid_ssm:
        return 0
    conv = arch.mamba_conv_dim * max(arch.mamba_d_conv - 1, 0)
    recurrent = arch.mamba_n_heads * arch.mamba_d_head * arch.mamba_d_state
    return int(
        conv * DTYPE_BYTES[state_dtype]
        + recurrent * DTYPE_BYTES[arch.mamba_ssm_state_dtype]
    )


def shares_hybrid_pool(arch: ModelArch) -> bool:
    """Whether Mamba state and attention KV share one paged vLLM pool."""
    return arch.is_hybrid_ssm and arch.n_kv_heads > 0 and kv_storing_layers(arch) > 0


def attention_token_bytes(
    arch: ModelArch, weight_dtype: WeightDtype = "bf16", tensor_parallel: int = 1
) -> int:
    """K + V bytes one full-attention layer stores per token on one vLLM rank."""
    _, _, full_dim = kv_layer_head_dims(arch)
    heads = kv_heads_per_rank(arch.n_kv_heads, tensor_parallel)
    return int(2 * heads * full_dim * DTYPE_BYTES[weight_dtype])


def aligned_kv_block_size(
    arch: ModelArch,
    weight_dtype: WeightDtype = "bf16",
    tensor_parallel: int = 1,
) -> int:
    """Block size vLLM adopts so an attention page holds a Mamba page.

    A hybrid model puts both caches in one block pool, so the page sizes
    must match. vLLM raises the attention block size until one rank's
    attention page is at least as large as its Mamba page, then pads the
    Mamba page up to it. The block size is then hundreds or thousands of
    tokens, and every sequence's KV rounds up to that page.
    """
    if not shares_hybrid_pool(arch):
        return KV_BLOCK_SIZE_DEFAULT
    per_token_per_layer = attention_token_bytes(arch, weight_dtype, tensor_parallel)
    mamba_page = mamba_page_per_rank(arch, weight_dtype, tensor_parallel)
    tokens = -(-mamba_page // per_token_per_layer)
    alignment = MAMBA_KERNEL_BLOCK_ALIGNMENT
    return max(KV_BLOCK_SIZE_DEFAULT, -(-tokens // alignment) * alignment)


def hybrid_page_bytes(
    arch: ModelArch, weight_dtype: WeightDtype = "bf16", tensor_parallel: int = 1
) -> int:
    """One layer's page in a hybrid pool on one rank; Mamba pages pad up to it."""
    return aligned_kv_block_size(
        arch, weight_dtype, tensor_parallel
    ) * attention_token_bytes(arch, weight_dtype, tensor_parallel)


def hybrid_kv_groups(arch: ModelArch) -> tuple[int, int, int]:
    """(layers per group, attention groups, Mamba groups) of a hybrid KV pool.

    vLLM splits each layer type into groups the size of the smaller type, or
    of the larger when it is under 1.5x the smaller, and pads the last group
    of each type. One pool block holds one page for every layer of a group,
    and a request takes blocks from every group.
    """
    attention = kv_storing_layers(arch)
    mamba = arch.n_mamba_layers
    size = min(attention, mamba)
    if max(attention, mamba) < 1.5 * size:
        size = max(attention, mamba)
    return size, -(-attention // size), -(-mamba // size)


def mamba_state_blocks_per_seq(arch: ModelArch) -> int:
    """Recurrent-state blocks one sequence holds in each Mamba group."""
    if arch.mamba_gated_delta:
        return MAMBA_STATE_BLOCKS_PER_SEQ
    return MAMBA_STATE_BLOCKS_PER_SEQ + MAMBA2_PREFILL_CHECKPOINT_BLOCKS


def mamba_page_per_rank(
    arch: ModelArch, state_dtype: WeightDtype = "bf16", tensor_parallel: int = 1
) -> int:
    """One Mamba layer's state on one vLLM rank.

    vLLM shards the conv window and the SSM state by the tensor-parallel
    size. Heads must divide that size. Groups that do not divide it are
    replicated up to one group per rank, which vLLM only allows when there
    is a single group.
    """
    if tensor_parallel == 1 or not arch.is_hybrid_ssm:
        return mamba_page_bytes(arch, state_dtype)
    heads = arch.mamba_n_heads
    groups = arch.mamba_n_groups
    if heads % tensor_parallel != 0:
        msg = (
            f"mamba_num_heads ({heads}) must be divisible by "
            f"tensor_parallel_size ({tensor_parallel})"
        )
        raise ValueError(msg)
    if groups % tensor_parallel != 0 and groups != 1:
        msg = (
            f"mamba n_groups ({groups}) must divide tensor_parallel_size "
            f"({tensor_parallel}), or be 1"
        )
        raise ValueError(msg)
    if groups % tensor_parallel != 0:
        groups = tensor_parallel
    width = heads * arch.mamba_d_head
    conv_dim = (width + 2 * groups * arch.mamba_d_state) // tensor_parallel
    recurrent = (heads // tensor_parallel) * arch.mamba_d_head * arch.mamba_d_state
    return int(
        conv_dim * max(arch.mamba_d_conv - 1, 0) * DTYPE_BYTES[state_dtype]
        + recurrent * DTYPE_BYTES[arch.mamba_ssm_state_dtype]
    )


def mamba_state_bytes(
    arch: ModelArch,
    concurrency: int,
    state_dtype: WeightDtype = "bf16",
    tensor_parallel: int = 1,
) -> int:
    """Recurrent-state cache for a hybrid model's Mamba layers.

    Two tensors per recurrent layer per slot, matching vLLM's
    ``mamba2_state_shape``: a causal-conv window of ``(conv_dim, d_conv - 1)``
    and a recurrent state of ``(n_heads, d_head, d_state)``. Both are
    constant in context length, so this term scales with concurrency.
    ``tensor_parallel`` applies :func:`mamba_page_per_rank`. Beside attention
    each layer is charged the padded :func:`hybrid_page_bytes`, over the
    padded layer groups of :func:`hybrid_kv_groups`.
    """
    if not arch.is_hybrid_ssm:
        return 0
    blocks = concurrency * mamba_state_blocks_per_seq(arch)
    if not shares_hybrid_pool(arch):
        per_layer = mamba_page_per_rank(arch, state_dtype, tensor_parallel)
        return int(arch.n_mamba_layers * per_layer * blocks)
    size, _, mamba_groups = hybrid_kv_groups(arch)
    page = hybrid_page_bytes(arch, state_dtype, tensor_parallel)
    return int(mamba_groups * size * page * blocks)


def hybrid_kv_demand_bytes(
    arch: ModelArch,
    weight_dtype: WeightDtype,
    concurrency: int,
    seq_len: int,
    tensor_parallel: int = 1,
) -> int:
    """Attention KV one rank of a hybrid pool holds for every sequence at ``seq_len``.

    Each sequence takes whole blocks of :func:`aligned_kv_block_size` tokens
    from every attention group of :func:`hybrid_kv_groups`.
    """
    block = aligned_kv_block_size(arch, weight_dtype, tensor_parallel)
    size, attention_groups, _ = hybrid_kv_groups(arch)
    page = hybrid_page_bytes(arch, weight_dtype, tensor_parallel)
    return int(concurrency * attention_groups * size * -(-seq_len // block) * page)


def kv_cache_demand_bytes(
    arch: ModelArch,
    weight_dtype: WeightDtype,
    concurrency: int,
    seq_len: int,
    in_flight_tokens: int,
) -> int:
    """Worst-case KV bytes: every concurrent sequence at full context.

    Sliding-window layers hold, per sequence, the window less one token plus
    the scheduler's in-flight tokens, in whole blocks plus one, at the sliding
    head width; full-attention layers keep the full sequence at
    :attr:`ModelArch.global_head_dim`.

    :param in_flight_tokens: Tokens scheduled but not yet freed, vLLM's
        ``max_in_flight_tokens``.
    """
    kv_bytes = DTYPE_BYTES[weight_dtype]
    layer_kv = 2 * kv_storing_layers(arch) * arch.n_kv_heads * kv_bytes
    windowed, window_dim, full_dim = kv_layer_head_dims(arch)
    if arch.sliding_window is None:
        return int(concurrency * seq_len * layer_kv * full_dim)
    held = min(arch.sliding_window - 1 + in_flight_tokens, seq_len)
    window_len = (-(-held // KV_BLOCK_SIZE_DEFAULT) + 1) * KV_BLOCK_SIZE_DEFAULT
    return int(
        concurrency
        * layer_kv
        * (windowed * window_len * window_dim + (1.0 - windowed) * seq_len * full_dim)
    )


def resolve_max_num_batched_tokens(
    max_num_seqs: int, max_model_len: int, explicit: int | None = None
) -> int:
    """Scheduler-step token cap the async rollout engine starts vLLM with.

    Unset takes :data:`ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS`. Either value is
    capped at every sequence at full context.
    """
    requested = ROLLOUT_ENGINE_MAX_NUM_BATCHED_TOKENS if explicit is None else explicit
    return min(requested, max_num_seqs * max_model_len)
