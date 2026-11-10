# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Closed-form memory terms for LLM RL training and generation.

Every function is a pure function of the specs in
:mod:`agilerl.arena.memory.specs`. No torch, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

from agilerl.arena.memory.resolve.chunk import resolve_chunk_rows
from agilerl.arena.memory.specs import (
    CSA_COMPRESS_RATIO,
    DTYPE_BYTES,
    HCA_COMPRESS_RATIO,
    BlockKind,
    CheckpointQuantization,
    GiB,
    LoraTargetScope,
    MiB,
    ModelArch,
    TrainingSettings,
    WeightDtype,
    WeightVariant,
)

# Upper bound on the CUDA-graph pool the vLLM engine captures. It cancels in
# the generation total: subtracted from the KV budget, added back as residency.
CUDA_GRAPH_POOL_BYTES = 256 * MiB
# Extra non-KV memory vLLM reserves with 4-token n-gram drafting: its CUDA-graph
# estimate covers the draft-width batches, and the profile runs the rejection
# sampler. Takes 0.52-0.61 GiB of KV pool on Qwen3-4B, TP 1, A100-80.
SPECULATIVE_DRAFTING_BYTES = 640 * MiB
# Extra non-KV memory vLLM reserves when LoRA layers run on two CUDA streams
# (``VLLM_LORA_ENABLE_DUAL_STREAM``), mostly a larger CUDA-graph estimate. Takes
# 0.5-1.1 GiB of KV pool (median 0.66) on Qwen3-4B, Granite 3.1 MoE and
# Granite 4.0-H, bf16 and fp8, TP 1, A100-80 and H100.
LORA_DUAL_STREAM_BYTES = 768 * MiB
# vLLM's non-torch memory (kernel workspaces, loaded kernels, NCCL), as a share
# of the unsharded engine weights at the checkpoint dtype. It sits inside the
# gpu_memory_utilization budget and every rank holds it whole: it does not
# shrink with TP or fp8 weights.
ENGINE_RUNTIME_FRACTION = 0.036
# Ceiling on that share. vLLM 0.32 logs 2.31-2.49 GiB non-torch for
# Qwen3.6-35B-A3B, the largest fitted model, and 0.86-1.93 GiB for
# DeepSeek-V4-Flash, GLM-5.3-Flash and MiMo-V2.6-Flash at 530-585 GiB of bf16
# weights (H100, TP 4-8).
ENGINE_RUNTIME_MAX_BYTES = 2560 * MiB
# Misc broadcast bucket of an ``nccl_reshard`` refit, on trainer rank 0 and
# every engine GPU. Mirrors agilerl_ray's ``nccl_reshard.plan.MISC_BUCKET_BYTES``.
NCCL_RESHARD_MISC_BUCKET_BYTES = 512 * MiB
# Bytes per LoRA parameter under flat data parallel. PEFT
# ``autocast_adapter_dtype=True`` stores adapters and their gradients in
# fp32. AdamW keeps two moments per parameter in the parameter's dtype.
ADAPTER_BYTES_PER_PARAM = 4.0
ADAM_MOMENTS_PER_PARAM = 2
# Widest [rows, features] activation of one routed-expert row chunk, one fp32
# expert-parallel combine chunk, and one grouped LoRA GEMM output when no FSDP
# config sets one. Each chunk adds a fixed set of small kernel launches.
DEFAULT_ROUTED_CHUNK_BYTES = 64 * MiB
# Budgets an unset ``FSDPConfig.routed_expert_chunk_mib`` picks from, largest
# first. ROUTED_CHUNK_LIVE_FRACTION is not valid above 512 MiB.
ROUTED_CHUNK_MIB_CHOICES = (512, 256, 128, 64)
# Share of one chunk budget each chunked buffer adds to the MoE backward peak.
ROUTED_CHUNK_LIVE_FRACTION = 0.45
# Chunk-sized buffers the sorted path's contracted expert LoRA keeps live in
# the MoE backward: the up-projection output, its gradient and their casts.
CONTRACTED_CHUNK_LIVE_MULTIPLE = 4
# ``[routed rows, expert width]`` buffers expert parallel adds to the busiest
# rank's MoE backward peak: the send rows held for the comm stream, the
# received rows, the saved combine rows, and the gradients through both
# all-to-alls.
EP_EXCHANGE_BUFFERS = 4
# The same when uncheckpointed MoE blocks recompute their expert LoRA in
# backward. granite-3.1-3b-a800m and granite-4.0-h-tiny, A100 80GB, GC off,
# 18k-34k packed tokens: the granite-3.1 peaks sit a flat ~1.6 GiB above the
# saved tensors at every budget.
RECOMPUTED_CHUNK_LIVE_MULTIPLE = 7

# Share of a block's checkpointed-backward peak
# (:func:`~agilerl.arena.memory.block_activations.block_recompute_bytes`)
# that a block left out of checkpointing keeps from its forward: the backward
# peak also holds a gradient that only the block in backward carries. Qwen3-4B
# LoRA r16 on A100 80GB (FA2 packed, 12k and 15k-token budgets, frozen base
# sharded) kept 0.90; 0.91 puts the per-token slope 1.4% above it. Fit on that
# one dense model only; other families are extrapolated.
STORED_BLOCK_SHARE = 0.91

# PyTorch caching-allocator slack: reserved above allocated.
# Charged on the torch-side subtotal. Training only: generation sits in
# vLLM's CuMem pool, reserved up front at ``gpu_memory_utilization``.
ALLOCATOR_RESERVE_FRACTION = 0.07
# Flat data parallel keeps an all-reduce buffer.
ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL = 0.054
# FSDP compute stream; the all-gather stream's pool is charged separately
# (:func:`agilerl.arena.memory.fsdp_gather.fsdp_gather_pool_params`). Blocks without fp32 temporaries free
# bf16 tensors the next activations reuse, so little stays free.
ALLOCATOR_RESERVE_FRACTION_FSDP = 0.02
# FSDP compute stream when freed blocks do not fit the next tensors: stock
# fp32 RMSNorm temporaries leave hidden-sized holes, and EP dispatch buffers
# change size with routing.
ALLOCATOR_RESERVE_FRACTION_FSDP_MIXED_SIZES = 0.064
# Largest share by which a measured peak has exceeded the estimate. A GPU
# recommendation keeps this share of usable memory free.
MAX_UNDERPREDICTION = 0.089
# Host RAM a node keeps for the raylet, Ray's object store and env hosts.
NODE_HOST_RESERVE_BYTES = 8 * GiB
# Host RAM of one trainer process outside its learn buffers: Python, torch and
# the CUDA runtime. Measured at about 5 GiB on L4 trainers.
TRAINER_PROCESS_HOST_BYTES = 6 * GiB


def recommendation_fits(predicted_bytes: int, usable_bytes: int) -> bool:
    """Whether a predicted peak still fits after the underprediction buffer."""
    return predicted_bytes <= int(usable_bytes * (1 - MAX_UNDERPREDICTION))


def host_bytes_per_gpu(node_bytes: int, gpus_per_node: int) -> int:
    """Host RAM one GPU's process may hold: the node's RAM less its reserve, split over its GPUs.

    :param node_bytes: Host RAM one node gives the job.
    :param gpus_per_node: GPUs on that node.
    :return: Bytes per GPU, never negative.
    """
    return max(node_bytes - NODE_HOST_RESERVE_BYTES, 0) // gpus_per_node


def allocator_reserve_bytes(
    allocated_bytes: float,
    n_ranks: int = 1,
    sharded: bool = False,
    mixed_sizes: bool = False,
) -> float:
    """Caching-allocator reserved-but-unallocated bytes on the compute stream.

    :param mixed_sizes: The step frees tensors the next activations do not
        fit (fp32 norm temporaries, expert-parallel dispatch buffers).
    """
    if sharded and mixed_sizes:
        fraction = ALLOCATOR_RESERVE_FRACTION_FSDP_MIXED_SIZES
    elif sharded:
        fraction = ALLOCATOR_RESERVE_FRACTION_FSDP
    elif n_ranks > 1:
        fraction = ALLOCATOR_RESERVE_FRACTION_DATA_PARALLEL
    else:
        fraction = ALLOCATOR_RESERVE_FRACTION
    return max(allocated_bytes, 0.0) * fraction


@dataclass(frozen=True)
class ParamCounts:
    """Parameter counts split by matrix group.

    LoRA targeting acts on a subset of groups. ``moe_experts`` holds the
    fused 3D expert parameters transformers stores per layer; PEFT wraps
    only ``nn.Linear``.
    """

    embedding: int
    lm_head: int
    attention: int
    mlp: int
    norms: int
    multimodal_towers: int
    moe_experts: int = 0
    # Mamba-2 mixer parameters (in/out projections, conv, gate norm).
    mamba: int = 0
    # Signed residual vs the checkpoint parameter count, held at the
    # checkpoint dtype.
    unattributed: int = 0

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.lm_head
            + self.attention
            + self.mlp
            + self.norms
            + self.multimodal_towers
            + self.moe_experts
            + self.mamba
            + self.unattributed
        )


def geometry_gap_warning(counts: ParamCounts) -> str | None:
    """Warn when parsed geometry leaves a large share of parameters unattributed.

    :param counts: Analytic parameter split from :func:`param_counts`.
    :return: Warning text, or ``None`` when the gap is small.
    """
    gap_warn_fraction = 0.02
    if counts.total <= 0:
        return None
    gap = counts.unattributed / counts.total
    if gap < gap_warn_fraction:
        return None
    return (
        f"{gap:.1%} of this checkpoint's parameters are not accounted for by "
        "the parsed geometry. Weight bytes are exact (checkpoint metadata); "
        "activation, KV and LoRA terms come from the geometry, so they are low."
    )


def tp_local(n: int, tp: int) -> int:
    """How many of ``n`` equal pieces one tensor-parallel rank holds.

    A count that does not divide ``tp`` stays whole. Key/value heads and
    Mamba groups that cannot be split are replicated.
    """
    if tp <= 1 or n % tp != 0:
        return n
    return n // tp


def mamba_shared_group_rows_params(arch: ModelArch, tp: int) -> int:
    """Frozen Mamba-2 B/C rows each TP rank gathers once and keeps resident.

    Only when ``n_groups`` does not divide ``tp``: the ``in_proj`` and
    ``conv1d`` rows of every group, ``2 * n_groups * d_state`` per layer.
    """
    if (
        tp <= 1
        or not arch.tensor_parallel_plan
        or arch.mamba_gated_delta
        or arch.mamba_n_groups % tp == 0
    ):
        return 0
    rows = 2 * arch.mamba_n_groups * arch.mamba_d_state
    return arch.n_mamba_layers * rows * (arch.hidden_size + arch.mamba_d_conv + 1)


def context_parallel_shard_len(arch: ModelArch, seq_len: int, cp: int) -> int:
    """Tokens one context-parallel rank runs of a ``seq_len``-token row.

    The trainer pads the ``seq_len - 1`` action frame to a multiple of ``cp``:
    of ``cp`` scan chunks on gated-delta-net and of whole blocks on FlexAttention.
    """
    if cp == 1:
        return seq_len
    gated_delta = arch.mamba_gated_delta and arch.n_mamba_layers > 0
    round_to = cp * GATED_DELTA_CHUNK if gated_delta else cp
    if arch.flex_attention:
        round_to = math.lcm(round_to, FLEX_BLOCK_SIZE)
    return -(-(seq_len - 1) // round_to) * round_to // cp


def tensor_parallel_params(
    counts: ParamCounts, arch: ModelArch, tensor_parallel: int
) -> int:
    """Parameters placed on the tensor-parallel mesh.

    Each rank keeps 1/tp, which FSDP then shards over ``dp_shard * cp`` and
    gathers per unit. An untied ``lm_head`` is in the set for every
    family; attention, dense MLP, shared experts, Mamba, per-layer
    embeddings and planned tower linears only for families with a training
    TP plan, less :func:`tensor_parallel_whole_params`. Norms, token
    embeddings, the MoE router, and packed experts stay on the FSDP mesh.
    """
    head = 0 if arch.tied_embeddings else counts.lm_head
    if not arch.tensor_parallel_plan:
        return head
    _, router = moe_params_per_layer(arch)
    dense_mlp = counts.mlp - arch.moe_layers * router
    shared = arch.moe_layers * shared_expert_params_per_layer(arch)
    gate = attention_output_gate_params(counts, arch)
    per_layer_embed = max(counts.embedding - token_embedding_params(arch), 0)
    mixers = (
        counts.attention
        + counts.mamba
        - tensor_parallel_whole_params(arch, tensor_parallel)
    )
    return (
        mixers
        + dense_mlp
        + shared
        + head
        + gate
        + per_layer_embed
        + arch.tensor_parallel_tower_params
    )


def tensor_parallel_whole_params(arch: ModelArch, tensor_parallel: int) -> int:
    """Attention and KDA parameters a family TP plan keeps whole on every rank.

    The low-rank query and MLA KV down-projections with their norms, the KV
    compressors and indexers, key/value heads that do not divide ``tp``,
    KDA's low-rank forget and output gate down-projections, and each
    gated-delta mixer's per-head output norm.
    """
    h = arch.hidden_size
    per_layer = arch.q_lora_rank * (h + 1) if arch.q_lora_rank else 0
    if arch.kv_lora_rank:
        per_layer += h * (arch.kv_lora_rank + arch.qk_rope_head_dim)
        per_layer += arch.kv_lora_rank
    elif arch.low_rank_attention:
        if arch.n_kv_heads % tensor_parallel:
            kv = arch.n_kv_heads * arch.head_dim
            per_layer += h * kv * (1 if arch.shared_kv else 2)
            per_layer += kv if arch.shared_kv else 0
    whole = arch.attention_layers * per_layer + attention_extra_params(arch)
    if not arch.low_rank_attention:
        for kind in attention_layer_kinds(arch):
            if kind.kv_heads % tensor_parallel:
                width = kind.kv_heads * (kind.head_dim + kind.v_head_dim)
                whole += kind.kv_layers * width * (h + int(arch.attn_bias))
    if arch.mamba_kda:
        whole += arch.n_mamba_layers * 2 * h * arch.mamba_d_head
    if arch.mamba_gated_delta:
        whole += arch.n_mamba_layers * arch.mamba_d_head
    return whole


def attention_output_gate_params(counts: ParamCounts, arch: ModelArch) -> int:
    """Output-gate rows of ``q_proj``, which the parameter split leaves unattributed."""
    if not arch.attention_output_gate:
        return 0
    gate = arch.attention_layers * arch.n_heads * arch.head_dim * arch.hidden_size
    if arch.attn_bias:
        gate += arch.attention_layers * arch.n_heads * arch.head_dim
    return min(gate, max(counts.unattributed, 0))


def qkv_width_per_rank(arch: ModelArch, tp: int, context_parallel: int = 1) -> int:
    """q+k+v elements one tensor-parallel rank materialises.

    Query heads always split. Key/value heads split only when they divide
    ``tp``; otherwise every rank keeps them. Ulysses context parallel
    replicates a rank's key/value heads up to ``context_parallel`` when it
    holds fewer. MLA expands keys and values per query head.
    """
    if arch.kv_lora_rank:
        v_dim = arch.v_head_dim or arch.head_dim
        return tp_local(arch.n_heads, tp) * (2 * arch.head_dim + v_dim)
    head = arch.head_dim
    if arch.global_head_dim and arch.global_head_dim > head:
        head = arch.global_head_dim
    q = tp_local(arch.n_heads, tp) * head
    kv_heads = max(tp_local(arch.n_kv_heads, tp), context_parallel)
    return q + kv_heads * (head + (arch.v_head_dim or head))


def attention_linears(
    arch: ModelArch, tensor_parallel: int = 1
) -> list[tuple[int, int]]:
    """``(in_features, out_features)`` of one attention block's ``nn.Linear`` layers, in forward order.

    MLA projects keys and values from a shared latent plus one rope key.
    A grouped low-rank output's per-group projection returns one output per
    group, which a LoRA delta cannot match, so only the group-mixing linear
    is listed.
    """
    h = arch.hidden_size
    heads = tp_local(arch.n_heads, tensor_parallel)
    if not arch.low_rank_attention:
        head_dim = max(arch.head_dim, arch.global_head_dim or 0)
        v_dim = arch.v_head_dim or head_dim
        kv_heads = tp_local(arch.n_kv_heads, tensor_parallel)
        return [
            (h, heads * head_dim),
            (h, kv_heads * head_dim),
            (h, kv_heads * v_dim),
            (heads * v_dim, h),
        ]
    q_out = heads * arch.head_dim
    v_dim = arch.v_head_dim or arch.head_dim
    query = [(h, q_out)]
    if arch.q_lora_rank:
        query = [(h, arch.q_lora_rank), (arch.q_lora_rank, q_out)]
    if arch.kv_lora_rank:
        nope = arch.head_dim - arch.qk_rope_head_dim
        kv = [
            (h, arch.kv_lora_rank + arch.qk_rope_head_dim),
            (arch.kv_lora_rank, heads * (nope + v_dim)),
        ]
    else:
        kv_out = tp_local(arch.n_kv_heads, tensor_parallel) * arch.head_dim
        kv = [(h, kv_out)] * (1 if arch.shared_kv else 2)
    out = [(heads * v_dim, h)]
    if arch.o_groups:
        out = [(tp_local(arch.o_groups, tensor_parallel) * arch.o_lora_rank, h)]
    return query + kv + out


def attention_linear_names(arch: ModelArch) -> list[str]:
    """PEFT module names of :func:`attention_linears`, in the same order."""
    if not arch.low_rank_attention:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    query = ["q_a_proj", "q_b_proj"] if arch.q_lora_rank else ["q_proj"]
    if arch.kv_lora_rank:
        kv = ["kv_a_proj_with_mqa", "kv_b_proj"]
    else:
        kv = ["kv_proj"] if arch.shared_kv else ["k_proj", "v_proj"]
    return query + kv + ["o_b_proj" if arch.o_groups else "o_proj"]


def kv_compressor_params(hidden: int, width: int, rate: int, overlap: int) -> int:
    """Parameters of one DeepSeek-V4 KV compressor: KV and gate projections, position bias, norm.

    :param overlap: Windows each compressed entry reads; projections widen by it.
    """
    return 2 * hidden * overlap * width + rate * overlap * width + width


def attention_extra_params(arch: ModelArch) -> int:
    """KV-compressor and sparse-attention indexer parameters over the whole stack."""
    h = arch.hidden_size
    # CSA entries read two overlapping windows; HCA entries one.
    csa = kv_compressor_params(h, arch.head_dim, CSA_COMPRESS_RATIO, 2)
    hca = kv_compressor_params(h, arch.head_dim, HCA_COMPRESS_RATIO, 1)
    total = arch.n_csa_layers * csa + arch.n_hca_layers * hca
    if not arch.n_indexed_layers:
        return total
    ihd = arch.index_head_dim
    queries = arch.q_lora_rank * arch.index_n_heads * ihd + h * arch.index_n_heads
    if arch.n_csa_layers:
        keys = kv_compressor_params(h, ihd, CSA_COMPRESS_RATIO, 2)
    else:
        # Key projection, key-pool gate, and a LayerNorm weight and bias.
        keys = 2 * h * ihd + 2 * ihd
    return total + arch.n_indexed_layers * (queries + keys)


def hyper_connection_params(arch: ModelArch) -> int:
    """Pre- and post-block mixing parameters of hyper-connections over the whole stack."""
    m = arch.hc_mult
    if m <= 1:
        return 0
    mix = (2 + m) * m
    # Each mixer: a (mix, m * hidden) weight, a bias per output, three scales.
    per_mixer = mix * m * arch.hidden_size + mix + 3
    return arch.n_layers * 2 * per_mixer


def attention_params_per_layer(arch: ModelArch) -> int:
    """q/k/v/o parameters of one attention block."""
    if arch.low_rank_attention:
        params = sum(i * o for i, o in attention_linears(arch))
        # RMSNorms on the query latent and the KV latent or shared KV.
        params += arch.q_lora_rank + arch.kv_lora_rank
        if arch.shared_kv:
            params += arch.n_kv_heads * arch.head_dim
        if arch.o_groups:
            params += (
                arch.o_lora_rank * arch.n_heads * (arch.v_head_dim or arch.head_dim)
            )
        return params
    return int(standard_attention_params(arch) / max(arch.attention_layers, 1))


@dataclass(frozen=True)
class AttentionLayerKind:
    """Standard attention layers that share one head geometry."""

    layers: int
    # Layers with their own k/v projections; KV-shared layers have none.
    kv_layers: int
    kv_heads: int
    head_dim: int
    v_head_dim: int


def attention_layer_kinds(
    arch: ModelArch,
) -> tuple[AttentionLayerKind, AttentionLayerKind]:
    """Sliding-window and full-attention layers of a standard attention stack."""
    f = arch.sliding_window_layer_fraction if arch.sliding_window else 0.0
    sliding = round(f * arch.attention_layers)
    full = arch.attention_layers - sliding
    shared_full = arch.kv_shared_full_layers
    full_dim = arch.global_head_dim or arch.head_dim
    return (
        AttentionLayerKind(
            layers=sliding,
            kv_layers=sliding - (arch.n_kv_shared_layers - shared_full),
            kv_heads=arch.n_kv_heads,
            head_dim=arch.head_dim,
            v_head_dim=arch.v_head_dim or arch.head_dim,
        ),
        AttentionLayerKind(
            layers=full,
            kv_layers=full - shared_full,
            kv_heads=arch.global_n_kv_heads or arch.n_kv_heads,
            head_dim=full_dim,
            v_head_dim=arch.v_head_dim or full_dim,
        ),
    )


def standard_attention_params(arch: ModelArch) -> int:
    """q/k/v/o parameters of every layer of a standard (not low-rank) attention stack."""
    h = arch.hidden_size
    bias = int(arch.attn_bias)
    return sum(
        kind.layers * arch.n_heads * (kind.head_dim * (h + bias) + kind.v_head_dim * h)
        + kind.kv_layers
        * kind.kv_heads
        * (kind.head_dim + kind.v_head_dim)
        * (h + bias)
        for kind in attention_layer_kinds(arch)
    )


def adapter_linears(
    names: Sequence[str],
    linears: list[tuple[int, int]],
    excluded: frozenset[str],
) -> list[tuple[int, int]]:
    """The ``linears`` whose PEFT module ``names`` are not in ``excluded``."""
    return [
        linear
        for name, linear in zip(names, linears, strict=True)
        if name not in excluded
    ]


def kda_linears(arch: ModelArch, tensor_parallel: int = 1) -> list[tuple[int, int]]:
    """``(in_features, out_features)`` of one KDA mixer's ``nn.Linear`` layers, in forward order.

    q, k and v; the low-rank forget gate; beta; the low-rank output gate; o.
    """
    h, dim = arch.hidden_size, arch.mamba_d_head
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    width = heads * dim
    gate = [(h, dim), (dim, width)]
    return [(h, width)] * 3 + gate + [(h, heads)] + gate + [(width, h)]


# PEFT module names of :func:`kda_linears`, in the same order.
KDA_LINEAR_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
    "g_a_proj",
    "g_b_proj",
    "o_proj",
)


def mamba_params_per_layer(arch: ModelArch) -> int:
    """Parameters of one Mamba-2 mixer, from the same geometry as its state."""
    if not arch.n_mamba_layers:
        return 0
    if arch.mamba_kda:
        width = arch.mamba_n_heads * arch.mamba_d_head
        linears = sum(i * o for i, o in kda_linears(arch))
        # Bias-free conv, dt bias, A_log, and the per-head output norm.
        conv = arch.mamba_conv_dim * arch.mamba_d_conv
        return linears + conv + width + arch.mamba_n_heads + arch.mamba_d_head
    h = arch.hidden_size
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    if arch.mamba_gated_delta:
        # z, qkv, beta and decay projections; a bias-free conv; dt bias,
        # A_log, and the per-head output norm.
        in_proj = h * (d_inner + arch.mamba_conv_dim + 2 * arch.mamba_n_heads)
        conv = arch.mamba_conv_dim * arch.mamba_d_conv
        heads = 2 * arch.mamba_n_heads
        return in_proj + d_inner * h + conv + heads + arch.mamba_d_head
    in_proj = h * (d_inner + arch.mamba_conv_dim + arch.mamba_n_heads)
    out_proj = d_inner * h
    conv = arch.mamba_conv_dim * (arch.mamba_d_conv + 1)
    return int(in_proj + out_proj + conv + d_inner + 3 * arch.mamba_n_heads)


def mlp_params_per_layer(arch: ModelArch) -> int:
    """Dense MLP parameters of one FFN block."""
    matrices = 3 if arch.gated_mlp else 2
    return int(
        matrices * arch.hidden_size * arch.intermediate_size * arch.mlp_width_factor
    )


def routed_expert_params_per_layer(arch: ModelArch) -> int:
    """Packed routed-expert parameters of one MoE block, the part expert parallel splits."""
    n_experts = arch.n_experts
    if n_experts is None or n_experts <= 1:
        return 0
    mlp_matrices = 3 if arch.gated_mlp else 2
    expert_inter = arch.expert_intermediate_size or arch.intermediate_size
    return n_experts * mlp_matrices * arch.expert_width * expert_inter


def shared_expert_params_per_layer(arch: ModelArch) -> int:
    """Always-on shared-expert MLP parameters of one MoE block."""
    if not arch.is_moe:
        return 0
    mlp_matrices = 3 if arch.gated_mlp else 2
    shared_inter = (
        arch.shared_expert_intermediate_size
        or arch.expert_intermediate_size
        or arch.intermediate_size
    )
    return arch.n_shared_experts * mlp_matrices * arch.hidden_size * shared_inter


def moe_params_per_layer(arch: ModelArch) -> tuple[int, int]:
    """(expert parameters, router parameters) of one MoE block.

    Latent MoE adds the hidden <-> latent projections to the expert count.
    """
    n_experts = arch.n_experts
    if n_experts is None or n_experts <= 1:
        return 0, 0
    h = arch.hidden_size
    experts = routed_expert_params_per_layer(arch)
    experts += shared_expert_params_per_layer(arch)
    if arch.moe_latent_size:
        experts += 2 * h * arch.moe_latent_size
    return experts, h * n_experts


@lru_cache(maxsize=256)
def param_counts(arch: ModelArch, n_params: int | None = None) -> ParamCounts:
    """Analytic parameter counts from the architecture geometry.

    Each group is charged only on the layers that hold it: the full stack
    for a uniform model, the declared subsets for hybrids and
    block-exclusive layouts.

    :param n_params: Checkpoint total. The difference against the analytic
        sum goes into :attr:`ParamCounts.unattributed`.
    """
    h = arch.hidden_size
    experts_per_layer, router_per_layer = moe_params_per_layer(arch)
    mlp_per_layer = mlp_params_per_layer(arch)

    embedding = arch.vocab_size * h
    # Per-layer embeddings: one vector per layer per vocabulary entry.
    embedding += arch.per_layer_input_vocab * arch.n_layers * arch.per_layer_input_dim
    lm_head = 0 if arch.tied_embeddings else arch.vocab_size * h
    norms = arch.n_layers * 2 * h + h + hyper_connection_params(arch)
    counts = ParamCounts(
        embedding=embedding,
        lm_head=lm_head,
        attention=(
            arch.attention_layers * attention_params_per_layer(arch)
            if arch.low_rank_attention
            else standard_attention_params(arch)
        )
        + attention_extra_params(arch)
        + arch.attention_sink_layers * arch.n_heads,
        mlp=arch.mlp_layers * mlp_per_layer + arch.moe_layers * router_per_layer,
        norms=norms,
        multimodal_towers=arch.multimodal_tower_params,
        moe_experts=arch.moe_layers * experts_per_layer,
        mamba=arch.n_mamba_layers * mamba_params_per_layer(arch),
    )
    if n_params is None:
        return counts
    return replace(counts, unattributed=n_params - counts.total)


def weight_bytes(
    counts: ParamCounts,
    dtype: WeightDtype,
    variant: WeightVariant,
) -> int:
    """Realised base-weight bytes for one loaded copy.

    Analytic at the checkpoint dtype over the reconciled counts.
    ``stripped_multimodal`` drops the counted towers only.
    """
    towers = 0 if variant.stripped_multimodal else counts.multimodal_towers
    return int((counts.total - counts.multimodal_towers + towers) * DTYPE_BYTES[dtype])


def checkpoint_quantized_params(
    arch: ModelArch, counts: ParamCounts, stored: CheckpointQuantization
) -> tuple[int, int]:
    """(linear, routed-expert) parameters a pre-quantized checkpoint stores quantized.

    Linears are the attention and recurrent-mixer projections the checkpoint
    does not list as unconverted, DeepSeek-V4's per-group output projection,
    the dense MLPs, and the shared experts. KV compressors, indexers, the
    router, norms, embeddings, the LM head and multimodal towers count at the
    checkpoint dtype.
    """
    routed = arch.moe_layers * routed_expert_params_per_layer(arch)
    attention = sum(
        i * o
        for i, o in adapter_linears(
            attention_linear_names(arch),
            attention_linears(arch),
            stored.unquantized_attention,
        )
    )
    if arch.o_groups:
        attention += (
            arch.o_lora_rank * arch.n_heads * (arch.v_head_dim or arch.head_dim)
        )
    mixer = 0
    if arch.mamba_kda:
        mixer = sum(
            i * o
            for i, o in adapter_linears(
                KDA_LINEAR_NAMES, kda_linears(arch), stored.unquantized_mixer
            )
        )
    linears = (
        arch.attention_layers * attention
        + arch.n_mamba_layers * mixer
        + arch.mlp_layers * mlp_params_per_layer(arch)
        + counts.moe_experts
        - routed
    )
    return linears, routed


def engine_weight_bytes(
    arch: ModelArch,
    counts: ParamCounts,
    dtype: WeightDtype,
    variant: WeightVariant,
    quantization: Literal["none", "fp8"],
) -> int:
    """Base-weight bytes of one unsharded vLLM engine copy.

    vLLM serves a pre-quantized checkpoint as stored
    (:func:`checkpoint_quantized_params`), whatever ``quantization`` says.
    Otherwise its online fp8 stores linear, expert and Mamba projection
    weights at one byte. Embeddings, the LM head, norms and multimodal towers
    stay at ``dtype``.
    """
    full = weight_bytes(counts, dtype, variant)
    stored = arch.checkpoint_quantization
    if stored is not None:
        linears, experts = checkpoint_quantized_params(arch, counts, stored)
        return int(
            full
            - linears * (DTYPE_BYTES[dtype] - stored.linear_bytes)
            - experts * (DTYPE_BYTES[dtype] - stored.expert_bytes)
        )
    if quantization == "none":
        return full
    quantized = counts.attention + counts.mlp + counts.moe_experts + counts.mamba
    return int(full - quantized * (DTYPE_BYTES[dtype] - 1))


def block_kinds(arch: ModelArch) -> tuple[BlockKind, ...]:
    """Layer kinds a block-exclusive stack holds."""
    kinds: list[BlockKind] = []
    if arch.attention_layers:
        kinds.append("attention")
    if arch.mlp_layers or (arch.is_moe and arch.moe_layers):
        kinds.append("ffn")
    if arch.n_mamba_layers:
        kinds.append("mamba")
    return tuple(kinds)


def checkpointed_blocks(
    n_blocks: int, gradient_checkpointing: bool, every_n_blocks: int
) -> int:
    """Blocks activation checkpointing wraps: the first, then every ``every_n_blocks``-th.

    The remaining blocks keep their full activations until their backward.

    :param n_blocks: Blocks checkpointing may wrap.
    :param gradient_checkpointing: Activation checkpointing is on.
    :param every_n_blocks: Checkpoint stride over those blocks.
    :return: Checkpointed blocks.
    """
    if not gradient_checkpointing:
        return 0
    return -(-n_blocks // every_n_blocks)


def training_checkpointed_blocks(arch: ModelArch, settings: TrainingSettings) -> int:
    """Blocks the trainer checkpoints at the resolved stride.

    :param arch: Model architecture.
    :param settings: Training settings with checkpointing resolved.
    :return: Checkpointed blocks.
    """
    return checkpointed_blocks(
        arch.n_layers,
        settings.gradient_checkpointing is True,
        settings.checkpoint_every_n_blocks,
    )


def lora_linears(
    arch: ModelArch,
    scope: LoraTargetScope = "all-linear",
    tensor_parallel: int = 1,
    width_factor: float = 1.0,
) -> dict[BlockKind, list[tuple[int, int]]]:
    """``(in_features, out_features)`` of the linears PEFT wraps, per layer kind, in forward order.

    Routed experts live in fused 3D parameters (see ParamCounts.moe_experts),
    so the FFN kind tracks dense MLP blocks.

    :param arch: Decoder geometry.
    :param scope: LoRA target scope.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param width_factor: MLP intermediate width relative to ``intermediate_size``.
    :return: Linears of the attention, FFN and recurrent kinds; empty when unwrapped.
    """
    h = arch.hidden_size
    excluded = arch.lora_excluded_modules
    linears: dict[BlockKind, list[tuple[int, int]]] = {
        "attention": adapter_linears(
            attention_linear_names(arch),
            attention_linears(arch, tensor_parallel),
            excluded,
        ),
        "ffn": [],
        "mamba": [],
    }
    if scope != "all-linear":
        return linears
    if arch.mlp_layers:
        inter = tp_local(int(arch.intermediate_size * width_factor), tensor_parallel)
        linears["ffn"] = [(h, inter)] * (2 if arch.gated_mlp else 1) + [(inter, h)]
    if arch.mamba_kda:
        linears["mamba"] = adapter_linears(
            KDA_LINEAR_NAMES, kda_linears(arch, tensor_parallel), excluded
        )
    elif arch.n_mamba_layers:
        heads_out = arch.mamba_n_heads * arch.mamba_d_head
        # Gated-delta-net LoRA wraps four residual-stream in_proj linears and
        # out_proj; Mamba-2 LoRA wraps in_proj only.
        linears["mamba"] = (
            [(h, heads_out)] * 4 + [(heads_out, h)]
            if arch.mamba_gated_delta
            else [(h, heads_out + arch.mamba_conv_dim + arch.mamba_n_heads)]
        )
    return linears


def lora_input_cast_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    scope: LoraTargetScope = "all-linear",
    gradient_checkpointing: bool = True,
    layer: BlockKind | None = None,
    tensor_parallel: int = 1,
) -> int:
    """fp32 LoRA inputs and outputs of an fp32 adapter compute over bf16 activations.

    Each wrapped linear casts its input to fp32
    (``BaseTunerLayer._cast_input_dtype``) and ``lora_A`` keeps that copy
    for backward. ``lora_B`` returns fp32 and is scaled out of place before
    the cast back to bf16, so two fp32 outputs are live while it runs.

    Under gradient checkpointing this is the recomputed block's peak: the
    casts kept so far plus one linear's two fp32 outputs, at whichever
    linear makes that largest. ``layer`` picks one layer kind; otherwise the
    block runs every kind it holds (the widest of a block-exclusive stack).
    A mixer on a stack whose layers carry an MLP beside it is followed by
    that MLP. Without checkpointing it is every layer's kept casts, which
    stay live until that layer's backward.
    """
    tokens_bytes = rows * seq_len * DTYPE_BYTES["fp32"]
    if not gradient_checkpointing:
        linears = lora_linears(arch, scope, tensor_parallel, arch.mlp_width_factor)
        layers = {
            "attention": arch.attention_layers,
            "ffn": arch.mlp_layers,
            "mamba": arch.n_mamba_layers,
        }
        kept = sum(
            layers[kind] * sum(width for width, _ in kind_linears)
            for kind, kind_linears in linears.items()
        )
        return int(tokens_bytes * kept)
    by_kind = lora_linears(arch, scope, tensor_parallel, arch.peak_mlp_width_factor)
    attention = by_kind["attention"] if arch.attention_layers else []
    mlp, mamba = by_kind["ffn"], by_kind["mamba"]
    if layer is not None:
        block = by_kind[layer]
        if layer != "ffn" and not arch.block_exclusive_layers:
            block = block + mlp
        candidates = [block]
    elif arch.block_exclusive_layers:
        candidates = [attention, mlp, mamba]
    elif arch.sublayer_transients:
        candidates = [attention + mlp, mamba + mlp]
    else:
        candidates = [attention + mamba + mlp]
    peak = 0
    for block in candidates:
        kept = 0
        for width, out in block:
            kept += width
            peak = max(peak, kept + 2 * out)
    return int(tokens_bytes * peak)


def lora_fp32_output_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    scope: LoraTargetScope = "all-linear",
    tensor_parallel: int = 1,
) -> int:
    """The two fp32 ``lora_B`` outputs of the widest wrapped linear.

    Live beside the kept casts while a layer that is not recomputed runs
    forward or backward.
    """
    linears = lora_linears(arch, scope, tensor_parallel, arch.peak_mlp_width_factor)
    present = {
        "attention": arch.attention_layers,
        "ffn": arch.mlp_layers,
        "mamba": arch.n_mamba_layers,
    }
    widest = max(
        (out for kind, block in linears.items() if present[kind] for _, out in block),
        default=0,
    )
    return int(rows * seq_len * 2 * widest * DTYPE_BYTES["fp32"])


def lora_only_stored_input_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int = 1,
) -> int:
    """Mean activation-dtype bytes per stored block that only ``lora_A`` keeps.

    That is the MLP down-projection input; the frozen projection saves none.
    A stored block drops it when ``lora_A`` keeps an fp32 cast instead, or
    when the MLP carries no adapter. The other wrapped inputs (normed rows,
    attention output) stay saved by the norm and attention ops: Qwen3-4B
    with fp32 adapter compute on H100, uncheckpointed, 6k tokens.
    """
    ffn = lora_linears(arch, "all-linear", tensor_parallel, arch.mlp_width_factor)[
        "ffn"
    ]
    if not ffn:
        return 0
    down_in = ffn[-1][0]
    return int(rows * seq_len * down_in * act_bytes * arch.mlp_layers / arch.n_layers)


# Expert-intermediate widths one gpt-oss routed row peaks at, forward and
# backward: gate_up, the clamped gate, its sigmoid, the GLU, up + 1 and the
# gated output stay saved.
CLAMPED_SWIGLU_FORWARD_INTERS = 8
CLAMPED_SWIGLU_BACKWARD_INTERS = 9


def routed_expert_bytes(
    arch: ModelArch, rows: int, seq_len: int, act_bytes: float, backward: bool = False
) -> int:
    """Live routed-expert tensors in one MoE block on the grouped-GEMM path.

    Per routed token (``tokens x top_k``): the gathered input and the expert
    output (:attr:`ModelArch.expert_width` each), gate/up (``2 x inter``) and
    the activation (``inter``). A checkpointed backward adds the activation's
    gradient. Nothing outlives the block.

    ``arch.chunked_routed_experts`` runs experts in row chunks and adds
    each chunk into one fp32 ``[tokens, expert_width]`` layer output, whose
    fp32 gradient backward holds. Backward keeps the gathered input, the
    expert output, the up projection and the activation for every chunk; a
    no-grad pass keeps only the gathered input. The per-chunk gradient
    buffers are :func:`routed_chunk_bytes`.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    h = arch.expert_width
    inter = arch.expert_intermediate_size or arch.intermediate_size
    if arch.transposed_experts:
        inters = (
            CLAMPED_SWIGLU_BACKWARD_INTERS
            if backward
            else CLAMPED_SWIGLU_FORWARD_INTERS
        )
        return int(routed * (2 * h + inters * inter) * act_bytes)
    if not arch.chunked_routed_experts:
        return int(routed * (2 * h + (4 if backward else 3) * inter) * act_bytes)
    width = 2 * h + (3 if arch.gated_mlp else 2) * inter if backward else h
    combined = rows * seq_len * h * DTYPE_BYTES["fp32"]
    return int(routed * width * act_bytes + combined)


def saved_router_topk_bytes(arch: ModelArch, rows: int, seq_len: int) -> int:
    """Router top-k outputs every checkpointed MoE block saves for backward.

    Selective checkpointing keeps each router's ``(tokens, top_k)`` scores
    and int64 expert indices on device until that block's backward, so
    recompute replays the forward routing. Activation offload does not
    move them. Scores are charged at fp32, the widest a router picks on.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    slot_bytes = DTYPE_BYTES["fp32"] + 8
    return int(routed * arch.moe_layers * slot_bytes)


def moe_collectives_saveable(
    arch: ModelArch, expert_parallel: int, tensor_parallel: int
) -> bool:
    """Whether a checkpointed MoE block runs collectives ``fsdp.save_moe_collectives`` saves.

    :param expert_parallel: Expert-parallel degree.
    :param tensor_parallel: Degree the family's tensor-parallel plan shards by.
    :return: True under expert parallel, or under a tensor-parallel plan
        that shards a saved shared expert or latent projections inside the MoE.
    """
    if not arch.is_moe:
        return False
    if expert_parallel > 1:
        return True
    return tensor_parallel > 1 and (
        arch.shared_expert_reduce_saved or bool(arch.moe_latent_size)
    )


def saved_moe_collective_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    expert_parallel: int,
    routed_split: int,
    tensor_parallel: int,
) -> int:
    """Collective outputs one checkpointed MoE block saves for its backward.

    Selective checkpointing keeps the collectives the MoE child's recompute
    reaches, so it reuses them instead of communicating. Expert
    parallel saves the dispatched and the combined rows, ``tokens x top_k /
    routed_split`` each at :attr:`ModelArch.expert_width`. A routed split
    saves the all-gather of every rank's padded row span. A tensor-parallel
    plan saves the shared expert's all-reduce when
    :attr:`ModelArch.shared_expert_reduce_saved`, and on a latent MoE the
    gathered input projection and the reduced output projection.

    :param expert_parallel: Expert-parallel degree.
    :param routed_split: Tensor-parallel ranks that split each token's routing.
    :param tensor_parallel: Degree the family's tensor-parallel plan shards by.
    :return: Bytes per checkpointed MoE block.
    """
    if not arch.is_moe:
        return 0
    tokens = rows * seq_len
    width = arch.expert_width
    elements = 0
    if expert_parallel > 1:
        routed = tokens * (arch.n_experts_per_tok or 1) // routed_split
        elements += 2 * routed * width
        if routed_split > 1:
            elements += routed_split * -(-tokens // routed_split) * width
    if tensor_parallel > 1:
        if arch.shared_expert_reduce_saved:
            elements += tokens * arch.hidden_size
        if arch.moe_latent_size:
            elements += tokens * (arch.moe_latent_size + arch.hidden_size)
    return int(elements * act_bytes)


def checkpointed_moe_blocks(arch: ModelArch, settings: TrainingSettings) -> int:
    """MoE blocks among the blocks the trainer checkpoints.

    :param arch: Model architecture.
    :param settings: Training settings with checkpointing resolved.
    :return: Checkpointed MoE blocks.
    """
    if not arch.is_moe or settings.gradient_checkpointing is not True:
        return 0
    if not arch.layer_kinds:
        return training_checkpointed_blocks(arch, settings)
    stride = settings.checkpoint_every_n_blocks
    return sum(1 for kind in arch.layer_kinds[::stride] if kind == "moe")


def ep_exchange_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    expert_parallel: int,
    tensor_parallel: int = 1,
) -> int:
    """Token-exchange buffers of one expert-parallel MoE block at its backward peak.

    A rank sends its routed rows and receives about as many for its own
    experts. Tensor-parallel ranks share their tokens, so each routes 1/tp.
    """
    if not arch.is_moe or expert_parallel <= 1:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1) // tensor_parallel
    return int(EP_EXCHANGE_BUFFERS * routed * arch.expert_width * act_bytes)


def routed_chunk_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    chunk_bytes: int,
    expert_lora: bool,
    contracted: bool,
    ep: int,
    recomputed: bool = False,
) -> int:
    """Backward bytes of one MoE block that grow with the routed-expert chunk budget.

    Each buffer is capped by the rows it chunks. The chunked routed path
    (expert LoRA on ``arch.chunked_routed_experts``) and the expert-parallel
    fp32 combine each add :data:`ROUTED_CHUNK_LIVE_FRACTION` of a chunk. The
    sorted path's contracted LoRA holds
    :data:`CONTRACTED_CHUNK_LIVE_MULTIPLE` whole chunks, or
    :data:`RECOMPUTED_CHUNK_LIVE_MULTIPLE` when the experts recompute in
    backward; transposed experts (``arch.transposed_experts``) hold as many
    buffers over every routed row.

    :param chunk_bytes: Widest ``[rows, features]`` activation of one chunk.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param contracted: Expert LoRA runs split (``packed_moe_dispatch="contracted"``).
    :param ep: Expert-parallel degree.
    :param recomputed: Some MoE block runs outside checkpointing, so its
        expert LoRA recomputes the up-projection in backward.
    """
    if not arch.is_moe:
        return 0
    routed = rows * seq_len * (arch.n_experts_per_tok or 1)
    inter = arch.expert_intermediate_size or arch.intermediate_size
    widest_row = max(arch.expert_width, inter) * act_bytes
    fractional = 0.0
    whole = 0.0
    if expert_lora and arch.chunked_routed_experts:
        fractional += min(chunk_bytes, routed * widest_row)
    elif expert_lora and contracted:
        multiple = (
            RECOMPUTED_CHUNK_LIVE_MULTIPLE
            if recomputed
            else CONTRACTED_CHUNK_LIVE_MULTIPLE
        )
        held = routed * widest_row
        if not arch.transposed_experts:
            held = min(chunk_bytes, held)
        whole += multiple * held
    if ep > 1:
        fractional += min(chunk_bytes, routed * arch.expert_width * 4)
    return int(fractional * ROUTED_CHUNK_LIVE_FRACTION + whole)


def support_rows_bytes(
    arch: ModelArch, chunk_rows: int, width: int, act_bytes: float
) -> int:
    """Peak of one support-scored logprob chunk: its tokens' gathered lm_head rows.

    A chunk takes enough tokens for ``width`` head rows each to fill one vocab
    tile, or one token when its rows alone exceed the tile. Rows gather in the
    head's dtype; below fp32 the upcast holds both copies.

    :param chunk_rows: Tokens in one vocab tile.
    :param width: Widest sampling support, in token ids.
    :param act_bytes: Bytes per lm_head element.
    """
    tokens = max(1, chunk_rows * arch.vocab_size // (width * arch.hidden_size))
    elements = tokens * width * arch.hidden_size
    return int(elements * (act_bytes + 4 if act_bytes < 4 else act_bytes))


def logit_terms(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
) -> tuple[float, float]:
    """Loss-instant logits and no-grad head bytes."""
    # Two fp32 tiles are live at the loss instant: recomputed logits plus
    # that tile's gradient. Tiles are fixed-size, so a fused critic row adds
    # none. The head matmul reads the lm_head in its own dtype, so no copy of
    # it is live.
    logit_rows = resolve_chunk_rows(arch.vocab_size)
    if settings.sampling_support_width:
        support_rows = support_rows_bytes(
            arch, logit_rows, settings.sampling_support_width, act_bytes
        )
        return support_rows, support_rows
    return logit_rows * arch.vocab_size * 8, logit_rows * arch.vocab_size * 4


# Chunk length of fla's chunked gated-delta-rule kernel.
GATED_DELTA_CHUNK = 64

# FlexAttention block size; context-parallel widths on flex round to it.
FLEX_BLOCK_SIZE = 128

# Per-process job overhead charged when ``orchestrated`` is set.
JOB_OVERHEAD_BYTES = 50 * MiB

# Non-torch device memory a one-rank trainer holds beyond the CUDA context
# and job overhead. Multi-rank trainers charge :func:`agilerl.arena.memory.nccl.trainer_nccl_bytes`.
TRAINER_LIB_OVERHEAD_BYTES = 436 * MiB


def _lora_dropout_mixer_width(
    arch: ModelArch, excluded: frozenset[str], tensor_parallel: int
) -> int:
    """Input elements per token of one recurrent mixer's all-linear LoRA linears.

    :param excluded: Linear names LoRA leaves unwrapped.
    :param tensor_parallel: Degree the mixer's heads are split by.
    :return: Summed adapter input widths; zero without recurrent layers.
    """
    if arch.mamba_kda:
        kda = adapter_linears(
            KDA_LINEAR_NAMES, kda_linears(arch, tensor_parallel), excluded
        )
        return sum(i for i, _ in kda)
    if not arch.n_mamba_layers:
        return 0
    # Gated-delta-net LoRA wraps four residual-stream in_proj linears and
    # out_proj; Mamba-2 LoRA wraps in_proj only.
    h = arch.hidden_size
    return (
        4 * h + arch.mamba_n_heads * arch.mamba_d_head if arch.mamba_gated_delta else h
    )


def lora_dropout_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    adapter_bytes: float,
    scope: LoraTargetScope = "all-linear",
    gradient_checkpointing: bool = True,
    layer: BlockKind | None = None,
    tensor_parallel: int = 1,
) -> int:
    """Dropped-out adapter inputs and their masks that autograd keeps.

    With ``lora_dropout > 0`` PEFT feeds each wrapped linear's adapter a
    dropped-out copy of its input at the adapter dtype and keeps a one-byte
    mask for backward. Under gradient checkpointing one block's copies are
    live at once; otherwise every layer's are. When
    :attr:`ModelArch.sublayer_transients` holds, a block's peak sits in one
    attention, MLP or recurrent sub-layer: ``layer`` picks one, else the
    widest counts. A mixer on a stack whose layers carry an MLP beside it
    adds that MLP. A parallel mixer stack charges all three in one block.
    """
    h = arch.hidden_size
    excluded = arch.lora_excluded_modules
    # q/k/v read the residual stream, o reads the concatenated heads.
    head_out = tp_local(
        arch.n_heads * max(arch.head_dim, arch.global_head_dim or 0), tensor_parallel
    )
    inputs = {"q_proj": h, "k_proj": h, "v_proj": h, "o_proj": head_out}
    attention = sum(width for name, width in inputs.items() if name not in excluded)
    if arch.low_rank_attention:
        wrapped = adapter_linears(
            attention_linear_names(arch),
            attention_linears(arch, tensor_parallel),
            excluded,
        )
        attention = sum(i for i, _ in wrapped)
    mlp = 0
    if scope == "all-linear" and arch.mlp_layers:
        inter = tp_local(
            int(arch.intermediate_size * arch.peak_mlp_width_factor), tensor_parallel
        )
        mlp = (2 * h if arch.gated_mlp else h) + inter
    mamba = (
        _lora_dropout_mixer_width(arch, excluded, tensor_parallel)
        if scope == "all-linear"
        else 0
    )
    if layer is not None:
        width = {"attention": attention, "ffn": mlp, "mamba": mamba}[layer]
        if layer != "ffn" and not arch.block_exclusive_layers:
            width += mlp
    elif arch.block_exclusive_layers:
        width = max(attention, mlp, mamba)
    elif arch.sublayer_transients:
        width = max(attention, mamba) + mlp
    else:
        width = attention + mlp + mamba
    per_block = rows * seq_len * width * (adapter_bytes + 1)
    if gradient_checkpointing:
        return int(per_block)
    return int(per_block * arch.n_layers)


def token_embedding_params(arch: ModelArch) -> int:
    """Token-embedding elements replicated on every FSDP rank."""
    return arch.vocab_size * arch.hidden_size
