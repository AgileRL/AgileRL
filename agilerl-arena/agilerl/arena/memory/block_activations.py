# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Per-block activation terms of the trainer's decoder stack.

Transient peaks of one recomputed or no-grad block (attention, MLP and
recurrent mixers) and the tensors an uncheckpointed block keeps for backward.
Pure functions of the specs in :mod:`agilerl.arena.memory.specs`.
"""

from __future__ import annotations

from collections.abc import Sequence

from agilerl.arena.memory.formulas import (
    GATED_DELTA_CHUNK,
    STORED_BLOCK_SHARE,
    lora_only_stored_input_bytes,
    qkv_width_per_rank,
    routed_expert_bytes,
    tp_local,
    training_checkpointed_layers,
)
from agilerl.arena.memory.specs import (
    CSA_COMPRESS_RATIO,
    DTYPE_BYTES,
    HCA_COMPRESS_RATIO,
    BlockKind,
    LoraTargetScope,
    ModelArch,
    TrainingSettings,
)

# Extra backward bytes per token per Mamba inner channel when the mixer runs
# HF's unfused path: separate conv, chunk scan, gated norm and output
# projection, with the SSM inputs upcast to fp32.
UNFUSED_MAMBA_SCAN_BYTES = 28

# fp32 elements in one query chunk of DeepSeek-V4's no-grad indexer scores.
INDEXER_SCORE_ELEMENTS = 1 << 28


def mamba_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    unfused_scan: bool = False,
    tensor_parallel: int = 1,
) -> int:
    """Live tensors of one Mamba-2 layer's chunked scan in the trainer.

    Per token: the ``in_proj`` output, the conv1d output, the gated scan
    input and an fp32 copy of the residual stream. Per chunk: the fp32 SSM
    state ``(heads, d_head, d_state)`` and the ``chunk x chunk`` CB matrix
    per group. A checkpointed backward adds the gated-norm gradient, fp32
    gradients of the ``in_proj`` output and the residual, and two more
    state buffers.

    ``unfused_scan`` adds the backward of HF's unfused path (separate conv,
    chunk scan, gated norm and output projection, the SSM inputs upcast to
    fp32), :data:`UNFUSED_MAMBA_SCAN_BYTES` per token per inner channel.
    """
    if not arch.n_mamba_layers:
        return 0
    tokens = rows * seq_len
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    groups = tp_local(arch.mamba_n_groups, tensor_parallel)
    d_inner = heads * arch.mamba_d_head
    conv_dim = arch.mamba_conv_dim
    if heads != arch.mamba_n_heads and arch.mamba_n_heads:
        conv_dim = arch.mamba_conv_dim * heads // arch.mamba_n_heads
    in_proj_out = d_inner + conv_dim + heads
    chunks = rows * -(-seq_len // arch.mamba_chunk_size)
    state = chunks * heads * arch.mamba_d_head * arch.mamba_d_state * 4
    cb = chunks * groups * arch.mamba_chunk_size**2 * 4
    per_token = (in_proj_out + arch.mamba_conv_dim + d_inner) * act_bytes
    per_token += arch.hidden_size * 4
    total = tokens * per_token + state + cb
    if backward:
        per_token_grads = d_inner * act_bytes + (in_proj_out + arch.hidden_size) * 4
        total += tokens * per_token_grads + 2 * state
    if unfused_scan and backward:
        total += tokens * d_inner * UNFUSED_MAMBA_SCAN_BYTES
    return int(total)


def gated_delta_conv_dim(arch: ModelArch, tensor_parallel: int = 1) -> int:
    """Conv channels of one rank's gated-delta-net heads; tensor parallel splits every head group."""
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    return arch.mamba_conv_dim * heads // arch.mamba_n_heads


def gated_delta_qk_width(arch: ModelArch, tensor_parallel: int = 1) -> int:
    """Per-token width of one rank's gated-delta-net query (or key): key heads x key head dim.

    The conv channels are ``[q | k | v]``, so this is what is left of the
    rank's conv channels after its value heads, halved.
    """
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    return (
        gated_delta_conv_dim(arch, tensor_parallel) - heads * arch.mamba_d_head
    ) // 2


def gated_delta_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    tensor_parallel: int = 1,
) -> int:
    """Live tensors of one gated-delta-net layer's fla chunk kernel in the trainer.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param backward: Whether the layer runs a checkpointed backward.
    :param tensor_parallel: Degree the mixer's heads are split by.
    :return: Peak bytes of the kernel's tensors; zero without gated-delta layers.
    """
    if not (arch.n_mamba_layers and arch.mamba_gated_delta):
        return 0
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    conv_dim = gated_delta_conv_dim(arch, tensor_parallel)
    qk = gated_delta_qk_width(arch, tensor_parallel)
    key = heads * arch.mamba_d_state
    value = heads * arch.mamba_d_head
    state = heads * arch.mamba_d_state * arch.mamba_d_head / GATED_DELTA_CHUNK
    tokens = rows * seq_len
    if not backward or arch.mamba_kda:
        # Per token: conv1d output, contiguous q/k at the key-head count and v,
        # l2-normed q and k, gate z, WY w and u, updated values, output, the
        # intra-chunk matrix, and one (d_state, d_head) state per head per chunk.
        width = conv_dim + 4 * qk + key + 5 * value + heads * GATED_DELTA_CHUNK + state
        # An unmeasured KDA backward also keeps the in_proj_qkv output and its gate.
        if backward:
            width += conv_dim + value
        return int(tokens * width * act_bytes)
    key_heads = qk // arch.mamba_d_state
    gqa_key_grad = qk if qk < key else 0
    # Per token in fla's chunk backward.
    width = (
        # in_proj_qkv output
        conv_dim
        # l2-normed q and k and their gradients, at the key-head count
        + 4 * qk
        # WY key gradient summed to the key heads when value heads outnumber them
        + gqa_key_grad
        # WY w, its gradient and the WY key gradient, at the value-head count
        + 3 * key
        # contiguous v, WY u, updated values, the gated RMSNorm's input and
        # gate gradients, and two value gradients
        + 7 * value
        # intra-chunk matrix
        + heads * GATED_DELTA_CHUNK
        # chunk states and their gradient
        + 2 * state
        # beta and its gradient
        + 2 * heads
    )
    # The gate, its chunk cumsum, two gate gradients, and q and k's l2-norm rstd.
    fp32 = (4 * heads + 2 * key_heads) * DTYPE_BYTES["fp32"]
    return int(tokens * (width * act_bytes + fp32))


def gated_delta_saved_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int = 1,
) -> int:
    """Tensors a recomputed gated-delta-net layer keeps for its own backward.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param tensor_parallel: Degree the mixer's heads are split by.
    :return: Saved bytes; zero without gated-delta layers.
    """
    if not (arch.n_mamba_layers and arch.mamba_gated_delta):
        return 0
    heads = tp_local(arch.mamba_n_heads, tensor_parallel)
    qk = gated_delta_qk_width(arch, tensor_parallel)
    value = heads * arch.mamba_d_head
    # Per token: in_proj_qkv output, l2-normed q and k at the key-head
    # count, contiguous v, the intra-chunk matrix, and fla's fused gated
    # RMSNorm's input, gate and output.
    width = (
        gated_delta_conv_dim(arch, tensor_parallel)
        + 2 * qk
        + 4 * value
        + heads * GATED_DELTA_CHUNK
    )
    if arch.mamba_kda:
        # An unmeasured KDA layer also keeps one fp32 RMSNorm rstd per head.
        return int(rows * seq_len * (width * act_bytes + heads * DTYPE_BYTES["fp32"]))
    # Beta; in fp32 the gate, its chunk cumsum, one RMSNorm rstd per head
    # and q and k's l2-norm rstd.
    width += heads
    fp32 = (3 * heads + 2 * (qk // arch.mamba_d_state)) * DTYPE_BYTES["fp32"]
    return int(rows * seq_len * (width * act_bytes + fp32))


def _block_mlp_recompute(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    engine_forward: bool,
    tensor_parallel: int,
    routed_split: int = 1,
    lora_target_scope: LoraTargetScope | None = None,
) -> tuple[int, int]:
    """Per-token MLP width and routed-expert bytes of one block recompute.

    :param routed_split: Ranks that share one sequence's routed tokens
        (TP ranks under expert parallel).
    :param lora_target_scope: LoRA targets; prices a dense gated MLP's
        checkpointed backward. MoE models ignore it.
    :return: ``(mlp_width, routed_bytes)``. ``mlp_width`` is elements per
        token. ``routed_bytes`` is the grouped-GEMM workspace; zero on the
        engine path and on dense models.
    """
    mlp_matrices = 3 if arch.gated_mlp else 2
    routed = 0
    if arch.is_moe:
        inter = arch.expert_intermediate_size or arch.intermediate_size
        shared_inter = arch.shared_expert_intermediate_size or inter
        # The always-on shared expert runs at its own (possibly wider) width.
        active_width = tp_local(arch.n_shared_experts * shared_inter, tensor_parallel)
        if engine_forward:
            active_width += (arch.n_experts_per_tok or 1) * inter
        else:
            routed = routed_expert_bytes(
                arch, rows, -(-seq_len // routed_split), act_bytes, backward
            )
    else:
        active_width = tp_local(
            int(arch.intermediate_size * arch.peak_mlp_width_factor), tensor_parallel
        )
    if (
        backward
        and arch.gated_mlp
        and not arch.is_moe
        and lora_target_scope is not None
    ):
        # Dense gated LoRA backward: the MLP input, gate, up and the activated
        # gate when the MLP is not fused; then the fused output, or with
        # adapters on the MLP the down projection's base and LoRA input
        # gradients and their sum.
        saved = 3 if arch.stock_gated_mlp else 2
        grads = 3 if lora_target_scope == "all-linear" else 1
        return (saved + grads) * active_width + arch.hidden_size, routed
    # Gated backward: gate, up, fused output, and one gradient, plus the
    # activated gate when the MLP is not fused.
    mlp_count = mlp_matrices
    if backward and arch.gated_mlp:
        mlp_count += 2 if arch.stock_gated_mlp else 1
    mlp = mlp_count * active_width
    if backward and not arch.gated_mlp:
        # Activation plus its gradient for each of the two matrices.
        mlp *= 2
    return mlp, routed


def compressed_entry_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    document_len: int,
    context_parallel: int = 1,
) -> int:
    """Per-query indexer selection of one DeepSeek-V4 CSA attention layer.

    Each query sees the entries of its own document, one per
    ``CSA_COMPRESS_RATIO`` tokens. A document with at most ``index_topk``
    entries is masked by blocks alone and holds nothing per query. Past that,
    each query keeps a boolean per entry of its row, and the no-grad indexer
    scores one document's queries in chunks of at most
    :data:`INDEXER_SCORE_ELEMENTS` fp32 scores per head and entry, plus their
    fp32 sum over heads. Under context parallel the row is one document of
    every shard's entries.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Query tokens per sequence on this rank.
    :param document_len: Longest document in a sequence; packed rows hold several.
    :param context_parallel: Ranks the sequence is sharded over.
    :return: Bytes, or 0 without CSA layers or without a scored document.
    """
    if not arch.n_csa_layers:
        return 0
    row_tokens = seq_len * context_parallel
    document_tokens = row_tokens if context_parallel > 1 else min(seq_len, document_len)
    entries = document_tokens // CSA_COMPRESS_RATIO
    if entries <= arch.index_topk:
        return 0
    visible = rows * seq_len * (row_tokens // CSA_COMPRESS_RATIO)
    queries = seq_len if context_parallel > 1 else document_tokens
    chunk = max(1, INDEXER_SCORE_ELEMENTS // (rows * arch.index_n_heads * entries))
    scored = rows * min(queries, chunk) * entries
    return int(visible + scored * (arch.index_n_heads + 1) * DTYPE_BYTES["fp32"])


def pooled_index_bytes(
    arch: ModelArch, rows: int, seq_len: int, context_parallel: int = 1
) -> int:
    """No-grad k-pool indexer scores and top-k mask of one GLM DSA layer.

    Each query scores one pool per ``index_kpool`` keys in fp32 per indexer
    head, holding the product, its scaled and relu copies, and a boolean
    visibility row over the keys. The top-k mask that follows holds one
    boolean per key, no more than the scoring. Under context parallel a
    rank's queries score the keys of every shard.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Query tokens per sequence on this rank.
    :param context_parallel: Ranks the sequence is sharded over.
    :return: Bytes, or 0 without a k-pool indexer.
    """
    if not arch.index_kpool:
        return 0
    keys = seq_len * context_parallel
    scores = 3 * arch.index_n_heads * (keys // arch.index_kpool) * DTYPE_BYTES["fp32"]
    return int(rows * seq_len * (scores + keys))


def context_parallel_key_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int = 1,
    backward: bool = False,
    context_parallel: int = 1,
) -> int:
    """Whole-sequence key side one key-gather attention layer holds on each context-parallel rank.

    MLA gathers the latent KV with its rope key and expands keys and values
    per local query head; its DSA indexer gathers packed no-grad keys. The
    backward adds each tensor's gradient, the reduce-scatter's stacked copy of
    the widest gather, and the ``1 / context_parallel`` share it receives.

    DeepSeek-V4 pools each rank's own windows, then one all-gather carries the
    sliding KV, the widest compressor's entries and, when ``index_topk``
    cannot keep every entry, the CSA indexer's no-grad keys. The forward
    holds three copies of the sliding KV and entries (the gathered pieces,
    their contiguous copies, and the keys with the entries appended) and the
    indexer keys once. The backward holds three as well: the saved keys and
    their key and value gradients, then the gather's gradient and its
    reduce-scatter copy, which also span the indexer keys. Ranks
    also gather the last ``rate - 1`` tokens of each shard's compressor
    projections (``2 * rate - 1`` for CSA's overlapping windows) when a
    window straddles a shard boundary.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens of the whole sequence.
    :param act_bytes: Bytes per activation element.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param backward: Size the checkpointed backward.
    :param context_parallel: Ranks the sequence is sharded over.
    :return: Bytes; 0 for families whose attention runs Ulysses.
    """
    if not arch.context_parallel_key_gather:
        return 0
    if arch.kv_lora_rank:
        gathered = arch.kv_lora_rank + arch.qk_rope_head_dim
        v_dim = arch.v_head_dim or arch.head_dim
        with_grad = gathered + tp_local(arch.n_heads, tensor_parallel) * (
            arch.head_dim + v_dim
        )
        no_grad = 2 * arch.index_head_dim + 1 if arch.n_indexed_layers else 0
        width = (
            2 * with_grad + gathered + gathered / context_parallel
            if backward
            else with_grad + no_grad
        )
        return int(rows * seq_len * width * act_bytes)
    kv = arch.n_kv_heads * arch.head_dim
    if not (arch.n_csa_layers or arch.n_hca_layers):
        return int(rows * seq_len * (4 * kv if backward else kv) * act_bytes)
    # CSA pools two overlapping windows per entry, HCA one.
    rate, windows = (
        (CSA_COMPRESS_RATIO, 2) if arch.n_csa_layers else (HCA_COMPRESS_RATIO, 1)
    )
    scored = arch.n_csa_layers > 0 and seq_len // rate > arch.index_topk
    index_keys = arch.index_head_dim / rate if scored else 0
    keys = kv + arch.head_dim / rate
    width = 3 * keys + (2 if backward else 1) * index_keys
    shard_len = seq_len // context_parallel
    straddles = windows == 2 or shard_len % rate
    tail = (
        min(windows * rate - 1, shard_len) if context_parallel > 1 and straddles else 0
    )
    projected = 2 * windows * arch.head_dim + (4 * arch.index_head_dim if scored else 0)
    boundary = context_parallel * tail * projected
    return int(rows * (seq_len * width + boundary) * act_bytes)


def _block_mixer_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    residual: int,
    backward: bool,
    unfused_mamba_scan: bool,
    tensor_parallel: int,
) -> float:
    """Recurrent-mixer transients of one trainer block.

    :param residual: Residual-stream elements per token the block holds.
    :return: The gated-delta-net kernel plus the layer's residual copy, else
        the Mamba-2 chunked scan.
    """
    if arch.mamba_gated_delta:
        return rows * seq_len * residual * act_bytes + gated_delta_block_bytes(
            arch, rows, seq_len, act_bytes, backward, tensor_parallel
        )
    return mamba_block_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        backward,
        unfused_scan=unfused_mamba_scan,
        tensor_parallel=tensor_parallel,
    )


def _attention_saved_width(
    arch: ModelArch, tensor_parallel: int, context_parallel: int
) -> int:
    """Elements per token an attention that projects q, k and v from hidden saves for its backward.

    A checkpointed Qwen3.5-2B or 4B full-attention layer holds these beside
    its FFN's backward: its measured backward peak is 0.3% over the price
    (LoRA r=16 on every linear or on attention only).
    """
    q = tp_local(arch.n_heads, tensor_parallel) * arch.head_dim
    kv = (qkv_width_per_rank(arch, tensor_parallel, context_parallel) - q) // 2
    # Per token: the rope'd q and k, v and the attention output flash
    # attention saves; a Q/K norm's inputs q and k; and an output gate's
    # contiguous attention output and sigmoid(gate), which its product saves,
    # and the gated output o_proj's LoRA reads.
    width = 2 * q + 2 * kv
    if arch.qk_norm:
        width += q + kv
    if arch.attention_output_gate:
        width += 3 * q
    return width


def _block_attention_terms(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    tensor_parallel: int,
    document_len: int | None,
    context_parallel: int,
) -> tuple[int, int, float]:
    """Attention-side widths and bytes of one trainer block.

    :return: ``(qkv_dim, q_norm, scores)``: q/k/v elements per token on this
        rank (on a gated attention's backward, the tensors its forward saves),
        the fp32 Q copy a Q/K-norm backward holds, and the eager softmax
        tensors, CSA and DSA indexer terms and gathered context-parallel keys.
    """
    key_gather = context_parallel > 1 and arch.context_parallel_key_gather
    q_heads = tp_local(arch.n_heads, tensor_parallel)
    gated_backward = backward and arch.attention_output_gate and not key_gather
    if key_gather:
        qkv_dim = q_heads * arch.head_dim
    elif gated_backward:
        qkv_dim = _attention_saved_width(arch, tensor_parallel, context_parallel)
    else:
        qkv_dim = qkv_width_per_rank(arch, tensor_parallel, context_parallel)
    tokens = rows * seq_len
    q_norm = (
        tokens * q_heads * arch.head_dim * 4
        if backward and arch.qk_norm and not gated_backward
        else 0
    )
    # The softmax holds the masked logits, their sink-extended copy and the
    # probabilities; the backward holds the probabilities and two gradients.
    scores = (
        3 * rows * q_heads * seq_len * seq_len * act_bytes
        if arch.eager_attention
        else 0
    )
    document_len = document_len or seq_len * context_parallel
    scores += compressed_entry_bytes(
        arch, rows, seq_len, document_len, context_parallel
    )
    scores += pooled_index_bytes(arch, rows, seq_len, context_parallel)
    if key_gather:
        scores += context_parallel_key_bytes(
            arch,
            rows,
            seq_len * context_parallel,
            act_bytes,
            tensor_parallel,
            backward,
            context_parallel,
        )
    return qkv_dim, q_norm, scores


def _engine_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    routed_split: int,
) -> int:
    """Peak transients of one block in the engine's layer-at-a-time forward.

    :param routed_split: Ranks that share one sequence's routed tokens.
    """
    h = arch.hidden_size
    mlp, _ = _block_mlp_recompute(
        arch, rows, seq_len, act_bytes, backward, True, 1, routed_split
    )
    ple = arch.n_layers * arch.per_layer_input_dim * (2 if backward else 1)
    # Two of the four residual copies are the hyper-connection streams
    # entering and leaving the block; their pre-block mixer also
    # normalises the flattened streams in fp32.
    hidden = 2 * h + 2 * arch.residual_width
    if arch.hc_mult > 1:
        hidden += arch.residual_width * 4 / act_bytes
    tokens = rows * seq_len
    return int(tokens * (hidden + arch.peak_qkv_dim + mlp + ple) * act_bytes)


def _hc_mixer_bytes(
    arch: ModelArch, tokens: int, residual: int, act_bytes: float, backward: bool
) -> float:
    """Backward peak of one block's hyper-connection mixer.

    :param tokens: Tokens in the micro-batch.
    :param residual: Residual-stream elements per token the block holds.
    :return: Bytes; zero on the forward or without hyper-connections.
    """
    # A hyper-connection mixer saves only its bf16 streams and parameters; its
    # own backward rebuilds the fp32 flattened streams once the sub-layer's
    # other transients are freed.
    if not (backward and arch.hc_mult > 1):
        return 0
    return tokens * (residual * act_bytes + arch.residual_width * 4)


def _block_ffn_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool,
    tensor_parallel: int,
    routed_split: int,
    lora_target_scope: LoraTargetScope | None,
) -> float:
    """MLP, per-layer-input and routed-expert transients of one trainer block."""
    mlp, routed = _block_mlp_recompute(
        arch,
        rows,
        seq_len,
        act_bytes,
        backward,
        False,
        tensor_parallel,
        routed_split,
        lora_target_scope,
    )
    ple = arch.n_layers * arch.per_layer_input_dim * (2 if backward else 1)
    return rows * seq_len * (mlp + ple) * act_bytes + routed


def _sublayer_peak(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int,
    backward: bool,
    layer: BlockKind | None,
    residual: int,
    attention: float,
    ffn: float,
    mamba: float,
    norm_input: int,
    hc_mixer: float,
) -> int:
    """Peak of one trainer block whose transients peak in one sub-layer.

    :param layer: Sub-layer to size, or None for the widest.
    :param residual: Residual-stream elements per token the block holds.
    :param attention: Attention sub-layer bytes.
    :param ffn: FFN bytes beyond the residual copy.
    :param mamba: Recurrent-mixer bytes before norm inputs.
    :param norm_input: fp32 norm inputs one sub-layer saves.
    :param hc_mixer: Hyper-connection mixer backward bytes.
    """
    tokens = rows * seq_len
    # Both recurrent terms hold the layer's residual copy.
    by_layer = {
        "attention": attention,
        "ffn": tokens * residual * act_bytes + ffn,
        "mamba": mamba + norm_input,
    }
    if backward and not arch.block_exclusive_layers:
        # Recompute rebuilds the whole layer, so the mixer's saved
        # tensors stay live through the backward of the FFN beside it.
        # The recurrent kernel's own backward runs once the FFN's are freed.
        by_layer["attention"] += ffn
        mamba_saved = (
            tokens * residual * act_bytes
            + gated_delta_saved_bytes(arch, rows, seq_len, act_bytes, tensor_parallel)
            + norm_input
        )
        by_layer["mamba"] = max(by_layer["mamba"], mamba_saved + ffn)
    if layer is not None:
        return int(max(by_layer[layer], hc_mixer))
    return int(max(*by_layer.values(), hc_mixer))


def block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    backward: bool = False,
    engine_forward: bool = False,
    layer: BlockKind | None = None,
    unfused_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    routed_split: int = 1,
    document_len: int | None = None,
    context_parallel: int = 1,
    lora_target_scope: LoraTargetScope | None = None,
) -> int:
    """Peak transient activations of one transformer block.

    The forward peak holds four residual copies, qkv, and the MLP outputs.
    A checkpointed backward holds two residual copies at the full residual
    width (every hyper-connection stream), qkv, and the gated MLP: gate, up,
    the fused output, and one gradient. Given ``lora_target_scope``, a dense
    gated MLP holds its input, gate and up, then the fused output, or with
    adapters on the MLP the down projection's base and LoRA input gradients
    and their sum. A hyper-connection mixer's own
    backward holds the residual copies and an fp32 copy of the flattened
    streams, and the block peaks at the larger of that and the rest. A Q/K-norm model
    (``arch.qk_norm``) also holds one fp32 copy of Q. An attention with an
    output gate (``arch.attention_output_gate``) holds the tensors its forward
    saves as its qkv, and no fp32 Q copy. Flash-style backends
    tile the score matrix; eager attention (``arch.eager_attention``) holds
    three full ``(rows, heads, seq, seq)`` score tensors. A stock RMSNorm
    (``arch.fp32_norm_inputs``) keeps an fp32 copy of each norm input on the
    backward. DeepSeek-V4 CSA attention adds :func:`compressed_entry_bytes`
    over documents of ``document_len`` tokens, ``seq_len`` when None, and a
    GLM DSA layer :func:`pooled_index_bytes`.

    Under context parallel ``seq_len`` is one rank's shard of a sequence
    sharded over ``context_parallel`` ranks. Ulysses attention adds the
    key/value heads it replicates
    (:func:`~agilerl.arena.memory.formulas.qkv_width_per_rank`); key-gather
    attention keeps only queries local and adds
    :func:`context_parallel_key_bytes` over the whole sequence.

    In the trainer, routed experts run the grouped-GEMM path
    (:func:`~agilerl.arena.memory.formulas.routed_expert_bytes`), Mamba
    mixers their chunked scan (:func:`mamba_block_bytes`) and gated-delta-net
    mixers the fla chunk kernel (:func:`gated_delta_block_bytes`). When
    :attr:`ModelArch.sublayer_transients` holds, a block peaks in one
    attention, FFN or recurrent sub-layer: ``layer`` picks one, else the
    widest counts; otherwise a block holds all of them. On a stack whose
    layers carry an FFN beside the mixer, a checkpointed backward charges
    the mixer's saved tensors and that FFN together; a gated-delta-net
    layer peaks at the larger of that and its kernel's own backward
    (:func:`gated_delta_saved_bytes`).
    ``engine_forward=True`` sizes routed experts as the active FFN width
    (vLLM's in-kernel routing).
    """
    # One block's transient peaks at the widest block: checkpointed
    # recompute and the engine's layer-at-a-time forward run block by block.
    # The residual stays full width. Query/MLP/Mamba projections narrow on
    # the training mesh; the engine path sizes its own tensor parallel.
    if engine_forward:
        return _engine_block_bytes(
            arch, rows, seq_len, act_bytes, backward, routed_split
        )
    h = arch.hidden_size
    # The backward keeps the block's input and mid-block streams, which the
    # second hyper-connection mixer and the closing stream matmul both save.
    residual = 2 * arch.residual_width if backward else 4 * h
    qkv_dim, q_norm, scores = _block_attention_terms(
        arch,
        rows,
        seq_len,
        act_bytes,
        backward,
        tensor_parallel,
        document_len,
        context_parallel,
    )
    ffn = _block_ffn_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        backward,
        tensor_parallel,
        routed_split,
        lora_target_scope,
    )
    tokens = rows * seq_len
    attention = tokens * (residual + qkv_dim) * act_bytes + q_norm + scores
    mamba = _block_mixer_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        residual,
        backward,
        unfused_mamba_scan,
        tensor_parallel,
    )
    # fp32 RMSNorm inputs saved for backward, one per norm in the block.
    norm_input = tokens * h * 4 if backward and arch.fp32_norm_inputs else 0
    hc_mixer = _hc_mixer_bytes(arch, tokens, residual, act_bytes, backward)
    if arch.sublayer_transients:
        return _sublayer_peak(
            arch,
            rows,
            seq_len,
            act_bytes,
            tensor_parallel,
            backward,
            layer,
            residual,
            attention + norm_input,
            ffn + norm_input,
            mamba,
            norm_input,
            hc_mixer,
        )
    return int(
        max(attention + ffn + mamba + arch.norms_per_block * norm_input, hc_mixer)
    )


def _stored_ffn_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int,
    routed_split: int,
) -> int:
    """Hidden copies, MLP tensors and fp32 norm inputs one uncheckpointed block keeps."""
    tokens = rows * seq_len
    h = arch.hidden_size
    norm_inputs = 2 * tokens * h * 4 if arch.fp32_norm_inputs else 0
    mlp, routed = _block_mlp_recompute(
        arch, rows, seq_len, act_bytes, False, False, tensor_parallel, routed_split
    )
    # Per token in a dense block: the block's input and mid-block residual,
    # both norms' outputs, and the gated MLP's gate, up and fused output.
    width = 2 * arch.residual_width + 2 * h + mlp
    return int(tokens * width * act_bytes) + routed + norm_inputs


def stored_gated_delta_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int = 1,
    routed_split: int = 1,
) -> int:
    """Tensors one uncheckpointed gated-delta-net block keeps for its backward.

    The hidden copies, the MLP and the mixer's :func:`gated_delta_saved_bytes`;
    the fla kernel's backward transients come only inside the block's own
    backward. Uncheckpointed Qwen3.5-0.8B, 2B and 4B layers (LoRA r=1 on every
    linear, one bf16 row, A100-40, 16k-64k tokens) keep this within 0.03%.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param routed_split: Ranks that share one sequence's routed tokens.
    :return: Bytes.
    """
    saved = gated_delta_saved_bytes(arch, rows, seq_len, act_bytes, tensor_parallel)
    return (
        _stored_ffn_bytes(arch, rows, seq_len, act_bytes, tensor_parallel, routed_split)
        + saved
    )


def stored_attention_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    tensor_parallel: int = 1,
    routed_split: int = 1,
    context_parallel: int = 1,
) -> int:
    """Tensors one uncheckpointed attention block keeps for its backward.

    The hidden copies, the MLP and the attention's saved tensors; the
    attention backward's transients come only inside the block's own
    backward. Uncheckpointed Qwen3.5-0.8B, 2B and 4B full-attention layers
    (Liger, FA2, LoRA r=1 on every linear, one bf16 row, A100-40, 16k-64k
    tokens) keep 0.1-0.2% more than this.

    :param arch: Model architecture; attention projects q, k and v from hidden.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param routed_split: Ranks that share one sequence's routed tokens.
    :param context_parallel: Ranks the sequence is sharded over.
    :return: Bytes.
    """
    width = _attention_saved_width(arch, tensor_parallel, context_parallel)
    return _stored_ffn_bytes(
        arch, rows, seq_len, act_bytes, tensor_parallel, routed_split
    ) + int(rows * seq_len * width * act_bytes)


def mean_block_recompute_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    unfused_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    routed_split: int = 1,
    document_len: int | None = None,
    context_parallel: int = 1,
    layer: BlockKind | None = None,
) -> float:
    """Tensors one uncheckpointed block keeps for its backward, averaged over the stack's layers.

    Each uncheckpointed layer keeps its own mixer's tensors, so a stack whose
    every layer runs one mixer beside its FFN weights its attention and
    recurrent layers by their counts (Qwen3.5 LoRA on A100 80GB, GC off and
    every second or fourth block checkpointed, 16k-32k packed tokens), or
    with ``layer`` prices that mixer's layers alone. On a dense stack, a gated-delta-net layer other than KDA keeps
    :func:`stored_gated_delta_block_bytes`, and an attention layer that
    projects q, k and v from hidden :func:`stored_attention_block_bytes`.
    Other layers, and every layer of a MoE stack, keep
    :data:`~agilerl.arena.memory.formulas.STORED_BLOCK_SHARE` of their
    :func:`block_recompute_bytes`; other stacks keep that share of the
    widest block.

    :param arch: Model architecture.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param unfused_mamba_scan: Mamba layers run HF's unfused path.
    :param tensor_parallel: Degree the decoder layers are sharded by.
    :param routed_split: Ranks that share one sequence's routed tokens.
    :param document_len: Longest document in a sequence, or None for ``seq_len``.
    :param context_parallel: Ranks the sequence is sharded over.
    :param layer: Mixer whose layers to price, or None for the stack's mean.
    :return: Mean bytes of one block.
    """

    def peak(layer: BlockKind | None) -> float:
        return STORED_BLOCK_SHARE * block_recompute_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            backward=True,
            layer=layer,
            unfused_mamba_scan=unfused_mamba_scan,
            tensor_parallel=tensor_parallel,
            routed_split=routed_split,
            document_len=document_len,
            context_parallel=context_parallel,
        )

    def attention() -> float:
        if arch.low_rank_attention or arch.is_moe:
            return peak("attention")
        return stored_attention_block_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            tensor_parallel,
            routed_split,
            context_parallel,
        )

    def mixer() -> float:
        if arch.mamba_gated_delta and not (arch.mamba_kda or arch.is_moe):
            return stored_gated_delta_block_bytes(
                arch, rows, seq_len, act_bytes, tensor_parallel, routed_split
            )
        return peak("mamba")

    if not arch.sublayer_transients or arch.block_exclusive_layers:
        return peak(None)
    if layer == "attention":
        return attention()
    if layer == "mamba":
        return mixer()
    return (
        arch.attention_layers * attention() + arch.n_mamba_layers * mixer()
    ) / arch.n_layers


def prices_stored_tensors(arch: ModelArch) -> bool:
    """Whether :func:`stored_block_bytes` prices this model's uncheckpointed blocks.

    :param arch: Model architecture.
    :return: True for a MoE stack whose every layer holds a mixer and the
        experts, on the self-routing expert LoRA path.
    """
    return (
        arch.is_moe
        and arch.routed_expert_recompute
        and not arch.block_exclusive_layers
        and not arch.mamba_gated_delta
    )


def stored_block_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    expert_lora: bool,
    mamba_lora: bool,
    expert_parallel: int = 1,
    routed_split: int = 1,
) -> int:
    """Tensors one uncheckpointed block keeps for its backward, averaged over layer kinds.

    Per token: each RMSNorm's input (fp32 under the stock module) and the
    normed rows the experts read. An attention layer adds the normed rows its
    LoRA reads, q, k, v and the attention output. A Mamba-2 layer adds the
    fused kernel's ``in_proj`` and scan outputs, and with LoRA on ``in_proj``
    and ``out_proj`` their inputs. The router keeps an fp32 score and an
    int64 index per routed slot. Expert LoRA recomputes the experts in
    backward and keeps another score and index per slot; frozen experts keep
    gate, up and the expert output. A shared expert keeps gate and up, plus
    the activated gate under the stock MLP. Expert parallel keeps one
    ``expert_width`` row per routed slot for the combine's router-weight
    product, and with expert LoRA four: the received rows the recompute
    reads, the combine rows, and two more the measured peaks show.

    :param arch: Model architecture; :func:`prices_stored_tensors` holds.
    :param rows: Sequences in the micro-batch.
    :param seq_len: Tokens per sequence.
    :param act_bytes: Bytes per activation element.
    :param expert_lora: Packed expert matrices carry LoRA.
    :param mamba_lora: Mamba ``in_proj`` and ``out_proj`` carry LoRA.
    :param expert_parallel: Expert-parallel degree.
    :param routed_split: Ranks that share one sequence's routed tokens
        (TP ranks under expert parallel).
    :return: Mean saved bytes of one block.
    """
    h = arch.hidden_size
    tokens = rows * seq_len
    norm_bytes = DTYPE_BYTES["fp32"] if arch.fp32_norm_inputs else act_bytes
    per_token = arch.norms_per_block * h * norm_bytes + h * act_bytes
    slots = arch.n_experts_per_tok or 1
    slot_bytes = DTYPE_BYTES["fp32"] + 8
    inter = arch.expert_intermediate_size or arch.intermediate_size
    if expert_lora:
        per_token += 2 * slots * slot_bytes
    else:
        per_token += slots * (slot_bytes + (2 * inter + arch.expert_width) * act_bytes)
    if expert_parallel > 1:
        # granite-3.1-3b-a800m GRPO LoRA, ep=2, GC off, 2x A100 80GB: peaks
        # grow 3.70 GiB per 1k packed tokens at 8k-16k, 3.9 rows per slot
        # above the same model without expert parallel.
        ep_rows = 4 if expert_lora else 1
        per_token += ep_rows * slots * arch.expert_width * act_bytes / routed_split
    shared_inter = arch.shared_expert_intermediate_size or inter
    shared_matrices = 3 if arch.stock_gated_mlp else 2
    per_token += arch.n_shared_experts * shared_matrices * shared_inter * act_bytes
    q_dim = arch.n_heads * arch.head_dim
    attention = (h + arch.peak_qkv_dim + q_dim) * act_bytes
    if arch.qk_norm:
        attention += q_dim * DTYPE_BYTES["fp32"]
    mamba = mamba_saved_width(arch, mamba_lora) * act_bytes
    mixers = arch.attention_layers * attention + arch.n_mamba_layers * mamba
    return int(tokens * (per_token + mixers / arch.n_layers))


def kept_layer_bytes(
    arch: ModelArch,
    settings: TrainingSettings,
    act_bytes: float,
    seq_len: int,
    live_rows: int,
    unfused_mamba_scan: bool,
    tensor_parallel: int,
    routed_split: int,
    casts_apply: bool,
) -> list[float]:
    """Bytes each layer keeps from its forward for its backward, in layer order.

    A checkpointed layer keeps its input. Any other layer keeps its forward's
    saved tensors: :func:`stored_block_bytes` where :func:`prices_stored_tensors`
    holds, else :func:`mean_block_recompute_bytes` of its own mixer when
    :attr:`ModelArch.layer_mixers` lists it, or of the whole stack.

    :param casts_apply: PEFT casts wrapped inputs to wider adapters.
    """
    rows = settings.grad_forward_rows
    mixers = arch.layer_mixers or (None,) * arch.n_layers
    if prices_stored_tensors(arch):
        saved = stored_block_bytes(
            arch,
            rows,
            seq_len,
            act_bytes,
            expert_lora=settings.lora_packed_target_matrices > 0,
            mamba_lora=settings.lora_target_scope == "all-linear",
            expert_parallel=1 if settings.fsdp is None else settings.fsdp.ep,
            routed_split=routed_split,
        )
        prices = dict.fromkeys(mixers, float(saved))
    else:
        dropped = (
            lora_only_stored_input_bytes(
                arch, rows, seq_len, act_bytes, tensor_parallel
            )
            if casts_apply or settings.lora_target_scope == "attention-only"
            else 0
        )
        prices = {
            kind: mean_block_recompute_bytes(
                arch,
                rows,
                seq_len,
                act_bytes,
                unfused_mamba_scan,
                tensor_parallel,
                routed_split,
                settings.max_model_len,
                1 if settings.fsdp is None else settings.fsdp.cp,
                kind,
            )
            - dropped
            for kind in set(mixers)
        }
    checkpointed = training_checkpointed_layers(arch, settings)
    boundary = live_rows * seq_len * arch.residual_width * act_bytes
    return [
        boundary if index in checkpointed else prices[kind] * settings.grad_graph_rows
        for index, kind in enumerate(mixers)
    ]


def backward_freed_bytes(
    arch: ModelArch, settings: TrainingSettings, kept: Sequence[float]
) -> list[float]:
    """Bytes each layer's backward takes off the device's saved set, in layer order.

    Backward runs the last layer first and frees what a layer kept once its
    backward ends. A layer outside :func:`training_checkpointed_layers` holds
    the tensors it kept inside its backward transients. Each entry is one
    graph's share: ``kept`` spans every graph of
    :attr:`TrainingSettings.grad_graph_rows`, and the layer in backward
    belongs to one. Activation offload keeps the saved set on the host.

    :param kept: :func:`kept_layer_bytes`.
    """
    if settings.activation_offload:
        return [0.0] * len(kept)
    checkpointed = training_checkpointed_layers(arch, settings)
    freed: list[float] = []
    after = 0.0
    for index in reversed(range(len(kept))):
        freed.append(after + (0.0 if index in checkpointed else kept[index]))
        after += kept[index]
    return [layer_freed / settings.grad_graph_rows for layer_freed in freed[::-1]]


def mamba_saved_width(arch: ModelArch, lora: bool) -> int:
    """Elements per token one Mamba-2 layer on the fused split-scan kernel saves for backward.

    The kernel keeps its ``in_proj`` output and its scan output. LoRA on
    ``in_proj`` and ``out_proj`` adds their inputs: the normed hidden rows and
    the gated-norm output.

    :param arch: Model architecture.
    :param lora: ``in_proj`` and ``out_proj`` carry LoRA.
    :return: Saved elements per token; zero without Mamba layers.
    """
    if not arch.n_mamba_layers:
        return 0
    d_inner = arch.mamba_n_heads * arch.mamba_d_head
    width = 2 * d_inner + arch.mamba_conv_dim + arch.mamba_n_heads
    if lora:
        width += arch.hidden_size + d_inner
    return width


def nograd_forward_bytes(
    arch: ModelArch,
    rows: int,
    seq_len: int,
    act_bytes: float,
    unfused_mamba_scan: bool = False,
    tensor_parallel: int = 1,
    routed_split: int = 1,
    document_len: int | None = None,
    context_parallel: int = 1,
) -> int:
    """Peak transient memory of the fused no-grad logprob forward.

    Nothing is saved for backward: one block's transients plus the
    full-sequence hidden state handed to the fused logprob kernel.
    """
    block = block_recompute_bytes(
        arch,
        rows,
        seq_len,
        act_bytes,
        unfused_mamba_scan=unfused_mamba_scan,
        tensor_parallel=tensor_parallel,
        routed_split=routed_split,
        document_len=document_len,
        context_parallel=context_parallel,
    )
    hidden_out = rows * seq_len * arch.hidden_size * act_bytes
    return int(block + hidden_out)
