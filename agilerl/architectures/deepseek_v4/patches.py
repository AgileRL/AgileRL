# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 patches: flex attention over the compressed KV, packed documents, and a cheaper indexer.

Flex attention: each CSA / HCA layer appends compressor entries to the
sliding keys inside the attention block, so the model-level ``BlockMask``
does not cover them. The patched attention pads the sliding keys to whole
mask blocks, appends the entries, and joins the model's block lists with the
entries' own. The entries' mask builds once per forward and compress rate.
The per-head sink joins the softmax by rescaling the flex output with
``sigmoid(lse - sink)``.

Packed documents: the compressor pools fixed windows of each document's tokens
and places entries at in-document positions. The patched attention projects
the whole row once and gathers each document's windows by index; a query sees
only its own document's entries. The model forward derives the document
lengths once and passes them to every attention layer.

Indexer: it returns integer top-k indices that ``DeepseekV4CSACompressor``
scatters into a constant block bias, so no gradient reaches its parameters.
Without a cache it compresses ``seq_len // compress_rate`` entries; when that
is at most ``index_topk``, top-k keeps all of them and only the causal
threshold decides which are valid. Without grad it scores queries in chunks of
at most :data:`INDEXER_SCORE_ELEMENTS` fp32 scores.

Context parallel: queries stay on the rank's sequence shard. Each rank pools
the windows that end in its shard, taking the tokens before the shard from the
earlier shards' tails. One all-gather then carries every shard's sliding keys,
entries, and indexer keys, so the indexer scores each local query against
every compressed entry.
"""

from __future__ import annotations

import functools
import itertools
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
)

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.flex_attention import (
    flex_decode_kernel_options,
    large_head_dim_kernel_options,
)
from agilerl.architectures.hyper_connections import compile_hyper_connection_mix
from agilerl.architectures.runtime import PatchRuntimeConfig
from agilerl.utils.llm_packing import FLEX_BLOCK_SIZE, packed_layout
from agilerl.utils.patching import class_is_patched, library_shim

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers.integrations.flex_attention import (
        compile_friendly_flex_attention,
    )
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4Attention,
        DeepseekV4CSACompressor,
        DeepseekV4HCACompressor,
        DeepseekV4HyperConnection,
        DeepseekV4Indexer,
        DeepseekV4IndexerScorer,
        DeepseekV4Model,
        DeepseekV4PreTrainedModel,
        apply_rotary_pos_emb,
    )
else:
    compile_friendly_flex_attention: Any = None
    DeepseekV4Attention: Any = None
    DeepseekV4CSACompressor: Any = None
    DeepseekV4HCACompressor: Any = None
    DeepseekV4HyperConnection: Any = None
    DeepseekV4Indexer: Any = None
    DeepseekV4IndexerScorer: Any = None
    DeepseekV4Model: Any = None
    DeepseekV4PreTrainedModel: Any = None
    apply_rotary_pos_emb: Any = None

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel
    from transformers.cache_utils import Cache

__all__ = [
    "CHUNKS_INDEXER_SCORES",
    "COMPILED_CREATE_BLOCK_MASK",
    "CONTEXT_PARALLEL_GROUP",
    "ENTRY_MEMO",
    "FLEX_COMPRESSED_KV",
    "INDEXER_SCORE_ELEMENTS",
    "PACKED_DOCUMENTS",
    "PACKED_DOCUMENT_LENGTHS",
    "SELECTS_ALL_WHEN_TOPK_COVERS",
    "EntryLayout",
    "EntryMemo",
    "EntryWindows",
    "append_entries",
    "append_kv_blocks",
    "build_block_mask",
    "causal_entry_blocks",
    "causal_entry_indices",
    "chunked_scorer_forward",
    "compress_documents",
    "document_entries",
    "document_keys",
    "flex_attention_with_sinks",
    "flex_compressed_attention",
    "full_kv_blocks",
    "full_q_blocks",
    "gather_shards",
    "indexer_scores",
    "install_deepseek_v4_patches",
    "packed_document_lengths",
    "pass_packed_document_lengths",
    "pool_entries",
    "previous_tokens",
    "select_all_when_topk_covers",
    "select_entries",
    "sequence_shard_keys",
    "shard_entries",
    "shard_sliding_mask",
    "shard_windows",
    "visible_mod",
]

SELECTS_ALL_WHEN_TOPK_COVERS = "agilerl_selects_all_when_topk_covers"

# Marks ``DeepseekV4IndexerScorer`` whose no-grad forward scores queries in chunks.
CHUNKS_INDEXER_SCORES = "agilerl_chunks_indexer_scores"

# Marks ``DeepseekV4Attention`` whose forward runs flex over the compressed KV.
FLEX_COMPRESSED_KV = "agilerl_flex_compressed_kv"

# Marks ``DeepseekV4Model`` whose forward passes packed document lengths.
PACKED_DOCUMENTS = "agilerl_packed_documents"

# Decoder-layer kwarg carrying the host tuple of packed document lengths.
PACKED_DOCUMENT_LENGTHS = "agilerl_packed_document_lengths"

# Attention kwarg carrying the context-parallel group while a sequence shard trains.
CONTEXT_PARALLEL_GROUP = "agilerl_context_parallel_group"

# fp32 elements in one chunk of indexer scores, [batch, queries, heads, entries]: 1 GiB.
INDEXER_SCORE_ELEMENTS = 1 << 28

# Evaluates a mask block by block, never as a dense [batch, queries, keys] tensor.
COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)

MaskMod = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
Built = TypeVar("Built")


class EntryWindows(NamedTuple):
    """Token windows that pool into compressed entries.

    :param starts: ``[entries]`` index of each window's first token in the pooled tokens.
    :param positions: ``[entries]`` RoPE position of each entry, its window's start
        within the document; 0 for a document's first window, which has none before it.
    """

    starts: torch.Tensor
    positions: torch.Tensor


class EntryLayout(NamedTuple):
    """Documents of the queries and of the compressed entries they may read.

    :param query_document: ``[queries]`` document of each query.
    :param entry_document: ``[entries]`` document of each entry, ``-1`` for padding.
    :param entry_window: ``[entries]`` window index of each entry within its document.
    :param spans: Host ``(query_start, query_end, entry_start, entry_end)`` of each document.
    """

    query_document: torch.Tensor
    entry_document: torch.Tensor
    entry_window: torch.Tensor
    spans: tuple[tuple[int, int, int, int], ...]


class EntryMemo:
    """Entry layouts and masks of the most recent ``position_ids`` tensor.

    Every attention layer of one forward, and its checkpoint recompute, receives
    the same ``position_ids`` tensor, so each value builds once per forward.
    Callers must not modify ``position_ids`` in place after the first lookup.
    """

    def __init__(self) -> None:
        self.key: weakref.ref[torch.Tensor] | None = None
        self.values: dict[tuple[object, ...], object] = {}

    def get(
        self,
        position_ids: torch.Tensor,
        key: tuple[object, ...],
        build: Callable[[], Built],
    ) -> Built:
        """Value under ``key`` for ``position_ids``, built on the first lookup.

        :param position_ids: Positions of the forward the value belongs to.
        :param key: Everything besides ``position_ids`` the value depends on.
        :param build: Builds the value.
        :return: The memoized value.
        """
        if self.key is None or self.key() is not position_ids:
            self.key = weakref.ref(position_ids)
            self.values = {}
        if key not in self.values:
            self.values[key] = build()
        return cast("Built", self.values[key])


ENTRY_MEMO = EntryMemo()


def causal_entry_indices(
    position_ids: torch.Tensor, batch: int, compressed_len: int, compress_rate: int
) -> torch.Tensor:
    """Every compressed entry per query, with entries past the query's causal threshold set to ``-1``.

    :param position_ids: Query positions, ``[batch or 1, seq_len]``.
    :param batch: Batch size.
    :param compressed_len: Number of compressed entries.
    :param compress_rate: Source tokens per compressed entry.
    :return: ``int64`` indices ``[batch, seq_len, compressed_len]``, ascending per query.
    """
    entry_indices = torch.arange(compressed_len, device=position_ids.device)
    entry_indices = entry_indices.expand(batch, position_ids.shape[1], compressed_len)
    causal_threshold = (position_ids + 1) // compress_rate
    return entry_indices.masked_fill(
        entry_indices >= causal_threshold.unsqueeze(-1), -1
    )


def select_all_when_topk_covers(
    forward: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Wrap ``DeepseekV4Indexer.forward`` to skip scoring when top-k keeps every entry, else score without grad.

    :param forward: Stock indexer forward.
    :return: Indexer forward returning the same selected entries per query.
    """

    @functools.wraps(forward)
    def indexer_forward(
        self: DeepseekV4Indexer,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        compressed_len = seq_len // self.compress_rate
        # The compressor reads these indices as a set, so ascending order matches score order.
        if past_key_values is None and compressed_len <= self.index_topk:
            return causal_entry_indices(
                position_ids, batch, compressed_len, self.compress_rate
            )
        with torch.no_grad():
            return forward(
                self,
                hidden_states,
                q_residual,
                position_ids,
                past_key_values,
                layer_idx,
            )

    return indexer_forward


def indexer_scores(
    scorer: DeepseekV4IndexerScorer,
    query: torch.Tensor,
    keys: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """``DeepseekV4IndexerScorer`` scores without grad, with relu, scale, and head weights applied in place.

    :param scorer: The indexer's scorer.
    :param query: ``[batch, queries, heads, index_head_dim]`` rotated indexer queries.
    :param keys: ``[batch, entries, index_head_dim]`` indexer keys.
    :param hidden_states: ``[batch, queries, hidden]`` attention input of the queries.
    :return: ``[batch, queries, entries]`` fp32 scores.
    """
    scores = torch.matmul(query.float(), keys.transpose(-1, -2).float().unsqueeze(1))
    scores.relu_().mul_(scorer.softmax_scale)
    weights = scorer.weights_proj(hidden_states).float() * scorer.weights_scaling
    return scores.mul_(weights.unsqueeze(-1)).sum(dim=2)


def chunked_scorer_forward(
    self: DeepseekV4IndexerScorer,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """``DeepseekV4IndexerScorer.forward`` scoring chunks of queries; the patched indexer runs it without grad.

    :param q: ``[batch, queries, heads, index_head_dim]`` rotated indexer queries.
    :param compressed_kv: ``[batch, entries, index_head_dim]`` indexer keys.
    :param hidden_states: ``[batch, queries, hidden]`` attention input of the queries.
    :return: ``[batch, queries, entries]`` fp32 scores.
    """
    chunk = max(
        1,
        INDEXER_SCORE_ELEMENTS
        // max(1, q.shape[0] * q.shape[2] * compressed_kv.shape[1]),
    )
    return torch.cat(
        [
            indexer_scores(self, query, compressed_kv, hidden)
            for query, hidden in zip(
                q.split(chunk, dim=1), hidden_states.split(chunk, dim=1), strict=True
            )
        ],
        dim=1,
    )


def packed_document_lengths(position_ids: torch.Tensor) -> tuple[int, ...] | None:
    """Document lengths shared by every packed row, or None when each row holds one document.

    :param position_ids: ``[batch, seq_len]`` positions; each packed document restarts at 0.
    :return: Lengths of the documents in each row, in order.
    """
    layout = packed_layout(position_ids)
    if layout is None:
        return None
    row_len = position_ids.shape[1]
    rows: list[list[int]] = [[] for _ in range(position_ids.shape[0])]
    for start, end in itertools.pairwise(layout.cu_seqlens.tolist()):
        rows[start // row_len].append(end - start)
    if any(row != rows[0] for row in rows):
        msg = "DeepSeek-V4 packed rows must share one document layout"
        raise ValueError(msg)
    return tuple(rows[0])


def document_entries(
    document_lengths: tuple[int, ...], rate: int, device: torch.device
) -> tuple[EntryWindows, EntryLayout]:
    """Windows and layout of every document's compressed entries, as the stock compressor makes them per document.

    The layout ends with padding entries in document ``-1``, which no query sees,
    so a row of ``seq_len`` tokens has ``seq_len // rate`` entries for any packing
    and compiled flex attention sees one key length.

    :param document_lengths: Lengths of the documents in each row.
    :param rate: Tokens per compressed entry.
    :param device: Device of the index tensors.
    :return: The windows over the row's tokens and the entries' layout, in document order.
    """
    lengths = torch.tensor(document_lengths)
    windows = lengths // rate
    documents = torch.arange(len(document_lengths))
    entry_document = documents.repeat_interleave(windows)
    entry_window = (
        torch.arange(int(windows.sum())) - (windows.cumsum(0) - windows)[entry_document]
    )
    starts = (lengths.cumsum(0) - lengths)[entry_document] + entry_window * rate
    query_document = documents.repeat_interleave(lengths)
    entries = entry_document.numel()
    padding = torch.full((sum(document_lengths) // rate - entries,), -1)
    padded = entries + padding.numel()
    # One host-to-device copy for every index tensor.
    starts, entry_window, entry_document, query_document = (
        torch.cat(
            [starts, entry_window, padding, entry_document, padding, query_document]
        )
        .to(device, non_blocking=True)
        .split([entries, padded, padded, query_document.numel()])
    )
    token_bounds = list(itertools.accumulate(document_lengths, initial=0))
    entry_bounds = list(
        itertools.accumulate((length // rate for length in document_lengths), initial=0)
    )
    spans = tuple(
        zip(
            token_bounds[:-1],
            token_bounds[1:],
            entry_bounds[:-1],
            entry_bounds[1:],
            strict=True,
        )
    )
    return (
        EntryWindows(starts, entry_window[:entries] * rate),
        EntryLayout(query_document, entry_document, entry_window, spans),
    )


def shard_windows(
    shard_len: int, rank: int, rate: int, overlap: bool
) -> tuple[int, int, int]:
    """First window ending in a sequence shard, how many end in it, and how many tokens before the shard they pool.

    :param shard_len: Tokens per shard.
    :param rank: Shard index.
    :param rate: Tokens per window.
    :param overlap: Each entry also pools the window before its own, as CSA does.
    :return: ``(first_window, windows, halo)``.
    """
    first = rank * shard_len // rate
    count = (rank + 1) * shard_len // rate - first
    halo = rank * shard_len - max(first - int(overlap), 0) * rate
    return first, count, halo


def shard_entries(
    shard_len: int,
    rank: int,
    world: int,
    rate: int,
    overlap: bool,
    device: torch.device,
) -> tuple[EntryWindows, EntryLayout, tuple[int, ...], int, int]:
    """Windows this shard pools, the layout of the whole sequence's entries for its queries, and the halo it needs.

    :param shard_len: Tokens per shard.
    :param rank: This shard's index.
    :param world: Number of shards.
    :param rate: Tokens per window.
    :param overlap: Each entry also pools the window before its own, as CSA does.
    :param device: Device of the index tensors.
    :return: Windows over ``halo`` earlier tokens then this shard's, the entries' layout,
        every shard's entry count, ``halo``, and the tail length each shard sends.
    """
    first, count, halo = shard_windows(shard_len, rank, rate, overlap)
    shards = [shard_windows(shard_len, other, rate, overlap) for other in range(world)]
    entries = world * shard_len // rate
    tail = 0
    if any(other_halo > 0 for _, _, other_halo in shards):
        tail = min((1 + int(overlap)) * rate - 1, shard_len)
    window = torch.arange(first, first + count, device=device)
    return (
        EntryWindows(window * rate - (rank * shard_len - halo), window * rate),
        EntryLayout(
            torch.zeros(shard_len, dtype=torch.long, device=device),
            torch.zeros(entries, dtype=torch.long, device=device),
            torch.arange(entries, device=device),
            ((0, shard_len, 0, entries),),
        ),
        tuple(other_count for _, other_count, _ in shards),
        halo,
        tail,
    )


def pool_entries(
    compressor: DeepseekV4CSACompressor | DeepseekV4HCACompressor | DeepseekV4Indexer,
    kv: torch.Tensor,
    gate: torch.Tensor,
    windows: EntryWindows,
    overlap: bool,
) -> torch.Tensor:
    """Pool each window of tokens into one entry, as the cacheless stock compressor and indexer do.

    :param compressor: CSA / HCA compressor or indexer that owns the pooling weights.
    :param kv: ``[batch, tokens, width]`` per-token key/value projections.
    :param gate: ``[batch, tokens, width]`` per-token gate projections.
    :param windows: Windows to pool.
    :param overlap: Pool each window with the previous window's first-half series, as CSA does.
    :return: ``[batch, entries, head_dim]`` entries, RoPE at each entry's position.
    """
    rate = int(compressor.compress_rate)
    head_dim = int(compressor.head_dim)
    position_bias = compressor.position_bias
    tokens = windows.starts.unsqueeze(-1) + torch.arange(rate, device=kv.device)
    if overlap:
        previous = (tokens - rate).clamp(min=0)
        missing = (windows.positions == 0).view(-1, 1, 1)
        pooled_kv = torch.cat(
            [
                kv[..., :head_dim][:, previous].masked_fill(missing, 0.0),
                kv[..., head_dim:][:, tokens],
            ],
            dim=2,
        )
        pooled_gate = torch.cat(
            [
                (
                    gate[..., :head_dim][:, previous] + position_bias[:, :head_dim]
                ).masked_fill(missing, float("-inf")),
                gate[..., head_dim:][:, tokens] + position_bias[:, head_dim:],
            ],
            dim=2,
        )
    else:
        pooled_kv = kv[:, tokens]
        pooled_gate = gate[:, tokens] + position_bias
    weights = pooled_gate.softmax(dim=2, dtype=torch.float32).to(pooled_kv.dtype)
    compressed = compressor.kv_norm((pooled_kv * weights).sum(dim=2))
    cos, sin = compressor.rotary_emb(
        compressed,
        position_ids=windows.positions.expand(kv.shape[0], -1),
        layer_type=compressor.rope_layer_type,
    )
    return apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)


def select_entries(
    indexer: DeepseekV4Indexer,
    keys: torch.Tensor,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    layout: EntryLayout,
) -> torch.Tensor:
    """Entries each query sees: the indexer's top-k of its document's causal entries, or all of them where top-k keeps every one.

    :param indexer: The CSA compressor's indexer.
    :param keys: ``[batch, entries, index_head_dim]`` indexer keys.
    :param hidden_states: ``[batch, queries, hidden]`` attention input.
    :param q_residual: ``[batch, queries, q_lora_rank]`` normed query latent.
    :param position_ids: ``[batch or 1, queries]`` positions.
    :param layout: Documents of the queries and entries.
    :return: ``[batch, queries, entries]`` bool.
    """
    batch = hidden_states.shape[0]
    rate = int(indexer.compress_rate)
    topk = int(indexer.index_topk)
    threshold = ((position_ids + 1) // rate).expand(batch, -1).unsqueeze(-1)
    visible = layout.entry_window < threshold
    visible &= layout.query_document.unsqueeze(-1) == layout.entry_document
    for query_start, query_end, entry_start, entry_end in layout.spans:
        if entry_end - entry_start <= topk:
            continue
        span_hidden = hidden_states[:, query_start:query_end]
        span_positions = position_ids[:, query_start:query_end]
        cos, sin = indexer.rotary_emb(
            span_hidden, position_ids=span_positions, layer_type=indexer.rope_layer_type
        )
        query = indexer.q_b_proj(q_residual[:, query_start:query_end])
        query = query.view(batch, query_end - query_start, -1, indexer.head_dim)
        query = apply_rotary_pos_emb(query.transpose(1, 2), cos, sin).transpose(1, 2)
        span_keys = keys[:, entry_start:entry_end]
        span_len = query_end - query_start
        chunk = max(
            1,
            INDEXER_SCORE_ELEMENTS
            // (batch * int(indexer.num_heads) * (entry_end - entry_start)),
        )
        for start in range(0, span_len, chunk):
            end = min(start + chunk, span_len)
            causal = visible[
                :, query_start + start : query_start + end, entry_start:entry_end
            ]
            scores = indexer_scores(
                indexer.scorer,
                query[:, start:end],
                span_keys,
                span_hidden[:, start:end],
            )
            top = scores.masked_fill_(~causal, float("-inf")).topk(topk, dim=-1).indices
            causal &= torch.zeros_like(causal).scatter_(-1, top, True)
    return visible


def compress_documents(
    compressor: DeepseekV4CSACompressor | DeepseekV4HCACompressor,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    document_lengths: tuple[int, ...],
) -> tuple[torch.Tensor, EntryLayout, torch.Tensor | None]:
    """Compressed entries of every document of the rows, as the stock compressor makes them per document.

    :param compressor: The attention layer's compressor.
    :param hidden_states: ``[batch, seq_len, hidden]`` attention input.
    :param q_residual: ``[batch, seq_len, q_lora_rank]`` normed query latent.
    :param position_ids: ``[batch or 1, seq_len]`` positions.
    :param document_lengths: Lengths of the documents in each row, summing to ``seq_len``.
    :return: ``[batch, 1, entries, head_dim]`` entries in document order then zero
        padding, their layout, and the ``[batch, seq_len, entries]`` entries each query
        sees where the indexer selects, else None: each query then sees its document's
        causal entries.
    """
    rate = int(compressor.compress_rate)
    windows, layout = ENTRY_MEMO.get(
        position_ids,
        ("documents", rate, document_lengths),
        lambda: document_entries(document_lengths, rate, hidden_states.device),
    )
    overlap = isinstance(compressor, DeepseekV4CSACompressor)
    entries = pool_entries(
        compressor,
        compressor.kv_proj(hidden_states),
        compressor.gate_proj(hidden_states),
        windows,
        overlap,
    )
    visible = None
    if overlap and any(
        entry_end - entry_start > compressor.indexer.index_topk
        for _, _, entry_start, entry_end in layout.spans
    ):
        indexer = compressor.indexer
        with torch.no_grad():
            keys = pool_entries(
                indexer,
                indexer.kv_proj(hidden_states),
                indexer.gate_proj(hidden_states),
                windows,
                True,
            )
            visible = select_entries(
                indexer, keys, hidden_states, q_residual, position_ids, layout
            )
    padding = layout.entry_document.shape[0] - entries.shape[1]
    return F.pad(entries, (0, 0, 0, padding)).unsqueeze(1), layout, visible


def build_block_mask(
    mask_mod: MaskMod,
    batch: int | None,
    queries: int,
    keys: int,
    block_size: tuple[int, int],
    device: torch.device,
    separate_full_blocks: bool,
) -> BlockMask:
    """``create_block_mask`` over ``queries`` x ``keys``, compiled on CUDA.

    :param mask_mod: Mask to evaluate.
    :param batch: Batch size, or None for a mask shared by every row.
    :param queries: Query length.
    :param keys: Key length.
    :param block_size: Query and key block sizes.
    :param device: Device the mask lives on.
    :param separate_full_blocks: List fully visible blocks apart from partial ones.
    :return: The block mask, shared by every head.
    """
    # Uncompiled masks materialise queries x keys on the device.
    build = COMPILED_CREATE_BLOCK_MASK if device.type == "cuda" else create_block_mask
    return build(
        mask_mod,
        B=batch,
        H=None,
        Q_LEN=queries,
        KV_LEN=keys,
        device=device,
        BLOCK_SIZE=block_size,
        separate_full_blocks=separate_full_blocks,
    )


def causal_entry_blocks(
    layout: EntryLayout,
    position_ids: torch.Tensor,
    rate: int,
    batch: int,
    block_size: tuple[int, int],
    separate_full_blocks: bool,
) -> BlockMask:
    """Mask of each query over its own document's entries below its causal threshold.

    :param layout: Documents of the queries and entries.
    :param position_ids: ``[batch or 1, queries]`` positions.
    :param rate: Tokens per compressed entry.
    :param batch: Batch size.
    :param block_size: Query and key block sizes.
    :param separate_full_blocks: List fully visible blocks apart; else every
        non-empty block is partial, a superset any narrower ``mask_mod`` can share.
    :return: Mask over ``[batch or 1, queries, entries]``.
    """
    threshold = ((position_ids + 1) // rate).expand(batch, -1)
    query_document = layout.query_document
    entry_document = layout.entry_document
    entry_window = layout.entry_window

    def causal(
        batch_idx: torch.Tensor,
        _head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        entry: torch.Tensor,
    ) -> torch.Tensor:
        same_document = query_document[q_idx] == entry_document[entry]
        return same_document & (entry_window[entry] < threshold[batch_idx, q_idx])

    return build_block_mask(
        causal,
        position_ids.shape[0],
        position_ids.shape[1],
        entry_window.shape[0],
        block_size,
        position_ids.device,
        separate_full_blocks,
    )


def visible_mod(visible: torch.Tensor) -> MaskMod:
    """Mask reading ``[batch, queries, entries]`` visibility."""

    def mask_mod(
        batch_idx: torch.Tensor,
        _head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        entry: torch.Tensor,
    ) -> torch.Tensor:
        return visible[batch_idx, q_idx, entry]

    return mask_mod


def append_kv_blocks(
    first_num: torch.Tensor,
    first_indices: torch.Tensor,
    second_num: torch.Tensor,
    second_indices: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each query block's key blocks from ``first``, then those of ``second`` shifted by ``offset``.

    :param first_num: ``[batch, heads, query_blocks]`` key blocks of each query block.
    :param first_indices: ``[batch, heads, query_blocks, width]`` key blocks, valid prefix first.
    :param second_num: Same as ``first_num`` for the second list.
    :param second_indices: Same as ``first_indices`` for the second list.
    :param offset: Key blocks ahead of the second list.
    :return: The joined counts and indices, valid prefix first.
    """
    width = first_indices.shape[-1]
    joined = torch.cat([first_indices, second_indices + offset], dim=-1)
    slots = torch.arange(joined.shape[-1], device=joined.device)
    count = first_num.unsqueeze(-1)
    source = torch.where(slots < count, slots, slots - count + width)
    source = source.clamp(max=joined.shape[-1] - 1)
    return first_num + second_num, joined.gather(-1, source)


def full_kv_blocks(mask: BlockMask) -> tuple[torch.Tensor, torch.Tensor]:
    """Full key blocks of ``mask``, empty when it lists none apart."""
    if mask.full_kv_num_blocks is None or mask.full_kv_indices is None:
        return (
            torch.zeros_like(mask.kv_num_blocks),
            mask.kv_indices.new_zeros((*mask.kv_indices.shape[:-1], 1)),
        )
    return mask.full_kv_num_blocks, mask.full_kv_indices


def full_q_blocks(mask: BlockMask) -> tuple[torch.Tensor, torch.Tensor]:
    """Full query blocks of ``mask``, empty when it lists none apart."""
    q_num_blocks = cast("torch.Tensor", mask.q_num_blocks)
    q_indices = cast("torch.Tensor", mask.q_indices)
    if mask.full_q_num_blocks is None or mask.full_q_indices is None:
        return torch.zeros_like(q_num_blocks), torch.zeros_like(q_indices)
    return mask.full_q_num_blocks, mask.full_q_indices


def append_full_blocks(
    sliding_mask: BlockMask,
    entry_blocks: BlockMask,
    rows: Callable[[torch.Tensor], torch.Tensor],
    sliding_blocks: int,
) -> tuple[torch.Tensor | None, ...]:
    """Full key and query blocks of the sliding mask then the entries, or None when neither mask lists any apart.

    :param sliding_mask: Mask over the sliding keys.
    :param entry_blocks: Block lists over the entries, in the sliding mask's block size.
    :param rows: Expands a block tensor to every batch row.
    :param sliding_blocks: Key blocks the padded sliding keys span.
    :return: Full key block counts and indices, then full query block counts and indices.
    """
    if (
        sliding_mask.full_kv_num_blocks is None
        and entry_blocks.full_kv_num_blocks is None
    ):
        return None, None, None, None
    sliding_full, entry_full = (
        full_kv_blocks(sliding_mask),
        full_kv_blocks(entry_blocks),
    )
    full_kv_num, full_kv_indices = append_kv_blocks(
        rows(sliding_full[0]),
        rows(sliding_full[1]),
        rows(entry_full[0]),
        rows(entry_full[1]),
        sliding_blocks,
    )
    sliding_full_q, entry_full_q = (
        full_q_blocks(sliding_mask),
        full_q_blocks(entry_blocks),
    )
    return (
        full_kv_num,
        full_kv_indices,
        torch.cat([rows(sliding_full_q[0]), rows(entry_full_q[0])], -1),
        torch.cat([rows(sliding_full_q[1]), rows(entry_full_q[1])], -2),
    )


def append_entries(
    kv: torch.Tensor,
    entries: torch.Tensor,
    sliding_mask: BlockMask,
    entry_blocks: BlockMask,
    entry_mod: MaskMod,
) -> tuple[torch.Tensor, BlockMask]:
    """Sliding keys padded to whole mask blocks, then the compressed entries, and the mask over both.

    :param kv: ``[batch, 1, sliding_len, head_dim]`` sliding keys.
    :param entries: ``[batch, 1, entries, head_dim]`` compressed entries.
    :param sliding_mask: Mask over the sliding keys.
    :param entry_blocks: Block lists over the entries, in the sliding mask's block size.
    :param entry_mod: Mask of each query over the entries, within ``entry_blocks``.
    :return: The keys and their mask; the padding keys are masked out.
    """
    sliding_len = kv.shape[2]
    sliding_blocks = -(-sliding_len // sliding_mask.BLOCK_SIZE[1])
    padded_len = sliding_blocks * sliding_mask.BLOCK_SIZE[1]
    padding = kv.new_zeros((*kv.shape[:2], padded_len - sliding_len, kv.shape[3]))
    keys = torch.cat([kv, padding, entries], dim=2)
    batch = max(
        sliding_mask.kv_num_blocks.shape[0], entry_blocks.kv_num_blocks.shape[0]
    )

    def rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.expand(batch, *tensor.shape[1:])

    kv_num, kv_indices = append_kv_blocks(
        rows(sliding_mask.kv_num_blocks),
        rows(sliding_mask.kv_indices),
        rows(entry_blocks.kv_num_blocks),
        rows(entry_blocks.kv_indices),
        sliding_blocks,
    )
    full_kv_num, full_kv_indices, full_q_num, full_q_indices = append_full_blocks(
        sliding_mask, entry_blocks, rows, sliding_blocks
    )
    q_num = torch.cat(
        [
            rows(cast("torch.Tensor", sliding_mask.q_num_blocks)),
            rows(cast("torch.Tensor", entry_blocks.q_num_blocks)),
        ],
        dim=-1,
    )
    q_indices = torch.cat(
        [
            rows(cast("torch.Tensor", sliding_mask.q_indices)),
            rows(cast("torch.Tensor", entry_blocks.q_indices)),
        ],
        dim=-2,
    )
    sliding_mod = sliding_mask.mask_mod

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        sliding = sliding_mod(
            batch_idx, head_idx, q_idx, kv_idx.clamp(max=sliding_len - 1)
        )
        entry = entry_mod(
            batch_idx, head_idx, q_idx, (kv_idx - padded_len).clamp(min=0)
        )
        return torch.where(
            kv_idx < sliding_len, sliding, (kv_idx >= padded_len) & entry
        )

    return keys, BlockMask(
        seq_lengths=(sliding_mask.seq_lengths[0], keys.shape[2]),
        kv_num_blocks=kv_num,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num,
        full_kv_indices=full_kv_indices,
        q_num_blocks=q_num,
        q_indices=q_indices,
        full_q_num_blocks=full_q_num,
        full_q_indices=full_q_indices,
        BLOCK_SIZE=sliding_mask.BLOCK_SIZE,
        mask_mod=mask_mod,
    )


def document_keys(
    module: DeepseekV4Attention,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    kv: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Cache | None,
    sliding_mask: BlockMask,
    document_lengths: tuple[int, ...],
) -> tuple[torch.Tensor, BlockMask]:
    """Sliding keys then every document's compressed entries, and their mask.

    :param module: CSA / HCA attention layer.
    :param hidden_states: ``[batch, seq_len, hidden]`` attention input.
    :param q_residual: ``[batch, seq_len, q_lora_rank]`` normed query latent.
    :param kv: ``[batch, 1, sliding_len, head_dim]`` sliding keys.
    :param position_ids: ``[batch or 1, seq_len]`` positions.
    :param past_key_values: Cache, or None.
    :param sliding_mask: The model's mask over the sliding keys.
    :param document_lengths: Lengths of the documents in each row, summing to ``seq_len``.
    :return: The keys and their mask; a query sees only its own document's entries.
    """
    compressor = cast(
        "DeepseekV4CSACompressor | DeepseekV4HCACompressor", module.compressor
    )
    batch, seq_len, _ = hidden_states.shape
    if past_key_values is not None:
        compressed, block_bias = compressor(
            hidden_states, q_residual, position_ids, past_key_values, module.layer_idx
        )
        if compressed.shape[2] == 0:
            return kv, sliding_mask
        visible = (
            compressed.new_ones((batch, seq_len, compressed.shape[2]), dtype=torch.bool)
            if block_bias is None
            else block_bias[:, 0] == 0
        )
        entry_mod = visible_mod(visible)
        entry_blocks = build_block_mask(
            entry_mod,
            batch,
            seq_len,
            compressed.shape[2],
            sliding_mask.BLOCK_SIZE,
            kv.device,
            True,
        )
        return append_entries(kv, compressed, sliding_mask, entry_blocks, entry_mod)
    compressed, layout, visible = compress_documents(
        compressor, hidden_states, q_residual, position_ids, document_lengths
    )
    if compressed.shape[2] == 0:
        return kv, sliding_mask
    rate = int(compressor.compress_rate)
    separate_full = visible is None
    entry_blocks = ENTRY_MEMO.get(
        position_ids,
        ("mask", rate, document_lengths, batch, sliding_mask.BLOCK_SIZE, separate_full),
        lambda: causal_entry_blocks(
            layout, position_ids, rate, batch, sliding_mask.BLOCK_SIZE, separate_full
        ),
    )
    entry_mod = entry_blocks.mask_mod if visible is None else visible_mod(visible)
    return append_entries(kv, compressed, sliding_mask, entry_blocks, entry_mod)


@functools.lru_cache(maxsize=8)
def shard_sliding_mask(
    query_len: int, key_len: int, offset: int, window: int, device: torch.device
) -> BlockMask:
    """Sliding-window causal mask of one sequence shard's queries over the whole sequence's keys.

    :param query_len: Queries on this shard.
    :param key_len: Keys of the whole sequence.
    :param offset: Sequence position of this shard's first query.
    :param window: Keys per query, counting the query.
    :param device: Device the mask lives on.
    :return: A ``BlockMask`` shared by every batch row and head.
    """

    def sliding(
        _batch: torch.Tensor,
        _head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        query = q_idx + offset
        return (kv_idx <= query) & (kv_idx > query - window)

    return build_block_mask(
        sliding,
        None,
        query_len,
        key_len,
        (FLEX_BLOCK_SIZE, FLEX_BLOCK_SIZE),
        device,
        True,
    )


def gather_shards(
    pieces: list[torch.Tensor], group: dist.ProcessGroup
) -> list[torch.Tensor]:
    """Every shard's copy of each ``[batch, ...]`` piece, in one differentiable all-gather.

    :param pieces: This shard's tensors, each with a leading batch dimension.
    :param group: Context-parallel group; shards are in rank order.
    :return: Per piece, ``[batch, world, *piece.shape[1:]]`` in the piece's dtype.
    """
    # Import cycle: agilerl.distributed imports agilerl.architectures, whose catalog imports this module.
    from agilerl.distributed.context_parallel.sequence import gather_sequence_for_cp

    batch = pieces[0].shape[0]
    world = dist.get_world_size(group)
    flat = torch.cat([piece.flatten(1) for piece in pieces], dim=1)
    gathered = gather_sequence_for_cp(flat, group, 1).view(batch, world, -1)
    return [
        part.reshape(batch, world, *piece.shape[1:]).to(piece.dtype)
        for part, piece in zip(
            gathered.split([piece[0].numel() for piece in pieces], dim=-1),
            pieces,
            strict=True,
        )
    ]


def previous_tokens(
    tails: torch.Tensor, group: dist.ProcessGroup, halo: int
) -> torch.Tensor:
    """The ``halo`` tokens before this shard, from every shard's equal-length tail.

    :param tails: ``[batch, tail, width]`` last tokens of this shard.
    :param group: Context-parallel group; shards are contiguous, in rank order.
    :param halo: Tokens to return; at most the earlier shards' tails hold.
    :return: ``[batch, halo, width]``.
    """
    # Import cycle: agilerl.distributed imports agilerl.architectures, whose catalog imports this module.
    from agilerl.distributed.context_parallel.sequence import gather_sequence_for_cp

    gathered = gather_sequence_for_cp(tails, group, 1)
    end = dist.get_rank(group) * tails.shape[1]
    return gathered[:, end - halo : end]


def pool_shard_entries(
    compressor: DeepseekV4CSACompressor | DeepseekV4HCACompressor,
    indexer: DeepseekV4Indexer | None,
    hidden_states: torch.Tensor,
    group: dist.ProcessGroup,
    windows: EntryWindows,
    overlap: bool,
    halo: int,
    tail: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Entries this shard pools, over the ``halo`` tokens before it then its own.

    :param compressor: CSA / HCA compressor.
    :param indexer: The CSA indexer, or None.
    :param hidden_states: ``[batch, shard_len, hidden]`` attention input of this shard.
    :param group: Context-parallel group; shards are contiguous, in rank order.
    :param windows: Windows this shard pools.
    :param overlap: Each entry also pools the window before its own, as CSA does.
    :param halo: Earlier tokens this shard's windows read.
    :param tail: Last tokens each shard sends; 0 when no shard reads earlier tokens.
    :return: ``[batch, entries, head_dim]`` compressed entries, and the indexer's
        ``[batch, entries, index_head_dim]`` keys or None without an indexer.
    """
    shard_len = hidden_states.shape[1]
    projected = [compressor.kv_proj(hidden_states), compressor.gate_proj(hidden_states)]
    if indexer is not None:
        with torch.no_grad():
            projected += [
                indexer.kv_proj(hidden_states),
                indexer.gate_proj(hidden_states),
            ]
    if tail > 0:
        earlier = previous_tokens(
            torch.cat([tokens[:, shard_len - tail :] for tokens in projected], dim=-1),
            group,
            halo,
        ).split([tokens.shape[-1] for tokens in projected], dim=-1)
        projected = [
            torch.cat([before.to(tokens.dtype), tokens], dim=1)
            for before, tokens in zip(earlier, projected, strict=True)
        ]
    local = pool_entries(compressor, projected[0], projected[1], windows, overlap)
    index_keys = None
    if indexer is not None:
        with torch.no_grad():
            index_keys = pool_entries(
                indexer, projected[2], projected[3], windows, True
            )
    return local, index_keys


def join_shard_entries(padded: torch.Tensor, counts: tuple[int, ...]) -> torch.Tensor:
    """Every shard's entries in sequence order, without the padding.

    :param padded: ``[batch, world, most, width]`` gathered entries, padded to the largest count.
    :param counts: Entries per shard.
    :return: ``[batch, sum(counts), width]``.
    """
    return torch.cat(
        [padded[:, other, :count] for other, count in enumerate(counts)], dim=1
    )


def sequence_shard_keys(
    module: DeepseekV4Attention,
    group: dist.ProcessGroup,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    kv: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, BlockMask]:
    """Keys of the whole sequence for this shard's queries, and their mask.

    :param module: Attention layer.
    :param group: Context-parallel group; shards are contiguous, in rank order.
    :param hidden_states: ``[batch, shard_len, hidden]`` attention input of this shard.
    :param q_residual: ``[batch, shard_len, q_lora_rank]`` normed query latent of this shard.
    :param kv: ``[batch, 1, shard_len, head_dim]`` sliding keys of this shard.
    :param position_ids: ``[batch or 1, shard_len]`` sequence positions of this shard.
    :return: ``[batch, 1, keys, head_dim]`` sliding keys of every shard, padding to a
        whole mask block, then the compressed entries; and the mask over them.
    """
    batch, shard_len, _ = hidden_states.shape
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    sliding_len = shard_len * world
    sliding_mask = shard_sliding_mask(
        shard_len, sliding_len, rank * shard_len, int(module.sliding_window), kv.device
    )
    compressor = module.compressor
    if compressor is None or sliding_len < int(compressor.compress_rate):
        (keys,) = gather_shards([kv], group)
        return keys.transpose(1, 2).flatten(2, 3), sliding_mask
    rate = int(compressor.compress_rate)
    overlap = isinstance(compressor, DeepseekV4CSACompressor)
    entries = sliding_len // rate
    # The CSA indexer, bound only when top-k drops entries.
    indexer = None
    if overlap and entries > compressor.indexer.index_topk:
        indexer = compressor.indexer
    windows, layout, counts, halo, tail = ENTRY_MEMO.get(
        position_ids,
        ("shard", rate, overlap, shard_len, rank, world),
        lambda: shard_entries(shard_len, rank, world, rate, overlap, kv.device),
    )
    local, index_keys = pool_shard_entries(
        compressor, indexer, hidden_states, group, windows, overlap, halo, tail
    )
    most = max(counts)
    pieces = [kv, F.pad(local, (0, 0, 0, most - local.shape[1]))]
    if index_keys is not None:
        pieces.append(F.pad(index_keys, (0, 0, 0, most - index_keys.shape[1])))
    gathered = gather_shards(pieces, group)
    keys = gathered[0].transpose(1, 2).flatten(2, 3)
    compressed = join_shard_entries(gathered[1], counts)
    visible = None
    if indexer is not None:
        with torch.no_grad():
            visible = select_entries(
                indexer,
                join_shard_entries(gathered[2], counts),
                hidden_states,
                q_residual,
                position_ids,
                layout,
            )
    separate_full = visible is None
    entry_blocks = ENTRY_MEMO.get(
        position_ids,
        (
            "shard mask",
            rate,
            shard_len,
            world,
            batch,
            sliding_mask.BLOCK_SIZE,
            separate_full,
        ),
        lambda: causal_entry_blocks(
            layout, position_ids, rate, batch, sliding_mask.BLOCK_SIZE, separate_full
        ),
    )
    entry_mod = entry_blocks.mask_mod if visible is None else visible_mod(visible)
    return append_entries(
        keys, compressed.unsqueeze(1), sliding_mask, entry_blocks, entry_mod
    )


def flex_attention_with_sinks(
    query: torch.Tensor,
    key_value: torch.Tensor,
    block_mask: BlockMask,
    sinks: torch.Tensor,
    scale: float,
    training: bool,
) -> torch.Tensor:
    """Flex attention over one shared key/value head, with one sink logit per head in the softmax.

    :param query: ``[batch, heads, seq_q, head_dim]`` queries.
    :param key_value: ``[batch, 1, seq_kv, head_dim]`` keys, also read as values.
    :param block_mask: Mask over ``seq_kv``.
    :param sinks: ``[heads]`` sink logits.
    :param scale: Softmax scale.
    :param training: Whether the module trains; selects the compiled flex variant.
    :return: ``[batch, heads, seq_q, head_dim]`` output in the query dtype.
    """
    kernel_options = (
        large_head_dim_kernel_options(
            query.shape[-1],
            key_value.shape[-1],
            query.element_size(),
            torch.cuda.get_device_properties(
                query.device
            ).shared_memory_per_block_optin,
        )
        if query.is_cuda
        else flex_decode_kernel_options(query, block_mask)
    )
    output, aux = cast(
        "tuple[torch.Tensor, AuxOutput]",
        compile_friendly_flex_attention(
            query,
            key_value,
            key_value,
            block_mask=block_mask,
            enable_gqa=True,
            scale=scale,
            kernel_options=kernel_options,
            training=training,
            return_aux=AuxRequest(lse=True),
        ),
    )
    lse = cast("torch.Tensor", aux.lse)
    # exp(lse) / (exp(lse) + exp(sink)): the keys' share of the sink-augmented softmax.
    key_share = torch.sigmoid(lse.float() - sinks.float().view(1, -1, 1))
    return output * key_share.to(output.dtype).unsqueeze(-1)


def flex_compressed_attention(
    forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
    """Wrap ``DeepseekV4Attention.forward`` to run flex over the sliding and compressed keys.

    Under a ``BlockMask`` the keys are the call's own. Given :data:`CONTEXT_PARALLEL_GROUP`,
    the queries are one sequence shard and the keys cover the whole causal sequence; the
    shard's ``attention_mask`` is ignored.

    :param forward: Stock attention forward, run for any other mask outside context parallel.
    :return: Attention forward returning ``(output, None)`` on the flex path.
    """

    @functools.wraps(forward)
    def attention_forward(
        self: DeepseekV4Attention,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | BlockMask | None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        document_lengths = kwargs.pop(PACKED_DOCUMENT_LENGTHS, None)
        cp_group = kwargs.pop(CONTEXT_PARALLEL_GROUP, None)
        if cp_group is None and not isinstance(attention_mask, BlockMask):
            return forward(
                self,
                hidden_states,
                position_embeddings,
                position_ids,
                attention_mask,
                past_key_values,
                **kwargs,
            )
        if self.training and self.attention_dropout > 0:
            msg = "DeepSeek-V4 flex attention does not support attention dropout"
            raise ValueError(msg)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = apply_rotary_pos_emb(self.q_b_norm(q), cos, sin)
        kv = self.kv_norm(self.kv_proj(hidden_states))
        kv = apply_rotary_pos_emb(kv.view(*hidden_shape).transpose(1, 2), cos, sin)
        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_mask = cast("BlockMask", attention_mask)
        if cp_group is not None:
            kv, block_mask = sequence_shard_keys(
                self, cp_group, hidden_states, q_residual, kv, position_ids
            )
        elif self.compressor is not None:
            kv, block_mask = document_keys(
                self,
                hidden_states,
                q_residual,
                kv,
                position_ids,
                past_key_values,
                block_mask,
                document_lengths or (hidden_states.shape[1],),
            )

        attn_output = flex_attention_with_sinks(
            q, kv, block_mask, self.sinks, self.scaling, self.training
        )
        # K = V, so the output carries RoPE on its rope slice; rotate back at the query position.
        attn_output = apply_rotary_pos_emb(attn_output, cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), None

    return attention_forward


def pass_packed_document_lengths(
    forward: Callable[..., object],
) -> Callable[..., object]:
    """Wrap ``DeepseekV4Model.forward`` to pass packed document lengths to every attention layer.

    :param forward: Stock model forward.
    :return: Model forward adding :data:`PACKED_DOCUMENT_LENGTHS` for packed rows.
    """

    @functools.wraps(forward)
    def model_forward(
        self: DeepseekV4Model,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Any,
    ) -> object:
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = self.config.use_cache
        # A padded row repeats positions under its mask; only maskless rows are packed.
        if (
            position_ids is not None
            and attention_mask is None
            and past_key_values is None
            and not use_cache
        ):
            # Kept out of the checkpointed layers: the lengths need one host sync.
            document_lengths = packed_document_lengths(position_ids)
            if document_lengths is not None:
                kwargs = {**kwargs, PACKED_DOCUMENT_LENGTHS: document_lengths}
        return forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    return model_forward


@library_shim("transformers", "==5.19.*")
def install_deepseek_v4_patches(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    use_sequence_packing: bool = False,
) -> None:
    """Patch the DeepSeek-V4 indexer and its scorer, flex attention over the compressed KV, the hyper-connection mix, and packed documents.

    :param _patch: Catalog patch config.
    :param _model: Already-built model, or None; the class-level patches cover it.
    :param use_sequence_packing: Also pass packed document lengths from the model forward.
    """
    indexer_cls: Any = DeepseekV4Indexer
    scorer_cls: Any = DeepseekV4IndexerScorer
    attention_cls: Any = DeepseekV4Attention
    if not class_is_patched(indexer_cls, SELECTS_ALL_WHEN_TOPK_COVERS):
        indexer_cls.forward = select_all_when_topk_covers(
            indexer_cls.__dict__["forward"]
        )
        setattr(indexer_cls, SELECTS_ALL_WHEN_TOPK_COVERS, True)
    if not class_is_patched(scorer_cls, CHUNKS_INDEXER_SCORES):
        scorer_cls.forward = chunked_scorer_forward
        setattr(scorer_cls, CHUNKS_INDEXER_SCORES, True)
    if not class_is_patched(attention_cls, FLEX_COMPRESSED_KV):
        attention_cls.forward = flex_compressed_attention(
            attention_cls.__dict__["forward"]
        )
        setattr(DeepseekV4Attention, FLEX_COMPRESSED_KV, True)
        DeepseekV4PreTrainedModel._supports_flex_attn = True
    compile_hyper_connection_mix(DeepseekV4HyperConnection)
    if use_sequence_packing and not class_is_patched(DeepseekV4Model, PACKED_DOCUMENTS):
        DeepseekV4Model.forward = pass_packed_document_lengths(
            DeepseekV4Model.__dict__["forward"]
        )
        setattr(DeepseekV4Model, PACKED_DOCUMENTS, True)
